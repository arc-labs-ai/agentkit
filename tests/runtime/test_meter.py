"""Meter — Budget (run) and Quota (tenant) as one protocol, driven by the meter middleware."""

import asyncio

import pytest

from agentkit.kernel.middleware import Call
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.runtime import Budget, MeterExceeded, Quota
from agentkit.runtime.context import RunContext


def _run(coro):
    return asyncio.run(coro)


def _call(scope=Scope(1, 2), msgs=None):
    ctx = RunContext("r", scope)
    return Call("chat", ChatRequest(messages=msgs or [Message("user", "hi")], model="m"), ctx)


def test_budget_guards_and_charges():
    b = Budget(max_cost_usd=0.015)
    _run(b.charge(_call(), Usage(0, 0, 0.01)))
    _run(b.guard(_call()))  # under ceiling → ok
    with pytest.raises(MeterExceeded):
        _run(b.charge(_call(), Usage(0, 0, 0.01)))  # 0.02 > 0.015 → raises after accruing
    assert b.calls == 2 and b.spent_usd == pytest.approx(0.02)


def test_quota_rpm_blocks_and_isolates_tenants():
    q = Quota(max_rpm=1, clock=lambda: 1000.0)
    _run(q.guard(_call(Scope(1, 1))))
    with pytest.raises(MeterExceeded):
        _run(q.guard(_call(Scope(1, 1))))
    _run(q.guard(_call(Scope(2, 2))))  # different tenant → isolated


def test_quota_tpm_after_charge():
    q = Quota(max_tpm=100, clock=lambda: 1000.0)
    _run(q.guard(_call(Scope(1, 1))))
    _run(q.charge(_call(Scope(1, 1)), Usage(150, 0, 0.0)))
    with pytest.raises(MeterExceeded):
        _run(q.guard(_call(Scope(1, 1))))


def test_quota_usd_window():
    q = Quota(max_usd=0.05, clock=lambda: 1000.0)
    _run(q.guard(_call()))
    _run(q.charge(_call(), Usage(0, 0, 0.06)))
    with pytest.raises(MeterExceeded):
        _run(q.guard(_call()))


# --- asyncio.Lock semantics on Budget + Quota --------------------------------
# Pin that the internal lock is the async kind, that two concurrent ``charge``
# calls serialize cleanly with no lost-update race, and that cancellation of a
# waiter doesn't deadlock the lock.


def test_budget_lock_is_asyncio_lock():
    """Regression guard: never accidentally revert to threading.Lock.

    The lock is lazily constructed via ``_get_lock`` so it binds to the
    running loop at first use; this test forces that construction and
    then asserts the type.
    """
    b = Budget(max_calls=10, max_cost_usd=1.0)

    async def _mint() -> asyncio.Lock:
        return b._get_lock()

    assert isinstance(asyncio.run(_mint()), asyncio.Lock)


def test_quota_lock_is_asyncio_lock():
    """Same guard for Quota — both meters live or die together."""
    q = Quota(max_rpm=10)

    async def _mint() -> asyncio.Lock:
        return q._get_lock()

    assert isinstance(asyncio.run(_mint()), asyncio.Lock)


def test_budget_lock_lazy_bound_to_running_loop():
    """Regression: a ``Budget`` built off the eventual event loop must still
    work inside ``asyncio.run(...)``. An eager ``default_factory=asyncio.Lock``
    would bind the lock to whatever loop happened to exist at construction
    time and later raise ``RuntimeError: ... attached to a different loop``.
    The lazy ``_get_lock`` fix means the lock is minted inside the running
    loop the first time a coroutine method touches it.
    """
    # Force the "wrong loop at construction" scenario: build Budget while a
    # short-lived event loop is the current one, then discard that loop.
    tmp_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(tmp_loop)
    try:
        b = Budget(max_cost_usd=1.0)
        assert b._lock is None  # lazy — no loop touched at construction
    finally:
        asyncio.set_event_loop(None)
        tmp_loop.close()

    async def use_it() -> None:
        await b.charge(_call(), Usage(0, 0, 0.5))

    # A fresh asyncio.run creates its own loop; the lock must bind here,
    # not to the now-closed tmp_loop.
    asyncio.run(use_it())
    assert b.spent_usd == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_concurrent_charges_serialize():
    """Two tasks charging concurrently must produce the same total as the
    same operations done sequentially — no lost-update race."""
    b = Budget(max_calls=200, max_cost_usd=10.0)

    async def charge_n(n: int) -> None:
        for _ in range(n):
            await b.charge(_call(), Usage(0, 0, 0.01))

    await asyncio.gather(charge_n(50), charge_n(50))
    # 50 + 50 charges at $0.01 each → $1.00 total, 100 calls.
    assert b.calls == 100
    assert abs(b.spent_usd - 1.0) < 1e-9


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_deadlock():
    """A task cancelled while waiting on the lock must release it cleanly,
    so a later acquirer proceeds without deadlock."""
    b = Budget(max_calls=10, max_cost_usd=1.0)

    started = asyncio.Event()
    proceed = asyncio.Event()

    async def long_holder() -> None:
        async with b._get_lock():
            started.set()
            await proceed.wait()

    holder = asyncio.create_task(long_holder())
    await started.wait()

    async def waiter() -> None:
        async with b._get_lock():
            pass  # would set this if reached, but we expect cancel

    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)  # let waiter park on the lock
    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    # Holder releases; a fresh acquirer must proceed cleanly.
    proceed.set()
    await holder

    async with b._get_lock():
        pass  # no deadlock — we acquired and released


# ── Per-tenant Quota enforced end-to-end through the meter middleware ────────
#
# Above tests exercise Quota's guard/charge in isolation. This block
# closes the loop: a real ``Invoker`` with ``meter()`` in the chain,
# a real ``LLMPort``, and two tenants sharing the same infrastructure.
# Tenant A exhausts its quota; tenant B must still run.


def _tiny_llm(cost: float = 0.02):
    """A minimal LLMPort that returns one delta carrying `cost`.
    Enough to drive the meter middleware's guard + charge round-trip."""
    from agentkit.kernel.types import Delta

    class _LLM:
        async def stream(self, **kw):
            yield Delta(
                text="ok",
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=cost),
                finish_reason="stop",
            )

    return _LLM()


def test_quota_isolates_tenants_end_to_end_through_meter_middleware():
    """Two tenants share ONE Invoker, ONE LLMPort, ONE Quota instance.
    Tenant A saturates its per-window budget; tenant B still runs.
    Verifies the meter middleware routes charges to the right scope
    key and that Quota partitions its per-window state by that key.

    Quota's ``guard`` fires when accrued cost is ``>= max_usd``. Each
    LLM call charges 0.03; the window cap is 0.05. Sequence per
    tenant: call 1 guards at 0 → passes → charges 0.03. Call 2
    guards at 0.03 → passes → charges 0.06. Call 3 guards at 0.06 → trips."""
    from agentkit.middlewares import meter
    from agentkit.runtime import Invoker, Services
    from agentkit.runtime.context import RunContext

    quota = Quota(max_usd=0.05, clock=lambda: 1000.0)
    inv = Invoker(llm=_tiny_llm(cost=0.03), chat_middleware=[meter()])

    def ctx_for(tenant_org: int) -> RunContext:
        return RunContext(
            correlation_id=f"run-{tenant_org}",
            scope=Scope(org_id=tenant_org, domain_id=1),
            services=Services(invoker=inv),
            meters=[quota],
        )

    req = ChatRequest(messages=[Message("user", "hi")], model="m")

    # Tenant A: first two calls fit under the 0.05 window; third trips.
    _run(inv.chat(req, ctx_for(1)))
    _run(inv.chat(req, ctx_for(1)))
    with pytest.raises(MeterExceeded):
        _run(inv.chat(req, ctx_for(1)))

    # Tenant B, running through the SAME Invoker with the SAME Quota,
    # sees the load-bearing isolation: their partition is untouched
    # by tenant A's exhaustion.
    _run(inv.chat(req, ctx_for(2)))
    _run(inv.chat(req, ctx_for(2)))
    # And their partition trips symmetrically at the same 0.06 spend.
    with pytest.raises(MeterExceeded):
        _run(inv.chat(req, ctx_for(2)))

    # The Quota's internal per-scope charge lists confirm partitioning
    # rather than a shared bucket.
    assert len(quota._charges["org1:dom1"]) == 2  # two successful calls
    assert len(quota._charges["org2:dom1"]) == 2  # tenant B's independent partition


def test_budget_and_quota_both_engaged_through_meter_middleware():
    """When both meters are attached, ``ctx.all_meters`` yields
    ``[budget, quota, …]`` and every meter guards + charges. Budget's
    ``charge`` records the spend then raises when the new total
    crosses the ceiling — the third call at 0.04 pushes the total
    from 0.08 to 0.12, so ``spent_usd`` lands at 0.12 and the
    ``MeterExceeded`` fires on the post-charge check."""
    from agentkit.middlewares import meter
    from agentkit.runtime import Invoker, Services
    from agentkit.runtime.context import RunContext

    budget = Budget(max_cost_usd=0.10)
    quota = Quota(max_usd=1.00, clock=lambda: 1000.0)  # generous — Budget trips first
    inv = Invoker(llm=_tiny_llm(cost=0.04), chat_middleware=[meter()])

    ctx = RunContext(
        correlation_id="r",
        scope=Scope(org_id=1, domain_id=1),
        budget=budget,
        services=Services(invoker=inv),
        meters=[quota],
    )
    req = ChatRequest(messages=[Message("user", "hi")], model="m")

    # Two calls at 0.04 → 0.08, under Budget's 0.10.
    _run(inv.chat(req, ctx))
    _run(inv.chat(req, ctx))
    # Third call's ``charge`` pushes to 0.12 and raises. The spend is
    # accrued before the raise (charge-then-check semantics).
    with pytest.raises(MeterExceeded):
        _run(inv.chat(req, ctx))

    # Post-condition: Budget's books reflect the accrued state.
    assert budget.spent_usd == pytest.approx(0.12)
    # Meter iteration order matters: ``all_meters`` yields
    # ``[budget, *extra_meters]``, so Budget's third charge raises
    # before Quota's ``charge`` runs. Quota only sees the two
    # successful calls' spend — observable evidence that meter
    # short-circuit is a hard boundary, not a best-effort fan-out.
    assert len(quota._charges["org1:dom1"]) == 2
