"""The meter must charge what the provider billed — no more, no less.

Two failures, opposite in sign, met here:

* **Under-charging.** `MeterMiddleware` charged in `on_response`, which
  `BaseMiddleware.as_middleware` reaches only on normal stream exhaustion. Its
  `except Exception` cannot catch `GeneratorExit` / `CancelledError`, so a
  consumer that `break`s out of the `async for` bypassed the charge entirely.
  Measured: `provider stream calls: 2` while the budget read
  `spent=0.25 calls=1` from the single fully-consumed call.

* **Over-charging.** A `memoize` hit re-emits the cached result, `usage` and
  all, and `on_response` charged on any `usage`. Measured: four identical chats
  behind `memoize` gave `provider_calls=1` but drove `spent 0.25 → 1.0`, and a
  fifth raised `MeterExceeded` on money that was never spent.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from agentkit.adapters.store.memory import InMemoryStore
from agentkit.kernel.types import ChatRequest, Delta, Message, Scope, Usage
from agentkit.middlewares import memoize, meter, tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.runtime.meter import MeterExceeded

PRICE = Usage(input_tokens=1000, output_tokens=500, cost_usd=0.25)


class _StreamLLM:
    """Bills on the FIRST delta (as a provider that reports usage up front
    does), then keeps streaming — so an abandoned stream has real spend
    attached to it."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls = 0
        self._fail_after = fail_after

    async def stream(self, **_kw: Any):
        self.calls += 1
        if self._fail_after == 0:
            raise RuntimeError("provider 500")  # pre-stream: nothing was produced
        yield Delta(text="hel", model="m", provider="p", usage=PRICE)
        if self._fail_after == 1:
            raise RuntimeError("provider 500 mid-stream")
        yield Delta(text="lo", model="m", provider="p")
        yield Delta(finish_reason="stop", model="m", provider="p")


def _wire(llm: Any, *middleware: Any, ceiling: float | None = 10.0) -> tuple[Invoker, Budget, RunContext]:
    inv = Invoker(llm=llm, chat_middleware=list(middleware))
    budget = Budget(max_cost_usd=ceiling)
    ctx = RunContext("run", Scope(), budget, Services(invoker=inv))
    return inv, budget, ctx


REQ = ChatRequest([Message("user", "hi")], "m")


# ── under-charging: the abandoned / interrupted stream ──────────────────────


def test_a_fully_consumed_stream_is_charged_exactly_once() -> None:
    """POSITIVE CONTROL for the whole file. Every other test here asserts that
    something is NOT charged; a "fix" that stops metering would sail through
    them and die here."""
    llm = _StreamLLM()
    inv, budget, ctx = _wire(llm, meter())

    async def go() -> None:
        async for _ in inv.stream(REQ, ctx):
            pass

    asyncio.run(go())
    assert budget.spent_usd == pytest.approx(0.25)
    assert budget.calls == 1


def test_an_abandoned_stream_is_still_charged() -> None:
    """A consumer that breaks after one delta: the provider billed, so the
    budget must see it. Before the fix `spent` and `calls` never moved."""
    llm = _StreamLLM()
    inv, budget, ctx = _wire(llm, meter())

    async def go() -> None:
        gen = inv.stream(REQ, ctx)
        async with contextlib.aclosing(gen):
            async for _ in gen:
                break  # walk away mid-stream → GeneratorExit at the yield

    asyncio.run(go())
    assert llm.calls == 1
    assert budget.spent_usd == pytest.approx(0.25), "an abandoned stream escaped the meter"
    assert budget.calls == 1


def test_a_bare_break_is_charged_too_but_not_synchronously() -> None:
    """The common pattern, and the trap in it.

    ``async with aclosing(...)`` closes the generator deterministically, so the
    charge lands before the block exits. A bare ``break`` — which is what most
    consumers actually write — leaves finalization to asyncio's async-generator
    hooks, so the charge lands a couple of event-loop turns LATER. Measured:

        immediately after the consumer returned: spent=0.0
        after ~2 loop turns + gc:                spent=0.25

    The money is never lost, but a caller that reads ``budget.spent()`` on the
    line after a ``break`` sees zero and may conclude the run was free. Pinned
    so the behaviour is a documented property rather than a surprise, and so a
    future change that drops the charge entirely fails here.
    """
    llm = _StreamLLM()
    inv, budget, ctx = _wire(llm, meter())

    async def go() -> tuple[float, float]:
        async def consume() -> None:
            async for _ in inv.stream(REQ, ctx):
                break  # no aclosing: finalization is the loop's business now

        await consume()
        immediately = budget.spent_usd
        for _ in range(4):  # a couple of turns is enough; four is slack
            await asyncio.sleep(0)
        return immediately, budget.spent_usd

    immediately, eventually = asyncio.run(go())

    assert immediately == 0.0, "if this starts passing synchronously, update the docs"
    assert eventually == pytest.approx(0.25), "the charge must land at finalization"
    assert llm.calls == 1


def test_a_cancelled_stream_is_still_charged() -> None:
    """`CancelledError` is the other `BaseException` that used to slip past the
    charge — a cancelled run bills exactly like an abandoned one."""
    llm = _StreamLLM()
    inv, budget, ctx = _wire(llm, meter())

    async def go() -> None:
        gen = inv.stream(REQ, ctx)
        await gen.__anext__()
        with contextlib.suppress(asyncio.CancelledError):
            await gen.athrow(asyncio.CancelledError())

    asyncio.run(go())
    assert budget.spent_usd == pytest.approx(0.25), "a cancelled stream escaped the meter"
    assert budget.calls == 1


def test_a_MID_stream_failure_is_charged_for_what_arrived() -> None:
    """Tokens already streamed were billed by the provider — a later 500 does
    not un-send them."""
    llm = _StreamLLM(fail_after=1)
    inv, budget, ctx = _wire(llm, meter())

    async def go() -> None:
        with contextlib.suppress(RuntimeError):
            async for _ in inv.stream(REQ, ctx):
                pass

    asyncio.run(go())
    assert budget.spent_usd == pytest.approx(0.25)
    assert budget.calls == 1


def test_a_PRE_stream_failure_is_not_charged() -> None:
    """The edge that keeps the fix honest in the other direction: nothing was
    produced, so nothing is charged and `calls` is not inflated — which matters
    because `retry()` sits INSIDE `meter()` in the documented chat chain."""
    llm = _StreamLLM(fail_after=0)
    inv, budget, ctx = _wire(llm, meter())

    async def go() -> None:
        with contextlib.suppress(RuntimeError):
            async for _ in inv.stream(REQ, ctx):
                pass

    asyncio.run(go())
    assert budget.spent_usd == pytest.approx(0.0)
    assert budget.calls == 0


def test_abandoning_an_over_ceiling_stream_records_the_spend_without_raising() -> None:
    """`Budget.charge` books the money BEFORE it checks the ceiling. On the
    abandon path we keep the booking and swallow `MeterExceeded` rather than
    replacing a consumer's `GeneratorExit` with an unrelated error — the next
    `on_request` guard is what stops the run."""
    llm = _StreamLLM()
    inv, budget, ctx = _wire(llm, meter(), ceiling=0.10)  # one call already blows it

    async def go() -> None:
        gen = inv.stream(REQ, ctx)
        async with contextlib.aclosing(gen):
            async for _ in gen:
                break

    asyncio.run(go())  # must NOT raise out of the close
    assert budget.spent_usd == pytest.approx(0.25), "the spend was dropped along with the exception"

    with pytest.raises(MeterExceeded):  # the guard on the NEXT call is the stop
        asyncio.run(inv.chat(REQ, ctx))


# ── over-charging: the cache hit ────────────────────────────────────────────


class _OneShotLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, **_kw: Any):
        self.calls += 1
        yield Delta(text="answer", model="m", provider="p")
        yield Delta(usage=PRICE, finish_reason="stop", model="m", provider="p")


class _CountingTrace:
    """Counts spans so we can prove a cache hit is still OBSERVED even though
    it is no longer BILLED."""

    def __init__(self) -> None:
        self.spans: list[str] = []
        self.events: list[str] = []

    @contextlib.contextmanager
    def span(self, name: str, kind: str, **attrs: Any):
        self.spans.append(name)
        yield type("S", (), {"set": lambda *_a: None, "add_event": lambda *_a, **_k: None})()

    def current_span_id(self) -> None:
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        self.events.append(name)


def _cached_wire(ceiling: float = 1.0) -> tuple[Invoker, Budget, RunContext, _OneShotLLM, _CountingTrace]:
    store, llm, trace = InMemoryStore(), _OneShotLLM(), _CountingTrace()
    inv = Invoker(llm=llm, chat_middleware=[tracing(), meter(), memoize(store=store)])
    budget = Budget(max_cost_usd=ceiling)
    ctx = RunContext("run", Scope(), budget, Services(invoker=inv, store=store, trace=trace))
    return inv, budget, ctx, llm, trace


def test_a_cache_hit_is_not_charged_again() -> None:
    """Four identical chats, one provider call — one charge."""
    inv, budget, ctx, llm, _ = _cached_wire()

    async def go() -> None:
        for _ in range(4):
            await inv.chat(REQ, ctx)

    asyncio.run(go())
    assert llm.calls == 1
    assert budget.spent_usd == pytest.approx(0.25), "cache hits were billed as provider calls"
    assert budget.calls == 1


def test_a_cache_hit_never_trips_a_budget_it_did_not_spend() -> None:
    """The user-visible consequence: the fifth identical call used to raise
    `MeterExceeded: cost $1.25 > $1` on $1.00 of money that was never spent."""
    inv, budget, ctx, llm, _ = _cached_wire(ceiling=1.0)

    async def go() -> None:
        for _ in range(20):
            await inv.chat(REQ, ctx)

    asyncio.run(go())  # must not raise
    assert llm.calls == 1
    assert budget.spent_usd == pytest.approx(0.25)


def test_a_cache_hit_is_still_TRACED() -> None:
    """Not charging must not mean not observing: `tracing()` sits outside
    `meter()`, so every call — hit or miss — still gets its chat span, and the
    hit additionally carries the `cache.hit` event that explains the missing
    provider span underneath it."""
    inv, budget, ctx, llm, trace = _cached_wire()

    async def go() -> None:
        await inv.chat(REQ, ctx)
        await inv.chat(REQ, ctx)

    asyncio.run(go())
    assert trace.spans == ["chat", "chat"], "the cache hit lost its span"
    assert "cache.hit" in trace.events
    assert llm.calls == 1
    assert budget.calls == 1


def test_a_real_MISS_is_still_charged_behind_the_cache() -> None:
    """POSITIVE CONTROL for the cache-hit fix: two DIFFERENT questions both
    miss, so both are billed. A meter that simply stopped charging when a store
    is wired would pass every test above and fail this one."""
    inv, budget, ctx, llm, _ = _cached_wire(ceiling=10.0)

    async def go() -> None:
        await inv.chat(ChatRequest([Message("user", "first")], "m"), ctx)
        await inv.chat(ChatRequest([Message("user", "second")], "m"), ctx)

    asyncio.run(go())
    assert llm.calls == 2
    assert budget.spent_usd == pytest.approx(0.50)
    assert budget.calls == 2
