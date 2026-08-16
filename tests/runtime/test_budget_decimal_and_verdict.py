"""Budget — exact money, surviving token totals, and a recoverable exhaustion.

Three defects in one class, so three groups of tests:

1. Money was ``float``. Binary floating point cannot represent ``0.01``, so a
   metered run could not be reconciled to the cent.
2. Token counts were discarded. ``Usage`` arrived carrying input / output /
   cache-read / cache-write and was reduced to one scalar, forcing every
   application to re-aggregate what the framework had already seen.
3. ``charge()`` raised from inside itself. The framework also ships a
   checkpointer — so exhausting a budget aborted BEFORE the checkpoint was
   written, and everything spent was unrecoverable.

``test_exhausting_a_budget_writes_a_checkpoint_before_it_stops`` is the
load-bearing test. It fails against the pre-change code: ``MeterExceeded``
unwinds out of ``invoker.stream`` past every ``_save``, so the only checkpoint
on disk is the previous iteration's — stale, and pointing at a budget that
will raise again on the first guard of any resume.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from _assertions import assert_money, assert_no_float_money

from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import Checkpointer
from agentkit.kernel.middleware import Call
from agentkit.kernel.types import ChatRequest, Delta, Message, Scope, ToolCall, Usage
from agentkit.middlewares import meter
from agentkit.runtime import Budget, MeterExceeded, Quota
from agentkit.runtime.context import RunContext
from agentkit.runtime.meter import Charge, MoneyPrecisionError, to_money
from agentkit.testing import make_test_ctx
from agentkit.tools import tool


def _run(coro):
    return asyncio.run(coro)


def _call(scope=Scope(1, 2)):
    ctx = RunContext("r", scope)
    return Call("chat", ChatRequest(messages=[Message("user", "hi")], model="m"), ctx)


# ── 1. money is exact ────────────────────────────────────────────────────────


def test_a_hundred_charges_of_one_cent_sum_to_exactly_one_dollar() -> None:
    """The brief's headline. As floats this lands at 1.0000000000000007 and a
    run cannot be reconciled; as Decimal it is exactly 1.00.

    Asserted with ``==`` on ``Decimal``, deliberately — ``pytest.approx``
    here would be testing nothing, since approximate equality is precisely
    the property that was broken."""
    b = Budget()
    for _ in range(100):
        _run(b.charge(_call(), Usage(0, 0, 0.01)))
    assert_money(b.spent(), "1.00", label="spent()")
    assert_money(b.spent_cents(), "1.00", label="spent_cents()")


def test_float_mirror_is_kept_in_sync_for_existing_readers() -> None:
    """``spent_usd`` stays a readable float — ~20 doc references, 28 test
    references and every application reading it keep working. It is a
    RENDERING of the ledger, re-derived after each charge, so the two can
    never drift."""
    b = Budget()
    for _ in range(3):
        _run(b.charge(_call(), Usage(0, 0, 0.01)))
    assert b.spent_usd == pytest.approx(0.03)
    assert_money(b.spent(), "0.03")
    assert float(b.spent()) == b.spent_usd


def test_a_budget_rebuilt_from_a_checkpoint_keeps_working() -> None:
    """``Budget(spent_usd=saved.state["spent_usd"])`` is a DOCUMENTED resume
    path (docs/mental-models/03). Making ``spent_usd`` a read-only property
    would have broken it, so it stays an init-accepting field that seeds the
    exact ledger."""
    b = Budget(max_cost_usd=50, spent_usd=45.31)
    assert_money(b.spent(), "45.31")
    assert_money(b.remaining(), "4.69")


def test_an_over_precise_ceiling_is_refused_not_rounded() -> None:
    """A ceiling is the operator's stated INTENT, so quietly rounding it
    changes what they asked for. Refused at construction, when it is free to
    fix."""
    with pytest.raises(MoneyPrecisionError) as exc:
        Budget(max_cost_usd="0.0000001")
    assert "decimal places" in str(exc.value)


def test_an_over_precise_charge_is_quantized_not_refused() -> None:
    """The deliberate asymmetry with the rule above.

    A ceiling is intent; a charge is a MEASUREMENT. Raising here would mean a
    custom ``pricing=`` callable returning full float precision aborts a run
    mid-flight — re-creating the unrecoverable abort this class was rewritten
    to remove."""
    b = Budget()
    _run(b.charge(_call(), Usage(0, 0, 0.00000049)))
    assert_money(b.spent(), "0.000000", label="quantized charge")  # 6dp, run intact


def test_money_coercion_goes_through_str_not_the_binary_float() -> None:
    """``Decimal(0.01)`` is the binary value 0.01000000000000000020816...;
    ``Decimal(str(0.01))`` is the number the caller meant."""
    assert to_money(0.01) == Decimal("0.01")
    assert to_money("0.01") == Decimal("0.01")
    assert to_money(Decimal("0.01")) == Decimal("0.01")


def test_ceilings_accept_every_sane_spelling() -> None:
    for value in (Decimal("1.50"), 1.5, "1.50"):
        assert Budget(max_cost_usd=value).ceiling() == Decimal("1.50")
    assert Budget(max_cost_usd=2).ceiling() == Decimal("2")


def test_sub_cent_calls_are_not_rounded_away() -> None:
    """Why quantization happens at READ time, not per-charge: rounding each
    charge to cents would round every sub-cent call to zero and undercount the
    whole run. Ten thousand tenth-of-a-cent calls really is a dollar."""
    b = Budget()
    for _ in range(10_000):
        _run(b.charge(_call(), Usage(0, 0, 0.0001)))
    assert_money(b.spent(), "1.0000")
    assert_money(b.spent_cents(), "1.00")


# ── 2. token counts survive ──────────────────────────────────────────────────


def test_token_totals_survive_to_the_end_of_a_multi_agent_run() -> None:
    """``Budget`` is shared BY REFERENCE across ``ctx.child()``, so a tree of
    agents accumulates into one ``Usage``. The application no longer
    re-aggregates what the framework already summed."""
    budget = Budget()
    root = RunContext("r", Scope(1, 1), budget)
    kids = [root.child(), root.child(), root.child().child()]

    for depth, ctx in enumerate(kids, start=1):
        _run(
            ctx.budget.charge(
                _call(),
                Usage(
                    input_tokens=100 * depth,
                    output_tokens=10 * depth,
                    cost_usd=0.01,
                    cache_read_tokens=5 * depth,
                    cache_write_tokens=depth,
                ),
            )
        )

    total = budget.usage
    assert total.input_tokens == 600 and total.output_tokens == 60
    assert total.cache_read_tokens == 30 and total.cache_write_tokens == 6
    assert total.total_tokens == 660
    assert_money(budget.spent(), "0.03")


def test_usage_is_replaced_never_mutated() -> None:
    """``Usage`` is frozen. The accumulator goes through ``__add__``, so a
    caller holding a reference to an earlier total is never surprised by it
    changing underneath them."""
    b = Budget()
    _run(b.charge(_call(), Usage(10, 1, 0.001)))
    snapshot = b.usage
    _run(b.charge(_call(), Usage(10, 1, 0.001)))
    assert snapshot.input_tokens == 10, "an earlier snapshot was mutated in place"
    assert b.usage.input_tokens == 20


def test_the_verdict_carries_the_token_totals() -> None:
    """A caller acting on a verdict shouldn't have to go back to the Budget
    for the numbers that justified it."""
    b = Budget(max_cost_usd="0.01", on_exceeded="stop")
    verdict = _run(b.charge(_call(), Usage(500, 50, 0.02)))
    assert isinstance(verdict, Charge)
    assert verdict.usage.input_tokens == 500
    assert_money(verdict.spent, "0.02")


# ── 3. exhaustion returns a verdict, and is recoverable ──────────────────────


def test_raise_remains_the_default_so_existing_wiring_is_unchanged() -> None:
    """Flipping the default would silently change control flow everywhere: a
    run that used to abort would continue past its ceiling in any caller that
    ignores the return value — a worse failure than the one being fixed."""
    b = Budget(max_cost_usd="0.015")
    _run(b.charge(_call(), Usage(0, 0, 0.01)))
    with pytest.raises(MeterExceeded):
        _run(b.charge(_call(), Usage(0, 0, 0.01)))
    assert b.calls == 2
    assert_money(b.spent(), "0.02", label="spend recorded before the raise")


def test_stop_mode_returns_a_verdict_instead_of_raising() -> None:
    b = Budget(max_cost_usd="0.015", on_exceeded="stop")
    assert _run(b.charge(_call(), Usage(0, 0, 0.01))).ok is True
    verdict = _run(b.charge(_call(), Usage(0, 0, 0.01)))
    assert verdict.ok is False
    assert "0.02" in verdict.reason and "0.015" in verdict.reason
    assert_money(verdict.remaining, "0")
    assert b.exhausted() is True


def test_a_verdict_can_be_turned_back_into_the_exception() -> None:
    """For a caller who wants the old control flow at one specific site."""
    b = Budget(max_cost_usd="0.001", on_exceeded="stop")
    with pytest.raises(MeterExceeded):
        _run(b.charge(_call(), Usage(0, 0, 0.5))).raise_if_exceeded()
    assert _run(Budget().charge(_call(), Usage())).raise_if_exceeded().ok


def test_call_ceiling_also_produces_a_verdict() -> None:
    b = Budget(max_calls=1, on_exceeded="stop")
    assert _run(b.charge(_call(), Usage())).ok
    assert not _run(b.charge(_call(), Usage())).ok


# ── the load-bearing test ────────────────────────────────────────────────────


@tool(side_effecting=False)
def echo(text: str) -> str:
    """A tool, so the ReAct loop has something to run and a reason to
    checkpoint between iterations."""
    return text


class _CostlyLLM:
    """Requests a tool every turn, and every turn costs $0.60 — so the second
    chat call crosses a $1.00 ceiling."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, **_kw):
        self.calls += 1
        yield Delta(text=f"turn {self.calls}", model="costly", provider="fake")
        yield Delta(
            tool_calls=(ToolCall(f"c{self.calls}", "echo", {"text": "hi"}),),
            usage=Usage(input_tokens=1000, output_tokens=100, cost_usd=0.60),
            finish_reason="tool_calls",
            model="costly",
            provider="fake",
        )


def _react_ctx(budget: Budget, checkpointer: Checkpointer):
    return make_test_ctx(
        llm=_CostlyLLM(),
        budget=budget,
        checkpointer=checkpointer,
        chat_middleware=[meter()],
        correlation_id="budget-run",
    )


def _fresh_checkpointer() -> Checkpointer:
    from agentkit.adapters.checkpoint.in_memory import InMemoryCheckpointStore

    return Checkpointer(port=InMemoryCheckpointStore())


def test_exhausting_a_budget_writes_a_checkpoint_before_it_stops() -> None:
    """THE load-bearing test — it must go red against the pre-change code.

    Under the old behaviour ``MeterExceeded`` is raised from inside
    ``charge()``, which runs in the meter middleware's ``on_response``, which
    is inside ``ctx.invoker.stream`` — so it unwinds past every ``_save`` in
    the loop. The run aborts holding at best a STALE checkpoint from the
    previous iteration, this turn's spend is unrecorded against it, and a
    resume re-enters a budget that is still over its ceiling and raises again
    on the first guard. Everything spent is unrecoverable.

    With ``on_exceeded="stop"`` the meter records the spend and returns a
    verdict, control reaches the cognition, and the cognition — which still
    holds the live context — writes a ``suspended`` checkpoint carrying the
    CURRENT state before ending the run.
    """
    budget = Budget(max_cost_usd="1.00", on_exceeded="stop")
    cp = _fresh_checkpointer()
    agent = Agent("spender", "m", cognition=ReActCognition(tools=[echo], checkpointer=cp))

    result = asyncio.run(agent.run("go", _react_ctx(budget, cp)))

    # 1. The run ended as a recorded outcome, not an exception.
    assert result.stop_reason == "budget_exhausted"
    assert result.partial is True

    # 2. It is distinguishable from a completed run AND flagged resumable.
    assert result.is_resumable is True
    assert result.is_suspended is False  # resumable, but not waiting on a human

    # 3. The checkpoint exists, is marked suspended, and is CURRENT — not the
    #    stale previous-iteration snapshot the old code would have left.
    saved = asyncio.run(cp.resume("budget-run"))
    assert saved is not None, "no checkpoint was written before the budget stopped the run"
    assert saved.status == "suspended"
    assert saved.state["iteration"] == 2, "the checkpoint is a whole iteration behind"
    assert saved.state["usage"]["cost"] > 0, "the checkpoint records the spend that happened"
    assert any(m["role"] == "assistant" for m in saved.state["messages"])

    # 4. And the books reconcile exactly.
    assert_money(budget.spent(), "1.20")
    assert budget.usage.input_tokens == 2000


def test_the_same_run_under_raise_mode_loses_the_current_checkpoint() -> None:
    """The counter-example that shows what the verdict path buys.

    Same agent, same budget, ``on_exceeded="raise"``: the exception escapes,
    and whatever checkpoint survives does NOT reflect the state at the moment
    the ceiling was crossed. This test documents the old behaviour rather than
    endorsing it — it is why ``"stop"`` exists.
    """
    budget = Budget(max_cost_usd="1.00")  # default: raise
    cp = _fresh_checkpointer()
    agent = Agent("spender", "m", cognition=ReActCognition(tools=[echo], checkpointer=cp))

    with pytest.raises(MeterExceeded):
        asyncio.run(agent.run("go", _react_ctx(budget, cp)))

    saved = asyncio.run(cp.resume("budget-run"))
    assert saved is not None
    # Marked ``running``, not ``suspended`` — auto-resume reads this status to
    # tell "engine in motion" from "waiting on the world", so the stopped run
    # looks like one that is still going.
    assert saved.status == "running"
    # And it is a whole iteration behind: the second turn — the one that
    # actually crossed the ceiling — never reached a ``_save``.
    assert saved.state["iteration"] == 1

    # The damage, stated numerically: $1.20 left the account, and the durable
    # record accounts for $0.60 of it. Half the spend is unrecoverable.
    assert_money(budget.spent(), "1.20")
    assert saved.state["usage"]["cost"] == pytest.approx(0.60)


def test_a_stopped_run_can_be_resumed_after_the_ceiling_is_raised() -> None:
    """"Recoverable" has to mean it actually resumes, not just that a file
    exists. Raise the ceiling, re-drive the same correlation id, and the run
    continues from the checkpoint rather than restarting."""
    cp = _fresh_checkpointer()
    agent = Agent("spender", "m", cognition=ReActCognition(tools=[echo], checkpointer=cp))

    tight = Budget(max_cost_usd="1.00", on_exceeded="stop")
    first = asyncio.run(agent.run("go", _react_ctx(tight, cp)))
    assert first.stop_reason == "budget_exhausted"
    saved = asyncio.run(cp.resume("budget-run"))
    assert saved is not None
    messages_at_stop = len(saved.state["messages"])

    # Operator raises the ceiling and re-runs the same correlation id. The
    # cognition's ``_load`` rehydrates rather than starting fresh.
    roomy = Budget(max_cost_usd="100.00", on_exceeded="stop")
    second = asyncio.run(agent.run("go", _react_ctx(roomy, cp)))
    assert second.stop_reason in ("complete", "max_iterations", "budget_exhausted")
    resumed = asyncio.run(cp.resume("budget-run"))
    if resumed is not None:
        assert len(resumed.state["messages"]) > messages_at_stop, "the run restarted from scratch"


def test_single_call_cognition_also_stops_cleanly() -> None:
    """No checkpointer on this cognition — a single call has no mid-flight
    state worth resuming — so "recoverable" means a typed terminal event with
    the text and the spend, instead of an exception discarding both."""

    class _Pricey:
        async def stream(self, **_kw):
            yield Delta(text="partial answer", model="p", provider="fake")
            yield Delta(usage=Usage(10, 5, 5.0), finish_reason="stop", model="p", provider="fake")

    budget = Budget(max_cost_usd="1.00", on_exceeded="stop")
    ctx = make_test_ctx(llm=_Pricey(), budget=budget, chat_middleware=[meter()])
    result = asyncio.run(Agent("a", "m").run("go", ctx))

    assert result.stop_reason == "budget_exhausted"
    assert result.output == "partial answer"
    assert_money(budget.spent(), "5.00")


# ── Quota keeps protocol parity ──────────────────────────────────────────────


def test_quota_returns_verdicts_too() -> None:
    """Two ``Meter`` impls, one protocol. Before this change ``Budget`` raised
    on charge and ``Quota`` never checked at all — the middleware iterated
    both uniformly and got two different behaviours."""
    q = Quota(max_rpm=1, clock=lambda: 1000.0, on_exceeded="stop")
    assert _run(q.guard(_call(Scope(1, 1)))).ok is True
    assert _run(q.guard(_call(Scope(1, 1)))).ok is False
    assert _run(q.charge(_call(Scope(1, 1)), Usage(1, 1, 0.01))).ok is True


def test_quota_window_spend_is_exact() -> None:
    q = Quota(clock=lambda: 1000.0)
    for _ in range(100):
        _run(q.charge(_call(Scope(1, 1)), Usage(0, 0, 0.01)))
    assert_money(q.spent_in_window("org1:dom1"), "1.00", label="quota window")


def test_quota_over_precise_ceiling_is_refused() -> None:
    with pytest.raises(MoneyPrecisionError):
        Quota(max_usd="0.00000001")


# ── regressions found during review ──────────────────────────────────────────


def test_resuming_an_unraised_ceiling_costs_nothing() -> None:
    """Regression: each retry of an exhausted run used to burn another call.

    ``guard()`` returns a not-ok verdict under ``on_exceeded="stop"``, but the
    meter middleware deliberately ignores it (a middleware cannot write a
    checkpoint, so acting on the verdict is the cognition's job). With only a
    POST-call check, every retry made one more chat call before noticing —
    $0.60 a go, forever. The loop now pre-flights the budget.
    """
    cp = _fresh_checkpointer()
    budget = Budget(max_cost_usd="1.00", on_exceeded="stop")
    llm = _CostlyLLM()
    agent = Agent("spender", "m", cognition=ReActCognition(tools=[echo], checkpointer=cp))

    def go():
        return asyncio.run(
            agent.run(
                "go",
                make_test_ctx(
                    llm=llm,
                    budget=budget,
                    checkpointer=cp,
                    chat_middleware=[meter()],
                    correlation_id="budget-run",
                ),
            )
        )

    first = go()
    assert first.stop_reason == "budget_exhausted"
    calls_at_stop, spent_at_stop = llm.calls, budget.spent()

    for _ in range(3):
        again = go()
        assert again.stop_reason == "budget_exhausted"
    assert llm.calls == calls_at_stop, "a retry against an unraised ceiling made an LLM call"
    assert budget.spent() == spent_at_stop, "a retry against an unraised ceiling spent money"


def test_single_call_also_pre_flights_the_budget() -> None:
    """Same guard on the other cognition — an already-exhausted budget must
    not buy one more call just to discover it is exhausted."""

    class _Counting:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **_kw):
            self.calls += 1
            yield Delta(text="hi", model="m", provider="fake")
            yield Delta(usage=Usage(1, 1, 5.0), finish_reason="stop", model="m", provider="fake")

    llm = _Counting()
    budget = Budget(max_cost_usd="1.00", on_exceeded="stop", spent_usd=99.0)
    ctx = make_test_ctx(llm=llm, budget=budget, chat_middleware=[meter()])

    result = asyncio.run(Agent("a", "m").run("go", ctx))
    assert result.stop_reason == "budget_exhausted"
    assert llm.calls == 0, "an already-exhausted budget still made a call"


def test_raising_the_ceiling_after_construction_actually_raises_it() -> None:
    """Regression: caching the normalised ceiling in ``__post_init__`` made
    ``budget.max_cost_usd = 10.0`` silently do nothing — which is exactly what
    the spend recipe tells an operator to do to resume an exhausted run."""
    b = Budget(max_cost_usd=1.0)
    assert_money(b.ceiling(), "1.00", label="ceiling")
    assert_money(b.remaining(), "1.00", label="remaining")

    b.max_cost_usd = 10.0
    assert_money(b.ceiling(), "10.00", label="reassigned ceiling")
    assert_money(b.remaining(), "10.00")
    assert b.exhausted() is False

    b.max_cost_usd = None  # unlimited
    assert b.ceiling() is None and b.remaining() is None


def test_raising_a_quota_ceiling_after_construction_works_too() -> None:
    q = Quota(max_usd=1.0)
    q.max_usd = 5.0
    assert_money(q.ceiling(), "5.00", label="quota ceiling")


def test_a_post_construction_ceiling_is_quantized_not_raised() -> None:
    """``ceiling()`` is reached from inside ``charge()``. Raising
    ``MoneyPrecisionError`` from a read path would abort a run mid-flight —
    the exact unrecoverable abort being designed out. Strict refusal stays at
    construction, where it is free to fix."""
    b = Budget(max_cost_usd=1.0)
    b.max_cost_usd = "0.00000001"
    assert_money(b.ceiling(), "0.000000", label="post-hoc ceiling")  # quantized, no exception
    _run(b.charge(_call(), Usage(0, 0, 0.0)))  # and a charge still completes


def test_charge_verdict_uses_decimal_names_not_the_float_mirror_names() -> None:
    """``Budget.spent_usd`` is a float and ``Charge.spent`` is a Decimal. One
    NAME meaning two types across two classes is how a float creeps back into
    a ledger, so the Decimal accessors are ``spent``/``remaining`` everywhere
    and ``*_usd`` is float everywhere."""
    b = Budget(max_cost_usd="1.00", on_exceeded="stop")
    verdict = _run(b.charge(_call(), Usage(0, 0, 0.25)))
    assert isinstance(verdict.spent, Decimal) and isinstance(verdict.remaining, Decimal)
    assert_no_float_money(verdict.spent, verdict.remaining, label="Charge")
    assert isinstance(b.spent_usd, float) and isinstance(b.remaining_usd(), float)
    assert not hasattr(verdict, "spent_usd"), "Charge must not reuse the float mirror's name"


# ── tests written to kill specific surviving mutants ─────────────────────────
#
# Each of these exists because a deliberate break in ``meter.py`` passed the
# whole suite. They are the cases where the numbers that had been chosen
# happened to be the numbers that hide the bug.


def test_a_large_balance_still_records_a_sub_cent_charge() -> None:
    """Kills: ``_spent`` accumulating through ``float``.

    Every cent-scale test in this module survived that mutation, because a
    float round-trip through six-decimal quantization recovers the same value
    at small magnitudes. Float64 carries ~15-16 significant digits, so the
    error only becomes visible once the INTEGER part is large enough to crowd
    the fraction out:

        Decimal path : 10000000000.000001
        float   path : 10000000000.000002

    A long-running enrichment job that has spent five figures and is still
    being charged per row lands exactly here — which is precisely the workload
    ``docs/mental-models/03`` describes.
    """
    b = Budget(spent_usd=10_000_000_000.0)
    _run(b.charge(_call(), Usage(0, 0, 0.000001)))
    assert_money(b.spent(), "10000000000.000001", label="big balance + sub-cent charge")


def test_a_non_binary_exact_float_ceiling_is_accepted() -> None:
    """Kills: ``to_money`` building ``Decimal(value)`` instead of ``Decimal(str(value))``.

    ``Decimal(0.01)`` is the BINARY value —
    ``0.01000000000000000020816681711721685...`` — which carries far more than
    six decimal places, so a strict ceiling built from it raises
    ``MoneyPrecisionError`` and ``Budget(max_cost_usd=0.01)`` becomes
    unconstructable.

    The existing spelling test used ``1.5`` and ``2``, and both are exactly
    representable in binary, so the mutant sailed through. The number had to
    be one that ISN'T.
    """
    assert_money(Budget(max_cost_usd=0.01).ceiling(), "0.01", label="0.01 ceiling")
    assert_money(Budget(max_cost_usd=0.07).ceiling(), "0.07", label="0.07 ceiling")
    assert_money(to_money(0.1), "0.1", label="to_money(0.1)")
    # The classic: three tenths as a float sum is 0.30000000000000004.
    assert_money(to_money(0.1 + 0.2), "0.3", label="to_money(0.1 + 0.2)")


def test_charges_of_a_non_binary_exact_amount_still_reconcile() -> None:
    """The cent case everyone tests is ``0.01``; the case that actually breaks
    float ledgers is a repeating-binary amount like ``0.07``."""
    b = Budget()
    for _ in range(100):
        _run(b.charge(_call(), Usage(0, 0, 0.07)))
    assert_money(b.spent(), "7.00", label="100 x $0.07")


def test_guard_alone_refuses_when_already_over_the_ceiling() -> None:
    """Kills: ``guard()`` unconditionally returning ok.

    Every existing ceiling test drove ``charge()``, which also refuses — so
    neutering ``guard()`` changed nothing observable. But ``guard`` is the
    PRE-flight half: the meter middleware calls it in ``on_request``, before
    the provider is touched. A guard that always passes means an
    already-exhausted budget still pays for one more call to rediscover it,
    which is the exact defect the cognitions' pre-flight checks exist to
    prevent. Asserted on ``guard`` in isolation, with no ``charge`` nearby.
    """
    b = Budget(max_cost_usd="0.01", on_exceeded="stop")
    assert _run(b.guard(_call())).ok is True  # fresh budget: nothing spent

    _run(b.charge(_call(), Usage(0, 0, 0.05)))  # now well over

    verdict = _run(b.guard(_call()))
    assert verdict.ok is False, "guard() passed a budget that is already over its ceiling"
    assert "0.05" in verdict.reason and "0.01" in verdict.reason

    # And under the default raise-mode it must actually raise from guard.
    strict = Budget(max_cost_usd="0.01", spent_usd=0.05)
    with pytest.raises(MeterExceeded):
        _run(strict.guard(_call()))


def test_guard_refuses_on_the_call_ceiling_too() -> None:
    """The other axis, same reasoning."""
    b = Budget(max_calls=1, on_exceeded="stop")
    _run(b.charge(_call(), Usage()))
    _run(b.charge(_call(), Usage()))  # calls == 2 > 1
    assert _run(b.guard(_call())).ok is False


def test_spent_cents_rounds_half_up_not_down() -> None:
    """Kills: ``spent_cents`` quantizing with ``ROUND_DOWN``.

    Nothing asserted the rounding MODE, so silently truncating every invoice
    passed the suite. Half a cent is the discriminating case: half-up bills
    $0.01, truncation bills $0.00. Applied across a month of runs that is
    systematic under-billing, and it is invisible unless a test names the
    boundary.
    """
    b = Budget(spent_usd=0.005)
    assert_money(b.spent_cents(), "0.01", label="half a cent rounds up")

    below = Budget(spent_usd=0.004)
    assert_money(below.spent_cents(), "0.00", label="just under half a cent rounds down")

    # And the exact-boundary case one place further out, so a change to
    # MONEY_SCALE cannot quietly shift the rounding point.
    b2 = Budget(spent_usd=1.234567)
    assert_money(b2.spent_cents(), "1.23", label="ordinary truncation-vs-rounding case")
    b3 = Budget(spent_usd=1.235)
    assert_money(b3.spent_cents(), "1.24", label="round-half-up at the cent boundary")
