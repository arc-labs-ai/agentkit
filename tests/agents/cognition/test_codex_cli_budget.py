"""Codex spend is on the framework's books — and only half of it can be enforced.

``CodexCliCognition`` bypasses the ``Invoker``, so the ``meter()`` middleware
never sees its usage and every meter on the context would stay at zero no
matter what the run spent. That is the same silent-$0.00 ledger the Claude
cognition shipped once, and the charge happens in ``_charge_meters`` for the
same reason.

What is DIFFERENT, and is the thing a reader moving between the two cognitions
will get wrong: there is no ``--max-budget-usd`` on ``codex``. The Claude
cognition hands the run's remaining headroom to the CLI so it stops itself
mid-flight; nothing here can. All this cognition can do is refuse to spawn when
the budget is already gone and charge what the run cost once it ends, so a run
that starts with $0.01 of headroom can spend $5 and only be caught afterwards.
The tests below say that in both directions, because "the budget is wired"
reads as a hard ceiling and here it is not one.

And the CLI reports no cost at all — its ``turn.completed`` carries token counts
and nothing else. So ``Usage.cost_usd`` is COMPUTED, ``evals["cost_source"]`` is
always ``"estimated"``, and an unpriced model costs 0.00. A caller who needs a
number they can bill against injects ``pricing=``.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentkit import Agent, Scope, Usage
from agentkit.agents.cognition import CodexCliCognition
from agentkit.agents.control.budget import ActorBudget
from agentkit.runtime import Budget, Quota, RunContext, Services
from agentkit.testing.fakes import FakeCodexCli, codex_turn
from agentkit.testing.fakes.ctx import FakeCtx
from tests._assertions import assert_money
from tests.agents.cognition.test_codex_cli import drive, final_of


def ctx_with(budget: Budget, **kw: Any) -> RunContext:
    return RunContext("run-1", Scope(), services=Services(), budget=budget, **kw)


def flat(usd: float) -> Any:
    """A ``pricing=`` callable that charges a fixed amount, so a test can assert
    on the WIRING without also asserting on a public price table that is
    documented to go stale."""

    def price(model: str | None, usage: Usage) -> float:
        del model, usage
        return usd

    return price


# ─────────────────────────────────────────────────────────────────────────────
# 1. no ceiling goes out — and the tests say so on purpose
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_budget_flag_is_sent_because_the_cli_has_none() -> None:
    """A negative assertion that earns its place. If a future ``codex`` gains a
    spend cap, THIS is the test that fails and points at the field to add —
    rather than the cognition quietly continuing to enforce nothing."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1000, 0, 500)))
    await drive(CodexCliCognition(spawn=cli), ctx=ctx_with(Budget(max_cost_usd=2.50)))

    argv = list(cli.invocations[-1].argv)
    assert not [a for a in argv if "budget" in a.lower()]


@pytest.mark.asyncio
async def test_an_exhausted_budget_refuses_to_spawn_at_all() -> None:
    """A pre-flight refusal: spawning a subprocess to be told what we already
    know costs seconds of CLI warm-up. And the reason is the RESUMABLE one —
    raise the ceiling and run again — not a failure."""
    budget = Budget(max_cost_usd=1.00)
    await budget.charge(None, Usage(0, 0, 1.00))
    cli = FakeCodexCli.script(codex_turn(text="never runs", usage=(1, 0, 1)))

    result = final_of(await drive(CodexCliCognition(spawn=cli), ctx=ctx_with(budget)))

    assert cli.spawns == 0
    assert result.partial is True
    assert result.evals["stop_reason"] == "budget_exhausted"
    assert result.stop_reason == "budget_exhausted"
    assert result.is_resumable is True


@pytest.mark.asyncio
async def test_a_budget_with_headroom_spawns() -> None:
    budget = Budget(max_cost_usd=1.00)
    cli = FakeCodexCli.script(codex_turn(text="ran", usage=(1, 0, 1)))
    result = final_of(await drive(CodexCliCognition(spawn=cli), ctx=ctx_with(budget)))
    assert cli.spawns == 1
    assert result.output == "ran"


@pytest.mark.asyncio
async def test_a_budget_with_no_ceiling_does_not_refuse() -> None:
    """``Budget()`` sets no ``max_cost_usd``. Inventing a limit here would
    impose a cap the caller never asked for."""
    cli = FakeCodexCli.script(codex_turn(text="ran", usage=(1, 0, 1)))
    result = final_of(await drive(CodexCliCognition(spawn=cli), ctx=ctx_with(Budget())))
    assert result.output == "ran"


@pytest.mark.asyncio
async def test_a_ctx_with_no_budget_is_fine() -> None:
    """``FakeCtx`` and hand-rolled contexts carry no budget at all, and a
    cognition that required one would be unusable in half the framework's own
    tests."""
    cli = FakeCodexCli.script(codex_turn(text="ran", usage=(1, 0, 1)))
    result = final_of(await drive(CodexCliCognition(spawn=cli), ctx=FakeCtx()))
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
async def test_meter_spend_false_opts_out_of_both_ends() -> None:
    """A warm-up call, or an eval harness with its own accounting: it should
    neither be refused by the shared envelope nor draw on it."""
    budget = Budget(max_cost_usd=1.00)
    await budget.charge(None, Usage(0, 0, 1.00))
    cli = FakeCodexCli.script(codex_turn(text="ran anyway", usage=(1000, 0, 500)))

    result = final_of(
        await drive(
            CodexCliCognition(meter_spend=False, pricing=flat(0.50), spawn=cli),
            ctx=ctx_with(budget),
        )
    )

    assert result.output == "ran anyway"
    # Charged nothing beyond the 1.00 the test put there itself.
    assert_money(budget.spent(), "1.000000")


# ─────────────────────────────────────────────────────────────────────────────
# 2. the spend comes back
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_spend_lands_on_every_meter() -> None:
    """THE gap: without this the ledger reads $0.00 for any amount of CLI
    spend, because the middleware that would have charged it never runs."""
    budget = Budget(max_cost_usd=10.0)
    actor = ActorBudget(max_cost_usd=5.0, max_tokens=100_000, max_steps=10, max_wall_seconds=600.0)
    quota = Quota(max_usd=10.0)
    ctx = ctx_with(budget, actor_budget=actor, meters=[quota])
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1200, 1000, 500)))

    await drive(CodexCliCognition(pricing=flat(0.75), spawn=cli), ctx=ctx)

    assert_money(budget.spent(), "0.750000")
    # Fresh input + output. The 1000 cached tokens are on
    # ``cache_read_tokens`` and are deliberately not in ``total_tokens``.
    assert budget.usage.total_tokens == 700
    assert_money(actor.used_cost(), "0.75")
    assert actor.used_tokens == 700
    assert actor.used_steps == 1
    # ``Quota`` partitions per tenant, so it is read by scope key — which is
    # also the reason ``_CliCall`` carries a ``ctx`` at all rather than being a
    # bare ``None``: ``Quota.charge`` reads ``call.ctx.scope.key()``.
    assert_money(quota.spent_in_window(Scope().key()), "0.750000")


@pytest.mark.asyncio
async def test_a_meter_that_refuses_the_charge_does_not_cost_the_answer() -> None:
    """The spend already happened and the run already produced an answer; the
    terminal-event guarantee says the caller gets that answer. A ceiling crossed
    on the LAST call is recorded and reported, not converted into a lost
    result."""
    budget = Budget(max_cost_usd=0.10)
    cli = FakeCodexCli.script(codex_turn(text="the answer", usage=(1, 0, 1)))

    result = final_of(
        await drive(CodexCliCognition(pricing=flat(5.00), spawn=cli), ctx=ctx_with(budget))
    )

    assert result.output == "the answer"
    assert "meter_error" in result.evals
    assert "MeterExceeded" in result.evals["meter_error"]


@pytest.mark.asyncio
async def test_a_cancelled_run_still_charges_what_it_used() -> None:
    """Tokens spent before a cancel are spent. Skipping the charge on a partial
    run is how a retry loop bills nothing while costing money."""
    from tests.agents.cognition.test_codex_cli import CancellingCtx

    class _Cancelling(CancellingCtx):
        def __init__(self, budget: Budget) -> None:
            super().__init__(after=3)
            self.budget = budget
            self.all_meters = [budget]

    budget = Budget(max_cost_usd=10.0)
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "turn.completed", "usage": {"input_tokens": 900, "output_tokens": 100}},
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "partial"}},
            {"type": "item.completed", "item": {"id": "n", "type": "agent_message", "text": " more"}},
        ]
    )
    result = final_of(
        await drive(CodexCliCognition(pricing=flat(0.20), spawn=cli), ctx=_Cancelling(budget))
    )

    assert result.evals["stop_reason"] == "cancelled"
    assert_money(budget.spent(), "0.200000")


# ─────────────────────────────────────────────────────────────────────────────
# 3. where the cost number comes from, and what it is worth
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_cost_is_always_marked_estimated() -> None:
    """The CLI reports none, so a caller reading ``usage.cost_usd`` without
    reading this would treat a table lookup as a billed number. Marked on every
    result rather than only on the uncertain ones — there are no certain
    ones."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    result = final_of(await drive(CodexCliCognition(pricing=flat(0.01), spawn=cli)))
    assert result.evals["cost_source"] == "estimated"


@pytest.mark.asyncio
async def test_an_unpriced_model_costs_zero_rather_than_guessing() -> None:
    """The framework's public table does not know most Codex models, and
    ``pricing.cost`` returns 0.0 for a model it has never heard of rather than
    inventing a rate. Zero is the honest answer AND a trap, which is what
    ``cost_source`` is for."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1000, 0, 500)))
    result = final_of(await drive(CodexCliCognition(model="gpt-5-codex", spawn=cli)))
    assert result.usage.cost_usd == 0.0
    assert result.usage.total_tokens == 1500
    assert result.evals["cost_source"] == "estimated"


@pytest.mark.asyncio
async def test_the_default_price_table_is_used_when_it_knows_the_model() -> None:
    """Wired to the framework's own ``pricing.cost``, not to a private copy —
    so a caller who registers a rate there gets it here too."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1_000_000, 0, 0)))
    result = final_of(await drive(CodexCliCognition(model="gpt-4.1", spawn=cli)))
    # 1M input tokens of gpt-4.1 at the table's $2.00/1M.
    assert result.usage.cost_usd == pytest.approx(2.00)


@pytest.mark.asyncio
async def test_the_model_the_cli_reports_wins_over_the_one_we_asked_for() -> None:
    """Codex resolves aliases and profiles on its side, so what ran can differ
    from what was requested — and the cost has to be priced against what
    ran."""
    seen: list[str | None] = []

    def price(model: str | None, usage: Usage) -> float:
        del usage
        seen.append(model)
        return 0.0

    cli = FakeCodexCli.script(
        [
            {"id": "0", "msg": {"type": "session_configured", "session_id": "s", "model": "gpt-4.1-mini"}},
            {"id": "1", "msg": {"type": "agent_message", "message": "x"}},
            {"id": "2", "msg": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1}}}},
        ]
    )
    result = final_of(await drive(CodexCliCognition(model="gpt-4.1", pricing=price, spawn=cli)))

    assert seen == ["gpt-4.1-mini"]
    assert result.evals["cli_model"] == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_a_pricing_callable_that_raises_does_not_cost_the_run() -> None:
    """A caller-supplied hook that blows up must not lose them the answer. The
    tokens are real and the terminal-event guarantee is unconditional, so the
    cost falls back to zero and everything else survives."""

    def price(model: str | None, usage: Usage) -> float:
        raise RuntimeError("rate service unreachable")

    cli = FakeCodexCli.script(codex_turn(text="the answer", usage=(10, 0, 5)))
    result = final_of(await drive(CodexCliCognition(pricing=price, spawn=cli)))

    assert result.output == "the answer"
    assert result.usage.cost_usd == 0.0
    assert result.usage.input_tokens == 10
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
async def test_the_final_event_and_the_result_carry_the_same_usage() -> None:
    """The cost is computed inside ``_finalise``, and the terminal event's own
    ``usage`` field has to be the priced copy — not the raw one folded off the
    stream, which is what a consumer reading ``ev.usage`` would otherwise get.
    """
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    events = await drive(CodexCliCognition(pricing=flat(0.42), spawn=cli))
    terminal = events[-1]
    assert terminal.usage is not None
    assert terminal.usage.cost_usd == 0.42
    assert terminal.usage == terminal.result.usage


@pytest.mark.asyncio
async def test_an_agent_run_reports_the_same_usage_the_cognition_computed() -> None:
    """Through the real ``Agent``, because that is the path a caller takes and
    it re-wraps the result."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1200, 1000, 40)))
    cog = CodexCliCognition(pricing=flat(0.05), spawn=cli)
    result = await Agent(name="local", cognition=cog).run("t", FakeCtx())

    assert result.usage.cost_usd == 0.05
    assert result.usage.input_tokens == 200
    assert result.usage.cache_read_tokens == 1000
