"""CLI spend is on the framework's books, and the framework's ceiling is on the CLI.

`ClaudeCliCognition` bypasses the `Invoker`, so the `meter()` middleware never
sees its usage. Every meter on the context therefore stayed at zero no matter
what the CLI spent — the class docstring admitted as much ("callers who need a
hard ceiling on CLI spend must impose it externally"). A $50 CLI run against a
$1 `Budget` completed happily and the ledger read `$0.00`. Same shape as the
`ActorBudget` that was wired to nothing.

Two halves, and both are needed:

* **Before**: the run's remaining headroom goes out as `--max-budget-usd`, so
  the CLI stops ITSELF mid-flight rather than being audited afterwards. An
  already-exhausted budget refuses to spawn at all.
* **After**: what it actually spent is charged to every meter on the context
  plus the per-actor envelope.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from agentkit import Agent, Scope, Usage
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.agents.control.budget import ActorBudget
from agentkit.context import WorkingContext
from agentkit.runtime import Budget, Quota, RunContext, Services
from agentkit.testing.fakes.ctx import FakeCtx
from tests._assertions import assert_money
from tests.agents.cognition.test_claude_cli import _FakeProcess, _line


def _stream(cost: float, *, tokens: tuple[int, int] = (1000, 500)) -> list[bytes]:
    return [
        _line({"type": "system", "subtype": "init", "session_id": "s"}),
        _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}),
        _line(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": "s",
                "duration_ms": 5,
                "total_cost_usd": cost,
                "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
            }
        ),
    ]


def _run(cog: ClaudeCliCognition, ctx: Any, lines: list[bytes]) -> tuple[Any, tuple[str, ...] | None]:
    proc = _FakeProcess(stdout_lines=lines)
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        agent = Agent(name="x", cognition=cog)

        async def _go() -> list[Any]:
            return [ev async for ev in cog.drive(agent, "t", ctx, WorkingContext())]

        events = asyncio.run(_go())
    return events[-1].result, (tuple(spawn.await_args.args) if spawn.await_args else None)


def _ctx(budget: Budget, **kw: Any) -> RunContext:
    return RunContext("run-1", Scope(), services=Services(), budget=budget, **kw)


# ── 1. the ceiling goes OUT ─────────────────────────────────────────────────


def test_the_runs_headroom_becomes_the_clis_own_cap() -> None:
    """Enforced by the CLI mid-run, not audited after the money is gone. The
    value is the remaining headroom, not the original ceiling — a second call
    on a partly-spent budget must not get the full amount again."""
    budget = Budget(max_cost_usd=2.50)
    asyncio.run(budget.charge(None, Usage(0, 0, 1.00)))

    _, argv = _run(ClaudeCliCognition(), _ctx(budget), _stream(0.25))

    assert argv is not None
    assert argv[argv.index("--max-budget-usd") + 1] == "1.500000"


def test_no_ceiling_means_no_flag() -> None:
    """A `Budget` with no `max_cost_usd` sets no limit, and inventing one here
    would impose a cap the caller never asked for."""
    _, argv = _run(ClaudeCliCognition(), _ctx(Budget()), _stream(0.25))
    assert argv is not None and "--max-budget-usd" not in argv


def test_a_ctx_without_a_budget_is_fine() -> None:
    """`FakeCtx` and hand-rolled contexts carry no budget at all."""
    result, argv = _run(ClaudeCliCognition(), FakeCtx(), _stream(0.25))
    assert argv is not None and "--max-budget-usd" not in argv
    assert result.stop_reason == "complete"


# ── 2. the spend comes BACK ─────────────────────────────────────────────────


def test_the_spend_lands_on_every_meter() -> None:
    """THE gap: the ledger used to read $0.00 for any amount of CLI spend."""
    budget = Budget(max_cost_usd=10.0)
    actor = ActorBudget(
        max_cost_usd=5.0, max_tokens=100_000, max_steps=10, max_wall_seconds=600.0
    )
    quota = Quota(max_usd=10.0)
    ctx = _ctx(budget, actor_budget=actor, meters=[quota])

    result, _ = _run(ClaudeCliCognition(), ctx, _stream(0.75))

    assert_money(budget.spent(), "0.750000")
    assert budget.usage.total_tokens == 1500
    assert_money(actor.used_cost(), "0.75")
    assert actor.used_tokens == 1500
    assert actor.used_steps == 1
    assert result.usage.cost_usd == 0.75


def test_a_tenant_quota_is_charged_through_the_call_shim() -> None:
    """`Quota` reads `call.ctx.scope.key()` to partition per tenant, so a bare
    `None` call would charge the budget and crash the quota. Pinned because
    the shim is invisible until a Quota is wired."""
    quota = Quota(max_usd=10.0)
    ctx = RunContext(
        "run-1", Scope(org_id=7), services=Services(), budget=Budget(), meters=[quota]
    )
    _run(ClaudeCliCognition(), ctx, _stream(0.40))
    assert_money(quota.spent_in_window(Scope(org_id=7).key()), "0.400000")


def test_a_ceiling_crossed_by_this_very_run_is_recorded_not_raised() -> None:
    """The money is spent and the answer exists. Converting the charge into an
    exception would lose a result the caller already paid for — and would break
    the terminal-event guarantee on top."""
    budget = Budget(max_cost_usd=1.0, on_exceeded="raise")
    result, _ = _run(ClaudeCliCognition(), _ctx(budget), _stream(3.00))

    assert result.stop_reason == "complete"  # the run itself succeeded
    assert "MeterExceeded" in result.evals["meter_error"]
    assert_money(budget.spent(), "3.000000")  # the books still tell the truth


# ── 3. an exhausted budget short-circuits ───────────────────────────────────


def test_an_exhausted_budget_refuses_to_spawn() -> None:
    """Two to five seconds of CLI warm-up to be told what we already know. The
    stop reason is the resumable one: raise the ceiling and run again."""
    budget = Budget(max_cost_usd=0.10)
    asyncio.run(budget.charge(None, Usage(0, 0, 0.10)))

    result, argv = _run(ClaudeCliCognition(), _ctx(budget), _stream(5.0))

    assert argv is None, "no subprocess should have been spawned"
    assert result.evals["stop_reason"] == "budget_exhausted"
    assert result.stop_reason == "budget_exhausted"
    assert result.is_resumable and result.partial


def test_the_refusal_still_yields_exactly_one_terminal_event() -> None:
    """The guarantee this whole cognition is built around holds on the new path
    too."""
    budget = Budget(max_cost_usd=0.10)
    asyncio.run(budget.charge(None, Usage(0, 0, 0.10)))
    cog = ClaudeCliCognition()

    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=AssertionError("must not spawn")),
    ):

        async def _go() -> list[Any]:
            return [
                ev async for ev in cog.drive(Agent(name="x"), "t", _ctx(budget), WorkingContext())
            ]

        events = asyncio.run(_go())

    assert [e.type for e in events] == ["final"]


# ── 4. the opt-out ──────────────────────────────────────────────────────────


def test_meter_spend_false_touches_neither_end() -> None:
    """A warm-up call or an eval harness with its own accounting should not
    draw on the shared envelope — and must not be capped by it either."""
    budget = Budget(max_cost_usd=2.0)
    result, argv = _run(ClaudeCliCognition(meter_spend=False), _ctx(budget), _stream(0.9))

    assert argv is not None and "--max-budget-usd" not in argv
    assert_money(budget.spent(), "0.000000")
    assert result.usage.cost_usd == 0.9  # still reported, just not charged


def test_meter_spend_false_does_not_short_circuit_an_exhausted_budget() -> None:
    """Opting out of the envelope means opting out of ITS refusal too."""
    budget = Budget(max_cost_usd=0.10)
    asyncio.run(budget.charge(None, Usage(0, 0, 0.10)))
    _, argv = _run(ClaudeCliCognition(meter_spend=False), _ctx(budget), _stream(1.0))
    assert argv is not None
