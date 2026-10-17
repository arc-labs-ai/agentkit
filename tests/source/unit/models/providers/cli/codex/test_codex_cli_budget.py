"""What a Codex run costs, and how that number reaches the framework's books.

WHY THIS FILE EXISTS
--------------------
This cognition bypasses the ``Invoker``, so the ``meter()`` middleware never
sees a CLI run's usage. Every meter on the context therefore stays at zero
unless this module charges them itself — which is the silent ``$0.00`` ledger
the Claude cognition shipped once: a documented safety mechanism that was wired,
visible in the code, and doing nothing.

``codex`` makes the second half harder than its sibling does. There is no
``--max-budget-usd``, so the CLI cannot stop itself mid-flight against the run's
remaining headroom. All the budget can do is refuse to spawn when the money is
already gone — and that refusal is the ONLY in-flight power it has, so dropping
it turns an exhausted budget into a completed run that overspent, with
``stop_reason="complete"`` on it.

And the CLI reports no cost AT ALL. ``ProviderUsage.cost_usd`` here is always
computed, so ``tests.evals["cost_source"]`` must never say ``reported``:
"provider billing receipt" and "our best guess from a public price table" are
the two things a caller invoices against, and labelling the second as the first
is a number nobody can audit back to anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agentkit.core.contracts.types import CostSource, ProviderUsage, StreamEvent
from agentkit.core.testing.fakes import FakeCodexCli, codex_turn
from agentkit.models.providers.cli.codex.driver import CodexCliCognition


@dataclass
class _Agent:
    prompt: str | None = None


@dataclass
class _Meter:
    """A meter that records what it was charged, in charge order."""

    charges: list[ProviderUsage] = field(default_factory=list)

    async def charge(self, call: Any, usage: ProviderUsage) -> None:
        assert call.ctx is not None, "a Quota partitions on call.ctx.scope; None would charge the wrong books"
        self.charges.append(usage)


@dataclass
class _AngryMeter:
    async def charge(self, call: Any, usage: ProviderUsage) -> None:
        raise RuntimeError("ceiling crossed")


@dataclass
class _Budget:
    """Just the ``remaining()`` the pre-flight refusal reads."""

    headroom: float | None

    def remaining(self) -> float | None:
        return self.headroom


@dataclass
class _Ctx:
    budget: Any = None
    all_meters: tuple[Any, ...] = ()
    correlation_id: str = "run-1"
    invoker: Any = None

    def check_cancelled(self) -> None:
        return None


async def _drive(cognition: CodexCliCognition, ctx: Any) -> list[StreamEvent]:
    return [ev async for ev in cognition.drive(_Agent(), "go", ctx, None)]


def _final(events: list[StreamEvent]) -> Any:
    assert [e.type for e in events].count("final") == 1, [e.type for e in events]
    return events[-1].result


@pytest.mark.asyncio
async def test_what_the_cli_spent_reaches_the_meters_the_invoker_never_showed_it_to() -> None:
    """The ``meter()`` middleware sits on the ``Invoker``, and this cognition
    does not go through it, so without an explicit charge here every meter on
    the context reads zero for a run that really spent money. Nothing fails:
    the budget never trips, the quota never partitions, the ledger balances at
    $0.00 and the spend is simply invisible."""
    meter = _Meter()
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(1200, 1000, 40)))
    cognition = CodexCliCognition(spawn=cli, pricing=lambda model, usage: 0.25)

    result = _final(await _drive(cognition, _Ctx(all_meters=(meter,))))

    assert len(meter.charges) == 1, "the run's spend was never charged"
    charged = meter.charges[0]
    assert charged.cost_usd == 0.25
    # The meters and the terminal event must see the SAME number, cached prefix
    # already subtracted — a ledger and a result that disagree is unauditable.
    assert charged.input_tokens == 200
    assert charged.cache_read_tokens == 1000
    assert charged.cost_usd == result.usage.cost_usd
    assert charged.input_tokens == result.usage.input_tokens


@pytest.mark.asyncio
async def test_an_exhausted_budget_refuses_to_spawn_instead_of_overspending() -> None:
    """This is the WHOLE of the budget's in-flight power here — ``codex`` has no
    ``--max-budget-usd`` to stop itself with. Skip the pre-flight refusal and a
    run with no money left spawns anyway, spends, and comes back as a completed
    run; the resumable ``budget_exhausted`` stop reason, which tells a caller to
    raise the ceiling and try again, becomes a success that quietly went over."""
    cli = FakeCodexCli.script(codex_turn(text="should never run", usage=(10, 0, 5)))
    cognition = CodexCliCognition(spawn=cli)

    result = _final(await _drive(cognition, _Ctx(budget=_Budget(0.0))))

    assert cli.spawns == 0, "the CLI was spawned with no budget left"
    assert result.evals["stop_reason"] == "budget_exhausted"
    assert result.stop_reason == "budget_exhausted"
    assert result.partial is True
    assert result.output == ""


@pytest.mark.asyncio
async def test_a_budget_with_headroom_spawns_and_is_charged_what_the_run_cost() -> None:
    """The refusal has to be a refusal and not a brake: a budget that still has
    money must not cost the caller the run. Paired with the test above because
    "refuses when exhausted" is only half a guarantee — the other half is that
    it does nothing otherwise."""
    meter = _Meter()
    cli = FakeCodexCli.script(codex_turn(text="ran", usage=(10, 0, 5)))
    cognition = CodexCliCognition(spawn=cli, pricing=lambda model, usage: 0.01)

    result = _final(await _drive(cognition, _Ctx(budget=_Budget(5.0), all_meters=(meter,))))

    assert cli.spawns == 1
    assert result.output == "ran"
    assert [c.cost_usd for c in meter.charges] == [0.01]


@pytest.mark.asyncio
async def test_a_context_with_no_ceiling_configured_is_not_treated_as_exhausted() -> None:
    """``remaining()`` returning ``None`` means "no ceiling", which is the common
    case. Reading it as zero headroom would refuse every run on every context
    that merely HAS a budget object."""
    cli = FakeCodexCli.script(codex_turn(text="ran", usage=(10, 0, 5)))
    result = _final(await _drive(CodexCliCognition(spawn=cli), _Ctx(budget=_Budget(None))))

    assert cli.spawns == 1
    assert result.output == "ran"


@pytest.mark.asyncio
async def test_meter_spend_false_opts_a_run_out_of_both_the_refusal_and_the_charge() -> None:
    """One flag, both ends, deliberately: a caller who is accounting for this
    run somewhere else must not get half of the wiring — charged but never
    refused, or refused by a ceiling nothing is being charged against."""
    meter = _Meter()
    cli = FakeCodexCli.script(codex_turn(text="ran anyway", usage=(10, 0, 5)))
    cognition = CodexCliCognition(spawn=cli, meter_spend=False)

    result = _final(await _drive(cognition, _Ctx(budget=_Budget(0.0), all_meters=(meter,))))

    assert cli.spawns == 1
    assert result.output == "ran anyway"
    assert meter.charges == []


@pytest.mark.asyncio
async def test_a_computed_cost_is_never_presented_as_a_provider_billing_receipt() -> None:
    """``codex``'s ``turn.completed`` carries token counts and nothing else, so
    every cost on this path is an estimate. Stamping it ``reported`` tells a
    caller they are holding a provider's own figure, and they bill against a
    public price table — or against a host callable they passed for an entirely
    different purpose — believing it came off an invoice."""
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
    host = _final(await _drive(CodexCliCognition(spawn=cli, pricing=lambda m, u: 0.5), _Ctx()))

    assert host.evals["cost_source"] == CostSource.HOST_PRICING
    assert host.usage.cost_source == CostSource.HOST_PRICING
    assert host.evals["cost_source"] != CostSource.REPORTED

    table_cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
    table = _final(await _drive(CodexCliCognition(spawn=table_cli, model="gpt-4.1"), _Ctx()))

    assert table.evals["cost_source"] == CostSource.PRICE_TABLE
    assert table.usage.cost_source == CostSource.PRICE_TABLE
    assert table.usage.cost_usd is not None
    assert table.evals["cost_source"] != CostSource.REPORTED


@pytest.mark.asyncio
async def test_a_model_the_price_table_does_not_know_is_unpriced_rather_than_free() -> None:
    """Most Codex models are not in the public table. A priced zero would put a
    real run on the books as costing nothing, which is a worse lie than "we do
    not know" — ``None`` plus ``unpriced`` is the answer a caller can branch
    on."""
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
    result = _final(await _drive(CodexCliCognition(spawn=cli, model="gpt-5-codex-unknown"), _Ctx()))

    assert result.usage.cost_usd is None
    assert result.evals["cost_source"] == CostSource.UNPRICED
    # The token facts are real and survive the missing price.
    assert result.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_a_pricing_callable_that_blows_up_leaves_the_run_unpriced_not_answerless() -> None:
    """The tokens are real and the terminal-event guarantee is unconditional, so
    a caller's own broken ``pricing=`` must not cost them the run's answer. It
    leaves the cost unknown, which is the only honest value, and keeps
    everything else."""

    def exploding(model: str | None, usage: ProviderUsage) -> float:
        raise ZeroDivisionError("rate card divided by zero")

    cli = FakeCodexCli.script(codex_turn(text="the answer survives", usage=(10, 0, 5)))
    result = _final(await _drive(CodexCliCognition(spawn=cli, pricing=exploding), _Ctx()))

    assert result.output == "the answer survives"
    assert result.usage.cost_usd is None
    assert result.evals["cost_source"] == CostSource.UNPRICED
    assert result.partial is False


@pytest.mark.asyncio
async def test_a_pricing_callable_returning_a_nonsense_number_is_refused() -> None:
    """A NaN or a negative cost reaching a meter is worse than no cost at all: a
    negative charge makes a budget go UP, and a NaN poisons every comparison the
    ledger makes against it afterwards."""
    for bad in (float("nan"), -1.0, float("inf")):
        cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
        cognition = CodexCliCognition(spawn=cli, pricing=lambda m, u, v=bad: v)
        result = _final(await _drive(cognition, _Ctx()))

        assert result.usage.cost_usd is None, bad
        assert result.evals["cost_source"] == CostSource.UNPRICED, bad


@pytest.mark.asyncio
async def test_a_meter_that_raises_is_reported_rather_than_costing_the_caller_the_answer() -> None:
    """The spend already happened and the run already produced an answer. A
    ceiling crossed on the LAST call is recorded and reported, not converted
    into a lost result — and a custom meter that simply misbehaves is contained
    for the same reason."""
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 0, 5)))
    cognition = CodexCliCognition(spawn=cli, pricing=lambda m, u: 0.01)

    result = _final(await _drive(cognition, _Ctx(all_meters=(_AngryMeter(),))))

    assert result.output == "ok"
    assert result.partial is False
    assert result.evals["meter_error"] == "RuntimeError: ceiling crossed"
