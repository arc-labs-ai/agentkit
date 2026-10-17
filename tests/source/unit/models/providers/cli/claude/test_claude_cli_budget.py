"""Who pays for a CLI-delegated run, and who finds out.

WHY THIS FILE EXISTS
--------------------
``ClaudeCliCognition`` bypasses the ``Invoker``, so the ``meter()`` middleware
never sees its usage. That is how a documented safety mechanism ends up doing
nothing, and this cognition has already shipped the failure once: a run that
spent real money charged ``$0.00`` to every meter on the context. The fix lives
in two halves and both are invisible to a type check:

    BEFORE the spawn   the run's remaining headroom goes out as
                       ``--max-budget-usd`` so the CLI stops ITSELF mid-flight,
                       and a budget with no headroom left refuses to spawn at
                       all rather than paying two to five seconds of CLI
                       warm-up to be told what we already knew.

    AFTER the run      ``_charge_meters`` puts what the CLI actually spent on
                       every meter on the context PLUS the per-actor envelope
                       — and must never raise, because the spend already
                       happened and the answer is already paid for.

Nothing in either half fails loudly when it is removed. An argv that is missing
a flag still runs; a charge that never happens still returns; a ceiling crossed
on the last call still has an answer to hand back. So each test below pins one
of those observable consequences — the flag and its VALUE, the spawn that did
not happen, the stop reason an operator reads, the meter that was called, the
exception that was contained — rather than the shape of the code that produces
them.

WHY THE COST IS THE CLI'S OWN NUMBER
------------------------------------
``ProviderUsage.cost_usd`` on this path comes from the CLI's own
``result.total_cost_usd``, never from agentkit's price table. The two disagree —
the CLI knows about discounts, cache tiers and server-side retries that a
published per-token table does not — and a test that recomputed the expected
charge from ``PriceTable`` would be asserting the wrong authority. One test
below states that difference explicitly so the next person to "fix" the
accounting finds it written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from agentkit.core.contracts.types import Mode, ProviderUsage, StreamEvent
from agentkit.core.testing.fakes import FakeClaudeCli
from agentkit.models.catalog.pricing import PriceTable, normalize_usage, price
from agentkit.models.contracts.constants import CostSource
from agentkit.models.providers.cli._common import CliResult, CliTimeouts
from agentkit.models.providers.cli.claude.driver import ClaudeCliCognition

# ─────────────────────────────────────────────────────────────────────────────
# Doubles.
#
# Every one of these is read STRUCTURALLY by the cognition (``getattr`` on the
# context, duck-typed ``charge``), which is deliberate on the production side —
# it keeps ``_common`` free of an import of the metering runtime. The
# consequence for a test is that the double IS the contract: ``remaining()``,
# ``ceiling()``, ``all_meters``, ``actor_budget.charge(tokens=, cost_usd=,
# steps=)``. Spelling one of them differently here would make these tests pass
# against a cognition that charges nothing.
# ─────────────────────────────────────────────────────────────────────────────

# One ordinary finished run. ``total_cost_usd`` is the CLI's own figure and the
# only cost authority on this path; the token counts are what the per-actor
# envelope is charged in.
FINISHED_RUN: dict[str, Any] = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "done",
    "session_id": "01HSESSION",
    "duration_ms": 1234,
    "total_cost_usd": 0.25,
    "usage": {"input_tokens": 100, "output_tokens": 50},
}


class FakeBudget:
    """A run budget with a ceiling and a ledger, read as the cognition reads it.

    ``remaining()`` and ``ceiling()`` return DIFFERENT numbers whenever
    anything has been spent. That is the whole point of the double: a cognition
    that sends the ceiling instead of the headroom hands the CLI permission to
    spend the already-spent money a second time, and the two would be
    indistinguishable on a budget nobody had drawn from yet.
    """

    def __init__(self, ceiling: str, spent: str = "0") -> None:
        self._ceiling = Decimal(ceiling)
        self._spent = Decimal(spent)

    def remaining(self) -> Decimal:
        return self._ceiling - self._spent

    def ceiling(self) -> Decimal:
        return self._ceiling


class UncappedBudget:
    """A budget wired up with no ceiling at all — ``remaining()`` is ``None``."""

    def remaining(self) -> None:
        return None

    def ceiling(self) -> None:
        return None


class RecordingMeter:
    """A ``Meter`` that records what it was charged and accepts everything."""

    def __init__(self) -> None:
        self.charges: list[tuple[Any, ProviderUsage]] = []

    async def charge(self, call: Any, usage: ProviderUsage) -> None:
        self.charges.append((call, usage))


class RefusingMeter:
    """A ``Meter`` whose ceiling this very run crossed, so the charge raises."""

    def __init__(self) -> None:
        self.attempts = 0

    async def charge(self, call: Any, usage: ProviderUsage) -> None:
        self.attempts += 1
        raise RuntimeError("run budget ceiling crossed by this call")


class RecordingActorBudget:
    """The per-actor envelope. Synchronous ``charge``, keyword-only, as wired."""

    def __init__(self) -> None:
        self.charges: list[tuple[int, float | None, int]] = []

    def charge(self, *, tokens: int, cost_usd: float | None, steps: int) -> None:
        self.charges.append((tokens, cost_usd, steps))


@dataclass
class FakeCtx:
    """The execution context, holding only what this cognition reads off it."""

    scope: Any = None
    correlation_id: str = "01HCORRELATION"
    cancel: Any = None
    budget: Any = None
    all_meters: tuple[Any, ...] = ()
    actor_budget: Any = None

    def check_cancelled(self) -> None:
        return None


class FakeAgent:
    """An agent-like with the two attributes the cognition reads.

    ``_output_adapter`` is ``None`` throughout this file: structured output is
    the other file's subject, and leaving it set here would put a
    ``--json-schema`` argument in every argv assertion below for no reason.
    """

    prompt = "You are a concise assistant."
    # A model the bundled price table actually prices, so the "the CLI's number
    # wins over the table's" test below compares two real figures.
    model = "claude-sonnet-4-6"
    _output_adapter = None


def cognition(cli: FakeClaudeCli, **kwargs: Any) -> ClaudeCliCognition:
    """A cognition wired to the offline CLI, with every liveness bound off.

    The bounds are off rather than defaulted because a budget test that fails
    on a loaded machine's clock teaches nothing. The fake returns from an
    already-materialised buffer, so no bound can legitimately fire.
    """
    return ClaudeCliCognition(spawn=cli, timeouts=CliTimeouts.off(), **kwargs)


async def drive(cog: ClaudeCliCognition, ctx: Any, *, agent: Any = None) -> CliResult:
    """Drive one run and return the result off its single terminal event.

    Asserts the terminal-event guarantee on the way past: exactly one ``final``
    event, on every path through this file including the ones where the budget
    refuses the spawn and the one where a meter raises.
    """
    events: list[StreamEvent] = []
    async for event in cog.drive(agent or FakeAgent(), "summarise the diff", ctx, None):
        events.append(event)
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1, f"expected exactly one terminal event, got {[e.type for e in events]}"
    result = finals[0].result
    assert isinstance(result, CliResult)
    return result


def flag_value(argv: tuple[str, ...], flag: str) -> str:
    """The single value following ``flag`` in an argv. Fails if absent."""
    assert flag in argv, f"{flag} missing from argv: {argv}"
    return argv[argv.index(flag) + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Before the spawn — the headroom handed to the CLI.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_run_hands_the_cli_its_remaining_headroom_not_its_original_ceiling() -> None:
    """The CLI enforces the cap itself mid-flight; agentkit only ever learns the
    total afterwards. So the number in the argv is the only thing that can stop
    a run before it overspends — and it has to be what is LEFT. Sending the
    original ceiling of a budget that is already $4 down authorises $4 of spend
    the caller's ledger has already recorded, and sending nothing at all leaves
    a long autonomous session with no limit in a flag the docs promise sets
    one."""
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="10", spent="4"))

    await drive(cognition(cli), ctx)

    argv = cli.invocations[0].argv
    assert Decimal(flag_value(argv, "--max-budget-usd")) == Decimal("6")


@pytest.mark.asyncio
async def test_an_exhausted_budget_refuses_to_spawn_the_cli_at_all() -> None:
    """Spawning a subprocess to be told what the ledger already knows costs two
    to five seconds of CLI warm-up per attempt, and a retry loop over an
    exhausted budget pays it every time around. The refusal is a pre-flight
    check for that reason: the measurable guarantee is that NO process starts,
    not merely that the run ends badly."""
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="10", spent="10"))

    result = await drive(cognition(cli), ctx)

    assert cli.spawns == 0
    assert result.evals["execution_started"] is False


@pytest.mark.asyncio
async def test_a_pre_flight_budget_refusal_is_reported_as_resumable_not_as_a_spawn_failure() -> None:
    """``budget_exhausted`` and ``spawn_failed`` send an operator to opposite
    places. The first says "raise the ceiling and run this again" — nothing is
    wrong, the run is resumable. The second says the ``claude`` binary or its
    working directory is broken, which is an infrastructure hunt that ends in
    nothing. Because the refusal travels as an exception out of ``_prepare``,
    exactly like a missing binary does, the only thing keeping them apart is
    the classification."""
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="2.50", spent="2.50"))

    result = await drive(cognition(cli), ctx)

    assert result.stop_reason == "budget_exhausted"
    assert result.evals["stop_reason"] == "budget_exhausted"
    assert result.partial is True
    assert "MeterExceeded" in result.evals["error"]


@pytest.mark.asyncio
async def test_a_budget_with_no_ceiling_invents_no_limit_for_the_cli() -> None:
    """A context can carry a budget that counts spend without capping it. There
    is no number to enforce in that case, and passing one anyway would invent a
    ceiling the caller never set — a run killed mid-flight at a limit that
    exists nowhere in the application's configuration."""
    cli = FakeClaudeCli.script([FINISHED_RUN])

    await drive(cognition(cli), FakeCtx(budget=UncappedBudget()))

    assert "--max-budget-usd" not in cli.invocations[0].argv


@pytest.mark.asyncio
async def test_a_run_driven_without_a_context_neither_caps_nor_charges() -> None:
    """``drive`` is reachable with no context at all — a bare
    ``cognition.drive(...)`` in a script, a harness that has not built one yet.
    There is then no ledger to read a cap out of and none to charge, and both
    halves of the spend integration have to notice that rather than raising on
    a ``None`` and converting a working run into ``spawn_failed``."""
    cli = FakeClaudeCli.script([FINISHED_RUN])

    result = await drive(cognition(cli), None)

    assert result.stop_reason == "complete"
    assert result.partial is False
    assert "--max-budget-usd" not in cli.invocations[0].argv


# ─────────────────────────────────────────────────────────────────────────────
# After the run — the spend on the framework's books.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_what_the_cli_spent_is_charged_to_every_meter_on_the_context() -> None:
    """This is the ``$50 run, $0.00 ledger`` bug, and it is silent by
    construction: the meters are not consulted on this path by the framework,
    so a missing charge produces no error, no warning and a perfectly ordinary
    result. Every meter is charged — not the first one — because a context
    routinely carries a run budget AND a tenant quota, and charging one of the
    two is the same defect with a smaller blast radius."""
    first, second = RecordingMeter(), RecordingMeter()
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="10"), all_meters=(first, second))

    await drive(cognition(cli), ctx)

    for meter in (first, second):
        assert len(meter.charges) == 1
        call, usage = meter.charges[0]
        assert usage.cost_usd == pytest.approx(0.25)
        assert (usage.input_tokens, usage.output_tokens) == (100, 50)
        # ``Quota`` partitions per tenant off ``call.ctx.scope``, so a bare
        # ``None`` call would charge the budget and crash the quota.
        assert call.ctx is ctx


@pytest.mark.asyncio
async def test_the_per_actor_envelope_is_charged_alongside_the_shared_meters() -> None:
    """The per-actor envelope is a SECOND ledger with its own ceiling, and
    ``ActorBudget`` has already shipped exactly this hole once: the shared
    meters were charged, the per-actor one was not, and an actor whose whole
    purpose is a spend limit ran without one. Charging the meters is therefore
    not evidence that this happened — it needs its own assertion, in tokens,
    dollars and steps, because the envelope caps all three."""
    actor = RecordingActorBudget()
    meter = RecordingMeter()
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="10"), all_meters=(meter,), actor_budget=actor)

    await drive(cognition(cli), ctx)

    assert actor.charges == [(150, pytest.approx(0.25), 1)]


@pytest.mark.asyncio
async def test_a_meter_that_refuses_the_charge_never_costs_the_caller_the_answer() -> None:
    """The commonest refusal is a ceiling crossed by THIS very call — the run
    that took the budget over is the run being charged. The money is spent, the
    CLI has produced its answer, and the terminal-event guarantee says the
    caller receives it. Letting the refusal propagate would throw away a result
    that has already been paid for, out of a generator, and leave the caller
    with neither the answer nor a terminal event. The refusal still has to be
    VISIBLE, which is what ``meter_error`` is for."""
    meter = RefusingMeter()
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="10"), all_meters=(meter,))

    result = await drive(cognition(cli), ctx)

    assert meter.attempts == 1
    assert result.output == "done"
    assert result.stop_reason == "complete"
    assert result.partial is False
    assert "run budget ceiling crossed by this call" in result.evals["meter_error"]


@pytest.mark.asyncio
async def test_a_refused_charge_on_one_meter_does_not_stop_the_others_being_charged() -> None:
    """A context carrying a crossed run budget and an untouched tenant quota
    must still record the spend against the quota. Stopping at the first
    refusal would leave the quota permanently under-counting every tenant
    whose run budget happens to run out first."""
    refusing, recording = RefusingMeter(), RecordingMeter()
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="10"), all_meters=(refusing, recording))

    result = await drive(cognition(cli), ctx)

    assert len(recording.charges) == 1
    assert "meter_error" in result.evals


@pytest.mark.asyncio
async def test_meter_spend_false_opts_a_run_out_of_both_ends_of_the_accounting() -> None:
    """``meter_spend=False`` is the documented opt-out, and it has to mean both
    halves: nothing is charged afterwards AND no cap is handed to the CLI. A
    flag that silenced only the charge would leave a caller who deliberately
    took a run off the books running against a ceiling they had just disowned,
    and one that silenced only the cap would still move someone else's ledger
    for a run they asked to be invisible to it."""
    meter, actor = RecordingMeter(), RecordingActorBudget()
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="10", spent="4"), all_meters=(meter,), actor_budget=actor)

    result = await drive(cognition(cli, meter_spend=False), ctx)

    assert meter.charges == []
    assert actor.charges == []
    assert "--max-budget-usd" not in cli.invocations[0].argv
    assert "meter_error" not in result.evals


@pytest.mark.asyncio
async def test_the_charge_is_the_clis_own_cost_and_not_a_price_table_estimate() -> None:
    """Two sources disagree about what a run cost, and only one of them is
    right. The CLI reports ``total_cost_usd`` from the provider's own billing,
    which accounts for cache tiers, server-side retries and whatever discount
    the account carries; agentkit's ``PriceTable`` is a dated snapshot of
    published per-token rates. Recomputing the charge from the table would
    quietly replace a measured number with an estimate — and on these token
    counts the two differ by two orders of magnitude, which is the size of the
    error that would land in the ledger."""
    meter = RecordingMeter()
    cli = FakeClaudeCli.script([FINISHED_RUN])
    ctx = FakeCtx(budget=FakeBudget(ceiling="10"), all_meters=(meter,))

    await drive(cognition(cli), ctx)

    _, usage = meter.charges[0]
    assert usage.cost_usd == pytest.approx(0.25)

    estimated, source = price(
        FakeAgent.model,
        normalize_usage(Mode.CLAUDE_CODE, FINISHED_RUN["usage"]),
        PriceTable.default(),
        mode=Mode.CLAUDE_CODE,
    )
    # Not an assertion about the table's numbers — only that it HAS one here
    # and that it is not the number the ledger received, so the assertion
    # above is pinning the right authority rather than a coincidence.
    assert source is CostSource.PRICE_TABLE
    assert estimated is not None and estimated != Decimal("0.25")
