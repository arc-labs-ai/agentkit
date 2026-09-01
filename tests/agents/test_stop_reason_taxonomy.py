"""Every producer stamps ``AgentResult.stop_reason`` — the typed field, not just ``evals``.

``AgentResult.stop_reason`` is the CLOSED taxonomy a caller branches on, and
``is_suspended`` / ``is_resumable`` are derived from it. When it was introduced,
only the tool loop set it: every COORDINATOR policy (round-robin, selector,
ledger, plan) and the Claude-CLI cognition left it at its ``"complete"``
default while writing the real reason into ``evals["stop_reason"]``.

The consequence was not cosmetic. A plan parked on a human gate reported
``is_suspended is False`` with its checkpoint sitting in the store, so an
application that branched on the typed field never prompted its human and never
called ``resume`` — while a run that hit a turn ceiling was indistinguishable
from one that finished its work.

These tests pin the mapping at both ends: the pure function
(:func:`agentkit.agents.result.stop_reason_for`), and each producer actually
calling it. The last test is a source-level ratchet so a new producer cannot
invent a reason string that silently falls through to ``"terminated"``.
"""

from __future__ import annotations

import asyncio
import pathlib
import re
from typing import get_args

import pytest

from agentkit import Agent
from agentkit.adapters.store import InMemoryStore
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.cognition.claude_cli import (
    _CLI_FAILURE_REASONS,
    _CLI_INVALID_OUTPUT_REASONS,
    _cli_stop_reason,
)
from agentkit.agents.cognition.codex_cli import (
    _CODEX_FAILURE_REASONS,
    _CODEX_INVALID_OUTPUT_REASONS,
    _codex_stop_reason,
)
from agentkit.agents.control.termination import FunctionalTermination, MaxTurns
from agentkit.agents.policies.ledger import LedgerPolicy
from agentkit.agents.policies.plan import PlanPolicy, StaticPlanner, Step
from agentkit.agents.policies.roundrobin import RoundRobinPolicy
from agentkit.agents.result import (
    _REASON_TO_STOP,
    RESUMABLE_STOP_REASONS,
    AgentResult,
    AgentStopReason,
    stop_reason_for,
)
from agentkit.capabilities.checkpointer.persistence import dict_to_result, result_to_dict
from agentkit.kernel.types import Usage
from agentkit.testing import FakeLLM, make_test_ctx


class _Rec:
    """A child agent stub that records its dispatches and never calls an LLM."""

    def __init__(self, name: str, out: str = "OUT") -> None:
        self.name, self.output, self.calls = name, out, []

    async def run(self, task, ctx, *, context=None):  # noqa: ANN001, ANN202
        self.calls.append(task)
        return AgentResult(output=self.output, usage=Usage())


def _coord(policy, children: dict) -> Agent:  # noqa: ANN001
    return Agent(name="c", cognition=CoordinatorCognition(children=children, policy=policy))


# ── 1. the pure mapping ─────────────────────────────────────────────────────


def test_the_mapping_is_total_and_never_guesses() -> None:
    """Unrecognised → ``"terminated"``; ``None`` → ``"complete"``. Totality is
    what lets a producer stamp the field unconditionally."""
    assert stop_reason_for("a custom condition's own wording") == "terminated"
    assert stop_reason_for("") == "terminated"
    assert stop_reason_for(None) == "complete"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("awaiting_decision", "suspended"),  # a plan parked on a human gate
        ("awaiting_approval", "suspended"),  # a tool call parked on approval
        ("plan_complete", "complete"),
        ("satisfied", "complete"),
        ("rejected", "terminated"),
        ("stalled", "terminated"),
        ("no_children", "terminated"),
        ("cancelled", "terminated"),
        ("interrupted", "terminated"),
        ("max_turns", "max_iterations"),
        ("max_rounds", "max_iterations"),
        ("max_messages", "max_iterations"),
        ("budget_exhausted", "budget_exhausted"),
        ("expired", "expired"),
    ],
)
def test_each_framework_reason_maps_to_the_intended_category(reason: str, expected: str) -> None:
    """Spelled out one by one rather than asserting the dict equals itself: the
    VALUES are the contract (``suspended`` implies resumable, ``complete``
    implies nothing left to do), and a typo'd value would otherwise be pinned
    by a test that read the same typo."""
    assert stop_reason_for(reason) == expected


def test_only_suspended_and_budget_exhausted_are_resumable() -> None:
    """A rejected gate and an exhausted round budget are terminal: no amount of
    resuming will advance them. Pinning this stops a future reason from being
    quietly added to the resumable set."""
    resumable = {r for r in set(_REASON_TO_STOP.values()) if r in RESUMABLE_STOP_REASONS}
    assert resumable == {"suspended", "budget_exhausted"}
    assert AgentResult("", Usage(), stop_reason="terminated").is_resumable is False
    assert AgentResult("", Usage(), stop_reason="suspended").is_resumable is True


# ── 2. the bug that started this: a suspended plan claiming completion ──────


def test_a_plan_parked_on_a_gate_reports_itself_suspended() -> None:
    """THE regression. Before the fix: ``stop_reason == "complete"``,
    ``is_suspended is False``, and a checkpoint nobody would ever resume."""
    synth = _Rec("synth")
    policy = PlanPolicy(
        planner=StaticPlanner(
            [Step("r1", "q", group=0), Step.gate("review", group=1), Step("synth", "go", group=2)]
        )
    )
    coord = _coord(policy, {"r1": _Rec("r1"), "synth": synth})
    ctx = make_test_ctx(llm=FakeLLM("ok"), store=InMemoryStore(), correlation_id="p1")

    res = asyncio.run(coord.run("goal", ctx))

    assert res.stop_reason == "suspended"
    assert res.is_suspended and res.is_resumable
    assert res.evals["stop_reason"] == "awaiting_decision"  # free-form detail kept verbatim
    assert synth.calls == []


def test_a_rejected_gate_is_terminal_not_complete() -> None:
    """``reject`` is a deliberate stop by a person: terminal, and NOT the same
    typed state as a plan that finished its work."""
    policy = PlanPolicy(
        planner=StaticPlanner([Step("r1", "q", group=0), Step.gate("review", group=1)])
    )
    coord = _coord(policy, {"r1": _Rec("r1")})
    ctx = make_test_ctx(llm=FakeLLM("ok"), store=InMemoryStore(), correlation_id="p2")
    asyncio.run(coord.run("goal", ctx))

    res = asyncio.run(policy.resume(coord, {"review": "reject"}, ctx))

    assert res.stop_reason == "terminated"
    assert res.evals["stop_reason"] == "rejected"
    assert not res.is_resumable


def test_a_finished_plan_is_complete() -> None:
    """The positive control: without a gate, a plan that runs every group IS
    completion — so the fix did not just relabel everything as terminated."""
    policy = PlanPolicy(planner=StaticPlanner([Step("r1", "q", group=0)]))
    coord = _coord(policy, {"r1": _Rec("r1")})
    res = asyncio.run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), correlation_id="p3")))
    assert res.stop_reason == "complete"
    assert res.evals["stop_reason"] == "plan_complete"


# ── 3. the other coordinators ───────────────────────────────────────────────


def test_a_ledger_that_runs_out_of_rounds_says_so() -> None:
    """``max_rounds`` is an iteration ceiling, not a met goal."""
    coord = _coord(LedgerPolicy(max_rounds=2), {"w": _Rec("w")})
    res = asyncio.run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), correlation_id="l1")))
    assert res.evals["stop_reason"] == "max_rounds"
    assert res.stop_reason == "max_iterations"


def test_a_round_robin_that_runs_out_of_turns_says_so() -> None:
    """The policy's own ceiling. ``max_turns=1`` with a never-firing termination
    forces the loop to exhaust rather than terminate early."""
    never = FunctionalTermination(lambda msgs, ctx: False, reason="never")
    coord = Agent(
        name="c",
        cognition=CoordinatorCognition(
            children={"a": _Rec("a")}, policy=RoundRobinPolicy(max_turns=1), termination=never
        ),
    )
    res = asyncio.run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), correlation_id="rr1")))
    assert res.evals["stop_reason"] == "max_turns"
    assert res.stop_reason == "max_iterations"


def test_a_custom_termination_condition_reads_as_terminated() -> None:
    """A user's own wording cannot be categorised, and the framework does not
    guess: ``terminated`` says "something stopped this deliberately"."""
    coord = Agent(
        name="c",
        cognition=CoordinatorCognition(
            children={"a": _Rec("a")},
            policy=RoundRobinPolicy(max_turns=5),
            termination=FunctionalTermination(lambda msgs, ctx: True, reason="reviewer_said_stop"),
        ),
    )
    res = asyncio.run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), correlation_id="rr2")))
    assert res.evals["stop_reason"] == "reviewer_said_stop"
    assert res.stop_reason == "terminated"


def test_a_round_robin_stopped_by_max_turns_condition_agrees_with_the_ceiling() -> None:
    """``MaxTurns`` firing and the policy's own ceiling tripping mean the same
    thing to a reader deciding whether to raise a limit, so they must not map
    to different categories."""
    coord = Agent(
        name="c",
        cognition=CoordinatorCognition(
            children={"a": _Rec("a")}, policy=RoundRobinPolicy(max_turns=9), termination=MaxTurns(1)
        ),
    )
    res = asyncio.run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), correlation_id="rr3")))
    assert res.stop_reason == "max_iterations"


# ── 4. the CLI cognitions — the producers that report failure as data ───────


@pytest.mark.parametrize("reason", sorted(_CLI_FAILURE_REASONS) + ["cli_exit_1", "cli_exit_137"])
def test_cli_failures_map_to_failed(reason: str) -> None:
    """A subprocess that never started is not a deliberate stop. ``cli_exit_<n>``
    is dynamic, which is why this mapping is local to the cognition."""
    assert _cli_stop_reason(reason) == "failed"


@pytest.mark.parametrize("reason", sorted(_CODEX_FAILURE_REASONS) + ["cli_exit_1", "cli_exit_137"])
def test_codex_failures_map_to_failed(reason: str) -> None:
    """The same contract for the second CLI cognition, asserted against its OWN
    table rather than a shared one. The two sets overlap heavily and differ
    where the binaries do — ``turn_failed`` exists only for Codex, ``turn_refused``
    only for Claude — so a single parametrised list over a merged set would stop
    noticing when one of them lost an entry."""
    assert _codex_stop_reason(reason) == "failed"


@pytest.mark.parametrize("reason", sorted(_CODEX_INVALID_OUTPUT_REASONS))
def test_codex_structured_output_failures_are_invalid_output(reason: str) -> None:
    """Codex constrains its final MESSAGE to the schema rather than returning a
    separate validated field, so "the answer is not JSON" is the shape failing,
    not the run — ``invalid_output``, same as everywhere else."""
    assert _codex_stop_reason(reason) == "invalid_output"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [(None, "complete"), ("success", "complete"), ("cancelled", "terminated")],
)
def test_codex_non_failures_defer_to_the_shared_table(reason: str | None, expected: str) -> None:
    assert _codex_stop_reason(reason) == expected


def test_budget_exhausted_stays_resumable_through_both_cli_cognitions() -> None:
    """Both cognitions refuse to spawn when the budget is already gone, and both
    have to report that as the RESUMABLE reason — raise the ceiling and run
    again — rather than as a failure. It is deliberately absent from both
    failure sets so the shared table answers, and this is the test that says so:
    listing it locally would flip it to ``failed`` and silently strand a run an
    operator could have released."""
    assert _cli_stop_reason("budget_exhausted") == "budget_exhausted"
    assert _codex_stop_reason("budget_exhausted") == "budget_exhausted"
    assert "budget_exhausted" in RESUMABLE_STOP_REASONS


def test_an_interrupt_is_terminated_not_failed() -> None:
    """Somebody pressed stop. Not ``failed`` — nothing broke — and distinct
    from ``cancelled``, which in the CLI cognition means the process was killed
    and the conversation is gone. It lives in the SHARED table rather than a
    CLI-local one: the fallback would land on ``terminated`` anyway, and a
    second table that only restates the default is one more place to drift."""
    assert _cli_stop_reason("interrupted") == "terminated"
    assert stop_reason_for("interrupted") == "terminated"


@pytest.mark.parametrize("reason", sorted(_CLI_INVALID_OUTPUT_REASONS))
def test_cli_structured_output_failures_are_invalid_output(reason: str) -> None:
    """The run worked; the SHAPE did not. That is ``invalid_output``, the same
    category the tool loop uses when parse-and-repair is exhausted — not
    ``failed``, which would read as "the subprocess fell over"."""
    assert _cli_stop_reason(reason) == "invalid_output"


@pytest.mark.parametrize(("reason", "expected"), [(None, "complete"), ("success", "complete"), ("cancelled", "terminated")])
def test_cli_non_failures_defer_to_the_shared_table(reason: str | None, expected: str) -> None:
    assert _cli_stop_reason(reason) == expected


# ── 5. persistence round-trip ───────────────────────────────────────────────


def test_a_rehydrated_result_keeps_its_stop_reason() -> None:
    """``result_to_dict``/``dict_to_result`` back a coordinator's durable resume.
    Dropping the field there re-introduced the bug one layer down: a restored
    suspended child read back as ``complete``."""
    r = AgentResult(output="x", usage=Usage(1, 2, 0.5), stop_reason="suspended")
    assert dict_to_result(result_to_dict(r)).stop_reason == "suspended"


def test_a_legacy_record_upgrades_from_its_free_form_reason() -> None:
    """Records written before the field was persisted have no ``stop_reason``
    key but DO carry ``evals["stop_reason"]``. They must upgrade, not read back
    as a bare completion."""
    legacy = {"output": "x", "usage": {"input": 0, "output": 0, "cost": 0.0},
              "evals": {"stop_reason": "awaiting_decision"}}
    assert dict_to_result(legacy).stop_reason == "suspended"
    assert dict_to_result({"output": ""}).stop_reason == "complete"  # nothing recorded at all


# ── 6. the ratchet ──────────────────────────────────────────────────────────


_REASON_LITERAL = re.compile(
    r"""(?:final_)?stop_reason(?::\s*str)?\s*=\s*"([a-z_0-9]+)" """  # assignment
    r"""|"stop_reason":\s*"([a-z_0-9]+)" """  # evals dict literal
    r"""|reason="([a-z_0-9]+)\"""",  # _final_events(...)/Suspended(...)
    re.VERBOSE,
)


def test_every_reason_the_framework_itself_passes_is_categorised() -> None:
    """Source-level ratchet. A new producer writing
    ``stop_reason = "waiting_on_vendor"`` gets ``"terminated"`` from the
    fallback — correct, but silent, and if the reason actually means "resumable"
    the typed field is then WRONG in exactly the way this whole file exists to
    prevent. Forcing an entry in one of the two tables makes that a decision.

    ``control/termination.py`` is excluded on purpose: a ``TerminationCondition``
    names its own stop, those strings are user-facing vocabulary rather than
    framework categories, and ``terminated`` is the honest answer for all of
    them.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "agentkit"
    # A producer may pass a taxonomy member DIRECTLY (``stop_reason="suspended"``
    # in the tool loop) — that needs no mapping entry, it IS the category.
    known = (
        set(_REASON_TO_STOP)
        | _CLI_FAILURE_REASONS
        | _CLI_INVALID_OUTPUT_REASONS
        | _CODEX_FAILURE_REASONS
        | _CODEX_INVALID_OUTPUT_REASONS
        | set(get_args(AgentStopReason))
    )
    uncategorised: dict[str, list[str]] = {}
    scanned = 0
    for path in sorted((root / "agents").rglob("*.py")) + sorted(
        (root / "capabilities").rglob("*.py")
    ):
        if path.name == "termination.py":
            continue
        scanned += 1
        for m in _REASON_LITERAL.finditer(path.read_text()):
            reason = next(g for g in m.groups() if g)
            if reason not in known and not reason.startswith("cli_exit_"):
                uncategorised.setdefault(reason, []).append(path.name)
    assert scanned > 20, f"the scan found almost nothing ({scanned} files) — the glob broke"
    assert not uncategorised, (
        "these stop reasons are passed by the framework but categorised nowhere; add each to "
        "`_REASON_TO_STOP` in agents/result.py (or the `_CLI_*` / `_CODEX_*` sets in the "
        f"relevant cognition if it is a CLI failure mode): {uncategorised}"
    )
