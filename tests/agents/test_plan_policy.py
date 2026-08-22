"""``PlanPolicy`` human-gate suspend/resume (mid-plan pause between groups).

Companion to ``test_orchestrator.py`` (basic group dispatch); this file locks the
coordinator-level suspend contract: reaching a ``Step.gate("name")`` checkpoints the
accumulated results/errors/usage at the plan's durable slot, yields a ``Suspended``
result, and returns without dispatching further groups. Resume routes on the decision
dict: ``approve`` clears the checkpoint and continues at the next group; ``reject`` (or
a missing key) terminates with ``stop_reason="rejected"`` and never runs downstream
groups. Mirrors ``Workflow.human_gate`` / ``Workflow.resume`` — the same primitive at
the coordinator layer. Offline & deterministic via ``FakeLLM`` + ``asyncio.run``."""

import asyncio

import pytest

from agentkit import Agent
from agentkit.adapters.store import InMemoryStore
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies.plan import (
    PlanPolicy,
    StaticPlanner,
    Step,
    checkpoint_slot,
)
from agentkit.agents.result import Suspended
from agentkit.kernel.types import Scope
from agentkit.testing import FakeLLM, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


class _RecordingAgent:
    """A minimal agent stub that logs each ``.run`` call — lets a test assert whether
    the downstream agent was actually dispatched across a suspend/resume."""

    def __init__(self, name: str, output: str = "SYNTH") -> None:
        self.name = name
        self.output = output
        self.calls: list[str] = []

    async def run(self, task, ctx, *, context=None):
        from agentkit.agents.result import AgentResult
        from agentkit.kernel.types import Usage

        self.calls.append(task)
        return AgentResult(output=self.output, usage=Usage())


def _plan_with_gate() -> list[Step]:
    """Canonical shape from doc 04: two researchers @ group 0 → human gate @ group 1
    → synthesizer @ group 2. The gate sits between researchers and synth by group."""
    return [
        Step("researcher_1", "q1", group=0),
        Step("researcher_2", "q2", group=0),
        Step.gate("review", group=1),
        Step("synthesizer", "compose", group=2),
    ]


def _make_children(synth: _RecordingAgent) -> dict:
    return {
        "researcher_1": Agent("researcher_1", "m"),
        "researcher_2": Agent("researcher_2", "m"),
        "synthesizer": synth,
    }


def _coord(policy: PlanPolicy, children: dict) -> Agent:
    return Agent(
        name="coordinator",
        cognition=CoordinatorCognition(children=children, policy=policy),
    )


# ── 1. Suspend at the gate — synthesizer never runs ─────────────────────────


def test_plan_policy_suspends_at_human_gate():
    """A plan with a gate step between researcher and synthesizer groups pauses on the
    gate: only researcher outputs land in ``evals['results']``; the synthesizer's ``run``
    is never called; the result carries a ``Suspended(pending=("review",))`` in its evals
    and ``stop_reason == "awaiting_decision"``."""
    synth = _RecordingAgent("synthesizer")
    children = _make_children(synth)
    policy = PlanPolicy(planner=StaticPlanner(_plan_with_gate()))
    coord = _coord(policy, children)

    store = InMemoryStore()
    ctx = make_test_ctx(
        llm=FakeLLM("ok"), store=store, scope=Scope(1, 2), correlation_id="plan-run-1"
    )
    res = _run(coord.run("goal", ctx))

    assert res.evals["stop_reason"] == "awaiting_decision"
    suspended = res.evals["suspended"]
    assert isinstance(suspended, Suspended)
    assert suspended.pending == ("review",)
    assert suspended.run_id == "plan-run-1"
    # Only the two researchers ran — the synth step was NEVER dispatched.
    assert len(res.evals["results"]) == 2
    assert synth.calls == []
    # Checkpoint was persisted for resume, through the shared ``Checkpointer``
    # seam (bridged over this store) at the plan's own namespaced slot.
    assert _run(store.get(f"checkpoint:{checkpoint_slot('plan-run-1')}")) is not None


# ── 2. Approve → resume dispatches the synthesizer ──────────────────────────


def test_plan_policy_resume_approves_and_continues():
    """After a suspend, ``resume(coord, {"review": "approve"}, ctx)`` clears the
    checkpoint and executes the group AFTER the gate: the synthesizer runs exactly once
    and the final result includes all three agent groups' outputs (researcher_1,
    researcher_2, synthesizer)."""
    synth = _RecordingAgent("synthesizer", output="FINAL")
    children = _make_children(synth)
    policy = PlanPolicy(planner=StaticPlanner(_plan_with_gate()))
    coord = _coord(policy, children)

    store = InMemoryStore()
    ctx = make_test_ctx(
        llm=FakeLLM("ok"), store=store, scope=Scope(1, 2), correlation_id="plan-run-2"
    )
    res = _run(coord.run("goal", ctx))
    assert res.evals["stop_reason"] == "awaiting_decision"
    assert synth.calls == []  # baseline: pre-approve, synth did not run

    resumed = _run(policy.resume(coord, {"review": "approve"}, ctx))
    assert resumed.evals["stop_reason"] == "plan_complete"
    # Two researcher results (from before the gate) + one synth result (post-approve).
    assert len(resumed.evals["results"]) == 3
    assert resumed.evals["errors"] == []
    # The synthesizer was dispatched exactly once, with its planned input.
    assert synth.calls == ["compose"]
    # The last result's output floats to the top-level ``output`` field.
    assert resumed.output == "FINAL"


# ── 3. Reject → resume terminates without running the synthesizer ───────────


def test_plan_policy_resume_rejects_and_stops():
    """``resume(coord, {"review": "reject"}, ctx)`` returns
    ``stop_reason="rejected"`` and never dispatches the post-gate group. A missing
    decision falls into the same terminal branch — reject is the safe default."""
    synth = _RecordingAgent("synthesizer")
    children = _make_children(synth)
    policy = PlanPolicy(planner=StaticPlanner(_plan_with_gate()))
    coord = _coord(policy, children)

    store = InMemoryStore()
    ctx = make_test_ctx(
        llm=FakeLLM("ok"), store=store, scope=Scope(1, 2), correlation_id="plan-run-3"
    )
    _run(coord.run("goal", ctx))

    resumed = _run(policy.resume(coord, {"review": "reject"}, ctx))
    assert resumed.evals["stop_reason"] == "rejected"
    assert resumed.evals["gate"] == "review"
    assert resumed.evals["decision"] == "reject"
    # Synthesizer NEVER ran: the resume terminates before the post-gate group.
    assert synth.calls == []
    # Researcher results from before the gate survive on the terminal result.
    assert len(resumed.evals["results"]) == 2


# ── 4. Regression: a gate-less plan runs every group as before ──────────────


def test_plan_policy_without_gate_runs_all_groups():
    """The gate machinery is opt-in — a plan with no ``Step.gate`` step behaves exactly
    like the pre-gate PlanPolicy: all groups run to completion, ``stop_reason`` is
    ``plan_complete``, no checkpoint is written, no ``Suspended`` in evals."""
    plan = [
        Step("a", "x", group=0),
        Step("b", "y", group=0),
        Step("c", "z", group=1),
    ]
    children = {"a": Agent("a", "m"), "b": Agent("b", "m"), "c": Agent("c", "m")}
    policy = PlanPolicy(planner=StaticPlanner(plan))
    coord = _coord(policy, children)

    store = InMemoryStore()
    ctx = make_test_ctx(
        llm=FakeLLM("ok"), store=store, scope=Scope(1, 2), correlation_id="plan-run-4"
    )
    res = _run(coord.run("goal", ctx))
    assert res.evals["stop_reason"] == "plan_complete"
    assert len(res.evals["results"]) == 3
    assert res.evals["errors"] == []
    assert "suspended" not in res.evals
    # No checkpoint should have been written when no gate was in play.
    assert _run(store.get(f"checkpoint:{checkpoint_slot('plan-run-4')}")) is None


# ── 5. Approve reclaims the checkpoint ──────────────────────────────────────


def test_plan_policy_gate_checkpoint_deleted_on_approve():
    """After a successful approve-resume, the checkpoint is deleted — a re-resume under
    the same run id must fail loudly. This is the counterpart to
    ``Workflow.resume``'s ``store.delete`` on terminal-not-suspended: a run cannot be
    silently re-approved."""
    synth = _RecordingAgent("synthesizer")
    children = _make_children(synth)
    policy = PlanPolicy(planner=StaticPlanner(_plan_with_gate()))
    coord = _coord(policy, children)

    store = InMemoryStore()
    ctx = make_test_ctx(
        llm=FakeLLM("ok"), store=store, scope=Scope(1, 2), correlation_id="plan-run-5"
    )
    slot = f"checkpoint:{checkpoint_slot('plan-run-5')}"
    _run(coord.run("goal", ctx))
    assert _run(store.get(slot)) is not None  # checkpoint exists

    _run(policy.resume(coord, {"review": "approve"}, ctx))
    # State cleaned up: the store no longer holds the run's checkpoint.
    assert _run(store.get(slot)) is None
    # A second resume under the same id must not silently succeed — the checkpoint is
    # gone, so ``resume`` raises rather than replaying a phantom.
    with pytest.raises(ValueError, match="no suspended plan"):
        _run(policy.resume(coord, {"review": "approve"}, ctx))
