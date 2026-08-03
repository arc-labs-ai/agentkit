"""Smoke tests for the `Agent` + `Policy` shape — the cross-cutting surface.
A leaf `Agent` is the single-call form; a coordinator `Agent` dispatches to
`children` per its `Policy`. `RunPolicy` (Rule of Two) is the run-wide safety
check."""

import asyncio

import pytest

from agentkit import Agent, RunPolicy
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies.plan import PlanPolicy, StaticPlanner, Step
from agentkit.kernel.types import Scope
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import FunctionTool


def _run(coro):
    return asyncio.run(coro)


def test_agent_single_call_goes_through_invoker():
    ctx = make_test_ctx(llm=FakeLLM("answer"), scope=Scope(1, 2), correlation_id="r")
    res = _run(Agent(name="a", model="m", prompt="s").run("q", ctx))
    assert res.output == "answer"


def test_coordinator_plan_policy_runs_groups_in_order():
    ctx = make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="r")
    coord = Agent(
        name="orch",
        cognition=CoordinatorCognition(
            children={"ex": Agent("ex", "m"), "ju": Agent("ju", "m")},
            policy=PlanPolicy(
                planner=StaticPlanner([Step("ex", "c", group=0), Step("ju", "c", group=1)])
            ),
        ),
    )
    res = _run(coord.run("goal", ctx))
    # Two steps across two groups → two per-child results recorded in evals.
    results = res.evals["results"]
    assert len(results) == 2 and results[0].output == "x"


def _t(name, caps):
    return FunctionTool(
        name,
        lambda a, c: None,
        description="test stub tool used for RunPolicy capability checks",
        side_effecting=False,
        caps=caps,
    )


def test_run_policy_flags_lethal_trifecta():
    v = RunPolicy().check(
        [_t("a", ("private_data",)), _t("b", ("untrusted_content",)), _t("c", ("egress",))]
    )
    assert not v.allowed and "lethal trifecta" in v.reason


def test_run_policy_allows_two_of_three():
    assert RunPolicy().check([_t("a", ("private_data",)), _t("b", ("egress",))]).allowed


# ── Suspended handshake is fully immutable ──────────────────────────────────
#
# ``Suspended`` is the value the run yields when it pauses for human
# approval. The operator UI renders the ``pending`` items; the resume
# path threads them back verbatim. If the collection carrying those
# items were mutable, a stray ``suspended.pending.append(...)`` on
# either side would desync the operator's view from what actually
# runs after resume — the classic approve-then-modify attack surface.


def test_suspended_pending_is_a_frozen_tuple() -> None:
    """``pending`` is a tuple, not a list. Reassignment is blocked by
    the frozen dataclass; item assignment / append is blocked by the
    tuple itself. Two doors, both locked."""
    from dataclasses import FrozenInstanceError

    from agentkit.agents.result import Suspended
    from agentkit.kernel.types import ToolCall

    tc = ToolCall("call-1", "delete", {"path": "/etc/hosts"})
    susp = Suspended(run_id="run-x", pending=(tc,))

    assert isinstance(susp.pending, tuple)
    # Frozen shell — cannot rebind the whole collection.
    with pytest.raises(FrozenInstanceError):
        susp.pending = ()  # type: ignore[misc]
    # Frozen collection — cannot mutate items in-place either.
    with pytest.raises(TypeError):
        susp.pending[0] = ToolCall("call-2", "wipe", {})  # type: ignore[index]
    with pytest.raises(AttributeError):
        susp.pending.append(ToolCall("call-2", "wipe", {}))  # type: ignore[attr-defined]


def test_suspended_pending_survives_replace_for_intentional_edits() -> None:
    """``dataclasses.replace`` is the sanctioned rewrite path — a caller
    that legitimately needs to add or drop a pending item builds a
    fresh ``Suspended`` instead of mutating one in flight."""
    from dataclasses import replace

    from agentkit.agents.result import Suspended
    from agentkit.kernel.types import ToolCall

    original = Suspended(run_id="run-x", pending=(ToolCall("c1", "a", {}),))
    extended = replace(original, pending=(*original.pending, ToolCall("c2", "b", {})))

    assert len(original.pending) == 1  # untouched
    assert len(extended.pending) == 2
    assert original is not extended
