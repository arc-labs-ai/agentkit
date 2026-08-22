"""``PlanPolicy`` checks the plan against the roster BEFORE it dispatches anything.

Two failure shapes used to surface deep inside the dispatch loop, or not at all:

* A step naming a child that is not on the coordinator raised a bare
  ``KeyError('reseacher')`` from ``children[s.agent]`` — after the earlier
  groups had already run and spent money, and with their results unreachable
  (the accumulator is a local of ``_run_groups``). Under ``best_effort=True``
  that broke the mode's one promise: partial progress survives.
* A gate sharing a group with dispatch steps silently dropped those steps. The
  group suspends before any step runs, and resume continues at the group AFTER
  the gate, so the steps were announced in the trace and then never executed —
  in BOTH branches of the human's decision.

Everything here is offline and deterministic: the children are stubs that record
their dispatches, so "was anything dispatched before the error" is directly
observable.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit import Agent
from agentkit.adapters.store import InMemoryStore
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies.plan import (
    PlanPolicy,
    PlanShapeError,
    StaticPlanner,
    Step,
    _validate_plan,
)
from agentkit.agents.result import AgentResult
from agentkit.kernel.errors import Failure
from agentkit.kernel.resilience import ErrorClass
from agentkit.kernel.types import Usage
from agentkit.testing import FakeLLM, make_test_ctx


class _Rec:
    def __init__(self, name: str, out: str = "OUT") -> None:
        self.name, self.output, self.calls = name, out, []

    async def run(self, task, ctx, *, context=None):  # noqa: ANN001, ANN202
        self.calls.append(task)
        return AgentResult(output=self.output, usage=Usage())


def _coord(policy: PlanPolicy, children: dict) -> Agent:
    return Agent(name="c", cognition=CoordinatorCognition(children=children, policy=policy))


def _ctx(cid: str, store: InMemoryStore | None = None):  # noqa: ANN202
    return make_test_ctx(llm=FakeLLM("ok"), store=store, correlation_id=cid)


# ── 1. an unknown child, fail-fast mode ─────────────────────────────────────


def test_an_unknown_child_is_refused_before_anything_is_dispatched() -> None:
    """THE regression. The error names the child AND the roster, and the earlier
    step in the plan never ran — so nothing was spent on a plan that could not
    finish."""
    first = _Rec("r1")
    policy = PlanPolicy(
        planner=StaticPlanner([Step("r1", "q", group=0), Step("reseacher", "typo", group=1)])
    )
    coord = _coord(policy, {"r1": first})

    with pytest.raises(PlanShapeError) as exc:
        asyncio.run(coord.run("goal", _ctx("v1")))

    assert "'reseacher'" in str(exc.value)
    assert "['r1']" in str(exc.value)  # the roster, so the typo is obvious
    assert first.calls == [], "a plan that cannot finish must not spend first"


def test_the_shape_error_is_still_a_value_error() -> None:
    """``PlanShapeError`` subclasses ``ValueError`` so callers already catching
    the plan-construction ``ValueError`` keep working."""
    assert issubclass(PlanShapeError, ValueError)


# ── 2. an unknown child, best-effort mode ───────────────────────────────────


def test_best_effort_isolates_an_unknown_child_and_runs_the_rest() -> None:
    """A live ``Planner`` names the child it wants, so an unknown name can be
    runtime data rather than a typo. ``best_effort=True`` records it and keeps
    going — which is what the mode promises and what a mid-loop ``KeyError``
    could not deliver."""
    first, third = _Rec("r1"), _Rec("r3")
    policy = PlanPolicy(
        planner=StaticPlanner(
            [
                Step("r1", "q", group=0),
                Step("ghost", "nowhere", group=1),
                Step("r3", "last", group=2),
            ]
        ),
        best_effort=True,
    )
    coord = _coord(policy, {"r1": first, "r3": third})

    res = asyncio.run(coord.run("goal", _ctx("v2")))

    assert res.stop_reason == "complete"
    assert first.calls == ["q"] and third.calls == ["last"]  # both real steps ran
    (name, failure), = res.evals["errors"]
    assert name == "ghost"
    assert isinstance(failure, Failure)
    # PERMANENT: re-dispatching a name that is not on the roster cannot succeed,
    # and a caller reading ``retriable`` must not be told to try again.
    assert failure.category is ErrorClass.PERMANENT and failure.retriable is False
    assert "ghost" in failure.message


# ── 3. a gate colliding with work in one group ──────────────────────────────


def test_a_gate_sharing_a_group_with_work_is_refused() -> None:
    """The plan does not say whether the work belongs before or after the
    human's decision, so the framework refuses instead of picking one. Before
    this, step 'a' was announced in the trace and never ran — on approve OR on
    reject."""
    a, b = _Rec("a"), _Rec("b")
    policy = PlanPolicy(
        planner=StaticPlanner(
            [Step("a", "first", group=0), Step.gate("g", group=0), Step("b", "after", group=1)]
        )
    )
    coord = _coord(policy, {"a": a, "b": b})

    with pytest.raises(PlanShapeError) as exc:
        asyncio.run(coord.run("goal", _ctx("v3", InMemoryStore())))

    msg = str(exc.value)
    assert "group 0" in msg and "'g'" in msg and "'a'" in msg
    assert a.calls == [] and b.calls == []


def test_a_gate_alone_in_its_group_is_fine() -> None:
    """The positive control: the canonical shape — work, then a gate, then more
    work — is untouched by the check."""
    a, b = _Rec("a"), _Rec("b")
    policy = PlanPolicy(
        planner=StaticPlanner(
            [Step("a", "first", group=0), Step.gate("g", group=1), Step("b", "after", group=2)]
        )
    )
    coord = _coord(policy, {"a": a, "b": b})
    ctx = _ctx("v4", InMemoryStore())

    res = asyncio.run(coord.run("goal", ctx))
    assert res.is_suspended and a.calls == ["first"] and b.calls == []

    resumed = asyncio.run(policy.resume(coord, {"g": "approve"}, ctx))
    assert resumed.stop_reason == "complete" and b.calls == ["after"]


# ── 4. a malformed step ─────────────────────────────────────────────────────


def test_a_step_with_neither_an_agent_nor_a_gate_is_refused() -> None:
    """``Step()`` is constructible because gate steps need ``agent=None``. A step
    that is neither has nothing to dispatch and nothing to wait for; it used to
    reach ``children[None]``."""
    policy = PlanPolicy(planner=StaticPlanner([Step(input="orphan", group=0)]))
    coord = _coord(policy, {"r1": _Rec("r1")})
    with pytest.raises(PlanShapeError, match="neither an agent nor a gate"):
        asyncio.run(coord.run("goal", _ctx("v5")))


# ── 5. resume re-validates against the RE-SUPPLIED roster ───────────────────


def test_resume_revalidates_because_the_roster_is_re_supplied() -> None:
    """``resume`` takes a coordinator, so the roster on the way back need not be
    the one the plan was validated against — a service that rebuilds its
    coordinator from config can lose a child between suspend and resume. The
    checkpoint is already consumed at that point, so a raw ``KeyError`` here
    would strand the accumulated results with no way to retry."""
    a = _Rec("a")
    plan = [Step("a", "first", group=0), Step.gate("g", group=1), Step("b", "after", group=2)]
    policy = PlanPolicy(planner=StaticPlanner(plan))
    ctx = _ctx("v6", InMemoryStore())

    full = _coord(policy, {"a": a, "b": _Rec("b")})
    assert asyncio.run(full.run("goal", ctx)).is_suspended

    shrunk = _coord(policy, {"a": a})  # 'b' is gone on the way back
    with pytest.raises(PlanShapeError, match="'b'"):
        asyncio.run(policy.resume(shrunk, {"g": "approve"}, ctx))


# ── 6. the validator itself ─────────────────────────────────────────────────


def test_validation_preserves_step_order_and_gates() -> None:
    """A pure-function check: validation filters, it does not reorder. Group
    ordering is what makes a plan a plan, so a validator that sorted its output
    would silently rewrite the plan it was asked to check."""
    steps = [
        Step("b", "2", group=5),
        Step("a", "1", group=0),
        Step.gate("g", group=3),
        Step("a", "3", group=9),
    ]
    keep, dropped = _validate_plan(steps, {"a": _Rec("a"), "b": _Rec("b")}, best_effort=False)
    assert keep == steps and dropped == []


def test_an_empty_plan_validates_to_nothing() -> None:
    """No steps is not an error — ``_run_groups`` returns an empty completion,
    which is the honest result for an empty plan."""
    assert _validate_plan([], {}, best_effort=False) == ([], [])
