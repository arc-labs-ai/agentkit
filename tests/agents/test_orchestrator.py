"""Coordinator `Agent` + `PlanPolicy` (supervisor for a plan of named-child Steps):
groups run sequentially, steps within a group concurrently under the shared tree budget;
fail-fast by default, best_effort isolates failures. Offline & deterministic."""

import asyncio

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies.plan import PlanPolicy, StaticPlanner, Step
from agentkit.kernel.errors import Failure
from agentkit.kernel.types import Scope
from agentkit.testing import FakeLLM, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


class _Boom:
    name = "boom"

    async def run(self, task, ctx):
        raise ValueError("agent exploded")


def test_groups_run_in_order_and_merge_usage():
    children = {"a": Agent("a", "m"), "b": Agent("b", "m"), "c": Agent("c", "m")}
    plan = [Step("a", "x", group=0), Step("b", "y", group=0), Step("c", "z", group=1)]
    coord = Agent(
        name="orch",
        cognition=CoordinatorCognition(
            children=children, policy=PlanPolicy(planner=StaticPlanner(plan))
        ),
    )
    res = _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2))))
    assert len(res.evals["results"]) == 3  # all three steps ran
    assert res.usage.input_tokens == 30  # merged across the shared budget (3 × FakeLLM's 10)
    assert res.evals["errors"] == []


def test_best_effort_isolates_a_failed_step_and_keeps_partial_results():
    children = {"ok": Agent("ok", "m"), "boom": _Boom()}
    plan = [Step("ok", "x", group=0), Step("boom", "y", group=0)]
    coord = Agent(
        name="orch",
        cognition=CoordinatorCognition(
            children=children,
            policy=PlanPolicy(planner=StaticPlanner(plan), best_effort=True),
        ),
    )
    res = _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2))))
    assert len(res.evals["results"]) == 1  # the good step survived
    errors = res.evals["errors"]
    assert len(errors) == 1 and errors[0][0] == "boom"
    # The failed slot is now a ``Failure`` (first-class error data) — not the raw
    # ``ValueError``. The originating exception is preserved on ``.cause`` so a caller
    # can still switch on its type.
    failure = errors[0][1]
    assert isinstance(failure, Failure)
    assert isinstance(failure.cause, ValueError)


def test_fail_fast_default_raises_when_a_step_fails():
    children = {"ok": Agent("ok", "m"), "boom": _Boom()}
    plan = [Step("ok", "x", group=0), Step("boom", "y", group=0)]
    coord = Agent(
        name="orch",
        cognition=CoordinatorCognition(
            children=children,
            policy=PlanPolicy(planner=StaticPlanner(plan)),  # best_effort=False (default)
        ),
    )
    with pytest.raises(BaseExceptionGroup):  # the group is cancelled and raises
        _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2))))


def test_run_needs_steps_or_a_planner():
    coord = Agent(
        name="orch", cognition=CoordinatorCognition(children={}, policy=PlanPolicy())
    )  # no planner, no steps
    with pytest.raises(ValueError):
        _run(coord.run("goal", make_test_ctx(llm=FakeLLM("ok"), scope=Scope(1, 2))))
