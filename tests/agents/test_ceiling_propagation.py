"""A ceiling breached below an `as_tool` boundary must reach the caller.

`_invoke_tool_safe` catches `except Exception` to reflect tool failures back to
the model — which is right: a tool that failed is information the model can act
on, a different argument or another approach. But `MeterExceeded` and
`BudgetExhausted` are both `RuntimeError` subclasses, so a hard ceiling raised
inside a sub-agent was caught by that same clause and handed to the model as a
retryable-looking string. Measured with `Budget(max_depth=4)` and an `as_tool`
chain 8 deep:

    tool msg the model saw: "ERROR: tool 'sub3' failed: MeterExceeded:
                             agent depth 5 > max_depth 4"
    result.stop_reason:     complete
    result.evals:           {}

The model retried, the ceiling was violated, and the caller — whom
`docs/cheatsheet.md` tells to `except MeterExceeded` — got `complete` and an
empty `evals`. Nothing in the public result said a limit had been crossed.

Letting it escape was only half the fix. The tool fan-out runs under
`asyncio.TaskGroup`, which re-raises everything wrapped, so the first attempt
produced `ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)`
and the documented `except MeterExceeded` still missed it.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit import Agent
from agentkit.agents._agent_helpers import _ceiling_errors, _unwrap_ceiling
from agentkit.agents.cognition import ReActCognition
from agentkit.agents.control.budget import ActorBudget, BudgetExhausted
from agentkit.kernel.types import ToolCall
from agentkit.runtime.meter import MeterExceeded
from agentkit.testing import FakeLLM, Turn, make_test_ctx
from agentkit.tools import tool
from agentkit.tools.from_agent import as_tool


def _calls_a_tool(name: str = "sub") -> FakeLLM:
    return FakeLLM.script(
        [Turn(tool_calls=[ToolCall("1", name, {"task": "go"})]), Turn(content="final")]
    )


# ── 1. the regression ───────────────────────────────────────────────────────


def test_a_depth_ceiling_reaches_the_caller_as_itself() -> None:
    """THE regression. `except MeterExceeded` is what the cheatsheet documents,
    so that is what must catch it — not an ExceptionGroup, not a `complete`
    result carrying the error as prose."""
    inner = Agent(name="inner", model="m")
    outer = Agent(
        name="outer", model="m", cognition=ReActCognition(tools=[as_tool(inner, name="sub")])
    )
    ctx = make_test_ctx(llm=_calls_a_tool())
    ctx.budget.max_depth = 0  # any ctx.child() must be refused

    with pytest.raises(MeterExceeded, match="max_depth"):
        asyncio.run(outer.run("go", ctx))


def test_an_actor_budget_ceiling_also_reaches_the_caller() -> None:
    """`BudgetExhausted` is a sibling of the same bug — also a `RuntimeError`,
    so also swallowed by the same clause.

    It is raised by `reserve_for_child`, not by `charge`: `charge` is documented
    to soft-exceed and never raise, so the loop can stop cleanly on the next
    `exhausted()` check. (I wrote this test against `charge` first and it
    failed — correctly. The design there is deliberate and worth not breaking.)
    """

    @tool(side_effecting=False)
    async def spawn(task: str, ctx=None) -> str:  # noqa: ANN001
        """Reserve a child slice larger than the envelope allows."""
        ctx.actor_budget.reserve_for_child(tokens=10_000, cost_usd=0.0, steps=1)
        return "reserved"

    agent = Agent(name="a", model="m", cognition=ReActCognition(tools=[spawn]))
    ctx = make_test_ctx(llm=_calls_a_tool("spawn"))
    ctx.actor_budget = ActorBudget(
        max_tokens=10, max_cost_usd=1.0, max_steps=5, max_wall_seconds=60.0
    )

    with pytest.raises(BudgetExhausted):
        asyncio.run(agent.run("go", ctx))


def test_charge_still_soft_exceeds_without_raising() -> None:
    """Guarding the design the test above discovered: `charge` records spend
    past the cap and lets the in-flight call finish, so a just-charged call
    never surfaces as a crash. The loop stops on `exhausted()` instead."""
    budget = ActorBudget(max_tokens=10, max_cost_usd=1.0, max_steps=5, max_wall_seconds=60.0)
    budget.charge(tokens=10_000, cost_usd=0.0, steps=1)  # must not raise
    assert budget.exhausted()


# ── 2. a genuine tool failure is still reflected, not raised ───────────────


def test_an_ordinary_tool_failure_still_reaches_the_model() -> None:
    """THE positive control, and the reason the broad catch exists. A fix that
    simply narrowed the clause would break the loop's whole recovery story."""

    @tool(side_effecting=False)
    async def flaky(task: str) -> str:
        """Fail in the ordinary way a tool fails, so the model can adapt."""
        raise RuntimeError("upstream 503")

    agent = Agent(name="a", model="m", cognition=ReActCognition(tools=[flaky]))
    result = asyncio.run(agent.run("go", make_test_ctx(llm=_calls_a_tool("flaky"))))

    assert result.stop_reason == "complete"
    tool_msgs = [m for m in result.evals.get("messages", []) if m.role == "tool"]
    assert not tool_msgs or "upstream 503" in tool_msgs[0].content


def test_a_tool_shape_error_is_still_reflected() -> None:
    """The other deliberate reflection path must be unaffected."""

    @tool(side_effecting=False, output_schema={"type": "object", "required": ["x"]})
    async def shaped(task: str) -> dict:
        """Return a value that does not match the declared output schema."""
        return {"wrong": 1}

    agent = Agent(name="a", model="m", cognition=ReActCognition(tools=[shaped]))
    result = asyncio.run(agent.run("go", make_test_ctx(llm=_calls_a_tool("shaped"))))
    assert result.stop_reason == "complete"


# ── 3. the unwrapping itself ────────────────────────────────────────────────


def test_a_ceiling_is_unwrapped_from_a_nested_group() -> None:
    """Nested fan-out nests the groups, so the search recurses."""
    ceiling = MeterExceeded("depth")
    nested = BaseExceptionGroup("outer", [BaseExceptionGroup("inner", [ceiling])])
    assert _unwrap_ceiling(nested) is ceiling


def test_a_group_without_a_ceiling_is_left_alone() -> None:
    """A genuine multi-failure fan-out IS a group, and must keep its shape —
    unwrapping one arbitrary member would hide the others."""
    group = BaseExceptionGroup("g", [ValueError("a"), KeyError("b")])
    assert _unwrap_ceiling(group) is None


def test_both_ceiling_types_are_recognised() -> None:
    """A third ceiling type added later must be added here too; this test is
    where that gets noticed."""
    assert set(_ceiling_errors()) == {MeterExceeded, BudgetExhausted}


def test_a_multi_failure_fan_out_still_raises_its_group() -> None:
    """The negative control for the unwrapping: with no ceiling present, the
    group propagates unchanged."""

    @tool(side_effecting=False)
    async def boom(task: str) -> str:
        """Fail with an error that is not a ceiling."""
        raise KeyError("nope")

    agent = Agent(name="a", model="m", cognition=ReActCognition(tools=[boom]))
    # A tool failure is caught and reflected, so this must NOT raise at all —
    # confirming the unwrap did not turn ordinary failures into exceptions.
    result = asyncio.run(agent.run("go", make_test_ctx(llm=_calls_a_tool("boom"))))
    assert result.stop_reason == "complete"
