"""Composition / nesting (ch17 §17.4): `as_tool` wraps any runnable as a tool an Agent can call
(explicit/coordinator inside emergent), and `Workflow.coordinator` runs a coordinator Agent
as a graph node (emergent inside explicit). Offline & deterministic."""

import asyncio

from agentkit.agents import Agent, AgentResult, RoundRobinPolicy, Workflow, WorkflowResult
from agentkit.agents.cognition import CoordinatorCognition, ReActCognition
from agentkit.agents.control import MaxMessages
from agentkit.kernel.types import Message, Scope, ToolCall, Usage
from agentkit.testing import FakeLLM, Turn, make_test_ctx
from agentkit.tools import ToolRegistry
from agentkit.tools.from_agent import as_tool, render_result


def _run(coro):
    return asyncio.run(coro)


# ---- render_result over the result types ------------------------------------------------------


def test_render_result_handles_agent_and_workflow_results():
    """`render_result` is the re-entry shim that turns any run result into text for an outer
    loop. AgentResult → `.output` (already a string, including the last assistant reply for a
    coordinator); WorkflowResult → terminal `outputs` as `k: v` lines."""
    assert render_result(AgentResult(output="hi", usage=Usage())) == "hi"

    # A coordinator's AgentResult — `.output` IS the last assistant message by construction.
    coord_res = AgentResult(
        output="ans",
        usage=Usage(),
        evals={
            "stop_reason": "max_messages",
            "messages": [Message("user", "q"), Message("assistant", "ans")],
            "results": [],
        },
    )
    assert render_result(coord_res) == "ans"

    wf = WorkflowResult(outputs={"k": "v"}, usage=Usage(), steps=1, stop_reason="complete")
    assert render_result(wf) == "k: v"


# ---- as_tool: wrap a runnable so a loop can call it -------------------------------------------


def test_as_tool_runs_a_workflow_and_renders_its_output():
    sub = Workflow()
    sub.fn("compute", lambda inp: "42")

    async def go():
        out = await as_tool(sub, name="subflow").run(
            {"task": "go"}, make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2))
        )
        assert out == "compute: 42"

    _run(go())


def test_agentloop_calls_a_subworkflow_as_a_tool():
    ran = {"n": 0}
    sub = Workflow()
    sub.fn(
        "step", lambda inp: ran.__setitem__("n", ran["n"] + 1) or "DONE"
    )  # fn-only → no LLM contention
    reg = ToolRegistry()
    reg.register(as_tool(sub, name="pipeline", description="run the pipeline"))

    # turn 1: the loop calls the sub-workflow tool; turn 2: it finishes
    llm = FakeLLM.script(
        [Turn(tool_calls=(ToolCall("c1", "pipeline", {"task": "x"}),)), Turn(content="final")]
    )
    res = _run(
        Agent("orch", "m", cognition=ReActCognition(tools=reg)).run(
            "go", make_test_ctx(llm=llm, scope=Scope(1, 2))
        )
    )
    # the graph really ran, inside the emergent loop
    assert res.output == "final" and ran["n"] == 1


# ---- Workflow.coordinator: a coordinator Agent as a graph node ---------------------------------


def test_workflow_coordinator_node_runs_a_coordinator_inside_the_graph():
    """A coordinator `Agent` plugged into a `Workflow.coordinator(...)` node runs the
    multi-agent loop on a `ctx.child()`, returns its last assistant message as the node's
    output, and that output threads into the next node like any other typed edge."""
    coord = Agent(
        name="discuss-coord",
        cognition=CoordinatorCognition(
            children={"a": Agent("a", "m")},
            policy=RoundRobinPolicy(),
            termination=MaxMessages(1),
        ),
    )
    wf = Workflow()
    wf.coordinator("discuss", coord)
    wf.fn("use", lambda inp: f"used: {inp['discuss']}", after="discuss")
    res = _run(wf.run("topic", make_test_ctx(llm=FakeLLM("hello"), scope=Scope(1, 2))))
    assert res.outputs["discuss"] == "hello"  # the coordinator's last assistant reply
    assert res.outputs["use"] == "used: hello"  # threaded into the next node
