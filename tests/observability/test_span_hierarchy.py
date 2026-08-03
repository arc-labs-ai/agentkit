"""Tests for the OTel GenAI v1.41 span hierarchy (B4b).

Verifies that ``Workflow.run`` opens an ``invoke_workflow`` root, ``Agent.run`` opens an
``invoke_agent`` span with the documented ``gen_ai.*`` + ``agentkit.*`` attributes, and that
the existing ``chat`` span fires as a child of ``invoke_agent``. Coordinator-mode agents
flip ``agentkit.agent.is_coordinator=True`` and stamp the policy class name.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any

from agentkit.agents import Agent, RoundRobinPolicy, Workflow
from agentkit.agents.cognition import CoordinatorCognition, ReActCognition
from agentkit.kernel.types import Scope
from agentkit.middlewares import tracing
from agentkit.runtime import Invoker, RunContext, Services
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


# ---- A capturing TracePort that records opens / closes / parent-child --------------------


@dataclass
class _RecordedSpan:
    name: str
    kind: str
    attrs: dict[str, Any]
    parent_name: str | None
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def add_event(self, name: str, **fields: Any) -> None:
        self.events.append((name, fields))


class _CapturingTrace:
    """Records every ``span()`` open with parent context. ``current_span_id`` /
    ``add_event_to_current_span`` are wired so middlewares that target the surrounding
    span still work."""

    def __init__(self) -> None:
        self.spans: list[_RecordedSpan] = []
        self._stack: list[_RecordedSpan] = []

    @contextlib.contextmanager
    def span(self, name: str, kind: str = "", **attrs: Any):
        parent = self._stack[-1].name if self._stack else None
        rec = _RecordedSpan(name=name, kind=kind, attrs=dict(attrs), parent_name=parent)
        self.spans.append(rec)
        self._stack.append(rec)
        try:
            yield rec
        finally:
            self._stack.pop()

    def current_span_id(self):
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        if self._stack:
            self._stack[-1].events.append((name, fields))

    # Helpers for tests --------------------------------------------------------
    def by_name(self, name: str) -> list[_RecordedSpan]:
        return [s for s in self.spans if s.name == name]


def _ctx(*, llm, trace) -> RunContext:
    services = Services(
        invoker=Invoker(llm=llm, chat_middleware=[tracing()]),
        trace=trace,
    )
    return RunContext(correlation_id="r", scope=Scope(), services=services)


# ---- 1. Agent.run opens invoke_agent with expected attributes ----------------------------


def test_agent_run_opens_invoke_agent_span():
    """``Agent.run`` opens exactly one ``invoke_agent`` span carrying ``gen_ai.agent.name``,
    ``gen_ai.agent.tools_count``, ``gen_ai.agent.has_output_schema``, and ``agentkit.agent.id``."""

    async def go():
        llm = FakeLLM("hi")
        trace = _CapturingTrace()
        ctx = _ctx(llm=llm, trace=trace)
        agent = Agent(name="leaf", model="m", prompt="you are leaf")
        await agent.run("task", ctx)

        opens = trace.by_name("invoke_agent")
        assert len(opens) == 1
        s = opens[0]
        assert s.attrs["gen_ai.agent.name"] == "leaf"
        assert s.attrs["gen_ai.agent.tools_count"] == 0
        assert s.attrs["gen_ai.agent.has_output_schema"] is False
        assert "agentkit.agent.id" in s.attrs

    _run(go())


def test_agent_with_tools_count_reflects_registry():
    """The ``gen_ai.agent.tools_count`` attribute matches the registered tool count."""

    async def go():
        def get_weather(args, ctx):  # noqa: ARG001
            """Return the current weather conditions for the named city."""
            return "sunny"

        def get_time(args, ctx):  # noqa: ARG001
            """Return the current UTC wall-clock time for the caller's region."""
            return "noon"

        llm = FakeLLM("hi")
        trace = _CapturingTrace()
        ctx = _ctx(llm=llm, trace=trace)
        agent = Agent(
            name="leaf",
            model="m",
            prompt="p",
            cognition=ReActCognition(tools=[get_weather, get_time]),
        )
        await agent.run("task", ctx)
        s = trace.by_name("invoke_agent")[0]
        assert s.attrs["gen_ai.agent.tools_count"] == 2

    _run(go())


# ---- 2. Workflow.run opens invoke_workflow containing child invoke_agent spans -----------


def test_workflow_run_opens_invoke_workflow_and_child_invoke_agents():
    """A workflow with two agent nodes opens one ``invoke_workflow`` root with two
    child ``invoke_agent`` spans, each parented by the workflow span."""

    async def go():
        llm = FakeLLM("answer")
        trace = _CapturingTrace()
        ctx = _ctx(llm=llm, trace=trace)
        wf = Workflow("planner")
        a1 = Agent(name="a1", model="m", prompt="p1")
        a2 = Agent(name="a2", model="m", prompt="p2")
        wf.agent("n1", a1)
        wf.agent("n2", a2, after="n1")
        await wf.run("goal", ctx)

        wf_spans = trace.by_name("invoke_workflow")
        ag_spans = trace.by_name("invoke_agent")
        assert len(wf_spans) == 1
        assert len(ag_spans) == 2
        wf_span = wf_spans[0]
        assert wf_span.attrs["gen_ai.workflow.name"] == "planner"
        assert wf_span.attrs["agentkit.workflow.nodes_count"] == 2
        assert wf_span.attrs["gen_ai.workflow.autonomy"] in ("auto", "gated", "manual")
        for s in ag_spans:
            assert s.parent_name == "invoke_workflow"

    _run(go())


# ---- 3. chat span fires as a child of invoke_agent ----------------------------------------


def test_chat_span_is_child_of_invoke_agent():
    """The ``tracing()`` middleware opens its ``chat`` span inside the agent's
    ``invoke_agent`` scope."""

    async def go():
        llm = FakeLLM("hi")
        trace = _CapturingTrace()
        ctx = _ctx(llm=llm, trace=trace)
        agent = Agent(name="leaf", model="m", prompt="p")
        await agent.run("task", ctx)
        chats = trace.by_name("chat")
        assert chats, "expected a `chat` span from the tracing middleware"
        assert all(c.parent_name == "invoke_agent" for c in chats)

    _run(go())


# ---- 4. Coordinator agent stamps is_coordinator=True + policy class name ------------------


def test_coordinator_agent_stamps_is_coordinator_and_policy():
    """An ``Agent(children=...)`` with a policy gets ``agentkit.agent.is_coordinator=True``
    and ``agentkit.agent.policy=<PolicyClassName>`` on its ``invoke_agent`` span."""

    async def go():
        llm = FakeLLM("done")
        trace = _CapturingTrace()
        ctx = _ctx(llm=llm, trace=trace)
        leaf = Agent(name="worker", model="m", prompt="p")
        # Round-robin one tick over a single child — minimal viable coordinator wiring
        # for the span-attribute assertion.
        from agentkit.agents.control.termination import MaxTurns

        coord = Agent(
            name="coord",
            cognition=CoordinatorCognition(
                children={"worker": leaf},
                policy=RoundRobinPolicy(),
                termination=MaxTurns(1),
            ),
        )
        await coord.run("task", ctx)
        opens = [
            s for s in trace.by_name("invoke_agent") if s.attrs["gen_ai.agent.name"] == "coord"
        ]
        assert len(opens) == 1
        s = opens[0]
        assert s.attrs["agentkit.agent.is_coordinator"] is True
        assert s.attrs["agentkit.agent.policy"] == "RoundRobinPolicy"

    _run(go())


def test_leaf_agent_does_not_stamp_is_coordinator():
    """A leaf ``Agent`` (no ``children=``) leaves ``agentkit.agent.is_coordinator`` unset —
    not present is the canonical "not a coordinator" signal."""

    async def go():
        llm = FakeLLM("hi")
        trace = _CapturingTrace()
        ctx = _ctx(llm=llm, trace=trace)
        agent = Agent(name="leaf", model="m", prompt="p")
        await agent.run("task", ctx)
        s = trace.by_name("invoke_agent")[0]
        assert "agentkit.agent.is_coordinator" not in s.attrs
        assert "agentkit.agent.policy" not in s.attrs

    _run(go())
