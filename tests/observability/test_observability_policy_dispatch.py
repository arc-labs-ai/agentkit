"""Phase 3 D3 — policy.dispatch span event per coordinator child selection.

Each policy drops a ``policy.dispatch`` event on the open invoke_agent span when it
picks the next child. Carries the policy class name, the selected child, and a
human-readable selection reason ("round-robin index 2", "selector returned 'bob'",
"plan step 1/3", "ledger assessor picked 'alice'").
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any

from agentkit.agents import Agent
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.control.termination import MaxTurns
from agentkit.agents.policies.ledger import LedgerPolicy, ProgressLedger
from agentkit.agents.policies.plan import PlanPolicy, StaticPlanner, Step
from agentkit.agents.policies.roundrobin import RoundRobinPolicy
from agentkit.agents.policies.selector_policy import SelectorPolicy
from agentkit.kernel.types import Scope
from agentkit.middlewares import tracing
from agentkit.runtime import Invoker, RunContext, Services
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


# Local capturing TracePort — mirrors test_span_hierarchy._CapturingTrace so we
# can read ``policy.dispatch`` events off the open ``invoke_agent`` span without
# standing up the full OTel SDK.
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


def _ctx(*, llm, trace) -> RunContext:
    services = Services(
        invoker=Invoker(llm=llm, chat_middleware=[tracing()]),
        trace=trace,
    )
    return RunContext(correlation_id="r", scope=Scope(), services=services)


def _dispatch_events(trace: _CapturingTrace) -> list[dict[str, Any]]:
    """Flatten every ``policy.dispatch`` event from every span the test ran."""
    out: list[dict[str, Any]] = []
    for s in trace.spans:
        for name, fields in s.events:
            if name == "policy.dispatch":
                out.append(fields)
    return out


# ---- RoundRobinPolicy ---------------------------------------------------------


def test_roundrobin_policy_emits_dispatch_events():
    """Each iteration drops a policy.dispatch event with selection_reason."""

    async def go():
        trace = _CapturingTrace()
        ctx = _ctx(llm=FakeLLM("ok"), trace=trace)
        a = Agent(name="a", model="m", prompt="p")
        b = Agent(name="b", model="m", prompt="p")
        coord = Agent(
            name="coord",
            cognition=CoordinatorCognition(
                children={"a": a, "b": b},
                policy=RoundRobinPolicy(),
                termination=MaxTurns(2),
            ),
        )
        await coord.run("task", ctx)
        events = _dispatch_events(trace)
        # Two children, MaxTurns(2) → exactly 2 dispatch events.
        assert len(events) == 2
        assert events[0]["policy"] == "RoundRobinPolicy"
        assert events[0]["selected_child"] == "a"
        assert events[0]["selection_reason"] == "round-robin index 0"
        assert events[1]["selected_child"] == "b"
        assert events[1]["selection_reason"] == "round-robin index 1"

    _run(go())


# ---- SelectorPolicy -----------------------------------------------------------


def test_selector_policy_emits_dispatch_event():
    """Selector dispatch event names the selected child + selector reason."""

    def _pick_b(_transcript, _agents) -> str:
        return "b"

    async def go():
        trace = _CapturingTrace()
        ctx = _ctx(llm=FakeLLM("ok"), trace=trace)
        a = Agent(name="a", model="m", prompt="p")
        b = Agent(name="b", model="m", prompt="p")
        coord = Agent(
            name="coord",
            cognition=CoordinatorCognition(
                children={"a": a, "b": b},
                policy=SelectorPolicy(selector=_pick_b),
                termination=MaxTurns(1),
            ),
        )
        await coord.run("task", ctx)
        events = _dispatch_events(trace)
        assert len(events) == 1
        assert events[0]["policy"] == "SelectorPolicy"
        assert events[0]["selected_child"] == "b"
        assert "selector returned 'b'" in events[0]["selection_reason"]

    _run(go())


def test_selector_dispatch_event_reports_fallback_when_choice_unknown():
    """An unknown selector choice falls back to round-robin — the event's reason
    names BOTH the (unknown) choice and the fallback path."""

    def _pick_unknown(_transcript, _agents) -> str:
        return "unknown_child"

    async def go():
        trace = _CapturingTrace()
        ctx = _ctx(llm=FakeLLM("ok"), trace=trace)
        a = Agent(name="a", model="m", prompt="p")
        b = Agent(name="b", model="m", prompt="p")
        coord = Agent(
            name="coord",
            cognition=CoordinatorCognition(
                children={"a": a, "b": b},
                policy=SelectorPolicy(selector=_pick_unknown),
                termination=MaxTurns(1),
            ),
        )
        await coord.run("task", ctx)
        events = _dispatch_events(trace)
        assert len(events) == 1
        reason = events[0]["selection_reason"]
        assert "selector returned 'unknown_child'" in reason
        assert "fallback round-robin" in reason

    _run(go())


# ---- PlanPolicy ---------------------------------------------------------------


def test_plan_policy_emits_dispatch_event_per_step():
    """PlanPolicy fires one policy.dispatch event per planned step, with the
    1-based step position in the reason."""

    async def go():
        trace = _CapturingTrace()
        ctx = _ctx(llm=FakeLLM("ok"), trace=trace)
        a = Agent(name="a", model="m", prompt="p")
        b = Agent(name="b", model="m", prompt="p")
        steps = [Step("a", "do-a", group=0), Step("b", "do-b", group=1)]
        coord = Agent(
            name="coord",
            cognition=CoordinatorCognition(
                children={"a": a, "b": b},
                policy=PlanPolicy(planner=StaticPlanner(steps=steps)),
            ),
        )
        await coord.run("task", ctx)
        events = _dispatch_events(trace)
        assert len(events) == 2
        assert events[0]["policy"] == "PlanPolicy"
        assert events[0]["selected_child"] == "a"
        assert events[0]["selection_reason"] == "plan step 1/2"
        assert events[1]["selected_child"] == "b"
        assert events[1]["selection_reason"] == "plan step 2/2"

    _run(go())


# ---- LedgerPolicy -------------------------------------------------------------


def test_ledger_policy_emits_dispatch_event():
    """LedgerPolicy fires policy.dispatch with the assessor-picked child."""

    def _assess(_ledger, _transcript, names):
        # Always pick the first child and never declare satisfied — we'll let the
        # MaxTurns/max_rounds cap end the run after exactly one dispatch.
        return ProgressLedger(
            satisfied=False,
            in_a_loop=False,
            progress_being_made=True,
            next_speaker=names[0] if names else "",
            instruction="go",
        )

    async def go():
        trace = _CapturingTrace()
        ctx = _ctx(llm=FakeLLM("DONE"), trace=trace)  # heuristic won't apply: custom assessor
        a = Agent(name="a", model="m", prompt="p")
        coord = Agent(
            name="coord",
            cognition=CoordinatorCognition(
                children={"a": a},
                policy=LedgerPolicy(assessor=_assess, max_rounds=1),
            ),
        )
        await coord.run("task", ctx)
        events = _dispatch_events(trace)
        assert len(events) == 1
        assert events[0]["policy"] == "LedgerPolicy"
        assert events[0]["selected_child"] == "a"
        assert "ledger assessor picked 'a'" in events[0]["selection_reason"]

    _run(go())
