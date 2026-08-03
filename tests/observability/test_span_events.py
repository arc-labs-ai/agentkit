"""Tests for narrative span events (B4b).

Verifies that the right span events land on the currently-open span for the documented
moments: a ``retry.attempt`` from the retry middleware, an ``output.coercion_failed`` from
the output_coerce middleware, and a ``context.compacted`` from the compaction middleware.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentkit.capabilities.compaction import SlidingWindowCompactor
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.middlewares import compaction, output_coerce, retry, tracing
from agentkit.runtime import Invoker, RunContext, Services
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


async def _nosleep(_d: float) -> None:
    return None


# ---- Capturing tracer that records add_event_to_current_span on a span stack ----------------


@dataclass
class _Span:
    name: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def add_event(self, name: str, **fields: Any) -> None:
        self.events.append((name, fields))


class _CapturingTrace:
    """Records spans + events on the currently-open span. Tests poll
    ``all_events()`` to assert what landed where without depending on
    real OTel infrastructure."""

    def __init__(self) -> None:
        self.spans: list[_Span] = []
        self._stack: list[_Span] = []

    @contextlib.contextmanager
    def span(self, name: str, kind: str = "", **attrs: Any):
        s = _Span(name=name, kind=kind, attrs=dict(attrs))
        self.spans.append(s)
        self._stack.append(s)
        try:
            yield s
        finally:
            self._stack.pop()

    def current_span_id(self):
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        if self._stack:
            self._stack[-1].events.append((name, fields))

    # Helpers --------------------------------------------------------------
    def all_events(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Flat list of ``(span_name, event_name, fields)`` across every recorded span."""
        return [(s.name, n, f) for s in self.spans for (n, f) in s.events]


def _ctx(*, llm, trace, chat_mw=()) -> RunContext:
    services = Services(
        invoker=Invoker(llm=llm, chat_middleware=list(chat_mw)),
        trace=trace,
    )
    return RunContext(correlation_id="r", scope=Scope(), services=services)


# ---- 1. retry middleware drops `retry.attempt` ---------------------------------------------


def test_retry_attempt_event_fires_with_attempt_and_error_type():
    """A transient pre-stream failure → the retry middleware drops a ``retry.attempt``
    event carrying ``attempt``, ``last_error_type``, and ``backoff_ms``."""

    async def go():
        llm = FakeLLM("ok", fail_times=1, fail_exc=TimeoutError("t"))
        trace = _CapturingTrace()
        ctx = _ctx(
            llm=llm,
            trace=trace,
            chat_mw=[tracing(), retry(max_attempts=3, sleep=_nosleep)],
        )
        await ctx.invoker.chat(ChatRequest(messages=[Message("user", "hi")], model="m"), ctx)
        events = [(n, f) for (_s, n, f) in trace.all_events()]
        attempt_events = [f for (n, f) in events if n == "retry.attempt"]
        assert len(attempt_events) == 1
        ev = attempt_events[0]
        assert ev["last_error_type"] == "TimeoutError"
        assert ev["attempt"] >= 1
        assert isinstance(ev["backoff_ms"], int)
        assert ev["backoff_ms"] >= 0

    _run(go())


def test_retry_event_fires_per_failed_attempt():
    """Two transient failures + a success → two ``retry.attempt`` events."""

    async def go():
        llm = FakeLLM("ok", fail_times=2, fail_exc=TimeoutError("t"))
        trace = _CapturingTrace()
        ctx = _ctx(
            llm=llm,
            trace=trace,
            chat_mw=[tracing(), retry(max_attempts=5, sleep=_nosleep)],
        )
        await ctx.invoker.chat(ChatRequest(messages=[Message("user", "hi")], model="m"), ctx)
        retry_events = [f for (_s, n, f) in trace.all_events() if n == "retry.attempt"]
        assert len(retry_events) == 2

    _run(go())


# ---- 2. output_coerce drops `output.coercion_failed` ---------------------------------------


def test_output_coercion_failure_drops_event():
    """A schema-violating response → the output_coerce middleware drops an
    ``output.coercion_failed`` event with the error type and re-raises."""

    from agentkit.capabilities.output_schema import OutputCoercionError, adapt

    pydantic = pytest.importorskip("pydantic")

    class Plan(pydantic.BaseModel):
        subject: str
        steps: list[str]

    async def go():
        llm = FakeLLM('{"subject": "x"}')  # missing required `steps`
        trace = _CapturingTrace()
        ctx = _ctx(
            llm=llm,
            trace=trace,
            chat_mw=[tracing(), output_coerce()],
        )
        ctx._output_adapter = adapt(Plan)  # type: ignore[attr-defined]

        with pytest.raises(OutputCoercionError):
            await ctx.invoker.chat(ChatRequest(messages=[Message("user", "p")], model="m"), ctx)

        coercion_events = [f for (_s, n, f) in trace.all_events() if n == "output.coercion_failed"]
        assert len(coercion_events) == 1
        ev = coercion_events[0]
        assert ev["error_type"] == "OutputCoercionError"
        assert ev["errors_count"] >= 1

    _run(go())


# ---- 3. compaction middleware drops `context.compacted` ------------------------------------


def test_context_compacted_event_fires_with_before_after_counts():
    """A transcript longer than the compactor's window → the compaction middleware
    rewrites ``ctx.request.messages`` and drops a ``context.compacted`` event with
    ``messages_before`` / ``messages_after`` / ``tokens_saved``."""

    async def go():
        # 10 messages, sliding window keeps the last 3 — guaranteed to drop.
        msgs = [
            Message("user" if i % 2 == 0 else "assistant", f"turn {i} " * 30) for i in range(10)
        ]
        llm = FakeLLM("ok", usage=Usage(1, 1, 0.0))
        trace = _CapturingTrace()
        ctx = _ctx(
            llm=llm,
            trace=trace,
            chat_mw=[tracing(), compaction(SlidingWindowCompactor(keep_recent=3))],
        )
        await ctx.invoker.chat(ChatRequest(messages=msgs, model="m"), ctx)
        compacted_events = [f for (_s, n, f) in trace.all_events() if n == "context.compacted"]
        assert len(compacted_events) == 1
        ev = compacted_events[0]
        assert ev["messages_before"] == 10
        assert ev["messages_after"] <= 4  # window + optional system
        assert ev["tokens_saved"] > 0

    _run(go())


def test_compaction_opens_context_compact_span():
    """A pass that reduces the transcript also opens a dedicated ``context.compact``
    span carrying ``agentkit.context.strategy`` + token counts — the structural
    record of the reduction (the event is the narrative pointer)."""

    async def go():
        msgs = [Message("user" if i % 2 == 0 else "assistant", f"x {i} " * 30) for i in range(10)]
        llm = FakeLLM("ok", usage=Usage(1, 1, 0.0))
        trace = _CapturingTrace()
        ctx = _ctx(
            llm=llm,
            trace=trace,
            chat_mw=[tracing(), compaction(SlidingWindowCompactor(keep_recent=3))],
        )
        await ctx.invoker.chat(ChatRequest(messages=msgs, model="m"), ctx)
        compact_spans = [s for s in trace.spans if s.name == "context.compact"]
        assert len(compact_spans) == 1
        s = compact_spans[0]
        assert s.attrs["agentkit.context.strategy"] == "SlidingWindowCompactor"
        assert "agentkit.context.tokens_before" in s.attrs
        assert "agentkit.context.tokens_after" in s.attrs

    _run(go())


def test_compaction_with_short_transcript_emits_no_event():
    """A transcript under the threshold — the compactor returns unchanged messages
    and no narrative event fires (the ``context.compact`` span still opens — it's
    the structural record of the *attempt*)."""

    async def go():
        msgs = [Message("user", "hi")]
        llm = FakeLLM("ok", usage=Usage(1, 1, 0.0))
        trace = _CapturingTrace()
        ctx = _ctx(
            llm=llm,
            trace=trace,
            chat_mw=[tracing(), compaction(SlidingWindowCompactor(keep_recent=10))],
        )
        await ctx.invoker.chat(ChatRequest(messages=msgs, model="m"), ctx)
        compacted_events = [f for (_s, n, f) in trace.all_events() if n == "context.compacted"]
        assert compacted_events == []

    _run(go())
