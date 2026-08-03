"""Tests for the Observation ↔ TracePort link (B4a).

Verifies that ``Observation.trace_context`` is attached when a span is open via the run's
``TracePort``, is None when no tracer is configured, and that the link is structural-equality
stable (the dataclass is frozen). Also verifies the CRITICAL-kind span-event mirror.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from agentkit.adapters.observer import CollectingObserver
from agentkit.kernel.observation import Observation, TraceContext
from agentkit.kernel.types import Scope
from agentkit.runtime.context import NoopTrace, RunContext, Services


def _run(coro):
    return asyncio.run(coro)


# ---- Value-type tests --------------------------------------------------------------------------


def test_observation_trace_context_default_is_none():
    obs = Observation(kind="result")
    assert obs.trace_context is None


def test_observation_with_trace_context_round_trips():
    tc = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    obs = Observation(kind="result", trace_context=tc)
    assert obs.trace_context == tc
    assert obs.trace_context.trace_id == "a" * 32
    assert obs.trace_context.span_id == "b" * 16


def test_trace_context_is_frozen():
    from dataclasses import FrozenInstanceError

    tc = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    # frozen dataclasses raise FrozenInstanceError on assignment.
    with pytest.raises(FrozenInstanceError):
        tc.trace_id = "c" * 32  # type: ignore[misc]


def test_trace_context_equality_is_structural():
    a = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    b = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    c = TraceContext(trace_id="a" * 32, span_id="c" * 16)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)  # frozen → hashable


# ---- TracePort default-impl tests --------------------------------------------------------------


def test_noop_trace_current_span_id_returns_none():
    """Default trace port (no real tracer) — current_span_id() must return None so cold-emit
    doesn't crash and Observations land without a trace link."""
    assert NoopTrace().current_span_id() is None


def test_noop_trace_add_event_to_current_span_is_noop():
    """The default `add_event_to_current_span` must accept any args and never raise so
    `RunContext.emit` can call it unconditionally on critical kinds without a tracer attached."""
    NoopTrace().add_event_to_current_span("observation.emitted", kind="result")
    NoopTrace().add_event_to_current_span("anything")
    # No assertion needed — the contract is "never raises".


# ---- ctx.emit attachment tests -----------------------------------------------------------------


class _RecordingTrace:
    """A TracePort double that surfaces a fixed span-id and records calls.

    `current_span_id` returns ``self.fixed_ctx`` (None or a TraceContext) so tests can
    pick the regime. `add_event_to_current_span` appends each invocation to ``events``."""

    def __init__(self, fixed_ctx: TraceContext | None) -> None:
        self.fixed_ctx = fixed_ctx
        self.events: list[tuple[str, dict]] = []

    @contextlib.contextmanager
    def span(self, name, kind, **attrs):
        class _S:
            def set(self, *a, **kw):
                pass

            def add_event(self, *a, **kw):
                pass

        yield _S()

    def current_span_id(self):
        return self.fixed_ctx

    def add_event_to_current_span(self, name: str, **fields) -> None:
        self.events.append((name, fields))


def test_emit_attaches_trace_context_when_span_open():
    """When emit runs with a tracer that reports a current span, the Observation carries
    the same trace_id + span_id."""

    async def go():
        tc = TraceContext(trace_id="t" * 32, span_id="s" * 16)
        obs = CollectingObserver()
        trace = _RecordingTrace(fixed_ctx=tc)
        ctx = RunContext("run-1", Scope(1, 2), services=Services(observer=obs, trace=trace))
        await ctx.emit("summary", "step done", payload={"i": 1})
        assert len(obs.items) == 1
        assert obs.items[0].trace_context == tc
        assert obs.items[0].run_id == "run-1"

    _run(go())


def test_emit_with_no_open_span_leaves_context_none():
    """When the tracer is attached but no span is open (current_span_id() returns None),
    Observations still emit cleanly with trace_context=None."""

    async def go():
        obs = CollectingObserver()
        trace = _RecordingTrace(fixed_ctx=None)
        ctx = RunContext("run-2", Scope(1, 2), services=Services(observer=obs, trace=trace))
        await ctx.emit("progress", "tick")
        assert obs.items[0].trace_context is None

    _run(go())


def test_emit_with_noop_trace_leaves_context_none():
    """When the run has the default no-op tracer, Observations still emit cleanly with
    trace_context=None (the canonical zero-config path)."""

    async def go():
        obs = CollectingObserver()
        ctx = RunContext("run-3", Scope(1, 2), services=Services(observer=obs))
        await ctx.emit("progress", "tick")
        assert obs.items[0].trace_context is None

    _run(go())


def test_emit_critical_kind_mirrors_span_event():
    """For CRITICAL kinds (result/error), ctx.emit also drops a span event so the trace
    timeline carries the critical observation without cross-referencing the observer stream."""

    async def go():
        tc = TraceContext(trace_id="t" * 32, span_id="s" * 16)
        obs = CollectingObserver()
        trace = _RecordingTrace(fixed_ctx=tc)
        ctx = RunContext("run-4", Scope(1, 2), services=Services(observer=obs, trace=trace))
        await ctx.emit("result", "final", payload={"ok": True})
        # The Observation made it through.
        assert obs.items[0].kind == "result"
        assert obs.items[0].trace_context == tc
        # ...and the tracer was asked to drop a span event mirroring it.
        assert ("observation.emitted", {"kind": "result"}) in trace.events

    _run(go())


def test_emit_non_critical_kind_does_not_mirror_span_event():
    """Only CRITICAL kinds get the mirrored span event — progress/summary do not, so the
    trace timeline isn't drowned in routine telemetry."""

    async def go():
        tc = TraceContext(trace_id="t" * 32, span_id="s" * 16)
        obs = CollectingObserver()
        trace = _RecordingTrace(fixed_ctx=tc)
        ctx = RunContext("run-5", Scope(1, 2), services=Services(observer=obs, trace=trace))
        await ctx.emit("progress", "tick")
        await ctx.emit("summary", "step done")
        assert trace.events == []  # no mirror events for non-critical kinds

    _run(go())


def test_emit_never_raises_when_trace_port_misbehaves():
    """`current_span_id` and `add_event_to_current_span` are best-effort — if a tracer throws,
    the run must still get its Observation through."""

    class ExplodingTrace:
        @contextlib.contextmanager
        def span(self, name, kind, **attrs):
            yield object()

        def current_span_id(self):
            raise RuntimeError("boom")

        def add_event_to_current_span(self, name, **fields):
            raise RuntimeError("boom")

    async def go():
        obs = CollectingObserver()
        ctx = RunContext(
            "run-6",
            Scope(1, 2),
            services=Services(observer=obs, trace=ExplodingTrace()),
        )
        await ctx.emit("error", "crashed")  # CRITICAL → exercises add_event path too
        assert len(obs.items) == 1
        assert obs.items[0].kind == "error"
        assert obs.items[0].trace_context is None  # swallowed → None

    _run(go())
