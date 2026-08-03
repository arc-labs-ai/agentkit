"""Tracing middleware × SamplerPort integration.

The sampler is consulted at span open. ``True`` → the span opens
normally (and the chat result flows through unchanged). ``False`` →
no span opens, no attributes stamp, no replay write — but METRICS
still emit and the underlying handler STILL runs so the chat result
flows through. Sampling controls TRACE FIDELITY; metrics roll up
unconditionally. Gating metrics on the sampler would scale production
histograms by the sample rate.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.sampling import AlwaysOnSampler
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.middlewares import tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


# ── Capturing tracer (records span open/close) ───────────────────────────────


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


# ── Recording sampler (lets us assert how it was called + force a decision) ──


@dataclass
class _SamplerInvocation:
    operation: str
    correlation_id: str
    attrs: dict[str, Any] = field(default_factory=dict)


class _RecordingSampler:
    """Always returns the configured ``decision`` and records every call."""

    def __init__(self, decision: bool) -> None:
        self.decision = decision
        self.calls: list[_SamplerInvocation] = []

    def should_sample(
        self,
        *,
        operation: str,
        correlation_id: str,
        attrs: Mapping[str, str | int | float] | None = None,
    ) -> bool:
        self.calls.append(
            _SamplerInvocation(
                operation=operation,
                correlation_id=correlation_id,
                attrs=dict(attrs or {}),
            )
        )
        return self.decision


# ── Always-on path: span recorded normally ───────────────────────────────────


def test_always_on_sampler_records_chat_span_normally():
    """The default sampler keeps every span — the tracing middleware
    behaves exactly as before sampling existed."""

    async def go() -> None:
        trace = _CapturingTrace()
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=5, output_tokens=3, cost_usd=0.0)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r-always-on",
            Scope(),
            Budget(),
            Services(invoker=inv, trace=trace, sampler=AlwaysOnSampler()),
        )
        res = await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)

        assert res.content == "ok"
        chat_spans = [s for s in trace.spans if s.name == "chat"]
        assert len(chat_spans) == 1
        # Attributes still stamp.
        assert chat_spans[0].attrs.get("gen_ai.request.model") == "m"
        assert chat_spans[0].attrs.get("gen_ai.usage.input_tokens") == 5

    _run(go())


# ── Sampler-drops path: span skipped, result still flows ─────────────────────


def test_sampler_returning_false_skips_chat_span():
    """When the sampler drops the chat span, no chat span is opened —
    no attributes stamp, no replay write. The chat call still returns
    a normal result; sampling controls trace fidelity. (Metrics still
    emit — see ``test_metrics_emit_even_when_sampler_drops_span``.)"""

    async def go() -> None:
        trace = _CapturingTrace()
        sampler = _RecordingSampler(decision=False)
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=5, output_tokens=3, cost_usd=0.0)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r-drop-me",
            Scope(),
            Budget(),
            Services(invoker=inv, trace=trace, sampler=sampler),
        )
        res = await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)

        # Result still flows through normally.
        assert res.content == "ok"
        # No chat span recorded.
        chat_spans = [s for s in trace.spans if s.name == "chat"]
        assert chat_spans == []

    _run(go())


# ── metrics emit even when the sampler drops the span ───────────────────────


@dataclass
class _Hist:
    name: str
    value: int | float
    tags: dict[str, str] = field(default_factory=dict)


class _RecordingMetrics:
    """Minimal MetricsPort capturing every counter + histogram call."""

    def __init__(self) -> None:
        self.counters: list[_Hist] = []
        self.histograms: list[_Hist] = []

    def add_counter(self, name: str, value: int | float = 1, *, tags=None) -> None:
        self.counters.append(_Hist(name=name, value=value, tags=dict(tags or {})))

    def record_histogram(self, name: str, value: int | float, *, tags=None) -> None:
        self.histograms.append(_Hist(name=name, value=value, tags=dict(tags or {})))


def test_metrics_emit_even_when_sampler_drops_span():
    """Token-usage + duration histograms must still record on the
    sampler-dropped path. Gating metrics on the sampler would cause
    production dashboards to under-count by the sample rate."""

    async def go() -> None:
        trace = _CapturingTrace()
        metrics = _RecordingMetrics()
        sampler = _RecordingSampler(decision=False)  # always drop
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.002)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r-metrics-survive",
            Scope(),
            Budget(),
            Services(invoker=inv, trace=trace, sampler=sampler, metrics=metrics),
        )
        await inv.chat(ChatRequest([Message("user", "q")], "claude-sonnet-4-6"), ctx)

        # No chat span (sampler dropped it) — that's the existing
        # trace-fidelity behavior.
        assert [s for s in trace.spans if s.name == "chat"] == []

        # But the token histograms STILL fire.
        token_hists = [h for h in metrics.histograms if h.name == "gen_ai.client.token.usage"]
        assert len(token_hists) == 2
        by_type = {h.tags["gen_ai.token.type"]: h for h in token_hists}
        assert by_type["input"].value == 100
        assert by_type["output"].value == 20

        # And the operation duration histogram fires too.
        dur = [h for h in metrics.histograms if h.name == "gen_ai.client.operation.duration"]
        assert len(dur) == 1
        assert dur[0].value > 0  # some positive duration

    _run(go())


def test_error_counter_emits_even_when_sampler_drops_span():
    """Symmetric to the histogram test: a failing call on the
    sampler-dropped path still emits ``gen_ai.client.error.count`` so
    error-rate dashboards reflect reality."""

    class _BoomLLM:
        async def stream(self, **kwargs):
            raise RuntimeError("provider 503")
            yield  # pragma: no cover — make this an async generator

    async def go() -> None:
        metrics = _RecordingMetrics()
        sampler = _RecordingSampler(decision=False)
        inv = Invoker(llm=_BoomLLM(), chat_middleware=[tracing()])
        ctx = RunContext(
            "r-error",
            Scope(),
            Budget(),
            Services(invoker=inv, sampler=sampler, metrics=metrics),
        )
        try:
            await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)
        except RuntimeError:
            pass  # expected propagation

        error_counts = [c for c in metrics.counters if c.name == "gen_ai.client.error.count"]
        assert len(error_counts) == 1
        assert error_counts[0].tags.get("error.type") == "RuntimeError"

    _run(go())


def test_sampler_called_with_chat_operation_and_correlation_id():
    """The sampler is consulted with ``operation="chat"`` and the run's
    ``correlation_id``. ``attrs`` carries the request-time model id so
    rule-based samplers can branch on it."""

    async def go() -> None:
        sampler = _RecordingSampler(decision=True)
        inv = Invoker(
            llm=FakeLLM("ok", usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "run-id-xyz",
            Scope(),
            Budget(),
            Services(invoker=inv, sampler=sampler),
        )
        await inv.chat(ChatRequest([Message("user", "q")], "claude-sonnet"), ctx)

        chat_calls = [c for c in sampler.calls if c.operation == "chat"]
        assert len(chat_calls) == 1
        assert chat_calls[0].correlation_id == "run-id-xyz"
        # request-model attribute is forwarded for rule-based sampling.
        assert chat_calls[0].attrs.get("gen_ai.request.model") == "claude-sonnet"

    _run(go())
