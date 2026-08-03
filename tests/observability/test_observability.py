"""OTel observability adapter — exercises the `otel_tracer()` TracePort end-to-end.

The adapter wraps an `opentelemetry.trace.Tracer` so that `ctx.trace.span(...)` opens a real
OTel span, the generic `tracing()` middleware stamps `gen_ai.usage.*` attributes on it, and the
span lands in whatever exporter the OTel SDK is configured with. Tests use an in-memory
exporter so assertions can read the recorded spans without standing up a real backend.

Skipped cleanly when the `[observability]` extra (opentelemetry-sdk) isn't installed.
"""

import asyncio

import pytest

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from agentkit.adapters.observability import otel_tracer  # noqa: E402
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage  # noqa: E402
from agentkit.middlewares import tracing  # noqa: E402
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402
from agentkit.testing import FakeLLM  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Build a fresh ``TracerProvider`` + ``InMemorySpanExporter`` for ONE test.

    The provider is LOCAL — we do NOT call ``ot.set_tracer_provider`` here. OTel
    forbids overriding the global TracerProvider once set, so sharing a global
    across tests gives stale-provider failures the second test onward. Each test
    passes its provider into ``otel_tracer(tracer_provider=…)`` for isolation."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_chat_span_lands_with_gen_ai_attributes():
    """The full path: tracing() middleware opens a span via the OTel TracePort, the FakeLLM
    returns a usage-bearing result, and the middleware stamps gen_ai.usage.* attributes on
    the span — which the InMemorySpanExporter then sees."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM("ok", usage=Usage(100, 20, 0.002)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    res = _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    assert res.content == "ok"

    spans = exporter.get_finished_spans()
    chat = next(s for s in spans if s.name == "chat")
    assert chat.attributes["gen_ai.request.model"] == "m"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 100
    assert chat.attributes["gen_ai.usage.output_tokens"] == 20


def test_current_span_id_returns_trace_context_inside_open_span():
    """``current_span_id()`` returns the in-context span's identifiers as a TraceContext —
    so ``RunContext.emit`` can deep-link an observation to the span that produced it."""
    provider, _exporter = _make_in_memory_provider()
    tracer = otel_tracer(provider)
    # Outside any span: no context.
    assert tracer.current_span_id() is None
    with tracer.span("work", "internal"):
        tc = tracer.current_span_id()
        assert tc is not None
        assert len(tc.trace_id) == 32 and len(tc.span_id) == 16


def test_add_event_to_current_span_is_a_noop_when_no_span_open():
    """The affordance must not raise when there is no open span (cold emit path)."""
    provider, _exporter = _make_in_memory_provider()
    tracer = otel_tracer(provider)
    # Must not raise.
    tracer.add_event_to_current_span("observation.emitted", kind="result")


def test_span_event_lands_on_open_span():
    """``add_event_to_current_span`` drops the event on whatever span is currently open."""
    provider, exporter = _make_in_memory_provider()
    tracer = otel_tracer(provider)
    with tracer.span("work", "internal"):
        tracer.add_event_to_current_span("observation.emitted", kind="result")
    spans = exporter.get_finished_spans()
    work = next(s for s in spans if s.name == "work")
    assert any(e.name == "observation.emitted" for e in work.events)
