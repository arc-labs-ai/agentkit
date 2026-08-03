"""Cache attribute stamping (prompt cache) + ``cache.hit`` span event (memoize).

Two facets of cache visibility:

- ``tracing()`` mirrors ``Usage.cache_read_tokens`` / ``cache_write_tokens``
  onto ``gen_ai.usage.cache_*_tokens`` span attributes, but ONLY when non-zero
  so the 95% of calls without prompt caching don't carry noisy zeros.
- ``memoize()`` drops a ``cache.hit`` span event on a hit so the trace timeline
  explains why no provider call happened on that turn.
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
from agentkit.adapters.store import InMemoryStore  # noqa: E402
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage  # noqa: E402
from agentkit.middlewares import memoize, tracing  # noqa: E402
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402
from agentkit.testing import FakeLLM  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _chat_span(exporter: InMemorySpanExporter):
    return next(s for s in exporter.get_finished_spans() if s.name == "chat")


def test_cache_read_tokens_attribute_lands_when_nonzero():
    """A provider that reports cache_read_tokens has the attribute mirrored on the span."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM(
            "ok",
            usage=Usage(
                input_tokens=100,
                output_tokens=20,
                cost_usd=0.002,
                cache_read_tokens=80,
                cache_write_tokens=5,
            ),
        ),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    assert chat.attributes["gen_ai.usage.cache_read_tokens"] == 80
    assert chat.attributes["gen_ai.usage.cache_write_tokens"] == 5


def test_cache_attributes_absent_when_zero():
    """Default Usage has 0 cache_read/write — both attributes MUST be absent (no noise)."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM("ok", usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.002)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    assert "gen_ai.usage.cache_read_tokens" not in chat.attributes
    assert "gen_ai.usage.cache_write_tokens" not in chat.attributes


def test_memoize_hit_drops_cache_hit_span_event():
    """First call (miss) hits the LLM and produces no cache.hit; second (hit) drops the event."""
    provider, exporter = _make_in_memory_provider()
    store = InMemoryStore()

    def _key(_call):
        return "fixed-cache-key"

    inv = Invoker(
        llm=FakeLLM("ok", usage=Usage(10, 5, 0.0001)),
        chat_middleware=[tracing(), memoize(key=_key, store=store)],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))

    # First call — MISS. memoize runs the inner LLM; no cache.hit event.
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))
    first = _chat_span(exporter)
    assert not any(e.name == "cache.hit" for e in first.events)

    # Reset exporter so we look at only the second span.
    exporter.clear()

    # Second call — HIT. memoize short-circuits; cache.hit event lands on the open chat span.
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))
    second = _chat_span(exporter)
    hit_events = [e for e in second.events if e.name == "cache.hit"]
    assert len(hit_events) == 1
    assert hit_events[0].attributes.get("cache_key") == "fixed-cache-key"
