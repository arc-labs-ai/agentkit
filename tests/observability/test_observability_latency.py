"""First-token + total-duration attributes on chat spans.

The ``tracing()`` middleware time-stamps two latencies on every chat span:

- ``gen_ai.response.first_token_latency_ms`` — perf-counter delta from the
  span open to the FIRST non-empty content delta. Genuinely undefined for
  empty streams (tool-only turns), so it's only set when content arrives.
- ``gen_ai.response.duration_ms`` — total turn duration; stamped at stream
  end regardless of whether content was produced.

Skipped cleanly when ``opentelemetry-sdk`` (the ``[observability]`` extra)
isn't installed.
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
from agentkit.kernel.types import ChatRequest, Delta, Message, Scope, Usage  # noqa: E402
from agentkit.middlewares import tracing  # noqa: E402
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    """A fresh, LOCAL ``TracerProvider`` per test (the global one cannot be re-set)."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class _SlowFirstTokenLLM:
    """Yield one content delta after a small sleep, then a terminal delta with usage."""

    def __init__(self, *, pre_token_sleep: float, usage: Usage):
        self._pre_token_sleep = pre_token_sleep
        self._usage = usage

    async def stream(self, *, messages, model, **_kw):
        await asyncio.sleep(self._pre_token_sleep)
        yield Delta(text="ok", model=model, provider="fake")
        yield Delta(usage=self._usage, finish_reason="stop", model=model, provider="fake")


class _EmptyStreamLLM:
    """Emit ONLY a terminal delta (no content) — first-token latency is undefined."""

    def __init__(self, *, usage: Usage):
        self._usage = usage

    async def stream(self, *, messages, model, **_kw):
        yield Delta(usage=self._usage, finish_reason="stop", model=model, provider="fake")


def _chat_span(exporter: InMemorySpanExporter):
    return next(s for s in exporter.get_finished_spans() if s.name == "chat")


def test_first_token_latency_ms_lands_on_chat_span():
    """The attribute is set and >= the induced pre-token sleep."""
    provider, exporter = _make_in_memory_provider()
    sleep_s = 0.02  # 20 ms — well above scheduler jitter
    inv = Invoker(
        llm=_SlowFirstTokenLLM(pre_token_sleep=sleep_s, usage=Usage(100, 20, 0.002)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    ftl = chat.attributes.get("gen_ai.response.first_token_latency_ms")
    assert ftl is not None
    assert ftl >= sleep_s * 1000


def test_total_duration_ms_lands_on_chat_span():
    """``duration_ms`` is set and is no smaller than ``first_token_latency_ms``."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=_SlowFirstTokenLLM(pre_token_sleep=0.01, usage=Usage(100, 20, 0.002)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    dur = chat.attributes.get("gen_ai.response.duration_ms")
    ftl = chat.attributes.get("gen_ai.response.first_token_latency_ms")
    assert dur is not None
    assert ftl is not None
    assert dur >= ftl


def test_first_token_latency_absent_on_empty_stream():
    """No content delta => attribute MUST be absent (not 0) so backends show 'no value'."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=_EmptyStreamLLM(usage=Usage(100, 0, 0.0)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    assert "gen_ai.response.first_token_latency_ms" not in chat.attributes
    # duration is still defined for an empty stream (the span was alive) — stamp it.
    assert "gen_ai.response.duration_ms" in chat.attributes
