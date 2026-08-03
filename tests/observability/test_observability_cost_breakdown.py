"""Phase 3 D3 — cost-breakdown signals on the chat span.

Two facets:

- ``gen_ai.usage.cost_saved_by_cache_usd`` — computed from ``cache_read_tokens`` +
  the cached-input price ratio (10% of full input for both OpenAI + Anthropic).
- ``budget.checkpoint`` event — fired by the meter middleware after each charge,
  carrying ``spent_usd`` / ``remaining_usd`` / ``calls`` so operators can see the
  call that crossed a budget threshold without correlating against an external metric.
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
from agentkit.middlewares import meter, tracing  # noqa: E402
from agentkit.middlewares.tracing import CACHED_INPUT_PRICE_RATIO  # noqa: E402
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


def test_cost_saved_by_cache_usd_lands_on_chat_span():
    """cache_read_tokens > 0 → gen_ai.usage.cost_saved_by_cache_usd computed + attached."""
    provider, exporter = _make_in_memory_provider()
    usage = Usage(
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.002,
        cache_read_tokens=80,
        cache_write_tokens=0,
    )
    inv = Invoker(
        llm=FakeLLM("ok", usage=usage),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    assert "gen_ai.usage.cost_saved_by_cache_usd" in chat.attributes
    # Expected: cache_read_tokens * per_token_cost * (1 - cached_ratio).
    per_token = 0.002 / 100
    expected = 80 * per_token * (1 - CACHED_INPUT_PRICE_RATIO)
    actual = chat.attributes["gen_ai.usage.cost_saved_by_cache_usd"]
    assert abs(actual - expected) < 1e-9


def test_cost_saved_by_cache_usd_absent_when_no_cache_reads():
    """No cache read tokens → savings attribute absent (no noisy zero)."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM("ok", usage=Usage(100, 20, 0.002)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    assert "gen_ai.usage.cost_saved_by_cache_usd" not in chat.attributes


def test_budget_remaining_stamped_when_budget_configured():
    """Budget set + a chat call → budget.checkpoint event lands on the chat span
    with spent_usd, remaining_usd, and calls."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM("ok", usage=Usage(100, 20, 0.5)),
        chat_middleware=[tracing(), meter()],
    )
    ctx = RunContext(
        "r",
        Scope(1, 2),
        Budget(max_cost_usd=2.0),
        Services(invoker=inv, trace=otel_tracer(provider)),
    )
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    events = [e for e in chat.events if e.name == "budget.checkpoint"]
    assert len(events) == 1
    ev = events[0]
    assert ev.attributes["spent_usd"] == 0.5
    assert ev.attributes["remaining_usd"] == 1.5  # 2.0 cap minus 0.5 charged
    assert ev.attributes["calls"] == 1


def test_budget_checkpoint_event_absent_when_no_ceiling():
    """No max_cost_usd configured → budget.checkpoint event MUST NOT fire (no signal)."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM("ok", usage=Usage(100, 20, 0.5)),
        chat_middleware=[tracing(), meter()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat = _chat_span(exporter)
    events = [e for e in chat.events if e.name == "budget.checkpoint"]
    assert events == []
