"""Phase 3 D3 — cost-breakdown signals on the chat span.

Two facets:

- Cache token counts and measured cost provenance; a total cost cannot reveal
  provider-specific cache savings, so those dollars are never invented.
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

from agentkit.core.contracts.types import ChatRequest, Message, Scope, Usage
from agentkit.core.policies import meter, tracing
from agentkit.core.runtime import Budget, Invoker, RunContext, Services
from agentkit.core.testing import FakeLLM
from agentkit.integrations.observability import otel_tracer


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _chat_span(exporter: InMemorySpanExporter):
    return next(s for s in exporter.get_finished_spans() if s.name == "chat")


def test_cache_counts_do_not_invent_a_model_independent_dollar_saving():
    """The total includes output; it cannot reveal a provider's fresh/cached input rates."""
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
    assert "gen_ai.usage.cost_saved_by_cache_usd" not in chat.attributes
    assert chat.attributes["gen_ai.usage.cache_read_tokens"] == 80
    assert chat.attributes["gen_ai.usage.cost_usd"] == 0.002
    assert chat.attributes["gen_ai.usage.cost_source"] == "reported"


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


def test_unknown_cost_trace_exposes_provenance_and_omits_a_fake_dollar_measurement():
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM("unpriced", usage=Usage(input_tokens=10, output_tokens=2, cost_usd=None)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("unknown-cost", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    result = _run(inv.chat(ChatRequest([Message("user", "q")], "unknown-model"), ctx))
    assert result.usage.cost_usd is None
    span = _chat_span(exporter)
    assert span.attributes["gen_ai.usage.cost_source"] == "unpriced"
    assert "gen_ai.usage.cost_usd" not in span.attributes
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
