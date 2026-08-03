"""Phase 3 D1 — chat-span carries OTel GenAI v1.41 per-message events.

Each request message lands as a ``gen_ai.{role}.message`` event on the chat span
with the role, content (length-capped), and a chars/4 token estimate. The
assembled assistant reply lands as a single ``gen_ai.choice`` event.

PII / redaction is operator-side (collector regex rules) by stance — these tests
just check the events fire with the right shape; what content lands is what came in.
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
from agentkit.middlewares.tracing import (  # noqa: E402
    MAX_MESSAGE_CONTENT_BYTES,
    _truncate_content,
)
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


def _events_named(span, name: str) -> list:
    return [e for e in span.events if e.name == name]


def test_chat_span_emits_message_events_per_role():
    """gen_ai.{system|user|assistant}.message events land on chat span — one per role
    in the request — plus exactly one gen_ai.choice for the assistant reply."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM("answer", usage=Usage(10, 5, 0.0001)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))

    req = ChatRequest(
        messages=[
            Message("system", "you are helpful"),
            Message("user", "first question"),
            Message("user", "second question"),
        ],
        model="m",
    )
    _run(inv.chat(req, ctx))

    span = _chat_span(exporter)
    assert len(_events_named(span, "gen_ai.system.message")) == 1
    assert len(_events_named(span, "gen_ai.user.message")) == 2
    assert len(_events_named(span, "gen_ai.choice")) == 1

    choice = _events_named(span, "gen_ai.choice")[0]
    assert choice.attributes["gen_ai.choice.role"] == "assistant"
    assert choice.attributes["gen_ai.choice.content"] == "answer"
    assert choice.attributes["gen_ai.choice.tokens"] == 5  # from FakeLLM usage


def test_per_message_tokens_attribute_is_present():
    """gen_ai.message.tokens populated; sum approximates input_tokens scale."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=FakeLLM("ok", usage=Usage(20, 5, 0.0)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))

    req = ChatRequest(
        messages=[Message("user", "x" * 40)],  # ~10 tokens at chars/4
        model="m",
    )
    _run(inv.chat(req, ctx))

    span = _chat_span(exporter)
    user_event = _events_named(span, "gen_ai.user.message")[0]
    # chars/4 heuristic: 40 chars / 4 = 10
    assert user_event.attributes["gen_ai.message.tokens"] == 10
    assert user_event.attributes["gen_ai.message.role"] == "user"


def test_long_message_content_is_truncated():
    """8KB+ message → event content truncated with [truncated] suffix."""
    big = "x" * (MAX_MESSAGE_CONTENT_BYTES + 4096)  # 12 KB
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(llm=FakeLLM("ok"), chat_middleware=[tracing()])
    ctx = RunContext("r", Scope(), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))

    req = ChatRequest(messages=[Message("user", big)], model="m")
    _run(inv.chat(req, ctx))

    span = _chat_span(exporter)
    ev = _events_named(span, "gen_ai.user.message")[0]
    content = ev.attributes["gen_ai.message.content"]
    assert content.endswith(" [truncated]")
    # Truncated content must fit within the cap (UTF-8 bytes).
    assert len(content.encode("utf-8")) <= MAX_MESSAGE_CONTENT_BYTES


def test_truncation_preserves_leading_content():
    """The truncated content starts with the original's leading chars."""
    head = "HELLO_AGENT_KIT"
    body = head + "x" * (MAX_MESSAGE_CONTENT_BYTES + 100)
    truncated = _truncate_content(body)
    assert truncated.startswith(head)
    assert truncated.endswith(" [truncated]")


def test_no_message_events_on_empty_request():
    """0-message request → no per-message events fire (only gen_ai.choice for the
    assistant reply, since the LLM still produces an output)."""
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(llm=FakeLLM("ok"), chat_middleware=[tracing()])
    ctx = RunContext("r", Scope(), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))

    req = ChatRequest(messages=[], model="m")
    _run(inv.chat(req, ctx))

    span = _chat_span(exporter)
    # No per-message events for the (empty) request.
    assert _events_named(span, "gen_ai.user.message") == []
    assert _events_named(span, "gen_ai.system.message") == []
    assert _events_named(span, "gen_ai.assistant.message") == []
