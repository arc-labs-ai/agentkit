"""Phase 3 D3 — retry.attempt span event carries the truncated error message.

The retry middleware's ``retry.attempt`` event was already wired before — this test
pins the additional ``last_error_message`` field (capped at 500 chars).

A full request-diff between attempts is a future addition; today the error type +
message is the bigger win, and the test guards against silent regression of the
500-char cap.
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
from agentkit.middlewares.resilience import retry  # noqa: E402
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402
from agentkit.testing import FakeLLM  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_retry_attempt_event_carries_error_message():
    """retry.attempt event now has last_error_message field, capped at 500 chars."""
    long_err_message = "x" * 800  # over the cap

    # FakeLLM raises a single transient with a long message → retry fires once,
    # then succeeds on attempt 2. The retry.attempt event lands on the chat span
    # carrying both ``last_error_type`` (TimeoutError) and ``last_error_message``.
    llm = FakeLLM(
        "ok",
        usage=Usage(10, 5, 0.0),
        fail_times=1,
        fail_exc=TimeoutError(long_err_message),
    )
    provider, exporter = _make_in_memory_provider()

    # Sleep is mocked to a no-op so the test stays fast.
    async def _no_sleep(_seconds: float) -> None:  # noqa: ARG001
        return

    inv = Invoker(
        llm=llm,
        chat_middleware=[tracing(), retry(max_attempts=3, sleep=_no_sleep)],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))

    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))

    chat_span = next(s for s in exporter.get_finished_spans() if s.name == "chat")
    retry_events = [e for e in chat_span.events if e.name == "retry.attempt"]
    assert len(retry_events) == 1
    ev = retry_events[0]
    # Existing fields still present (no removed attributes — additive only).
    assert ev.attributes["last_error_type"] == "TimeoutError"
    assert ev.attributes["attempt"] == 2
    # New field: last_error_message, capped at 500 chars.
    msg_attr = ev.attributes["last_error_message"]
    assert msg_attr.startswith("xxxx")  # leading content preserved
    assert len(msg_attr) == 500  # cap enforced
