"""``gen_ai.response.tool_calls`` is capped at ``MAX_TOOL_NAMES_IN_ATTR``.

A tool-heavy turn (e.g. a 50-tool re-entrant loop) would otherwise pin every
tool name into a single span attribute, blowing the OTel SDK's per-attribute
budget. The middleware truncates the joined names list and stamps the raw
count separately on ``gen_ai.response.tool_calls_count`` so backends that care
about cardinality (e.g. histograms) can read it without parsing the names.
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
from agentkit.kernel.types import (  # noqa: E402
    ChatRequest,
    Delta,
    Message,
    Scope,
    ToolCall,
    Usage,
)
from agentkit.middlewares import tracing  # noqa: E402
from agentkit.middlewares.tracing import MAX_TOOL_NAMES_IN_ATTR  # noqa: E402
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class _ToolCallsLLM:
    """Emit a terminal delta carrying ``n`` tool calls — no content."""

    def __init__(self, n: int, *, usage: Usage):
        self._n = n
        self._usage = usage

    async def stream(self, *, messages, model, **_kw):
        tool_calls = tuple(
            ToolCall(id=f"t{i}", name=f"tool_{i}", arguments={}) for i in range(self._n)
        )
        yield Delta(
            tool_calls=tool_calls,
            usage=self._usage,
            finish_reason="tool_calls",
            model=model,
            provider="fake",
        )


def _chat_span(exporter: InMemorySpanExporter):
    return next(s for s in exporter.get_finished_spans() if s.name == "chat")


def _run_chat(n: int) -> InMemorySpanExporter:
    provider, exporter = _make_in_memory_provider()
    inv = Invoker(
        llm=_ToolCallsLLM(n, usage=Usage(10, 5, 0.0001)),
        chat_middleware=[tracing()],
    )
    ctx = RunContext("r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider)))
    _run(inv.chat(ChatRequest([Message("user", "q")], "m"), ctx))
    return exporter


def test_tool_calls_attribute_under_cap_is_full_list():
    """5 tools (well under the cap) => attribute is the 5 names comma-joined, no overflow suffix."""
    exporter = _run_chat(5)
    chat = _chat_span(exporter)
    expected = ",".join(f"tool_{i}" for i in range(5))
    assert chat.attributes["gen_ai.response.tool_calls"] == expected
    assert "+" not in chat.attributes["gen_ai.response.tool_calls"]


def test_tool_calls_attribute_at_cap_is_capped_with_overflow():
    """MAX + 9 tools => first MAX names + ``,+9 more`` suffix."""
    overflow = 9
    n = MAX_TOOL_NAMES_IN_ATTR + overflow
    exporter = _run_chat(n)
    chat = _chat_span(exporter)
    head = ",".join(f"tool_{i}" for i in range(MAX_TOOL_NAMES_IN_ATTR))
    assert chat.attributes["gen_ai.response.tool_calls"] == f"{head},+{overflow} more"


def test_tool_calls_count_is_separate_int_attr():
    """The raw count is on its own attribute regardless of the names cap."""
    n = MAX_TOOL_NAMES_IN_ATTR + 9
    exporter = _run_chat(n)
    chat = _chat_span(exporter)
    assert chat.attributes["gen_ai.response.tool_calls_count"] == n
