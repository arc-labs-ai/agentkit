"""Phase 3 D2 — execute_tool span carries a gen_ai.tool.result event.

After the tool runs, the tracing middleware drops a single ``gen_ai.tool.result``
event on the open execute_tool span with the tool name, the (truncated) result
content, the un-capped raw byte size, and the wall-clock duration.

PII / redaction is operator-side (collector regex rules) — what the tool returns
is what lands on the event (subject to the 4 KB content cap).
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
from agentkit.kernel.types import Scope, ToolRequest  # noqa: E402
from agentkit.middlewares import tracing  # noqa: E402
from agentkit.middlewares.tracing import (  # noqa: E402
    MAX_TOOL_RESULT_CONTENT_BYTES,
    _serialize_tool_result,
)
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402
from agentkit.testing import FakeLLM  # noqa: E402
from agentkit.tools import FunctionTool  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _execute_tool_span(exporter: InMemorySpanExporter):
    return next(s for s in exporter.get_finished_spans() if s.name == "execute_tool")


def _events_named(span, name: str) -> list:
    return [e for e in span.events if e.name == name]


def _ctx(provider, *, tool_payload):
    """Build a RunContext with a single FunctionTool wired through tracing()."""
    inv = Invoker(
        llm=FakeLLM("unused"),
        chat_middleware=[tracing()],
        tool_middleware=[tracing()],
    )
    return inv, RunContext(
        "r", Scope(1, 2), Budget(), Services(invoker=inv, trace=otel_tracer(provider))
    )


def test_execute_tool_span_emits_result_event():
    """gen_ai.tool.result event with name + content + size_bytes + duration_ms."""
    provider, exporter = _make_in_memory_provider()
    inv, ctx = _ctx(provider, tool_payload={"s": 200})
    tool = FunctionTool(
        "fetch",
        lambda a, c: {"status": 200, "body": "ok"},
        description="fetch a resource",
        side_effecting=False,
    )
    req = ToolRequest("fetch", {"url": "x"}, tool, side_effecting=False)
    _run(inv.invoke_tool(req, ctx))

    span = _execute_tool_span(exporter)
    events = _events_named(span, "gen_ai.tool.result")
    assert len(events) == 1
    ev = events[0]
    assert ev.attributes["gen_ai.tool.name"] == "fetch"
    # JSON-encoded dict result lands as content (not str(dict)).
    content = ev.attributes["gen_ai.tool.result_content"]
    assert '"status"' in content and '"body"' in content
    # Size + duration are present and sensible.
    assert ev.attributes["gen_ai.tool.result_size_bytes"] > 0
    assert ev.attributes["gen_ai.tool.duration_ms"] >= 0


def test_tool_result_content_truncates_at_4kb():
    """Large dict tool returns → result_content capped at 4 KB, size_bytes UNCAPPED."""
    big_value = "x" * (MAX_TOOL_RESULT_CONTENT_BYTES + 2048)  # 6 KB
    provider, exporter = _make_in_memory_provider()
    inv, ctx = _ctx(provider, tool_payload={"big": big_value})
    tool = FunctionTool(
        "scrape",
        lambda a, c: {"body": big_value},
        description="scrape a page",
        side_effecting=False,
    )
    req = ToolRequest("scrape", {}, tool, side_effecting=False)
    _run(inv.invoke_tool(req, ctx))

    span = _execute_tool_span(exporter)
    ev = _events_named(span, "gen_ai.tool.result")[0]
    content = ev.attributes["gen_ai.tool.result_content"]
    # Truncated content must fit within the cap (UTF-8 bytes).
    assert len(content.encode("utf-8")) <= MAX_TOOL_RESULT_CONTENT_BYTES
    assert content.endswith(" [truncated]")
    # Raw size reflects the actual JSON length — NOT capped.
    raw_size = ev.attributes["gen_ai.tool.result_size_bytes"]
    assert raw_size > MAX_TOOL_RESULT_CONTENT_BYTES


def test_tool_result_serializes_via_json_when_possible():
    """dict result → JSON-string content; opaque (non-JSON) result → str() form."""

    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque object>"

    # Pure-Python helper coverage: dict → JSON.
    json_form = _serialize_tool_result({"a": 1, "b": [2, 3]})
    assert "{" in json_form and '"a"' in json_form

    # Opaque object — falls back to str()/repr().
    str_form = _serialize_tool_result(Opaque())
    assert str_form == "<Opaque object>"

    # End-to-end on the span: a string result lands verbatim (no JSON wrapping).
    provider, exporter = _make_in_memory_provider()
    inv, ctx = _ctx(provider, tool_payload="raw-string-output")
    tool = FunctionTool(
        "echo",
        lambda a, c: "raw-string-output",
        description="echo",
        side_effecting=False,
    )
    req = ToolRequest("echo", {}, tool, side_effecting=False)
    _run(inv.invoke_tool(req, ctx))

    span = _execute_tool_span(exporter)
    ev = _events_named(span, "gen_ai.tool.result")[0]
    assert ev.attributes["gen_ai.tool.result_content"] == "raw-string-output"
