"""Phase 2B Gap 2 — ``invoke_agent`` span carries the per-agent subtree usage rollup.

The agent's ``invoke_agent`` span aggregates the run's cumulative ``Usage`` (the same
value that lands on ``AgentResult.usage``) into ``gen_ai.usage.*_subtree`` attributes.
This lets an operator answer "what did this agent + descendants cost?" without summing
the nested ``chat`` spans by hand.

Streaming agents must NOT close the ``invoke_agent`` span before the terminal delta —
otherwise the rollup loses any after-final-yield usage.
"""

import asyncio
import math

import pytest

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from agentkit.adapters.observability import otel_tracer  # noqa: E402
from agentkit.agents import Agent  # noqa: E402
from agentkit.kernel.types import Scope, Usage  # noqa: E402
from agentkit.middlewares import tracing  # noqa: E402
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402
from agentkit.testing import FakeLLM  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Fresh ``TracerProvider`` + ``InMemorySpanExporter`` per test — LOCAL only.

    OTel forbids overriding the global TracerProvider once set, so each test wires
    its own provider through ``otel_tracer(tracer_provider=…)`` for isolation."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _ctx(*, llm, provider) -> RunContext:
    inv = Invoker(llm=llm, chat_middleware=[tracing()])
    return RunContext(
        correlation_id="r",
        scope=Scope(),
        budget=Budget(),
        services=Services(invoker=inv, trace=otel_tracer(provider)),
    )


def _invoke_agent_span(exporter: InMemorySpanExporter):
    spans = exporter.get_finished_spans()
    return next(s for s in spans if s.name == "invoke_agent")


def test_invoke_agent_span_stamps_subtree_usage_attributes():
    """``AgentResult.usage`` lands on the span as ``gen_ai.usage.*_subtree``."""
    provider, exporter = _make_in_memory_provider()
    llm = FakeLLM("ok", usage=Usage(100, 20, 0.002))
    ctx = _ctx(llm=llm, provider=provider)
    agent = Agent(name="leaf", model="m", prompt="p")

    res = _run(agent.run("task", ctx))
    assert res.output == "ok"

    span = _invoke_agent_span(exporter)
    assert span.attributes["gen_ai.usage.input_tokens_subtree"] == 100
    assert span.attributes["gen_ai.usage.output_tokens_subtree"] == 20
    assert math.isclose(
        span.attributes["gen_ai.usage.cost_usd_subtree"], 0.002, rel_tol=0, abs_tol=1e-9
    )


def test_subtree_cache_attrs_absent_when_zero():
    """Default ``Usage`` (no cache hits/writes) — cache subtree attrs MUST be absent."""
    provider, exporter = _make_in_memory_provider()
    llm = FakeLLM("ok", usage=Usage(10, 5, 0.0))  # no cache fields
    ctx = _ctx(llm=llm, provider=provider)
    agent = Agent(name="leaf", model="m", prompt="p")

    _run(agent.run("task", ctx))

    span = _invoke_agent_span(exporter)
    assert "gen_ai.usage.cache_read_tokens_subtree" not in span.attributes
    assert "gen_ai.usage.cache_write_tokens_subtree" not in span.attributes


def test_subtree_cache_attrs_stamped_when_nonzero():
    """When usage carries cache fields, they land on the span."""
    provider, exporter = _make_in_memory_provider()
    llm = FakeLLM("ok", usage=Usage(100, 20, 0.002, cache_read_tokens=80, cache_write_tokens=15))
    ctx = _ctx(llm=llm, provider=provider)
    agent = Agent(name="leaf", model="m", prompt="p")

    _run(agent.run("task", ctx))

    span = _invoke_agent_span(exporter)
    assert span.attributes["gen_ai.usage.cache_read_tokens_subtree"] == 80
    assert span.attributes["gen_ai.usage.cache_write_tokens_subtree"] == 15


def test_streaming_agent_subtree_usage_stamped_at_terminal_delta():
    """``Agent.stream`` MUST keep the ``invoke_agent`` span open until the final event
    yields — and the subtree usage MUST be stamped before the span closes.

    We assert the order: the consumer sees the ``final`` event WHILE the span is still
    open (it has not landed in the exporter yet), then after the iterator exhausts the
    span lands with the rollup attributes set."""
    provider, exporter = _make_in_memory_provider()
    llm = FakeLLM("ok", usage=Usage(50, 10, 0.001))
    ctx = _ctx(llm=llm, provider=provider)
    agent = Agent(name="leaf", model="m", prompt="p")

    async def go():
        saw_final = False
        spans_at_final = None
        async for ev in agent.stream("task", ctx):
            if ev.type == "final":
                saw_final = True
                # While we're inside the consumer loop, the span is still
                # open — the InMemoryExporter has NOT received it yet.
                spans_at_final = [s.name for s in exporter.get_finished_spans()]
        return saw_final, spans_at_final

    saw_final, spans_at_final = _run(go())
    assert saw_final, "consumer never saw a final event"
    assert "invoke_agent" not in (spans_at_final or []), (
        "invoke_agent span closed BEFORE the terminal yield — subtree usage would be lost"
    )

    span = _invoke_agent_span(exporter)
    assert span.attributes["gen_ai.usage.input_tokens_subtree"] == 50
    assert span.attributes["gen_ai.usage.output_tokens_subtree"] == 10
