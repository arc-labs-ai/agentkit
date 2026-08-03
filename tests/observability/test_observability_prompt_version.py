"""Phase 2B Gap 8 — ``gen_ai.system.prompt_version`` lands on the ``invoke_agent`` span.

``AgentResult.prompt_version`` (stamped by the ``RequestBuilder``) is mirrored onto the
``invoke_agent`` span so a trace shows which template produced the output without
inspecting the result object. The ``gen_ai.system.*`` prefix is OTel's convention for
system-level facts (versus per-request facts under ``gen_ai.request.*``).
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
from agentkit.agents import Agent  # noqa: E402
from agentkit.kernel.types import Scope  # noqa: E402
from agentkit.middlewares import tracing  # noqa: E402
from agentkit.prompts.prompt import Prompt  # noqa: E402
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402
from agentkit.testing import FakeLLM  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
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


def test_prompt_version_stamped_on_invoke_agent_span():
    """A Prompt with version "2026-06-22" → that string lands on the span."""
    provider, exporter = _make_in_memory_provider()
    ctx = _ctx(llm=FakeLLM("ok"), provider=provider)
    prompt = Prompt(id="leaf.v1", version="2026-06-22", template="you are leaf")
    agent = Agent(name="leaf", model="m", prompt=prompt)

    res = _run(agent.run("task", ctx))
    assert res.prompt_version == "2026-06-22"

    span = _invoke_agent_span(exporter)
    assert span.attributes["gen_ai.system.prompt_version"] == "2026-06-22"


def test_prompt_version_absent_when_empty():
    """An ``AgentResult.prompt_version`` of "" → the attribute is NOT present on the span.

    A leaf Agent with no ``prompt=`` falls through to an inline Prompt with version
    ``"inline"`` — so to assert the absent case we use a Prompt with an empty version
    string (the documented "no prompt wired" value on ``AgentResult.prompt_version``).
    Empty string is falsy so the stamp guard skips it."""
    provider, exporter = _make_in_memory_provider()
    ctx = _ctx(llm=FakeLLM("ok"), provider=provider)
    prompt = Prompt(id="leaf.empty", version="", template="t")
    agent = Agent(name="leaf", model="m", prompt=prompt)

    _run(agent.run("task", ctx))

    span = _invoke_agent_span(exporter)
    assert "gen_ai.system.prompt_version" not in span.attributes
