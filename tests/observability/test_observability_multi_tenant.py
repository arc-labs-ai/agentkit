"""Phase 2B Gap 3 — ``agentkit.scope.*`` attributes stamp from ``RunContext.scope``.

Tenant-isolation key (``Scope.org_id`` + ``Scope.domain_id``) lands on the ``invoke_agent``
span so an operator can filter "show me org 42's traces". Stamped ONLY when set — the
dev-default zero-tenant case (both ``None``) leaves the keys absent so backends see
"no tenant" as missing, not as a bogus 0.
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
from agentkit.runtime import Budget, Invoker, RunContext, Services  # noqa: E402
from agentkit.testing import FakeLLM  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _make_in_memory_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _ctx(*, llm, provider, scope: Scope) -> RunContext:
    inv = Invoker(llm=llm, chat_middleware=[tracing()])
    return RunContext(
        correlation_id="r",
        scope=scope,
        budget=Budget(),
        services=Services(invoker=inv, trace=otel_tracer(provider)),
    )


def _invoke_agent_span(exporter: InMemorySpanExporter):
    spans = exporter.get_finished_spans()
    return next(s for s in spans if s.name == "invoke_agent")


def test_scope_org_id_lands_on_invoke_agent_span():
    """Scope(org_id=42, domain_id=7) → both keys land on the span."""
    provider, exporter = _make_in_memory_provider()
    ctx = _ctx(llm=FakeLLM("ok"), provider=provider, scope=Scope(org_id=42, domain_id=7))
    agent = Agent(name="leaf", model="m", prompt="p")

    _run(agent.run("task", ctx))

    span = _invoke_agent_span(exporter)
    assert span.attributes["agentkit.scope.org_id"] == 42
    assert span.attributes["agentkit.scope.domain_id"] == 7


def test_scope_attrs_absent_when_none():
    """Default Scope() (both None) → neither key is present on the span."""
    provider, exporter = _make_in_memory_provider()
    ctx = _ctx(llm=FakeLLM("ok"), provider=provider, scope=Scope())
    agent = Agent(name="leaf", model="m", prompt="p")

    _run(agent.run("task", ctx))

    span = _invoke_agent_span(exporter)
    assert "agentkit.scope.org_id" not in span.attributes
    assert "agentkit.scope.domain_id" not in span.attributes


def test_scope_attrs_stamp_only_set_half():
    """Scope(org_id=42, domain_id=None) → only org_id lands; domain_id stays absent."""
    provider, exporter = _make_in_memory_provider()
    ctx = _ctx(llm=FakeLLM("ok"), provider=provider, scope=Scope(org_id=42, domain_id=None))
    agent = Agent(name="leaf", model="m", prompt="p")

    _run(agent.run("task", ctx))

    span = _invoke_agent_span(exporter)
    assert span.attributes["agentkit.scope.org_id"] == 42
    assert "agentkit.scope.domain_id" not in span.attributes
