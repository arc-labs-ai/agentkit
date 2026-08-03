"""Tracing middleware defensive-span coverage.

The tracer is observability only — every seam the middleware touches
(metrics, sampler, replay, per-message events) already wraps calls in
``contextlib.suppress`` so a misbehaving implementation cannot break
the run. Until now the ``TracePort.span`` context manager itself was
the sole exception: an exception raised by ``span(...)`` propagated
straight into the chat chain and killed the LLM call.

This test pins the fix: a ``TracePort`` whose ``span(...)`` raises
does NOT stop the middleware chain — the underlying handler still
runs and the LLM result flows through unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.middlewares import tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


class _BrokenTrace:
    """TracePort whose ``span`` raises on every call. Structurally
    satisfies the port so ``Services.trace`` accepts it; every other
    seam (``current_span_id``, ``add_event_to_current_span``) returns
    inert values so the middleware's defensive wraps don't fire."""

    def __init__(self) -> None:
        self.span_calls: int = 0

    def span(self, name: str, kind: str, **attrs: Any) -> Any:
        self.span_calls += 1
        raise RuntimeError(f"boom on span open: {name!r}")

    def current_span_id(self) -> None:
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        return None


def test_tracing_middleware_survives_broken_span():
    """A ``TracePort.span`` that raises must NOT break the chat chain.
    The middleware falls back to a no-op span; the LLM handler runs;
    the assembled result flows through unchanged."""

    async def go() -> None:
        broken = _BrokenTrace()
        inv = Invoker(
            llm=FakeLLM("hello", usage=Usage(input_tokens=3, output_tokens=1, cost_usd=0.0)),
            chat_middleware=[tracing()],
        )
        ctx = RunContext(
            "r-broken-span",
            Scope(),
            Budget(),
            Services(invoker=inv, trace=broken),
        )
        result = await inv.chat(ChatRequest([Message("user", "q")], "m"), ctx)

        # 1. The chain completed — we got a real assembled result.
        assert result.content == "hello"
        assert result.usage.input_tokens == 3
        # 2. The tracer WAS consulted (proves the middleware still tried to open a span).
        assert broken.span_calls == 1

    _run(go())
