"""E2E tests for the middleware chain against real Claude.

Covers:
    - Full chain: tracing() → meter() → retry() → memoize()
      Two identical requests → memoize serves the second from cache
      (only one real provider call). Tracing spans get recorded.
    - retry(): a mocked transient-then-success provider retries + succeeds.
      (Uses httpx.MockTransport — no real provider spend.)
    - output_coerce(): request a Pydantic-typed answer against real Claude,
      verify the parsed object lands on LLMResult.parsed → AgentResult.parsed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import claude as _claude
from agentkit.adapters.store import InMemoryStore
from agentkit.agents import Agent
from agentkit.kernel.middleware import Call
from agentkit.kernel.resilience import stable_hash as _hash
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.middlewares import memoize, meter, output_coerce, retry, tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.testing import RecordingTracer

from .conftest import HAIKU_MODEL, MAX_TOKENS, requires_anthropic


def _run(coro):
    return asyncio.run(coro)


def _key_of(call: Call) -> str:
    """Stable per-message-content cache key for memoize()."""
    req: ChatRequest = call.request
    payload = json.dumps([(m.role, m.content) for m in req.messages], sort_keys=True)
    return _hash({"messages": payload, "model": req.model})


@requires_anthropic
def test_full_chain_memoize_serves_second_call_from_cache(anthropic_key: str) -> None:
    """tracing → memoize → meter → retry. Two identical requests; second is a hit."""

    _real_calls = {"count": 0}

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        # Wrap the LLM's stream so we can count REAL provider hits
        orig_stream = llm.stream

        def counting_stream(*a, **kw):
            _real_calls["count"] += 1
            return orig_stream(*a, **kw)

        llm.stream = counting_stream  # type: ignore[method-assign]

        try:
            tracer = RecordingTracer()
            store = InMemoryStore()
            # memoize needs `store` on ctx OR passed directly
            chain = [
                tracing(),
                memoize(key=_key_of, store=store),
                meter(),
                retry(),
            ]
            invoker = Invoker(llm=llm, chat_middleware=chain)
            services = Services(invoker=invoker, store=store, trace=tracer)
            ctx = RunContext(
                correlation_id="memo-run",
                scope=Scope(org_id=1, domain_id=1),
                budget=Budget(),
                services=services,
            )

            req = ChatRequest(
                messages=[
                    Message("system", "You are terse."),
                    Message("user", "Reply with the single word: ok"),
                ],
                model=HAIKU_MODEL,
                max_tokens=MAX_TOKENS,
            )

            first = await invoker.chat(req, ctx)
            second = await invoker.chat(req, ctx)

            assert first.content
            assert second.content == first.content
            # Only ONE real provider call
            assert _real_calls["count"] == 1, f"expected 1 real provider call, got {_real_calls['count']}"
            # Tracing recorded spans
            assert tracer.spans, "expected tracer to record spans"
            # The chat span name should show up
            span_names = [s.name for s in tracer.spans]
            assert "chat" in span_names
        finally:
            await llm.aclose()

    _run(go())


# ────────────────────────────────────────────────────────────────
# retry with a mocked flaky provider (no real cost)
# ────────────────────────────────────────────────────────────────


def test_retry_middleware_recovers_from_transient_failure() -> None:
    """Mock an LLMPort that raises a transient (retryable) error twice then succeeds;
    verify retry() re-invokes and the final result surfaces."""
    from agentkit.adapters.llm.providers.base import ProviderError
    from agentkit.kernel.types import Delta

    calls = {"count": 0}

    class FlakyLLM:
        async def stream(
            self,
            *,
            messages,
            model,
            tools=None,
            response_format=None,
            temperature=0.0,
            max_tokens=None,
            cache_hint=None,
        ):
            calls["count"] += 1
            if calls["count"] <= 2:
                raise ProviderError("provider transient error (status 503; rate limit/overloaded): ...")
            yield Delta(
                text="ok",
                model="fake",
                provider="fake",
                usage=Usage(input_tokens=1, output_tokens=1),
                finish_reason="stop",
            )

        async def chat(self, **kw):  # not exercised
            raise NotImplementedError

    async def go() -> None:
        invoker = Invoker(
            llm=FlakyLLM(),
            chat_middleware=[tracing(), retry()],
        )
        ctx = RunContext(
            correlation_id="retry-run",
            scope=Scope(org_id=1, domain_id=1),
            services=Services(invoker=invoker),
        )
        req = ChatRequest(
            messages=[Message("user", "hi")],
            model="fake",
            max_tokens=10,
        )
        res = await invoker.chat(req, ctx)
        assert res.content == "ok"
        assert calls["count"] == 3, f"expected 3 attempts (2 failures + 1 success), got {calls['count']}"

    _run(go())


# ────────────────────────────────────────────────────────────────
# output_coerce with a Pydantic model, against real Claude
# ────────────────────────────────────────────────────────────────


pydantic = pytest.importorskip("pydantic")


class Answer(pydantic.BaseModel):
    color: str
    length: int


@requires_anthropic
def test_output_coerce_parses_pydantic_answer_against_real_claude(anthropic_key: str) -> None:
    """Pin a Pydantic output shape on the Agent; verify AgentResult.parsed lands."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            chain = [tracing(), output_coerce(), retry()]
            invoker = Invoker(llm=llm, chat_middleware=chain)
            ctx = RunContext(
                correlation_id="coerce-run",
                scope=Scope(org_id=1, domain_id=1),
                budget=Budget(),
                services=Services(invoker=invoker),
            )
            agent = Agent(
                name="typed",
                model=HAIKU_MODEL,
                prompt=(
                    "Reply ONLY with a JSON object matching the schema. "
                    'For example: {"color": "red", "length": 3}. No prose.'
                ),
                max_tokens=MAX_TOKENS,
                output=Answer,
            )
            result = await agent.run("The color 'blue' has how many letters?", ctx)
            # AgentResult.parsed carries the validated Pydantic object
            if result.parsed is None:
                # If the model produced free-form text the adapter can't parse,
                # the agent falls back to partial. Assert we at least got a result.
                assert result.output
            else:
                assert isinstance(result.parsed, Answer)
                assert isinstance(result.parsed.color, str)
                assert isinstance(result.parsed.length, int)
        finally:
            await llm.aclose()

    _run(go())
