"""agentkit.client — the batteries-included Chat builder: a provider client wrapped in the standard
tracing→meter→retry chain. Offline via an httpx.MockTransport-backed provider + FakeLLM."""

import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import OpenAICompatibleLLM
from agentkit.client import Chat, claude, deepseek, openai, openrouter
from agentkit.testing import FakeLLM


def _run(coro):
    return asyncio.run(coro)


_OPENAI_SSE = (
    'data: {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "po"}, "finish_reason": null}]}\n\n'
    'data: {"choices": [{"delta": {"content": "ng"}, "finish_reason": "stop"}]}\n\n'
    'data: {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}\n\n'
    "data: [DONE]\n\n"
)


def _sse_client(body: str):
    def handler(request):
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_chat_runs_a_provider_through_the_resilience_chain():
    # Chat collects the streaming primitive under the hood, so the provider speaks SSE
    llm = OpenAICompatibleLLM(api_key="k", base_url="http://x", client=_sse_client(_OPENAI_SSE))
    chat = Chat(llm, model="gpt-4o-mini")
    res = _run(chat("ping", system="be terse"))
    assert res.content == "pong"  # reassembled from the "po" + "ng" deltas
    assert chat._budget.calls == 1  # the `meter` middleware charged the call → chain ran
    assert res.usage.cost_usd > 0  # cache-aware cost applied (gpt-4o-mini in the table)


def test_chat_works_with_any_llmport_fake():
    chat = Chat(FakeLLM("hi"), model="m", breaker=False)
    assert _run(chat("q")).content == "hi"


def test_presets_build_chat_over_the_right_provider():
    # constructed without a network call (real api_key path); assert the wrapped provider/base_url
    assert isinstance(claude(api_key="k"), Chat)
    for preset, host in (
        (openai, "api.openai.com"),
        (deepseek, "api.deepseek.com"),
        (openrouter, "openrouter.ai"),
    ):
        c = preset(api_key="k")
        assert isinstance(c, Chat) and host in c._invoker._llm._base_url


def test_chat_aclose_closes_owned_provider_client():
    """Regression: the batteries-included Chat must close the provider's httpx client (no socket leak).
    As an async context manager on exit, and via aclose()."""
    closed = {"n": 0}

    class _Provider:
        async def chat(self, **kw):
            from agentkit.kernel.types import LLMResult

            return LLMResult(content="ok", model=kw.get("model"), provider="rec")

        async def stream(self, **kw):
            from agentkit.adapters.llm._mapping import result_to_delta

            yield result_to_delta(await self.chat(**kw))

        async def aclose(self):
            closed["n"] += 1

    async def go():
        prov = _Provider()
        async with Chat(prov, model="m", breaker=False) as chat:
            r = await chat("hi")
        return r

    res = _run(go())
    assert res.content == "ok" and closed["n"] == 1  # closed exactly once on context exit


def test_chat_aclose_is_safe_when_provider_has_no_close():
    # an injected FakeLLM has no aclose → aclose is a no-op, never raises
    chat = Chat(FakeLLM("hi"), model="m", breaker=False)
    _run(chat.aclose())
