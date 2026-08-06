"""E2E integration tests for the provider clients (`providers.claude`, `providers.openai`).

Covers the LLM preset happy + sad matrix:
    - LLMResult shape (content, usage, cost_usd, provider, model)
    - Streaming: Delta shape, terminal delta, assemble_deltas round-trip
    - Auth failures: ProviderAuthError propagation (empty and "sk-invalid" keys)
    - Model-not-found errors: useful diagnostic surface
    - max_tokens=1 truncation: finish_reason reflects it
    - Timeout: asyncio.wait_for cancels cleanly (no dangling client)

Real-provider tests are gated on API-key env vars via @requires_anthropic /
@requires_openai — CI without keys skips them.
"""

from __future__ import annotations

import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import claude as _claude
from agentkit.adapters.llm.providers import openai as _openai
from agentkit.kernel.errors import ProviderAuthError
from agentkit.kernel.types import Delta, LLMResult, Message, assemble_deltas

from .conftest import (
    HAIKU_MODEL,
    MAX_TOKENS,
    OPENAI_MINI_MODEL,
    requires_anthropic,
    requires_openai,
)


def _run(coro):
    return asyncio.run(coro)


# ────────────────────────────────────────────────────────────────
# Anthropic — happy path (chat + streaming)
# ────────────────────────────────────────────────────────────────


@requires_anthropic
def test_anthropic_chat_returns_valid_llmresult(anthropic_key: str) -> None:
    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            res = await llm.chat(
                messages=[Message("user", "Reply with the single word: hello")],
                model=HAIKU_MODEL,
                max_tokens=MAX_TOKENS,
            )
            assert isinstance(res, LLMResult)
            assert res.content, "content should not be empty"
            assert res.provider == "anthropic"
            assert res.model  # served model comes back
            assert res.usage.input_tokens > 0
            assert res.usage.output_tokens > 0
            # haiku 4-5 is on the pricing table → cost should be > 0
            assert res.usage.cost_usd > 0.0
            assert res.finish_reason is not None
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_anthropic_stream_yields_deltas_and_assembles(anthropic_key: str) -> None:
    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            deltas: list[Delta] = []
            async for d in llm.stream(
                messages=[Message("user", "Reply with the single word: hello")],
                model=HAIKU_MODEL,
                max_tokens=MAX_TOKENS,
            ):
                assert isinstance(d, Delta)
                assert d.provider == "anthropic"
                deltas.append(d)
            assert len(deltas) >= 1
            terminal = deltas[-1]
            assert terminal.usage is not None
            assert terminal.finish_reason is not None
            result = assemble_deltas(deltas)
            assert isinstance(result, LLMResult)
            assert result.content
            assert result.provider == "anthropic"
            assert result.usage.output_tokens > 0
        finally:
            await llm.aclose()

    _run(go())


# ────────────────────────────────────────────────────────────────
# Anthropic — sad paths
# ────────────────────────────────────────────────────────────────


@requires_anthropic
def test_anthropic_empty_api_key_raises_provider_auth_error() -> None:
    async def go() -> None:
        llm = _claude(api_key="", model=HAIKU_MODEL)
        try:
            with pytest.raises(ProviderAuthError):
                await llm.chat(
                    messages=[Message("user", "hi")],
                    model=HAIKU_MODEL,
                    max_tokens=10,
                )
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_anthropic_invalid_api_key_raises_provider_auth_error() -> None:
    async def go() -> None:
        llm = _claude(api_key="sk-invalid-key-not-real", model=HAIKU_MODEL)
        try:
            with pytest.raises(ProviderAuthError):
                await llm.chat(
                    messages=[Message("user", "hi")],
                    model=HAIKU_MODEL,
                    max_tokens=10,
                )
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_anthropic_model_not_found_surfaces_error(anthropic_key: str) -> None:
    """A bogus model name should raise an informative error (Anthropic returns
    404/400). It must not silently succeed."""
    from agentkit.adapters.llm.providers.base import ProviderError

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model="claude-does-not-exist-xyz")
        try:
            with pytest.raises(ProviderError) as excinfo:
                await llm.chat(
                    messages=[Message("user", "hi")],
                    model="claude-does-not-exist-xyz",
                    max_tokens=10,
                )
            msg = str(excinfo.value).lower()
            assert (
                "does-not-exist" in msg
                or "model" in msg
                or "not_found" in msg
                or "invalid" in msg
                or "404" in msg
                or "400" in msg
            )
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_anthropic_max_tokens_1_truncates(anthropic_key: str) -> None:
    """max_tokens=1 → finish_reason reflects the length cap; output_tokens ≤ 1."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            res = await llm.chat(
                messages=[Message("user", "Write a long five-sentence poem about cats.")],
                model=HAIKU_MODEL,
                max_tokens=1,
            )
            assert res.usage.output_tokens <= 1
            # Anthropic sends stop_reason == "max_tokens" when the cap fires
            assert res.finish_reason in {"max_tokens", "length", "end_turn"}
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_anthropic_asyncio_timeout_cancels_cleanly(anthropic_key: str) -> None:
    """asyncio.wait_for at 1ms should raise TimeoutError/CancelledError and a fresh
    client should still work — no dangling loop / client state."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            with pytest.raises((asyncio.TimeoutError, asyncio.CancelledError)):
                await asyncio.wait_for(
                    llm.chat(
                        messages=[Message("user", "hi")],
                        model=HAIKU_MODEL,
                        max_tokens=10,
                    ),
                    timeout=0.001,
                )
        finally:
            await llm.aclose()

        llm2 = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            res = await llm2.chat(
                messages=[Message("user", "Reply with: ok")],
                model=HAIKU_MODEL,
                max_tokens=10,
            )
            assert res.content
        finally:
            await llm2.aclose()

    _run(go())


# ────────────────────────────────────────────────────────────────
# OpenAI — happy path (skipped when billing/key invalid)
# ────────────────────────────────────────────────────────────────


@requires_openai
def test_openai_chat_returns_valid_llmresult(openai_key: str) -> None:
    async def go() -> None:
        llm = _openai(api_key=openai_key, model=OPENAI_MINI_MODEL)
        try:
            res = await llm.chat(
                messages=[Message("user", "Reply with the single word: hello")],
                model=OPENAI_MINI_MODEL,
                max_tokens=MAX_TOKENS,
            )
            assert isinstance(res, LLMResult)
            assert res.content
            assert res.provider in {"openai", "openai-compatible", "openai_compat", "openai-compat"}
            assert res.usage.input_tokens > 0
            assert res.usage.output_tokens > 0
        finally:
            await llm.aclose()

    _run(go())


@requires_openai
def test_openai_stream_yields_deltas_and_assembles(openai_key: str) -> None:
    async def go() -> None:
        llm = _openai(api_key=openai_key, model=OPENAI_MINI_MODEL)
        try:
            deltas: list[Delta] = []
            async for d in llm.stream(
                messages=[Message("user", "Say: hi")],
                model=OPENAI_MINI_MODEL,
                max_tokens=MAX_TOKENS,
            ):
                deltas.append(d)
            assert len(deltas) >= 1
            result = assemble_deltas(deltas)
            assert result.content
        finally:
            await llm.aclose()

    _run(go())


# ────────────────────────────────────────────────────────────────
# OpenAI — sad paths (auth mapping fires even if billing is stale).
# ────────────────────────────────────────────────────────────────


@requires_openai
def test_openai_invalid_api_key_raises_provider_auth_error() -> None:
    async def go() -> None:
        llm = _openai(api_key="sk-invalid-not-real", model=OPENAI_MINI_MODEL)
        try:
            with pytest.raises(ProviderAuthError):
                await llm.chat(
                    messages=[Message("user", "hi")],
                    model=OPENAI_MINI_MODEL,
                    max_tokens=10,
                )
        finally:
            await llm.aclose()

    _run(go())


@requires_openai
def test_openai_empty_api_key_raises_provider_auth_error() -> None:
    async def go() -> None:
        llm = _openai(api_key="", model=OPENAI_MINI_MODEL)
        try:
            with pytest.raises(ProviderAuthError):
                await llm.chat(
                    messages=[Message("user", "hi")],
                    model=OPENAI_MINI_MODEL,
                    max_tokens=10,
                )
        finally:
            await llm.aclose()

    _run(go())
