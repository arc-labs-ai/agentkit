"""A provider failure arriving INSIDE a 200 response must not read as a complete answer.

Both providers can deliver an error frame mid-stream, once the headers are
long gone:

    Anthropic:  {"type": "error", "error": {"type": "overloaded_error", ...}}
    OpenAI:     {"error": {"message": "...", "type": "server_error", ...}}

Neither translator had a branch for it, so the frame fell through every
`elif`, the loop ended normally, and the caller received a TRUNCATED answer
presented as a complete one — partial text, `finish_reason=None`, no exception
anywhere. An agent takes that half-sentence as the model's final word.

And because nothing raised, `retry()` never fired: the most retryable provider
failure there is — an overload — was the one the resilience layer never saw.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.kernel.types import Message

httpx = pytest.importorskip("httpx")


def _sse(*lines: str) -> str:
    return "".join(line + "\n" for line in lines)


def _client(body: str):
    def handler(_request):
        # 200 OK. The failure is in the BODY, which is the whole point: status
        # -based error handling never sees it.
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _anthropic(body: str):
    from agentkit.adapters.llm.providers import AnthropicLLM

    return AnthropicLLM(api_key="k", client=_client(body))


def _openai(body: str):
    from agentkit.adapters.llm.providers import OpenAICompatibleLLM

    return OpenAICompatibleLLM(api_key="k", base_url="https://x", client=_client(body))


def _drain(llm, model: str):
    async def go():
        return [d async for d in llm.stream(messages=[Message("user", "hi")], model=model)]

    return asyncio.run(go())


# ── the happy path still works ───────────────────────────────────────────────


def test_anthropic_normal_stream_is_unaffected() -> None:
    deltas = _drain(
        _anthropic(
            _sse(
                'data: {"type":"message_start","message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":10}}}',
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
                "data: [DONE]",
            )
        ),
        "claude-sonnet-4-6",
    )
    assert "".join(d.text for d in deltas) == "Hello world"
    assert deltas[-1].finish_reason == "end_turn"
    assert deltas[-1].usage.input_tokens == 10 and deltas[-1].usage.output_tokens == 5


def test_openai_normal_stream_is_unaffected() -> None:
    deltas = _drain(
        _openai(
            _sse(
                'data: {"model":"gpt-4o","choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"model":"gpt-4o","choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            )
        ),
        "gpt-4o",
    )
    assert "".join(d.text for d in deltas) == "Hello world"


# ── an in-band error must raise, not truncate ────────────────────────────────


def test_anthropic_midstream_error_raises_instead_of_truncating() -> None:
    llm = _anthropic(
        _sse(
            'data: {"type":"message_start","message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":10}}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The answer is "}}',
            "event: error",
            'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
        )
    )
    from agentkit.adapters.llm.providers import ProviderError

    with pytest.raises(ProviderError, match="overloaded_error"):
        _drain(llm, "claude-sonnet-4-6")


def test_openai_midstream_error_raises_instead_of_truncating() -> None:
    llm = _openai(
        _sse(
            'data: {"model":"gpt-4o","choices":[{"delta":{"content":"The answer is "}}]}',
            'data: {"error":{"message":"upstream overloaded","type":"server_error","code":500}}',
        )
    )
    from agentkit.adapters.llm.providers import ProviderError

    with pytest.raises(ProviderError, match="server_error"):
        _drain(llm, "gpt-4o")


def test_an_error_frame_before_any_token_also_raises() -> None:
    """The rate-limit shape: the failure arrives first, with a 200 status."""
    from agentkit.adapters.llm.providers import ProviderError

    llm = _anthropic(
        _sse('data: {"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}')
    )
    with pytest.raises(ProviderError, match="rate_limit_error"):
        _drain(llm, "claude-sonnet-4-6")


# ── and the raised error must route correctly ────────────────────────────────


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("overloaded_error", "TRANSIENT"),
        ("rate_limit_error", "TRANSIENT"),
        ("server_error", "TRANSIENT"),
        ("invalid_request_error", "PERMANENT"),
    ],
)
def test_the_raised_error_classifies_for_the_resilience_layer(kind: str, expected: str) -> None:
    """Raising is only half the fix. `retry()` and `fallback()` branch on
    `classify`, so an overload that raised but classified as PERMANENT would
    fail fast on the one failure most worth retrying.

    Provider error TYPES use underscores (`rate_limit_error`) while the
    classifier's vocabulary had spaces (`rate limit`), so these landed in
    UNKNOWN — still retried, since only PERMANENT fails fast, but classified
    on nothing.
    """
    from agentkit.adapters.llm.providers import ProviderError
    from agentkit.kernel.resilience import classify

    exc = ProviderError(f"provider stream error [{kind}]: something happened")
    assert classify(exc).name == expected


def test_the_classifier_does_not_read_5000_as_a_500() -> None:
    """Bare `500` is deliberately absent from the TRANSIENT vocabulary: it is a
    substring of `5000`, which appears in ordinary text like `max_tokens 5000`,
    and a false TRANSIENT there retries a request that can never succeed."""
    from agentkit.kernel.resilience import classify

    assert classify(RuntimeError("model max_tokens 5000 exceeded")).name == "UNKNOWN"


def test_a_malformed_error_frame_still_raises_something_useful() -> None:
    """A provider that sends `{"error": {}}` — no type, no message — must still
    fail loudly rather than falling through to a silent truncation."""
    from agentkit.adapters.llm.providers import ProviderError

    llm = _openai(
        _sse(
            'data: {"model":"gpt-4o","choices":[{"delta":{"content":"partial"}}]}',
            'data: {"error":{}}',
        )
    )
    with pytest.raises(ProviderError, match="mid-stream"):
        _drain(llm, "gpt-4o")
