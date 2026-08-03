"""Provider clients (httpx) tested fully offline via httpx.MockTransport — request shape, response/usage/
tool-call parsing, cache-aware cost, and error→classify mapping (429/5xx transient, 4xx permanent)."""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import (
    AnthropicLLM,
    OpenAICompatibleLLM,
    ProviderError,
    claude,
    openrouter,
)
from agentkit.kernel.resilience import ErrorClass, classify
from agentkit.kernel.types import Message, ToolSchema


def _run(coro):
    return asyncio.run(coro)


def _client_with(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _sse(body: str):
    def handler(request):
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---- OpenAI-compatible -------------------------------------------------------------------------


def test_openai_compat_chat_shape_parse_and_cost():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi there"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "prompt_tokens_details": {"cached_tokens": 200},
                },
            },
        )

    llm = OpenAICompatibleLLM(
        api_key="sk-x", base_url="https://api.openai.com/v1", client=_client_with(handler)
    )
    res = _run(
        llm.chat(messages=[Message("system", "s"), Message("user", "q")], model="gpt-4o-mini")
    )

    assert seen["url"].endswith("/chat/completions") and seen["auth"] == "Bearer sk-x"
    assert seen["body"]["model"] == "gpt-4o-mini" and [
        m["role"] for m in seen["body"]["messages"]
    ] == ["system", "user"]
    assert res.content == "hi there" and res.finish_reason == "stop"
    assert (
        res.usage.input_tokens == 800
        and res.usage.cache_read_tokens == 200
        and res.usage.output_tokens == 500
    )
    # 800/1e6*0.15 + 500/1e6*0.60 + 200/1e6*0.075
    assert res.usage.cost_usd == pytest.approx(0.000435)


def test_openai_compat_parses_tool_calls():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "fetch", "arguments": '{"url": "x"}'},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    llm = OpenAICompatibleLLM(api_key="k", base_url="http://x", client=_client_with(handler))
    res = _run(
        llm.chat(
            messages=[Message("user", "go")],
            model="m",
            tools=[ToolSchema("fetch", "Fetch", {"type": "object"})],
        )
    )
    assert len(res.tool_calls) == 1 and res.tool_calls[0].name == "fetch"
    assert res.tool_calls[0].arguments == {"url": "x"}


def test_error_mapping_429_transient_400_permanent():
    def make(status):
        def handler(request):
            return httpx.Response(status, json={"error": "nope"})

        return OpenAICompatibleLLM(api_key="k", base_url="http://x", client=_client_with(handler))

    with pytest.raises(ProviderError) as e429:
        _run(make(429).complete(system="s", user="u", model="m"))
    assert classify(e429.value) is ErrorClass.TRANSIENT  # rate-limited → retried

    with pytest.raises(ProviderError) as e400:
        _run(make(400).complete(system="s", user="u", model="m"))
    assert classify(e400.value) is ErrorClass.PERMANENT  # bad request → fail fast


# ---- OpenAI-compatible SSE streaming -----------------------------------------------------------

_OPENAI_STREAM = (
    'data: {"model": "gpt-4o-mini", "choices": [{"delta": {"role": "assistant", "content": "hi "}}]}\n\n'
    'data: {"choices": [{"delta": {"content": "there"}}]}\n\n'
    'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
    'data: {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 500, '
    '"prompt_tokens_details": {"cached_tokens": 200}}}\n\n'
    "data: [DONE]\n\n"
)


def test_openai_compat_stream_yields_incremental_deltas_with_usage():
    llm = OpenAICompatibleLLM(api_key="k", base_url="http://x", client=_sse(_OPENAI_STREAM))
    deltas = _run(_collect(llm.stream(messages=[Message("user", "q")], model="gpt-4o-mini")))
    texts = [d.text for d in deltas if d.text]
    assert texts == ["hi ", "there"]  # genuinely incremental
    assert "".join(texts) == "hi there"
    assert deltas[-1].finish_reason == "stop"
    u = deltas[-1].usage
    assert u.input_tokens == 800 and u.cache_read_tokens == 200 and u.output_tokens == 500
    assert u.cost_usd == pytest.approx(0.000435)  # same cache-aware cost as the non-stream path


_OPENAI_TOOL_STREAM = (
    'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function", '
    '"function": {"name": "fetch", "arguments": ""}}]}}]}\n\n'
    'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"url\\""}}]}}]}\n\n'
    'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": \\"x\\"}"}}]}, '
    '"finish_reason": "tool_calls"}]}\n\n'
    "data: [DONE]\n\n"
)


def test_openai_compat_stream_assembles_tool_call_fragments():
    llm = OpenAICompatibleLLM(api_key="k", base_url="http://x", client=_sse(_OPENAI_TOOL_STREAM))
    deltas = _run(
        _collect(
            llm.stream(
                messages=[Message("user", "go")],
                model="m",
                tools=[ToolSchema("fetch", "Fetch", {"type": "object"})],
            )
        )
    )
    final = deltas[-1]
    assert final.finish_reason == "tool_calls" and len(final.tool_calls) == 1
    assert final.tool_calls[0].name == "fetch" and final.tool_calls[0].arguments == {"url": "x"}


async def _collect(stream):
    return [d async for d in stream]


_OPENAI_PARALLEL_NOID_STREAM = (
    'data: {"choices": [{"delta": {"tool_calls": ['
    '{"index": 0, "function": {"name": "fetch", "arguments": "{}"}}, '
    '{"index": 1, "function": {"name": "fetch", "arguments": "{}"}}]}}]}\n\n'
    'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}\n\n'
    "data: [DONE]\n\n"
)


def test_openai_compat_stream_gives_distinct_ids_to_parallel_same_tool_calls():
    """Regression: two parallel calls to the SAME tool with no provider id must get distinct ids (from the
    stream index), not collapse onto the tool name."""
    llm = OpenAICompatibleLLM(
        api_key="k", base_url="http://x", client=_sse(_OPENAI_PARALLEL_NOID_STREAM)
    )
    final = _run(_collect(llm.stream(messages=[Message("user", "go")], model="m")))[-1]
    ids = [tc.id for tc in final.tool_calls]
    assert (
        len(final.tool_calls) == 2
        and len(set(ids)) == 2
        and all(tc.name == "fetch" for tc in final.tool_calls)
    )


# ---- Anthropic ---------------------------------------------------------------------------------


def test_anthropic_chat_shape_cache_and_cost():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["api_key"] = request.headers.get("x-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "answer"}],
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cache_read_input_tokens": 2000,
                    "cache_creation_input_tokens": 100,
                },
            },
        )

    llm = AnthropicLLM(api_key="ak", client=_client_with(handler))
    res = _run(
        llm.chat(
            messages=[Message("system", "be concise"), Message("user", "q")],
            model="claude-sonnet-4-6",
            cache_hint="prefix",
        )
    )

    assert seen["url"].endswith("/v1/messages") and seen["api_key"] == "ak"
    # cache_hint → system carried as a cache_control block (prompt caching)
    assert (
        isinstance(seen["body"]["system"], list)
        and seen["body"]["system"][0]["cache_control"]["type"] == "ephemeral"
    )
    assert [m["role"] for m in seen["body"]["messages"]] == [
        "user"
    ]  # system is top-level, not a message
    assert res.content == "answer"
    assert res.usage.cache_read_tokens == 2000 and res.usage.cache_write_tokens == 100
    # 1000/1e6*3 + 500/1e6*15 + 2000/1e6*0.30 + 100/1e6*3.75
    assert res.usage.cost_usd == pytest.approx(0.011475)


# ---- Anthropic SSE streaming -------------------------------------------------------------------

_ANTHROPIC_STREAM = (
    'data: {"type": "message_start", "message": {"model": "claude-sonnet-4-6", "usage": '
    '{"input_tokens": 1000, "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 100}}}\n\n'
    'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n'
    'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ans"}}\n\n'
    'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "wer"}}\n\n'
    'data: {"type": "content_block_stop", "index": 0}\n\n'
    'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 500}}\n\n'
    'data: {"type": "message_stop"}\n\n'
)


def test_anthropic_stream_yields_text_deltas_and_split_usage():
    llm = AnthropicLLM(api_key="ak", client=_sse(_ANTHROPIC_STREAM))
    deltas = _run(_collect(llm.stream(messages=[Message("user", "q")], model="claude-sonnet-4-6")))
    texts = [d.text for d in deltas if d.text]
    assert texts == ["ans", "wer"] and "".join(texts) == "answer"  # incremental text_deltas
    final = deltas[-1]
    assert final.finish_reason == "end_turn"
    u = final.usage
    assert (
        u.input_tokens == 1000 and u.output_tokens == 500
    )  # split across message_start/message_delta
    assert u.cache_read_tokens == 2000 and u.cache_write_tokens == 100
    assert u.cost_usd == pytest.approx(0.011475)  # matches the non-stream cache-aware cost


_ANTHROPIC_TOOL_STREAM = (
    'data: {"type": "message_start", "message": {"model": "claude-sonnet-4-6", "usage": {"input_tokens": 10}}}\n\n'
    'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", '
    '"name": "fetch"}}\n\n'
    'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", '
    '"partial_json": "{\\"url\\":"}}\n\n'
    'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", '
    '"partial_json": " \\"x\\"}"}}\n\n'
    'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}}\n\n'
    'data: {"type": "message_stop"}\n\n'
)


def test_anthropic_stream_assembles_tool_use_input_json():
    llm = AnthropicLLM(api_key="ak", client=_sse(_ANTHROPIC_TOOL_STREAM))
    deltas = _run(_collect(llm.stream(messages=[Message("user", "go")], model="claude-sonnet-4-6")))
    final = deltas[-1]
    assert final.finish_reason == "tool_use" and len(final.tool_calls) == 1
    assert final.tool_calls[0].name == "fetch" and final.tool_calls[0].arguments == {"url": "x"}


# ---- presets -----------------------------------------------------------------------------------


def test_anthropic_max_tokens_default_is_explicit_and_configurable():
    """Regression: Anthropic requires max_tokens; an unset call must use the documented default (4096),
    not a silent 1024, and the default must be overridable per-client."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "x"}], "usage": {}})

    llm = AnthropicLLM(api_key="k", client=_client_with(handler))
    _run(llm.chat(messages=[Message("user", "q")], model="claude-sonnet-4-6"))
    assert seen["body"]["max_tokens"] == 4096  # explicit default, not 1024

    llm2 = AnthropicLLM(api_key="k", default_max_tokens=200, client=_client_with(handler))
    _run(llm2.chat(messages=[Message("user", "q")], model="m"))
    assert seen["body"]["max_tokens"] == 200  # configurable per client


def test_presets_wire_base_url_and_defaults():
    assert openrouter(api_key="k")._base_url == "https://openrouter.ai/api/v1"
    c = claude(api_key="k")
    assert c._base_url == "https://api.anthropic.com" and c._default_model == "claude-sonnet-4-6"


def test_owned_client_rebuilt_across_event_loops():
    """Regression: an owned httpx client is bound to its creating loop. Reusing one provider instance
    across separate `asyncio.run(...)` calls must rebuild the client (not error / not reuse a dead loop's)."""
    llm = OpenAICompatibleLLM(
        api_key="k", base_url="http://x", default_model="m"
    )  # owns its client

    async def touch():
        return id(llm._http())  # creating an AsyncClient opens no sockets

    first = _run(touch())
    second = _run(touch())  # a different event loop
    assert first != second  # rebuilt for the new loop, no RuntimeError


def test_injected_client_is_reused_and_never_owned():
    shared = _client_with(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})
    )
    llm = OpenAICompatibleLLM(api_key="k", base_url="http://x", client=shared)

    async def go():
        a = llm._http()
        await llm.aclose()  # must NOT close an injected client
        return a is shared and llm._http() is shared

    assert _run(go())


# ── Multi-turn tool-call history serializes cleanly ──────────────────────────
#
# When a ``ToolCall`` from a prior turn is embedded in the assistant
# message history, ``_to_msg`` must convert its ``MappingProxyType``
# arguments to a plain ``dict`` before JSON-serialization. A raw
# ``mappingproxy`` in the JSON payload raises ``TypeError`` — every
# multi-turn tool-calling flow would break at request build time.


def test_openai_compat_serializes_assistant_history_with_prior_tool_call() -> None:
    """A follow-up chat with a prior assistant ``tool_calls`` message
    in the history must serialize cleanly. The prior ``ToolCall``
    holds its arguments as a ``MappingProxyType``; the request
    builder must unwrap it before ``json.dumps`` sees it."""
    from agentkit.kernel.types import ToolCall

    seen_body: dict = {}

    def handler(request):
        seen_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    llm = OpenAICompatibleLLM(api_key="k", base_url="http://x", client=_client_with(handler))

    # A prior turn's assistant message carrying a ToolCall (arguments = MappingProxyType).
    prior_call = ToolCall("call-1", "search", {"query": "hello", "k": 5})
    history = [
        Message("user", "find something"),
        Message("assistant", "", tool_calls=(prior_call,)),
        Message("tool", "search results here", tool_call_id="call-1"),
        Message("user", "summarise"),
    ]

    _run(llm.chat(messages=history, model="gpt-4o-mini"))

    # The serialized payload landed with a real dict as arguments — no proxy leak.
    assistant_msg = next(m for m in seen_body["messages"] if m["role"] == "assistant")
    tool_calls_payload = assistant_msg["tool_calls"]
    args_json = tool_calls_payload[0]["function"]["arguments"]
    assert json.loads(args_json) == {"query": "hello", "k": 5}


def test_anthropic_serializes_assistant_history_with_prior_tool_use() -> None:
    """Same invariant on the Anthropic side: a prior ``tool_use``
    block in the assistant history must serialize the arguments as
    a plain dict, not a ``MappingProxyType``. httpx's ``json=``
    payload path calls stdlib ``json.dumps`` which cannot encode
    the proxy view."""
    from agentkit.kernel.types import ToolCall

    seen_body: dict = {}

    def handler(request):
        seen_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "answer"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    llm = AnthropicLLM(api_key="ak", client=_client_with(handler))

    prior_call = ToolCall("tool_use_1", "search", {"query": "hello", "topk": 5})
    history = [
        Message("user", "find something"),
        Message("assistant", "", tool_calls=(prior_call,)),
        Message("tool", "search results here", tool_call_id="tool_use_1"),
        Message("user", "summarise"),
    ]

    _run(llm.chat(messages=history, model="claude-sonnet-4-6"))

    # The serialized payload's ``input`` is a real dict.
    assistant_msg = next(m for m in seen_body["messages"] if m["role"] == "assistant")
    tool_use = next(b for b in assistant_msg["content"] if b["type"] == "tool_use")
    assert tool_use["input"] == {"query": "hello", "topk": 5}
    assert isinstance(tool_use["input"], dict)


# ── Malformed tool-call arguments preserve raw text ──────────────────────────
#
# Providers occasionally hallucinate malformed JSON in tool_call
# ``arguments`` (unclosed braces, truncated strings, escaped quotes
# gone wrong). ``coerce_tool_args`` is the single choke point: it
# accepts anything, returns a dict, and stashes non-parseable inputs
# under ``{"_raw": …}`` so a downstream tool can still see what the
# model tried to say. Never crashes on the provider parse path.


def test_coerce_tool_args_wraps_unclosed_json_under_raw() -> None:
    """Truncated JSON — the classic streaming-cutoff shape — never
    raises. The downstream tool sees ``{"_raw": <original text>}``
    and decides how to handle it (raise, retry, treat as query text)."""
    from agentkit.adapters.llm._mapping import coerce_tool_args

    result = coerce_tool_args('{"query": "hello')  # unclosed brace + missing quote
    assert result == {"_raw": '{"query": "hello'}


def test_coerce_tool_args_wraps_non_object_json_under_raw() -> None:
    """Valid JSON that isn't a dict (list, scalar, bool) can't satisfy
    the tool-args shape either. The raw text survives so downstream
    diagnostics can show the model exactly what came back."""
    from agentkit.adapters.llm._mapping import coerce_tool_args

    assert coerce_tool_args("[1, 2, 3]") == {"_raw": "[1, 2, 3]"}
    assert coerce_tool_args('"just a string"') == {"_raw": '"just a string"'}
    assert coerce_tool_args("true") == {"_raw": "true"}


def test_coerce_tool_args_passes_dict_through_untouched() -> None:
    """When the provider hands us a real dict (already parsed by its
    own client library), we don't re-serialize + re-parse it."""
    from agentkit.adapters.llm._mapping import coerce_tool_args

    original = {"query": "hello", "k": 5}
    assert coerce_tool_args(original) is original


def test_coerce_tool_args_treats_none_and_empty_as_empty_dict() -> None:
    """A missing ``arguments`` field on the provider payload (``None``,
    ``""``) resolves to ``{}`` — the empty-args case. Downstream tools
    that accept no arguments (or have all-optional params) work
    without further guards."""
    from agentkit.adapters.llm._mapping import coerce_tool_args

    assert coerce_tool_args(None) == {}
    assert coerce_tool_args("") == {}
    assert coerce_tool_args("{}") == {}


def test_coerce_tool_args_handles_non_string_input_without_crashing() -> None:
    """A provider client that hands us a non-string non-dict (an int,
    a bytes object with binary garbage) must not blow up. Preserves
    the raw value under ``_raw`` for the tool to inspect."""
    from agentkit.adapters.llm._mapping import coerce_tool_args

    assert coerce_tool_args(42) == {"_raw": 42}
    # Bytes with non-UTF-8 content — a provider handing us binary
    # nonsense. Still no crash, still preserved.
    binary = b"\xff\xfe\x00\x01"
    assert coerce_tool_args(binary) == {"_raw": binary}
