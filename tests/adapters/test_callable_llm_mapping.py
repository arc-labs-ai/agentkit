"""`CallableLLM` / `_duck_to_result` — the duck-typed coercion an injected provider fn goes through.

The coercion is best-effort by contract, which makes it exactly the place where a wrong answer is
invisible: it has no schema to violate and no exception to raise, so a mis-read response becomes an
empty string the agent reads as the model's answer, and a zero `Usage` the budget bills as free.
"""

import asyncio

import pytest

from agentkit.adapters.llm._mapping import _duck_to_result, result_to_delta
from agentkit.adapters.llm.callable import CallableLLM
from agentkit.kernel.types import LLMResult, Message, Usage


def _run(coro):
    return asyncio.run(coro)


_DICT_RESPONSE = {
    "content": "the answer is 42",
    "model": "m1",
    "provider": "acme",
    "finish_reason": "stop",
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "tool_calls": [{"id": "t1", "function": {"name": "search", "arguments": '{"q": "x"}'}}],
}


class _SdkUsage:
    """A real SDK usage OBJECT — attributes, no ``.get``."""

    input_tokens = 7
    output_tokens = 3


class _SdkResponse:
    content = "from an object"
    model = "m2"
    provider = "acme"
    finish_reason = "end_turn"
    usage = _SdkUsage()


# ── a dict-shaped response is a real response, not an empty one ─────────────


def test_a_dict_shaped_response_keeps_content_usage_cost_and_tool_calls() -> None:
    """Regression: `_duck_to_result` read every field with bare `getattr`, so the most common
    shape an injected fn returns — a plain dict — matched nothing. Measured before the fix:
    `content='' usage=Usage(0, 0, 0.0)`, tool calls gone, no exception. The agent took the empty
    string as the model's final word and the call was billed as free."""
    llm = CallableLLM(lambda **_kw: _DICT_RESPONSE, cost_fn=lambda model, i, o: 0.5 * i)
    res = _run(llm.complete(system="s", user="u", model="m"))

    assert res.content == "the answer is 42"
    assert res.model == "m1" and res.provider == "acme" and res.finish_reason == "stop"
    assert res.usage.input_tokens == 10 and res.usage.output_tokens == 5
    assert res.usage.cost_usd == 5.0  # cost_fn ran on the REAL token counts, not zeros
    assert [(t.id, t.name, dict(t.arguments)) for t in res.tool_calls] == [
        ("t1", "search", {"q": "x"})
    ]


def test_an_object_shaped_response_still_works() -> None:
    """Positive control for the fix above: teaching the coercion about dicts must not cost it the
    attribute path it already handled. A fix that only ever calls `.get` fails here."""
    res = _duck_to_result(_SdkResponse())
    assert res.content == "from an object" and res.model == "m2"
    assert res.usage.input_tokens == 7 and res.usage.output_tokens == 3


def test_an_llmresult_is_passed_through_untouched() -> None:
    """Positive control: the identity shortcut must survive. A fix that re-coerced everything
    would rebuild (and flatten) a result the caller had already assembled."""
    original = LLMResult(content="done", usage=Usage(1, 2, 3.0))
    assert _duck_to_result(original) is original


# ── partial / hostile shapes degrade, never crash ───────────────────────────


def test_a_dict_missing_most_keys_degrades_field_by_field() -> None:
    """Edge: only `content` present. Everything else takes its documented default rather than
    dragging the whole result down to empty."""
    res = _duck_to_result({"content": "just text"})
    assert res.content == "just text"
    assert res.model is None and res.provider is None and res.finish_reason is None
    assert res.usage == Usage(0, 0, 0.0)
    assert res.tool_calls == ()


def test_a_dict_with_no_content_key_yields_an_empty_string_not_none() -> None:
    """Edge: `LLMResult.content` is typed `str`. A response carrying only tool calls (content
    `None`, the OpenAI shape) must still produce `""`."""
    res = _duck_to_result({"content": None, "model": "m"})
    assert res.content == "" and res.model == "m"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"usage": {"prompt_tokens": 4, "completion_tokens": 2}}, (4, 2)),  # OpenAI naming
        ({"usage": {"prompt_token_count": 4, "candidates_token_count": 2}}, (4, 2)),  # Gemini
        ({"usage": {"input_tokens": 4, "output_tokens": 2}}, (4, 2)),  # Anthropic
        ({"usage": {}}, (0, 0)),  # present but empty
        ({}, (0, 0)),  # absent entirely
        ({"usage": None}, (0, 0)),  # explicitly null
    ],
)
def test_usage_is_read_from_every_shape_a_provider_might_send(raw, expected) -> None:
    """Edge matrix: dict usage under each vendor's key naming, plus empty / missing / null."""
    res = _duck_to_result(raw)
    assert (res.usage.input_tokens, res.usage.output_tokens) == expected


def test_an_object_shaped_usage_degrades_instead_of_raising() -> None:
    """Regression: `getattr(raw, "usage", {})` then `usage.get(...)` raised
    `AttributeError: 'SdkUsage' object has no attribute 'get'` on a real SDK response object —
    a hard crash out of a helper whose entire contract is best-effort duck typing."""

    class _OnlyUsage:
        usage = _SdkUsage()

    assert _duck_to_result(_OnlyUsage()).usage.input_tokens == 7


def test_a_response_with_no_recognisable_field_still_returns_a_result() -> None:
    """Edge: nothing we know about. Empty result is the CORRECT answer here — the point of the
    fix is that it stops being the answer for shapes we do understand."""
    res = _duck_to_result(object())
    assert res.content == "" and res.usage == Usage(0, 0, 0.0)


# ── the chat() path ─────────────────────────────────────────────────────────


def test_chat_on_a_dict_response_reports_content_and_tool_calls_together() -> None:
    """`CallableLLM.chat` back-fills tool calls onto whatever `_to_result` produced. It used to be
    the ONLY thing rescued from a dict response; now the content and usage arrive with them."""
    llm = CallableLLM(lambda **_kw: {}, chat_fn=lambda **_kw: _DICT_RESPONSE)
    res = _run(llm.chat(messages=[Message("user", "u")], model="m"))
    assert res.content == "the answer is 42"
    assert res.usage.input_tokens == 10
    assert [t.name for t in res.tool_calls] == ["search"]


def test_an_injected_to_result_still_overrides_the_default_coercion() -> None:
    """Positive control: `to_result=` is the documented escape hatch for a provider shape the
    duck-typing can't read. A fix that hard-wired the default coercion would break it."""
    llm = CallableLLM(lambda **_kw: _DICT_RESPONSE, to_result=lambda raw: LLMResult(content="mine"))
    assert _run(llm.complete(system="s", user="u", model="m")).content == "mine"


# ── one seam, one answer ────────────────────────────────────────────────────
#
# `result_to_delta` carried six of `LLMResult`'s seven fields, dropping
# `parsed` — the identical drop that `kernel.middleware._result_to_stream` had.
# It is easy to assume it cannot matter, because providers build their result
# from wire data where `parsed` has no source. But `CallableLLM` passes a
# user-supplied callable's return value through `coerce` verbatim, so a
# callable returning `LLMResult(parsed=X)` reaches the mapping with a real
# object. Measured before the fix:
#
#     deltas: 1  parsed on terminal delta: None
#     chat() parsed: <Plan>
#
# The same LLM, the same call, two different answers depending on whether the
# caller used `chat()` or `stream()`.


class _Plan:
    """Stand-in for a typed output object. Deliberately not comparable by
    value, so the tests below assert on IDENTITY — a fresh equal object would
    let a re-parsing implementation pass without carrying anything through."""

    def __repr__(self) -> str:
        return "<Plan>"


def _typed_result(obj: object) -> LLMResult:
    return LLMResult(
        content='{"v": 1}',
        model="m",
        provider="p",
        finish_reason="stop",
        usage=None,
        tool_calls=(),
        parsed=obj,
    )


def test_result_to_delta_carries_parsed() -> None:
    plan = _Plan()
    assert result_to_delta(_typed_result(plan)).parsed is plan


def test_chat_and_stream_agree_on_parsed() -> None:
    """The property that actually matters to a caller: which method you reach
    for must not change the answer."""
    plan = _Plan()

    def chat_fn(**_kw: object) -> LLMResult:
        return _typed_result(plan)

    llm = CallableLLM(chat_fn)
    msgs = [Message(role="user", content="hi")]

    async def _go() -> tuple[object, object]:
        deltas = [d async for d in llm.stream(messages=msgs, model="m")]
        streamed = deltas[-1].parsed
        completed = (await llm.chat(messages=msgs, model="m")).parsed
        return streamed, completed

    streamed, completed = asyncio.run(_go())
    assert streamed is plan
    assert completed is plan
    assert streamed is completed, "one seam must not give two answers"


def test_the_replayed_delta_is_not_marked_partial() -> None:
    """POSITIVE CONTROL for the deliberate non-fix. `partial` means 'this is an
    in-progress partial parse'. A terminal delta rebuilt from a COMPLETE result
    is not one, so carrying `parsed` must not drag `partial` along with it."""
    assert result_to_delta(_typed_result(_Plan())).partial is None


def test_a_result_without_parsed_is_unaffected() -> None:
    """POSITIVE CONTROL: the overwhelmingly common case — every provider —
    must round-trip exactly as before."""
    d = result_to_delta(
        LLMResult(content="hello", model="m", provider="p", finish_reason="stop", usage=None)
    )
    assert d.parsed is None
    assert d.text == "hello" and d.model == "m" and d.provider == "p"
