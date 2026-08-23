"""`response_format` on the Anthropic adapter — a parameter the Messages API does not have.

It used to be accepted and dropped: measured, the request payload keys were
`['max_tokens', 'messages', 'model', 'temperature']`, so a caller that asked for JSON got prose,
got no error, and had no way to find out. The adapter now translates the shapes that have a
faithful prompt-level equivalent (announcing the downgrade once per client) and refuses the rest.
"""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import AnthropicLLM
from agentkit.kernel.types import Message

_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}
_JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "Out", "schema": _SCHEMA, "strict": True},
}


def _run(coro):
    return asyncio.run(coro)


def _client():
    """An AnthropicLLM plus the dict its last request payload landed in."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"answer": "42"}'}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    llm = AnthropicLLM(
        api_key="k",
        base_url="http://x",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return llm, seen


def _chat(llm, response_format, messages=None):
    return _run(
        llm.chat(
            messages=messages or [Message("user", "q")],
            model="claude-sonnet-4-6",
            response_format=response_format,
        )
    )


# ── translated ──────────────────────────────────────────────────────────────


def test_json_object_becomes_a_system_instruction() -> None:
    """Regression: the request now CARRIES the caller's contract. Before the fix the payload had
    no `system` key at all for this call."""
    llm, seen = _client()
    with pytest.warns(UserWarning, match="no response_format parameter"):
        _chat(llm, {"type": "json_object"})
    assert "single valid JSON object" in seen["body"]["system"]


def test_json_schema_puts_the_schema_itself_in_the_prompt() -> None:
    """A `json_schema` format names a concrete shape. Asking for "some JSON" would satisfy the
    type and lose the schema, which is the part the caller actually cares about."""
    llm, seen = _client()
    with pytest.warns(UserWarning):
        _chat(llm, _JSON_SCHEMA_FORMAT)
    system = seen["body"]["system"]
    assert "answer" in system and '"type": "object"' in system


def test_the_instruction_is_appended_to_an_existing_system_prompt() -> None:
    """Edge: the caller's own system turns must survive — the instruction is added to them, not
    substituted for them."""
    llm, seen = _client()
    with pytest.warns(UserWarning):
        _chat(
            llm,
            {"type": "json_object"},
            messages=[Message("system", "You are terse."), Message("user", "q")],
        )
    system = seen["body"]["system"]
    assert system.startswith("You are terse.")
    assert "single valid JSON object" in system


def test_the_streaming_path_translates_it_too() -> None:
    """Edge: `stream()` builds its own payload. A fix applied to `chat()` alone leaves the
    streamed path exactly as broken as it was."""
    body = (
        'data: {"type": "message_start", "message": {"model": "claude-sonnet-4-6", '
        '"usage": {"input_tokens": 1}}}\n\n'
        'data: {"type": "message_stop"}\n\n'
    )
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    llm = AnthropicLLM(
        api_key="k",
        base_url="http://x",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def drain():
        return [
            d
            async for d in llm.stream(
                messages=[Message("user", "q")],
                model="claude-sonnet-4-6",
                response_format={"type": "json_object"},
            )
        ]

    with pytest.warns(UserWarning):
        _run(drain())
    assert "single valid JSON object" in seen["body"]["system"]


def test_the_downgrade_warning_fires_once_per_client() -> None:
    """The warning says a provider-enforced contract became a best-effort prompt. That is worth
    saying — and worth saying ONCE, not on every turn of an agent loop, or it gets filtered out
    along with the warnings that matter."""
    llm, _seen = _client()
    with pytest.warns(UserWarning) as first:
        _chat(llm, {"type": "json_object"})
    assert len(first) == 1

    import warnings

    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        _chat(llm, {"type": "json_object"})
    assert second == []


# ── untouched / refused ─────────────────────────────────────────────────────


def test_no_response_format_leaves_the_payload_exactly_as_it_was() -> None:
    """POSITIVE CONTROL. The overwhelmingly common call passes nothing — it must gain no system
    prompt, no instruction and no warning. A "fix" that always appended guidance would quietly
    edit every Anthropic prompt in the framework."""
    llm, seen = _client()
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _chat(llm, None)
    assert caught == []
    assert "system" not in seen["body"]


def test_a_text_response_format_is_a_no_op() -> None:
    """Edge: `{"type": "text"}` IS the default. Translating it into a JSON instruction would be
    the opposite of what the caller asked for."""
    llm, seen = _client()
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _chat(llm, {"type": "text"})
    assert caught == []
    assert "system" not in seen["body"]


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "xml"},  # a mode we have no translation for
        {"type": "json_schema"},  # json_schema with no schema
        {"type": "json_schema", "json_schema": {"name": "Out"}},  # ... nor under the wrapper
        "json_object",  # not a dict at all
    ],
)
def test_an_untranslatable_response_format_is_refused_loudly(bad) -> None:
    """The other half of the honest behaviour: what cannot be translated must not be dropped.
    The error names the parameter and the way out, at the call that used it."""
    llm, seen = _client()
    with pytest.raises(ValueError, match="response_format"):
        _chat(llm, bad)
    assert seen == {}  # refused BEFORE the request went out


def test_the_result_still_parses_normally_when_a_format_was_requested() -> None:
    """Positive control: translating the parameter must not disturb the response mapping."""
    llm, _seen = _client()
    with pytest.warns(UserWarning):
        res = _chat(llm, {"type": "json_object"})
    assert res.content == '{"answer": "42"}'
    assert res.finish_reason == "end_turn" and res.usage.input_tokens == 1
