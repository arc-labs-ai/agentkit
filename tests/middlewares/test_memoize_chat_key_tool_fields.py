"""`default_key`'s CHAT key must include a transcript's tool fields.

The key reduced each `Message` to `(role, content)`, dropping the three fields
that carry a ReAct loop's entire state: `tool_calls` (what the assistant asked
for), `tool_call_id` (which request a result answers) and `name` (which tool
produced it). A tool-requesting assistant turn has `content == ""`, so under
the old key it was indistinguishable from any other empty assistant turn.

Measured before the fix: two transcripts differing ONLY in
`assistant.tool_calls` — `weather(SF)` vs `weather(NYC)` — and in their
`tool_call_id`s both hashed to `memo:9fc800ba328158c24129902b`. One branch of a
ReAct loop could therefore be served another branch's answer.

`cache_hint` was ignored too — the one `ChatRequest` field the key never
touched. It is typed `Any` and `CallableLLM` forwards it verbatim into a
user-supplied `chat_fn(**kw)`, where it CAN change the answer, so it is hashed
now; the cost is at most one extra miss when the hint itself changes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentkit.adapters.store.memory import InMemoryStore
from agentkit.kernel.types import ChatRequest, Message, Scope, ToolCall
from agentkit.middlewares import memoize
from agentkit.middlewares.memoize import _message_identity, default_key
from agentkit.runtime import Budget, Invoker, RunContext, Services


def _key(request: ChatRequest) -> str:
    return default_key(type("C", (), {"kind": "chat", "request": request})())


def _react(city: str, call_id: str = "call-1", result: str = "sunny") -> list[Message]:
    """A minimal ReAct transcript: ask → assistant requests `weather(city)` →
    tool result comes back tagged with the matching id."""
    return [
        Message("user", "what is the weather?"),
        Message("assistant", "", tool_calls=(ToolCall(call_id, "weather", {"city": city}),)),
        Message("tool", result, name="weather", tool_call_id=call_id),
    ]


def test_transcripts_differing_only_in_tool_calls_get_different_keys() -> None:
    """The load-bearing case: `weather(SF)` and `weather(NYC)` both hashed to
    `memo:9fc800ba328158c24129902b`, because the arguments live on `tool_calls`
    and the requesting turn's `content` is `""` in both."""
    sf = _key(ChatRequest(_react("SF"), "m"))
    nyc = _key(ChatRequest(_react("NYC"), "m"))

    assert sf != nyc, "two ReAct branches collapsed into one cache entry"


def test_transcripts_differing_only_in_tool_call_id_get_different_keys() -> None:
    """`tool_call_id` is what pairs a result with its request. Two transcripts
    that agree on every visible token but pair them differently are different
    conversations."""
    assert _key(ChatRequest(_react("SF", "call-1"), "m")) != _key(
        ChatRequest(_react("SF", "call-2"), "m")
    )


def test_transcripts_differing_only_in_message_name_get_different_keys() -> None:
    """`name` says WHICH tool produced a result. `42` from `stock` and `42` from
    `weather` are not the same observation."""
    a = [Message("tool", "42", name="weather", tool_call_id="c1")]
    b = [Message("tool", "42", name="stock", tool_call_id="c1")]

    assert _key(ChatRequest(a, "m")) != _key(ChatRequest(b, "m"))


def test_a_tool_requesting_turn_differs_from_a_plain_empty_turn() -> None:
    """The mechanism behind the collision, isolated: an assistant turn carrying
    `tool_calls` has empty `content`, so `(role, content)` made it identical to
    an assistant turn that said nothing at all."""
    requesting = [Message("assistant", "", tool_calls=(ToolCall("c1", "weather", {"city": "SF"}),))]
    silent = [Message("assistant", "")]

    assert _key(ChatRequest(requesting, "m")) != _key(ChatRequest(silent, "m"))


def test_transcripts_differing_only_in_the_number_of_tool_calls_differ() -> None:
    """A parallel-tool turn (two calls) is not the same request as a single-call
    turn that happens to share the first call."""
    one = (ToolCall("c1", "weather", {"city": "SF"}),)
    two = (*one, ToolCall("c2", "stock", {"ticker": "AAPL"}))

    assert _key(ChatRequest([Message("assistant", "", tool_calls=one)], "m")) != _key(
        ChatRequest([Message("assistant", "", tool_calls=two)], "m")
    )


def test_cache_hint_is_part_of_the_key() -> None:
    """`cache_hint` reaches the seam verbatim; `CallableLLM` hands it to a
    caller-supplied `chat_fn`, so a cache cannot assume the opaque field is
    inert."""
    msgs = [Message("user", "hi")]

    assert _key(ChatRequest(list(msgs), "m")) != _key(ChatRequest(list(msgs), "m", cache_hint="v2"))
    assert _key(ChatRequest(list(msgs), "m", cache_hint="v1")) != _key(
        ChatRequest(list(msgs), "m", cache_hint="v2")
    )


def test_identical_transcripts_still_share_a_key() -> None:
    """POSITIVE CONTROL. Hashing more fields must not amount to switching the
    chat cache off — two structurally identical ReAct transcripts, built from
    separate objects, still produce ONE key. A "fix" that folded in something
    per-instance (object identity, a timestamp) fails here."""
    assert _key(ChatRequest(_react("SF"), "m")) == _key(ChatRequest(_react("SF"), "m"))


def test_the_key_is_stable_across_dict_ORDERING() -> None:
    """POSITIVE CONTROL. Tool-call arguments and `response_format` are dicts;
    insertion order is not identity. `stable_hash` sorts keys, and
    `_message_identity` copies the frozen arguments mapping without
    re-ordering it."""
    a = [Message("assistant", "", tool_calls=(ToolCall("c1", "f", {"a": 1, "b": 2, "c": 3}),))]
    b = [Message("assistant", "", tool_calls=(ToolCall("c1", "f", {"c": 3, "b": 2, "a": 1}),))]

    assert _key(ChatRequest(a, "m")) == _key(ChatRequest(b, "m"))
    assert _key(ChatRequest([Message("user", "hi")], "m", response_format={"x": 1, "y": 2})) == _key(
        ChatRequest([Message("user", "hi")], "m", response_format={"y": 2, "x": 1})
    )


def test_unhashable_and_mutable_tool_call_arguments_key_cleanly() -> None:
    """Edge case: tool-call arguments are arbitrary JSON — lists, nested dicts,
    sets — none of which `hash()` accepts. `stable_hash` JSON-encodes them
    (sets sorted, keys sorted), so the key is computable AND stable."""
    args: dict[str, Any] = {"ids": [3, 1, 2], "tags": {"b", "a"}, "meta": {"deep": {"k": [1]}}}
    same = {"meta": {"deep": {"k": [1]}}, "tags": {"a", "b"}, "ids": [3, 1, 2]}
    other = {"ids": [1, 2, 3], "tags": {"a", "b"}, "meta": {"deep": {"k": [1]}}}

    def k(a: dict[str, Any]) -> str:
        return _key(ChatRequest([Message("assistant", "", tool_calls=(ToolCall("c1", "f", a),))], "m"))

    assert k(args) == k(same)
    assert k(args) != k(other), "list ORDER is meaningful and must be in the key"


def test_a_mutated_argument_dict_does_not_reuse_the_old_key() -> None:
    """Different arguments must produce a different key.

    `ToolCall` used to wrap `arguments` in a `MappingProxyType` over the
    caller's dict — a live VIEW, where mutating that dict silently changed
    what an already-built turn meant. `deep_freeze` COPIES, so it no longer
    does, and this test builds a second `ToolCall` from the mutated dict
    rather than relying on the old aliasing. What is locked either way is
    the part that matters: the key follows the arguments."""
    live: dict[str, Any] = {"city": "SF"}
    tc = ToolCall("c1", "weather", live)
    before = _key(ChatRequest([Message("assistant", "", tool_calls=(tc,))], "m"))

    live["city"] = "NYC"
    assert _key(ChatRequest([Message("assistant", "", tool_calls=(ToolCall("c1", "weather", live),))], "m")) != before


def test_two_react_branches_are_not_served_each_others_answer() -> None:
    """End to end through the real `Invoker`: the SF branch and the NYC branch
    of a ReAct loop each reach the LLM. Before the fix the second branch was
    answered from the first's cache entry without a provider call."""
    seen: list[str] = []

    class _LLM:
        async def stream(self, **kw: Any) -> Any:
            city = kw["messages"][1].tool_calls[0].arguments["city"]
            seen.append(city)
            for d in _deltas(f"it is sunny in {city}"):
                yield d

    def _deltas(text: str) -> list[Any]:
        from agentkit.kernel.types import Delta

        return [Delta(text=text, finish_reason="stop")]

    store = InMemoryStore()
    inv = Invoker(llm=_LLM(), chat_middleware=[memoize(store=store)])
    ctx = RunContext("run", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))

    async def go() -> tuple[Any, Any]:
        a = await inv.chat(ChatRequest(_react("SF"), "m"), ctx)
        b = await inv.chat(ChatRequest(_react("NYC"), "m"), ctx)
        return a, b

    a, b = asyncio.run(go())

    assert seen == ["SF", "NYC"], "one ReAct branch was served the other branch's cached answer"
    assert a.content == "it is sunny in SF"
    assert b.content == "it is sunny in NYC"


def test_an_identical_chat_turn_is_still_served_from_cache() -> None:
    """POSITIVE CONTROL, end to end: the same transcript twice must still make
    exactly ONE provider call and set `cache_hit`."""
    calls: list[int] = []

    class _LLM:
        async def stream(self, **kw: Any) -> Any:
            from agentkit.kernel.types import Delta

            calls.append(1)
            yield Delta(text="sunny", finish_reason="stop")

    store = InMemoryStore()
    inv = Invoker(llm=_LLM(), chat_middleware=[memoize(store=store)])
    ctx = RunContext("run", Scope(org_id=1), Budget(), Services(invoker=inv, store=store))

    async def go() -> tuple[Any, Any]:
        return (
            await inv.chat(ChatRequest(_react("SF"), "m"), ctx),
            await inv.chat(ChatRequest(_react("SF"), "m"), ctx),
        )

    a, b = asyncio.run(go())

    assert len(calls) == 1, "an identical chat turn was no longer memoized"
    assert a.content == b.content == "sunny"


# ── the chat key must survive a duck-typed tool call ───────────────────────


def test_the_chat_key_survives_an_openai_shaped_tool_call() -> None:
    """`_message_identity` reads everything through `getattr` because `m` and
    `tc` are duck-typed — a provider SDK object, a test double, a replayed
    record. One line broke that contract: a bare `dict(tc.arguments)`.

    The OpenAI wire shape carries `arguments` as the raw JSON STRING, and
    `dict('{"q": "hi"}')` raises `ValueError: dictionary update sequence
    element #0 has length 1; 2 is required`. So the single defensive-looking
    line was the only way this function could fail, and it failed on the most
    common foreign shape there is.
    """
    from dataclasses import dataclass

    @dataclass
    class _OpenAIShapedCall:
        id: str
        name: str
        arguments: str  # the wire format: a JSON string, not a mapping

    @dataclass
    class _ShapedMessage:
        role: str
        content: str
        tool_calls: tuple

    msg = _ShapedMessage(
        role="assistant",
        content="",
        tool_calls=(_OpenAIShapedCall("c1", "search", '{"q": "hi"}'),),
    )
    identity = _message_identity(msg)  # must not raise
    assert identity["tool_calls"][0]["arguments"] == '{"q": "hi"}', (
        "a string argument is passed through, not guessed at — parsing it would "
        "invent a normalisation this function has no business performing"
    )


def test_guarding_the_unwrap_did_not_change_any_existing_cache_key() -> None:
    """POSITIVE CONTROL, and the reason this change was safe to make.

    The unwrap never affected the key: `_stable_default` already normalises
    every mapping shape, so the unwrapped and raw forms hash identically for a
    plain dict, a `MappingProxyType` and a `ChainMap` alike. Guarding it
    therefore cannot re-key a single live cache entry — it only decides whether
    we crash. If this ever fails, someone has changed the key's shape and every
    persisted entry silently misses.
    """
    from collections import ChainMap
    from types import MappingProxyType

    from agentkit.kernel.resilience import stable_hash

    base = {"id": "c1", "name": "search"}
    for shape in ({"q": "hi"}, MappingProxyType({"q": "hi"}), ChainMap({"q": "hi"})):
        assert stable_hash({**base, "arguments": dict(shape)}) == stable_hash(
            {**base, "arguments": shape}
        )
