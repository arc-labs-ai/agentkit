"""`semantic_memoize` must never cache a tool-REQUESTING turn.

The store condition was `result.content is not None`. A turn where the model
asks for a tool assembles to `content == ""` with `tool_calls` set, so it
qualified — and the hit path rebuilds an `LLMResult` from the chunk's metadata
only, which never carried `tool_calls`.

Measured before the fix: call 1 returned
`tool_calls=(weather(SF),) content=''`, call 2 returned `tool_calls=() content=''`.
A ReAct loop reading that answers with an empty string instead of calling the
tool — a silent wrong answer, not a crash.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agentkit.kernel.types import ChatRequest, Chunk, Delta, Message, Scope, ToolCall, Usage
from agentkit.middlewares import semantic_memoize
from agentkit.runtime import Budget, Invoker, RunContext, Services


class _ExactVector:
    """A VectorPort whose "similarity" is exact text equality scored 1.0 —
    enough to exercise the hit/miss paths without a real embedder."""

    def __init__(self) -> None:
        self.rows: list[Chunk] = []

    async def search(self, scope: Any, query: str, k: int = 1) -> list[tuple[float, Chunk]]:
        return [(1.0, c) for c in self.rows if c.text == query][:k]

    async def upsert(self, scope: Any, chunks: list[Chunk]) -> None:
        self.rows.extend(chunks)


class _ScriptedLLM:
    """Replays a list of delta-lists, one list per call."""

    def __init__(self, turns: list[list[Delta]]) -> None:
        self._turns = turns
        self.calls = 0

    async def stream(self, **_kw: Any):
        turn = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        for d in turn:
            yield d


TOOL_TURN = [
    Delta(tool_calls=(ToolCall(id="1", name="weather", arguments={"city": "SF"}),), model="m", provider="p"),
    Delta(finish_reason="tool_calls", usage=Usage(5, 5, 0.0), model="m", provider="p"),
]
ANSWER_TURN = [
    Delta(text="It is sunny.", model="m", provider="p"),
    Delta(finish_reason="stop", usage=Usage(5, 5, 0.0), model="m", provider="p"),
]
EMPTY_TURN = [Delta(finish_reason="stop", usage=Usage(1, 0, 0.0), model="m", provider="p")]


def _wire(llm: _ScriptedLLM, vec: _ExactVector) -> tuple[Invoker, RunContext]:
    inv = Invoker(llm=llm, chat_middleware=[semantic_memoize(vector=vec)])
    ctx = RunContext("run", Scope(), Budget(), Services(invoker=inv, vector=vec))
    return inv, ctx


REQ = ChatRequest([Message("user", "weather in SF?")], "m")


def test_a_tool_requesting_turn_is_never_cached() -> None:
    """The load-bearing test: the second identical turn must still reach the
    provider and still carry its `tool_calls`."""
    vec, llm = _ExactVector(), _ScriptedLLM([TOOL_TURN, TOOL_TURN])
    inv, ctx = _wire(llm, vec)

    async def go() -> tuple[Any, Any]:
        return await inv.chat(REQ, ctx), await inv.chat(REQ, ctx)

    first, second = asyncio.run(go())

    assert [t.name for t in first.tool_calls] == ["weather"]
    assert [t.name for t in second.tool_calls] == ["weather"], "the cached replay dropped the tool call"
    assert vec.rows == [], "a tool-requesting turn was written to the semantic cache"
    assert llm.calls == 2


def test_an_empty_answer_is_never_cached() -> None:
    """The other half of the condition: `content == ""` is not an answer, so
    caching it would serve an empty string forever."""
    vec, llm = _ExactVector(), _ScriptedLLM([EMPTY_TURN, ANSWER_TURN])
    inv, ctx = _wire(llm, vec)

    async def go() -> tuple[Any, Any]:
        return await inv.chat(REQ, ctx), await inv.chat(REQ, ctx)

    first, second = asyncio.run(go())

    assert first.content == ""
    assert second.content == "It is sunny.", "an empty reply was cached and replayed"
    assert llm.calls == 2


def test_a_real_answer_IS_cached_and_reused() -> None:
    """POSITIVE CONTROL. A "fix" that stopped caching altogether would pass both
    tests above; it fails here. A final textual answer is exactly what
    `semantic_memoize` exists to reuse."""
    vec, llm = _ExactVector(), _ScriptedLLM([ANSWER_TURN])
    inv, ctx = _wire(llm, vec)

    async def go() -> tuple[Any, Any]:
        return await inv.chat(REQ, ctx), await inv.chat(REQ, ctx)

    first, second = asyncio.run(go())

    assert first.content == second.content == "It is sunny."
    assert llm.calls == 1, "a reusable answer was not cached"
    assert [c.metadata["content"] for c in vec.rows] == ["It is sunny."]
    assert second.provider == "semantic-cache"


def test_a_tool_turn_does_not_block_caching_the_answer_that_follows() -> None:
    """The realistic ReAct sequence: turn 1 asks for a tool (not cached),
    turn 2 answers (cached). The cache must end up holding the ANSWER."""
    vec, llm = _ExactVector(), _ScriptedLLM([TOOL_TURN, ANSWER_TURN, ANSWER_TURN])
    inv, ctx = _wire(llm, vec)

    async def go() -> list[Any]:
        return [await inv.chat(REQ, ctx) for _ in range(3)]

    a, b, c = asyncio.run(go())

    assert [t.name for t in a.tool_calls] == ["weather"]
    assert b.content == "It is sunny."
    assert c.content == "It is sunny."
    assert llm.calls == 2, "the third call should have been served from the cache"
    assert [r.metadata["content"] for r in vec.rows] == ["It is sunny."]
