"""A `memoize()`d chat must return the same TYPED output as an un-memoized one.

The user-visible shape of the `_result_to_stream` field drop. `memoize()` never returns the inner
stream: it collects `next` to an `LLMResult` (to store it) and re-expresses that result as a
one-shot stream via `_result_to_stream` — on the HIT path *and* on the MISS path. That copy carried
six of `LLMResult`'s seven fields and dropped `parsed`, the typed object `output_coerce()` exists to
produce, so putting a cache in front of a structured chat silently deleted the structure.

Measured before the fix, `Invoker(chat_middleware=[memoize(), output_coerce()])` with an
`output=Plan` adapter and a store wired, the same request twice::

    llm.calls    : 1                                        (the cache worked)
    call 1 parsed: None                                     (expected Plan(subject='ship', …))
    call 2 parsed: None
    call 1 content: '{"subject": "ship", "steps": ["a","b"]}'   (content was fine — hence silent)

After the fix both calls return `Plan(subject='ship', steps=['a','b'])` and `llm.calls` is still 1.

`store=InMemoryStore()` on the ctx is load-bearing: `memoize()` with no store passes straight
through to `next` and caches NOTHING, so the whole reproduction quietly evaporates and the test
passes against the broken code. `test_the_store_is_actually_caching` pins that so this file cannot
rot into a no-op.

Chain ORDER is load-bearing too: `[memoize(), output_coerce()]` puts memoize OUTSIDE, so the parse
happens inside the cached region and the typed object has to survive the cache. Reversing it hides
the bug — `output_coerce()` would simply re-parse the replayed content every time.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentkit.adapters.store.memory import InMemoryStore
from agentkit.capabilities.output_schema import adapt
from agentkit.kernel.types import ChatRequest, Message, ToolRequest
from agentkit.middlewares import memoize, output_coerce
from agentkit.runtime import Invoker
from agentkit.testing import FakeLLM, make_test_ctx

pydantic = pytest.importorskip("pydantic")


class Plan(pydantic.BaseModel):
    subject: str
    steps: list[str]


VALID = '{"subject": "ship", "steps": ["a", "b"]}'


def _wire(response: str = VALID) -> tuple[FakeLLM, Invoker, Any]:
    """A structured chat behind a cache: memoize OUTERMOST, a real store on the ctx."""
    llm = FakeLLM(response)
    inv = Invoker(llm=llm, chat_middleware=[memoize(), output_coerce()])
    ctx = make_test_ctx(invoker=inv, store=InMemoryStore())
    return llm, inv, ctx


def _request() -> ChatRequest:
    # A fresh, EQUAL request each time — `default_key` hashes the request's fields, so two identical
    # requests are one cache entry. Reusing the same object would also hide a key that keyed on
    # identity rather than content.
    return ChatRequest(messages=[Message("user", "plan the release")], model="m")


def _twice(inv: Invoker, ctx: Any, *, adapter: Any) -> tuple[Any, Any]:
    async def go() -> tuple[Any, Any]:
        meta = {"output_adapter": adapter} if adapter is not None else None
        first = await inv.chat(_request(), ctx, meta=dict(meta) if meta else None)
        second = await inv.chat(_request(), ctx, meta=dict(meta) if meta else None)
        return first, second

    return asyncio.run(go())


# ── the bug ────────────────────────────────────────────────────────────────────────────────


def test_a_cached_chat_returns_the_same_typed_output_as_the_first_call() -> None:
    """The load-bearing case: the second (CACHED) call must not be a different answer."""
    llm, inv, ctx = _wire()

    first, second = _twice(inv, ctx, adapter=adapt(Plan))

    assert llm.calls == 1, "the second call was not served from the cache — nothing was proven"
    assert isinstance(first.parsed, Plan), "the typed output was lost before it ever hit the cache"
    assert isinstance(second.parsed, Plan), "a cache HIT returned parsed=None"
    assert second.parsed == first.parsed, "the cached call returned a different typed object"
    assert second.parsed.subject == "ship" and second.parsed.steps == ["a", "b"]


def test_the_cached_result_is_otherwise_identical() -> None:
    """`parsed` was the field that drifted; assert the WHOLE result matches so a future copy that
    fixes `parsed` by rewriting the rebuild cannot lose something else on the way."""
    _, inv, ctx = _wire()
    first, second = _twice(inv, ctx, adapter=adapt(Plan))
    assert second == first


def test_the_store_is_actually_caching() -> None:
    """The trap this file is built to avoid: `memoize()` without a store is a pass-through, and a
    reproduction that never caches passes against broken code. Measured: with `store=None` the
    FakeLLM is entered TWICE, so `llm.calls` is the honest witness that the hit path ran at all."""
    llm_cached, inv_cached, ctx_cached = _wire()
    _twice(inv_cached, ctx_cached, adapter=adapt(Plan))

    llm_uncached = FakeLLM(VALID)
    inv_uncached = Invoker(llm=llm_uncached, chat_middleware=[memoize(), output_coerce()])
    ctx_uncached = make_test_ctx(invoker=inv_uncached)  # no store → memoize is a no-op
    _twice(inv_uncached, ctx_uncached, adapter=adapt(Plan))

    assert llm_cached.calls == 1
    assert llm_uncached.calls == 2, "no-store memoize was expected to cache nothing"


# ── positive controls (pass BEFORE and AFTER the fix) ──────────────────────────────────────


def test_an_unstructured_cached_chat_is_unaffected() -> None:
    """No adapter → `parsed` is `None` on both calls and the text answer still round-trips through
    the cache. The fix must not invent a typed object where the caller asked for none."""
    llm, inv, ctx = _wire("just prose")

    first, second = _twice(inv, ctx, adapter=None)

    assert llm.calls == 1
    assert first.content == second.content == "just prose"
    assert first.parsed is None and second.parsed is None
    assert second.usage == first.usage and second.finish_reason == first.finish_reason


def test_a_cached_tool_call_is_unaffected() -> None:
    """A tool result is an opaque payload, not an `LLMResult` — `_result_to_stream` passes it
    through verbatim, so the replay path for tools must be byte-identical to the first call."""

    class _ReadOnlyTool:
        side_effecting = False

        def __init__(self) -> None:
            self.runs = 0

        async def run(self, arguments: dict[str, Any], ctx: Any) -> dict[str, Any]:
            self.runs += 1
            return {"tool": "lookup", "n": self.runs, **arguments}

    tool = _ReadOnlyTool()
    inv = Invoker(llm=None, tool_middleware=[memoize()])
    ctx = make_test_ctx(invoker=inv, store=InMemoryStore())

    async def go() -> tuple[Any, Any]:
        req = lambda: ToolRequest("lookup", {"id": 7}, tool)  # noqa: E731
        return await inv.invoke_tool(req(), ctx), await inv.invoke_tool(req(), ctx)

    first, second = asyncio.run(go())

    assert tool.runs == 1  # the cache hit really happened
    assert first == second == {"tool": "lookup", "n": 1, "id": 7}
