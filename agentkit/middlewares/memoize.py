"""memoize middlewares — one family for exact reuse, idempotency, and near-duplicate caching.

- `memoize(key=…)`          — exact, single-flight, scope-keyed (chat or read-only tool reuse).
- `idempotent()`            — exact, keyed by (run, scope, tool, args), never caches a failure; wired
                              for side-effecting tools so an at-least-once retry doesn't re-fire.
- `semantic_memoize(…)`     — near-duplicate reuse over a VectorPort (read-only), thresholded.

All three are the same idea (skip `next` when a prior result is reusable); they differ only in how the
key/match is derived. Exact/idempotent ride `StorePort.get_or_set` (single-flight) — a raised producer is
never stored, so failures are never cached.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

from agentkit.kernel.middleware import Call, Handler, Middleware, _assemble, _result_to_stream
from agentkit.kernel.resilience import idempotency_key, stable_hash
from agentkit.kernel.resilience import stable_hash as _hash
from agentkit.kernel.types import Chunk, LLMResult, Usage


def _is_tool_call(call: Call) -> bool:
    """Is this unit of work a tool execution rather than a chat turn?

    ``call.kind`` is the authority, but ``default_key`` is also called
    directly in tests and by callers holding a bare object with only a
    ``request``, so fall back to the shape of the request itself: a
    ``ToolRequest`` has ``name`` + ``arguments``; a ``ChatRequest`` has
    neither."""
    kind = getattr(call, "kind", None)
    if kind is not None:
        return bool(kind == "tool")
    r = call.request
    return getattr(r, "name", None) is not None and hasattr(r, "arguments")


def default_key(call: Call) -> str:
    """The exact-match cache key for a unit of work: every field that changes the
    answer, and nothing else.

    Exists so ``memoize()`` works with no arguments. It previously required a
    ``key=`` callable, which pushed the single most dangerous decision in a
    cache — what counts as "the same call" — onto every caller, and the key the
    docs taught (``lambda c: c.request.messages[-1].content``) ignored the
    model, the tools, the temperature AND the tenant.

    Tools are reduced to their names: two registries advertising the same tools
    should share a cache entry, and a schema's description text changing does
    not change the answer.

    **Tool calls key on (tool name, arguments).** This module's own docstring
    advertises ``memoize`` for "read-only tool reuse", but the key hashed only
    chat fields — ``model`` / ``messages`` / ``tools`` / ``temperature`` /
    ``max_tokens`` / ``response_format`` — and a ``ToolRequest`` has *none* of
    them, so every tool call in a scope hashed to the SAME key. Measured:
    ``weather(SF)`` ran once, then ``stock(AAPL)`` and ``weather(NYC)`` both
    returned ``{'city': 'SF'}`` and the stock tool never executed at all. A
    cache that returns one tool's answer for another tool is worse than no
    cache. The ``memo:tool:`` prefix keeps the two key spaces disjoint so a
    tool key can never alias a chat key either.
    """
    r = call.request
    if _is_tool_call(call):
        return "memo:tool:" + stable_hash(
            {"tool": getattr(r, "name", None), "arguments": getattr(r, "arguments", None)},
            length=24,
        )
    return "memo:" + stable_hash(
        {
            "model": getattr(r, "model", None),
            "messages": [
                (getattr(m, "role", ""), getattr(m, "content", "")) for m in getattr(r, "messages", []) or []
            ],
            "tools": sorted(getattr(t, "name", "") for t in (getattr(r, "tools", None) or ())),
            "response_format": getattr(r, "response_format", None),
            "temperature": getattr(r, "temperature", None),
            "max_tokens": getattr(r, "max_tokens", None),
        },
        length=24,
    )


def _scoped(call: Call, raw: str) -> str:
    """Namespace a cache key by tenant.

    The framework's own ``Scope`` docstring calls itself the key "threaded
    through every memory recall / cache key / meter / callback" — but
    ``memoize`` took an arbitrary ``key`` callable and added nothing, so
    isolation depended on every caller remembering. They did not: the key in
    the cheatsheet and the LangChain migration guide was the last message's
    text, which meant two tenants asking the same question shared one cached
    LLM response. Measured: tenant 999 received tenant 1's answer, one provider
    call for both.

    Partitioning here rather than documenting harder is the point. A
    tenant-isolation boundary that relies on a caller-supplied key is not a
    boundary. A caller who genuinely wants cross-tenant sharing uses a
    scope-less store or a shared ``Scope``, which is an explicit act.
    """
    return f"{call.ctx.scope.key()}:{raw}"


def _emit_cache_hit(call: Call, key: str) -> None:
    """Drop a ``cache.hit`` event on the currently-open span — best-effort.

    Fired when ``memoize`` short-circuits to a stored result so the trace timeline
    explains why this call has no provider span underneath it. A misbehaving tracer
    must never break the cache path, so the call is wrapped in ``contextlib.suppress``."""
    trace = getattr(call.ctx, "trace", None)
    if trace is None:
        return
    with contextlib.suppress(Exception):
        trace.add_event_to_current_span("cache.hit", cache_key=key)


def memoize(
    *,
    key: Callable[[Call], str] = default_key,
    store: Any = None,
    ttl: int | None = None,
    when: Callable[[Call], bool] | None = None,
) -> Middleware:
    """Single-flight, never-cache-failure reuse over `StorePort.get_or_set`: the producer
    *collects* the inner stream to the assembled result and stores it; a hit re-emits the stored result as
    a one-shot stream. Memoized calls are therefore not incremental — the correctness of single-flight +
    a raised producer never being stored wins over incremental delivery for a cache layer.

    Every key is namespaced by ``ctx.scope.key()`` before it reaches the store,
    so a cache entry can never cross a tenant boundary regardless of what
    ``key`` returns — see :func:`_scoped`. ``key`` defaults to
    :func:`default_key`, an exact match over the fields that change the answer.
    """

    async def mw(call: Call, nxt: Handler) -> AsyncIterator[Any]:
        if (when is not None and not when(call)) or (store or call.ctx.store) is None:
            async for x in nxt(call):
                yield x
            return
        s = store or call.ctx.store
        ran = {"v": False}

        async def produce() -> Any:
            ran["v"] = True  # runs only on a MISS
            return _assemble(
                call, [x async for x in nxt(call)]
            )  # a raised stream propagates → unstored

        cache_key = _scoped(call, key(call))
        result = await s.get_or_set(cache_key, produce, ttl=ttl)
        if not ran["v"]:
            call.meta["cache_hit"] = True  # signal a hit to outer middlewares (audit → "deduped")
            _emit_cache_hit(call, cache_key)  # trace timeline: explain the missing provider span
        for item in _result_to_stream(call, result):
            yield item

    return mw


def idempotent(*, store: Any = None) -> Middleware:
    """Dedupe SIDE-EFFECTING tool calls on (run, scope, tool, args); a failure is never stored."""

    def key(call: Call) -> str:
        r = call.request
        return idempotency_key(call.ctx.correlation_id, call.ctx.scope.key(), r.name, r.arguments)

    return memoize(key=key, store=store, when=lambda c: getattr(c.request, "side_effecting", False))


def semantic_memoize(
    *,
    vector: Any = None,
    threshold: float = 0.85,
    text: Callable[[Call], str] | None = None,
    when: Callable[[Call], bool] | None = None,
) -> Middleware:
    """Near-duplicate reuse for READ-ONLY calls over a VectorPort (scored search), scope-isolated.
    NEVER attach to a side-effecting call — masking a real action behind a hit would be a correctness
    bug, so guard with `when=` (defaults to chat calls / explicitly read-only)."""
    _text = text or (
        lambda c: getattr(c.request, "messages", None) and c.request.messages[-1].content or ""
    )

    async def mw(call: Call, nxt: Handler) -> AsyncIterator[Any]:
        if (when is not None and not when(call)) or (vector or call.ctx.vector) is None:
            async for x in nxt(call):
                yield x
            return
        vec = vector or call.ctx.vector
        query = _text(call)
        hits = await vec.search(call.ctx.scope, query, k=1)
        if (
            hits and hits[0][0] >= threshold
        ):  # hit → reconstruct a result, re-stream (no new tokens)
            md = hits[0][1].metadata
            call.meta["cache_hit"] = True
            _emit_cache_hit(call, hits[0][1].id)
            cached = LLMResult(
                content=md.get("content", ""),
                model=md.get("model"),
                provider="semantic-cache",
                finish_reason="cached",
                usage=Usage(),
            )
            for it in _result_to_stream(call, cached):
                yield it
            return
        items = []  # miss → tee (incremental) while collecting to cache
        async for x in nxt(call):
            items.append(x)
            yield x
        result = _assemble(call, items)
        content = getattr(result, "content", None)  # only cache LLM-shaped, serializable results
        # Only a FINAL, textual answer is reusable. The condition used to be
        # ``content is not None``, which happily stored a tool-REQUESTING turn:
        # such a turn assembles to ``content == ""`` with ``tool_calls`` set, and
        # the hit path above reconstructs an ``LLMResult`` from ``metadata`` only
        # — ``tool_calls`` is not stored, so it cannot be restored. Measured: a
        # ReAct turn cached on call 1 (``tool_calls=(weather(SF),) content=''``)
        # came back on call 2 as ``tool_calls=() content=''``, i.e. the loop
        # silently returned an empty answer instead of calling the tool.
        # Both halves matter: an empty ``content`` is not an answer, and a turn
        # carrying ``tool_calls`` is an instruction to act, not a result to reuse.
        if content and not getattr(result, "tool_calls", None):
            cid = _hash(query)
            await vec.upsert(
                call.ctx.scope,
                [
                    Chunk(
                        id=cid,
                        text=query,
                        metadata={"content": content, "model": getattr(result, "model", None)},
                    )
                ],
            )

    return mw
