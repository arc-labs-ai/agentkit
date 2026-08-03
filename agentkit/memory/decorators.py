"""Memory decorators — composable wrappers over any ``MemorySource``.

Each decorator IS a ``MemorySource`` (structurally), so they nest
arbitrarily. The discipline is single-responsibility: ``ScopedMemory``
enforces tenant boundaries, ``CompactedMemory`` shrinks results,
``CachedMemory`` short-circuits identical queries. Stack them in the
order that matches your concerns.
"""

from __future__ import annotations

import contextlib
import json
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message
from agentkit.memory.base import MemoryItem, MemorySource


def _default_enforce(scope: Any) -> bool:
    """Default policy: both tenant axes (``org_id`` AND ``domain_id``)
    must be populated. A query against an unscoped run is a programmer
    error — refuse loudly rather than silently leak across tenants.

    Explicit ``is not None`` checks — a valid tenant ID may be ``0``
    (e.g. the ``root`` org or the ``default`` domain), and ``bool(0)``
    would misclassify it as unscoped.
    """
    if scope is None:
        return False
    return (
        getattr(scope, "org_id", None) is not None and getattr(scope, "domain_id", None) is not None
    )


@dataclass(slots=True)
class ScopedMemory:
    """Fail-loud multi-tenant guard around any ``MemorySource``.

    Every ``query`` and every ``write`` runs ``enforce(ctx)`` first.
    If ``enforce`` is ``None``, the default check fires — both
    ``ctx.scope.org_id`` and ``ctx.scope.domain_id`` must be set.

    A custom ``enforce`` callable lets callers pin a stricter policy
    (e.g. "must match this specific tenant", "must be inside an
    approved checkpoint"). Returning a falsy value OR raising any
    exception inside ``enforce`` results in a ``PermissionError`` — the
    inner source is NOT touched.

    The decorator's ``name`` mirrors the wrapped source so downstream
    consumers attributing ``MemoryItem.source`` see the same label
    they'd see without the wrapper — the guard is invisible on the
    happy path.
    """

    inner: MemorySource
    enforce: Callable[[Ctx], Any] | None = None
    name: str = field(default="")

    def __post_init__(self) -> None:
        # Mirror the inner source's diagnostic label so attribution
        # stays stable when callers wrap/unwrap the guard.
        if not self.name:
            self.name = getattr(self.inner, "name", "scoped")

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        self._check(ctx)
        return await self.inner.query(query, k=k, ctx=ctx, where=where)

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        self._check(ctx)
        # Materialize once so we can (a) forward a list to the inner
        # source and (b) report ``n`` on the observation without
        # re-iterating an exhausted generator.
        materialized = list(items)
        await self.inner.write(materialized, ctx=ctx)
        # Best-effort observability — a memory write is the kind of
        # side-effect operators want to see on the run's timeline
        # (e.g. "the Researcher just cached 12 items to the vector
        # store"). ``RunContext.emit`` is already defensive; the guard
        # here is only for ctx stubs that don't expose ``emit``.
        emit = getattr(ctx, "emit", None)
        if emit is not None:
            with contextlib.suppress(Exception):
                await emit(
                    "memory.written",
                    payload={"n": len(materialized), "source": self.name},
                )

    def _check(self, ctx: Ctx) -> None:
        # Custom ``enforce`` receives the whole ``ctx`` (its typed contract);
        # the default policy only reads the scope, so we hand it the scope
        # directly — that lets tests exercise ``_default_enforce(scope)`` in
        # isolation and keeps the default's responsibility crisp.
        try:
            if self.enforce is not None:
                ok = self.enforce(ctx)
            else:
                ok = _default_enforce(getattr(ctx, "scope", None))
        except Exception as exc:  # enforce raised → translate to PermissionError
            raise PermissionError(
                f"ScopedMemory: enforce check raised on {self.inner!r}: {exc}"
            ) from exc
        if not ok:
            raise PermissionError(
                f"ScopedMemory: enforce check failed on {self.inner!r} "
                f"(scope={getattr(ctx, 'scope', None)!r})"
            )


@dataclass(slots=True)
class CompactedMemory:
    """Decorator: shrink each ``MemoryItem.content`` via a ``Compactor``.

    Used when raw chunks would blow the prefix budget. Adapts the
    framework's ``Compactor`` capability (which operates on
    ``list[Message]``) to operate on individual items: each item's
    content is wrapped as a single message, compacted, and the result
    becomes the new content.

    ``max_items`` caps the post-compaction list — useful when the
    inner source returns more items than the cognition wants to fold
    into the prompt.
    """

    inner: MemorySource
    compactor: Any  # Compactor capability
    max_items: int | None = None
    name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.inner, "name", "compacted")

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        items = await self.inner.query(query, k=k, ctx=ctx, where=where)
        compacted: list[MemoryItem] = []
        for item in items:
            messages = [Message("user", item.content)]
            shrunk = await self.compactor.compact(messages, ctx)
            text = "\n".join(m.content for m in shrunk if m.content)
            compacted.append(
                MemoryItem(
                    content=text or item.content,
                    source=item.source,
                    score=item.score,
                    metadata=item.metadata,
                )
            )
        if self.max_items is not None:
            return compacted[: self.max_items]
        return compacted

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        await self.inner.write(items, ctx=ctx)


@dataclass(slots=True)
class CachedMemory:
    """Decorator: cache query → results with a TTL.

    The cache key is ``(query, k, frozenset(where or {}))``. Hits
    served without touching the inner source. Writes invalidate the
    cache (best-effort — a write may surface stale results to an
    in-flight query, which is the typical "read-after-write
    eventual consistency" trade-off).

    ``ttl_seconds`` is wall-clock monotonic — the cache never serves
    an entry older than this. ``max_entries`` caps the cache size;
    eviction is LRU-ish (oldest insertion drops when full).

    Not thread-safe; agents are single-flow per loop.
    """

    inner: MemorySource
    ttl_seconds: float = 60.0
    max_entries: int = 256
    name: str = field(default="")
    strict_scope: bool = False
    _cache: dict[tuple[Any, ...], tuple[float, list[MemoryItem]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.inner, "name", "cached")

    @staticmethod
    def _key(query: str, k: int, where: dict[str, Any] | None, scope_key: str) -> tuple[Any, ...]:
        # Two invariants on the cache key:
        #
        # 1. Include ``scope_key`` — the tenant axis. Without it,
        #    stacking ``ScopedMemory(CachedMemory(Inner))`` would serve
        #    tenant A's items to tenant B on a cache hit (their
        #    ``enforce`` pass runs, but the underlying items come from
        #    A's cached fetch).
        # 2. Serialize ``where`` via canonical JSON, not
        #    ``frozenset(...items())``. A ``where={"tags":
        #    ["security","finance"]}`` breaks the set fallback because
        #    lists aren't hashable. JSON dump handles nested collections
        #    and stays deterministic across processes.
        where_key = json.dumps(where or {}, sort_keys=True, default=str)
        return (scope_key, query, k, where_key)

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        scope_key = getattr(getattr(ctx, "scope", None), "key", lambda: "")()
        if not scope_key:
            # No scope on ctx (or an empty ``key()``) → the cache would
            # bucket every tenant under the same empty key, silently
            # crossing tenants on hits. In strict mode this is a hard
            # error; otherwise warn once so the drift is at least visible.
            msg = (
                "CachedMemory requires a scoped RunContext — "
                "wrap with ScopedMemory or provide ctx.scope"
            )
            if self.strict_scope:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
        key = self._key(query, k, where, scope_key)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None:
            ts, items = cached
            if now - ts < self.ttl_seconds:
                # Return a defensive copy. Without it, a caller that
                # extends / sorts / dedupes the returned list would
                # corrupt every subsequent hit against the same key for
                # the TTL window.
                return list(items)
            # expired — drop and miss-through
            del self._cache[key]
        items = await self.inner.query(query, k=k, ctx=ctx, where=where)
        if len(self._cache) >= self.max_entries:
            # Drop the oldest entry (insertion-order dicts give us FIFO).
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        # Store a defensive copy so post-fetch mutation by an outer
        # decorator (rerank/dedup) doesn't leak into the cache either.
        self._cache[key] = (now, list(items))
        return items

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        self._cache.clear()
        await self.inner.write(items, ctx=ctx)


__all__ = ["CachedMemory", "CompactedMemory", "ScopedMemory"]
