"""``CachedMemory`` — TTL-bounded cache over any ``MemorySource``.

These tests pin:

1. Identical query within TTL hits the cache (inner source called once).
2. Different ``query`` / different ``k`` / different ``where`` are all
   separate cache keys (inner source called per distinct combination).
3. After the TTL expires, the next query misses and re-invokes inner.
4. ``write`` invalidates the cache; the next read re-invokes inner.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.memory import CachedMemory, MemoryItem
from agentkit.memory import decorators as memdec
from agentkit.testing import make_test_ctx


def _run(coro):
    return asyncio.run(coro)


# ── fakes ─────────────────────────────────────────────────────────────────


@dataclass
class _CountingMemory:
    """Records every query / write so we can assert on cache hit/miss counts."""

    items: list[MemoryItem] = field(default_factory=list)
    query_calls: int = 0
    write_calls: int = 0
    name: str = "inner"

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del query, ctx, where
        self.query_calls += 1
        return list(self.items[:k])

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        del items, ctx
        self.write_calls += 1


# ── tests ─────────────────────────────────────────────────────────────────


def test_repeated_query_within_ttl_hits_cache():
    """Same (query, k, where) within the TTL → inner called exactly once."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=60.0)
    ctx = make_test_ctx()

    a = _run(mem.query("q", k=3, ctx=ctx))
    b = _run(mem.query("q", k=3, ctx=ctx))

    assert a == b
    assert inner.query_calls == 1


def test_different_query_misses_cache():
    """A different query string is a different cache key."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=60.0)
    ctx = make_test_ctx()

    _run(mem.query("q1", k=3, ctx=ctx))
    _run(mem.query("q2", k=3, ctx=ctx))

    assert inner.query_calls == 2


def test_different_k_misses_cache():
    """Different ``k`` → different cache key."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner)
    ctx = make_test_ctx()

    _run(mem.query("q", k=3, ctx=ctx))
    _run(mem.query("q", k=5, ctx=ctx))

    assert inner.query_calls == 2


def test_different_where_misses_cache():
    """Different ``where`` filter → different cache key."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner)
    ctx = make_test_ctx()

    _run(mem.query("q", k=3, ctx=ctx, where={"site": "a.com"}))
    _run(mem.query("q", k=3, ctx=ctx, where={"site": "b.com"}))
    _run(mem.query("q", k=3, ctx=ctx, where={"site": "a.com"}))  # cached

    assert inner.query_calls == 2


def test_expired_entry_misses(monkeypatch):
    """After the TTL elapses (simulated by monkeypatching ``time.monotonic``),
    the next read misses and re-invokes the inner source."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=1.0)
    ctx = make_test_ctx()

    # control the monotonic clock via the module the decorator imports
    now = [1000.0]
    monkeypatch.setattr(memdec.time, "monotonic", lambda: now[0])

    _run(mem.query("q", k=3, ctx=ctx))
    assert inner.query_calls == 1

    # advance past the TTL
    now[0] += 2.0
    _run(mem.query("q", k=3, ctx=ctx))
    assert inner.query_calls == 2


def test_write_invalidates_cache():
    """Any ``write`` drops the cache so the next ``query`` hits inner anew."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=60.0)
    ctx = make_test_ctx()

    _run(mem.query("q", k=3, ctx=ctx))
    _run(mem.query("q", k=3, ctx=ctx))  # cached
    assert inner.query_calls == 1

    _run(mem.write([MemoryItem(content="new", source="x")], ctx=ctx))
    assert inner.write_calls == 1

    _run(mem.query("q", k=3, ctx=ctx))  # now a miss
    assert inner.query_calls == 2


# ── Cross-tenant isolation ──────────────────────────────────────────────────
#
# The cache key includes the scope (tenant identity). Without it, a
# stack of ``ScopedMemory(CachedMemory(Inner))`` would run tenant B's
# ``enforce`` check successfully, then serve tenant A's items from the
# cache — a silent cross-tenant read.


from agentkit.kernel.types import Scope


def test_cache_is_partitioned_by_scope() -> None:
    """Two runs with different tenant scopes share the cache instance
    but see distinct partitions. Even if every other cache-key axis
    (query / k / where) is identical, tenant B never sees tenant A's
    items."""

    tenant_a_items = [MemoryItem(content="A's data", source="inner")]
    tenant_b_items = [MemoryItem(content="B's data", source="inner")]

    @dataclass
    class _PerScopeMemory:
        """Returns different items depending on the caller's tenant scope."""

        query_calls: int = 0
        name: str = "inner"

        async def query(self, query, *, k, ctx, where=None):
            self.query_calls += 1
            org = getattr(getattr(ctx, "scope", None), "org_id", None)
            return list(tenant_a_items if org == 1 else tenant_b_items)

        async def write(self, items, *, ctx):
            return None

    inner = _PerScopeMemory()
    mem = CachedMemory(inner=inner, ttl_seconds=60.0)

    ctx_a = make_test_ctx(scope=Scope(org_id=1, domain_id=1))
    ctx_b = make_test_ctx(scope=Scope(org_id=2, domain_id=1))

    result_a = _run(mem.query("q", k=1, ctx=ctx_a))
    result_b = _run(mem.query("q", k=1, ctx=ctx_b))

    # Each tenant sees its own data, and both fetches hit the inner
    # source (distinct cache partitions).
    assert result_a[0].content == "A's data"
    assert result_b[0].content == "B's data"
    assert inner.query_calls == 2

    # A repeat query from tenant A hits its cache partition — no
    # duplicated inner fetch.
    _run(mem.query("q", k=1, ctx=ctx_a))
    assert inner.query_calls == 2


def test_cache_accepts_unhashable_where_values() -> None:
    """``where`` may contain lists or nested dicts — canonical JSON
    serialization handles either, so a legitimate filter like
    ``{"tags": ["security", "finance"]}`` doesn't crash the cache
    lookup."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=60.0)
    ctx = make_test_ctx()

    # First call: unhashable ``list`` value in the where dict must not
    # raise. The cache produces a valid key and stores the result.
    _run(mem.query("q", k=3, ctx=ctx, where={"tags": ["security", "finance"]}))
    assert inner.query_calls == 1

    # A repeat query with the same where (list order preserved) is a hit.
    _run(mem.query("q", k=3, ctx=ctx, where={"tags": ["security", "finance"]}))
    assert inner.query_calls == 1

    # A different list content is a miss — the JSON serializer produces
    # a different key.
    _run(mem.query("q", k=3, ctx=ctx, where={"tags": ["security", "compliance"]}))
    assert inner.query_calls == 2


def test_cached_result_is_isolated_from_caller_mutation() -> None:
    """A caller that mutates the returned list — sorts it, dedupes it,
    appends to it — must not corrupt the cache. Returning the same
    reference on hits would let the caller poison every subsequent
    hit inside the TTL window."""
    inner = _CountingMemory(
        items=[
            MemoryItem(content="first", source="inner"),
            MemoryItem(content="second", source="inner"),
            MemoryItem(content="third", source="inner"),
        ]
    )
    mem = CachedMemory(inner=inner, ttl_seconds=60.0)
    ctx = make_test_ctx()

    a = _run(mem.query("q", k=3, ctx=ctx))
    assert len(a) == 3
    # Caller mutates the returned list — remove all items.
    a.clear()

    # The next cached read (within the TTL) must still see 3 items.
    b = _run(mem.query("q", k=3, ctx=ctx))
    assert len(b) == 3
    # And the two returned lists are independent objects — not the
    # same reference into the cache.
    assert a is not b


def test_cache_key_stable_across_where_dict_ordering() -> None:
    """Two ``where`` dicts with the same entries in different insertion
    orders share a cache key — otherwise a caller who happened to
    build the filter with a different order would silently double-
    fetch."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=60.0)
    ctx = make_test_ctx()

    _run(mem.query("q", k=3, ctx=ctx, where={"a": 1, "b": 2, "c": 3}))
    _run(mem.query("q", k=3, ctx=ctx, where={"c": 3, "b": 2, "a": 1}))

    assert inner.query_calls == 1  # second call was a cache hit


# ── scope-missing defensive path ─────────────────────────────────────────
#
# Without a scope on ctx, every tenant would collapse onto the same
# empty cache key — silently crossing tenants on hits. Default
# behavior warns loudly; ``strict_scope=True`` upgrades that to a
# hard ``ValueError`` so misconfigured wiring fails fast.

import warnings as _warnings  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402


def test_cached_memory_warns_when_scope_missing() -> None:
    """Default (non-strict) behavior: a scope-less ctx produces a
    ``UserWarning`` so the misconfiguration is visible, but the query
    still proceeds. The warning message names the required fix so
    grep-based debugging lands on it."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=60.0)
    ctx = SimpleNamespace()  # no ``scope`` attribute at all

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _run(mem.query("q", k=3, ctx=ctx))

    matching = [w for w in caught if "requires a scoped RunContext" in str(w.message)]
    assert matching, f"expected a scope-missing warning; got {caught!r}"
    # And the inner source was still called — non-strict is warn-and-continue.
    assert inner.query_calls == 1


def test_cached_memory_strict_scope_raises_when_scope_missing() -> None:
    """Opt-in strict mode: a scope-less ctx is a hard error, not a
    warning. The inner source must NOT have been touched."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=60.0, strict_scope=True)
    ctx = SimpleNamespace()

    with pytest.raises(ValueError, match="requires a scoped RunContext"):
        _run(mem.query("q", k=3, ctx=ctx))
    assert inner.query_calls == 0


def test_cached_memory_strict_scope_passes_when_scope_present() -> None:
    """Happy path for strict mode: a ctx with a real ``scope.key()``
    proceeds normally — the guard only fires on the missing-scope
    edge case."""
    inner = _CountingMemory(items=[MemoryItem(content="x", source="inner")])
    mem = CachedMemory(inner=inner, ttl_seconds=60.0, strict_scope=True)
    ctx = make_test_ctx(scope=Scope(org_id=1, domain_id=2))

    out = _run(mem.query("q", k=3, ctx=ctx))
    assert [i.content for i in out] == ["x"]
    assert inner.query_calls == 1
