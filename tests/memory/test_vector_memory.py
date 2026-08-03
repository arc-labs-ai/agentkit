"""``VectorMemory`` — the canonical ``MemorySource`` over a ``VectorPort``.

These tests pin the contract:

1. ``query`` forwards ``(ctx.scope, query, k, where)`` to ``VectorPort.search``
   and adapts each ``(score, Chunk)`` to a ``MemoryItem`` stamped with the
   source's ``name``.
2. The constructor's default ``where`` filter merges with the call-time
   ``where``; call-time keys win on collision.
3. ``write`` converts each ``MemoryItem`` to a ``Chunk`` (id from
   ``metadata["id"]`` or generated) and calls ``VectorPort.upsert``.
4. ``ctx.scope`` is honoured end-to-end — two tenants do not see each
   other's data via the same ``VectorMemory`` instance.
"""

from __future__ import annotations

import asyncio

from agentkit.adapters.vector import InMemoryVector
from agentkit.kernel.types import Chunk, Scope
from agentkit.memory import MemoryItem, VectorMemory
from agentkit.testing import make_test_ctx


def _run(coro):
    return asyncio.run(coro)


# ── recording fake VectorPort ─────────────────────────────────────────────


class _RecordingVector:
    """Captures the (scope, query, k, where) of every call so the test can
    assert the exact forwarding. Returns canned hits."""

    def __init__(self, hits: list[tuple[float, Chunk]] | None = None) -> None:
        self.hits = hits or []
        self.searches: list[tuple[Scope, str, int, dict | None]] = []
        self.upserts: list[tuple[Scope, list[Chunk]]] = []

    async def search(
        self, scope: Scope, query: str, k: int = 5, where: dict | None = None
    ) -> list[tuple[float, Chunk]]:
        self.searches.append((scope, query, k, where))
        return self.hits

    async def upsert(self, scope: Scope, chunks: list[Chunk]) -> None:
        self.upserts.append((scope, list(chunks)))


# ── query forwards + adapts ───────────────────────────────────────────────


def test_query_forwards_scope_query_k_where_and_adapts_results():
    """The Protocol contract: VectorMemory.query is a thin adapter over
    VectorPort.search — same scope, same query string, same k, the
    merged where filter. Each (score, Chunk) becomes one MemoryItem
    with ``source`` stamped from the instance's ``name``."""
    canned = [
        (0.91, Chunk(id="a", text="alpha", metadata={"tag": "x"})),
        (0.42, Chunk(id="b", text="beta", metadata={"tag": "y"})),
    ]
    vec = _RecordingVector(hits=canned)
    mem = VectorMemory(vector=vec, name="kb")
    ctx = make_test_ctx(scope=Scope(1, 2))

    items = _run(mem.query("the query", k=3, ctx=ctx, where={"tag": "x"}))

    assert vec.searches == [(Scope(1, 2), "the query", 3, {"tag": "x"})]
    assert [i.content for i in items] == ["alpha", "beta"]
    assert all(i.source == "kb" for i in items)
    assert items[0].score == 0.91
    assert items[1].score == 0.42
    assert items[0].metadata == {"tag": "x"}
    assert items[1].metadata == {"tag": "y"}


# ── constructor where merges with call-time where ─────────────────────────


def test_default_where_merges_with_call_time_where_call_wins_on_collision():
    """The constructor's ``where`` is the wired-in narrow filter (e.g.
    ``{"kind": "memory"}``). A per-call ``where=`` widens or overrides
    it. On a key collision the call site wins — the caller's explicit
    intent overrides the wired default."""
    vec = _RecordingVector()
    mem = VectorMemory(vector=vec, where={"kind": "memory", "tag": "default"})
    ctx = make_test_ctx(scope=Scope(1, 2))

    _run(mem.query("q", k=2, ctx=ctx, where={"tag": "override", "extra": True}))

    (_scope, _q, _k, where) = vec.searches[0]
    assert where == {"kind": "memory", "tag": "override", "extra": True}


def test_default_where_alone_is_forwarded_when_no_call_time_where():
    """When the caller doesn't pass ``where=``, only the constructor's
    default applies — but it's still forwarded (copied) to the
    VectorPort, not dropped."""
    vec = _RecordingVector()
    mem = VectorMemory(vector=vec, where={"kind": "memory"})
    _run(mem.query("q", k=2, ctx=make_test_ctx(scope=Scope(1, 2))))
    assert vec.searches[0][3] == {"kind": "memory"}


def test_no_where_anywhere_forwards_none():
    """No constructor default + no call-time filter ⇒ ``where=None`` at
    the VectorPort. The adapter treats that as "match everything"."""
    vec = _RecordingVector()
    mem = VectorMemory(vector=vec)
    _run(mem.query("q", k=2, ctx=make_test_ctx(scope=Scope(1, 2))))
    assert vec.searches[0][3] is None


# ── write converts + upserts ──────────────────────────────────────────────


def test_write_converts_memory_items_to_chunks_and_upserts():
    """A MemoryItem's ``content`` becomes ``Chunk.text``; an ``id`` in
    its metadata becomes the chunk's id (and is popped out so it's not
    duplicated on the chunk's metadata). Other metadata keys ride
    through unchanged."""
    vec = _RecordingVector()
    mem = VectorMemory(vector=vec)
    ctx = make_test_ctx(scope=Scope(1, 2))

    _run(
        mem.write(
            [
                MemoryItem(content="A", source="x", metadata={"id": "id-A", "k": 1}),
                MemoryItem(content="B", source="x", metadata={"k": 2}),
            ],
            ctx=ctx,
        )
    )

    assert len(vec.upserts) == 1
    (scope, chunks) = vec.upserts[0]
    assert scope == Scope(1, 2)
    assert [c.text for c in chunks] == ["A", "B"]
    assert chunks[0].id == "id-A"
    assert chunks[0].metadata == {"k": 1}  # ``id`` popped, ``k`` preserved
    # second chunk got a generated id (no ``id`` in its metadata)
    assert chunks[1].id and chunks[1].id != "id-A"
    assert chunks[1].metadata == {"k": 2}


def test_write_with_no_items_is_a_noop():
    """Empty write must not touch the underlying VectorPort — the
    in-memory adapter would allocate an empty bucket for the scope."""
    vec = _RecordingVector()
    mem = VectorMemory(vector=vec)
    _run(mem.write([], ctx=make_test_ctx(scope=Scope(1, 2))))
    assert vec.upserts == []


# ── end-to-end against the real InMemoryVector adapter ────────────────────


def test_scope_is_respected_via_underlying_vector_port():
    """Run against the real InMemoryVector to confirm tenant isolation
    actually holds — writes in scope A are invisible from scope B even
    when both queries go through the same VectorMemory instance."""
    v = InMemoryVector()
    mem = VectorMemory(vector=v)

    ctx_a = make_test_ctx(scope=Scope(1, 1), vector=v)
    ctx_b = make_test_ctx(scope=Scope(2, 2), vector=v)

    _run(
        mem.write(
            [MemoryItem(content="alpha fact", source="agent", metadata={"id": "1"})],
            ctx=ctx_a,
        )
    )

    items_a = _run(mem.query("alpha", k=3, ctx=ctx_a))
    items_b = _run(mem.query("alpha", k=3, ctx=ctx_b))

    assert items_a and "alpha" in items_a[0].content
    assert items_b == []
