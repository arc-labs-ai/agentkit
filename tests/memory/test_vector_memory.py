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


# ── the chunk id survives the round trip ──────────────────────────────────


def test_query_carries_the_chunk_id_onto_the_item():
    """``Chunk.id`` was the one field the read adapter dropped: the write side
    has always keyed chunks by it, so the store knew the identity of every row
    it handed back and the item did not. That is the identity a
    ``CompositeMemory`` needs to notice that the journal and the vector store
    are returning the same fact — which is the normal case, since the journal
    is usually what the store was built from."""
    canned = [(0.91, Chunk(id="a", text="alpha", metadata={"tag": "x"}))]
    mem = VectorMemory(vector=_RecordingVector(hits=canned), name="kb")
    items = _run(mem.query("q", k=3, ctx=make_test_ctx(scope=Scope(1, 2))))
    assert [i.id for i in items] == ["a"]
    # NOT also smuggled into metadata: ``write`` pops ``metadata["id"]`` as the
    # chunk key, so a read-then-write round trip would otherwise depend on
    # which of the two copies of the id won.
    assert items[0].metadata == {"tag": "x"}


def test_write_keys_the_chunk_by_the_items_id_before_metadata_or_a_uuid():
    """Read-modify-write is the loop this closes: an item recalled from the
    store now carries its ``id``, and writing it back has to UPDATE that row
    rather than insert a second copy under a fresh uuid. ``metadata["id"]``
    stays supported underneath for callers who were setting it that way.

    The instance is named ``kb`` to match the items' ``source``: ``item.id`` is
    a provenance record, honoured only for items this instance handed out (see
    ``test_a_foreign_sources_id_never_addresses_this_stores_keyspace``), and
    ``query`` stamps ``source`` with ``name``, so that is what a recalled item
    actually looks like."""
    vec = _RecordingVector()
    mem = VectorMemory(vector=vec, name="kb")
    ctx = make_test_ctx(scope=Scope(1, 2))
    _run(mem.write([MemoryItem(content="c", source="kb", id="r7")], ctx=ctx))
    _run(mem.write([MemoryItem(content="c", source="kb", metadata={"id": "m9"})], ctx=ctx))
    _run(mem.write([MemoryItem(content="c", source="kb")], ctx=ctx))
    ids = [chunks[0].id for _scope, chunks in vec.upserts]
    assert ids[:2] == ["r7", "m9"]
    assert ids[2] not in ("r7", "m9")
    assert len(ids[2]) == 32  # uuid4().hex fallback


def test_a_vector_round_trip_updates_the_row_instead_of_duplicating_it():
    """End to end against the real in-memory adapter, because the uuid
    fallback made this silently wrong: recall a row, re-write it, and the store
    held TWO copies of one fact — which is precisely the duplicate the
    composite dedupe then has to clean up downstream."""
    vec = InMemoryVector()
    mem = VectorMemory(vector=vec, name="kb")
    ctx = make_test_ctx(scope=Scope(1, 2))
    _run(mem.write([MemoryItem(content="certs rotate every 90d", source="kb", id="r7")], ctx=ctx))
    recalled = _run(mem.query("certs", k=5, ctx=ctx))
    _run(mem.write(recalled, ctx=ctx))
    assert len(_run(mem.query("certs", k=5, ctx=ctx))) == 1


# ── item.id is provenance, not an address into THIS store's keyspace ────────


def test_a_foreign_sources_id_never_addresses_this_stores_keyspace():
    """The destructive half of keying chunks by ``item.id``.

    ``CompositeMemory.write`` BROADCASTS every item to every source, which is
    the documented write semantics. So an item recalled from a journal whose
    row key happens to be ``"3"`` reached ``VectorMemory.write`` and upserted
    over vector chunk ``"3"`` — an unrelated fact, silently destroyed, on the
    default write path. Ids are only unique WITHIN a backend; the composite's
    own ``dedupe="content"`` docstring makes exactly that point ("two stores
    can both call their first row ``1``").

    ``item.id`` is therefore trusted only when the item came from this
    instance (``item.source == self.name``), which is what ``query`` stamps —
    so read-modify-write still updates its row. A caller who genuinely means
    "write this to row X of this store" still says so with ``metadata["id"]``,
    which is an instruction rather than a provenance record."""
    vec = _RecordingVector()
    mem = VectorMemory(vector=vec, name="kb")
    ctx = make_test_ctx(scope=Scope(1, 2))
    _run(mem.write([MemoryItem(content="prod db is postgres 16", source="kb", id="3")], ctx=ctx))
    _run(mem.write([MemoryItem(content="user prefers dark mode", source="journal", id="3")], ctx=ctx))
    own_id, foreign_id = (chunks[0].id for _scope, chunks in vec.upserts)
    assert own_id == "3"
    assert foreign_id != "3"
    assert len(foreign_id) == 32  # uuid4().hex — inserted alongside, not over


def test_an_explicit_metadata_id_still_addresses_the_row_from_any_source():
    """The escape hatch the rule above leaves intact: ``metadata["id"]`` is the
    caller SAYING which row to write, so it outranks the missing provenance
    match and still updates in place rather than inserting a copy."""
    vec = InMemoryVector()
    mem = VectorMemory(vector=vec, name="kb")
    ctx = make_test_ctx(scope=Scope(1, 2))
    _run(mem.write([MemoryItem(content="stale", source="kb", id="r7")], ctx=ctx))
    _run(mem.write([MemoryItem(content="fresh", source="elsewhere", metadata={"id": "r7"})], ctx=ctx))
    out = _run(mem.query("stale fresh", k=10, ctx=ctx))
    assert [i.content for i in out] == ["fresh"]


def test_the_dedupe_stamp_is_not_persisted_as_record_metadata():
    """``dedupe_sources``/``dedupe_count`` describe ONE fan-out query, not the
    record. Persisting them made a later read look like the store itself was
    asserting corroboration, and — now that a merge absorbs an existing stamp
    rather than overwriting it — a written-back item would inflate its own
    count on every round trip. They are stripped at the persistence boundary,
    exactly as ``metadata["id"]`` is."""
    vec = _RecordingVector()
    mem = VectorMemory(vector=vec, name="kb")
    ctx = make_test_ctx(scope=Scope(1, 2))
    item = MemoryItem(
        content="certs rotate every 90d",
        source="kb",
        id="r7",
        metadata={"topic": "ops", "dedupe_sources": ["kb", "journal"], "dedupe_count": 2},
    )
    _run(mem.write([item], ctx=ctx))
    (_scope, chunks) = vec.upserts[0]
    assert chunks[0].metadata == {"topic": "ops"}
