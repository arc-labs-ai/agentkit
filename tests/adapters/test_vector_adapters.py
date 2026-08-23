"""VectorPort adapters: `PgVectorStore` schema/embedding agreement, and `InMemoryVector`'s
upsert-by-id contract.

`PgVectorStore` is exercised through a recording fake pool — the DDL/SQL it emits is the whole
contract under test, so no live Postgres is needed."""

import asyncio

import pytest

from agentkit.adapters.vector import InMemoryVector
from agentkit.adapters.vector.pgvector import _DIM, PgVectorStore, _embed
from agentkit.kernel.types import Chunk, Scope


def _run(coro):
    return asyncio.run(coro)


class _RecordingPool:
    """Records every statement executed, so `init`/`upsert` can be asserted
    offline. Mirrors the asyncpg surface `PgVectorStore` uses."""

    def __init__(self):
        self.executed: list[str] = []
        self.batches: list[tuple[str, list]] = []

    def acquire(self):
        pool = self

        class _Con:
            async def execute(self, sql, *args):
                pool.executed.append(sql)

            async def executemany(self, sql, args):
                pool.batches.append((sql, list(args)))

            async def fetch(self, sql, *args):
                return []

        class _CM:
            async def __aenter__(self):
                return _Con()

            async def __aexit__(self, *exc):
                return False

        return _CM()

    async def close(self):
        return None


# ── PgVectorStore: the table and the vectors must be the same width ─────────
#
# ``_DDL`` was a class-level constant interpolating the module default
# ``_DIM`` (256) while ``upsert`` wrote ``_embed(text, self._dim)``. So the
# ``dim=`` constructor argument built a table that could never accept its own
# writes — Postgres rejects a 64-dim vector into a ``vector(256)`` column.


@pytest.mark.parametrize("dim", [_DIM, 64, 1, 1536])
def test_pgvector_ddl_column_width_matches_the_embeddings_it_writes(dim):
    """The regression, stated as the invariant it broke: whatever width the
    DDL declares, ``upsert`` must produce vectors of exactly that width.
    Measured before the fix: ``PgVectorStore(dim=64)`` reported ``_dim=64``
    and embedded 64 floats, while the DDL still said ``vector(256)``."""
    pool = _RecordingPool()
    store = PgVectorStore(pool=pool, dim=dim)
    _run(store.init())

    create = next(s for s in pool.executed if "CREATE TABLE" in s)
    assert f"embedding vector({dim}) NOT NULL" in create

    _run(store.upsert(Scope(1, 1), [Chunk("a", "pricing and billing", {})]))
    _, rows = pool.batches[0]
    literal = rows[0][-1]
    assert literal.count(",") + 1 == dim  # the vector literal is exactly `dim` wide
    assert len(_embed("pricing and billing", dim)) == dim


def test_pgvector_default_dim_is_still_256():
    """POSITIVE CONTROL: dropping the dimension from the DDL entirely (or
    defaulting it to something else) would satisfy the parametrized test's
    "they agree" property while silently changing the shipped schema."""
    pool = _RecordingPool()
    _run(PgVectorStore(pool=pool).init())
    assert "embedding vector(256) NOT NULL" in next(s for s in pool.executed if "CREATE TABLE" in s)


def test_pgvector_init_still_creates_the_extension_and_index():
    """The DDL moved from a class constant to a per-instance property — the
    other two statements must survive the move."""
    pool = _RecordingPool()
    _run(PgVectorStore(pool=pool, dim=8).init())
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in pool.executed)
    assert any("agentkit_vectors_scope" in s for s in pool.executed)
    assert len(pool.executed) == 3


@pytest.mark.parametrize("dim", [0, -1])
def test_pgvector_rejects_a_nonpositive_dim(dim):
    """Edge: ``vector(0)`` is not valid DDL and ``_embed`` would index into an
    empty list. Fail at construction, where the caller can see it, rather than
    at the first ``init()`` against a real server."""
    with pytest.raises(ValueError, match="dim must be >= 1"):
        PgVectorStore(pool=_RecordingPool(), dim=dim)


# ── InMemoryVector: upsert-by-id holds WITHIN a call, not just across ────────


def test_inmemory_vector_dedupes_a_repeated_id_inside_one_upsert():
    """The old loop only rewrote the bucket when the id was in the
    PRE-CALL id set, so a new id appearing twice in the same batch was
    appended twice. Measured: ``search`` returned two hits both with id
    ``doc1``. Last write wins, matching pgvector's ``ON CONFLICT DO UPDATE``."""
    v = InMemoryVector()
    scope = Scope(1, 1)
    _run(v.upsert(scope, [Chunk("doc1", "alpha beta", {}), Chunk("doc1", "alpha gamma", {})]))
    hits = _run(v.search(scope, "alpha", k=10))
    assert [c.id for _, c in hits] == ["doc1"]
    assert hits[0][1].text == "alpha gamma"


def test_inmemory_vector_keeps_distinct_ids_from_one_upsert():
    """POSITIVE CONTROL: a "fix" that collapsed the batch to a single chunk,
    or dropped every repeat regardless of id, would fail here."""
    v = InMemoryVector()
    scope = Scope(1, 1)
    _run(
        v.upsert(
            scope,
            [Chunk("a", "alpha one", {}), Chunk("b", "alpha two", {}), Chunk("c", "alpha three", {})],
        )
    )
    assert sorted(c.id for _, c in _run(v.search(scope, "alpha", k=10))) == ["a", "b", "c"]


def test_inmemory_vector_upsert_across_calls_still_replaces():
    """Existing contract, unchanged by the batch dedupe: re-upserting an id in
    a LATER call replaces the stored chunk rather than duplicating it."""
    v = InMemoryVector()
    scope = Scope(1, 1)
    _run(v.upsert(scope, [Chunk("doc1", "alpha old", {"v": 1})]))
    _run(v.upsert(scope, [Chunk("doc1", "alpha new", {"v": 2})]))
    hits = _run(v.search(scope, "alpha", k=10))
    assert len(hits) == 1 and hits[0][1].metadata == {"v": 2}


def test_inmemory_vector_empty_upsert_touches_nothing():
    """Edge: an empty batch must not wipe the scope — and must not
    materialize a bucket for a scope that has never been written."""
    v = InMemoryVector()
    scope = Scope(1, 1)
    _run(v.upsert(scope, [Chunk("doc1", "alpha", {})]))
    _run(v.upsert(scope, []))
    assert len(_run(v.search(scope, "alpha", k=10))) == 1
    _run(v.upsert(Scope(9, 9), []))
    assert Scope(9, 9).key() not in v._store


def test_inmemory_vector_repeated_id_stays_scope_isolated():
    """Edge: deduping is per-scope. The same id in another tenant's scope is a
    different document and must survive untouched."""
    v = InMemoryVector()
    _run(v.upsert(Scope(1, 1), [Chunk("doc1", "alpha tenant one", {})]))
    _run(v.upsert(Scope(2, 2), [Chunk("doc1", "alpha two", {}), Chunk("doc1", "alpha three", {})]))
    assert _run(v.search(Scope(1, 1), "alpha", k=5))[0][1].text == "alpha tenant one"
    assert len(_run(v.search(Scope(2, 2), "alpha", k=5))) == 1
