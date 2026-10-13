"""SQL vector binding, row hydration and acquisition cleanup boundaries."""

import asyncio
import json

import pytest

from agentkit.core.contracts.types import Scope
from agentkit.state.vectors.pgvector import PgVectorStore


class RowPool:
    def __init__(self, rows, error=None):
        self.rows, self.error = rows, error
        self.requests = []
        self.exits = []

    def acquire(self):
        pool = self

        class Connection:
            async def fetch(self, sql, *args):
                pool.requests.append((sql, args))
                if pool.error:
                    raise pool.error
                return pool.rows

        class Acquired:
            async def __aenter__(self):
                return Connection()

            async def __aexit__(self, typ, exc, traceback):
                pool.exits.append(typ)
                return False

        return Acquired()


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [False, True])
async def test_pgvector_search_hydrates_json_and_mapping_metadata_without_zero_similarity_hits(encoded):
    metadata = {"approved": True, "nested": {"owners": ["host"]}}
    pool = RowPool([
        {"id": "match", "text": "approved policy", "score": .75,
         "metadata": json.dumps(metadata) if encoded else metadata},
        {"id": "zero", "text": "unrelated", "score": 0, "metadata": {}},
        {"id": "negative", "text": "unrelated", "score": -.1, "metadata": {}},
    ])
    hits = await PgVectorStore(pool=pool, dim=8).search(Scope(3, 7), "approved", k=3)
    assert len(hits) == 1 and hits[0][0] == .75
    assert hits[0][1].id == "match" and hits[0][1].metadata == metadata
    with pytest.raises(TypeError):
        hits[0][1].metadata["nested"]["owners"].append("caller")
    assert pool.exits == [None]


@pytest.mark.asyncio
async def test_pgvector_scope_and_adversarial_filter_values_are_parameters_not_query_text():
    pool = RowPool([])
    needle = "'); DROP TABLE agentkit_vectors; --"
    where = {"owner": needle, "approved": True}
    await PgVectorStore(pool=pool, dim=8).search(Scope(4, 9), needle, k=2, where=where)
    sql, arguments = pool.requests[0]
    assert "WHERE scope = $2" in sql and "metadata @> $3::jsonb" in sql
    assert sql.endswith("LIMIT 2") and needle not in sql
    assert arguments[1] == Scope(4, 9).key() and json.loads(arguments[2]) == where


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ConnectionError("backend unreachable"), asyncio.CancelledError()])
async def test_pgvector_query_failure_or_cancellation_propagates_and_exits_connection_scope(error):
    pool = RowPool([], error=error)
    with pytest.raises(type(error)):
        await PgVectorStore(pool=pool).search(Scope(1, 1), "query")
    assert pool.exits == [type(error)]
