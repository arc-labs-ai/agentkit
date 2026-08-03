"""`PgVectorStore` — durable `VectorPort` over Postgres + pgvector (extra: `agentkit[postgres]`).

The production counterpart to `InMemoryVector`, with the same `upsert`/`search` contract and the same
dependency-free, deterministic embedding (a feature-hashed, L2-normalized term-frequency vector) — so RAG
behaves identically offline and in production, but the index is durable and shared across processes.

Embeddings are fixed-dimension dense vectors stored in a `vector` column; `search` ranks by cosine distance
(`<=>`) and returns `(score, chunk)` where `score = 1 - distance`, scope-isolated and metadata-filterable.
Swap in a real embedding model by overriding `_embed` — the schema and queries are unchanged.

Call `await init()` once to create the extension, table, and index. Inject a `pool` for tests; else an
asyncpg pool is created from `dsn`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

from agentkit.kernel.types import Chunk, Scope

_TOKEN = re.compile(r"[a-z0-9]+")
_DIM = 256


def _embed(text: str, dim: int = _DIM) -> list[float]:
    """A deterministic, dependency-free embedding: term frequencies feature-hashed into `dim` dimensions and
    L2-normalized. Stable across processes (uses blake2b, not the salted built-in hash)."""
    vec = [0.0] * dim
    for term, tf in Counter(_TOKEN.findall(text.lower())).items():
        h = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += float(tf)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _literal(vec: list[float]) -> str:
    """pgvector text literal — sent as text and cast with `::vector`, so no asyncpg codec is required."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


class PgVectorStore:
    _DDL = (
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"CREATE TABLE IF NOT EXISTS agentkit_vectors (scope TEXT NOT NULL, id TEXT NOT NULL, "
        f"text TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{{}}', embedding vector({_DIM}) NOT NULL, "
        f"PRIMARY KEY (scope, id))",
        "CREATE INDEX IF NOT EXISTS agentkit_vectors_scope ON agentkit_vectors (scope)",
    )

    def __init__(self, dsn: str | None = None, *, pool: Any = None, dim: int = _DIM) -> None:
        self._dsn = dsn
        self._pool = pool
        self._dim = dim

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg  # type: ignore[import-not-found]  # the [postgres] extra, no py.typed marker

            self._pool = await asyncpg.create_pool(self._dsn)
        return self._pool

    async def init(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as con:
            for ddl in self._DDL:
                await con.execute(ddl)

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def upsert(self, scope: Scope, chunks: list[Chunk]) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as con:
            await con.executemany(
                "INSERT INTO agentkit_vectors (scope, id, text, metadata, embedding) "
                "VALUES ($1,$2,$3,$4,$5::vector) "
                "ON CONFLICT (scope, id) DO UPDATE SET text = EXCLUDED.text, "
                "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                [
                    (
                        scope.key(),
                        ch.id,
                        ch.text,
                        json.dumps(ch.metadata),
                        _literal(_embed(ch.text, self._dim)),
                    )
                    for ch in chunks
                ],
            )

    async def search(
        self, scope: Scope, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[tuple[float, Chunk]]:
        pool = await self._get_pool()
        qvec = _literal(_embed(query, self._dim))
        sql = (
            "SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS score "
            "FROM agentkit_vectors WHERE scope = $2"
        )
        params: list[Any] = [qvec, scope.key()]
        if where:
            params.append(json.dumps(where))
            sql += f" AND metadata @> ${len(params)}::jsonb"
        sql += " ORDER BY embedding <=> $1::vector LIMIT " + str(int(k))
        async with pool.acquire() as con:
            rows = await con.fetch(sql, *params)
        out: list[tuple[float, Chunk]] = []
        for r in rows:
            if r["score"] > 0:  # match InMemoryVector: drop non-overlapping hits
                meta = r["metadata"]
                out.append(
                    (
                        float(r["score"]),
                        Chunk(
                            r["id"], r["text"], json.loads(meta) if isinstance(meta, str) else meta
                        ),
                    )
                )
        return out
