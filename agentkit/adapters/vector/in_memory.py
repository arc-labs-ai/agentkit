"""VectorPort adapter — scope-isolated TF-cosine stand-in (offline Fake + contract reference).

`search` returns `(score, chunk)` so callers can threshold: RAG ignores the score, the semantic-cache
middleware uses it. Production adapter is pgvector (real embeddings + hybrid retrieval) of the same
shape.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from agentkit.kernel.types import Chunk, Scope

_TOKEN = re.compile(r"[a-z0-9]+")


def _vec(text: str) -> dict[str, float]:
    """L2-normalized term-frequency vector — a deterministic embedding stand-in."""
    tf = Counter(_TOKEN.findall(text.lower()))
    norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
    return {t: v / norm for t, v in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
    return sum(w * larger.get(t, 0.0) for t, w in smaller.items())


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    return where is None or all(metadata.get(k) == v for k, v in where.items())


class InMemoryVector:
    """VectorPort — scope-isolated, cosine over TF vectors, metadata-filtered, scored (tests/lean)."""

    def __init__(self) -> None:
        self._store: dict[str, list[tuple[Chunk, dict[str, float]]]] = defaultdict(list)

    async def upsert(self, scope: Scope, chunks: list[Chunk]) -> None:
        # Dedupe the INCOMING batch first, not just against what is already
        # stored. The old code only rewrote the bucket when ``ch.id`` was in
        # the pre-call id set, so an id appearing twice within a single
        # ``upsert`` call was appended twice — measured: one call with
        # ``Chunk("doc1", ...)`` twice, then ``search`` returned two hits both
        # with id ``doc1``. "Upsert by id" has to hold per call, not just
        # across calls, or a chunker that emits a repeated id corrupts the
        # index on first ingest. Last write wins, matching pgvector's
        # ``ON CONFLICT ... DO UPDATE``.
        incoming: dict[str, Chunk] = {ch.id: ch for ch in chunks}
        if not incoming:
            return
        bucket = self._store[scope.key()]
        bucket[:] = [(c, v) for c, v in bucket if c.id not in incoming]  # idempotent upsert by id
        bucket.extend((ch, _vec(ch.text)) for ch in incoming.values())

    async def search(
        self, scope: Scope, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[tuple[float, Chunk]]:
        qv = _vec(query)
        scored = [
            (_cosine(qv, vec), ch)
            for ch, vec in self._store.get(scope.key(), [])  # ← isolation: only this tenant
            if _matches(ch.metadata, where)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s, ch) for s, ch in scored[:k] if s > 0]
