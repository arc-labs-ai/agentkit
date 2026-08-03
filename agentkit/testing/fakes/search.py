"""`FakeSearch` — deterministic offline SearchPort for tests. Either a flat list of `SearchHit`s
returned for every query, or a dict `{query_substring: [SearchHit, ...]}` that picks the first
matching bucket — same ergonomics as `FakeLLM`."""

from __future__ import annotations

from typing import Any

from agentkit.kernel.ports import SearchHit


class FakeSearch:
    """Deterministic offline SearchPort. Pass a list (returned for every query) or a dict
    {substring_of_query: [SearchHit, ...]} (first substring match wins; unknown queries → [])."""

    def __init__(self, hits: list[SearchHit] | dict[str, list[SearchHit]]):
        self._hits = hits
        self.calls = 0

    def _select(self, query: str) -> list[SearchHit]:
        if isinstance(self._hits, dict):
            return next((v for k, v in self._hits.items() if k in query), [])
        return list(self._hits)

    async def search(
        self, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        self.calls += 1
        hits = self._select(query)
        if where is not None:
            hits = [h for h in hits if all(h.metadata.get(mk) == mv for mk, mv in where.items())]
        return hits[:k]
