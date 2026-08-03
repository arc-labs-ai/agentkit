"""MemorySource Protocol — the single seam for "what can this agent reach for?"

One Protocol, many backends (vector, key-value, files, journal, tool-wrapped,
composite). Adding a new memory backend is implementing this Protocol; no
Agent subclassing, no parallel class hierarchy.

The shape is deliberately narrow:

- ``query(query, k, ctx)`` — ask for items relevant to ``query``.
- ``write(items, ctx)`` — index items for later recall (no-op when the
  backend is read-only, e.g. a wrapped search tool).

Every backend returns the same ``MemoryItem`` shape regardless of native
representation — the cognition that consumes results never sees backend
specifics. This is the Adapter discipline applied at the Protocol boundary.

The Protocol coexists with ``Tool`` (LLM-decided side effects) — they
overlap (a web-search tool is also a knowledge probe) but answer different
questions: tools are what the *model* decides to call mid-loop; memory is
what the *cognition* fetches around an LLM call to ground reasoning. The
``ToolMemory`` adapter bridges the two when a single backend serves both.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentkit.kernel.protocols import Ctx


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """The uniform result shape every ``MemorySource`` returns.

    ``content`` is what the agent reads (already-formatted text).
    ``source`` names the backend that produced it (the ``MemorySource.name``)
    so consumers can attribute results and cognitions can format citations.
    ``score`` is the backend's relevance signal (cosine similarity for
    vector search, recency for journal, etc.); ``None`` when the backend
    doesn't rank.
    ``metadata`` carries backend-specific extras (chunk id, file path,
    timestamp) that the cognition or the rendering layer may consult.
    """

    content: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemorySource(Protocol):
    """A queryable source of information the agent can reach for.

    ``name`` is the stable diagnostic label — stamped on returned
    ``MemoryItem.source``, also used for trace attribution. Implementations
    SHOULD set a sensible class-level default.

    Implementations MUST honour ``ctx.scope`` if the underlying backend
    crosses tenants. Multi-tenant enforcement is the implementation's job
    OR a ``ScopedMemory`` decorator's job; the Protocol doesn't enforce it
    itself.
    """

    name: str

    async def query(
        self, query: str, *, k: int, ctx: Ctx, where: dict[str, Any] | None = None
    ) -> list[MemoryItem]: ...

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    """Optional second pass over a set of merged ``MemoryItem``s.

    A composite memory may pull from many sources, each returning items
    scored on its own scale (vector cosine vs lexical match vs recency).
    A Reranker scores them on a common axis so the top-k cut is sensible.

    The default reranker is ``score_sort_rerank`` — sort by ``score`` desc
    with ``None`` last. Real applications wire a cross-encoder or an LLM
    judge here.
    """

    async def rerank(self, query: str, items: list[MemoryItem], *, k: int) -> list[MemoryItem]: ...


async def score_sort_rerank(query: str, items: list[MemoryItem], *, k: int) -> list[MemoryItem]:
    """Default: stable sort by score descending. ``None`` scores sink to
    the bottom but aren't dropped — a tool-wrapped source that doesn't
    rank still gets to surface in the top-k."""
    del query
    scored = [
        (i.score if i.score is not None else float("-inf"), idx, i) for idx, i in enumerate(items)
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [i for _s, _idx, i in scored[:k]]


__all__ = [
    "MemoryItem",
    "MemorySource",
    "Reranker",
    "score_sort_rerank",
]
