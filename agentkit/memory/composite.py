"""Composite memory shapes — combine many ``MemorySource``s as one.

Two compositions cover the practical cases:

- ``CompositeMemory`` — parallel fan-out + merge + optional rerank. The
  default when "ask everywhere and rank the union" is right.
- ``SequentialMemory`` — try sources in order, stop at first hit (or
  short-circuit when k items are collected). The right shape when a
  cheap cache should answer before an expensive vector lookup.

Both implement the ``MemorySource`` Protocol, so they nest arbitrarily —
a ``CompositeMemory`` of two ``SequentialMemory`` chains is a valid
"two-arm rerank" topology.

Write semantics: ``CompositeMemory.write`` broadcasts to every source;
``SequentialMemory.write`` writes to the FIRST source (writes happen at
the cache tier in the typical setup). Backends that are read-only
(``ToolMemory``) ignore writes via a default no-op.

Partial-write failures surface a :class:`CompositeWriteError` naming
which backends accepted the write and which failed. A naive
``gather(return_exceptions=False)`` would propagate the first raising
source verbatim — the caller would have no signal that another source's
write committed. ``CompositeMemory.write`` instead collects per-source
outcomes and raises only when at least one failed, with the full split
attached.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem, MemorySource, Reranker, score_sort_rerank


class CompositeWriteError(RuntimeError):
    """At least one source in a ``CompositeMemory.write`` fan-out failed.

    Carries a structured per-source split so callers can compensate
    (retry the failed sources, or surface the partial-commit state to
    the operator). Raising the first child exception verbatim would
    discard everything else and postmortems would lose the information
    about which backends DID commit.

    Attributes:
        accepted: list of source names whose ``write`` succeeded.
        failed: dict mapping source name → BaseException raised by that
            source.
    """

    def __init__(self, *, accepted: list[str], failed: dict[str, BaseException]) -> None:
        self.accepted = accepted
        self.failed = failed
        failed_summary = ", ".join(
            f"{name}: {type(exc).__name__}: {exc}" for name, exc in failed.items()
        )
        super().__init__(
            f"CompositeMemory.write partial failure — "
            f"accepted: {accepted or '(none)'}; failed: {failed_summary}"
        )


@dataclass(slots=True)
class CompositeMemory:
    """Parallel fan-out across many sources.

    Each query runs against every source concurrently. Results are
    merged, then optionally reranked. Top-k is taken from the reranked
    list. The default reranker is ``score_sort_rerank`` (sort by score
    desc); pass a cross-encoder or LLM judge for richer semantics.

    Backpressure: ``asyncio.gather`` is bounded by Python's default
    semaphore behaviour — for very wide trees, prefer nesting
    ``CompositeMemory(CompositeMemory(...), ...)`` to keep fanout
    moderate at each level.
    """

    sources: list[MemorySource]
    reranker: Reranker | None = None
    name: str = field(default="composite")

    async def query(
        self, query: str, *, k: int, ctx: Ctx, where: dict[str, Any] | None = None
    ) -> list[MemoryItem]:
        if not self.sources:
            return []
        # Fan out concurrently; each source returns at most k items, so
        # the merged pool is at most ``len(sources) * k``. The reranker
        # narrows to k.
        batches = await asyncio.gather(
            *(s.query(query, k=k, ctx=ctx, where=where) for s in self.sources)
        )
        merged: list[MemoryItem] = [item for batch in batches for item in batch]
        rerank = self.reranker.rerank if self.reranker is not None else score_sort_rerank
        return await rerank(query, merged, k=k)

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        """Broadcast writes to every source. A backend that can't accept
        writes implements ``write`` as a no-op (see ``ToolMemory``).

        Partial failures surface as :class:`CompositeWriteError` with a
        per-source accepted/failed split. Without ``return_exceptions=True``
        the first child exception would propagate and the rest would be
        discarded, so a caller seeing a failure would have no way to know
        which backends had already committed."""
        materialised = list(items)
        if not materialised:
            return
        # Run every write to completion. Each source's outcome lands in
        # the results list (either ``None`` for success or the raised
        # exception); the wave never short-circuits on the first failure
        # so a slow backend isn't blamed for an early one's crash.
        outcomes = await asyncio.gather(
            *(s.write(materialised, ctx=ctx) for s in self.sources),
            return_exceptions=True,
        )
        accepted: list[str] = []
        failed: dict[str, BaseException] = {}
        for source, outcome in zip(self.sources, outcomes, strict=True):
            name = getattr(source, "name", source.__class__.__name__)
            if isinstance(outcome, BaseException):
                failed[name] = outcome
            else:
                accepted.append(name)
        if failed:
            raise CompositeWriteError(accepted=accepted, failed=failed)


@dataclass(slots=True)
class SequentialMemory:
    """Try sources in order; stop when ``k`` items are collected.

    Use for cache-then-fallback patterns: a fast in-memory source first,
    an expensive vector source second. The second source isn't hit if
    the first answered. Items returned in the order sources were
    declared (preserves the "cache hit beats fresh fetch" intent).
    """

    sources: list[MemorySource]
    name: str = field(default="sequential")

    async def query(
        self, query: str, *, k: int, ctx: Ctx, where: dict[str, Any] | None = None
    ) -> list[MemoryItem]:
        collected: list[MemoryItem] = []
        for s in self.sources:
            if len(collected) >= k:
                break
            remaining = k - len(collected)
            collected.extend(await s.query(query, k=remaining, ctx=ctx, where=where))
        return collected[:k]

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        """Write to the FIRST source only — typical cache-tier pattern.
        Downstream sources are read-only from this composite's POV."""
        materialised = list(items)
        if not materialised or not self.sources:
            return
        await self.sources[0].write(materialised, ctx=ctx)


__all__ = ["CompositeMemory", "CompositeWriteError", "SequentialMemory"]
