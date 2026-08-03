"""VectorMemory — the canonical ``MemorySource`` over a ``VectorPort``.

Replaces the old ``Retriever`` capability. Same wire (a ``VectorPort``)
and the same scope-isolated read/write contract, but exposed through
the unified ``MemorySource`` Protocol so vector lookup composes with
the other memory shapes (sequential cache+fallback, fan-out composite,
scope-enforcing decorator) instead of living off to the side as its
own capability.

The "memory vs RAG" distinction that ``Retriever`` documented is still
purely a metadata filter at the seam (``where=`` at write time + at
query time). The mechanism — scope-restricted upsert + scope-restricted
search + adapter to the uniform ``MemoryItem`` shape — is identical
for both.

Formatting moves OUT of this class. ``Retriever.context_block`` is
gone; consumers (the ``Grounder`` callable on ``RequestBuilder``, or
the cognition itself) format the returned ``MemoryItem``s as they see
fit. The Protocol returns data, not strings.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Chunk


@dataclass(slots=True)
class VectorMemory:
    """A ``MemorySource`` backed by a ``VectorPort``.

    ``where`` is the default metadata filter applied to every query
    and (informationally) attached at write time only if the caller
    embeds it in ``MemoryItem.metadata``. At query time, a call-time
    ``where=`` merges over the constructor default — call-time wins
    on key collisions, so a wired-in narrow filter can be widened or
    overridden per-query without rebuilding the source.

    Tenant isolation is delegated to the underlying ``VectorPort.upsert``
    / ``VectorPort.search``, which both take ``ctx.scope`` as their
    bucket key. Wrap with ``ScopedMemory`` for fail-loud enforcement
    at the framework boundary.
    """

    vector: Any  # VectorPort (Protocol — duck-typed to avoid the kernel import here)
    where: dict[str, Any] | None = None
    name: str = field(default="vector")

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Top-k scope-restricted lookup. The call-time ``where``
        merges over the constructor default (call-time wins). The
        underlying ``VectorPort.search`` returns ``(score, Chunk)``;
        we adapt each pair to a ``MemoryItem`` and stamp ``source``
        with this instance's ``name`` so downstream consumers can
        attribute the result."""
        from agentkit.memory.base import MemoryItem  # local to avoid cycles

        merged = self._merge_where(where)
        hits = await self.vector.search(ctx.scope, query, k, merged)
        return [
            MemoryItem(
                content=chunk.text,
                source=self.name,
                score=score,
                metadata=dict(chunk.metadata),
            )
            for score, chunk in hits
        ]

    async def write(self, items: Iterable[Any], *, ctx: Ctx) -> None:
        """Index a batch of ``MemoryItem``s. Each item becomes a
        ``Chunk`` keyed by ``metadata["id"]`` when the caller wrote
        one, else a fresh uuid. ``content`` becomes ``Chunk.text``;
        the rest of ``metadata`` is preserved so callers can later
        ``where=``-filter on whatever tags they wrote."""
        chunks: list[Chunk] = []
        for item in items:
            metadata = dict(item.metadata) if item.metadata else {}
            chunk_id = metadata.pop("id", None) or uuid.uuid4().hex
            chunks.append(Chunk(id=chunk_id, text=item.content, metadata=metadata))
        if not chunks:
            return
        await self.vector.upsert(ctx.scope, chunks)

    def _merge_where(self, call_where: dict[str, Any] | None) -> dict[str, Any] | None:
        """Combine the constructor default with the call-time filter;
        call-time keys WIN on collision. ``None`` on both sides means
        no filter at all."""
        if call_where is None:
            return dict(self.where) if self.where is not None else None
        if self.where is None:
            return dict(call_where)
        merged = dict(self.where)
        merged.update(call_where)
        return merged


__all__ = ["VectorMemory"]
