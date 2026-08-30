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
from agentkit.memory.composite import DEDUPE_COUNT_KEY, DEDUPE_SOURCES_KEY


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
        attribute the result.

        ``Chunk.id`` rides along on ``MemoryItem.id``. It used to be the one
        field the adapter dropped, which was an odd asymmetry — ``write`` has
        always keyed chunks by it, so the store knew the identity of every row
        it handed back and the item did not. Two consequences followed:
        ``CompositeMemory`` could not tell that the journal and the vector
        store had returned the same fact (the normal case, since the journal is
        usually what the store was built from), and writing a recalled item
        back inserted a SECOND copy under a fresh uuid instead of updating the
        row."""
        from agentkit.memory.base import MemoryItem  # local to avoid cycles

        merged = self._merge_where(where)
        hits = await self.vector.search(ctx.scope, query, k, merged)
        return [
            MemoryItem(
                content=chunk.text,
                source=self.name,
                id=chunk.id,
                score=score,
                metadata=dict(chunk.metadata),
            )
            for score, chunk in hits
        ]

    async def write(self, items: Iterable[Any], *, ctx: Ctx) -> None:
        """Index a batch of ``MemoryItem``s. Each item becomes a
        ``Chunk`` keyed by ``item.id``, else ``metadata["id"]`` when the
        caller wrote one, else a fresh uuid. ``content`` becomes
        ``Chunk.text``; the rest of ``metadata`` is preserved so callers
        can later ``where=``-filter on whatever tags they wrote.

        ``item.id`` takes precedence because read-modify-write is the loop
        this closes: an item recalled from this store carries its row's id, so
        writing it back must UPDATE that row. Under the uuid fallback it
        inserted a second copy of the same fact instead, which then had to be
        cleaned up downstream by ``CompositeMemory``'s dedupe — a duplicate
        manufactured by the store rather than found in it.

        It is trusted ONLY when the item came from this instance, though, and
        that guard is load-bearing rather than defensive. ``item.id`` is a
        PROVENANCE record — the key of the row a backend handed back — and ids
        are unique only within a backend. ``CompositeMemory.write`` broadcasts
        every item to every source, so without the guard an item recalled from
        a journal whose row key happened to be ``"3"`` upserted over vector
        chunk ``"3"``: an unrelated fact silently destroyed, on the default
        write path. ``query`` stamps ``source`` with this instance's ``name``,
        so the round trip above still matches and still updates in place.

        ``metadata["id"]`` stays supported underneath, and outranks the missing
        provenance match, because it is the caller INSTRUCTING this store which
        row to write rather than reporting where an item came from. It is
        popped either way so the key never survives into ``Chunk.metadata`` —
        two places claiming to be the id would make the next round trip depend
        on which one won."""
        chunks: list[Chunk] = []
        for item in items:
            metadata = dict(item.metadata) if item.metadata else {}
            legacy_id = metadata.pop("id", None)
            # Dropped for the same reason ``id`` is: these describe one
            # ``CompositeMemory`` fan-out, not the record. Persisted, they made
            # a later read look like the STORE was asserting corroboration, and
            # since a merge now absorbs an existing stamp rather than
            # overwriting it, a written-back item would inflate its own count
            # on every round trip.
            for transient in (DEDUPE_SOURCES_KEY, DEDUPE_COUNT_KEY):
                metadata.pop(transient, None)
            own_id = item.id if item.source == self.name else None
            chunk_id = own_id or legacy_id or uuid.uuid4().hex
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
