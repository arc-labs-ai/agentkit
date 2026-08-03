"""``JournalMemory`` — read-only ``MemorySource`` over a ``MutationJournal``.

The journal axis of ``WorkingContext`` is the canonical append-only
authored history. ``JournalMemory`` exposes it as a queryable memory
source so a cognition can pull "what happened earlier in this run"
through the same Protocol as vector recall or file notes.

Read-only by contract: journal authorship goes through
``journal.record()`` / ``apply()``; ``write`` is a no-op. This keeps
the journal's attribution invariant ("the journal never rewrites")
intact — a memory write seam would invite double-authoring.

Recency is the implicit order: results are the **most recent ``k``
entries**, no scoring (every item's ``score`` is ``None``). When a
lexical filter is wanted, callers pass ``query`` and we keep only
entries whose rendered text contains it (case-insensitive). The
optional ``where={"role": "..."}`` filter further narrows to entries
that carry a ``role`` attribute — useful for "what did the planner
say earlier" style lookups.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from agentkit.context.context import MutationJournal
from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem

T = TypeVar("T")


@dataclass(slots=True)
class JournalMemory(Generic[T]):
    """A ``MemorySource`` view on a ``MutationJournal[T]``.

    ``render`` turns each journal entry into the text the cognition
    will read — pushed to the caller so the journal's entry type
    stays unconstrained.

    Order semantics: ``query`` returns the **most recent ``k``**
    entries (no relevance ranking — journals are temporal). Score is
    always ``None`` on returned items.

    Filters:
        ``query`` — lexical filter; case-insensitive substring of the
            rendered text. Empty string disables the filter.
        ``where={"role": "..."}`` — keep only entries whose ``role``
            attribute matches. Entries without a ``role`` attribute
            are skipped under this filter.
    """

    journal: MutationJournal[T]
    render: Callable[[T], str]
    name: str = "journal"

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del ctx
        needle = query.lower() if query else ""
        role_filter: str | None = None
        if where is not None:
            role_filter = where.get("role")

        # Walk newest-first so the take-k below yields most-recent.
        entries = list(self.journal.view())
        selected: list[MemoryItem] = []
        for entry in reversed(entries):
            if role_filter is not None:
                entry_role = getattr(entry, "role", None)
                if entry_role != role_filter:
                    continue
            text = self.render(entry)
            if needle and needle not in text.lower():
                continue
            selected.append(
                MemoryItem(
                    content=text,
                    source=self.name,
                    score=None,
                    metadata={},
                )
            )
            if len(selected) >= k:
                break
        return selected

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        """No-op — the journal's authored history is owned by
        ``journal.record()``; a memory write seam would break that
        invariant."""
        del items, ctx
        return None


__all__ = ["JournalMemory"]
