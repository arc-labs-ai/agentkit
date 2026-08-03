"""``ScratchpadMemory`` — ``MemorySource`` over ``WorkingContext.scratchpad``.

The scratchpad axis of ``WorkingContext`` is a free-form
``dict[str, Any]`` of cross-step notes — the agent's working
notepad. ``ScratchpadMemory`` exposes it as a queryable key-value
memory so the same cognition path that pulls from vector / file /
journal sources can also reach for in-flight notes.

Query semantics are lexical and KV-shaped: an entry matches when
the lowercase key OR the lowercase ``str(value)`` contains the
query substring. Score is ``None`` (no relevance ranking — last
match wins on ties by insertion order, bounded by ``k``).

Writes are best-effort: each ``MemoryItem`` is added at
``metadata["key"]`` if set, otherwise at a truncated content key.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from agentkit.context.context import WorkingContext
from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem


@dataclass(slots=True)
class ScratchpadMemory:
    """A ``MemorySource`` view on a ``WorkingContext.scratchpad``.

    Keys are inspected as-is; values are rendered via ``str(value)``
    for matching. Returned items carry ``metadata={"key": key}`` so
    a downstream writer or reader can address the entry by name.
    """

    context: WorkingContext
    name: str = "scratchpad"

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del ctx, where
        needle = query.lower() if query else ""
        out: list[MemoryItem] = []
        for key, value in self.context.scratchpad.items():
            rendered = str(value)
            if needle:
                if needle not in key.lower() and needle not in rendered.lower():
                    continue
            out.append(
                MemoryItem(
                    content=rendered,
                    source=self.name,
                    score=None,
                    metadata={"key": key},
                )
            )
            if len(out) >= k:
                break
        return out

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        del ctx
        for item in items:
            key = item.metadata.get("key") if item.metadata else None
            if not key:
                # Truncated content key — best-effort addressing when the
                # caller didn't supply one explicitly.
                key = item.content[:32]
            self.context.scratchpad[key] = item.content


__all__ = ["ScratchpadMemory"]
