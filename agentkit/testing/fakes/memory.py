"""``FakeMemory`` — a MemorySource stub for unit tests.

Backed by an in-memory dict ``query -> [MemoryItem]``. Tests can
preload responses + assert on calls.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem


@dataclass
class FakeMemory:
    """A ``MemorySource``-Protocol-conforming fake.

    Two modes:

    * ``responses={"q": [MemoryItem(...), ...], ...}`` — per-query
      lookup. Misses fall through to ``default`` (defaults to ``[]``).
    * ``items=[MemoryItem(...), ...]`` — static fixture mode. ``query``
      ignores the query string entirely and returns the first ``k``
      items regardless. Useful when a test doesn't care about retrieval
      semantics, only that the agent received some items.

    Every ``query`` is recorded as ``(query, k, where)`` on
    ``self.queries``; every ``write`` batch is appended to
    ``self.writes``. Lets tests assert ``mem.queries[0] == ("foo", 3, None)``
    or ``mem.writes == [[MemoryItem("a", "src")]]``.
    """

    name: str = "fake_memory"
    responses: dict[str, list[MemoryItem]] = field(default_factory=dict)
    default: list[MemoryItem] = field(default_factory=list)
    items: list[MemoryItem] | None = None
    queries: list[tuple[str, int, dict[str, Any] | None]] = field(default_factory=list, init=False)
    writes: list[list[MemoryItem]] = field(default_factory=list, init=False)

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del ctx
        self.queries.append((query, k, where))
        if self.items is not None:
            return list(self.items[:k])
        return list(self.responses.get(query, self.default))[:k]

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        del ctx
        self.writes.append(list(items))


__all__ = ["FakeMemory"]
