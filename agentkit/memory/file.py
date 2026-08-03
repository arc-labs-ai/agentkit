"""``FileMemory`` — read/write seam over a file-tree backend.

Wraps any object with ``view(path)`` / ``create(path, text)`` / ``delete(path)``
async methods — the in-tree ``InMemoryFiles`` backend is the default
target, but a real durable backend that implements the same surface
plugs in unchanged.

This is the **read side** of agent-managed file memory: a uniform
``MemorySource`` view on top of the file tree the agent maintains.
``FileTool`` (the tool-call interface the model drives) is a
separate concern and stays unchanged — both wrap the same backend
but answer different questions ("what notes match this query?" vs
"the model wants to edit a note").

Scoring is **lexical**: lowercase-substring match count of the query
in each file's text. Files with zero matches are dropped; the top-k
are returned sorted by match count descending. This is deliberately
simple — when richer ranking is needed, layer a ``Reranker`` via
``CompositeMemory`` rather than enriching the backend.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class FileMemory:
    """Lexical ``MemorySource`` over a file-tree backend.

    ``files`` is duck-typed: any object exposing async ``view(path)``,
    ``create(path, text)``, ``delete(path)``. The default is the
    in-tree ``InMemoryFiles``; a durable / FS-backed implementation
    that conforms to the same surface drops in without changes.

    ``query(query, k, ctx, where=None)`` — walks every file, scores
    by lowercase-substring match count, returns the top ``k`` items
    with metadata ``{"path": …}``. Zero-match files are dropped.
    ``where={"path_prefix": "/foo"}`` scopes the walk to that prefix.

    ``write(items, ctx)`` — writes each item via ``create``. The
    target path comes from ``item.metadata["path"]`` if present,
    otherwise from a SHA-1 hash of the content.
    """

    files: Any  # InMemoryFiles or any object with .view/.create/.delete
    name: str = "files"

    async def _list_paths(self, prefix: str = "/") -> list[str]:
        """Enumerate file paths under ``prefix`` via the backend's
        ``view`` listing (sorted, newline-separated). Empty listings
        return ``[]``."""
        listing = await self.files.view(prefix)
        if not listing:
            return []
        return [line for line in listing.split("\n") if line]

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del ctx
        prefix = "/"
        if where is not None:
            scope = where.get("path_prefix")
            if scope:
                prefix = scope
        paths = await self._list_paths(prefix)
        needle = query.lower()
        scored: list[tuple[int, str, str]] = []  # (count, path, text)
        for path in paths:
            try:
                text = await self.files.view(path)
            except FileNotFoundError:
                # Listing yielded a directory-like entry; skip — only
                # leaf files survive the scoring pass.
                continue
            if not isinstance(text, str):
                continue
            count = text.lower().count(needle) if needle else 0
            if count <= 0:
                continue
            scored.append((count, path, text))
        # Highest match count first; ties broken by path order (stable).
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            MemoryItem(
                content=text,
                source=self.name,
                score=float(count),
                metadata={"path": path},
            )
            for count, path, text in scored[:k]
        ]

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        del ctx
        for item in items:
            path = item.metadata.get("path") if item.metadata else None
            if not path:
                # Deterministic path from content hash — keeps `write`
                # idempotent for identical items.
                digest = hashlib.sha1(item.content.encode("utf-8")).hexdigest()
                path = f"/memories/{digest}.md"
            await self.files.create(path, item.content)


__all__ = ["FileMemory"]
