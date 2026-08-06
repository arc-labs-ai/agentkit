"""Adapt an MCP server's resources into an agentkit ``MemorySource``.

MCP resources are named documents the server exposes for read. This
adapter presents them as a queryable knowledge source: on ``query`` it
lists resources, filters by substring match against name / description /
URI, reads the top-k matches, and returns them as ``MemoryItem``s.

For v1 the ranking is intentionally coarse (substring match, tie-broken
by first-seen order). Callers who want vector-quality ranking can wrap
the returned source in a ``CachedMemory`` / ``CompactedMemory`` decorator
or drop it into a ``CompositeMemory`` alongside a ``VectorMemory``.

Writes are unsupported — MCP resources are server-authored; a client-
side write path would be a protocol violation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Any

from agentkit.kernel.protocols import Ctx
from agentkit.memory.base import MemoryItem

if TYPE_CHECKING:
    from mcp import types as mcp_types

    from agentkit.integrations.mcp.client import MCPClient


_RESOURCE_LIST_TTL_S = 30.0


@dataclass
class _McpResourceMemory:
    """A ``MemorySource`` backed by an MCP server's resource list."""

    _client: MCPClient = field(repr=False)
    name: str = "mcp"
    _resource_cache: tuple[float, list[mcp_types.Resource]] | None = field(default=None, init=False, repr=False)

    async def _resources(self) -> list[mcp_types.Resource]:
        """Return the (short-TTL) cached resource list."""
        now = monotonic()
        if self._resource_cache is not None:
            cached_at, items = self._resource_cache
            if now - cached_at < _RESOURCE_LIST_TTL_S:
                return items
        items = await self._client.list_resources()
        self._resource_cache = (now, items)
        return items

    async def query(
        self,
        query: str,
        *,
        k: int,
        ctx: Ctx,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        del where  # v1: metadata filters unused — MCP resources have no shared schema
        ctx.check_cancelled()
        q = query.strip().lower()
        candidates = await self._resources()
        # Substring rank. Empty query returns first-seen order; otherwise
        # rank by (match_found, position_of_first_match) so an earlier
        # match wins ties.
        ranked: list[tuple[int, int, mcp_types.Resource]] = []
        for idx, r in enumerate(candidates):
            haystack = " ".join(
                s.lower()
                for s in (
                    r.name or "",
                    r.description or "",
                    str(r.uri),
                )
                if s
            )
            if not q:
                score = 0
            elif q in haystack:
                score = 1
            else:
                score = -1
            ranked.append((score, idx, r))
        # Sort: higher score first, earlier index first on ties. Filter
        # out negative-score entries only when the query was non-empty.
        ranked.sort(key=lambda t: (-t[0], t[1]))
        if q:
            ranked = [r for r in ranked if r[0] >= 1]
        top = ranked[:k]

        items: list[MemoryItem] = []
        for _score, _idx, r in top:
            uri = str(r.uri)
            text = await self._client.read_resource(uri)
            items.append(
                MemoryItem(
                    content=text,
                    source=self.name,
                    score=None,
                    metadata={"uri": uri, "name": r.name, "mimeType": r.mimeType},
                )
            )
        return items

    async def write(self, items: Iterable[MemoryItem], *, ctx: Ctx) -> None:
        del items, ctx
        raise NotImplementedError(
            "MCP resources are read-only from the client side. Write to the MCP server directly via its own tooling."
        )


def mcp_resources(client: MCPClient, *, name: str = "mcp") -> _McpResourceMemory:
    """Wrap an ``MCPClient`` as a ``MemorySource`` over its resources."""
    return _McpResourceMemory(_client=client, name=name)


__all__ = ["mcp_resources"]
