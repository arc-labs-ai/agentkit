"""Expose read-only MCP resources through the agentkit MemoryProvider contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING

from agentkit.core.contracts.types import Scope
from agentkit.state.context.memory import MemoryContext

if TYPE_CHECKING:
    from mcp import types as mcp_types

    from agentkit.integrations.mcp.client import MCPClient

_RESOURCE_LIST_TTL_S = 30.0


@dataclass
class McpResourceMemory:
    client: MCPClient = field(repr=False)
    name: str = "mcp"
    _resource_cache: tuple[float, list[mcp_types.Resource]] | None = field(default=None, init=False, repr=False)

    async def _resources(self) -> list[mcp_types.Resource]:
        now = monotonic()
        if self._resource_cache is not None and now - self._resource_cache[0] < _RESOURCE_LIST_TTL_S:
            return self._resource_cache[1]
        resources = await self.client.list_resources()
        self._resource_cache = (now, resources)
        return resources

    async def retrieve(self, query: str, *, scope: Scope, limit: int) -> tuple[MemoryContext, ...]:
        del scope
        if limit < 1:
            raise ValueError("memory limit must be positive")
        q = query.strip().casefold()
        ranked = []
        for index, resource in enumerate(await self._resources()):
            haystack = " ".join((resource.name or "", resource.description or "", str(resource.uri))).casefold()
            if not q or q in haystack:
                ranked.append((0 if not q else haystack.index(q) + 1, index, resource))
        ranked.sort(key=lambda item: (item[0], item[1]))
        items = []
        for _, _, resource in ranked[:limit]:
            uri = str(resource.uri)
            content = await self.client.read_resource(uri)
            items.append(MemoryContext(source=self.name, content=content, id=uri,
                                       metadata={"uri": uri, "name": resource.name,
                                                 "mime_type": resource.mimeType}))
        return tuple(items)


def mcp_resources(client: MCPClient, *, name: str = "mcp") -> McpResourceMemory:
    return McpResourceMemory(client=client, name=name)


__all__ = ["McpResourceMemory", "mcp_resources"]
