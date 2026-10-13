"""Sanity checks for ``MCPClient`` against an in-process FastMCP server."""

from __future__ import annotations

from typing import Any

import pytest

from agentkit.integrations.mcp import MCPClient


@pytest.mark.asyncio
async def test_list_and_call_tool(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        tools = await mcp_client.list_tools()
        names = [t.name for t in tools]
        assert "add" in names

        result = await mcp_client.call_tool("add", {"a": 3, "b": 4})
        assert result.isError is False
        # FastMCP wraps a scalar return in a TextContent block.
        texts = [getattr(b, "text", None) for b in result.content]
        assert "7" in texts


@pytest.mark.asyncio
async def test_list_and_read_resource(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        resources = await mcp_client.list_resources()
        uris = [str(r.uri) for r in resources]
        assert "demo://text" in uris

        text = await mcp_client.read_resource("demo://text")
        assert text == "hello demo world"


@pytest.mark.asyncio
async def test_list_and_get_prompt(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        prompts = await mcp_client.list_prompts()
        names = [p.name for p in prompts]
        assert "hello" in names

        rendered = await mcp_client.get_prompt("hello")
        assert "hello prompt speaking" in rendered


@pytest.mark.asyncio
async def test_list_tools_uses_cache(make_mcp_client: Any) -> None:
    """Second call within TTL returns the same list without hitting the server."""
    async with make_mcp_client() as mcp_client:
        first = await mcp_client.list_tools()
        # Mutate the cache entry to a sentinel — if the second call re-fetched,
        # it would overwrite this.
        from agentkit.integrations.mcp.client import _TTLEntry

        mcp_client._tools_cache = _TTLEntry(  # type: ignore[attr-defined]
            value=[first[0]], expires_at=float("inf")
        )
        second = await mcp_client.list_tools()
        assert len(second) == 1
        assert second[0].name == first[0].name


@pytest.mark.asyncio
async def test_session_property_before_entry_raises() -> None:
    from agentkit.integrations.mcp import StdioServer

    unopened = MCPClient(server=StdioServer(command="does-not-run"))
    with pytest.raises(RuntimeError, match="not entered"):
        _ = unopened.session
