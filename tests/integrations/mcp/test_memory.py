"""``mcp_resources`` — expose an MCP server's resources as a ``MemorySource``."""

from __future__ import annotations

from typing import Any

import pytest

from agentkit.integrations.mcp import mcp_resources
from agentkit.memory import MemorySource
from agentkit.testing.fakes.ctx import FakeCtx


@pytest.mark.asyncio
async def test_satisfies_memory_source_protocol(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        src = mcp_resources(mcp_client)
        assert isinstance(src, MemorySource)


@pytest.mark.asyncio
async def test_query_returns_matching_resource(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        src = mcp_resources(mcp_client)
        items = await src.query("demo", k=1, ctx=FakeCtx())
        assert len(items) == 1
        item = items[0]
        assert "hello demo world" in item.content
        assert item.source == "mcp"
        assert item.metadata["uri"].startswith("demo://")


@pytest.mark.asyncio
async def test_query_substring_filters(make_mcp_client: Any) -> None:
    """A query that matches only one resource returns just that resource."""
    async with make_mcp_client() as mcp_client:
        src = mcp_resources(mcp_client)
        items = await src.query("text", k=5, ctx=FakeCtx())
        assert len(items) >= 1
        # `text` matches "demo://text" but not "demo://other" — enforce.
        uris = [it.metadata["uri"] for it in items]
        assert any("://text" in u for u in uris)


@pytest.mark.asyncio
async def test_write_is_unsupported(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        src = mcp_resources(mcp_client)
        with pytest.raises(NotImplementedError, match="read-only"):
            await src.write([], ctx=FakeCtx())


@pytest.mark.asyncio
async def test_prompts_adapter_drops_argumented(make_mcp_client: Any) -> None:
    from agentkit.integrations.mcp import mcp_prompts

    async with make_mcp_client() as mcp_client:
        prompts = await mcp_prompts(mcp_client)
        assert "hello" in prompts
        # `personalized` requires `name` — v1 drops it.
        assert "personalized" not in prompts
        assert "hello prompt speaking" in prompts["hello"].template
        assert prompts["hello"].version == "mcp:1"
