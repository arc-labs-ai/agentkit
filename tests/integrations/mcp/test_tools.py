"""``mcp_tools`` — adapt an MCP server's tools into agentkit ``Tool`` objects."""

from __future__ import annotations

from typing import Any

import pytest

from agentkit.integrations.mcp import mcp_tools
from agentkit.testing.fakes.ctx import FakeCtx
from agentkit.tools import Tool


@pytest.mark.asyncio
async def test_returns_tool_protocol_conforming_objects(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        tools = await mcp_tools(mcp_client)
        assert len(tools) >= 1
        for t in tools:
            assert isinstance(t, Tool), f"{t.name!r} does not satisfy Tool Protocol"


@pytest.mark.asyncio
async def test_runs_add_via_adapter(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        tools = await mcp_tools(mcp_client)
        add = next(t for t in tools if t.name == "add")
        out = await add.run({"a": 5, "b": 6}, ctx=FakeCtx())
        assert "11" in out


@pytest.mark.asyncio
async def test_prefix_applied_to_names(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        tools = await mcp_tools(mcp_client, prefix="mcp_")
        names = {t.name for t in tools}
        assert "mcp_add" in names
        # The remote name stays unprefixed for the RPC call.
        add = next(t for t in tools if t.name == "mcp_add")
        out = await add.run({"a": 1, "b": 2}, ctx=FakeCtx())
        assert "3" in out
        # The advertised schema uses the prefixed name too.
        assert add.schema is not None
        assert add.schema.name == "mcp_add"


@pytest.mark.asyncio
async def test_tools_carry_input_schema(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        tools = await mcp_tools(mcp_client)
        add = next(t for t in tools if t.name == "add")
        assert add.schema is not None
        assert add.schema.parameters.get("type") == "object"
        props = add.schema.parameters.get("properties") or {}
        assert set(props.keys()) == {"a", "b"}
