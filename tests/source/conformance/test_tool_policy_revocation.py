"""Cached discovery and MCP advertisements never replace current host policy."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from agentkit.core.testing import make_test_ctx
from agentkit.tools import FunctionTool, ToolRegistry
from agentkit.tools.execution.policy import ToolSet, serve


@asynccontextmanager
async def channel(tools: ToolSet, route: str) -> AsyncIterator[Any]:
    """Exercise the same policy through API execution or a native MCP session."""
    ctx = make_test_ctx()
    if route.startswith("mcp"):
        pytest.importorskip("mcp")
        from mcp.shared.memory import create_connected_server_and_client_session

        spec = serve(tools, ctx=ctx)
        try:
            async with create_connected_server_and_client_session(spec.build_server()) as session:
                async def execute(name: str, arguments: dict[str, Any]) -> Any:
                    return await session.call_tool(name, arguments)

                yield execute
        finally:
            await spec.stop()
    else:
        async def execute(name: str, arguments: dict[str, Any]) -> Any:
            return await tools.execute(name, arguments, ctx)

        yield execute


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("api_direct", "api_dispatcher", "mcp_direct", "mcp_dispatcher"))
async def test_revoked_host_approval_is_checked_after_discovery_and_cached_mcp_advertisement(route: str) -> None:
    committed: list[str] = []

    async def publish_record(value: str) -> str:
        """Publish an authorized record after the host's current approval check."""
        committed.append(value)
        return "committed"

    declaration = FunctionTool.from_callable(publish_record, side_effecting=True, requires_approval=False)
    tools = ToolSet(ToolRegistry.from_tools([declaration]), deferred=(declaration.name,))
    async with channel(tools, route) as execute:
        await execute("search_tools", {"query": "publish record"})
        assert declaration.name in [schema.name for schema in tools.schemas()]
        # The tool was initially approved and may still be cached by a native
        # client. Revocation must be enforced at execution, independently.
        declaration.requires_approval = True
        name = "call_loaded_tool" if route.endswith("dispatcher") else declaration.name
        arguments = (
            {"name": declaration.name, "arguments_json": json.dumps({"value": "must-not-commit"})}
            if route.endswith("dispatcher") else {"value": "must-not-commit"}
        )
        result = await execute(name, arguments)
        error = result.isError if route.startswith("mcp") else result.is_error
        text = result.content[0].text if route.startswith("mcp") else result.text
        assert error and "host approval" in text
        assert committed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("api_direct", "api_dispatcher", "mcp_direct", "mcp_dispatcher"))
async def test_host_disabled_images_cannot_escape_through_a_discovered_tool(route: str) -> None:
    async def read_picture() -> dict[str, Any]:
        """Read a picture whose image content is disabled by this host policy."""
        return {"content": [{"type": "image", "mimeType": "image/png", "data": "private-image-bytes"}]}

    declaration = FunctionTool.from_callable(read_picture, side_effecting=False)
    tools = ToolSet(ToolRegistry.from_tools([declaration]), images=False, deferred=(declaration.name,))
    async with channel(tools, route) as execute:
        await execute("search_tools", {"query": "picture"})
        result = await execute(
            "call_loaded_tool" if route.endswith("dispatcher") else declaration.name,
            {"name": declaration.name, "arguments_json": "{}"} if route.endswith("dispatcher") else {},
        )
        error = result.isError if route.startswith("mcp") else result.is_error
        text = result.content[0].text if route.startswith("mcp") else result.text
        assert error and "image results are disabled" in text
        assert "private-image-bytes" not in text
        assert all((block.type if route.startswith("mcp") else block["type"]) == "text" for block in result.content)
