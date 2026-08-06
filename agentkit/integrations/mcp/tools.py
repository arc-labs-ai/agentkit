"""Adapt MCP tools into agentkit ``Tool``s.

Each MCP tool becomes an ``_McpTool`` — a small dataclass whose shape
satisfies the ``@runtime_checkable`` ``Tool`` Protocol structurally.
No inheritance; the adapter is the Adapter pattern applied at the
Protocol boundary, consistent with the rest of the framework.

The tools returned here can be handed to a ``ReActCognition`` unchanged:

    tools = await mcp_tools(mcp_client, prefix="mcp_")
    cognition = ReActCognition(tools=tools)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import ToolSchema

if TYPE_CHECKING:
    from agentkit.integrations.mcp.client import MCPClient


@dataclass
class _McpTool:
    """A ``Tool`` Protocol impl backed by an MCP server call.

    Fields mirror the ``Tool`` Protocol exactly so ``isinstance(t, Tool)``
    passes at runtime. The dataclass isn't ``frozen`` because the ``Tool``
    Protocol expects plain attributes (not descriptor properties);
    ``slots=False`` here keeps the class friendly to duck-typing tests
    that expect a ``__dict__``.
    """

    name: str
    description: str
    schema: ToolSchema | None
    output_schema: type | dict[str, Any] | None
    side_effecting: bool
    requires_approval: bool
    _client: MCPClient = field(repr=False)
    _remote_name: str = field(repr=False)

    async def run(self, args: Mapping[str, Any], ctx: Ctx) -> Any:
        """Invoke the MCP tool and reduce the return payload to a string.

        MCP tools return a ``CallToolResult`` with an opaque ``content``
        list of text / image / audio / resource blocks. Text blocks are
        concatenated; non-text blocks are dropped with a marker so the
        agent sees SOMETHING attributable rather than a silent hole.
        """
        result = await self._client.call_tool(self._remote_name, dict(args), ctx=ctx)
        parts: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            btype = getattr(block, "type", type(block).__name__)
            parts.append(f"[{btype} content omitted]")
        text_out = "\n".join(parts)
        if result.isError:
            return f"ERROR: {text_out or '<no error text>'}"
        return text_out


async def mcp_tools(client: MCPClient, *, prefix: str = "") -> list[_McpTool]:
    """List an MCP server's tools and adapt each into an agentkit ``Tool``.

    ``prefix`` is prepended to the tool name so many MCP servers can share
    one agent without name collisions (e.g. ``prefix="fs_"`` for a
    filesystem server + ``prefix="time_"`` for a clock server).
    """
    tools = await client.list_tools()
    return [
        _McpTool(
            name=f"{prefix}{tool.name}",
            description=tool.description or f"MCP tool {tool.name!r}",
            schema=ToolSchema(
                name=f"{prefix}{tool.name}",
                description=tool.description or "",
                parameters=dict(tool.inputSchema),
            ),
            output_schema=None,
            # Safe default: MCP tools may do anything. Agent-level
            # ``Autonomy`` still gates via ``should_gate``, and callers
            # who KNOW a specific server is read-only can wrap the
            # returned tools to flip the flag.
            side_effecting=True,
            requires_approval=False,
            _client=client,
            _remote_name=tool.name,
        )
        for tool in tools
    ]


__all__ = ["mcp_tools"]
