"""MCP client adapter — consume any Model Context Protocol server and expose
its tools / resources / prompts as agentkit-native ``Tool`` / ``MemorySource`` /
``Prompt`` objects.

This module is opt-in. Install the ``mcp`` extra:

    pip install "arc-agentkit[mcp]"

Then wire an MCP server the same way you wire any other agentkit collaborator:

    from agentkit import Agent
    from agentkit.agents.cognition import ReActCognition
    from agentkit.integrations.mcp import MCPClient, StdioServer, mcp_tools

    async with MCPClient(StdioServer(command="uvx", args=("mcp-server-time",))) as mcp:
        tools = await mcp_tools(mcp)
        agent = Agent(
            name="clock",
            model="gpt-4o-mini",
            prompt="Answer with the current time when asked.",
            cognition=ReActCognition(tools=tools),
        )
        result = await agent.run("What time is it in Tokyo?", ctx)

The module wraps every ``mcp`` import in a guarded ``try/except ImportError``
so that a bare ``agentkit`` install still imports cleanly; the informative
error surfaces only when a caller actually reaches for this integration.
"""

from __future__ import annotations

from agentkit.integrations.mcp.approvals import ApprovalServer
from agentkit.integrations.mcp.client import MCPClient, StdioServer, StreamableHttpServer
from agentkit.integrations.mcp.memory import mcp_resources
from agentkit.integrations.mcp.prompts import mcp_prompts
from agentkit.integrations.mcp.tools import mcp_tools

__all__ = [
    "ApprovalServer",
    "MCPClient",
    "StdioServer",
    "StreamableHttpServer",
    "mcp_prompts",
    "mcp_resources",
    "mcp_tools",
]
