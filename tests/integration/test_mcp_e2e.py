"""E2E tests for the MCP integration wired end-to-end into a ReActCognition
against real Claude.

Uses ``mcp.shared.memory.create_connected_server_and_client_session`` — the
official MCP test helper that stitches a FastMCP server and client via
in-memory streams. This exercises the full protocol round-trip (tool
list + call) without spawning a stdio subprocess (kept out of CI for
platform-portability reasons).

We spin an ``add`` tool on the server, expose it as an ``_McpTool``, and
verify it gets called from the model + surfaces a tool_result.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

httpx = pytest.importorskip("httpx")
mcp_pkg = pytest.importorskip("mcp")

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from agentkit.adapters.llm.providers import claude as _claude
from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.integrations.mcp import MCPClient, StdioServer, mcp_tools
from agentkit.kernel.types import Scope, StreamEvent
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services

from .conftest import HAIKU_MODEL, MAX_TOKENS, requires_anthropic


def _run(coro):
    return asyncio.run(coro)


def _ctx_with_llm(llm) -> RunContext:
    invoker = Invoker(
        llm=llm,
        chat_middleware=[tracing(), meter(), retry()],
        tool_middleware=[tracing(), meter(), retry()],
    )
    return RunContext(
        correlation_id="mcp-run",
        scope=Scope(org_id=1, domain_id=1),
        budget=Budget(),
        services=Services(invoker=invoker),
    )


def _build_test_server() -> FastMCP:
    """FastMCP server with one deterministic tool the model can call."""
    srv = FastMCP("agentkit-mcp-e2e")

    @srv.tool()
    def compute_add(a: int, b: int) -> int:
        """Add two integers. Use this whenever asked to add numbers."""
        return a + b

    return srv


@asynccontextmanager
async def _in_process_mcp() -> AsyncIterator[MCPClient]:
    """Yield an MCPClient bound to an in-process FastMCP server via memory streams."""
    server = _build_test_server()
    async with create_connected_server_and_client_session(server) as session:
        client = MCPClient(server=StdioServer(command="unused"))
        client._session = session  # type: ignore[attr-defined]
        try:
            yield client
        finally:
            client._session = None  # type: ignore[attr-defined]


@requires_anthropic
def test_mcp_tool_roundtrips_through_react_cognition(anthropic_key: str) -> None:
    """Real Claude decides to call the MCP ``compute_add`` tool for a math
    question. The tool round-trips through the MCP session and the result
    surfaces back through the ReAct loop."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            async with _in_process_mcp() as mcp:
                tools = await mcp_tools(mcp)
                assert tools, "expected the server to advertise at least one tool"
                # Sanity: name preserved
                assert any(t.name == "compute_add" for t in tools)

                ctx = _ctx_with_llm(llm)
                agent = Agent(
                    name="math",
                    model=HAIKU_MODEL,
                    prompt=(
                        "You add integers using the compute_add tool. "
                        "Always call the tool once and then report the result."
                    ),
                    max_tokens=MAX_TOKENS,
                    cognition=ReActCognition(tools=tools, max_iterations=3),
                )
                events: list[StreamEvent] = []
                async for ev in agent.stream("What is 17 plus 25?", ctx):
                    events.append(ev)
                tc = [e for e in events if e.type == "tool_call"]
                tr = [e for e in events if e.type == "tool_result"]
                finals = [e for e in events if e.type == "final"]
                assert tc, "expected the model to call compute_add"
                assert tr, "expected a tool_result event"
                assert len(finals) == 1
                # The tool result should have the numeric answer
                observed = str(tr[0].tool_result)
                assert "42" in observed, f"expected '42' in tool_result, got {observed!r}"
        finally:
            await llm.aclose()

    _run(go())


def test_mcp_client_call_tool_after_session_closed_raises() -> None:
    """When the MCP session is closed mid-run, subsequent ``tool.run`` must
    fail cleanly rather than crash the interpreter.

    NOTE: ``mcp_tools(...)`` (list_tools) currently returns cached results
    for up to ``_DEFAULT_TTL_S`` even after the session is torn down (see
    the client-side TTL cache). Only the actual ``call_tool`` seam catches
    the closed session — the MediUM finding in the report."""
    from agentkit.runtime import NullCtx

    async def go() -> None:
        async with _in_process_mcp() as mcp:
            tools = await mcp_tools(mcp)
            assert tools
        # Session is now closed. call_tool → RuntimeError from the `.session`
        # property guard on MCPClient.
        with pytest.raises(RuntimeError):
            await tools[0].run({"a": 1, "b": 2}, NullCtx("r"))

    _run(go())


def test_mcp_list_tools_after_close_does_not_serve_stale_cache() -> None:
    """Regression lock-in for the MEDIUM bug from the integration audit.

    After the session is closed, calling ``mcp_tools`` (or
    ``list_tools`` directly) must NOT return the pre-close cache — the
    transport is torn down and any tool the caller subsequently invokes
    would fail. list_tools now guards on ``_session is not None`` before
    trusting the cache, so a post-close call raises ``RuntimeError``
    with the "MCPClient not entered" message (matching the same-shape
    error a caller would hit on ``call_tool``)."""

    async def go() -> None:
        async with _in_process_mcp() as mcp:
            _ = await mcp_tools(mcp)
        # Session torn down → cache is not trusted; call falls through to
        # the (now-None) session and raises. Same for force_refresh=True.
        with pytest.raises(RuntimeError):
            await mcp_tools(mcp)
        with pytest.raises(RuntimeError):
            await mcp.list_tools(force_refresh=True)

    _run(go())
