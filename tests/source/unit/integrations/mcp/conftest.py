"""In-process MCP server for the integration tests.

Spins a ``FastMCP`` server with one tool (``add``), one resource
(``demo://text``), and one prompt (``hello``), then wires an
``MCPClient`` whose ``_session`` is set from
``create_connected_server_and_client_session`` — a helper the ``mcp``
package ships specifically for testing that stitches server and client
via in-memory streams. This bypasses the stdio subprocess spawn while
keeping the full protocol-level round trip under test.

The helper is exposed as an ``asynccontextmanager`` (not a
``pytest_asyncio.fixture``): ``anyio``'s task-scoped cancel scopes on
the underlying transport require enter+exit inside the SAME task, and
pytest-asyncio can hand the fixture back to the test in a different
task, which trips ``RuntimeError: Attempted to exit cancel scope in a
different task``. Using ``async with make_mcp_client() as c`` from
inside each test keeps enter/exit on one task.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from agentkit.integrations.mcp import MCPClient, StdioServer


def _build_server() -> FastMCP:
    srv = FastMCP("agentkit-tests")

    @srv.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @srv.tool()
    async def slow_echo(message: str) -> str:
        """Echo the input after a short async yield — used by the cancel test."""
        import anyio

        await anyio.sleep(0.1)
        return message

    @srv.resource("demo://text")
    def demo_text() -> str:
        """A tiny demo resource."""
        return "hello demo world"

    @srv.resource("demo://other")
    def other_text() -> str:
        """A second resource for substring-filter tests."""
        return "unrelated content"

    @srv.prompt()
    def hello() -> str:
        """A static hello prompt (no args)."""
        return "Hi there, hello prompt speaking."

    @srv.prompt()
    def personalized(name: str) -> str:
        """A prompt with a required argument — v1 adapter should drop it."""
        return f"Hi {name}!"

    return srv


@asynccontextmanager
async def _make_mcp_client() -> AsyncIterator[MCPClient]:
    server = _build_server()
    async with create_connected_server_and_client_session(server) as session:
        client = MCPClient(server=StdioServer(command="unused"))
        client._session = session  # type: ignore[attr-defined]
        try:
            yield client
        finally:
            client._session = None  # type: ignore[attr-defined]


@pytest.fixture
def make_mcp_client() -> object:
    """Sync-yielding fixture returning the async context manager factory.

    Tests use ``async with make_mcp_client() as client:`` — enter and
    exit occur in the test's own task so anyio's cancel-scope contract
    holds.
    """
    return _make_mcp_client
