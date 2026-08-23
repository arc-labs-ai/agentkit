"""Helper process: drive `MCPClient` over a REAL streamable-HTTP MCP server.

Run as ``python -m tests.integrations.mcp._http_transport_e2e``; prints one JSON
line describing what happened.

Its own process on purpose. Serving FastMCP's streamable-HTTP app leaves anyio
memory streams for the garbage collector, and this project runs
warnings-as-errors, so the finalisation surfaces as an unraisable
``ResourceWarning`` collected at pytest's SESSION teardown — where a per-test
filter cannot reach it, and where it also fails unrelated tests. Silencing it
globally would blind the suite to a class of real leak. The subprocess boundary
contains it exactly; the parent asserts on the verdict.

This is the coverage the transport migration was missing the first time, which
is why that migration got reverted rather than shipped untested.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP

from agentkit.integrations.mcp import MCPClient, StreamableHttpServer

SEEN_HEADERS: list[dict[str, str]] = []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _build() -> FastMCP:
    mcp = FastMCP("probe", stateless_http=True)

    @mcp.tool(structured_output=False)
    async def add(a: int, b: int) -> str:
        """Add two integers and return the sum as a string."""
        return str(a + b)

    return mcp


async def main() -> dict[str, Any]:
    port = _free_port()
    app = _build().streamable_http_app()

    # Capture the request headers the transport actually sends, which is how we
    # prove `headers=` still reaches the wire after moving onto a client we own.
    class _Capture:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope.get("type") == "http":
                SEEN_HEADERS.append(
                    {k.decode(): v.decode() for k, v in scope.get("headers", [])}
                )
            await self._inner(scope, receive, send)

    config = uvicorn.Config(_Capture(app), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            await task
        await asyncio.sleep(0.01)

    verdict: dict[str, Any] = {}
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        cfg = StreamableHttpServer(url=url, headers={"x-agentkit-probe": "1"}, timeout_s=10.0)
        async with MCPClient(cfg) as client:
            verdict["tools"] = [t.name for t in await client.list_tools()]
            result = await client.call_tool("add", {"a": 2, "b": 40})
            verdict["result"] = result.content[0].text
        verdict["header_seen"] = any(
            h.get("x-agentkit-probe") == "1" for h in SEEN_HEADERS
        )
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5.0)
    return verdict


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main())))
