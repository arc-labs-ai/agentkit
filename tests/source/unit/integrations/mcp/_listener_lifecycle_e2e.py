"""Real SSE sockets in a fresh process, isolating SDK stream finalizers."""

from __future__ import annotations

import asyncio
import json
import signal
from typing import Any

import httpx

from agentkit.core.testing import make_test_ctx
from agentkit.integrations.mcp import serve_registry
from agentkit.tools import FunctionTool, ToolRegistry


def echo(value: str) -> str:
    """Echo the supplied evidence string without modifying any state."""
    return value


async def main() -> dict[str, Any]:
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    completed: list[str] = []
    # The old SSE watcher attached to the first embedded uvicorn server and
    # noticed should_exit on its next 0.5s poll, permanently breaking server 2.
    for index in range(3):
        spec = serve_registry(ToolRegistry.from_tools([FunctionTool.from_callable(echo, side_effecting=False)]), name=f"server{index}", ctx=make_test_ctx())
        async with spec:
            assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())
            ready = await spec.check_ready()
            assert [entry["name"] for entry in ready["tools"]] == ["echo"]
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(
                    str(spec.url),
                    headers={**dict(spec.auth_headers), "Accept": "application/json, text/event-stream",
                             "MCP-Protocol-Version": "2025-11-25"},
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                          "params": {"name": "echo", "arguments": {"value": f"evidence-{index}"}}},
                )
                response.raise_for_status()
                assert "text/event-stream" in response.headers["content-type"]
                data = next(line[6:] for line in response.text.splitlines() if line.startswith("data: "))
                result = json.loads(data)["result"]
                assert not result.get("isError")
                completed.append(result["content"][0]["text"])
        assert not spec.config_path.exists()
        assert spec._transport is not None and not spec._transport.running
        await asyncio.sleep(0.6)
    return {"completed": completed, "handlers_preserved": True}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main())))
