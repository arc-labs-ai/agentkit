"""Real loopback transport boundaries; no provider or CLI confinement claim."""

import httpx
import pytest

from agentkit.core.testing import make_test_ctx
from agentkit.tools import ToolRegistry, ToolSet, tool
from agentkit.tools.execution.policy import serve


@pytest.mark.asyncio
async def test_bridge_credentials_do_not_cross_calls_and_stale_server_refuses():
    effects = []

    @tool(side_effecting=False)
    def read() -> str:
        """Read a harmless call-scoped test record."""
        effects.append(True)
        return "ok"

    tools = ToolSet(ToolRegistry.from_tools([read]))
    left, right = serve(tools, ctx=make_test_ctx()), serve(tools, ctx=make_test_ctx())
    async with left, right, httpx.AsyncClient(trust_env=False) as client:
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "read", "arguments": {}}}
        for endpoint, other in ((left, right), (right, left)):
            before = len(effects)
            response = await client.post(endpoint.url, json=request, headers=other.auth_headers)
            assert response.status_code == 401 and len(effects) == before
            clean = await client.post(
                endpoint.url,
                json=request,
                headers={
                    **endpoint.auth_headers,
                    "accept": "application/json, text/event-stream",
                },
            )
            assert clean.status_code == 200 and "ok" in clean.text
        assert len(effects) == 2
    stale = await left._dispatch("read", {})
    assert stale.isError and len(effects) == 2


@pytest.mark.asyncio
async def test_authenticated_bridge_bounds_messages_before_json_parsing():
    spec = serve(ToolSet(ToolRegistry()), ctx=make_test_ctx())
    async with spec, httpx.AsyncClient(trust_env=False) as client:
        response = await client.post(
            spec.url,
            content=b"x" * 1_048_577,
            headers={
                **spec.auth_headers,
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
            },
        )
    assert response.status_code == 413
