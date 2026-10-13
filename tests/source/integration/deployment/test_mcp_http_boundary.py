"""Real loopback HTTP/OAuth boundary; explicit host factory owns private endpoints."""

import asyncio
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import httpx
import pytest

from agentkit.agents.prompts.composition import digest
from agentkit.core.contracts.types import Scope
from agentkit.core.runtime import RunContext
from agentkit.integrations.mcp.gateway import (
    McpGateway,
    McpPolicy,
    McpProtocolVersion,
    ServerConfig,
    StatelessMcpClient,
)


@pytest.fixture
def local_oauth_mcp():
    state = {
        "definition": {
            "name": "read_status",
            "description": "Read controlled incident status",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        "issued": 0,
        "refresh_grants": [],
        "requests": [],
        "tool_calls": 0,
        "active_token": None,
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(self, status, payload):
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/oauth/token":
                grant = parse_qs(body.decode())
                state["refresh_grants"].append(grant)
                if grant != {"grant_type": ["refresh_token"], "refresh_token": ["synthetic-refresh-secret"]}:
                    self.respond(400, {"error": "invalid_grant"})
                    return
                state["issued"] += 1
                state["active_token"] = f"synthetic-access-{state['issued']}"
                self.respond(200, {"access_token": state["active_token"], "token_type": "Bearer", "expires_in": 1})
                return
            request = json.loads(body)
            state["requests"].append(
                {
                    "method": request["method"],
                    "auth": self.headers.get("Authorization"),
                    "protocol": self.headers.get("MCP-Protocol-Version"),
                }
            )
            if self.headers.get("Authorization") != f"Bearer {state['active_token']}":
                self.respond(401, {"error": "expired_token"})
                return
            method = request["method"]
            if method == "server/discover":
                result = {"supportedVersions": [McpProtocolVersion.STATELESS]}
            elif method == "tools/list":
                result = {"tools": [state["definition"]]}
            elif method == "tools/call":
                assert request["params"]["name"] == "read_status"
                state["tool_calls"] += 1
                result = {
                    "content": [{"type": "text", "text": "incident stable"}],
                    "structuredContent": {"stable": True},
                    "isError": False,
                }
            else:
                self.respond(400, {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32601}})
                return
            self.respond(200, {"jsonrpc": "2.0", "id": request["id"], "result": result})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


class _HostOAuthResolver:
    """Test harness owns refresh policy; gateway only asks for the current token."""

    def __init__(self, base):
        self.base = base
        self.cached = None
        self.lock = asyncio.Lock()

    async def __call__(self):
        async with self.lock:
            if self.cached is None:
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await client.post(
                        self.base + "/oauth/token",
                        data={"grant_type": "refresh_token", "refresh_token": "synthetic-refresh-secret"},
                    )
                response.raise_for_status()
                self.cached = response.json()["access_token"]
            return self.cached


def _local_factory(base, clients):
    # A host's explicit private-endpoint override is narrowly limited to this
    # controlled server. The default gateway resolver remains unchanged.
    async def factory(config):
        if config.url != base + "/mcp":
            raise PermissionError("private endpoint was not host-approved")
        client = StatelessMcpClient(config, config.url, "127.0.0.1")
        clients.append(client)
        try:
            await client.negotiate()
        except BaseException:
            await client.aclose()
            raise
        return client

    return factory


@pytest.mark.asyncio
async def test_real_http_oauth_rotation_definition_revocation_and_connection_teardown(local_oauth_mcp):
    base, state = local_oauth_mcp
    resolver, clients = _HostOAuthResolver(base), []
    config = ServerConfig("controlled-private", url=base + "/mcp", token=resolver)
    policy = McpPolicy(allow=(config.identity + "#read_status",), approved={"read_status": digest(state["definition"])})
    async with McpGateway(client_factory=_local_factory(base, clients)) as gateway:
        connected = await gateway.connect(config, policy)
        old = gateway.tools().registry.get(gateway.tools().schemas()[0].name)
        result = await old.run({}, RunContext("deployment", Scope()))
        assert result.structured == {"stable": True} and state["tool_calls"] == 1
        resolver.cached = None  # host observes expiry and obtains a real refreshed credential
        await old.run({}, RunContext("deployment", Scope()))
        assert state["issued"] == 2 and len(state["refresh_grants"]) == 2
        auth = [request["auth"] for request in state["requests"] if request["method"] == "tools/call"]
        assert auth == ["Bearer synthetic-access-1", "Bearer synthetic-access-2"]
        assert all(request["protocol"] == McpProtocolVersion.STATELESS for request in state["requests"])
        state["definition"] = {**state["definition"], "description": "Changed contract requires host review"}
        await connected.refresh()
        assert not gateway.tools().schemas()
        before = len(state["requests"])
        with pytest.raises(PermissionError, match="definition changed"):
            await old.run({}, RunContext("deployment", Scope()))
        assert len(state["requests"]) == before and state["tool_calls"] == 2
        connected.policy = replace(policy, approved={"read_status": digest(state["definition"])})
        await connected.refresh()
        renewed = gateway.tools().registry.get(gateway.tools().schemas()[0].name)
        await renewed.run({}, RunContext("deployment", Scope()))
        assert state["tool_calls"] == 3
        with pytest.raises(PermissionError, match="definition changed"):
            await old.run({}, RunContext("deployment", Scope()))
    assert clients[0].client.is_closed
    with pytest.raises(PermissionError, match="closed"):
        await renewed.run({}, RunContext("deployment", Scope()))


@pytest.mark.asyncio
async def test_default_gateway_rejects_private_plaintext_endpoint_without_host_override(local_oauth_mcp):
    base, state = local_oauth_mcp
    async with McpGateway() as gateway:
        with pytest.raises(PermissionError, match="HTTPS"):
            await gateway.connect(ServerConfig("unapproved", url=base + "/mcp"), McpPolicy(allow=("*",)))
    assert state["requests"] == [] and state["refresh_grants"] == []


@pytest.mark.asyncio
async def test_real_http_expired_bearer_is_not_silently_retried_or_committed(local_oauth_mcp):
    base, state = local_oauth_mcp
    resolver, clients = _HostOAuthResolver(base), []
    config = ServerConfig("controlled-private", url=base + "/mcp", token=resolver)
    policy = McpPolicy(allow=("*",), approved={"read_status": digest(state["definition"])})
    async with McpGateway(client_factory=_local_factory(base, clients)) as gateway:
        await gateway.connect(config, policy)
        tool = gateway.tools().registry.get(gateway.tools().schemas()[0].name)
        state["active_token"] = "rotated-on-server"
        with pytest.raises(httpx.HTTPStatusError) as failure:
            await tool.run({}, RunContext("deployment", Scope()))
        assert failure.value.response.status_code == 401
        assert state["tool_calls"] == 0 and state["issued"] == 1
        resolver.cached = None
        await tool.run({}, RunContext("deployment", Scope()))
        assert state["tool_calls"] == 1 and state["issued"] == 2
