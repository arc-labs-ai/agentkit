from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import replace

import httpx
import pytest

from agentkit.agents.prompts.composition import digest
from agentkit.core.contracts.types import Scope
from agentkit.core.runtime import RunContext
from agentkit.integrations.mcp.gateway import (
    McpGateway,
    McpPolicy,
    McpProtocolError,
    McpProtocolVersion,
    ServerConfig,
    StatelessMcpClient,
    resolve_public_url,
)


def rpc(request, result=None, *, error=None, headers=None):
    payload = json.loads(request.content)
    body = {"jsonrpc": "2.0", "id": payload["id"]}
    body["error" if error is not None else "result"] = error if error is not None else result or {}
    return httpx.Response(200, json=body, headers=headers)


def client(handler, **options):
    configured = StatelessMcpClient(
        ServerConfig("test", url="https://public.test:9443/mcp", **options), "https://8.8.8.8:9443/mcp", "public.test"
    )
    configured.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return configured


@pytest.mark.asyncio
async def test_stateless_discovery_and_tool_request_use_pinned_host_and_fresh_credentials():
    requests = []
    credentials = iter(["first", "rotated"])

    async def credential():
        return next(credentials)

    def handler(request):
        payload = json.loads(request.content)
        requests.append(request)
        assert request.url.host == "8.8.8.8"
        assert request.headers["Host"] == "public.test:9443"
        assert request.extensions["sni_hostname"] == "public.test"
        if payload["method"] == "server/discover":
            return rpc(request, {"supportedVersions": [McpProtocolVersion.STATELESS]})
        assert payload["params"]["_meta"]["io.modelcontextprotocol/clientInfo"]["name"] == "agentkit"
        return rpc(request, {"content": [{"type": "text", "text": "done"}]})

    transport = client(handler, token=credential)
    try:
        await transport.negotiate()
        assert (await transport.call_tool("read", {}))["content"][0]["text"] == "done"
        assert [r.headers["Authorization"] for r in requests] == ["Bearer first", "Bearer rotated"]
        assert requests[-1].headers["Mcp-Name"] == "read"
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_legacy_negotiation_tracks_session_and_supported_version():
    methods = []

    def handler(request):
        payload = json.loads(request.content)
        methods.append(payload["method"])
        if payload["method"] == "server/discover":
            return rpc(request, error={"code": -32601, "message": "method missing"})
        if payload["method"] == "initialize":
            assert payload["params"]["protocolVersion"] == McpProtocolVersion.STREAMABLE
            return rpc(
                request,
                {"protocolVersion": McpProtocolVersion.STREAMABLE_JUNE},
                headers={"MCP-Session-Id": "session-1"},
            )
        assert request.headers["MCP-Session-Id"] == "session-1"
        assert request.headers["MCP-Protocol-Version"] == McpProtocolVersion.STREAMABLE_JUNE
        if payload["method"] == "notifications/initialized":
            assert "id" not in payload
            return httpx.Response(202)
        return rpc(request, {"tools": []})

    transport = client(handler)
    try:
        await transport.negotiate()
        assert await transport.list_tools() == []
        assert methods == ["server/discover", "initialize", "notifications/initialized", "tools/list"]
    finally:
        await transport.aclose()


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_sse_reads_incremental_frames_ignores_notifications_and_closes_at_result():
    stream = Chunks(
        [
            b': keepalive\r\n\r\ndata: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\r\n\r\n',
            b'event: message\ndata: {"jsonrpc":"2.0",\n',
            b'data: "id":1,"result":{"tools":[]}}\n\n',
            b"ignored after the result",
        ]
    )
    transport = client(lambda _: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream))
    try:
        assert await transport.list_tools() == []
        assert stream.closed
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,content_type,message",
    [
        ({"jsonrpc": "2.0", "id": 99, "result": {}}, "application/json", "match"),
        ({"jsonrpc": "2.0", "id": 1, "result": []}, "application/json", "object"),
        ({"jsonrpc": "2.0", "id": 1, "result": {}}, "text/html", "content type"),
    ],
)
async def test_transport_rejects_mismatched_or_invalid_responses(body, content_type, message):
    transport = client(lambda _: httpx.Response(200, json=body, headers={"Content-Type": content_type}))
    try:
        with pytest.raises(ValueError, match=message):
            await transport.list_tools()
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["application/json", "text/event-stream"])
async def test_transport_checks_size_before_materializing_entire_response(content_type):
    stream = Chunks([b" " * 32, b" " * 32])
    transport = client(lambda _: httpx.Response(200, headers={"Content-Type": content_type}, stream=stream))
    transport.MAX_RESPONSE_BYTES = 40
    try:
        with pytest.raises(ValueError, match="transport bound"):
            await transport.list_tools()
        assert stream.closed
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_protocol_error_retains_code_without_disclosing_body():
    transport = client(lambda request: rpc(request, error={"code": -32000, "message": "SECRET"}))
    try:
        with pytest.raises(McpProtocolError) as error:
            await transport.list_tools()
        assert error.value.code == -32000
        assert "SECRET" not in str(error.value)
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_pagination_accumulates_and_refuses_repeated_cursor():
    def handler(request):
        payload = json.loads(request.content)
        if payload["params"].get("cursor"):
            return rpc(request, {"tools": [{"name": "second"}], "nextCursor": "same"})
        return rpc(request, {"tools": [{"name": "first"}], "nextCursor": "same"})

    transport = client(handler)
    try:
        with pytest.raises(ValueError, match="pagination"):
            await transport.list_tools()
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_last_pagination_page_cannot_exceed_catalogue_bound():
    transport = client(lambda request: rpc(request, {"tools": [{"name": "a"}, {"name": "b"}]}))
    transport.MAX_TOOLS = 1
    try:
        with pytest.raises(ValueError, match="catalogue"):
            await transport.list_tools()
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_host_token_resolver_is_inside_total_transport_deadline():
    async def stalled():
        await asyncio.Event().wait()

    transport = client(lambda request: rpc(request), token=stalled, timeout_s=0.01)
    try:
        with pytest.raises(TimeoutError):
            await transport.list_tools()
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["", "bad\ntoken", 123])
async def test_invalid_resolved_credentials_refused_before_network(token):
    transport = client(lambda _: pytest.fail("invalid credential reached network"), token=lambda: token)
    try:
        with pytest.raises(ValueError, match="bearer token"):
            await transport.list_tools()
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url", ["http://public.test/mcp", "https://user:pass@public.test/mcp", "https://public.test/mcp#fragment"]
)
async def test_non_https_or_embedded_credentials_rejected(url):
    with pytest.raises(PermissionError):
        await resolve_public_url(url)


@pytest.mark.asyncio
async def test_dns_pin_requires_every_resolved_address_to_be_public(monkeypatch):
    async def addresses(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)) for address in ["8.8.8.8", "127.0.0.1"]]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", addresses)
    with pytest.raises(PermissionError, match="all be public"):
        await resolve_public_url("https://public.test/mcp")


@pytest.mark.asyncio
async def test_dns_pin_formats_ipv6_and_preserves_public_path(monkeypatch):
    async def addresses(*args, **kwargs):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:4860:4860::8888", 9443, 0, 0))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", addresses)
    pinned, hostname = await resolve_public_url("https://public.test:9443/mcp?tenant=1")
    assert pinned == "https://[2001:4860:4860::8888]:9443/mcp?tenant=1" and hostname == "public.test"


@pytest.mark.asyncio
async def test_external_policy_deny_and_invocation_approval_override_pinned_approval():
    definition = {"name": "read", "description": "read externally", "inputSchema": {"type": "object"}}

    class Client:
        async def list_tools(self):
            return [definition]

        async def call_tool(self, *args):
            pytest.fail("denied invocation reached external server")

    config = ServerConfig("display", url="https://public.test/mcp")
    policy = McpPolicy(
        allow=(config.identity + "#*",), approved={"read": digest(definition)}, approval=lambda *args: False
    )
    async with McpGateway(client_factory=lambda _: Client()) as gateway:
        connection = await gateway.connect(config, policy)
        tools = gateway.tools()
        ctx = RunContext(correlation_id="gateway", scope=Scope())
        name = tools.registry.names()[0]
        result = await tools.execute(name, {}, ctx)
        assert result.is_error and "not approved" in result.text
        connection.policy = replace(policy, deny=(config.identity + "#read",))
        assert (await tools.execute(name, {}, ctx)).is_error
        assert not gateway.tools(approved_only=False).registry.names()


@pytest.mark.asyncio
async def test_optional_failure_is_unavailable_and_duplicate_identity_refused():
    async def unavailable(config):
        raise ConnectionError("offline")

    config = ServerConfig("display", url="https://public.test/mcp", required=False)
    async with McpGateway(client_factory=unavailable) as gateway:
        connection = await gateway.connect(config, McpPolicy())
        assert connection.status == "unavailable" and not gateway.tools().registry.names()
        with pytest.raises(ValueError, match="already connected"):
            await gateway.connect(replace(config, name="another label"), McpPolicy())
    async with McpGateway(client_factory=unavailable) as gateway:
        with pytest.raises(ConnectionError):
            await gateway.connect(replace(config, required=True), McpPolicy())


@pytest.mark.asyncio
async def test_stdio_requires_explicit_host_sandbox():
    async with McpGateway() as gateway:
        with pytest.raises(PermissionError, match="sandbox"):
            await gateway.connect(ServerConfig("local", command=("not-run",)), McpPolicy())


@pytest.mark.parametrize(
    "config", [{}, {"url": "https://public.test", "command": ("run",)}, {"url": "https://public.test", "timeout_s": 0}]
)
def test_server_config_rejects_ambiguous_or_unbounded_setup(config):
    with pytest.raises(ValueError):
        ServerConfig("invalid", **config)


@pytest.mark.asyncio
async def test_modern_metadata_and_schema_header_mirroring_follow_spec():
    seen = []
    schema = {
        "type": "object",
        "properties": {
            "region": {"type": "string", "x-mcp-header": "Region"},
            "settings": {
                "type": "object",
                "properties": {
                    "strict": {"type": "boolean", "x-mcp-header": "Strict"},
                    "limit": {"type": "integer", "x-mcp-header": "Limit"},
                },
            },
        },
    }

    def handler(request):
        payload = json.loads(request.content)
        metadata = payload["params"]["_meta"]
        assert metadata["io.modelcontextprotocol/protocolVersion"] == request.headers["MCP-Protocol-Version"]
        assert metadata["io.modelcontextprotocol/clientCapabilities"] == {}
        if payload["method"] == "tools/list":
            return rpc(request, {"tools": [{"name": "read", "inputSchema": schema}]})
        seen.append(request)
        return rpc(request, {"content": []})

    transport = client(handler)
    try:
        await transport.list_tools()
        await transport.call_tool("read", {"region": "line1\nline2", "settings": {"strict": True, "limit": 42}})
        assert seen[0].headers["Mcp-Param-Region"] == "=?base64?bGluZTEKbGluZTI=?="
        assert seen[0].headers["Mcp-Param-Strict"] == "true"
        assert seen[0].headers["Mcp-Param-Limit"] == "42"
        await transport.call_tool("read", {"region": None, "settings": {}})
        assert "Mcp-Param-Region" not in seen[1].headers
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        {"type": "string", "x-mcp-header": "root"},
        {"type": "object", "properties": {"n": {"type": "number", "x-mcp-header": "Number"}}},
        {"type": "object", "properties": {"n": {"type": "integer", "x-mcp-header": "bad\nheader"}}},
        {
            "type": "object",
            "properties": {
                "a": {"type": "string", "x-mcp-header": "Same"},
                "b": {"type": "string", "x-mcp-header": "same"},
            },
        },
        {"type": "array", "items": {"type": "string", "x-mcp-header": "Item"}},
        {"type": "object", "anyOf": [{"type": "string", "x-mcp-header": "Branch"}]},
        {"type": "object", "$defs": {"hidden": {"type": "string", "x-mcp-header": "Hidden"}}},
    ],
)
async def test_invalid_header_annotations_are_excluded_without_blocking_valid_tools(invalid):
    definitions = [{"name": "invalid", "inputSchema": invalid}, {"name": "valid", "inputSchema": {"type": "object"}}]
    transport = client(lambda request: rpc(request, {"tools": definitions}))
    try:
        assert [t["name"] for t in await transport.list_tools()] == ["valid"]
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [2**53, True, "42"])
async def test_header_integer_value_must_be_safe_and_typed(value):
    definition = {
        "name": "tool",
        "inputSchema": {"type": "object", "properties": {"n": {"type": "integer", "x-mcp-header": "N"}}},
    }
    transport = client(lambda request: rpc(request, {"tools": [definition]}))
    try:
        await transport.list_tools()
        with pytest.raises(ValueError, match="header type"):
            await transport.call_tool("tool", {"n": value})
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_unsupported_version_error_on_http400_negotiates_declared_legacy_support():
    methods = []

    def handler(request):
        payload = json.loads(request.content)
        methods.append(payload["method"])
        if payload["method"] == "server/discover":
            return httpx.Response(
                400,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "error": {"code": -32022, "data": {"supported": [McpProtocolVersion.STREAMABLE_MARCH]}},
                },
            )
        if payload["method"] == "initialize":
            assert payload["params"]["protocolVersion"] == McpProtocolVersion.STREAMABLE_MARCH
            return rpc(request, {"protocolVersion": McpProtocolVersion.STREAMABLE_MARCH})
        return httpx.Response(202)

    transport = client(handler)
    try:
        await transport.negotiate()
        assert transport.version == McpProtocolVersion.STREAMABLE_MARCH
        assert methods == ["server/discover", "initialize", "notifications/initialized"]
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_unknown_supported_version_is_refused_without_an_initialize_probe():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            400, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32022, "data": {"supported": ["future-version"]}}}
        )

    transport = client(handler)
    try:
        with pytest.raises(ValueError, match="mutually supported"):
            await transport.negotiate()
        assert len(requests) == 1
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_recognized_modern_header_error_never_downgrades_to_legacy():
    transport = client(lambda request: httpx.Response(400, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32020}}))
    try:
        with pytest.raises(McpProtocolError) as error:
            await transport.negotiate()
        assert error.value.code == -32020
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_legacy_sse_stream_without_terminal_response_is_not_treated_as_success():
    transport = client(
        lambda request: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content='data: {"jsonrpc":"2.0","method":"notifications/message"}\n\n',
        )
    )
    try:
        with pytest.raises(ValueError, match="ended"):
            await transport.list_tools()
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_gateway_closes_owned_clients_and_revokes_previously_exported_tools():
    class Client:
        closed = False

        async def list_tools(self):
            return [{"name": "read", "inputSchema": {"type": "object"}}]

        async def aclose(self):
            self.closed = True

    owned = Client()
    config = ServerConfig("owned", url="https://public.test/mcp")
    pin = digest({"name": "read", "inputSchema": {"type": "object"}})
    async with McpGateway(client_factory=lambda _: owned) as gateway:
        await gateway.connect(config, McpPolicy(allow=(config.identity + "#*",), approved={"read": pin}))
        tools = gateway.tools()
        name = tools.registry.names()[0]
    assert owned.closed
    assert (await tools.execute(name, {}, RunContext(correlation_id="test", scope=Scope()))).is_error
    with pytest.raises(RuntimeError, match="closed"):
        await gateway.connect(config, McpPolicy())


@pytest.mark.asyncio
async def test_failed_connect_closes_client_immediately_and_allows_retry():
    class Client:
        closed = False

        async def list_tools(self):
            raise ConnectionError("offline")

        async def aclose(self):
            self.closed = True

    owned = Client()
    config = ServerConfig("owned", url="https://public.test/mcp")
    async with McpGateway(client_factory=lambda _: owned) as gateway:
        for _ in range(2):
            owned.closed = False
            with pytest.raises(ConnectionError):
                await gateway.connect(config, McpPolicy())
            assert owned.closed


@pytest.mark.asyncio
async def test_connect_reserves_identity_before_async_discovery():
    entered, release = asyncio.Event(), asyncio.Event()

    class Client:
        async def list_tools(self):
            entered.set()
            await release.wait()
            return []

    config = ServerConfig("owned", url="https://public.test/mcp")
    async with McpGateway(client_factory=lambda _: Client()) as gateway:
        task = asyncio.create_task(gateway.connect(config, McpPolicy()))
        await entered.wait()
        with pytest.raises(ValueError, match="already connected"):
            await gateway.connect(config, McpPolicy())
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {"resultType": "input_required", "inputRequests": {"question": {"method": "elicitation/create"}}},
        {"resultType": "input_required", "requestState": "opaque"},
        {"resultType": "task", "task": {"taskId": "opaque"}},
        {"inputRequests": {"question": {"method": "elicitation/create"}}},
    ],
)
async def test_unhandled_multi_round_trip_or_task_cannot_become_empty_success(result):
    transport = client(lambda request: rpc(request, result))
    try:
        with pytest.raises(NotImplementedError, match="host client"):
            await transport.call_tool("read", {})
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_host_factory_unhandled_input_requests_become_explicit_model_visible_error():
    definition = {"name": "read", "inputSchema": {"type": "object"}}

    class Client:
        async def list_tools(self):
            return [definition]

        async def call_tool(self, *args):
            return {"resultType": "input_required", "requestState": "opaque"}

    config = ServerConfig("host", url="https://public.test/mcp")
    policy = McpPolicy(allow=(config.identity + "#*",), approved={"read": digest(definition)})
    async with McpGateway(client_factory=lambda _: Client()) as gateway:
        await gateway.connect(config, policy)
        tools = gateway.tools()
        result = await tools.execute(tools.registry.names()[0], {}, RunContext(correlation_id="test", scope=Scope()))
        assert result.is_error and "host client" in result.text
