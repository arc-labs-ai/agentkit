"""Pinned external MCP tools re-exposed through the common registry.

OAuth tokens and stdio sandbox launchers are supplied by the host; this module
never persists credentials or interprets a server's approval claims as policy.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import re
import socket
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from fnmatch import fnmatchcase
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agentkit.agents.prompts.composition import digest
from agentkit.core.contracts.types import ToolSchema
from agentkit.tools import ToolRegistry
from agentkit.tools.execution.policy import ToolResult, ToolSet


class McpServerStatus(StrEnum):
    CONNECTED = "connected"
    UNAVAILABLE = "unavailable"
    CLOSED = "closed"


class McpProtocolVersion(StrEnum):
    STATELESS = "2026-07-28"
    STREAMABLE = "2025-11-25"
    STREAMABLE_JUNE = "2025-06-18"
    STREAMABLE_MARCH = "2025-03-26"


class McpProtocolError(RuntimeError):
    """Structured protocol failure without exposing a server's untrusted body."""

    def __init__(self, code: int | None, supported: tuple[str, ...] = ()) -> None:
        self.code = code
        self.supported = supported
        super().__init__(f"MCP protocol error code {code}")


@dataclass(frozen=True)
class ServerConfig:
    """Identity is the URL or command/args, never the display name."""

    name: str
    url: str | None = None
    command: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict, repr=False, hash=False)
    required: bool = True
    timeout_s: float = 30
    protocol_version: McpProtocolVersion = McpProtocolVersion.STATELESS
    token: Callable[[], Any] | None = field(default=None, repr=False)
    sandbox_launcher: Callable[[tuple[str, ...]], tuple[str, ...]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from agentkit.core.serialization._immutable import deep_freeze

        if bool(self.url) == bool(self.command):
            raise ValueError("configure exactly one MCP URL or command")
        if self.timeout_s <= 0:
            raise ValueError("MCP timeout must be positive")
        object.__setattr__(self, "protocol_version", McpProtocolVersion(self.protocol_version))
        object.__setattr__(self, "env", deep_freeze(dict(self.env)))

    @property
    def identity(self) -> str:
        return self.url or digest(self.command)


@dataclass(frozen=True)
class McpPolicy:
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    approved: Mapping[str, str] = field(default_factory=dict, hash=False)  # name -> raw definition digest
    approval: Callable[[str, str, Mapping[str, Any]], Any] | None = field(default=None, repr=False)
    max_description_chars: int = 4096
    max_tools: int = 1000

    def __post_init__(self) -> None:
        from agentkit.core.serialization._immutable import deep_freeze

        if self.max_tools <= 0 or self.max_description_chars < 0:
            raise ValueError("MCP catalogue bounds must be positive")
        object.__setattr__(self, "approved", deep_freeze(dict(self.approved)))

    def permits(self, identity: str, name: str) -> bool:
        key = f"{identity}#{name}"
        return not any(fnmatchcase(key, x) for x in self.deny) and any(fnmatchcase(key, x) for x in self.allow)


async def resolve_public_url(url: str) -> tuple[str, str]:
    """Resolve once and return a pinned address plus TLS/Host server name."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise PermissionError("external MCP requires HTTPS without embedded credentials")
    hostname = parts.hostname
    infos = await asyncio.get_running_loop().getaddrinfo(hostname, parts.port or 443, type=socket.SOCK_STREAM)
    addresses = {str(info[4][0]) for info in infos}
    if not addresses or any(not ipaddress.ip_address(x).is_global for x in addresses):
        raise PermissionError("external MCP addresses must all be public")
    address = sorted(addresses)[0]
    literal = f"[{address}]" if ":" in address else address
    netloc = literal + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, "")), hostname


def _header_value(value: str) -> str:
    if (
        value != value.strip()
        or any(not (0x20 <= ord(char) <= 0x7E or char == "\t") for char in value)
        or value.startswith("=?base64?")
        and value.endswith("?=")
    ):
        return "=?base64?" + base64.b64encode(value.encode()).decode() + "?="
    return value


def _tool_header_paths(schema: Mapping[str, Any]) -> list[tuple[tuple[str, ...], str, str]]:
    """Validate header annotations, including unreachable schema positions."""
    paths: list[tuple[tuple[str, ...], str, str]] = []
    seen: set[str] = set()

    def walk(node: Any, path: tuple[str, ...], reachable: bool) -> None:
        if not isinstance(node, dict):
            return
        annotation = node.get("x-mcp-header")
        if "x-mcp-header" in node:
            kind = node.get("type")
            if (
                not reachable
                or not path
                or not isinstance(annotation, str)
                or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", annotation)
                or annotation.lower() in seen
                or kind not in ("string", "integer", "boolean")
            ):
                raise ValueError("invalid MCP tool header annotation")
            seen.add(annotation.lower())
            paths.append((path, annotation, kind))
        for name, child in (node.get("properties") or {}).items():
            walk(child, path + (name,), reachable)
        for keyword in ("items", "contains", "additionalProperties", "not", "if", "then", "else", "propertyNames"):
            walk(node.get(keyword), path, False)
        for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
            for child in node.get(keyword) or []:
                walk(child, path, False)
        for keyword in ("$defs", "definitions", "patternProperties", "dependentSchemas"):
            for child in (node.get(keyword) or {}).values():
                walk(child, path, False)

    walk(dict(schema), (), True)
    return paths


class StatelessMcpClient:
    """Pinned HTTP transport for stateless and initialized Streamable HTTP.

    JSON and POST SSE responses are bounded and matched to the request ID.
    Server-initiated requests, background GET streams, SSE resumption and the
    deprecated two-endpoint HTTP+SSE transport are intentionally unsupported.
    The host may supply a client_factory when those capabilities are needed.
    """

    MAX_RESPONSE_BYTES = 8 * 1024 * 1024
    MAX_TOOLS = 1000

    def __init__(self, config: ServerConfig, pinned_url: str, hostname: str) -> None:
        import httpx

        self.config, self.url, self.hostname = config, pinned_url, hostname
        # Environment proxies would bypass address pinning and leak auth headers.
        self.client = httpx.AsyncClient(timeout=config.timeout_s, follow_redirects=False, trust_env=False)
        self.version = config.protocol_version
        self._id = 0
        self._session_id: str | None = None
        self._tool_headers: dict[str, list[tuple[tuple[str, ...], str, str]]] = {}

    async def aclose(self) -> None:
        await self.client.aclose()

    async def negotiate(self) -> None:
        import httpx

        if self.version == McpProtocolVersion.STATELESS:
            try:
                discovery = await self.request("server/discover")
            except McpProtocolError as exc:
                if exc.code == -32022:
                    supported = [
                        version
                        for version in (
                            McpProtocolVersion.STREAMABLE,
                            McpProtocolVersion.STREAMABLE_JUNE,
                            McpProtocolVersion.STREAMABLE_MARCH,
                        )
                        if version in exc.supported
                    ]
                    if not supported:
                        raise ValueError("server has no mutually supported MCP version") from exc
                    self.version = supported[0]
                elif exc.code != -32601:
                    raise
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 400:
                    raise
            else:
                versions = (
                    discovery.get("supportedVersions") or discovery.get("versions") or discovery.get("protocolVersions")
                )
                if versions and self.version not in versions:
                    raise ValueError("server does not support the configured MCP protocol version")
                return
            if self.version == McpProtocolVersion.STATELESS:
                self.version = McpProtocolVersion.STREAMABLE
        result = await self.request(
            "initialize",
            {
                "protocolVersion": self.version,
                "capabilities": {},
                "clientInfo": {"name": "agentkit", "version": "0.1.0"},
            },
        )
        negotiated = result.get("protocolVersion")
        if negotiated not in {
            McpProtocolVersion.STREAMABLE,
            McpProtocolVersion.STREAMABLE_JUNE,
            McpProtocolVersion.STREAMABLE_MARCH,
        }:
            raise ValueError("server negotiated an unsupported MCP protocol version")
        self.version = McpProtocolVersion(negotiated)
        await self._send("notifications/initialized", {}, notification=True)

    async def _headers(self, method: str, params: Mapping[str, Any]) -> dict[str, str]:
        headers = {
            "MCP-Protocol-Version": self.version,
            "Mcp-Method": method,
            "Host": self.hostname
            + (f":{urlsplit(self.config.url or self.url).port}" if urlsplit(self.config.url or self.url).port else ""),
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["MCP-Session-Id"] = self._session_id
        if "name" in params or "uri" in params:
            headers["Mcp-Name"] = _header_value(str(params.get("name", params.get("uri"))))
        if method == "tools/call" and self.version == McpProtocolVersion.STATELESS:
            for path, annotation, kind in self._tool_headers.get(str(params.get("name")), []):
                value: Any = params.get("arguments") or {}
                for key in path:
                    value = value.get(key) if isinstance(value, Mapping) else None
                if value is None:
                    continue
                if (
                    kind == "string"
                    and not isinstance(value, str)
                    or kind == "boolean"
                    and not isinstance(value, bool)
                    or kind == "integer"
                    and (not isinstance(value, int) or isinstance(value, bool) or abs(value) > 2**53 - 1)
                ):
                    raise ValueError("tool argument cannot be represented by its MCP header type")
                text = str(value).lower() if isinstance(value, bool) else str(value)
                headers["Mcp-Param-" + annotation] = _header_value(text)
        if self.config.token:
            token = self.config.token()
            if hasattr(token, "__await__"):
                token = await token
            if not isinstance(token, str) or not token or "\r" in token or "\n" in token:
                raise ValueError("host credential resolver returned an invalid bearer token")
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        return await self._send(method, dict(params or {}))

    async def _send(self, method: str, params: dict[str, Any], *, notification: bool = False) -> Any:
        self._id += 1
        request_id = self._id
        if self.version == McpProtocolVersion.STATELESS:
            params["_meta"] = {
                **dict(params.get("_meta") or {}),
                "io.modelcontextprotocol/protocolVersion": self.version,
                "io.modelcontextprotocol/clientInfo": {"name": "agentkit", "version": "0.1.0"},
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            body["id"] = request_id
        async with asyncio.timeout(self.config.timeout_s):
            headers = await self._headers(method, params)
            async with self.client.stream(
                "POST", self.url, json=body, headers=headers, extensions={"sni_hostname": self.hostname}
            ) as response:
                if response.status_code in (400, 404) and "application/json" in response.headers.get(
                    "content-type", ""
                ):
                    error_body = await self._read_json(response)
                    if isinstance(error_body, dict) and "error" in error_body:
                        self._result(error_body, request_id)
                response.raise_for_status()
                if method == "initialize":
                    session_id = response.headers.get("MCP-Session-Id")
                    if session_id is not None:
                        if not session_id or any(not 0x21 <= ord(x) <= 0x7E for x in session_id):
                            raise ValueError("server supplied an invalid MCP session ID")
                        self._session_id = session_id
                if notification:
                    if response.status_code != 202:
                        raise ValueError("MCP notification was not accepted")
                    return {}
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type == "text/event-stream":
                    return await self._read_sse(response, request_id)
                if content_type != "application/json":
                    raise ValueError("MCP response has unsupported content type")
                return self._result(await self._read_json(response), request_id)

    async def _read_json(self, response: Any) -> Any:
        blob = bytearray()
        async for chunk in response.aiter_bytes():
            if len(blob) + len(chunk) > self.MAX_RESPONSE_BYTES:
                raise ValueError("MCP response exceeds transport bound")
            blob.extend(chunk)
        return json.loads(blob)

    @staticmethod
    def _result(data: Any, request_id: int) -> Any:
        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0" or data.get("id") != request_id:
            raise ValueError("MCP response does not match its request")
        if "error" in data:
            error = data["error"]
            details = error.get("data") if isinstance(error, dict) else None
            supported = details.get("supported") if isinstance(details, dict) else None
            versions = tuple(str(x) for x in supported) if isinstance(supported, list) else ()
            raise McpProtocolError(error.get("code") if isinstance(error, dict) else None, versions)
        result = data.get("result")
        if not isinstance(result, dict):
            raise ValueError("MCP response result must be an object")
        if result.get("resultType", "complete") != "complete" or "inputRequests" in result:
            raise NotImplementedError("MCP multi-round-trip or task results require a host client")
        return result

    async def _read_sse(self, response: Any, request_id: int) -> Any:
        pending = bytearray()
        data_lines: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.MAX_RESPONSE_BYTES:
                raise ValueError("MCP response exceeds transport bound")
            pending.extend(chunk)
            while b"\n" in pending:
                line, _, rest = pending.partition(b"\n")
                pending = bytearray(rest)
                line = line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    data_lines.append(bytes(line[5:].lstrip(b" ")))
                elif not line and data_lines:
                    payload = b"\n".join(data_lines)
                    data_lines = []
                    if not payload:
                        continue
                    message = json.loads(payload)
                    if not isinstance(message, dict):
                        raise ValueError("MCP stream message must be an object")
                    if "method" in message:
                        if "id" in message:
                            raise ValueError("server-initiated MCP requests require a host client")
                        continue
                    return self._result(message, request_id)
        raise ValueError("MCP stream ended before its matching response")

    async def list_tools(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor = None
        seen: set[str] = set()
        while True:
            result = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            page = result.get("tools") or []
            if not isinstance(page, list) or not all(isinstance(tool, dict) for tool in page):
                raise ValueError("MCP tool catalogue must be a list of definitions")
            if len(out) + len(page) > self.MAX_TOOLS:
                raise ValueError("external tool catalogue exceeds bound")
            out.extend(page)
            cursor = result.get("nextCursor")
            if not cursor:
                if self.version == McpProtocolVersion.STATELESS:
                    headers: dict[str, list[tuple[tuple[str, ...], str, str]]] = {}
                    accepted = []
                    for definition in out:
                        try:
                            paths = _tool_header_paths(definition.get("inputSchema") or {})
                        except ValueError:
                            continue
                        accepted.append(definition)
                        headers[str(definition.get("name"))] = paths
                    self._tool_headers = headers
                    return accepted
                return out
            if not isinstance(cursor, str) or cursor in seen or len(seen) >= self.MAX_TOOLS:
                raise ValueError("MCP pagination did not terminate within bounds")
            seen.add(cursor)

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        return await self.request("tools/call", {"name": name, "arguments": dict(arguments)})


@dataclass
class ConnectedServer:
    config: ServerConfig
    policy: McpPolicy
    client: Any
    definitions: dict[str, dict[str, Any]] = field(default_factory=dict)
    status: McpServerStatus = McpServerStatus.CONNECTED
    approved: dict[str, str] = field(default_factory=dict)

    async def refresh(self) -> None:
        raw = await self.client.list_tools()
        if hasattr(raw, "tools"):
            raw = raw.tools
        if len(raw) > self.policy.max_tools:
            raise ValueError("external tool catalogue exceeds bound")
        definitions: dict[str, dict[str, Any]] = {}
        for tool in raw:
            definition = tool if isinstance(tool, dict) else tool.model_dump(by_alias=True, exclude_none=True)
            name = definition.get("name")
            if not isinstance(name, str) or not name or name in definitions:
                raise ValueError("external tools require unique nonempty names")
            if not isinstance(definition.get("inputSchema"), dict):
                raise ValueError("external tool requires an input schema")
            # A mutable host client's response must not mutate a pinned catalogue.
            definitions[name] = json.loads(json.dumps(definition))
        # Changed definitions immediately revoke approval, including wrappers
        # handed to an in-flight run before this refresh.
        self.approved = {
            name: pin for name, pin in self.approved.items() if name in definitions and digest(definitions[name]) == pin
        }
        for name, definition in definitions.items():
            if self.policy.approved.get(name) == digest(definition):
                self.approved[name] = digest(definition)
        self.definitions = definitions


class McpGateway:
    def __init__(self, *, client_factory: Callable[[ServerConfig], Any] | None = None) -> None:
        self.client_factory = client_factory
        self.servers: dict[str, ConnectedServer] = {}
        self._stack = AsyncExitStack()
        self._connecting: set[str] = set()
        self._closed = False

    async def __aenter__(self) -> McpGateway:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._closed = True
        try:
            await self._stack.aclose()
        finally:
            for server in self.servers.values():
                server.status = McpServerStatus.CLOSED

    async def connect(self, server: ServerConfig, policy: McpPolicy) -> ConnectedServer:
        if self._closed:
            raise RuntimeError("MCP gateway is closed")
        if server.identity in self.servers or server.identity in self._connecting:
            raise ValueError("server identity already connected")
        self._connecting.add(server.identity)
        attempt = AsyncExitStack()
        try:
            if self.client_factory:
                client = self.client_factory(server)
                if hasattr(client, "__await__"):
                    client = await client
                if hasattr(client, "aclose"):
                    attempt.push_async_callback(client.aclose)
            elif server.url:
                pinned, hostname = await resolve_public_url(server.url)
                client = StatelessMcpClient(server, pinned, hostname)
                attempt.push_async_callback(client.aclose)
                await client.negotiate()
            else:
                if not server.command or not server.sandbox_launcher:
                    raise PermissionError("stdio MCP needs an explicit sandbox launcher")
                from agentkit.integrations.mcp.client import MCPClient, StdioServer

                argv = server.sandbox_launcher(server.command)
                client = await attempt.enter_async_context(
                    MCPClient(StdioServer(argv[0], tuple(argv[1:]), dict(server.env)))
                )
            connected = ConnectedServer(server, policy, client)
            await connected.refresh()
        except BaseException as exc:
            await attempt.aclose()
            if server.required or not isinstance(exc, Exception):
                raise
            connected = ConnectedServer(server, policy, None, status=McpServerStatus.UNAVAILABLE)
        else:
            await self._stack.enter_async_context(attempt.pop_all())
        finally:
            self._connecting.discard(server.identity)
        self.servers[server.identity] = connected
        return connected

    def tools(self, approved_only: bool = True) -> ToolSet:
        registry = ToolRegistry()
        for connected in self.servers.values():
            if connected.status != McpServerStatus.CONNECTED:
                continue
            for name, definition in sorted(connected.definitions.items()):
                if not connected.policy.permits(connected.config.identity, name):
                    continue
                pin = digest(definition)
                if approved_only and connected.approved.get(name) != pin:
                    continue
                registry.register(_ExternalTool(connected, name, pin, definition))
        return ToolSet(registry)


class _ExternalTool:
    output_schema: Any = None
    side_effecting = True
    requires_approval = False

    def __init__(self, server: ConnectedServer, remote_name: str, pin: str, raw: dict[str, Any]) -> None:
        self.server, self.remote_name, self.pin = server, remote_name, pin
        self.name = f"mcp_{digest(server.config.identity)[:12]}_{remote_name}"
        self.description = (
            "Untrusted external tool: " + str(raw.get("description", ""))[: server.policy.max_description_chars]
        )
        self.schema = ToolSchema(self.name, self.description, raw.get("inputSchema") or {"type": "object"})
        self.output_schema = raw.get("outputSchema")

    async def run(self, args: Mapping[str, Any], ctx: Any) -> Any:
        ctx.check_cancelled()
        if self.server.status != McpServerStatus.CONNECTED:
            raise PermissionError("external MCP connection is closed or unavailable")
        current = self.server.definitions.get(self.remote_name)
        if current is None or digest(current) != self.pin or self.server.approved.get(self.remote_name) != self.pin:
            raise PermissionError("external tool definition changed; approval required")
        if not self.server.policy.permits(self.server.config.identity, self.remote_name):
            raise PermissionError("external tool is denied")
        if self.server.policy.approval:
            approved = self.server.policy.approval(self.server.config.identity, self.remote_name, args)
            if hasattr(approved, "__await__"):
                approved = await approved
            if not approved:
                raise PermissionError("external tool invocation was not approved")
        async with asyncio.timeout(self.server.config.timeout_s):
            result = await self.server.client.call_tool(self.remote_name, dict(args))
        raw = result if isinstance(result, dict) else result.model_dump(by_alias=True, exclude_none=True)
        if raw.get("resultType", "complete") != "complete" or "inputRequests" in raw:
            raise NotImplementedError("MCP multi-round-trip or task results require a host client")
        return ToolResult(tuple(raw.get("content") or []), raw.get("structuredContent"), bool(raw.get("isError")))
