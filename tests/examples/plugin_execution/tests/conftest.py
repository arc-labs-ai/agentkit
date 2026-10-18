"""Provider doubles only. No tool dispatch or application model loop.

Two shapes, because the four modes reach a host tool over two different wires.

`scripted_client` is an `httpx.MockTransport` behind a real `ApiAdapter`: the
API modes receive the granted schema, emit a provider tool call and read back
whatever the host returned.

`scripted_cli` stands in for the CLI BINARY at the spawn seam, the way
`agentkit.core.testing.fakes` does for AgentKit's own CLI tests. It is handed
the argv and environment a real `claude`/`codex` process would have received,
reads the host-tool MCP endpoint `CliAdapter` published there for that process,
and calls the tool over that same authenticated loopback transport. What that
proves is the TRANSPORT and the single policy path behind it: the bridge really
does arrive at `ToolSet.execute` with this call's permission contributions and
observers attached, rather than at a second copy of the decision. What it
cannot prove is containment of a real CLI — there is no process, no native
tool, no sandbox and no credential here. Real-process characterization lives in
`tests/source/integration/execution`, and the refusal this example's strict
profile produces for both CLI modes is pinned next door in
`test_cli_containment.py`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentkit.core.testing.fakes import FakeClaudeCli, FakeCodexCli, codex_turn


@pytest.fixture
def scripted_client():
    clients = []

    def build(mode, *, record_id="record-1", tool_name="lookup_record", status="ready", answer_record_id=None):
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            answer = json.dumps({"record_id": record_id if answer_record_id is None else answer_record_id, "status": status})
            arguments = json.dumps({"record_id": record_id})
            first = len(requests) == 1
            if mode == "anthropic_api":
                block = (
                    {"type": "tool_use", "id": "provider-call-1", "name": tool_name, "input": {}}
                    if first
                    else {"type": "text", "text": ""}
                )
                delta = (
                    {"type": "input_json_delta", "partial_json": arguments}
                    if first
                    else {"type": "text_delta", "text": answer}
                )
                events = [
                    {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
                    {"type": "content_block_start", "index": 0, "content_block": block},
                    {"type": "content_block_delta", "index": 0, "delta": delta},
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use" if first else "end_turn"},
                        "usage": {"output_tokens": 1},
                    },
                    {"type": "message_stop"},
                ]
            else:
                output = (
                    [{"type": "function_call", "call_id": "provider-call-1", "name": tool_name, "arguments": arguments}]
                    if first
                    else [
                        {"type": "message", "content": [{"type": "output_text", "text": answer}]},
                    ]
                )
                events = [
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "output": output,
                            "usage": {"input_tokens": 2, "output_tokens": 1},
                        },
                    }
                ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
        clients.append(client)
        return client, requests

    yield build
    # MockTransport owns no sockets; each async test closes its client.
    assert all(client.is_closed for client in clients)


@dataclass
class ScriptedCli:
    """One scripted CLI session and what it observed on the host-tool bridge.

    `results` holds one MCP `CallToolResult` per host tool call the session
    completed — the refusal text and `code` a real CLI would have handed its
    model. `cancellations` names the tools whose in-flight call the host tore
    down instead of answering. `spawns` is the argv of each spawn, which the
    fixture checks is non-empty: the example's strict profile refuses both CLI
    modes BEFORE dispatch, so a CLI test that never reached the binary would
    otherwise assert "no effect" about a call that never started.
    """

    spawn: Callable[..., Awaitable[Any]]
    results: list[Any] = field(default_factory=list)
    cancellations: list[str] = field(default_factory=list)
    spawns: list[tuple[str, ...]] = field(default_factory=list)


def _toml_string(value: str) -> str:
    """Codex config values arrive as TOML basic strings (`key="value"`)."""
    return json.loads(value) if value.startswith('"') else value


async def _host_tool_endpoint(argv: tuple[str, ...], env: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    """The MCP endpoint and credential this process was pointed at.

    Both CLIs are told about the same server in their own spelling: Claude Code
    reads a `--mcp-config` document, Codex reads `-c mcp_servers.<name>.<key>`
    overrides and takes the bearer token out of a named environment variable.
    A double that hard-coded either one would stop being a double the day the
    adapter changed how it wires the server in.
    """
    if "--mcp-config" in argv:
        path = Path(argv[argv.index("--mcp-config") + 1])
        document = json.loads(await asyncio.to_thread(path.read_text))
        entry = next(iter(document["mcpServers"].values()))
        return entry["url"], dict(entry.get("headers", {}))
    table = {
        key.rpartition(".")[2]: _toml_string(value)
        for key, _, value in (argv[index + 1].partition("=") for index, arg in enumerate(argv) if arg == "-c")
        if key.startswith("mcp_servers.")
    }
    return table["url"], {"Authorization": f"Bearer {env[table['bearer_token_env_var']]}"}


async def _call_host_tool(url: str, headers: Mapping[str, str], name: str, arguments: Mapping[str, Any]) -> Any:
    """Call one host tool the way the binary would: an MCP client, nothing else.

    A plain MCP client, and its own `httpx.AsyncClient` with `trust_env=False`,
    for the same reason `scripted_client` sets it: the server is on loopback and
    an ambient `HTTP_PROXY` in a reader's shell must not decide whether this
    example runs. `agentkit.integrations.mcp.MCPClient` would do the transport
    wiring, but it owns its client and honours the environment.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(headers=dict(headers), timeout=10, trust_env=False) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(name, dict(arguments))


@pytest.fixture
def scripted_cli():
    sessions: list[ScriptedCli] = []

    def build(mode, *, record_id="record-1", tool_name="lookup_record", status="ready", answer_record_id=None):
        answer = json.dumps({"record_id": record_id if answer_record_id is None else answer_record_id, "status": status})
        if mode == "claude_code":
            replay = FakeClaudeCli.script(
                [
                    {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": answer}]}},
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": answer,
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    },
                ]
            )
        elif mode == "codex":
            replay = FakeCodexCli.script(codex_turn(text=answer, usage=(2, 0, 1)))
        else:
            raise ValueError(f"{mode} is not a CLI mode; use scripted_client")

        async def spawn(*argv, **kwargs):
            session.spawns.append(argv)
            url, headers = await _host_tool_endpoint(argv, kwargs.get("env") or {})
            try:
                session.results.append(await _call_host_tool(url, headers, tool_name, {"record_id": record_id}))
            except asyncio.CancelledError:
                # The adapter cancelled startup while this request was pending.
                session.cancellations.append(tool_name)
                raise
            return await replay(*argv, **kwargs)

        session = ScriptedCli(spawn)
        sessions.append(session)
        return session

    yield build
    # Every double must have been reached. A CLI call refused before dispatch
    # satisfies "no effect" without ever starting, and that is the one way
    # these tests could pass while proving nothing. Each session closed its own
    # MCP client; the adapter owns the server.
    assert all(session.spawns for session in sessions)
