"""Codex adapter/MCP protocol probe, local provider and fake API key only.

This runs the ordinary WARN profile. Native-tool absence and containment stay
unsupported regardless of successful host tool dispatch.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import struct
import zlib
from dataclasses import replace

import pytest
from pydantic import BaseModel

from agentkit.core.runtime.calls import CallRuntime
from agentkit.models import Binding, CallLimits, CredentialRef, ExecutionProfile, ExecutionRequirements
from agentkit.models.providers.cli import CliAdapter, SubprocessExecutionBackend
from agentkit.tools import PermissionDecision, ToolRegistry, tool
from agentkit.tools.execution.policy import ToolSet


@pytest.mark.asyncio
@pytest.mark.timeout(50)
@pytest.mark.skipif(
    shutil.which("codex") is None or os.getenv("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="real Codex CLI unavailable or explicitly disabled",
)
@pytest.mark.parametrize("native_read", [False, True])
async def test_actual_codex_adapter_dispatches_granted_host_tool_with_typed_answer(tmp_path, monkeypatch, native_read):
    from mcp import types

    from agentkit.integrations.mcp.serve import McpServerSpec

    class Answer(BaseModel):
        status: str

    work = tmp_path / "job"
    work.mkdir()
    outside = tmp_path / "other-job"
    outside.mkdir()

    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))

    # A valid, synthetic image outside this job, with no actual credential data.
    image = outside / "private-canary.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00\xff"))
        + chunk(b"IEND", b"")
    )

    effects, requests, failures = [], [], []
    forwarded_provider_identity = []
    original_build = McpServerSpec.build_server

    def build(spec):
        server = original_build(spec)
        original_call = server.request_handlers[types.CallToolRequest]

        async def capture(request):
            metadata = request.params.meta.model_dump() if request.params.meta is not None else {}
            # Do not infer a provider identity from the JSON-RPC request ID.
            # Detect if a future binary begins forwarding the known provider
            # call ID under metadata which the adapter must then preserve.
            forwarded_provider_identity.append(
                [key for key, value in metadata.items() if value in ("call_local", "call_denied", "call_unattended")]
            )
            return await original_call(request)

        server.request_handlers[types.CallToolRequest] = capture
        return server

    monkeypatch.setattr(McpServerSpec, "build_server", build)

    @tool(side_effecting=False)
    def allowed(ctx, value: str = "clean") -> str:
        """Perform the explicitly granted local test operation."""
        effects.append(ctx.tool_invocation)
        return "host-operation-completed"

    policy_events = []

    def permission(request):
        return {"denied": PermissionDecision.DENY, "unattended": PermissionDecision.ASK}.get(
            request.arguments.get("value"), PermissionDecision.ALLOW
        )

    def policy():
        return ToolSet(ToolRegistry.from_tools([allowed]), permissions=(permission,), observers=(policy_events.append,))

    async def provider(reader, writer):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            length = next(
                (
                    int(line.split(b":", 1)[1])
                    for line in header.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            body = json.loads(await reader.readexactly(length)) if length else {}
            requests.append(body)
            names = {entry.get("name", "") for entry in body.get("tools", [])}
            if len(requests) == 1:
                namespace = next(entry for entry in body["tools"] if entry.get("name") == "mcp__agentkit")
                assert [tool["name"] for tool in namespace["tools"]] == ["allowed"]
                assert "view_image" in names, "WARN must not be mislabeled as native-tool absence"
                item = {
                    "type": "function_call",
                    "id": "fc_local",
                    "call_id": "call_local",
                    "namespace": "mcp__agentkit",
                    "name": "allowed",
                    "arguments": "{}",
                }
                items = [
                    item,
                    *(
                        {
                            "type": "function_call",
                            "id": "fc_" + value,
                            "call_id": "call_" + value,
                            "namespace": "mcp__agentkit",
                            "name": "allowed",
                            "arguments": json.dumps({"value": value}),
                        }
                        for value in ("denied", "unattended")
                    ),
                ]
                if native_read:
                    items.append(
                        {
                            "type": "function_call",
                            "id": "fc_image",
                            "call_id": "call_native_image",
                            "name": "view_image",
                            "arguments": json.dumps({"path": str(image)}),
                        }
                    )
            else:
                item = {
                    "type": "message",
                    "id": "msg_local",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": '{"status":"ok"}', "annotations": []}],
                }
                items = [item]
            events = [
                {"type": "response.created", "response": {"id": f"resp_{len(requests)}"}},
                *(
                    {"type": "response.output_item.done", "output_index": i, "item": value}
                    for i, value in enumerate(items)
                ),
                {
                    "type": "response.completed",
                    "response": {
                        "id": f"resp_{len(requests)}",
                        "status": "completed",
                        "output": items,
                        "usage": {
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "total_tokens": 3,
                            "input_tokens_details": {"cached_tokens": 0},
                        },
                    },
                },
            ]
            encoded = "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                + str(len(encoded)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + encoded
            )
            await writer.drain()
        except Exception as exc:
            failures.append(type(exc).__name__)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(provider, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    class LocalProvider(SubprocessExecutionBackend):
        async def start(self, invocation):
            assert invocation.execution.profile == ExecutionProfile.CLI_COMPATIBLE
            assert invocation.native_tool_policy == "warn"
            # Select a test-only local transport. This grants no additional
            # execution authority and uses no bypass/sandbox-disabling flags.
            config = {
                "model_provider": "local_test",
                "model_providers.local_test.name": "local_test",
                "model_providers.local_test.base_url": f"http://127.0.0.1:{port}/v1",
                "model_providers.local_test.env_key": "CODEX_API_KEY",
                "model_providers.local_test.wire_api": "responses",
                "features.enable_request_compression": False,
                "model_providers.local_test.request_max_retries": 0,
                "model_providers.local_test.stream_max_retries": 0,
            }
            args = []
            for key, value in config.items():
                args.extend(["-c", f"{key}={json.dumps(value)}"])
            position = invocation.argv.index("exec") + 1
            return await super().start(
                replace(invocation, argv=(*invocation.argv[:position], *args, *invocation.argv[position:]))
            )

    try:
        async with CallRuntime(
            adapters={"codex": CliAdapter(execution_backend=LocalProvider(), working_dir=work)},
            unpriced_policy="proceed_unpriced",
        ) as runtime:
            result = await runtime.run(
                Binding("codex", "gpt-5.4", CredentialRef(lambda: {"OPENAI_API_KEY": "fake-local-only"})),
                "Run the host operation and return the typed status.",
                policy(),
                output=Answer,
                call_id="offline-codex",
                limits=CallLimits(max_turns=2, wall_clock_s=35),
                execution=ExecutionRequirements(profile=ExecutionProfile.CLI_COMPATIBLE),
            )
    finally:
        server.close()
        await server.wait_closed()
    assert not failures, [
        [(entry.get("type"), entry.get("name")) for entry in body.get("tools", [])] for body in requests
    ]
    assert result.outcome == "answered", (result.failure, len(requests))
    assert result.session.execution.profile == ExecutionProfile.CLI_COMPATIBLE
    assert result.value == Answer(status="ok") and len(effects) == 1
    assert effects[0].host_invocation_id and effects[0].transport_request_id is not None
    # 0.162.1 forwards the Responses call_id in MCP _meta.callId. The host
    # handler must receive it, separately from the JSON-RPC and host IDs.
    assert forwarded_provider_identity == [["callId"], ["callId"], ["callId"]]
    assert effects[0].provider_tool_call_id == "call_local"
    assert len(requests) == 2
    finished = [event for event in policy_events if event.phase == "finished"]
    assert sorted(event.code for event in finished) == ["ok", "permission_denied", "permission_unattended"]
    # Actual native execution, not merely an advertised tool: the CLI reads an
    # outside image and sends its contents to the local provider under WARN.
    # Strict DENY must keep refusing this profile.
    assert ("data:image/" in json.dumps(requests[1])) is native_read
