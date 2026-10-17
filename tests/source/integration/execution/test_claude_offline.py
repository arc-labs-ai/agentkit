"""Real Claude CLI against a local scripted provider, no paid calls or credentials.

This proves only native tool denial for this exact invocation/version. It does
not attest saved OAuth, MCP, configuration, egress or filesystem containment.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil

import pytest

from agentkit.models.providers.cli._common import _terminate


@pytest.mark.asyncio
@pytest.mark.skipif(
    shutil.which("claude") is None or os.getenv("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="real Claude CLI unavailable or explicitly disabled",
)
async def test_real_claude_empty_tool_list_refuses_planted_bash_and_read(tmp_path):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    marker = tmp_path / "prohibited-write"
    fake_credential = home / ".credentials.json"
    fake_credential.write_text('{"planted":"NOT-A-REAL-CREDENTIAL"}')
    # Plant both user/project startup hooks and project-discovered MCP.
    import shlex

    hook_marker = tmp_path / "prohibited-hook"
    mcp_marker = tmp_path / "prohibited-mcp"
    hook = {
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "touch " + shlex.quote(str(hook_marker)),
                        }
                    ]
                }
            ]
        }
    }
    (home / "settings.json").write_text(json.dumps(hook))
    (work / ".claude").mkdir()
    (work / ".claude" / "settings.json").write_text(json.dumps(hook))
    (work / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "planted": {
                        "command": "/usr/bin/touch",
                        "args": [str(mcp_marker)],
                    }
                }
            }
        )
    )
    requests = []
    failures = []

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
            # Deliberately request two prohibited tools, even though unadvertised.
            calls = (
                [
                    ("Bash", {"command": "touch " + str(marker)}),
                    ("Read", {"file_path": str(fake_credential)}),
                ]
                if len(requests) == 1
                else []
            )
            blocks = (
                [
                    {"type": "tool_use", "id": f"planted-{i}", "name": name, "input": arguments}
                    for i, (name, arguments) in enumerate(calls)
                ]
                if calls
                else [{"type": "text", "text": "completed local protocol test"}]
            )
            events = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_local",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                }
            ]
            for i, block in enumerate(blocks):
                events.extend(
                    [
                        {"type": "content_block_start", "index": i, "content_block": block},
                        {"type": "content_block_stop", "index": i},
                    ]
                )
            events.extend(
                [
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    },
                    {"type": "message_stop"},
                ]
            )
            data = "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events)
            encoded = data.encode()
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
    env = {
        "PATH": os.defpath,
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(home),
        "ANTHROPIC_API_KEY": "test-only-not-real",
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
    }
    process = await asyncio.create_subprocess_exec(
        shutil.which("claude"),
        "--bare",
        "--tools",
        "",
        "--disable-slash-commands",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        "claude-sonnet-4-5",
        "--max-turns",
        "3",
        "-p",
        "Run the local protocol test",
        env=env,
        cwd=work,
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 25)
    finally:
        await _terminate(process, 0.1)
        server.close()
        await server.wait_closed()
    assert not failures
    assert requests, "The planted provider response must actually reach the CLI"
    assert not marker.exists()
    assert not hook_marker.exists() and not mcp_marker.exists()
    assert "NOT-A-REAL-CREDENTIAL" not in output.decode()
    for request in requests:
        names = {tool["name"] for tool in request.get("tools", [])}
        assert not names.intersection({"Bash", "Read", "Edit", "Write", "WebFetch", "WebSearch"})
        assert "NOT-A-REAL-CREDENTIAL" not in json.dumps(request)
    assert any(json.loads(line).get("type") == "result" for line in output.splitlines() if line.startswith(b"{"))
