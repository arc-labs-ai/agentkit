"""The fake CLI invokes agentkit's actual hook process, without model calls."""

import asyncio
import json
import shlex
from pathlib import Path

import pytest

from agentkit.models.contracts.types import Allow, Deny
from agentkit.models.providers.cli.claude.native_hooks import NativeToolFence


async def hook(command, payload):
    child = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await child.communicate(json.dumps(payload).encode())
    assert child.returncode == 0, stderr.decode()
    return json.loads(stdout)


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_callback", [False, True])
async def test_native_callback_denial_reaches_claude_hook_shape_and_marks_refused_event(raise_callback):
    invoked, events = [], []

    async def policy(tool, arguments):
        invoked.append((tool, dict(arguments)))
        if raise_callback:
            raise RuntimeError("synthetic-sensitive-detail")
        return Deny("destructive writes are prohibited")

    async with NativeToolFence(policy, events.append) as fence:
        command = fence.settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        result = await hook(
            command,
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": "rm sample"},
                "tool_use_id": "native-1",
            },
        )
    assert invoked == [("Bash", {"command": "rm sample"})]
    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse" and output["permissionDecision"] == "deny"
    assert "synthetic-sensitive-detail" not in json.dumps(result)
    assert events[-1].kind == "tool_call" and events[-1].fields["refused"] is True
    assert events[-1].fields["tool_call_id"] == "native-1"
    assert "argument_digest" not in events[-1].fields and "arguments" not in events[-1].fields


@pytest.mark.asyncio
async def test_native_callback_allow_and_private_config_cleanup():
    async def policy(tool, arguments):
        return Allow()

    async with NativeToolFence(policy, lambda _: None) as fence:
        command = fence.settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        parts = shlex.split(command)
        config = Path(parts[parts.index("--client-config") + 1])
        assert (await asyncio.to_thread(config.stat)).st_mode & 0o777 == 0o600
        result = await hook(command, {"tool_name": "Read", "tool_input": {"file_path": "sample"}})
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert not await asyncio.to_thread(config.exists)
    disconnected = await hook(command, {"tool_name": "Read", "tool_input": {"file_path": "sample"}})
    assert disconnected["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_invalid_callback_result_and_malformed_hook_input_deny():
    async def invalid(tool, arguments):
        return True

    async with NativeToolFence(invalid, lambda _: None) as fence:
        command = fence.settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        for payload in ({"tool_name": "Bash", "tool_input": {}}, {"tool_name": "Bash", "tool_input": []}):
            result = await hook(command, payload)
            assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_fake_claude_cli_invokes_workspace_fence_and_preserves_existing_hooks():
    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.core.testing.fakes import FakeClaudeCli
    from agentkit.models.contracts.constants import NativeToolPolicy
    from agentkit.models.contracts.types import Binding, CredentialRef, Workspace
    from agentkit.models.providers.cli.calls import CliAdapter
    from tests.source.conformance.test_calls import PRICES

    seen, decisions, events = [], [], []

    async def policy(name, arguments):
        seen.append((name, dict(arguments)))
        return Deny("production write refused")

    fake = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "write blocked"}]}},
            {
                "type": "result",
                "subtype": "success",
                "result": "write blocked",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ]
    )
    retained = {"PostToolUse": [{"matcher": "Read", "hooks": []}]}

    async def spawn(*argv, **kwargs):
        settings_path = argv[argv.index("--settings") + 1]
        settings = json.loads(await asyncio.to_thread(Path(settings_path).read_text))
        assert settings["hooks"]["PostToolUse"] == retained["PostToolUse"]
        assert "--bare" not in argv
        command = settings["hooks"]["PreToolUse"][-1]["hooks"][0]["command"]
        result = await hook(
            command, {"tool_name": "Bash", "tool_input": {"command": "write production"}, "tool_use_id": "native-gate"}
        )
        decisions.append(result["hookSpecificOutput"])
        return await fake(*argv, **kwargs)

    async with CallRuntime(
        adapters={"claude_code": CliAdapter(spawn=spawn)}, pricing=PRICES, native_tool_policy=NativeToolPolicy.ALLOW
    ) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "synthetic"})),
            "attempt a write",
            workspace=Workspace(native_tools=("Bash",), hooks=retained, pre_tool_use=policy),
            native_tool_policy=NativeToolPolicy.ALLOW,
            events=events.append,
            call_id="native-fence",
        )
    assert answer.outcome == "answered" and answer.text == "write blocked"
    assert seen == [("Bash", {"command": "write production"})]
    assert decisions == [
        {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "production write refused",
        }
    ]
    assert any(e.kind == "tool_call" and e.fields.get("refused") for e in events)
    assert retained == {"PostToolUse": [{"matcher": "Read", "hooks": []}]}


@pytest.mark.asyncio
async def test_callback_timeout_denies_and_awaits_callback_cleanup(monkeypatch):
    from agentkit.models.providers.cli.claude import native_hooks

    monkeypatch.setattr(native_hooks, "_CALLBACK_TIMEOUT_S", 0.02)
    cleaned = asyncio.Event()

    async def policy(tool, arguments):
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async with NativeToolFence(policy, lambda _: None) as fence:
        command = fence.settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        result = await hook(command, {"tool_name": "Bash", "tool_input": {}})
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert cleaned.is_set()


@pytest.mark.asyncio
async def test_unauthenticated_request_denies_without_invoking_callback():
    calls = []

    async def policy(tool, arguments):
        calls.append(tool)
        return Allow()

    async with NativeToolFence(policy, lambda _: None) as fence:
        command = fence.settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        parts = shlex.split(command)
        config = json.loads(await asyncio.to_thread(Path(parts[parts.index("--client-config") + 1]).read_text))
        reader, writer = await asyncio.open_unix_connection(config["socket"])
        writer.write(json.dumps({"token": "untrusted", "payload": {"tool_name": "Bash", "tool_input": {}}}).encode() + b"\n")
        await writer.drain()
        result = json.loads(await reader.readline())
        writer.close()
        await writer.wait_closed()
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert calls == []


@pytest.mark.asyncio
async def test_missing_client_script_emits_explicit_deny_instead_of_nonblocking_hook_error(monkeypatch):
    from agentkit.models.providers.cli.claude import native_hooks

    monkeypatch.setattr(native_hooks, "_CLIENT_SCRIPT", "/nonexistent/agentkit-owned-hook.py")

    async def policy(tool, arguments):
        return Allow()

    async with NativeToolFence(policy, lambda _: None) as fence:
        command = fence.settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        result = await hook(command, {"tool_name": "Bash", "tool_input": {}})
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_mcp_tools_keep_existing_host_policy_without_native_callback():
    async def policy(tool, arguments):
        pytest.fail("MCP tools must use hosted tool authorization, not native policy")

    events = []
    async with NativeToolFence(policy, events.append) as fence:
        command = fence.settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        result = await hook(command, {"tool_name": "mcp__agentkit__refund", "tool_input": {}})
        assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert events == []
