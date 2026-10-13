from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentkit.agents.prompts.composition import ImagePart, inline_prompt
from agentkit.core.contracts.types import Mode
from agentkit.core.execution.concurrency import CancellationToken
from agentkit.core.runtime.calls import CallRuntime, RetryPolicy, _mask_tool_arguments
from agentkit.core.testing.fakes import CliRun, FakeClaudeCli, FakeCodexCli, codex_turn
from agentkit.models.catalog.pricing import PriceTable, normalize_usage, price
from agentkit.models.contracts.constants import CostSource, EventKind, FailureClass, UnpricedPolicy
from agentkit.models.contracts.schema import OutputSpec, clean_nulls, compile_schema, tolerant_json_object
from agentkit.models.contracts.types import (
    Binding,
    CallEvent,
    CallLimits,
    Credential,
    CredentialRef,
    Failure,
    ModelSettings,
    Turn,
    Workspace,
)
from agentkit.models.providers.cli.calls import CliAdapter, _classify_cli_failure
from agentkit.models.providers.frontiers.calls import ApiAdapter

MODES = ("claude_code", "codex", "anthropic_api", "openai_api")
PRICES = PriceTable(
    "test",
    "2026-10-13",
    {
        "test": {
            "input_uncached": "1",
            "input_cache_read": "0.1",
            "input_cache_write_5m": "1.25",
            "input_cache_write_1h": "2",
            "output": "5",
        }
    },
)


@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
def test_cli_rate_limit_retry_after_and_bad_request_are_distinguished(mode) -> None:
    limited = _classify_cli_failure(mode, "turn_failed", {"api_error_status": 429, "retry_after_s": "0.025"})
    invalid = _classify_cli_failure(mode, "turn_failed", {"api_error_status": 400})

    assert limited.classification == FailureClass.RATE_LIMITED
    assert limited.retryable is True
    assert limited.retry_after_s == 0.025
    assert invalid.classification == FailureClass.INVALID_REQUEST
    assert invalid.retryable is False


SCHEMA = {
    "type": "object",
    "properties": {
        "required": {"type": ["string", "null"]},
        "nested": {"anyOf": [{"$ref": "#/$defs/nested"}, {"type": "string"}]},
        "optional": {"type": "string"},
    },
    "required": ["required", "nested"],
    "additionalProperties": False,
    "$defs": {
        "nested": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "note": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }
    },
}


class ScriptedAdapter:
    def __init__(self, *turns: Turn) -> None:
        self.turns = list(turns)
        self.requests = []

    async def turn(self, binding: Any, prompt: Any, **kwargs: Any) -> Turn:
        self.requests.append((binding, prompt, kwargs))
        if not self.turns:
            raise AssertionError("unexpected adapter call")
        return self.turns.pop(0)


def sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)


def real_adapter(mode: str, text: str, *, value: Any = None, commentary: str = "Working...") -> Any:
    if mode == "claude_code":
        payload = [
            {"type": "assistant", "message": {"id": "first", "content": [{"type": "text", "text": commentary}]}},
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": text}]}},
            {
                "type": "result",
                "subtype": "success",
                "result": text,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "structured_output": value,
                "session_id": "test-session",
                "total_cost_usd": 0.000035,
            },
        ]
        return CliAdapter(spawn=FakeClaudeCli.script(payload))
    if mode == "codex":
        return CliAdapter(
            spawn=FakeCodexCli.script(
                codex_turn(
                    text=text,
                    usage=(10, 0, 5),
                    items=[{"id": "commentary", "type": "agent_message", "text": commentary}],
                )
            )
        )
    if mode == "openai_api":
        events = [
            {"type": "response.output_text.delta", "delta": text},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": commentary}]},
                        {"type": "message", "content": [{"type": "output_text", "text": text}]},
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        ]
    else:
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=sse(events))))
    return ApiAdapter(client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_final_answer_through_actual_translators(mode: str) -> None:
    adapter = real_adapter(mode, "42")
    binding = Binding(
        mode,
        "test",
        CredentialRef(
            lambda: (
                {"ANTHROPIC_API_KEY": "test"}
                if mode in ("claude_code", "anthropic_api")
                else {"OPENAI_API_KEY": "test"}
            )
        ),
    )
    events = []
    try:
        async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
            answer = await runtime.ask(binding, "answer", events=events.append, call_id="test")
        assert answer.outcome == "answered", answer.failure
        assert answer.text == "42"
        assert answer.usage.input_uncached == 10
        assert answer.usage.output == 5
        assert events[0].kind == "call_started" and events[-1].kind == "call_finished"
        assert [event.kind for event in events if event.kind in ("attempt_started", "attempt_finished")] == [
            "attempt_started",
            "attempt_finished",
        ]
        assert not any("text" in e.fields for e in events)
    finally:
        if hasattr(adapter, "client"):
            await adapter.client.aclose()


@pytest.mark.asyncio
async def test_unpriced_policy_requires_a_cap_and_preserves_known_partial_cost() -> None:
    binding = Binding("openai_api", "unlisted-model")
    no_cap = ScriptedAdapter(Turn(messages=("answer",), raw_usage={"input_tokens": 5, "output_tokens": 3}))
    async with CallRuntime(adapters={"openai_api": no_cap}, unpriced_policy=UnpricedPolicy.PROCEED_UNPRICED) as runtime:
        refused = await runtime.run(binding, "q", call_id="unpriced-no-cap")
    assert refused.outcome == "failed"
    assert not no_cap.requests

    partial_prices = PriceTable("partial", "2026-10-13", {"unlisted-model": {"input_uncached": "2"}})
    adapter = ScriptedAdapter(Turn(messages=("answer",), raw_usage={"input_tokens": 5, "output_tokens": 3}))
    async with CallRuntime(
        adapters={"openai_api": adapter},
        pricing=partial_prices,
        unpriced_policy=UnpricedPolicy.PROCEED_UNPRICED,
    ) as runtime:
        answer = await runtime.run(binding, "q", limits=CallLimits(max_turns=1), call_id="unpriced-partial")
    assert answer.outcome == "answered"
    assert answer.cost_usd == Decimal("0.00001")
    assert answer.cost_source == CostSource.MIXED


@pytest.mark.asyncio
async def test_output_tolerant_decode_and_validator_correction() -> None:
    spec = OutputSpec(
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        validators=(lambda value: [] if value["x"] == 2 else ["x must be 2"],),
    )
    adapter = ScriptedAdapter(
        Turn(messages=('```json\n{"x": 1}\n```',), raw_usage={"input_tokens": 2, "output_tokens": 2}),
        Turn(messages=('prefix {"x": 2}',), raw_usage={"input_tokens": 2, "output_tokens": 2}),
    )
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding("openai_api", "test"), "return json", spec, call_id="validated-json")
    assert answer.outcome == "answered"
    assert answer.value == {"x": 2}
    assert len(adapter.requests) == 2


def test_tolerant_json_refuses_multiple_objects_and_ignores_stray_closing_brace() -> None:
    with pytest.raises(ValueError, match="unambiguous"):
        tolerant_json_object('draft {"x": 1}; final {"x": 2}')
    assert tolerant_json_object('final {"x": 2}}') == {"x": 2}


@pytest.mark.asyncio
async def test_structured_output_can_use_more_than_one_correction() -> None:
    output = OutputSpec(
        {"type": "object", "properties": {"x": {"type": "integer", "minimum": 1}}, "required": ["x"]},
        max_corrections=2,
    )
    adapter = ScriptedAdapter(Turn(("not json",)), Turn(('prefix {"x": 0}',)), Turn(('final {"x": 3}',)))
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding("openai_api", "test"), "JSON", output, call_id="multi-correction")
    assert answer.outcome == "answered" and answer.value == {"x": 3}
    assert len(adapter.requests) == 3


@pytest.mark.asyncio
async def test_call_runtime_bounds_cli_concurrency_across_unique_homes() -> None:
    entered = 0
    peak = 0
    acquired = 0
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowAdapter:
        async def turn(self, binding, prompt, **kwargs):
            nonlocal entered, peak
            entered += 1
            peak = max(peak, entered)
            started.set()
            await release.wait()
            entered -= 1
            return Turn(messages=("ok",), raw_usage={"input_tokens": 1, "output_tokens": 1})

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def acquire(mode):
        nonlocal acquired
        acquired += 1
        yield

    runtime = CallRuntime(adapters={"codex": SlowAdapter()}, pricing=PRICES, max_concurrent_cli=1, acquire=acquire)
    async with runtime:
        first = asyncio.create_task(runtime.run(Binding("codex", "test"), "q", call_id="one"))
        await started.wait()
        second = asyncio.create_task(runtime.run(Binding("codex", "test"), "q", call_id="two"))
        await asyncio.sleep(0)
        assert peak == entered == 1
        release.set()
        answers = await asyncio.gather(first, second)
    assert all(answer.outcome == "answered" for answer in answers)
    assert peak == 1
    assert acquired == 2


@pytest.mark.asyncio
async def test_cli_auth_retry_requires_a_rotated_credential() -> None:
    calls = 0

    def credentials():
        nonlocal calls
        calls += 1
        return {"ANTHROPIC_API_KEY": "old" if calls == 1 else "rotated"}

    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    {"type": "result", "subtype": "authentication_error", "is_error": True, "result": ""},
                ]
            ),
            CliRun.of(
                [
                    {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "ok"}]}},
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "ok",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ]
            ),
        ]
    )
    adapter = CliAdapter(spawn=cli)
    async with CallRuntime(
        adapters={"claude_code": adapter},
        pricing=PRICES,
        retry=RetryPolicy(max_attempts=3, base_s=0, cap_s=0, max_elapsed_s=2),
    ) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(credentials)), "task", call_id="auth-rotation"
        )
    assert answer.outcome == "answered"
    assert answer.attempts == 2
    assert [invocation.env["ANTHROPIC_API_KEY"] for invocation in cli.invocations] == ["old", "rotated"]


@pytest.mark.asyncio
async def test_cli_auth_failure_without_rotation_is_not_retried() -> None:
    cli = FakeClaudeCli.script([{"type": "result", "subtype": "authentication_error", "is_error": True, "result": ""}])
    adapter = CliAdapter(spawn=cli)
    async with CallRuntime(
        adapters={"claude_code": adapter},
        pricing=PRICES,
        retry=RetryPolicy(max_attempts=3, base_s=0, cap_s=0, max_elapsed_s=2),
    ) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "same"})),
            "task",
            call_id="auth-no-rotation",
        )
    assert answer.outcome == "failed"
    assert answer.attempts == 1


@pytest.mark.asyncio
async def test_claude_no_tools_supplies_a_real_empty_mcp_servers_record() -> None:
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "ok"}]}},
            {"type": "result", "subtype": "success", "result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    adapter = CliAdapter(spawn=cli, transport={"HTTPS_PROXY": "https://proxy.example"})
    async with CallRuntime(adapters={"claude_code": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            "task",
            call_id="empty-mcp-record",
        )
    assert answer.outcome == "answered"
    argv = cli.invocations[0].argv
    assert "--mcp-config" in argv
    assert cli.validated_mcp_configs == [{"mcpServers": {}}]
    assert cli.invocations[0].env["HTTPS_PROXY"] == "https://proxy.example"
    assert len(cli.invocations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credential_values", "expect_bare"),
    [
        ({"ANTHROPIC_API_KEY": "test-api-key"}, True),
        ({"ANTHROPIC_AUTH_TOKEN": "test-oauth-token"}, False),
    ],
)
async def test_claude_bare_mode_respects_api_key_vs_oauth_credentials(credential_values, expect_bare) -> None:
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "ok"}]}},
            {"type": "result", "subtype": "success", "result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    adapter = CliAdapter(spawn=cli)
    async with CallRuntime(adapters={"claude_code": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: credential_values)),
            "reply",
            call_id="claude-auth-kind",
        )
    assert answer.outcome == "answered", answer.failure
    argv = cli.invocations[0].argv
    assert ("--bare" in argv) is expect_bare
    if not expect_bare:
        assert "--strict-mcp-config" in argv
        assert "--mcp-config" in argv
    env = cli.invocations[0].env
    assert ("ANTHROPIC_API_KEY" in env) is expect_bare
    assert ("ANTHROPIC_AUTH_TOKEN" in env) is not expect_bare


@pytest.mark.asyncio
async def test_cli_async_credential_resolver_and_unsafe_environment_refuse_before_spawn() -> None:
    spawned = 0

    async def spawn(*args, **kwargs):
        nonlocal spawned
        spawned += 1
        raise AssertionError("invalid credential must be refused before spawn")

    async def resolver():
        return Credential(env={"ANTHROPIC_BASE_URL": "https://attacker.invalid"})

    async with CallRuntime(adapters={"claude_code": CliAdapter(spawn=spawn)}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(resolver)), "hello", call_id="unsafe-env"
        )
    assert answer.outcome == "failed"
    assert answer.failure.classification == FailureClass.INVALID_REQUEST
    assert spawned == 0


@pytest.mark.asyncio
async def test_credential_callback_error_degrades_and_keeps_answer(tmp_path: Path) -> None:
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "kept"}]}},
            {
                "type": "result",
                "subtype": "success",
                "result": "kept",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ]
    )

    async def persist(_files):
        raise OSError("read-only credential vault")

    events = []
    binding = Binding(
        "claude_code",
        "test",
        CredentialRef(lambda: Credential(home={".credentials.json": b'{"old":true}'}, on_changed=persist)),
    )

    async def spawn(*argv, **kwargs):
        config_dir = Path(kwargs["env"]["CLAUDE_CONFIG_DIR"])
        (config_dir / ".credentials.json").write_bytes(b'{"new":true}')
        return await cli(*argv, **kwargs)

    async with CallRuntime(
        adapters={"claude_code": CliAdapter(spawn=spawn, home_parent=tmp_path)}, pricing=PRICES
    ) as runtime:
        answer = await runtime.ask(binding, "reply", events=events.append, call_id="credential-callback")
    degraded = [event for event in events if event.kind == EventKind.DEGRADED]
    assert answer.outcome == "answered" and answer.text == "kept"
    assert any(event.fields.get("setting") == "credential_refresh" for event in degraded)
    assert not await asyncio.to_thread(lambda: list(tmp_path.rglob(".credentials.json")))


def test_cli_environment_is_an_explicit_allowlist() -> None:
    for variable in (
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "NODE_OPTIONS",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "AWS_ACCESS_KEY_ID",
    ):
        with pytest.raises(ValueError):
            CliAdapter(env={variable: "unsafe"})


def test_cli_transport_allowlist_is_separate_from_credentials_and_base_urls() -> None:
    from agentkit.models.providers.cli import CliTransportVariable

    adapter = CliAdapter(transport={CliTransportVariable.HTTPS_PROXY: "https://proxy.example"})
    assert adapter.transport == {"HTTPS_PROXY": "https://proxy.example"}
    for variable in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "AWS_PROFILE"):
        with pytest.raises(ValueError):
            CliAdapter(transport={variable: "unsafe"})
    with pytest.raises(ValueError):
        CliAdapter(env={"HTTPS_PROXY": "https://proxy.example"})


def test_default_tool_argument_masker_redacts_short_strings_recursively() -> None:
    value = {"secret-key": "cat /tmp/private", "nested": ["sk-secret", 7, True]}
    masked = _mask_tool_arguments("Bash", value)
    assert "secret-key" not in repr(masked)
    assert "cat /tmp/private" not in repr(masked)
    assert "sk-secret" not in repr(masked)
    assert "'value': 7" not in repr(masked)
    assert "'value': True" not in repr(masked)


@pytest.mark.asyncio
async def test_cli_tool_call_event_masks_arguments_by_default() -> None:
    class ToolEventAdapter:
        async def turn(self, binding, prompt, **kwargs):
            await kwargs["emit"](
                CallEvent(
                    EventKind.TOOL_CALL,
                    "masked-event",
                    {
                        "name": "Bash",
                        "arguments": {
                            "secret-key": "echo sk-live-secret",
                            "numeric-secret": 734291,
                            "boolean-secret": True,
                        },
                    },
                )
            )
            return Turn(messages=("done",), raw_usage={"input_tokens": 1, "output_tokens": 1})

    events = []
    async with CallRuntime(adapters={"claude_code": ToolEventAdapter()}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding("claude_code", "test"), "run", events=events.append, call_id="masked-event")
    event = next(item for item in events if item.kind == EventKind.TOOL_CALL)
    assert answer.outcome == "answered"
    assert "sk-live-secret" not in repr(event.fields)
    assert "secret-key" not in repr(event.fields)
    assert "734291" not in repr(event.fields)
    assert "True" not in repr(event.fields)


@pytest.mark.asyncio
async def test_claude_mcp_wiring_passes_strict_config_only_once() -> None:
    from agentkit.tools import FunctionTool, ToolRegistry
    from agentkit.tools.execution.policy import ToolSet

    async def lookup_record(key: str) -> str:
        """Return a record for the supplied key."""
        return key

    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "ok"}]}},
            {"type": "result", "subtype": "success", "result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    tools = ToolSet(ToolRegistry.from_tools([FunctionTool.from_callable(lookup_record, side_effecting=False)]))
    async with CallRuntime(adapters={"claude_code": CliAdapter(spawn=cli)}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test-key"})),
            "answer",
            tools=tools,
            call_id="claude-mcp-config",
        )
    assert answer.outcome == "answered", answer.failure
    argv = cli.invocations[0].argv
    assert argv.count("--strict-mcp-config") == 1
    assert "--mcp-config" in argv
    assert "--tools" in argv


@pytest.mark.asyncio
async def test_claude_large_schema_uses_prompt_file_and_local_validation(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "description": "Allowed value. " * 3000}},
        "required": ["value"],
        "additionalProperties": False,
    }
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": '{"value":"ok"}'}]}},
            {
                "type": "result",
                "subtype": "success",
                "result": '{"value":"ok"}',
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ]
    )
    schema_prompt: list[str] = []

    async def capture_spawn(*argv, **kwargs):
        prompt_flag = next(flag for flag in ("--append-system-prompt-file", "--system-prompt-file") if flag in argv)
        schema_prompt.append(await asyncio.to_thread(Path(argv[argv.index(prompt_flag) + 1]).read_text))
        return await cli(*argv, **kwargs)

    adapter = CliAdapter(spawn=capture_spawn, working_dir=tmp_path)
    events = []
    async with CallRuntime(adapters={"claude_code": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test-key"})),
            "Return the value.",
            OutputSpec(schema),
            call_id="claude-large-schema",
            events=events.append,
        )
    assert answer.outcome == "answered", answer.failure
    assert answer.value == {"value": "ok"}
    argv = cli.invocations[0].argv
    assert "--json-schema" not in argv
    assert len(schema_prompt) == 1 and '"properties"' in schema_prompt[0]
    assert any(
        event.kind == EventKind.DEGRADED and event.fields.get("setting") == "structured_output" for event in events
    )


@pytest.mark.asyncio
async def test_native_claude_agents_require_supplied_home_login_and_use_non_bare_flags() -> None:
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "ok"}]}},
            {"type": "result", "subtype": "success", "result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    adapter = CliAdapter(spawn=cli)
    agents = {"reviewer": {"description": "Review changes", "prompt": "Inspect carefully", "tools": ["Read"]}}
    async with CallRuntime(adapters={"claude_code": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: Credential(home={".credentials.json": b"{}"}))),
            "review",
            call_id="native-agents",
            native_agents=agents,
        )
    assert answer.outcome == "answered", answer.failure
    argv = cli.invocations[0].argv
    assert "--bare" not in argv
    assert json.loads(argv[argv.index("--agents") + 1]) == agents


@pytest.mark.asyncio
async def test_cli_workspace_applies_per_call_paths_permissions_tools_and_hook_settings(tmp_path) -> None:
    workdir, add_dir = tmp_path / "workspace", tmp_path / "extra"
    workdir.mkdir()
    add_dir.mkdir()
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "ok"}]}},
            {"type": "result", "subtype": "success", "result": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    captured_settings = []

    async def capture_spawn(*argv, **kwargs):
        settings_path = argv[argv.index("--settings") + 1]
        captured_settings.append(await asyncio.to_thread(Path(settings_path).read_text))
        return await cli(*argv, **kwargs)

    workspace = Workspace(
        working_dir=workdir,
        add_dirs=(add_dir,),
        permission_mode="acceptEdits",
        native_tools=("Read", "Edit"),
        hooks={"PreToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "fence"}]}]},
    )
    async with CallRuntime(adapters={"claude_code": CliAdapter(spawn=capture_spawn)}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: Credential(home={".credentials.json": b"{}"}))),
            "edit",
            call_id="workspace",
            workspace=workspace,
        )
    invocation = cli.invocations[0]
    argv = invocation.argv
    assert answer.outcome == "answered"
    assert invocation.cwd == str(workdir)
    assert argv[argv.index("--add-dir") + 1] == str(add_dir)
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert argv[argv.index("--tools") + 1 : argv.index("--tools") + 3] == ("Read", "Edit")
    assert '"PreToolUse"' in captured_settings[0]


@pytest.mark.asyncio
async def test_codex_workspace_applies_per_call_paths_and_sandbox(tmp_path) -> None:
    workdir, add_dir = tmp_path / "codex-workspace", tmp_path / "codex-extra"
    workdir.mkdir()
    add_dir.mkdir()
    cli = FakeCodexCli.script(codex_turn(text="ok"))
    workspace = Workspace(
        working_dir=workdir,
        add_dirs=(add_dir,),
        sandbox="workspace-write",
    )
    async with CallRuntime(adapters={"codex": CliAdapter(spawn=cli)}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("codex", "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test"})),
            "inspect the workspace",
            call_id="codex-workspace",
            workspace=workspace,
        )
    argv = cli.invocations[0].argv
    assert answer.outcome == "answered"
    assert cli.invocations[0].cwd == str(workdir)
    assert argv[argv.index("--cd") + 1] == str(workdir)
    assert argv[argv.index("--add-dir") + 1] == str(add_dir)
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"


@pytest.mark.asyncio
async def test_claude_cli_slices_resume_and_forks_with_stable_prefix() -> None:
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    {"type": "assistant", "message": {"id": "partial", "content": [{"type": "text", "text": "slice"}]}},
                    {
                        "type": "result",
                        "subtype": "error_max_turns",
                        "is_error": True,
                        "result": "slice",
                        "session_id": "claude-session",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ]
            ),
            CliRun.of(
                [
                    {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "done"}]}},
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "done",
                        "session_id": "claude-session",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ]
            ),
        ]
    )
    async with CallRuntime(adapters={"claude_code": CliAdapter(spawn=cli)}, pricing=PRICES) as runtime:
        first = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            "work in a bounded slice",
            call_id="slice-one",
            turns_per_slice=1,
        )
        second = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            "continue and finish",
            call_id="slice-two",
            resume=first.session,
            stable_prefix=True,
            fork=True,
        )
    first_argv, second_argv = (invocation.argv for invocation in cli.invocations)
    assert first.outcome == "limit_reached" and first.session is not None
    assert first_argv[first_argv.index("--max-turns") + 1] == "1"
    assert second.outcome == "answered"
    assert second_argv[second_argv.index("--resume") + 1] == "claude-session"
    assert "--fork-session" in second_argv
    assert "--exclude-dynamic-system-prompt-sections" in second_argv


@pytest.mark.asyncio
async def test_codex_resume_can_fork_a_cli_session() -> None:
    from agentkit.models.contracts.types import SessionHandle

    cli = FakeCodexCli.script(codex_turn(text="forked"))
    session = SessionHandle(
        mode="codex", run_id="prior", cli_session_id="thread-1", home_bundle={"sessions/thread.jsonl": "eA=="}
    )
    async with CallRuntime(adapters={"codex": CliAdapter(spawn=cli)}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("codex", "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test"})),
            "continue on a branch",
            call_id="codex-fork",
            resume=session,
            fork=True,
        )
    argv = cli.invocations[0].argv
    fork_index = argv.index("fork")
    assert answer.outcome == "answered"
    assert argv[fork_index + 1] == "thread-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["anthropic_api", "openai_api"])
async def test_frontier_apis_preserve_composed_image_parts(mode):
    captured = {}
    events = (
        [
            {"type": "response.output_text.delta", "delta": "ok"},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
            },
        ]
        if mode == "openai_api"
        else [
            {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
    )

    def respond(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse(events))

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    prompt = inline_prompt("Describe the picture")
    prompt = replace(prompt, images=(ImagePart("image/png", b"png-data"),))
    binding = Binding(
        mode,
        "test",
        CredentialRef(
            lambda: (
                {"ANTHROPIC_API_KEY": "test"}
                if mode in ("claude_code", "anthropic_api")
                else {"OPENAI_API_KEY": "test"}
            )
        ),
    )
    async with CallRuntime(adapters={mode: ApiAdapter(client=client)}, pricing=PRICES) as runtime:
        answer = await runtime.ask(binding, prompt, call_id=f"image-{mode}")
    await client.aclose()
    assert answer.outcome == "answered"
    if mode == "openai_api":
        block = captured["input"][0]["content"][1]
        assert block["type"] == "input_image" and block["image_url"].startswith("data:image/png;base64,")
    else:
        block = captured["messages"][0]["content"][1]
        assert block["type"] == "image" and block["source"]["data"]


@pytest.mark.asyncio
async def test_codex_images_are_private_files_and_claude_reports_unsupported_images(tmp_path):
    image = ImagePart("image/png", b"png-data")
    prompt = replace(inline_prompt("Describe"), images=(image,))
    cli = FakeCodexCli.script(codex_turn(text="picture", usage=(1, 0, 1)))
    inspected = []

    async def capture_spawn(*argv, **kwargs):
        image_path = Path(argv[argv.index("--image") + 1])
        inspected.append(await asyncio.to_thread(lambda: (image_path.read_bytes(), image_path.stat().st_mode & 0o777)))
        return await cli(*argv, **kwargs)

    async with CallRuntime(adapters={"codex": CliAdapter(spawn=capture_spawn)}, pricing=PRICES) as runtime:
        codex = await runtime.ask(
            Binding("codex", "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test"})),
            prompt,
            call_id="image-codex",
        )
    events = []
    async with CallRuntime(
        adapters={"claude_code": CliAdapter(spawn=FakeClaudeCli.script([]))}, pricing=PRICES
    ) as runtime:
        claude = await runtime.run(Binding("claude_code", "test"), prompt, events=events.append, call_id="image-claude")
    assert codex.outcome == "answered" and inspected == [(b"png-data", 0o600)]
    assert claude.outcome == "failed"
    assert any(event.kind == "degraded" and event.fields.get("setting") == "images" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_nested_nulls_through_actual_translators(mode: str) -> None:
    value = {"required": None, "nested": {"name": "ok", "note": None}, "optional": None}
    adapter = real_adapter(mode, json.dumps(value), value=value)
    binding = Binding(
        mode,
        "test",
        CredentialRef(
            lambda: (
                {"ANTHROPIC_API_KEY": "test"}
                if mode in ("claude_code", "anthropic_api")
                else {"OPENAI_API_KEY": "test"}
            )
        ),
    )
    try:
        async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
            answer = await runtime.ask(binding, "JSON", OutputSpec(SCHEMA), call_id="structured")
        assert answer.outcome == "answered", answer.failure
        assert answer.value == {"required": None, "nested": {"name": "ok"}}
    finally:
        if hasattr(adapter, "client"):
            await adapter.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_one_correction_turn_and_canonical_bounds(mode: str) -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3}}, "required": ["n"]}
    adapter = ScriptedAdapter(Turn(("preface", '{"n":1}')), Turn(('{"n":3}',)))
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding(mode, "test"), "JSON", OutputSpec(schema), call_id="repair")
    assert answer.value == {"n": 3} and answer.attempts == 2
    assert len(adapter.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["refusal", "truncation", "failure"])
async def test_terminal_outcomes_never_repair(mode: str, kind: str) -> None:
    turn = Turn(
        ("partial",),
        refused=kind == "refusal",
        truncated=kind == "truncation",
        failure=Failure("auth") if kind == "failure" else None,
    )
    adapter = ScriptedAdapter(turn)
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding(mode, "test"), "JSON", OutputSpec(SCHEMA), call_id="terminal")
    assert answer.outcome == {"refusal": "refused", "truncation": "truncated", "failure": "failed"}[kind]
    assert len(adapter.requests) == 1
    if kind != "truncation":
        assert answer.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_unpriced_is_refused_before_dispatch(mode: str) -> None:
    adapter = ScriptedAdapter()
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding(mode, "unknown-model"), "hello", call_id="unpriced")
    assert answer.cost_source == "unpriced" and answer.cost_usd is None
    assert answer.outcome == "failed" and not adapter.requests


@pytest.mark.asyncio
async def test_retry_after_minimum_and_quota_never_retries() -> None:
    events = []
    adapter = ScriptedAdapter(Turn(failure=Failure("rate_limited", True, 0.001)), Turn(("done",)))
    runtime = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES, retry=RetryPolicy(base_s=0, cap_s=1))
    answer = await runtime.ask(Binding("openai_api", "test"), "hello", events=events.append, call_id="retry")
    assert answer.text == "done" and answer.attempts == 2
    assert next(e for e in events if e.kind == "retry").fields["wait_seconds"] >= 0.001
    adapter.turns = [Turn(failure=Failure("quota_exhausted"))]
    answer = await runtime.ask(Binding("openai_api", "test"), "hello", call_id="quota")
    assert answer.attempts == 1 and answer.failure.classification == "quota_exhausted"


@pytest.mark.asyncio
async def test_retry_consumes_only_remaining_cost_and_failed_attempt_does_not_use_turn_cap() -> None:
    class CostAdapter:
        def __init__(self):
            self.limits = []
            self.turns = [
                Turn(failure=Failure("rate_limited", True), reported_cost=Decimal("0.4")),
                Turn(messages=("done",), reported_cost=Decimal("0.2")),
            ]

        async def turn(self, binding, prompt, **kwargs):
            self.limits.append(kwargs["limits"].max_cost_usd)
            return self.turns.pop(0)

    adapter = CostAdapter()
    retrying = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES, retry=RetryPolicy(base_s=0, cap_s=0))
    async with retrying:
        answer = await retrying.ask(
            Binding("openai_api", "test"),
            "retry",
            limits=CallLimits(max_cost_usd=Decimal("1"), max_turns=1),
            call_id="remaining-budget",
        )
    assert answer.outcome == "answered" and answer.attempts == 2
    assert adapter.limits == [Decimal("1"), Decimal("0.6")]


@pytest.mark.asyncio
async def test_unpriced_model_can_retry_with_required_turn_cap() -> None:
    adapter = ScriptedAdapter(
        Turn(failure=Failure("rate_limited", True)),
        Turn(messages=("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    async with CallRuntime(
        adapters={"openai_api": adapter},
        pricing=PRICES,
        unpriced_policy=UnpricedPolicy.PROCEED_UNPRICED,
        retry=RetryPolicy(base_s=0, cap_s=0),
    ) as runtime:
        answer = await runtime.ask(
            Binding("openai_api", "unknown-model"),
            "retry",
            limits=CallLimits(max_turns=1),
            call_id="unpriced-retry",
        )
    assert answer.outcome == "answered" and answer.attempts == 2


@pytest.mark.asyncio
async def test_codex_native_turn_slicing_fails_closed_before_dispatch() -> None:
    adapter = ScriptedAdapter()
    async with CallRuntime(adapters={"codex": adapter}, pricing=PRICES) as runtime:
        with pytest.raises(ValueError, match="verifiable agentic-turn slice limit"):
            await runtime.run(Binding("codex", "test"), "slice", turns_per_slice=1, call_id="codex-slice")
    assert not adapter.requests


@pytest.mark.asyncio
async def test_unpriced_usage_category_is_refused_by_default_and_can_be_allowed_explicitly() -> None:
    partial_prices = PriceTable("partial", "2026-10-13", {"test": {"input_uncached": "1"}})
    adapter = ScriptedAdapter(Turn(messages=("answer",), raw_usage={"input_tokens": 3, "output_tokens": 2}))
    events = []
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=partial_prices) as runtime:
        answer = await runtime.ask(
            Binding("openai_api", "test"), "q", events=events.append, call_id="unpriced-category"
        )
    assert answer.outcome == "failed" and answer.text == ""
    assert answer.cost_usd is None
    assert answer.failure is not None
    proceed_adapter = ScriptedAdapter(Turn(messages=("answer",), raw_usage={"input_tokens": 3, "output_tokens": 2}))
    proceed_events = []
    async with CallRuntime(
        adapters={"openai_api": proceed_adapter},
        pricing=partial_prices,
        unpriced_policy=UnpricedPolicy.PROCEED_UNPRICED,
    ) as runtime:
        proceed = await runtime.ask(
            Binding("openai_api", "test"), "q", events=proceed_events.append, call_id="unpriced-category-proceed"
        )
    assert proceed.outcome == "answered" and proceed.text == "answer"
    assert proceed.cost_source == "mixed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["claude_code", "codex"])
async def test_cli_rate_limit_retry_after_recovers_and_400_does_not(mode: str) -> None:
    async def run(first_payload: dict[str, Any], *, status: int) -> tuple[Any, Any]:
        if mode == "claude_code":
            cli = FakeClaudeCli(
                [
                    CliRun.of([first_payload], returncode=1),
                    CliRun.of(
                        [
                            {
                                "type": "assistant",
                                "message": {"id": "final", "content": [{"type": "text", "text": "done"}]},
                            },
                            {
                                "type": "result",
                                "subtype": "success",
                                "result": "done",
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                            },
                        ]
                    ),
                ]
            )
        else:
            cli = FakeCodexCli(
                [
                    CliRun.of([first_payload], returncode=1),
                    CliRun.of(codex_turn(text="done")),
                ]
            )
        events: list[Any] = []
        async with CallRuntime(
            adapters={mode: CliAdapter(spawn=cli)},
            pricing=PRICES,
            retry=RetryPolicy(base_s=0, cap_s=1, max_elapsed_s=2),
        ) as runtime:
            answer = await runtime.ask(
                Binding(
                    mode,
                    "test",
                    CredentialRef(
                        lambda: {"ANTHROPIC_API_KEY": "test"} if mode == "claude_code" else {"OPENAI_API_KEY": "test"}
                    ),
                ),
                "retry safely",
                events=events.append,
                call_id=f"cli-retry-{mode}-{status}",
            )
        return answer, events

    limited_payload = (
        {
            "type": "result",
            "subtype": "api_error",
            "terminal_reason": "api_error",
            "is_error": True,
            "api_error_status": 429,
            "retry_after": "0.01",
            "error": {"type": "rate_limit_error"},
        }
        if mode == "claude_code"
        else {
            "type": "turn.failed",
            "error": {"message": "limited", "code": "rate_limit_exceeded", "status": 429, "retry_after": "0.01"},
        }
    )
    answer, events = await run(limited_payload, status=429)
    assert answer.outcome == "answered" and answer.attempts == 2
    retry = next(event for event in events if event.kind == "retry")
    assert retry.fields["wait_seconds"] >= 0.01

    invalid_payload = (
        {
            "type": "result",
            "subtype": "api_error",
            "terminal_reason": "api_error",
            "is_error": True,
            "api_error_status": 400,
            "error": {"type": "invalid_request_error"},
        }
        if mode == "claude_code"
        else {"type": "turn.failed", "error": {"message": "invalid", "code": "invalid_request_error", "status": 400}}
    )
    answer, _ = await run(invalid_payload, status=400)
    assert answer.attempts == 1 and answer.failure.classification == FailureClass.INVALID_REQUEST


@pytest.mark.asyncio
async def test_claude_retry_keeps_the_original_resumed_session() -> None:
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    {"type": "assistant", "message": {"id": "first", "content": [{"type": "text", "text": "started"}]}},
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "started",
                        "session_id": "session-keep",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ]
            ),
            CliRun.of(
                [
                    {
                        "type": "result",
                        "subtype": "api_error",
                        "terminal_reason": "api_error",
                        "is_error": True,
                        "api_error_status": 429,
                        "retry_after": "0.001",
                        "error": {"type": "rate_limit_error"},
                    },
                ],
                returncode=1,
            ),
            CliRun.of(
                [
                    {
                        "type": "assistant",
                        "message": {"id": "final", "content": [{"type": "text", "text": "resumed answer"}]},
                    },
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "resumed answer",
                        "session_id": "session-keep",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ]
            ),
        ]
    )
    adapter = CliAdapter(spawn=cli)
    async with CallRuntime(
        adapters={"claude_code": adapter}, pricing=PRICES, retry=RetryPolicy(base_s=0, cap_s=1, max_elapsed_s=2)
    ) as runtime:
        binding = Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"}))
        first = await runtime.ask(binding, "start", call_id="resume-seed")
        answer = await runtime.ask(binding, "continue", resume=first.session, call_id="resume-retry")
    assert answer.outcome == "answered" and answer.attempts == 2
    for invocation in cli.invocations[1:]:
        assert "--resume" in invocation.argv
        assert invocation.argv[invocation.argv.index("--resume") + 1] == "session-keep"


@pytest.mark.asyncio
async def test_claude_retry_passes_only_remaining_provider_budget() -> None:
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    {
                        "type": "result",
                        "subtype": "api_error",
                        "terminal_reason": "api_error",
                        "is_error": True,
                        "api_error_status": 429,
                        "retry_after": "0.001",
                        "total_cost_usd": 0.4,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                        "error": {"type": "rate_limit_error"},
                    },
                ],
                returncode=1,
            ),
            CliRun.of(
                [
                    {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "done"}]}},
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "done",
                        "total_cost_usd": 0.2,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ]
            ),
        ]
    )
    async with CallRuntime(
        adapters={"claude_code": CliAdapter(spawn=cli)},
        pricing=PRICES,
        retry=RetryPolicy(base_s=0, cap_s=1, max_elapsed_s=2),
    ) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            "bounded retry",
            limits=CallLimits(max_cost_usd=Decimal("1")),
            call_id="budgeted-cli-retry",
        )
    assert answer.outcome == "answered" and answer.attempts == 2
    retry_argv = cli.invocations[1].argv
    assert Decimal(retry_argv[retry_argv.index("--max-budget-usd") + 1]) == Decimal("0.6")


@pytest.mark.asyncio
async def test_claude_budgeted_retry_treats_failed_attempt_without_usage_as_zero_spent() -> None:
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    {
                        "type": "result",
                        "subtype": "api_error",
                        "terminal_reason": "api_error",
                        "is_error": True,
                        "api_error_status": 429,
                        "retry_after": "0.001",
                        "error": {"type": "rate_limit_error"},
                    }
                ],
                returncode=1,
            ),
            CliRun.of(
                [
                    {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "done"}]}},
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "done",
                        "total_cost_usd": 0.2,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                ]
            ),
        ]
    )
    async with CallRuntime(
        adapters={"claude_code": CliAdapter(spawn=cli)},
        pricing=PRICES,
        retry=RetryPolicy(base_s=0, cap_s=1, max_elapsed_s=2),
    ) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            "retry without reported usage",
            limits=CallLimits(max_cost_usd=Decimal("1")),
            call_id="budgeted-cli-retry-no-cost",
        )
    assert answer.outcome == "answered" and answer.attempts == 2
    retry_argv = cli.invocations[1].argv
    assert Decimal(retry_argv[retry_argv.index("--max-budget-usd") + 1]) == Decimal("1")


@pytest.mark.asyncio
async def test_cli_budget_does_not_retry_after_output_with_unknown_cost() -> None:
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    {
                        "type": "assistant",
                        "message": {"id": "partial", "content": [{"type": "text", "text": "partial output"}]},
                    },
                    {
                        "type": "result",
                        "subtype": "api_error",
                        "terminal_reason": "api_error",
                        "is_error": True,
                        "api_error_status": 429,
                        "retry_after": "0.001",
                        "error": {"type": "rate_limit_error"},
                    },
                ],
                returncode=1,
            ),
            CliRun.of([{"type": "result", "subtype": "success", "result": "unexpected"}]),
        ]
    )
    async with CallRuntime(
        adapters={"claude_code": CliAdapter(spawn=cli)},
        pricing=PRICES,
        retry=RetryPolicy(base_s=0, cap_s=1, max_elapsed_s=2),
    ) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            "do not retry unknown spend",
            limits=CallLimits(max_cost_usd=Decimal("1")),
            call_id="budgeted-cli-partial-output",
        )
    assert answer.outcome == "limit_reached" and answer.attempts == 1
    assert cli.spawns == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_cancellation_unwinds_transport(mode: str) -> None:
    started, cleaned = asyncio.Event(), asyncio.Event()

    class Blocking:
        async def turn(self, *args: Any, **kwargs: Any) -> Turn:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
            return Turn()

    token = CancellationToken()
    events = []
    runtime = CallRuntime(adapters={mode: Blocking()}, pricing=PRICES)
    task = asyncio.create_task(
        runtime.ask(Binding(mode, "test"), "hello", cancel=token, events=events.append, call_id="cancel")
    )
    await started.wait()
    token.cancel()
    answer = await asyncio.wait_for(task, 1)
    assert cleaned.is_set() and answer.outcome == "cancelled"
    assert events[-1].fields["outcome"] == "cancelled"


@pytest.mark.parametrize("mode", MODES)
def test_schema_compilation_never_mutates_canonical(mode: str) -> None:
    snapshot = json.dumps(SCHEMA, sort_keys=True)
    compiled = compile_schema(SCHEMA, mode)
    assert json.dumps(SCHEMA, sort_keys=True) == snapshot
    if mode in ("openai_api", "codex"):
        assert compiled["required"] == list(SCHEMA["properties"])
    if mode == "claude_code":
        assert "definitions" in compiled and "$defs" not in compiled


def test_no_matching_union_preserves_invalid_value() -> None:
    bad = {"required": None, "nested": {"wrong": True, "note": None}, "optional": None}
    cleaned = clean_nulls(bad, SCHEMA)
    assert cleaned["nested"] == bad["nested"]


def test_usage_and_decimal_pricing_do_not_double_bill_cache_or_reasoning() -> None:
    u = normalize_usage(
        "openai_api",
        {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 40},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 10},
        },
    )
    assert u.input_uncached == 60 and u.reasoning == 10
    amount, source = price("test", u, PRICES)
    assert isinstance(amount, Decimal) and amount == Decimal("0.000164") and source == "price_table"


@pytest.mark.parametrize("mode", ("claude_code", "codex"))
@pytest.mark.asyncio
async def test_cli_sampling_refused_without_spawn(mode: str) -> None:
    runtime = CallRuntime(adapters={mode: ScriptedAdapter()}, pricing=PRICES)
    with pytest.raises(ValueError, match="temperature"):
        await runtime.ask(Binding(mode, "test"), "hello", settings=ModelSettings(temperature=0), call_id="settings")


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["wall", "cancel"])
async def test_tool_execution_shares_call_deadline_and_cancellation(cause: str) -> None:
    from agentkit.models.contracts.types import CallLimits

    entered, unwound = asyncio.Event(), asyncio.Event()

    class BlockingTools:
        def schemas(self):
            return []

        async def execute(self, *args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                unwound.set()

    adapter = ScriptedAdapter(Turn(tool_calls=({"id": "c", "name": "slow", "arguments": {}},)))
    runtime = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES)
    token = CancellationToken()
    task = asyncio.create_task(
        runtime.run(
            Binding("openai_api", "test"),
            "Do work",
            BlockingTools(),
            call_id="tool-deadline",
            cancel=token,
            limits=CallLimits(wall_clock_s=0.1 if cause == "wall" else None),
        )
    )
    await entered.wait()
    if cause == "cancel":
        token.cancel()
    answer = await asyncio.wait_for(task, 1)
    assert unwound.is_set()
    assert answer.outcome == ("cancelled" if cause == "cancel" else "limit_reached")
    assert len(adapter.requests) == 1


@pytest.mark.asyncio
async def test_cumulative_usage_and_cost_are_subtracted_only_once_on_resume() -> None:
    from agentkit.models.contracts.types import SessionHandle

    previous = SessionHandle("claude_code", "previous", reported_cost=Decimal("0.01"))
    claude = ScriptedAdapter(
        Turn(
            ("done",),
            raw_usage={"input_tokens": 10, "output_tokens": 2},
            reported_cost=Decimal("0.013"),
            reported_cost_cumulative=True,
        )
    )
    answer = await CallRuntime(adapters={"claude_code": claude}, pricing=PRICES).ask(
        Binding("claude_code", "test"), "next", resume=previous, call_id="claude-resume"
    )
    assert answer.cost_usd == Decimal("0.003")
    assert answer.usage.input_uncached == 10
    assert answer.session.reported_cost == Decimal("0.013")

    previous = SessionHandle("codex", "previous", raw_usage={"input_tokens": 100, "output_tokens": 10})
    codex = ScriptedAdapter(
        Turn(("done",), raw_usage={"input_tokens": 120, "output_tokens": 15}, usage_cumulative=True)
    )
    answer = await CallRuntime(adapters={"codex": codex}, pricing=PRICES).ask(
        Binding("codex", "test"), "next", resume=previous, call_id="codex-resume"
    )
    assert answer.usage.input_uncached == 20 and answer.usage.output == 5
    assert answer.cost_usd == Decimal("0.000045")


@pytest.mark.asyncio
async def test_retry_keeps_portable_history_and_native_usage_baseline() -> None:
    from agentkit.models.contracts.types import SessionHandle

    history = ({"role": "user", "content": "Remember code 123"},)
    previous = SessionHandle("openai_api", "previous", history=history)
    adapter = ScriptedAdapter(Turn(failure=Failure("transient_network", retryable=True)), Turn(("123",)))
    runtime = CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES, retry=RetryPolicy(base_s=0))
    answer = await runtime.ask(Binding("openai_api", "test"), "Which code?", resume=previous, call_id="retry-history")
    assert answer.outcome == "answered"
    assert adapter.requests[1][2]["session"].history[0] == history[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [(), ("",), ("  ",)])
async def test_empty_transport_turn_is_a_failure(messages) -> None:
    adapter = ScriptedAdapter(Turn(messages))
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).ask(
        Binding("openai_api", "test"), "question", call_id="empty"
    )
    assert answer.outcome == "failed" and answer.failure is not None


@pytest.mark.asyncio
async def test_anthropic_stream_cancellation_retains_observed_usage() -> None:
    class Transport:
        async def _stream_events(self, *args):
            yield {"type": "message_start", "message": {"usage": {"input_tokens": 42}}}
            yield {"type": "message_delta", "usage": {"output_tokens": 7}}
            raise asyncio.CancelledError

    async def emit(event):
        pass

    turn = await ApiAdapter()._anthropic(Transport(), {}, emit)
    assert turn.raw_usage == {"input_tokens": 42, "output_tokens": 7}
    assert turn.failure.classification == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_tool_results_continue_in_native_history_and_context(mode: str) -> None:
    from agentkit.tools.execution.policy import ToolResult

    class Tools:
        def schemas(self):
            return []

        async def execute(self, name, args, ctx):
            return ToolResult(({"type": "text", "text": "tool evidence"},), is_error=True)

    wire_call = ({"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "lookup", "input": {}}]},)
    if mode == "openai_api":
        wire_call = ({"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{}"},)
    adapter = ScriptedAdapter(
        Turn(tool_calls=({"id": "c", "name": "lookup", "arguments": {}},), history=wire_call),
        Turn(failure=Failure("transient_network", retryable=True)),
        Turn(("recovered",)),
    )
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES, retry=RetryPolicy(max_attempts=2, base_s=0))
    events = []
    answer = await runtime.run(Binding(mode, "test"), "lookup", Tools(), events=events.append, call_id="tool-loop")
    assert answer.outcome == "answered" and answer.attempts == 3
    history = adapter.requests[1][2]["session"].history
    assert "tool evidence" in json.dumps(history)
    assert any(m.role == "tool" and m.metadata["is_error"] for m in answer.context.messages)
    tool_call = next(event for event in events if event.kind == EventKind.TOOL_CALL)
    assert tool_call.fields["arguments"] == {}


@pytest.mark.asyncio
async def test_tool_result_events_include_digest_and_host_masked_bounded_preview() -> None:
    from agentkit.tools.execution.policy import ToolResult

    class Tools:
        timeout_s = None

        def schemas(self):
            return []

        async def execute(self, name, args, ctx):
            return ToolResult(({"type": "text", "text": "private secret evidence"},))

    adapter = ScriptedAdapter(
        Turn(tool_calls=({"id": "call-1", "name": "lookup", "arguments": {}},)),
        Turn(("done",)),
    )
    events = []
    runtime = CallRuntime(
        adapters={"openai_api": adapter},
        pricing=PRICES,
        result_masker=lambda _name, text: text.replace("secret", "[redacted]"),
    )
    answer = await runtime.run(
        Binding("openai_api", "test"), "lookup", Tools(), events=events.append, call_id="masked-result"
    )
    result_event = next(event for event in events if event.kind == EventKind.TOOL_RESULT)
    assert answer.outcome == "answered"
    assert result_event.fields["result_preview"] == "private [redacted] evidence"
    assert len(result_event.fields["result_digest"]) == 64


@pytest.mark.asyncio
async def test_anthropic_empty_tool_arguments_and_reasoning_signature_survive_stream() -> None:
    class Transport:
        async def _stream_events(self, *args):
            for event in [
                {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "opaque"}},
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "signed"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "c", "name": "ping"},
                },
                {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
                {"type": "message_stop"},
            ]:
                yield event

    emitted = []
    turn = await ApiAdapter()._anthropic(Transport(), {}, emitted.append)
    assert turn.tool_calls[0]["arguments"] == {}
    assert turn.history[0]["content"][0]["signature"] == "signed"
    assert "opaque" not in repr(emitted)
    assert turn.raw_usage["output_tokens"] == 7


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_clock_s", float("nan")),
        ("wall_clock_s", float("inf")),
        ("max_cost_usd", Decimal("NaN")),
        ("max_cost_usd", Decimal("Infinity")),
    ],
)
def test_nonfinite_call_limits_are_rejected(field, value):
    from agentkit.models.contracts.types import CallLimits

    with pytest.raises(ValueError, match="finite"):
        CallLimits(**{field: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_stream_disconnect_preserves_usage_already_observed(mode):
    class Transport:
        async def _stream_events(self, *args):
            if mode == "openai_api":
                yield {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 42, "output_tokens": 7}, "output": []},
                }
            else:
                yield {"type": "message_start", "message": {"usage": {"input_tokens": 42, "output_tokens": 7}}}
            raise ConnectionError("connection lost")

    adapter = ApiAdapter()
    parser = adapter._openai if mode == "openai_api" else adapter._anthropic
    turn = await parser(Transport(), {}, lambda event: None)
    assert turn.raw_usage == {"input_tokens": 42, "output_tokens": 7}
    assert turn.failure is not None


@pytest.mark.asyncio
async def test_external_task_cancellation_rethrows_and_emits_terminal_event():
    started, unwound = asyncio.Event(), asyncio.Event()

    class Blocking:
        async def turn(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                unwound.set()

    events = []
    runtime = CallRuntime(adapters={"openai_api": Blocking()}, pricing=PRICES)
    task = asyncio.create_task(
        runtime.ask(Binding("openai_api", "test"), "q", call_id="external-cancel", events=events.append)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert unwound.is_set()
    assert events[-1].kind == "call_finished" and events[-1].fields["outcome"] == "cancelled"
    assert events[-1].fields["cost_usd"] is None and events[-1].fields["cost_source"] == "unpriced"


@pytest.mark.asyncio
async def test_cli_snapshot_failure_preserves_answer_and_records_degradation(monkeypatch):
    from agentkit.models.providers.cli.isolation import IsolatedHome

    def unavailable(home):
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(IsolatedHome, "snapshot", unavailable)
    async with CallRuntime(adapters={"claude_code": real_adapter("claude_code", "answer")}, pricing=PRICES) as runtime:
        answer = await runtime.ask(
            Binding("claude_code", "test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test"})),
            "q",
            call_id="snapshot-failure",
        )
    assert answer.outcome == "answered" and answer.text == "answer"
    assert not answer.session.home_bundle
    assert any(d.setting == "native_session" for d in answer.degradations)
    assert answer.context.messages[-1].content == "answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_schema_correction_is_retained_in_session_and_context(mode):
    adapter = ScriptedAdapter(Turn(("not JSON",)), Turn(('{"answer":42}',)))
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES)
    output = OutputSpec(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    answer = await runtime.ask(Binding(mode, "test"), "q", output, call_id="correction-history")
    assert answer.value == {"answer": 42}
    assert any("Correct these validation errors" in m.content for m in answer.context.messages)
    correction_history = adapter.requests[1][2]["session"].history
    assert "not JSON" in json.dumps(correction_history)
    assert "Correct these validation errors" in json.dumps(correction_history)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["max_turns", "max_output_tokens", "max_cost_usd"])
async def test_call_limits_stop_continuation_before_second_dispatch(limit):
    from agentkit.models.contracts.types import CallLimits

    class Tools:
        def schemas(self):
            return []

        async def execute(self, *args):
            raise AssertionError("budget must stop before tool execution")

    value = Decimal("0.000001") if limit == "max_cost_usd" else 1
    # max_turns permits the first tool, then refuses the next dispatch.
    if limit == "max_turns":
        from agentkit.tools.execution.policy import ToolResult

        async def execute(*args):
            return ToolResult(({"type": "text", "text": "evidence"},))

        Tools.execute = execute
    turn = Turn(
        raw_usage={"input_tokens": 10, "output_tokens": 2}, tool_calls=({"id": "c", "name": "lookup", "arguments": {}},)
    )
    adapter = ScriptedAdapter(turn)
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).run(
        Binding("openai_api", "test"), "q", Tools(), call_id="bounded", limits=CallLimits(**{limit: value})
    )
    assert answer.outcome == "limit_reached"
    assert len(adapter.requests) == 1
    if limit == "max_output_tokens":
        assert adapter.requests[0][2]["settings"].max_output_tokens == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_tools_unavailable_fail_explicitly_and_preserve_usage(mode):
    adapter = ScriptedAdapter(
        Turn(
            raw_usage={"input_tokens": 7, "output_tokens": 1},
            tool_calls=({"id": "c", "name": "unknown", "arguments": {}},),
        )
    )
    answer = await CallRuntime(adapters={mode: adapter}, pricing=PRICES).run(
        Binding(mode, "test"), "q", call_id="missing-tools"
    )
    assert answer.failure.classification == "tool_error"
    assert answer.usage.input_uncached == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal,outcome", [("response.incomplete", "truncated"), ("response.failed", "failed")])
async def test_openai_terminal_status_preserves_usage(terminal, outcome):
    events = [
        {
            "type": terminal,
            "response": {
                "status": "incomplete" if outcome == "truncated" else "failed",
                "error": {"code": "invalid_request_error"},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "partial"}]}],
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        }
    ]
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=sse(events))))
    binding = Binding("openai_api", "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test"}))
    try:
        answer = await CallRuntime(adapters={"openai_api": ApiAdapter(client=client)}, pricing=PRICES).ask(
            binding, "q", call_id="partial"
        )
        assert answer.outcome == outcome
        assert answer.usage.output == 3 and answer.usage.input_uncached == 7
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("usage,complete", [({}, True), ({"input_tokens": 7}, False)])
async def test_missing_or_incomplete_accounting_is_unknown_instead_of_free(usage, complete):
    events = []
    adapter = ScriptedAdapter(Turn(("answer",), raw_usage=usage, usage_complete=complete))
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).ask(
        Binding("openai_api", "test"), "q", call_id="unknown-cost", events=events.append
    )
    assert answer.outcome == "answered" and answer.text == "answer"
    assert answer.cost_usd is None and answer.cost_source == "unpriced"
    assert answer.usage.input_uncached == usage.get("input_tokens", 0)
    assert events[-1].fields["cost_usd"] is None
    assert answer.usage_complete is complete


@pytest.mark.asyncio
async def test_reported_cost_does_not_mark_partial_usage_complete() -> None:
    adapter = ScriptedAdapter(
        Turn(
            messages=("answer",),
            raw_usage={"input_tokens": 7},
            usage_complete=False,
            reported_cost=Decimal("0.25"),
        )
    )
    async with CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.ask(Binding("openai_api", "test"), "q", call_id="partial-usage-reported-cost")
    assert answer.cost_usd == Decimal("0.25")
    assert answer.usage_complete is False


@pytest.mark.asyncio
async def test_unknown_accounting_stops_a_dollar_bounded_continuation():
    from agentkit.models.contracts.types import CallLimits

    class Tools:
        def schemas(self):
            return []

        async def execute(self, *args):
            raise AssertionError("unknown spend cannot permit another operation")

    adapter = ScriptedAdapter(Turn(tool_calls=({"id": "c", "name": "lookup", "arguments": {}},)))
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).run(
        Binding("openai_api", "test"),
        "q",
        Tools(),
        call_id="unknown-spend",
        limits=CallLimits(max_cost_usd=Decimal("1")),
    )
    assert answer.outcome == "limit_reached" and answer.cost_usd is None


@pytest.mark.asyncio
async def test_explicit_zero_usage_is_known_zero_cost():
    adapter = ScriptedAdapter(Turn(("answer",), raw_usage={"input_tokens": 0, "output_tokens": 0}))
    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).ask(
        Binding("openai_api", "test"), "q", call_id="zero-cost"
    )
    assert answer.cost_usd == Decimal(0) and answer.cost_source == "price_table"


@pytest.mark.asyncio
async def test_codex_native_deny_refuses_before_dispatch():
    from agentkit.models.contracts.constants import NativeToolPolicy

    adapter = ScriptedAdapter(Turn(("must not run",)))
    answer = await CallRuntime(
        adapters={"codex": adapter}, pricing=PRICES, native_tool_policy=NativeToolPolicy.DENY
    ).ask(Binding("codex", "test"), "q", call_id="native-deny")
    assert answer.outcome == "failed" and answer.failure.classification == "invalid_request"
    assert not adapter.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["warn", "allow"])
async def test_codex_native_warn_reduces_capabilities_and_reports_limit(policy):
    import tomllib

    from agentkit.tools import FunctionTool, ToolRegistry
    from agentkit.tools.execution.policy import ToolSet

    async def read_host_evidence() -> str:
        """Read evidence explicitly granted by the application host."""
        return "host evidence"

    tools = ToolSet(ToolRegistry.from_tools([FunctionTool.from_callable(read_host_evidence, side_effecting=False)]))
    adapter = real_adapter("codex", "done")
    adapter.native_tool_policy = policy
    binding = Binding("codex", "test", CredentialRef(lambda: {"OPENAI_API_KEY": "test"}))
    async with CallRuntime(adapters={"codex": adapter}, pricing=PRICES) as runtime:
        answer = await runtime.run(binding, "q", tools, call_id="native-policy")
    assert answer.outcome == "answered"
    argv = adapter.spawn.argv()
    assert ("features.shell_tool=false" in argv) == (policy == "warn")
    assert ("features.unified_exec=false" in argv) == (policy == "warn")
    for feature in ("apps", "plugins", "skill_mcp_dependency_install"):
        assert (f"features.{feature}=false" in argv) == (policy == "warn")
    assert ('web_search="disabled"' in argv) == (policy == "warn")
    assert any(argument.startswith("mcp_servers.agentkit.url=") for argument in argv)
    hosted = next(argument for argument in argv if argument.startswith("mcp_servers.agentkit.tools="))
    assert tomllib.loads(hosted)["mcp_servers"]["agentkit"]["tools"] == {
        "read_host_evidence": {"approval_mode": "approve"},
    }
    assert any(d.setting == "native_tools" for d in answer.degradations) == (policy == "warn")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_cancelled_tool_batch_closes_pending_pairs_and_retains_confirmed_results(mode):
    from agentkit.tools.execution.policy import ToolResult

    entered, cleaned = asyncio.Event(), asyncio.Event()
    calls = tuple({"id": name, "name": name, "arguments": {}} for name in ["first", "blocked", "last"])
    adapter = ScriptedAdapter(
        Turn(tool_calls=calls, raw_usage={"input_tokens": 3, "output_tokens": 1}), Turn(("resumed",))
    )

    class Tools:
        def schemas(self):
            return []

        async def execute(self, name, *args):
            if name == "first":
                return ToolResult(({"type": "text", "text": "confirmed evidence"},))
            assert name == "blocked"
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

    token = CancellationToken()
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES)
    task = asyncio.create_task(
        runtime.run(Binding(mode, "test"), "diagnose", Tools(), call_id="blocked-batch", cancel=token)
    )
    await entered.wait()
    token.cancel()
    interrupted = await task
    assert interrupted.outcome == "cancelled" and cleaned.is_set()
    results = {m.tool_call_id: m for m in interrupted.session.messages if m.role == "tool"}
    assert results["first"].content == "confirmed evidence"
    assert results["blocked"].metadata["interrupted"] and results["blocked"].metadata["is_error"]
    assert results["last"].metadata["not_executed"] and results["last"].metadata["is_error"]
    assert "confirmed evidence" in json.dumps(interrupted.session.history)
    resumed = await runtime.ask(
        Binding(mode, "test"), "Resume safely", resume=interrupted.session, call_id="safe-resume"
    )
    assert resumed.outcome == "answered"
    history = adapter.requests[-1][2]["session"].history
    if mode == "openai_api":
        assert {x["call_id"] for x in history if x.get("type") == "function_call"} == {
            x["call_id"] for x in history if x.get("type") == "function_call_output"
        }
    else:
        call_ids = {
            b["id"]
            for m in history
            for b in m.get("content", [])
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        result_ids = {
            b["tool_use_id"]
            for m in history
            for b in m.get("content", [])
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        assert call_ids == result_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_turn_limit_retains_latest_result_in_both_session_axes(mode):
    from agentkit.models.contracts.types import CallLimits
    from agentkit.tools.execution.policy import ToolResult

    class Tools:
        def schemas(self):
            return []

        async def execute(self, *args):
            return ToolResult(({"type": "text", "text": "durable diagnostic evidence"},))

    adapter = ScriptedAdapter(
        Turn(
            tool_calls=({"id": "c", "name": "diagnose", "arguments": {}},),
            raw_usage={"input_tokens": 3, "output_tokens": 1},
        )
    )
    answer = await CallRuntime(adapters={mode: adapter}, pricing=PRICES).run(
        Binding(mode, "test"), "diagnose", Tools(), call_id="one-turn", limits=CallLimits(max_turns=1)
    )
    assert answer.outcome == "limit_reached"
    assert answer.session.messages[-1].content == "durable diagnostic evidence"
    assert "durable diagnostic evidence" in json.dumps(answer.session.history)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["codex", "claude_code"])
async def test_cli_resume_floors_only_compact_budget_boundary(mode, monkeypatch):
    """Exact provider costs may leave a finer-than-microdollar invocation cap."""
    from agentkit.core import runtime as runtime_exports
    from agentkit.core.contracts.types import Message
    from agentkit.models.contracts.types import CallLimits
    from agentkit.state.context import WorkingContext

    ceilings = []
    original_budget = runtime_exports.Budget

    def capture_budget(*args, **kwargs):
        ceilings.append(kwargs.get("max_cost_usd"))
        return original_budget(*args, **kwargs)

    monkeypatch.setattr(runtime_exports, "Budget", capture_budget)
    if mode == "codex":
        cli = FakeCodexCli.script(codex_turn(text="submitted", usage=(10, 0, 5)), repeat_last=True)
    else:
        cli = FakeClaudeCli.script(
            [
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "submitted",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "session_id": "precision-session",
                    "total_cost_usd": 0.0363984,
                },
            ],
            repeat_last=True,
        )
    adapter = CliAdapter(spawn=cli)
    context = WorkingContext()
    remaining = Decimal(".9136016")
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        first = await runtime.run(
            Binding(
                mode,
                "test",
                CredentialRef(
                    lambda: (
                        {"ANTHROPIC_API_KEY": "offline-fake-key"}
                        if mode == "claude_code"
                        else {"OPENAI_API_KEY": "offline-fake-key"}
                    )
                ),
            ),
            "Repair source",
            context=context,
            limits=CallLimits(max_cost_usd=Decimal(".95")),
            call_id="precision",
        )
        context.append(Message("user", "Private verifier rejected the initial submission; recover."))
        second = await runtime.run(
            Binding(
                mode,
                "test",
                CredentialRef(
                    lambda: (
                        {"ANTHROPIC_API_KEY": "offline-fake-key"}
                        if mode == "claude_code"
                        else {"OPENAI_API_KEY": "offline-fake-key"}
                    )
                ),
            ),
            "Repair source",
            context=context,
            resume=first.session,
            limits=CallLimits(max_cost_usd=remaining),
            call_id="precision/verify-1",
        )
    assert first.outcome == second.outcome == "answered"
    assert len(cli.invocations) == 2
    assert ceilings == [Decimal(".950000"), Decimal(".913601")]
    assert ceilings[1] <= remaining and remaining == Decimal(".9136016")
    assert second.cost_usd == (Decimal(".0363984") if mode == "claude_code" else Decimal(".000035"))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
async def test_fast_cooperative_tool_abort_beats_monitor_poll_and_retains_context(mode):
    """A joined tool can raise cooperative cancellation before the 50ms monitor observes the token."""
    from agentkit.core.contracts.types import Message, ToolCall
    from agentkit.core.execution.concurrency import Cancelled

    token = CancellationToken()
    cleaned = asyncio.Event()
    if mode == "openai_api":
        history = ({"type": "function_call", "call_id": "fast-tool", "name": "sample", "arguments": "{}"},)
    else:
        history = (
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "fast-tool", "name": "sample", "input": {}},
                ],
            },
        )
    adapter = ScriptedAdapter(
        Turn(
            tool_calls=({"id": "fast-tool", "name": "sample", "arguments": {}},),
            canonical_messages=(Message("assistant", tool_calls=(ToolCall("fast-tool", "sample", {}),)),),
            history=history,
            raw_usage={"input_tokens": 10, "output_tokens": 2},
        )
    )

    class Tools:
        def schemas(self):
            return []

        async def execute(self, name, arguments, ctx):
            try:
                token.cancel()
                raise Cancelled("fast cooperative tool abort")
            finally:
                cleaned.set()

    events = []
    async with CallRuntime(adapters={mode: adapter}, pricing=PRICES) as runtime:
        answer = await runtime.run(
            Binding(mode, "test"),
            "Sample diagnostic",
            Tools(),
            cancel=token,
            events=events.append,
            call_id="fast-cooperative",
        )
    assert answer.outcome == "cancelled" and answer.failure.classification == "cancelled"
    assert cleaned.is_set() and len(adapter.requests) == 1
    assert answer.usage.input_uncached == 10 and answer.usage.output == 2
    assert answer.cost_usd == Decimal(".000020")
    result = next(message for message in answer.session.messages if message.role == "tool")
    assert result.tool_call_id == "fast-tool" and result.metadata["is_error"]
    assert "interrupted" in result.content.lower()
    assert events[-1].kind == "call_finished" and events[-1].fields["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_fast_cooperative_transport_abort_is_cancelled_and_unpriced():
    from agentkit.core.execution.concurrency import Cancelled

    token = CancellationToken()

    class Adapter:
        async def turn(self, *args, **kwargs):
            token.cancel()
            raise Cancelled("transport observed the token before monitor polling")

    async with CallRuntime(adapters={"openai_api": Adapter()}, pricing=PRICES) as runtime:
        answer = await runtime.run(Binding("openai_api", "test"), "Query", cancel=token, call_id="transport-abort")
    assert answer.outcome == "cancelled" and answer.failure.classification == "cancelled"
    assert answer.cost_usd is None and answer.cost_source == "unpriced"
