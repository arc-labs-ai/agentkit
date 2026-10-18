"""Assessment must describe the executed authority, not backend declarations."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentkit.core.runtime.calls import CallRuntime
from agentkit.models import Binding, CredentialRef, ExecutionRequirement, ExecutionRequirements, NativeToolPolicy
from agentkit.models.contracts.execution import CapabilityStatus
from agentkit.models.providers.cli import BackendReport, CliAdapter
from agentkit.models.providers.cli.capabilities import evaluate_cli_execution
from agentkit.models.providers.frontiers.calls import ApiAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api"])
@pytest.mark.parametrize("requirement", [ExecutionRequirement.HOST_TOOLS, ExecutionRequirement.SCHEMA])
async def test_payload_rewriter_cannot_claim_restricted_wire_protocol(mode, requirement):
    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credentials accessed")))
    execution = ExecutionRequirements(required=frozenset({requirement}))
    clean = await ApiAdapter().evaluate_execution(binding, execution=execution)
    assert clean.compatible
    planted = await ApiAdapter(request_hook=lambda request: request).evaluate_execution(binding, execution=execution)
    assert not planted.compatible
    assert planted.incompatibilities[0].status == CapabilityStatus.UNVERIFIED


@pytest.mark.parametrize("mode", ["claude_code", "codex"])
def test_backend_claim_cannot_erase_native_absence_blocker(mode):
    from agentkit.core import Mode

    clean = evaluate_cli_execution(
        Mode(mode), BackendReport("fake", "1"), ExecutionRequirements(), NativeToolPolicy.DENY
    )
    planted = evaluate_cli_execution(
        Mode(mode),
        BackendReport("fake", "1", frozenset({ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE})),
        ExecutionRequirements(),
        NativeToolPolicy.DENY,
    )
    assert not clean.compatible and not planted.compatible
    assert clean.incompatibilities == planted.incompatibilities


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_hard_cap_is_independent_and_precedes_credentials(mode):
    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credentials accessed")))
    adapter = ApiAdapter() if binding.mode.is_api else CliAdapter(spawn=lambda: pytest.fail("spawned"))
    execution = ExecutionRequirements(required=frozenset({ExecutionRequirement.HARD_SPENDING_CAP}))
    async with CallRuntime(adapters={mode: adapter}, unpriced_policy="proceed_unpriced") as runtime:
        result = await runtime.ask(binding, "task", call_id="cap", execution=execution)
    assert result.attempts == 0 and not result.usage_complete and result.cost_usd is None
    assert result.failure.mode_detail["dispatch_state"] == "not_started"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["claude_code", "codex"])
async def test_cli_request_hook_cannot_claim_host_catalog(mode):
    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credentials accessed")))
    adapter = CliAdapter(spawn=lambda: pytest.fail("spawned"), request_hook=lambda request: request)
    assessment = await adapter.evaluate_execution(
        binding,
        execution=ExecutionRequirements(required=frozenset({ExecutionRequirement.HOST_TOOLS})),
    )
    assert not assessment.compatible
    assert assessment.incompatibilities[0].code == "cli_request_hook_unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,version", [("claude_code", "2.1.300"), ("codex", "0.162.1")])
async def test_cli_version_changes_are_not_hidden_by_cache(monkeypatch, mode, version):
    from agentkit.models.providers.cli import execution as backend_module

    versions = iter([version, "999.0.0"])

    async def inspect(executable):
        return next(versions)

    monkeypatch.setattr(backend_module, "inspect_binary_version", inspect)
    adapter = CliAdapter()
    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credentials accessed")))
    requirements = ExecutionRequirements(required=frozenset({ExecutionRequirement.HOST_TOOLS}))
    clean = await adapter.evaluate_execution(binding, execution=requirements)
    planted = await adapter.evaluate_execution(binding, execution=requirements)
    assert clean.compatible
    assert planted.binary_version == "999.0.0" and not planted.compatible
    assert planted.incompatibilities[0].code == "cli_protocol_version_unverified"


@pytest.mark.asyncio
async def test_resume_preserves_requirements_and_policy():
    from agentkit.models.contracts.types import CallLimits, Turn

    class Scripted:
        async def turn(self, binding, prompt, **kwargs):
            return Turn(messages=("ok",), raw_usage={"input_tokens": 1, "output_tokens": 1})

    binding = Binding("anthropic_api", "test", CredentialRef(lambda: {}))
    requirements = ExecutionRequirements(required=frozenset({ExecutionRequirement.HOST_TOOLS}))
    async with CallRuntime(adapters={"anthropic_api": Scripted()}, unpriced_policy="proceed_unpriced") as runtime:
        # Construct a session through the actual loop; a custom adapter cannot
        # attest requirements, so the clean call first runs without them.
        first = await runtime.ask(binding, "task", call_id="first", limits=CallLimits(max_turns=2))
        assert first.session is not None
        protected = replace(first.session, execution=requirements, native_tool_policy=NativeToolPolicy.DENY)
        second = await runtime.ask(binding, "next", call_id="next", resume=protected, limits=CallLimits(max_turns=2))
    assert second.attempts == 0 and second.failure.classification == "execution_incompatible"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_changed_assessed_configuration_refuses_before_credentials(mode):
    from agentkit.agents.prompts.composition import inline_prompt
    from agentkit.models import CallLimits, ModelSettings

    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credentials accessed")))
    adapter = ApiAdapter() if binding.mode.is_api else CliAdapter(spawn=lambda: pytest.fail("spawned"))
    requirements = ExecutionRequirements(required=frozenset({ExecutionRequirement.HOST_TOOLS}))
    assessed = await adapter.evaluate_execution(binding, execution=requirements)
    assert assessed.compatible
    if binding.mode.is_api:
        adapter.base_url = "https://changed.invalid"
    else:
        adapter.working_dir = Path("/changed")
    result = await adapter.turn(
        binding,
        inline_prompt("task"),
        settings=ModelSettings(),
        limits=CallLimits(max_turns=2),
        schema=None,
        tools=None,
        session=None,
        cancel=None,
        emit=None,
        execution=requirements,
        assessment_id=assessed.configuration_id,
    )
    assert result.failure.mode_detail["dispatch_state"] == "not_started"
    assert result.failure.mode_detail["incompatibilities"][0]["code"] == "execution_configuration_changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["claude_code", "codex"])
@pytest.mark.parametrize("measured", [False, True])
async def test_cli_usage_absence_is_distinct_from_measured_zero(mode, measured):
    from agentkit.core.testing.fakes import FakeClaudeCli, FakeCodexCli, codex_turn
    from agentkit.models import CallLimits

    usage = {"input_tokens": 0, "output_tokens": 0} if measured else None
    if mode == "claude_code":
        result_frame = {"type": "result", "subtype": "success", "result": "done"}
        if measured:
            result_frame["usage"] = usage
        spawn = FakeClaudeCli.script([result_frame])
    else:
        spawn = FakeCodexCli.script(codex_turn(text="done", usage=(0, 0, 0) if measured else None))
    key = "ANTHROPIC_API_KEY" if mode == "claude_code" else "OPENAI_API_KEY"
    binding = Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-only"}))
    async with CallRuntime(adapters={mode: CliAdapter(spawn=spawn)}, unpriced_policy="proceed_unpriced") as runtime:
        result = await runtime.ask(binding, "task", call_id="measurement", limits=CallLimits(max_turns=2))
    assert result.outcome == "answered"
    assert result.usage_complete is measured
    if not measured:
        assert result.raw_usage == {} and result.cost_usd is None


@pytest.mark.asyncio
async def test_subprocess_port_refuses_protected_requirements_before_start(monkeypatch):
    from agentkit.models.providers.cli import CliInvocation, SubprocessExecutionBackend

    async def forbidden(*args, **kwargs):
        pytest.fail("protected process started")

    monkeypatch.setattr("asyncio.create_subprocess_exec", forbidden)
    invocation = CliInvocation(
        ("codex", "exec"),
        {"start_new_session": True},
        mode="codex",
        call_id="call",
        attempt_id="call:1",
        native_tool_policy=NativeToolPolicy.DENY,
        execution=ExecutionRequirements(),
    )
    with pytest.raises(ValueError, match="execution requirements"):
        await SubprocessExecutionBackend().start(invocation)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_implicit_deny_requires_assessment_even_for_an_injected_adapter(mode):
    class Unknown:
        async def turn(self, *args, **kwargs):
            pytest.fail("unassessed adapter dispatched under DENY")

    binding = Binding(mode, "test", CredentialRef(lambda: pytest.fail("credential released")))
    async with CallRuntime(adapters={mode: Unknown()}, native_tool_policy="deny") as runtime:
        result = await runtime.ask(binding, "task", call_id="implicit-deny")
    assert result.attempts == 0
    assert result.failure.classification == "execution_incompatible"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["openai_api", "anthropic_api", "claude_code", "codex"])
async def test_async_observer_cannot_replace_the_assessed_request_hook(mode):
    import httpx

    from agentkit.agents.prompts.composition import inline_prompt
    from agentkit.core.testing.fakes import FakeClaudeCli, FakeCodexCli, codex_turn
    from agentkit.models import CallLimits, ModelSettings

    client = None
    if mode.endswith("_api"):
        if mode == "openai_api":
            events = [
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
                }
            ]
        else:
            events = [
                {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"},
            ]
        body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))
        adapter = ApiAdapter(client=client)
    else:
        spawn = (
            FakeClaudeCli.script(
                [
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "done",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ]
            )
            if mode == "claude_code"
            else FakeCodexCli.script(codex_turn(text="done", usage=(1, 0, 1)))
        )
        adapter = CliAdapter(spawn=spawn)

    def malicious_replacement(request):
        pytest.fail("unassessed request hook was used")

    async def observer(event):
        if event.fields.get("mode_event") == "execution_policy":
            adapter.request_hook = malicious_replacement

    key = "ANTHROPIC_API_KEY" if mode in ("anthropic_api", "claude_code") else "OPENAI_API_KEY"
    binding = Binding(mode, "test", CredentialRef(lambda: {key: "synthetic-test-only"}))
    requirements = ExecutionRequirements(required=frozenset({ExecutionRequirement.HOST_TOOLS}))
    assessed = await adapter.evaluate_execution(binding, execution=requirements)
    try:
        result = await adapter.turn(
            binding,
            inline_prompt("task"),
            settings=ModelSettings(),
            limits=CallLimits(max_turns=2),
            schema=None,
            tools=None,
            session=None,
            cancel=None,
            emit=observer,
            call_id="pinned",
            execution=requirements,
            assessment_id=assessed.configuration_id,
        )
        assert result.failure is None
    finally:
        if hasattr(adapter, "aclose"):
            await adapter.aclose()
        if client is not None:
            await client.aclose()
