"""Security requirements are assessed before credentials or transport dispatch."""

from __future__ import annotations

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models import (
    Binding,
    CallLimits,
    CredentialRef,
    ExecutionRequirements,
    NativeExecution,
    NativeToolPolicy,
)
from agentkit.models.providers.cli import CliAdapter, SubprocessExecutionBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_containment_refuses_before_credentials_and_spawn(mode):
    def forbidden():
        raise AssertionError("must not resolve credentials")

    async def spawn(*args, **kwargs):
        raise AssertionError("must not dispatch")

    async with CallRuntime(adapters={mode: CliAdapter(spawn=spawn)}) as runtime:
        answer = await runtime.ask(
            Binding(mode, "test", CredentialRef(forbidden)),
            "task",
            call_id="contained",
            execution=ExecutionRequirements(native_execution=NativeExecution.CONTAINED),
            limits=CallLimits(max_turns=1),
        )
    assert answer.failure is not None
    assert answer.failure.classification == "execution_incompatible"
    assert answer.failure.mode_detail["dispatch_state"] == "not_started"
    assert answer.attempts == 0
    assert answer.cost_usd is None
    assert not answer.usage_complete
    assert any(x["code"] == "credential_boundary_unavailable" for x in answer.failure.mode_detail["incompatibilities"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_containment_cannot_weaken_deny(mode):
    adapter = CliAdapter(native_tool_policy=NativeToolPolicy.DENY)
    result = await adapter.evaluate_execution(
        Binding(mode, "test"),
        execution=ExecutionRequirements(native_execution=NativeExecution.CONTAINED),
    )
    assert not result.compatible
    assert any(x.code == "native_policy_conflict" for x in result.incompatibilities)
    assert adapter.native_tool_policy == NativeToolPolicy.DENY


@pytest.mark.asyncio
async def test_codex_deny_is_distinct_from_a_cost_limit():
    result = await CliAdapter(native_tool_policy=NativeToolPolicy.DENY).evaluate_execution(Binding(Mode.CODEX, "test"))
    assert not result.compatible
    assert [x.code for x in result.incompatibilities] == ["native_tools_unavailable"]
    assert result.incompatibilities[0].requirement == "native_tools_unavailable"


@pytest.mark.asyncio
async def test_codex_cannot_be_verified_later_by_accident() -> None:
    """UNVERIFIED and UNSUPPORTED are not interchangeable, and conflating them is
    how Codex would be promoted by accident.

    UNVERIFIED means "no verifier exists yet" — Claude's empty `--tools` may well
    deny native tools, and nobody has proved it for this binary. UNSUPPORTED
    means "cannot be done": Codex keeps file, patch and plan capabilities that
    an empty tool list does not remove, which is a property of the binary rather
    than a gap in our evidence.

    Both are refused today, because `unmet` requires SUPPORTED, so the
    difference is invisible from the compatible/incompatible verdict alone —
    which is exactly why it needs asserting. The day somebody writes a verifier
    for the UNVERIFIED case, anything mislabelled UNVERIFIED gets promoted with
    it. Codex must not be in that set.
    """
    claude = await CliAdapter(native_tool_policy=NativeToolPolicy.DENY).evaluate_execution(
        Binding(Mode.CLAUDE_CODE, "test")
    )
    codex = await CliAdapter(native_tool_policy=NativeToolPolicy.DENY).evaluate_execution(
        Binding(Mode.CODEX, "test")
    )

    def native_tools(result: object) -> object:
        return next(
            fact
            for fact in result.assessments  # type: ignore[attr-defined]
            if fact.requirement == "native_tools_unavailable"
        )

    assert native_tools(codex).status == "unsupported", (
        "Codex cannot prove native-tool isolation; calling that merely "
        "'unverified' invites a future verifier to promote it"
    )
    assert native_tools(claude).status == "unverified", (
        "Claude's denial is unproven for this binary, not impossible"
    )
    assert native_tools(codex).status != native_tools(claude).status
    assert "cannot prove" in native_tools(codex).explanation


def test_no_caller_boolean_can_attest_isolation():
    with pytest.raises(TypeError):
        ExecutionRequirements(isolated=True)


@pytest.mark.asyncio
async def test_subprocess_report_does_not_claim_containment():
    report = await SubprocessExecutionBackend().inspect()
    assert report.backend == "subprocess"
    assert not report.guarantees
    assert report.version


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.ANTHROPIC_API, Mode.OPENAI_API])
async def test_execution_requirements_are_not_silently_ignored_in_api_modes(mode):
    async with CallRuntime() as runtime:
        answer = await runtime.ask(
            Binding(mode, "test"),
            "task",
            call_id="api-execution",
            execution=ExecutionRequirements(native_execution=NativeExecution.CONTAINED),
            limits=CallLimits(max_turns=1),
        )
    assert answer.failure.classification == "execution_incompatible"
    assert answer.attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_supervisor_disconnect_during_start_is_uncertain_and_never_retried(mode):
    from agentkit.models.providers.cli import BackendReport, ExecutionOutcomeUnknown

    class Disconnected:
        starts = 0

        async def inspect(self):
            return BackendReport("test-supervisor", "1")

        async def start(self, invocation):
            self.starts += 1
            raise ExecutionOutcomeUnknown("planted-unstructured-private-material")

    backend = Disconnected()
    async with CallRuntime(
        adapters={mode: CliAdapter(execution_backend=backend)},
        unpriced_policy="proceed_unpriced",
    ) as runtime:
        answer = await runtime.ask(
            Binding(
                mode,
                "test",
                CredentialRef(
                    lambda: {
                        "ANTHROPIC_API_KEY" if mode == Mode.CLAUDE_CODE else "OPENAI_API_KEY": "test-only",
                    }
                ),
            ),
            "q",
            limits=CallLimits(max_turns=1),
            call_id="lost",
        )
    assert answer.failure.classification == "execution_uncertain"
    assert answer.failure.mode_detail["dispatch_state"] == "unknown"
    assert backend.starts == 1 and answer.attempts == 1
    assert not answer.usage_complete and answer.cost_usd is None
    assert "planted-unstructured-private-material" not in repr(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_backend_declaration_is_not_an_isolation_attestation(mode):
    from agentkit.models import ExecutionRequirement
    from agentkit.models.providers.cli import BackendReport

    class UnverifiedBackend:
        async def inspect(self):
            return BackendReport("self-declared", "1", frozenset(ExecutionRequirement))

        async def start(self, invocation):
            raise AssertionError("declarations do not authorize startup")

    report = await CliAdapter(execution_backend=UnverifiedBackend()).evaluate_execution(
        Binding(mode, "test"),
        execution=ExecutionRequirements(native_execution=NativeExecution.CONTAINED),
    )
    assert not report.compatible
    assert all(item.status == "unverified" for item in report.incompatibilities)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_failed_backend_inspection_is_a_typed_local_refusal(mode):
    class Unavailable:
        async def inspect(self):
            raise OSError("planted-private-path")

        async def start(self, invocation):
            raise AssertionError("inspection failed")

    result = await CliAdapter(execution_backend=Unavailable()).evaluate_execution(Binding(mode, "test"))
    assert not result.compatible
    assert result.incompatibilities[0].code == "backend_inspection_failed"
    assert "planted-private-path" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
@pytest.mark.parametrize(
    "boundary",
    [
        "filesystem_read",
        "filesystem_write",
        "network",
        "processes",
        "credentials",
        "configuration",
        "cancellation",
        "hard_spending_cap",
    ],
)
async def test_every_unproved_boundary_independently_refuses(mode, boundary):
    from agentkit.models import ExecutionRequirement

    report = await CliAdapter(spawn=lambda: None).evaluate_execution(
        Binding(mode, "test"),
        execution=ExecutionRequirements(required=frozenset({ExecutionRequirement(boundary)})),
    )
    assert not report.compatible
    assert len(report.incompatibilities) == 1
    assert report.incompatibilities[0].requirement == boundary


def test_containment_defaults_each_authority_to_deny():
    request = ExecutionRequirements(native_execution=NativeExecution.CONTAINED)
    assert request.read_roots == ()
    assert request.write_roots == ()
    assert request.network_destinations == ()


@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
@pytest.mark.parametrize("reason", ["cli_exit_1", "cli_exit_137"])
def test_unexplained_worker_exit_is_uncertain(mode, reason):
    from agentkit.models.providers.cli.calls import _classify_cli_failure

    failure = _classify_cli_failure(mode, reason, {})
    assert failure.classification == "execution_uncertain"
    assert failure.retryable is False
    assert failure.mode_detail["dispatch_state"] == "unknown"


@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
def test_confirmed_spawn_failure_is_distinct_from_worker_loss(mode):
    from agentkit.models.providers.cli.calls import _classify_cli_failure

    failure = _classify_cli_failure(mode, "spawn_failed", {})
    assert failure.classification == "transport"
    assert failure.mode_detail["dispatch_state"] == "not_started"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.CLAUDE_CODE, Mode.CODEX])
async def test_missing_binary_cannot_be_reported_executable(mode, tmp_path):
    report = await CliAdapter(executable={mode: str(tmp_path / "does-not-exist")}).evaluate_execution(
        Binding(mode, "test")
    )
    assert not report.compatible
    assert report.incompatibilities[0].code == "cli_binary_unverified"
    assert report.binary_version is None
