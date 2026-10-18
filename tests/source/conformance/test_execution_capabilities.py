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


@pytest.mark.asyncio
async def test_the_cli_observed_spending_threshold_is_credited_to_the_runtime_and_not_to_a_flag():
    """The CLI matrix reported `observed_spending_threshold` UNSUPPORTED by falling through to
    the module's default, which was wrong in the one direction that costs a caller nothing to
    read but everything to work around: a host that asks only for a post-hoc spending threshold
    was refused a threshold the shipped runtime does enforce for every mode, CLI included.

    What is proven is the runtime's own accounting, which does not depend on trusting the
    binary: observed CLI usage is priced (or taken from the run's reported cost), accumulated,
    and compared against `max_cost_usd` before the next dispatch, failing closed when the spend
    cannot be known. What is NOT proven is anything about `--max-budget-usd`. The CLI accepting
    that argument is argv acceptance, and this module's first line is that flags are not proofs —
    so the explanation must not rest on it, and this asserts the flag is absent from the reason.
    Without this row a correct capability is reported as missing; without the flag assertion the
    next person to touch the row can justify it with the CLI's documentation instead of our code.
    """
    from agentkit.models import ExecutionRequirement

    adapter = CliAdapter(spawn=lambda: None)
    report = await adapter.evaluate_execution(
        Binding(Mode.CLAUDE_CODE, "test"),
        execution=ExecutionRequirements(required=frozenset({ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD})),
    )
    fact = next(x for x in report.assessments if x.requirement == "observed_spending_threshold")

    assert fact.status == "supported"
    assert fact.code == "observed_spending_threshold_supported"
    assert report.compatible, "a host asking only for a post-hoc threshold must not be refused"
    assert "--max-budget-usd" not in fact.explanation, (
        "argv acceptance is not evidence; the reason must cite the runtime's accounting"
    )
    # The scope of the claim, stated in the reason itself: the unit of enforcement is a whole
    # CLI run, so the run that crosses the threshold still finishes, and that is not a cap.
    assert "between" in fact.explanation and "not a hard cap" in fact.explanation


@pytest.mark.asyncio
async def test_an_observed_cli_threshold_stops_the_next_spawn_while_a_hard_cap_stays_refused():
    """`observed_spending_threshold` and `hard_spending_cap` must stay two requirements, and the
    CLI side is where collapsing them is most tempting because `--max-budget-usd` looks like a
    cap. It is not one, and neither is what we actually ship: the threshold is observed, so the
    CLI run that crosses it runs to completion and is paid for, and only the NEXT dispatch is
    withheld. This asserts that bite directly — a first run that reports more spend than the
    caller's ceiling ends the invocation at `limit_reached` with one spawn, where the same
    script under a ceiling it does not reach is retried and answered.

    A host that genuinely needs a mid-flight dollar cap is still refused before dispatch. If
    these two ever answer together, one of the two answers is a lie: either a real threshold is
    withheld from callers who could use it, or a threshold is sold under a cap's name and a
    caller believes a run cannot overspend when it demonstrably can.
    """
    from decimal import Decimal

    from agentkit.core.runtime.calls import RetryPolicy
    from agentkit.core.testing.fakes import CliRun, FakeClaudeCli
    from agentkit.models import CredentialRef, ExecutionRequirement
    from agentkit.models.catalog.pricing import PriceTable

    prices = PriceTable("test", "2026-10-13", {"test": {"input_uncached": "1", "output": "5"}})
    threshold = ExecutionRequirements(required=frozenset({ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD}))

    def script():
        return FakeClaudeCli(
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
                        }
                    ],
                    returncode=1,
                ),
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
                            "total_cost_usd": 0.2,
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    ]
                ),
            ]
        )

    crossed, spare = script(), script()
    retry = RetryPolicy(base_s=0, cap_s=1, max_elapsed_s=2)
    credential = CredentialRef(lambda: {"ANTHROPIC_API_KEY": "test-only"})

    async with CallRuntime(
        adapters={Mode.CLAUDE_CODE: CliAdapter(spawn=crossed)}, pricing=prices, retry=retry
    ) as runtime:
        stopped = await runtime.ask(
            Binding(Mode.CLAUDE_CODE, "test", credential),
            "overspend the ceiling",
            execution=threshold,
            limits=CallLimits(max_cost_usd=Decimal("0.3")),
            call_id="cli-observed-threshold",
        )
    assert stopped.outcome == "limit_reached", stopped.failure
    assert crossed.spawns == 1, "the threshold must withhold the next spawn, not the first one"
    assert stopped.cost_usd == Decimal("0.4"), "the run that crossed the threshold is still paid for"

    async with CallRuntime(
        adapters={Mode.CLAUDE_CODE: CliAdapter(spawn=spare)}, pricing=prices, retry=retry
    ) as runtime:
        answered = await runtime.ask(
            Binding(Mode.CLAUDE_CODE, "test", credential),
            "stay under the ceiling",
            execution=threshold,
            limits=CallLimits(max_cost_usd=Decimal("5")),
            call_id="cli-observed-headroom",
        )
    assert answered.outcome == "answered" and spare.spawns == 2

    async def forbidden(*args, **kwargs):
        raise AssertionError("a hard cap has no implementation to dispatch to")

    async with CallRuntime(adapters={Mode.CLAUDE_CODE: CliAdapter(spawn=forbidden)}, pricing=prices) as runtime:
        capped = await runtime.ask(
            Binding(Mode.CLAUDE_CODE, "test", CredentialRef(forbidden)),
            "cap the run mid-flight",
            execution=ExecutionRequirements(required=frozenset({ExecutionRequirement.HARD_SPENDING_CAP})),
            limits=CallLimits(max_cost_usd=Decimal("0.3")),
            call_id="cli-hard-cap",
        )
    assert capped.failure.classification == "execution_incompatible"
    assert [x["code"] for x in capped.failure.mode_detail["incompatibilities"]] == ["hard_spending_cap_unavailable"]
    assert capped.attempts == 0


def test_the_execution_refusal_is_shared_contract_code_rather_than_borrowed_from_the_cli():
    """`execution_failure` turns an `ExecutionCompatibility` into the one public refusal both
    provider families must emit, and it consumes nothing but contract types. While it lived in
    the CLI adapter the API refusal path had to reach into `providers.cli` for it — a local
    import buried in the refusal branch, which is how a dependency nobody wants becomes
    permanent: it compiles, it is invisible in the import block, and it quietly says the API
    transport is downstream of the CLI one.

    Two things break without this. A refactor of the CLI package drags the API refusal with it,
    and the refusal's shape is free to drift per family — the whole point of one `Failure`
    builder is that a host parsing `mode_detail` sees the same keys whichever mode refused.
    The last assertion is the cycle guard: the contracts layer answers the providers and must
    never import them back, or the next import of a contract type deadlocks at module load.
    """
    import inspect

    from agentkit.models.contracts import execution as execution_contract
    from agentkit.models.contracts import failures
    from agentkit.models.contracts.failures import execution_failure
    from agentkit.models.providers.frontiers import calls as api_calls

    assert execution_failure.__module__ == "agentkit.models.contracts.failures"
    assert "providers.cli" not in inspect.getsource(api_calls), (
        "the API refusal path must not import the CLI package to refuse"
    )
    for module in (execution_contract, failures):
        assert "agentkit.models.providers" not in inspect.getsource(module), (
            f"{module.__name__} is below the providers; importing one back would cycle"
        )


@pytest.mark.asyncio
async def test_both_provider_families_refuse_an_impossible_request_in_the_same_shape():
    """One refusal builder, so a host parses one shape. A CLI mode and an API mode refuse a
    contained request for different reasons, and the reasons differ on purpose — but the
    envelope must not: same classification, same `mode_detail` keys, `dispatch_state` of
    `not_started`, zero attempts. If the two families ever build their own `Failure` the keys
    drift silently and a host that switched modes starts reading `None` out of a dict it was
    parsing correctly a release earlier.
    """
    async def forbidden(*args, **kwargs):
        raise AssertionError("a refusal precedes dispatch")

    contained = ExecutionRequirements(native_execution=NativeExecution.CONTAINED)
    refusals = {}
    for mode, adapters in ((Mode.CLAUDE_CODE, {Mode.CLAUDE_CODE: CliAdapter(spawn=forbidden)}), (Mode.OPENAI_API, {})):
        async with CallRuntime(adapters=adapters) as runtime:
            answer = await runtime.ask(
                Binding(mode, "test", CredentialRef(forbidden)),
                "task",
                execution=contained,
                limits=CallLimits(max_turns=1),
                call_id=f"shape-{mode}",
            )
        assert answer.attempts == 0
        assert answer.failure.classification == "execution_incompatible"
        assert answer.failure.mode_detail["dispatch_state"] == "not_started"
        refusals[mode] = set(answer.failure.mode_detail)

    assert refusals[Mode.CLAUDE_CODE] == refusals[Mode.OPENAI_API]


@pytest.mark.asyncio
async def test_codex_reports_the_spending_threshold_as_unsupported_because_dispatch_refuses_it() -> None:
    """The matrix must agree with what `run()` actually does, and for Codex it did not.

    `observed_spending_threshold` was credited to the runtime's accounting, which is
    mode-independent — so the row came out SUPPORTED for Codex too. But the runtime
    short-circuits any Codex call carrying `max_cost_usd` with `invalid_request` before
    dispatch, so the capability is unreachable on that mode whatever this module claims. An
    executable-looking assessment followed by a refusal is the exact failure a capability
    matrix exists to prevent.

    The refusal is not an oversight either, which is why the row moved rather than the guard:
    enforcement is between runs, and one Codex run is a whole autonomous session that can spend
    without bound before the threshold is ever consulted. A cost limit there would read far
    stronger than it is. If that guard is narrowed, this test and the row move together.
    """
    from decimal import Decimal

    from agentkit.models import ExecutionRequirement
    from agentkit.models.contracts.types import CallLimits

    adapter = CliAdapter(spawn=lambda: None)
    report = await adapter.evaluate_execution(
        Binding(Mode.CODEX, "test"),
        execution=ExecutionRequirements(required=frozenset({ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD})),
    )
    fact = next(x for x in report.assessments if x.requirement == "observed_spending_threshold")

    assert fact.status == "unsupported"
    assert fact.code == "observed_spending_threshold_refused_for_mode"
    assert not report.compatible, "a host asking for a threshold Codex cannot honour must be refused"

    # And the refusal the row is reporting is really there: a Codex run carrying the limit is
    # rejected locally, with zero provider attempts.
    #
    # The adapter is given a ``spawn`` rather than left to find the real binary. With a bare
    # ``CallRuntime()`` this assertion depended on whether ``codex`` happened to be INSTALLED on
    # the machine running it: present, and the limit guard refuses as asserted; absent, and
    # ``cli_binary_unverified`` refuses first with ``execution_incompatible``. It passed locally
    # and failed in CI for exactly that reason. A spawn makes the limit guard the thing under
    # test on every machine, and the order the two refusals take is pinned separately below.
    async def unreachable(*args, **kwargs):
        raise AssertionError("a threshold Codex cannot honour must be refused before any spawn")

    async with CallRuntime(adapters={Mode.CODEX: CliAdapter(spawn=unreachable)}) as runtime:
        answer = await runtime.ask(
            Binding(Mode.CODEX, "test"),
            "anything",
            limits=CallLimits(max_cost_usd=Decimal("1.00")),
            call_id="codex-threshold-refused",
        )
    assert answer.outcome == "failed"
    assert answer.failure is not None
    assert answer.failure.classification == "invalid_request", answer.failure
    assert answer.attempts == 0


@pytest.mark.asyncio
async def test_an_unidentifiable_cli_is_refused_as_incompatible_before_any_limit_is_consulted():
    """Which refusal wins when the binary is missing AND the limits are unhonourable.

    Both refusals are correct and only one can be reported, so the order is a contract rather
    than an accident — and it was an accident until it broke in CI. The binary is identified
    first, which means a host whose executable is missing is told to install it before being
    told anything about its limits. That is the useful order: the limits refusal is permanent
    for Codex and will still be there afterwards, whereas an unidentified binary makes every
    other assessment a guess about a program nobody has looked at.

    Deterministic on any machine because the executable is pointed at a path that cannot
    exist, rather than relying on ``codex`` being absent from PATH.
    """
    from decimal import Decimal

    from agentkit.models.contracts.types import CallLimits

    adapters = {Mode.CODEX: CliAdapter(executable={Mode.CODEX: "/nonexistent/codex-not-installed"})}
    async with CallRuntime(adapters=adapters) as runtime:
        answer = await runtime.ask(
            Binding(Mode.CODEX, "test"),
            "anything",
            limits=CallLimits(max_cost_usd=Decimal("1.00")),
            call_id="codex-binary-unverified",
        )
    assert answer.outcome == "failed"
    assert answer.failure is not None
    assert answer.failure.classification == "execution_incompatible", answer.failure
    assert answer.attempts == 0
    assert "cli_binary_unverified" in repr(answer.failure.mode_detail)
