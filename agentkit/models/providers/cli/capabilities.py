"""Conservative provider assessments. Documentation and flags are not proofs."""

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.constants import NativeToolPolicy
from agentkit.models.contracts.execution import (
    CapabilityStatus,
    ExecutionAssessment,
    ExecutionCompatibility,
    ExecutionRequirement,
    ExecutionRequirements,
    NativeExecution,
)
from agentkit.models.providers.cli.execution import BackendReport


def evaluate_cli_execution(
    mode: Mode,
    report: BackendReport,
    execution: ExecutionRequirements,
    policy: NativeToolPolicy,
) -> ExecutionCompatibility:
    """Evaluate requirements without accessing credentials or starting the CLI.

    There is currently no verified provider/backend containment combination.
    Declaring guarantees in BackendReport cannot enable one. New combinations
    require an evidence verifier and real planted-violation acceptance tests.
    """
    policy = execution.constrain_native_policy(policy)
    required = set(execution.required)
    if policy == NativeToolPolicy.DENY:
        required.add(ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE)
    facts = []
    for requirement in ExecutionRequirement:
        status = CapabilityStatus.UNSUPPORTED
        code = f"{requirement.value}_boundary_unavailable"
        explanation = f"The {report.backend} backend has no verified {requirement.value} enforcement."
        if requirement == ExecutionRequirement.CREDENTIALS:
            code = "credential_boundary_unavailable"
            explanation = (
                "CLI authentication and native tools share authority; use a separate credential-free executor."
            )
        elif requirement == ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE:
            code = "native_tools_unavailable"
            if mode == Mode.CODEX:
                explanation = "Codex cannot prove complete native-tool isolation; DENY policy refuses dispatch"
            else:
                status = CapabilityStatus.UNVERIFIED
                explanation = (
                    "Claude --tools disables native tools, but this binary/environment has no verified denial profile."
                )
        elif requirement in (ExecutionRequirement.HOST_TOOLS, ExecutionRequirement.SCHEMA, ExecutionRequirement.USAGE):
            status = CapabilityStatus.SUPPORTED
            code = "adapter_supported"
            explanation = (
                "The adapter supports this protocol; runtime validation and host tool grants remain authoritative."
            )
        elif requirement == ExecutionRequirement.MCP_ALLOWLIST:
            status = CapabilityStatus.UNVERIFIED
            explanation = "Explicit MCP configuration is supplied; discovery exclusion is not independently verified."
        elif requirement == ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD:
            if mode == Mode.CODEX:
                # Codex is UNSUPPORTED here for a reason that is NOT about the
                # mechanism: the runtime refuses a Codex call carrying
                # max_cost_usd outright (core/runtime/calls.py), so the
                # capability is unreachable whatever this module says. The one
                # rule this assessment exists to keep is that it must agree with
                # what dispatch actually does — a reachable-looking verdict
                # followed by a refusal is the precise failure the matrix is for.
                #
                # The refusal is itself defensible rather than an oversight.
                # Enforcement is between runs, and for Codex one run is a whole
                # autonomous session that can spend without bound before the
                # threshold is ever consulted, so a cost limit there reads far
                # stronger than it is. If that guard is ever narrowed, this row
                # and its test move together.
                code = "observed_spending_threshold_refused_for_mode"
                explanation = (
                    "The runtime refuses max_cost_usd on Codex: enforcement is between runs, and one Codex run is "
                    "a whole autonomous session that can spend without bound before the threshold is consulted."
                )
            else:
                status = CapabilityStatus.SUPPORTED
                code = "observed_spending_threshold_supported"
                explanation = (
                    "The runtime accumulates what each CLI run reports, or prices the usage it observed, and ends "
                    "the invocation once the caller's max_cost_usd is reached, failing closed when the spend "
                    "cannot be known. Enforcement is between runs and a whole autonomous run is one unit, so the "
                    "run that crosses the threshold completes and is billed: this is not a hard cap, and no "
                    "argument the binary accepts makes it one."
                )
        elif requirement == ExecutionRequirement.HARD_SPENDING_CAP:
            code = "hard_spending_cap_unavailable"
            explanation = "Usage metering and cancellation do not establish an exact hard dollar cap."
        if (
            requirement in report.guarantees
            and status != CapabilityStatus.SUPPORTED
            and requirement != ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE
        ):
            status = CapabilityStatus.UNVERIFIED
            explanation = "The trusted backend declares this guarantee, but agentkit has no verifier for its evidence."
        facts.append(ExecutionAssessment(requirement, status, code, explanation))
    unmet = [fact for fact in facts if fact.requirement in required and fact.status != CapabilityStatus.SUPPORTED]
    if execution.native_execution == NativeExecution.CONTAINED and policy == NativeToolPolicy.DENY:
        unmet.insert(
            0,
            ExecutionAssessment(
                ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE,
                CapabilityStatus.UNSUPPORTED,
                "native_policy_conflict",
                "Contained native execution cannot satisfy strict native-tool denial.",
            ),
        )
    return ExecutionCompatibility(
        str(mode), report.backend, report.version, None, tuple(facts), tuple(unmet), profile=execution.profile,
    )
