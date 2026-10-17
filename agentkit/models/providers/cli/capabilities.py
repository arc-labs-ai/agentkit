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
        elif requirement == ExecutionRequirement.HARD_SPENDING_CAP:
            code = "hard_spending_cap_unavailable"
            explanation = "Usage metering and cancellation do not establish an exact hard dollar cap."
        if requirement in report.guarantees and status != CapabilityStatus.SUPPORTED:
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
    return ExecutionCompatibility(str(mode), report.backend, report.version, None, tuple(facts), tuple(unmet))
