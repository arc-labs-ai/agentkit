"""API transport capabilities, scoped separately from arbitrary host callbacks."""

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.constants import NativeToolPolicy
from agentkit.models.contracts.execution import (
    CapabilityStatus,
    ExecutionAssessment,
    ExecutionCompatibility,
    ExecutionRequirement,
    ExecutionRequirements,
)


def evaluate_api_execution(
    mode: Mode,
    execution: ExecutionRequirements,
    policy: NativeToolPolicy,
    *,
    request_hook: object | None = None,
) -> ExecutionCompatibility:
    """Inspect the implemented wire protocol without credentials or network I/O."""
    implemented = {
        ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE: (
            "Only host function tool definitions are emitted; provider-native tools are never configured."
        ),
        ExecutionRequirement.HOST_TOOLS: (
            "Host callbacks use ToolSet validation and approval checks; arbitrary callback implementations "
            "are trusted host code and are not sandboxed."
        ),
        ExecutionRequirement.SCHEMA: (
            "Mode-specific schema compilation and canonical response validation are implemented."
        ),
        ExecutionRequirement.USAGE: "Captured token kinds are metered; missing usage remains incomplete.",
        ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD: (
            "The runtime checks observed cost between turns and cancels on observed thresholds; this is not a hard cap."
        ),
    }
    facts = tuple(
        ExecutionAssessment(
            requirement,
            CapabilityStatus.SUPPORTED if requirement in implemented else CapabilityStatus.UNSUPPORTED,
            "api_protocol_supported" if requirement in implemented else f"{requirement.value}_boundary_unavailable",
            implemented.get(
                requirement,
                (
                    "The API adapter does not sandbox host callbacks or establish this execution boundary. "
                    "A host execution mechanism is required."
                ),
            )
            if requirement != ExecutionRequirement.HARD_SPENDING_CAP
            else "Usage metering and cancellation do not establish an exact hard dollar cap.",
        )
        for requirement in ExecutionRequirement
    )
    if request_hook is not None:
        wire_requirements = {
            ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE,
            ExecutionRequirement.HOST_TOOLS,
            ExecutionRequirement.SCHEMA,
        }
        facts = tuple(
            ExecutionAssessment(
                fact.requirement,
                CapabilityStatus.UNVERIFIED,
                "request_hook_authority_unverified",
                "An unrestricted request hook can rewrite the tool catalog or schema; "
                "remove it for this execution profile.",
            )
            if fact.requirement in wire_requirements
            else fact
            for fact in facts
        )
    required = set(execution.required)
    if policy == NativeToolPolicy.DENY:
        required.add(ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE)
    return ExecutionCompatibility(
        str(mode),
        "host",
        "unversioned",
        None,
        facts,
        tuple(fact for fact in facts if fact.requirement in required and fact.status != CapabilityStatus.SUPPORTED),
        adapter="agentkit.models.providers.frontiers.calls.ApiAdapter",
    )
