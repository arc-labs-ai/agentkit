"""Conservative SDK assessments. A host callback is host authority, not a sandbox.

The CLI matrix lives in :mod:`agentkit.models.providers.cli.capabilities`; this is
its counterpart for the two API modes, and it answers a different shape of
question. A CLI invocation is a provider-owned process with provider-native
tools inside it, so the interesting doubt is what that process may do. An API
invocation has no process and no native tools: every tool the model can reach is
a function this host wrote, declared by the adapter and invoked by the runtime.

That inverts two verdicts relative to intuition, and both inversions are the
reason this module exists:

* ``HOST_TOOLS``/``SCHEMA``/``USAGE`` are SUPPORTED here. The adapter puts the
  host's tool declarations and compiled output schema on the wire, parses the
  provider's tool calls and token usage back off it, and the runtime invokes the
  host's own ``ToolSet`` — argument validation, approval gating, timeouts and
  cancellation included. Reporting these UNSUPPORTED (the previous behaviour)
  refused the SDK modes for precisely the hosts that had the most control.
* every CONTAINMENT boundary is UNSUPPORTED. Those callbacks run in this
  process, with this process's filesystem, network, descendants, environment and
  the provider credential itself in reach. As the design record puts it, SDK
  host callbacks do not prove arbitrary callback implementations are sandboxed.

Everything else follows the CLI module's rules, which are deliberately not
restated as new policy: a requirement is SUPPORTED only where the shipped code
path implements it, UNVERIFIED means "no verifier exists yet" and still refuses
when required, and a :class:`TransportReport`'s self-declared ``guarantees`` can
only DOWNGRADE a verdict to UNVERIFIED — never raise one to SUPPORTED, because
that would let whoever constructs the report decide which boundaries exist.
Identity (transport name, version, wire contract) is diagnostic metadata and
never participates in a status.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.constants import NativeToolPolicy
from agentkit.models.contracts.execution import (
    CapabilityStatus,
    ExecutionAssessment,
    ExecutionCompatibility,
    ExecutionRequirement,
    ExecutionRequirements,
)
from agentkit.models.providers.auth import ANTHROPIC_API_VERSION


@dataclass(frozen=True)
class TransportReport:
    """Transport observations, not an attestation agentkit independently verified.

    Mirrors ``BackendReport`` on the CLI side, including the rule that matters:
    no shipped verifier accepts a declared guarantee as isolation evidence.
    """

    backend: str
    version: str
    guarantees: frozenset[ExecutionRequirement] = frozenset()


# The remote interface each mode targets. This is the API counterpart of a CLI
# binary version: identity that belongs in an incompatibility report so an
# operator can tell which transport produced it, and that must never move a
# status. No base URL appears here — a custom endpoint is host configuration.
_WIRE_CONTRACT = {
    Mode.ANTHROPIC_API: f"anthropic-messages@{ANTHROPIC_API_VERSION}",
    Mode.OPENAI_API: "openai-responses@v1",
}

_PROTOCOL_SUPPORT = {
    ExecutionRequirement.HOST_TOOLS: (
        "host_tool_protocol_supported",
        "The adapter declares host tool schemas to the provider and parses its tool calls back; the runtime "
        "invokes them through the host ToolSet, which validates arguments and enforces approval and timeouts. "
        "What those tools may DO remains the host's own authority.",
    ),
    ExecutionRequirement.SCHEMA: (
        "structured_output_supported",
        "The compiled output schema is sent as this provider's structured-output format and every answer is "
        "validated against it by the runtime, with one bounded correction turn.",
    ),
    ExecutionRequirement.USAGE: (
        "usage_metering_supported",
        "Provider token usage is parsed from the response stream, reported incomplete when the stream ends "
        "without its terminal frame, and priced by the runtime's price table.",
    ),
    ExecutionRequirement.OBSERVED_SPENDING_THRESHOLD: (
        "observed_spending_threshold_supported",
        "The runtime prices observed usage and ends the invocation once the caller's max_cost_usd is reached, "
        "failing closed when a token kind is unpriced. Enforcement is between turns, so the turn that crosses "
        "the threshold completes.",
    ),
}


def evaluate_api_execution(
    mode: Mode,
    report: TransportReport,
    execution: ExecutionRequirements,
    policy: NativeToolPolicy,
    *,
    payload_rewriting: bool = False,
) -> ExecutionCompatibility:
    """Evaluate requirements without resolving credentials or contacting a provider.

    Returns the full matrix for every requirement, not just the requested
    subset, so a host can use it for discovery. ``payload_rewriting`` reports
    that a host request hook replaces the request body after the adapter builds
    it, which is the one way the dispatched payload stops being the payload this
    module can speak for.
    """
    if not mode.is_api:
        raise ValueError("evaluate_api_execution assesses API modes; use evaluate_cli_execution for CLI modes")
    required = set(execution.required)
    if policy == NativeToolPolicy.DENY:
        required.add(ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE)
    facts = []
    for requirement in ExecutionRequirement:
        status = CapabilityStatus.UNSUPPORTED
        code = f"{requirement.value}_boundary_unavailable"
        explanation = (
            "Host tool callbacks run inside this process with full host authority and agentkit does not "
            f"sandbox them, so no {requirement.value} boundary is enforced on this transport."
        )
        if requirement == ExecutionRequirement.CREDENTIALS:
            code = "credential_boundary_unavailable"
            explanation = (
                "The provider credential is held in this process alongside host tool callbacks, which run with "
                "full host authority; nothing keeps a callback away from it."
            )
        elif requirement == ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE:
            status = CapabilityStatus.SUPPORTED
            code = "native_tools_absent"
            explanation = (
                "An API request declares host functions only: this adapter can enable no provider-native or "
                "provider-hosted tool, so the model has no native tool to call."
            )
            if payload_rewriting:
                status = CapabilityStatus.UNVERIFIED
                code = "native_tools_absence_unverified"
                explanation = (
                    "A host request hook replaces the request body after this adapter builds it, so the absence "
                    "of provider-hosted tools in the dispatched payload is not verified."
                )
        elif requirement in _PROTOCOL_SUPPORT:
            status = CapabilityStatus.SUPPORTED
            code, explanation = _PROTOCOL_SUPPORT[requirement]
        elif requirement == ExecutionRequirement.CANCELLATION:
            status = CapabilityStatus.UNVERIFIED
            code = "cancellation_unverified"
            explanation = (
                "Cancelling the awaiting task closes the stream locally and reports partial usage, but the "
                "cooperative token alone does not interrupt a response in flight and the provider is not "
                "proven to stop, or to stop billing for, work already started."
            )
        elif requirement == ExecutionRequirement.MCP_ALLOWLIST:
            status = CapabilityStatus.UNVERIFIED
            code = "mcp_allowlist_unverified"
            explanation = (
                "No MCP configuration reaches the provider on this transport, but host tools may themselves be "
                "served by MCP clients in this process, which agentkit does not independently verify."
            )
        elif requirement == ExecutionRequirement.HARD_SPENDING_CAP:
            code = "hard_spending_cap_unavailable"
            explanation = (
                "Output token caps and post-hoc usage pricing do not establish an exact hard dollar cap; this "
                "transport enforces no per-request spending limit."
            )
        if payload_rewriting and requirement in {
            ExecutionRequirement.HOST_TOOLS,
            ExecutionRequirement.SCHEMA,
            ExecutionRequirement.MCP_ALLOWLIST,
        }:
            status = CapabilityStatus.UNVERIFIED
            code = "request_hook_authority_unverified"
            explanation = (
                "A host request hook can replace the assessed wire configuration; remove it for this requirement."
            )
        if requirement in report.guarantees and status != CapabilityStatus.SUPPORTED:
            status = CapabilityStatus.UNVERIFIED
            explanation = (
                "The trusted transport declares this guarantee, but agentkit has no verifier for its evidence."
            )
        facts.append(ExecutionAssessment(requirement, status, code, explanation))
    unmet = [fact for fact in facts if fact.requirement in required and fact.status != CapabilityStatus.SUPPORTED]
    return ExecutionCompatibility(
        str(mode), report.backend, report.version, _WIRE_CONTRACT[mode], tuple(facts), tuple(unmet)
    )


def api_transport_report() -> TransportReport:
    """Identify the shipped HTTPS/SSE transport, declaring no guarantee at all.

    The HTTP client version is the closest API analogue of a CLI binary version:
    useful in a refusal, and powerless over a verdict.
    """
    try:
        import httpx

        version = f"1/httpx-{httpx.__version__}"
    except Exception:  # pragma: no cover - httpx is a hard dependency of dispatch, not of assessment
        version = "1/httpx-unidentified"
    return TransportReport("https-sse", version)
