"""Evidence coverage distinguishes planned cases from demonstrated passes.

A feature witness is scoped to the case's acceptance checks. It never implies
that every model, transport, deployment or application behavior is covered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentkit.core.contracts.types import Mode


class FeatureAspect(StrEnum):
    FINAL_ANSWER = "final_answer"
    STRUCTURED_OUTPUT = "structured_output"
    SETTINGS = "settings"
    EVENTS = "events"
    USAGE_PRICING = "usage_pricing"
    FAILURE_RECOVERY = "failure_recovery"
    LIMITS = "limits"
    CANCELLATION = "cancellation"
    ISOLATION = "isolation"
    SESSIONS_CONTEXT = "sessions_context"
    PROMPT_CACHING = "prompt_caching"
    HOOKS = "hooks"
    TOOLS = "tools"
    MCP = "mcp"
    SKILLS = "skills"
    DELEGATION = "delegation"
    MINDS = "minds"
    MEMORY = "memory"
    CONTEXT_COMPACTION = "context_compaction"
    CHECKPOINTS = "checkpoints"
    JOURNALS = "journals"
    REPLAY = "replay"
    LEASES_FENCING = "leases_fencing"
    STORES = "stores"
    VECTORS = "vectors"


A = FeatureAspect
CONFORMANCE_COVERAGE: Mapping[str, tuple[FeatureAspect, ...]] = {
    "plain_answer": (A.FINAL_ANSWER, A.EVENTS, A.USAGE_PRICING),
    "nested_structured": (A.STRUCTURED_OUTPUT,),
    "provider_auth_rejection": (A.FAILURE_RECOVERY,),
    "claude_large_schema": (A.STRUCTURED_OUTPUT,),
    "tool_evidence": (A.TOOLS, A.EVENTS, A.MCP),
    "tool_error_recovery": (A.TOOLS, A.FAILURE_RECOVERY, A.MCP),
    "image_result": (A.TOOLS, A.MCP),
    "overflow": (A.TOOLS, A.LIMITS, A.MCP),
    "session_resume": (A.SESSIONS_CONTEXT,),
    "durable_resume": (A.SESSIONS_CONTEXT, A.ISOLATION),
    "model_switch": (A.SESSIONS_CONTEXT,),
    "pinned_skill": (A.SKILLS,),
    "single_call_mind": (A.MINDS, A.STRUCTURED_OUTPUT),
    "chain_of_thought_mind": (A.MINDS, A.STRUCTURED_OUTPUT),
    "deferred_tool": (A.TOOLS, A.MCP),
    "discoverable_skill": (A.SKILLS, A.TOOLS),
    "triggered_skill": (A.SKILLS,),
    "delegation": (A.DELEGATION, A.EVENTS),
    "event_reaction": (A.EVENTS, A.HOOKS),
    "tool_context_switch": (A.TOOLS, A.SESSIONS_CONTEXT),
    "cancellation": (A.CANCELLATION,),
    "settings_refusal": (A.SETTINGS,),
}
WORKLOAD_COVERAGE: Mapping[str, tuple[FeatureAspect, ...]] = {
    "dispatch_checkpoint_replay": (
        A.CHECKPOINTS,
        A.JOURNALS,
        A.REPLAY,
        A.LEASES_FENCING,
        A.STORES,
        A.SESSIONS_CONTEXT,
        A.TOOLS,
    ),
    "dispatch_stale_fence_refusal": (A.JOURNALS, A.LEASES_FENCING, A.STORES, A.TOOLS),
    "vector_support_grounding": (A.VECTORS, A.MEMORY, A.STORES, A.TOOLS),
    "vector_missing_authority": (A.VECTORS, A.MEMORY, A.STORES, A.TOOLS),
    "memory_customer_handoff": (A.MEMORY, A.TOOLS, A.SESSIONS_CONTEXT, A.STRUCTURED_OUTPUT),
    "memory_readonly_refusal": (A.MEMORY, A.TOOLS, A.STRUCTURED_OUTPUT),
    "context_compaction_evidence": (A.CONTEXT_COMPACTION, A.SESSIONS_CONTEXT, A.STRUCTURED_OUTPUT),
    "context_compaction_insufficient_evidence": (A.CONTEXT_COMPACTION, A.SESSIONS_CONTEXT, A.STRUCTURED_OUTPUT),
    "prompt_cache_policy_reuse": (A.PROMPT_CACHING, A.STRUCTURED_OUTPUT, A.USAGE_PRICING),
    "prompt_cache_policy_revoked": (A.PROMPT_CACHING, A.STRUCTURED_OUTPUT),
    "coding_checkpoint_repair": (A.TOOLS, A.FAILURE_RECOVERY),
    "coding_checkpoint_write_denied": (A.TOOLS,),
    "coding_schema_compatibility_repair": (A.TOOLS, A.FAILURE_RECOVERY),
    "coding_structured_error_repair": (A.TOOLS, A.FAILURE_RECOVERY),
    "sqlite_migration_research": (A.FINAL_ANSWER, A.STRUCTURED_OUTPUT, A.TOOLS, A.SKILLS, A.SESSIONS_CONTEXT, A.MINDS),
    "sqlite_incident_triage": (
        A.FINAL_ANSWER,
        A.STRUCTURED_OUTPUT,
        A.TOOLS,
        A.EVENTS,
        A.HOOKS,
        A.SESSIONS_CONTEXT,
        A.MINDS,
    ),
    "bank_campaign_analytics": (A.FINAL_ANSWER, A.STRUCTURED_OUTPUT, A.TOOLS, A.MINDS),
    "sqlite_research_insufficient_evidence": (A.TOOLS, A.STRUCTURED_OUTPUT, A.MINDS),
    "sqlite_incident_insufficient_evidence": (A.TOOLS, A.STRUCTURED_OUTPUT, A.MINDS),
    "bank_dataset_insufficient_evidence": (A.TOOLS, A.STRUCTURED_OUTPUT, A.MINDS),
    "operations_refund_denied": (A.TOOLS, A.STRUCTURED_OUTPUT),
    "operations_incident_blocked": (A.TOOLS, A.STRUCTURED_OUTPUT),
    "operations_partner_no_fallback": (A.TOOLS, A.STRUCTURED_OUTPUT, A.MCP),
    "framework_diagnostic_success": (A.TOOLS, A.STRUCTURED_OUTPUT),
    "framework_cancelled_diagnostic_resume": (A.TOOLS, A.CANCELLATION, A.SESSIONS_CONTEXT, A.STRUCTURED_OUTPUT),
    "framework_turn_limited_diagnostic_resume": (A.TOOLS, A.LIMITS, A.SESSIONS_CONTEXT, A.STRUCTURED_OUTPUT),
    "framework_api_auth_rejected": (A.FAILURE_RECOVERY,),
    "operations_inventory_allocation": (A.TOOLS,),
    "operations_return_recovery": (A.TOOLS, A.FAILURE_RECOVERY),
    "operations_incident_rollback": (A.TOOLS, A.FAILURE_RECOVERY),
    "operations_partner_revocation": (A.TOOLS, A.MCP),
}

WORKLOAD_PAIRS: Mapping[str, str] = {
    "dispatch_stale_fence_refusal": "dispatch_checkpoint_replay",
    "vector_missing_authority": "vector_support_grounding",
    "memory_readonly_refusal": "memory_customer_handoff",
    "context_compaction_insufficient_evidence": "context_compaction_evidence",
    "prompt_cache_policy_revoked": "prompt_cache_policy_reuse",
    "coding_checkpoint_write_denied": "coding_checkpoint_repair",
    "sqlite_research_insufficient_evidence": "sqlite_migration_research",
    "sqlite_incident_insufficient_evidence": "sqlite_incident_triage",
    "bank_dataset_insufficient_evidence": "bank_campaign_analytics",
    "operations_refund_denied": "operations_return_recovery",
    "operations_incident_blocked": "operations_incident_rollback",
    "operations_partner_no_fallback": "operations_partner_revocation",
    "framework_cancelled_diagnostic_resume": "framework_diagnostic_success",
    "framework_turn_limited_diagnostic_resume": "framework_diagnostic_success",
    "framework_api_auth_rejected": "framework_diagnostic_success",
}

SUPPORTED_CASE_MODES: Mapping[str, tuple[Mode, ...]] = {
    identifier: (Mode.OPENAI_API, Mode.ANTHROPIC_API)
    for identifier in (
        "prompt_cache_policy_reuse",
        "prompt_cache_policy_revoked",
        "framework_diagnostic_success",
        "framework_cancelled_diagnostic_resume",
        "framework_turn_limited_diagnostic_resume",
        "framework_api_auth_rejected",
    )
}

# MCP is the hosted CLI transport in conformance tool cases. API modes execute
# those registries in-process. The partner workload exercises gateway policy
# with an injected client, not external HTTP/stdio protocol interoperability.
_CLI_ONLY = {"durable_resume": frozenset({A.ISOLATION})}
COVERAGE_SCOPE: Mapping[FeatureAspect, str] = {
    A.CHECKPOINTS: "Business checkpoints over FileStore-backed latest-slot storage, reconstructed by fresh adapters; external Postgres deployments remain separate.",
    A.JOURNALS: "Authored SQLite event persistence and cursor replay; not exactly-once subscribers or multi-host delivery.",
    A.REPLAY: "Actual FileReplayStore tool receipt replay verified against checkpoint and journal after restart.",
    A.LEASES_FENCING: "SQLite lease renewal rejection and application fencing at an approved business-write boundary; not distributed database concurrency certification.",
    A.STORES: "Durable FileStore business effects and CAS idempotence within its documented process scope; external database operations remain separate.",
    A.VECTORS: "InMemoryVector TF-cosine reference retrieval through VectorMemory with fixed tenant/metadata guards; not external embedding quality or pgvector deployment.",
    A.MEMORY: "Durable StoreFiles/FileMemory CRUD, restart and read-only/tenant boundaries; external vector backends need deployment checks.",
    A.CONTEXT_COMPACTION: "Sliding-window compaction with explicitly protected attributed tool evidence and model switching; summarizer quality is separately evaluated.",
    A.LIMITS: "Live conformance checks tool-result overflow; deadline/spend/turn boundary faults are offline tests.",
    A.HOOKS: "Live cases exercise event subscriptions; request/result-hook variants remain offline checks.",
    A.MCP: "CLI hosted tools or injected-client gateway policy; not every external server/auth/transport.",
    A.ISOLATION: "Fresh-runtime CLI session continuation; tenant and cleanup fault boundaries are offline tests.",
    A.USAGE_PRICING: "Observed accounting and runner budget checks; price arithmetic/faults are offline tests.",
    A.PROMPT_CACHING: "API-only handbook reviews require actual cache reads and changed-prefix policy refusal; routing misses fail acceptance.",
    A.DELEGATION: "Explicit bounded child invocation; not distributed/global concurrency or durable handoffs.",
    A.MINDS: "Single-call, ReAct and prompt-level deliberation; hidden model reasoning is not exposed or graded.",
}


def case_features(case_id: str, mode: Mode | str | None = None) -> tuple[FeatureAspect, ...]:
    """Return declared witnesses, retaining transport-specific limitations."""
    features = CONFORMANCE_COVERAGE.get(case_id, WORKLOAD_COVERAGE.get(case_id))
    if features is None:
        raise KeyError(f"unknown coverage case: {case_id}")
    if mode is not None and Mode(mode) not in SUPPORTED_CASE_MODES.get(case_id, tuple(Mode)):
        return ()
    if mode is None or not Mode(mode).is_api:
        return features
    excluded = _CLI_ONLY.get(case_id, frozenset())
    if case_id in CONFORMANCE_COVERAGE:
        excluded = excluded | {A.MCP}
    return tuple(feature for feature in features if feature not in excluded)


@dataclass(frozen=True)
class CoverageSummary:
    planned_contract: tuple[FeatureAspect, ...]
    planned_workload: tuple[FeatureAspect, ...]
    passed_contract: tuple[FeatureAspect, ...]
    passed_workload: tuple[FeatureAspect, ...]
    missing_live: tuple[FeatureAspect, ...]
    missing_workload: tuple[FeatureAspect, ...]
    passed_cases: tuple[str, ...]
    failed_cases: tuple[str, ...]
    not_run_cases: tuple[str, ...]
    unknown_cases: tuple[str, ...] = ()
    mode: str | None = None
    complete_pairs: tuple[str, ...] = ()
    missing_paired_cases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            name: [str(item) for item in getattr(self, name)]
            for name in (
                "planned_contract",
                "planned_workload",
                "passed_contract",
                "passed_workload",
                "missing_live",
                "missing_workload",
                "passed_cases",
                "failed_cases",
                "not_run_cases",
                "unknown_cases",
                "complete_pairs",
                "missing_paired_cases",
            )
        } | {"mode": self.mode, "scope": {str(feature): note for feature, note in COVERAGE_SCOPE.items()}}


def summarize_coverage(
    results: Sequence[Mapping[str, Any]] = (),
    *,
    mode: Mode | str | None = None,
) -> CoverageSummary:
    """Use actual passed results; absent, failed or skipped cases prove nothing.

    Records use ``case`` (or ``id``), ``mode`` and ``status``. Supply a mode to
    avoid mistaking success on one backend for evidence on all four backends.
    """
    contract: set[FeatureAspect] = set()
    workload: set[FeatureAspect] = set()
    passed: set[str] = set()
    failed: set[str] = set()
    unknown: set[str] = set()
    passed_by_mode: dict[str, set[str]] = {}
    for record in results:
        if mode is not None and record.get("mode") != str(mode):
            continue
        identifier = str(record.get("case", record.get("id", "")))
        if identifier not in CONFORMANCE_COVERAGE and identifier not in WORKLOAD_COVERAGE:
            unknown.add(identifier)
            continue
        if record.get("status") != "passed":
            if record.get("status") == "failed":
                failed.add(identifier)
            continue
        passed.add(identifier)
        passed_by_mode.setdefault(str(record.get("mode", "unspecified")), set()).add(identifier)
        witnesses = case_features(identifier, mode or record.get("mode"))
        (contract if identifier in CONFORMANCE_COVERAGE else workload).update(witnesses)

    def planned(mapping: Mapping[str, tuple[FeatureAspect, ...]]) -> tuple[FeatureAspect, ...]:
        return tuple(sorted({feature for identifier in mapping for feature in case_features(identifier, mode)}))

    return CoverageSummary(
        planned(CONFORMANCE_COVERAGE),
        planned(WORKLOAD_COVERAGE),
        tuple(sorted(contract)),
        tuple(sorted(workload)),
        tuple(sorted(set(FeatureAspect) - contract - workload)),
        tuple(sorted(set(FeatureAspect) - workload)),
        tuple(sorted(passed)),
        tuple(sorted(failed)),
        tuple(
            sorted(
                {
                    identifier
                    for identifier in set(CONFORMANCE_COVERAGE) | set(WORKLOAD_COVERAGE)
                    if case_features(identifier, mode)
                }
                - passed
                - failed
            )
        ),
        tuple(sorted(unknown)),
        str(mode) if mode is not None else None,
        tuple(
            sorted(
                f"{backend}:{positive}/{negative}"
                for backend, identifiers in passed_by_mode.items()
                for negative, positive in WORKLOAD_PAIRS.items()
                if {negative, positive} <= identifiers
            )
        ),
        tuple(sorted({identifier for identifier in WORKLOAD_PAIRS if case_features(identifier, mode)} - passed)),
    )


__all__ = [
    "FeatureAspect",
    "CoverageSummary",
    "case_features",
    "summarize_coverage",
    "CONFORMANCE_COVERAGE",
    "WORKLOAD_COVERAGE",
    "WORKLOAD_PAIRS",
    "COVERAGE_SCOPE",
    "SUPPORTED_CASE_MODES",
]
