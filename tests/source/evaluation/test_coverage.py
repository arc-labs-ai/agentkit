"""Coverage declarations cannot silently omit cases or turn plans into passes."""

import pytest

from agentkit.core.contracts.types import Mode
from tests.evals.catalog import CASES, WORKLOAD_CASES
from tests.evals.coverage import (
    CONFORMANCE_COVERAGE,
    WORKLOAD_COVERAGE,
    FeatureAspect,
    case_features,
    summarize_coverage,
)


def test_every_catalogue_case_has_one_known_feature_declaration():
    identifiers = [case.id for case in CASES]
    assert len(identifiers) == len(set(identifiers))
    assert not set(CONFORMANCE_COVERAGE) & set(WORKLOAD_COVERAGE)
    assert set(identifiers) == set(CONFORMANCE_COVERAGE) | set(WORKLOAD_COVERAGE)
    assert {case.id for case in WORKLOAD_CASES} == set(WORKLOAD_COVERAGE)
    for identifier in identifiers:
        witnesses = case_features(identifier)
        assert witnesses and len(witnesses) == len(set(witnesses))
        assert all(isinstance(witness, FeatureAspect) for witness in witnesses)
    with pytest.raises(KeyError):
        case_features("invented_case")


def test_planned_coverage_is_distinct_from_actual_success():
    summary = summarize_coverage()
    assert summary.planned_contract and summary.planned_workload
    assert not summary.passed_contract and not summary.passed_workload
    assert set(summary.missing_live) == set(FeatureAspect)
    assert len(summary.not_run_cases) == len(CASES)
    assert FeatureAspect.PROMPT_CACHING not in summary.planned_contract
    assert FeatureAspect.PROMPT_CACHING in summary.planned_workload


def test_failed_skipped_and_unknown_results_never_prove_features():
    summary = summarize_coverage(
        [
            {"case": "plain_answer", "mode": "openai_api", "status": "passed"},
            {"case": "settings_refusal", "mode": "openai_api", "status": "failed"},
            {"case": "delegation", "mode": "openai_api", "status": "not_run"},
            {"case": "invented_case", "mode": "openai_api", "status": "passed"},
        ]
    )
    assert FeatureAspect.FINAL_ANSWER in summary.passed_contract
    assert FeatureAspect.SETTINGS in summary.missing_live
    assert FeatureAspect.DELEGATION in summary.missing_live
    assert summary.failed_cases == ("settings_refusal",)
    assert "delegation" in summary.not_run_cases and summary.unknown_cases == ("invented_case",)
    assert not summary.passed_workload


def test_transport_specific_witnesses_do_not_claim_api_cases_use_mcp_or_cli_homes():
    assert FeatureAspect.MCP in case_features("tool_evidence", Mode.CODEX)
    assert FeatureAspect.MCP not in case_features("tool_evidence", Mode.OPENAI_API)
    assert FeatureAspect.ISOLATION in case_features("durable_resume", Mode.CLAUDE_CODE)
    assert FeatureAspect.ISOLATION not in case_features("durable_resume", Mode.ANTHROPIC_API)
    assert FeatureAspect.MCP in case_features("operations_partner_revocation", Mode.OPENAI_API)


def test_per_mode_summary_does_not_treat_one_backend_as_proof_for_others():
    results = [
        {"case": "sqlite_migration_research", "mode": "anthropic_api", "status": "passed"},
        {"case": "tool_evidence", "mode": "codex", "status": "passed"},
    ]
    anthropic = summarize_coverage(results, mode=Mode.ANTHROPIC_API)
    codex = summarize_coverage(results, mode=Mode.CODEX)
    assert anthropic.passed_cases == ("sqlite_migration_research",)
    assert FeatureAspect.MCP in anthropic.missing_live
    assert FeatureAspect.MCP in codex.passed_contract and FeatureAspect.SKILLS in codex.missing_live
    assert anthropic.as_dict()["mode"] == "anthropic_api"
    assert "prompt_caching" in anthropic.as_dict()["missing_live"]
    assert "scope" in anthropic.as_dict()


def test_every_negative_pair_is_declared_and_points_to_a_matching_domain():
    from tests.evals.coverage import WORKLOAD_PAIRS

    declared = {case.id: case for case in WORKLOAD_CASES}
    assert {case.id: case.pair_id for case in WORKLOAD_CASES if case.pair_id is not None} == WORKLOAD_PAIRS
    for negative, positive in WORKLOAD_PAIRS.items():
        assert negative in declared and positive in declared
        if declared[negative].expectation != "auth_failure":
            assert declared[negative].domain == declared[positive].domain
        assert declared[negative].expectation in {
            "safe_refusal",
            "insufficient_evidence",
            "cancelled",
            "limit_reached",
            "auth_failure",
        }
        assert declared[positive].expectation in {"success", "recovery"}


def test_a_pair_requires_actual_passing_positive_and_negative_on_same_backend():
    positive = "bank_campaign_analytics"
    negative = "bank_dataset_insufficient_evidence"
    mixed = summarize_coverage(
        [
            {"case": positive, "mode": "openai_api", "status": "passed"},
            {"case": negative, "mode": "anthropic_api", "status": "passed"},
        ]
    )
    assert not mixed.complete_pairs
    complete = summarize_coverage(
        [
            {"case": positive, "mode": "openai_api", "status": "passed"},
            {"case": negative, "mode": "openai_api", "status": "passed"},
        ]
    )
    assert complete.complete_pairs == (f"openai_api:{positive}/{negative}",)
    assert negative not in complete.missing_paired_cases
    assert complete.as_dict()["complete_pairs"] == list(complete.complete_pairs)


def test_supported_mode_declarations_match_catalogue_and_do_not_plan_cli_api_lifecycle_checks():
    from tests.evals.coverage import SUPPORTED_CASE_MODES

    for case in WORKLOAD_CASES:
        assert case.supported_modes == SUPPORTED_CASE_MODES.get(case.id, tuple(Mode))
    assert not case_features("framework_api_auth_rejected", Mode.CODEX)
    assert FeatureAspect.CANCELLATION in case_features("framework_cancelled_diagnostic_resume", Mode.OPENAI_API)
    cli = summarize_coverage(mode=Mode.CLAUDE_CODE)
    assert "framework_api_auth_rejected" not in cli.not_run_cases
    assert "framework_cancelled_diagnostic_resume" not in cli.missing_paired_cases
