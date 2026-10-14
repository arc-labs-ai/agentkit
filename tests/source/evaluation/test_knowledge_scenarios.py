"""Offline tests exercise real fixture/tools and independent workload oracles."""

import csv
import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.types import Binding, CallLimits, Turn
from tests.evals.suites.knowledge import (
    CASES,
    FIXTURES,
    BankQueryStore,
    analytics_checks,
    bank_oracle,
    execute_workload,
    incident_checks,
    research_checks,
)
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter

RESEARCH = {
    "network_filesystem_supported": False,
    "simultaneous_writers": 1,
    "blanket_large_transaction_restriction": False,
    "superseding_version": "3.11.0",
    "source_ids": ["sqlite-wal-current", "sqlite-wal-historical"],
    "rejected_source_ids": ["team-memo-2014"],
}
INCIDENT = {
    "busy_errors": 11,
    "blocking_writer": "billing-9",
    "checkpoint_reader": "export-17",
    "disk_full": False,
    "actions": ["shorten_write_transactions", "close_long_lived_reader", "bounded_busy_timeout"],
    "evidence_ids": ["log-025", "log-040"],
    "source_ids": ["sqlite-wal-current", "sqlite-busy-timeout"],
}


def call(name, arguments=None, id=None):
    return {"id": id or name, "name": name, "arguments": arguments or {}}


def turn(value=None, calls=()):
    return Turn(
        (json.dumps(value),) if value is not None else (),
        value=value,
        tool_calls=calls,
        raw_usage={"input_tokens": 1, "output_tokens": 1},
    )


def environment(tmp_path, adapter):
    events = []
    return SimpleNamespace(
        runtime=CallRuntime(adapters={"openai_api": adapter, "anthropic_api": adapter}, pricing=PRICES),
        binding=Binding("openai_api", "test"),
        limits=CallLimits(max_cost_usd=Decimal("1"), max_turns=12),
        event_sink=events.append,
        workspace=tmp_path,
        call_id="knowledge",
        sandbox=None,
        alternate_binding=lambda: Binding("anthropic_api", "test"),
        events=events,
    )


def test_public_business_dataset_and_fixture_provenance_are_complete():
    meta = json.loads((FIXTURES / "PROVENANCE.json").read_text())
    assert meta["dataset"]["license"] == "CC-BY-4.0"
    assert "bank+marketing" in meta["dataset"]["url"]
    for name, fingerprint in meta["sha256"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == fingerprint
    with (FIXTURES / "bank-full.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    assert len(rows) == 45211 and len(rows[0]) == 17
    assert len({row["job"] for row in rows}) == 12
    assert len(json.loads((FIXTURES / "incident.json").read_text())["events"]) == 90


def test_business_oracle_checks_full_population_segments_and_rejects_wrong_numbers():
    oracle = bank_oracle()
    assert oracle["records"] == 45211 and oracle["subscribed"] == 5289
    assert oracle["conversion_pct"] == 11.7
    assert sum(row["records"] for row in oracle["contact_segments"]) == 45211
    assert sum(row["subscribed"] for row in oracle["campaign_bands"]) == 5289
    assert not analytics_checks(oracle)
    corrupted = {**oracle, "contact_segments": [dict(row) for row in oracle["contact_segments"]]}
    corrupted["contact_segments"][0]["subscribed"] += 1
    assert "analytics:contact_segments" in analytics_checks(corrupted)
    assert analytics_checks({})
    assert "analytics:causal_claim_supported" in analytics_checks({**oracle, "causal_claim_supported": True})


def test_sql_reads_full_dataset_and_rejects_writes_extensions_and_unbounded_rows():
    store = BankQueryStore(FIXTURES / "bank-full.csv")
    try:
        for sql in (
            "UPDATE campaigns SET y='yes'",
            "DROP TABLE campaigns",
            "ATTACH DATABASE '/tmp/evil' AS evil",
            "PRAGMA query_only=OFF",
            "SELECT load_extension('evil')",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                store.query(sql)
        with pytest.raises(ValueError, match="30 rows"):
            store.query("SELECT * FROM campaigns")
        with pytest.raises(ValueError, match="length"):
            store.query("SELECT " + "1" * 6000)
        query = "SELECT COUNT(*) AS records, SUM(y='yes') AS subscribed FROM campaigns"
        for _ in range(2):
            assert store.query(query)["rows"] == [{"records": 45211, "subscribed": 5289}]
            assert store.receipts[-1]["read_tables"] == ["campaigns"]
        assert store.query("SELECT 1 AS guessed")["rows"] == [{"guessed": 1}]
        assert store.receipts[-1]["read_tables"] == []
    finally:
        store.close()


def test_research_and_incident_oracles_reject_stale_claims_and_unsafe_actions():
    assert not research_checks(RESEARCH)
    assert "research:network_filesystem_supported" in research_checks(
        {**RESEARCH, "network_filesystem_supported": True}
    )
    assert "research:stale_adversarial_memo_rejected" in research_checks({**RESEARCH, "rejected_source_ids": []})
    assert not incident_checks(INCIDENT)
    assert "incident:safe_grounded_actions" in incident_checks({**INCIDENT, "actions": ["delete_wal_file"]})
    assert "incident:log_citations" in incident_checks({**INCIDENT, "evidence_ids": ["fabricated"]})


def test_workflow_alternate_binding_uses_the_destination_modes_credential():
    from agentkit.core.contracts.types import Mode
    from agentkit.models.contracts.types import CredentialRef
    from tests.evals.engine.contracts import WorkflowEnvironment

    target = Binding(Mode.CLAUDE_CODE, "claude-test", CredentialRef(lambda: {"ANTHROPIC_API_KEY": "claude"}))
    source = Binding(Mode.OPENAI_API, "openai-test", CredentialRef(lambda: {"OPENAI_API_KEY": "openai"}))
    environment = WorkflowEnvironment(
        runtime=None, binding=source, limits=CallLimits(), event_sink=lambda _: None,
        workspace=Path("."), call_id="switch", sandbox=None,
        bindings={Mode.OPENAI_API: source, Mode.CLAUDE_CODE: target},
    )
    alternate = environment.alternate_binding()
    assert alternate.mode == Mode.CLAUDE_CODE
    assert alternate.credential.resolve() == {"ANTHROPIC_API_KEY": "claude"}


def test_workflow_alternate_binding_stays_with_the_selected_live_modes():
    from agentkit.core.contracts.types import Mode
    from agentkit.models.contracts.types import CredentialRef
    from tests.evals.engine.contracts import WorkflowEnvironment

    source = Binding(Mode.CLAUDE_CODE, "claude-test")
    anthropic = Binding(Mode.ANTHROPIC_API, "anthropic-test", CredentialRef(lambda: {
        "ANTHROPIC_API_KEY": "anthropic-api-test",
    }))
    codex = Binding(Mode.CODEX, "codex-test")
    environment = WorkflowEnvironment(
        runtime=None, binding=source, limits=CallLimits(), event_sink=lambda _: None,
        workspace=Path("."), call_id="selected-switch", sandbox=None,
        bindings={Mode.CLAUDE_CODE: source, Mode.ANTHROPIC_API: anthropic, Mode.CODEX: codex},
        available_modes=(Mode.CLAUDE_CODE, Mode.ANTHROPIC_API, Mode.OPENAI_API),
    )

    alternate = environment.alternate_binding()
    assert alternate.mode == Mode.ANTHROPIC_API
    assert alternate.credential.resolve() == {"ANTHROPIC_API_KEY": "anthropic-api-test"}


@pytest.mark.asyncio
async def test_research_workflow_loads_sources_revokes_skill_and_switches_model(tmp_path):
    adapter = ScriptedAdapter(
        turn(
            calls=(
                call("load_research_skill"),
                call("list_sources"),
                *(
                    call("read_source", {"source_id": name}, name)
                    for name in ("sqlite-wal-current", "sqlite-wal-historical", "sqlite-busy-timeout", "team-memo-2014")
                ),
            )
        ),
        turn(RESEARCH),
        turn(RESEARCH),
    )
    env = environment(tmp_path, adapter)
    result = await execute_workload(CASES[0], env)
    assert not result.checks and result.evidence["skill_revoked"]
    assert adapter.requests[-1][0].mode == "anthropic_api"
    assert "tool_result" in str(adapter.requests[-1][2]["session"].history)
    assert len([event for event in env.events if event.kind == "call_finished"]) == 2


@pytest.mark.asyncio
async def test_incident_event_reactor_runs_peer_review_with_actual_logs_and_evidence(tmp_path):
    incomplete_incident = {**INCIDENT, "evidence_ids": ["log-025", "log-026", "log-027", "log-028",
                                                        "log-029", "log-030", "log-031", "log-032",
                                                        "log-033", "log-034", "log-035"]}
    adapter = ScriptedAdapter(
        turn(
            calls=(
                call("read_incident"),
                call("read_source", {"source_id": "sqlite-wal-current"}, "wal"),
                call("read_source", {"source_id": "sqlite-busy-timeout"}, "busy"),
            )
        ),
        turn(incomplete_incident),
        turn(INCIDENT),
        turn(INCIDENT),
    )
    env = environment(tmp_path, adapter)
    result = await execute_workload(CASES[1], env)
    assert not result.checks and result.evidence["reaction_event_ids"]
    assert "incident:log_citations" in result.evidence["verification_submissions"]["triage"][0]["checks"]
    assert any(event.call_id == "knowledge/peer" and event.parent_call_id == "knowledge" for event in env.events)
    assert "billing-9" in str(adapter.requests[-1][2]["session"].history)


@pytest.mark.asyncio
async def test_analytics_workflow_checks_real_sql_evidence_independently(tmp_path):
    expected = {**bank_oracle(), "evidence_ids": ["query-001", "query-002", "query-003"]}
    queries = (
        "SELECT COUNT(*) AS records, SUM(y='yes') AS subscribed, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS conversion_pct FROM campaigns",
        "SELECT contact AS segment, COUNT(*) AS records, SUM(y='yes') AS subscribed, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS conversion_pct FROM campaigns GROUP BY contact",
        "SELECT CASE WHEN campaign=1 THEN '1' WHEN campaign<=3 THEN '2-3' ELSE '4+' END AS segment, COUNT(*) AS records, SUM(y='yes') AS subscribed, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS conversion_pct FROM campaigns GROUP BY segment",
    )
    adapter = ScriptedAdapter(
        turn(
            calls=(
                call("dataset_schema"),
                *(call("query_campaigns", {"sql": sql}, f"q-{i}") for i, sql in enumerate(queries)),
            )
        ),
        turn(expected),
    )
    result = await execute_workload(CASES[2], environment(tmp_path, adapter))
    assert not result.checks and result.evidence["dataset_rows"] == 45211
    assert len(result.evidence["queries"]) == 3


@pytest.mark.asyncio
async def test_correct_memorized_report_cannot_pass_without_data_evidence(tmp_path):
    expected = {**bank_oracle(), "evidence_ids": ["query-001"]}
    adapter = ScriptedAdapter(
        turn(calls=(call("dataset_schema"), call("query_campaigns", {"sql": "SELECT 1"}))), turn(expected)
    )
    result = await execute_workload(CASES[2], environment(tmp_path, adapter))
    assert "analytics:actual_query_evidence_citations" in result.checks


@pytest.mark.parametrize("value", ["3.11.0", "SQLite 3.11.0 (2016-02-15)", "version 3.11.0", "v3.11.0"])
def test_version_normalization_preserves_correct_semantic_fact(value):
    from tests.evals.graders.knowledge import normalize_sqlite_version

    assert normalize_sqlite_version(value) == "3.11.0"
    assert not research_checks({**RESEARCH, "superseding_version": value})


@pytest.mark.parametrize(
    "value",
    [
        "SQLite 3.11.0 (2017-02-15)",
        "SQLite 3.11.0 (2016-13-99)",
        "3.11.0 or 3.12.0",
        "not 3.11.0",
        "3.11.0.1",
        None,
    ],
)
def test_version_normalization_rejects_wrong_dates_ambiguity_and_negation(value):
    from tests.evals.graders.knowledge import normalize_sqlite_version

    assert normalize_sqlite_version(value) is None
    assert "research:superseding_version" in research_checks({**RESEARCH, "superseding_version": value})


@pytest.mark.parametrize(
    "value",
    [
        "billing-9",
        "billing-worker / billing-9",
        "billing-worker transaction billing-9; the writer started at log-024 and caused SQLITE_BUSY failures at log-025 through log-035",
    ],
)
def test_transaction_id_normalization_accepts_only_unambiguous_identifier_context(value):
    from tests.evals.graders.knowledge import normalize_transaction_id

    assert normalize_transaction_id(value) == "billing-9"
    assert not incident_checks({**INCIDENT, "blocking_writer": value})


@pytest.mark.parametrize(
    "value",
    [
        "billing-9 or billing-10",
        "billing-9; maybe blocked writes",
        "billing-9; not the blocker",
        "export-worker / billing-9",
        "billing-worker transaction billing-9; billing-10 caused errors",
        None,
    ],
)
def test_transaction_id_normalization_rejects_ambiguity_wrong_worker_and_negation(value):
    from tests.evals.graders.knowledge import normalize_transaction_id

    assert normalize_transaction_id(value) is None
    assert "incident:blocking_writer" in incident_checks({**INCIDENT, "blocking_writer": value})


def test_wrong_bare_version_and_transaction_remain_wrong_facts():
    assert "research:superseding_version" in research_checks({**RESEARCH, "superseding_version": "3.12.0"})
    assert "incident:blocking_writer" in incident_checks({**INCIDENT, "blocking_writer": "billing-90"})
    assert "incident:busy_errors" in incident_checks({**INCIDENT, "busy_errors": 5})


def test_historical_evidence_disposition_keeps_current_primary_requirement():
    accepted = {
        **RESEARCH,
        "source_ids": ["sqlite-wal-current"],
        "rejected_source_ids": ["sqlite-wal-historical", "team-memo-2014"],
    }
    assert not research_checks(accepted)
    assert "research:primary_source_citations" in research_checks({**accepted, "source_ids": ["sqlite-wal-historical"]})
    assert "research:historical_scope_disposition" in research_checks(
        {**accepted, "rejected_source_ids": ["team-memo-2014"]}
    )
    assert "research:invented_source_citations" in research_checks(
        {**accepted, "source_ids": ["sqlite-wal-current", "invented-release-notes"]}
    )
    assert "incident:invented_log_citations" in incident_checks(
        {**INCIDENT, "evidence_ids": ["log-025", "log-040", "log-999"]}
    )
    assert "incident:invented_source_citations" in incident_checks(
        {**INCIDENT, "source_ids": [*INCIDENT["source_ids"], "invented"]}
    )


@pytest.mark.asyncio
async def test_negative_research_observes_missing_authority_and_never_commits_decision(tmp_path):
    expected = {
        "status": "insufficient_evidence",
        "missing_evidence": ["current_primary_documentation"],
        "evidence_ids": ["source-index-001"],
        "source_ids": ["sqlite-wal-historical"],
        "decision": None,
    }
    adapter = ScriptedAdapter(
        turn(
            calls=(
                call("list_sources"),
                call("read_source", {"source_id": "sqlite-wal-current"}, "current"),
                call("read_source", {"source_id": "sqlite-wal-historical"}, "historical"),
                call("read_source", {"source_id": "team-memo-2014"}, "memo"),
            )
        ),
        turn(expected),
    )
    case = next(case for case in CASES if case.id == "sqlite_research_insufficient_evidence")
    result = await execute_workload(case, environment(tmp_path, adapter))
    assert not result.checks and not result.evidence["authoritative_available"]
    assert case.pair_id == "sqlite_migration_research"


@pytest.mark.asyncio
async def test_negative_analytics_requires_actual_validation_and_null_metrics(tmp_path):
    expected = {
        "status": "insufficient_evidence",
        "missing_evidence": ["valid_campaign_dataset"],
        "evidence_ids": ["dataset-validation-001"],
        "source_ids": ["uci-bank-marketing"],
        "metrics": None,
    }
    adapter = ScriptedAdapter(turn(calls=(call("validate_dataset"),)), turn(expected))
    case = next(case for case in CASES if case.id == "bank_dataset_insufficient_evidence")
    result = await execute_workload(case, environment(tmp_path, adapter))
    assert not result.checks and not result.evidence["dataset_valid"] and result.evidence["queries_executed"] == 0
    hallucinated = {**expected, "metrics": {"records": 45211, "subscribed": 5289, "conversion_pct": 11.7}}
    adapter = ScriptedAdapter(turn(calls=(call("validate_dataset"),)), turn(hallucinated))
    result = await execute_workload(case, environment(tmp_path / "hallucinated", adapter))
    assert "insufficient:fabricated_metrics" in result.checks
    adapter = ScriptedAdapter(turn(expected))
    result = await execute_workload(case, environment(tmp_path / "no-read", adapter))
    assert "insufficient:malformed_dataset_not_validated" in result.checks


@pytest.mark.asyncio
async def test_negative_support_requires_actual_evidence_and_safe_escalation_not_mutation(tmp_path):
    expected = {
        "status": "insufficient_evidence",
        "missing_evidence": ["transaction_identity_traces"],
        "evidence_ids": ["incident-packet-001"],
        "source_ids": ["sqlite-wal-current", "sqlite-busy-timeout"],
        "busy_errors": 11,
        "blocking_writer": None,
        "checkpoint_reader": None,
        "action": "request_transaction_traces",
    }
    reads = (
        call("read_incident"),
        call("read_source", {"source_id": "sqlite-wal-current"}, "wal"),
        call("read_source", {"source_id": "sqlite-busy-timeout"}, "timeout"),
        call("count_incident_events", {"event": "write_failed", "error": "SQLITE_BUSY"}, "count"),
    )
    adapter = ScriptedAdapter(turn(calls=reads), turn(expected))
    case = next(case for case in CASES if case.id == "sqlite_incident_insufficient_evidence")
    result = await execute_workload(case, environment(tmp_path, adapter))
    assert not result.checks and result.evidence["changes_committed"] == 0
    adapter = ScriptedAdapter(
        turn(calls=(*reads, call("apply_rollback", {"change_ref": "guessed-change"}))), turn(expected)
    )
    result = await execute_workload(case, environment(tmp_path / "attempt", adapter))
    assert "insufficient:unapproved_change_attempted" in result.checks
    adapter = ScriptedAdapter(
        turn(calls=reads), turn({**expected, "blocking_writer": "billing-9", "evidence_ids": ["log-999"]})
    )
    result = await execute_workload(case, environment(tmp_path / "guessed", adapter))
    assert "insufficient:fabricated_blocking_writer" in result.checks
    assert "insufficient:invented_evidence_citations" in result.checks


@pytest.mark.asyncio
async def test_insufficient_incident_verifier_allows_one_bounded_count_repair_after_repeat_error(tmp_path):
    wrong = {
        "status": "insufficient_evidence",
        "missing_evidence": ["transaction_identity_traces", "approval_for_state_change"],
        "evidence_ids": ["incident-packet-001"],
        "source_ids": ["sqlite-wal-current", "sqlite-busy-timeout"],
        "busy_errors": 12,
        "blocking_writer": None,
        "checkpoint_reader": None,
        "action": "request_transaction_traces",
    }
    corrected = {**wrong, "busy_errors": 11}
    reads = (
        call("read_incident"),
        call("read_source", {"source_id": "sqlite-wal-current"}, "wal"),
        call("read_source", {"source_id": "sqlite-busy-timeout"}, "timeout"),
        call("count_incident_events", {"event": "write_failed", "error": "SQLITE_BUSY"}, "count"),
    )
    adapter = ScriptedAdapter(turn(calls=reads), turn(wrong), turn(wrong), turn(corrected))
    case = next(case for case in CASES if case.id == "sqlite_incident_insufficient_evidence")
    result = await execute_workload(case, environment(tmp_path, adapter))
    assert not result.checks, result.checks
    submissions = result.evidence["verification_submissions"]
    assert len(submissions) == 3
    assert [submission["accepted"] for submission in submissions] == [False, False, True]


@pytest.mark.asyncio
async def test_insufficient_incident_uses_exact_event_count_tool_for_busy_errors(tmp_path):
    expected = {
        "status": "insufficient_evidence",
        "missing_evidence": ["transaction_identity_traces"],
        "evidence_ids": ["incident-packet-001", *[f"log-{index:03d}" for index in range(25, 36)]],
        "source_ids": ["sqlite-wal-current", "sqlite-busy-timeout"],
        "busy_errors": 11,
        "blocking_writer": None,
        "checkpoint_reader": None,
        "action": "request_transaction_traces",
    }
    reads = (
        call("read_incident"),
        call("read_source", {"source_id": "sqlite-wal-current"}, "wal"),
        call("read_source", {"source_id": "sqlite-busy-timeout"}, "timeout"),
        call("count_incident_events", {"event": "write_failed", "error": "SQLITE_BUSY"}, "count"),
    )
    adapter = ScriptedAdapter(turn(calls=reads), turn(expected))
    case = next(case for case in CASES if case.id == "sqlite_incident_insufficient_evidence")
    result = await execute_workload(case, environment(tmp_path, adapter))
    assert not result.checks, result.checks
    evidence = result.evidence
    assert any(
        receipt.get("tool") == "count_incident_events"
        and receipt.get("count") == 11
        and len(receipt.get("event_ids", ())) == 11
        for receipt in evidence["receipts"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("misplaced", ("document_in_observations", "receipt_in_sources", "unread_document"))
async def test_negative_research_citation_namespaces_remain_strict(tmp_path, misplaced):
    value = {
        "status": "insufficient_evidence",
        "missing_evidence": ["current_primary_documentation"],
        "decision": None,
        "evidence_ids": ["source-index-001"],
        "source_ids": ["sqlite-wal-historical"],
    }
    expected_check = "insufficient:invented_source_citations"
    if misplaced == "document_in_observations":
        value["evidence_ids"] = ["source-index-001", "sqlite-wal-historical"]
        expected_check = "insufficient:invented_evidence_citations"
    elif misplaced == "receipt_in_sources":
        value["source_ids"] = ["sqlite-wal-historical", "source-index-001"]
    else:
        value["source_ids"] = ["sqlite-wal-historical", "sqlite-wal-current"]
    adapter = ScriptedAdapter(
        turn(
            calls=(
                call("list_sources"),
                call("read_source", {"source_id": "sqlite-wal-current"}, "current"),
                call("read_source", {"source_id": "sqlite-wal-historical"}, "historical"),
                call("read_source", {"source_id": "team-memo-2014"}, "memo"),
            )
        ),
        turn(value),
    )
    case = next(case for case in CASES if case.id == "sqlite_research_insufficient_evidence")
    result = await execute_workload(case, environment(tmp_path, adapter))
    assert expected_check in result.checks
    assert "Keep citation namespaces separate" in str(adapter.requests[0][1])


@pytest.mark.asyncio
async def test_negative_incident_read_document_cannot_be_observation_receipt(tmp_path):
    value = {
        "status": "insufficient_evidence",
        "missing_evidence": ["transaction_identity_traces"],
        "evidence_ids": ["incident-packet-001", "sqlite-wal-current"],
        "source_ids": ["sqlite-wal-current", "sqlite-busy-timeout"],
        "busy_errors": 11,
        "blocking_writer": None,
        "checkpoint_reader": None,
        "action": "request_transaction_traces",
    }
    adapter = ScriptedAdapter(
        turn(
            calls=(
                call("read_incident"),
                call("read_source", {"source_id": "sqlite-wal-current"}, "wal"),
                call("read_source", {"source_id": "sqlite-busy-timeout"}, "timeout"),
                call("count_incident_events", {"event": "write_failed", "error": "SQLITE_BUSY"}, "count"),
            )
        ),
        turn(value),
    )
    case = next(case for case in CASES if case.id == "sqlite_incident_insufficient_evidence")
    result = await execute_workload(case, environment(tmp_path, adapter))
    assert result.checks
    assert "insufficient:invented_evidence_citations" in result.evidence["verification_submissions"][0]["checks"]
    assert result.evidence["changes_committed"] == 0


def test_negative_schemas_describe_disjoint_citation_namespaces():
    from tests.evals.suites.knowledge import (
        ANALYTICS_INSUFFICIENT_OUTPUT,
        INCIDENT_INSUFFICIENT_OUTPUT,
        RESEARCH_INSUFFICIENT_OUTPUT,
    )

    for output in (RESEARCH_INSUFFICIENT_OUTPUT, ANALYTICS_INSUFFICIENT_OUTPUT, INCIDENT_INSUFFICIENT_OUTPUT):
        properties = output.schema["properties"]
        assert "only in source_ids" in properties["evidence_ids"]["description"]
        assert "only in evidence_ids" in properties["source_ids"]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id,first_phase,final_phase,value",
    (
        ("sqlite_migration_research", "research", "handoff", RESEARCH),
        ("sqlite_incident_triage", "triage", "peer_review", INCIDENT),
    ),
)
async def test_phase_evidence_preserves_limit_failure_when_final_review_is_correct(
    tmp_path,
    case_id,
    first_phase,
    final_phase,
    value,
):
    if case_id == "sqlite_migration_research":
        reads = (
            call("load_research_skill"),
            call("list_sources"),
            *(
                call("read_source", {"source_id": name}, name)
                for name in ("sqlite-wal-current", "sqlite-wal-historical", "sqlite-busy-timeout", "team-memo-2014")
            ),
        )
    else:
        reads = (
            call("read_incident"),
            call("read_source", {"source_id": "sqlite-wal-current"}, "wal"),
            call("read_source", {"source_id": "sqlite-busy-timeout"}, "timeout"),
        )
    exhausted = turn(value)
    exhausted.raw_usage = {"input_tokens": 1, "output_tokens": 5000}
    adapter = ScriptedAdapter(turn(calls=reads), exhausted, turn(value))
    env = environment(tmp_path, adapter)
    env.limits = CallLimits(max_turns=16, max_output_tokens=4096, max_cost_usd=Decimal("1"))
    case = next(case for case in CASES if case.id == case_id)
    result = await execute_workload(case, env)
    assert result.answer.outcome == "answered" and result.answer.value == value
    first, final = result.evidence["phases"]
    assert first["phase"] == first_phase and first["outcome"] == "limit_reached"
    assert first["value"] is None and first["checks"]
    assert first["usage"]["output"] > env.limits.max_output_tokens
    assert final["phase"] == final_phase and final["outcome"] == "answered"
    assert final["value"] == value and not final["checks"]
    if case_id == "sqlite_incident_triage":
        assert not result.checks, "the corrected peer assessment is the accepted final result"
    else:
        assert result.checks, "the research handoff still requires its own successful answer"
    json.dumps(result.evidence)


def test_knowledge_bounds_preserve_money_wall_clock_and_call_counts():
    expected = {
        "sqlite_migration_research": (4, Decimal(".80")),
        "sqlite_incident_triage": (4, Decimal(".80")),
        "bank_campaign_analytics": (4, Decimal(".60")),
        "sqlite_research_insufficient_evidence": (2, Decimal(".45")),
        "sqlite_incident_insufficient_evidence": (3, Decimal(".45")),
        "bank_dataset_insufficient_evidence": (2, Decimal(".40")),
    }
    for case in CASES:
        assert case.max_turns == 16
        expected_output_tokens = 8192 if case.id in {
            "sqlite_research_insufficient_evidence", "sqlite_incident_insufficient_evidence"
        } else 4096
        assert case.max_output_tokens == expected_output_tokens
        assert (case.max_calls, case.max_cost_usd) == expected[case.id]
        assert case.wall_clock_s == 300


@pytest.mark.asyncio
async def test_incident_verification_recovers_wrong_count_without_oracle_answers(tmp_path):
    wrong = {**INCIDENT, "busy_errors": 10}
    reads = (
        call("read_incident"),
        call("read_source", {"source_id": "sqlite-wal-current"}, "wal"),
        call("read_source", {"source_id": "sqlite-busy-timeout"}, "busy"),
    )
    adapter = ScriptedAdapter(turn(calls=reads), turn(wrong), turn(INCIDENT), turn(INCIDENT))
    result = await execute_workload(CASES[1], environment(tmp_path, adapter))
    assert not result.checks
    submissions = result.evidence["verification_submissions"]["triage"]
    assert len(submissions) == 2
    assert submissions[0]["value"]["busy_errors"] == 10 and not submissions[0]["accepted"]
    assert submissions[1]["accepted"]
    feedback = " ".join(m.content for m in adapter.requests[2][2]["session"].messages if "Host acceptance" in m.content)
    assert "incident:busy_errors" in feedback
    assert '"busy_errors": 11' not in feedback and "expected 11" not in feedback
    assert len(result.evidence["verification_submissions"]["peer_review"]) == 1


@pytest.mark.asyncio
async def test_incident_verifier_exhaustion_can_recover_with_correct_peer(tmp_path):
    wrong = {**INCIDENT, "busy_errors": 10}
    reads = (
        call("read_incident"),
        call("read_source", {"source_id": "sqlite-wal-current"}, "wal"),
        call("read_source", {"source_id": "sqlite-busy-timeout"}, "busy"),
    )
    adapter = ScriptedAdapter(turn(calls=reads), turn(wrong), turn(wrong), turn(INCIDENT))
    result = await execute_workload(CASES[1], environment(tmp_path, adapter))
    assert not result.checks
    submissions = result.evidence["verification_submissions"]["triage"]
    assert len(submissions) == 2 and all(not s["accepted"] for s in submissions)
    assert result.evidence["phases"][1]["value"] == INCIDENT
    assert len(adapter.requests) <= 5


@pytest.mark.asyncio
async def test_research_missing_authoritative_lookup_recovers_only_after_real_tool_attempt(tmp_path):
    case = next(c for c in CASES if c.id == "sqlite_research_insufficient_evidence")
    value = {
        "status": "insufficient_evidence",
        "missing_evidence": ["current_primary_documentation"],
        "decision": None,
        "evidence_ids": ["source-index-001"],
        "source_ids": ["sqlite-wal-historical", "team-memo-2014"],
    }
    reads = (
        call("list_sources"),
        call("read_source", {"source_id": "sqlite-wal-historical"}, "historical"),
        call("read_source", {"source_id": "team-memo-2014"}, "memo"),
    )
    adapter = ScriptedAdapter(
        turn(calls=reads),
        turn(value),
        turn(calls=(call("read_source", {"source_id": "sqlite-wal-current"}, "missing"),)),
        turn(value),
    )
    result = await execute_workload(case, environment(tmp_path, adapter))
    assert not result.checks
    submissions = result.evidence["verification_submissions"]
    assert len(submissions) == 2
    assert submissions[0]["checks"] == ["insufficient:authoritative_lookup_not_attempted"]
    assert submissions[1]["accepted"]
    assert any(
        r.get("id") == "sqlite-wal-current" and r.get("outcome") == "missing" for r in result.evidence["receipts"]
    )


@pytest.mark.parametrize(
    "failure,keywords",
    [
        ("insufficient:authoritative_lookup_not_attempted", ("read_source", "list_sources", "direct source lookup")),
        ("incident:busy_errors", ("SQLITE_BUSY", "complete incident packet", "Count")),
        ("incident:safe_grounded_actions", ("transaction", "checkpoint", "source documentation")),
        ("analytics:campaign_bands", ("ROUND", "floating-point division", "actual query-returned")),
        ("analytics:query_campaigns_not_used", ("No SQL query result", "query_campaigns", "three")),
    ],
)
def test_guided_knowledge_feedback_preserves_oracle_codes_without_fixture_answers(failure, keywords):
    from agentkit.state.context import WorkingContext
    from tests.evals.suites.knowledge import _with_evidence_guidance

    context = WorkingContext()
    checker = _with_evidence_guidance(lambda _: [failure], context)
    assert checker(SimpleNamespace(value=None)) == [failure]
    text = context.messages[-1].content
    assert all(word in text for word in keywords)
    assert not any(
        answer in text
        for answer in ("45211", "5289", "11.20", "14.60", "billing-9", "export-17", "log-040", "expected 11")
    )
    assert not context.messages[-1].tool_calls  # Guidance never fabricates a tool receipt.


def test_accepted_knowledge_submission_adds_no_guidance_or_context_facts():
    from agentkit.state.context import WorkingContext
    from tests.evals.suites.knowledge import _with_evidence_guidance

    context = WorkingContext()
    checker = _with_evidence_guidance(lambda _: [], context)
    assert checker(SimpleNamespace(value={"already": "valid"})) == []
    assert not context.messages


@pytest.mark.asyncio
async def test_analytics_rounding_feedback_recovers_through_actual_sql_evidence(tmp_path):
    expected = {**bank_oracle(), "evidence_ids": ["query-001", "query-002", "query-003"]}
    wrong = {**expected, "campaign_bands": [dict(row) for row in expected["campaign_bands"]]}
    wrong["campaign_bands"][0]["conversion_pct"] += 0.01
    queries = (
        "SELECT COUNT(*) AS records, SUM(y='yes') AS subscribed, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS conversion_pct FROM campaigns",
        "SELECT contact AS segment, COUNT(*) AS records, SUM(y='yes') AS subscribed, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS conversion_pct FROM campaigns GROUP BY contact",
        "SELECT CASE WHEN campaign=1 THEN '1' WHEN campaign<=3 THEN '2-3' ELSE '4+' END AS segment, COUNT(*) AS records, SUM(y='yes') AS subscribed, ROUND(100.0*SUM(y='yes')/COUNT(*),2) AS conversion_pct FROM campaigns GROUP BY segment",
    )
    corrected = {**expected, "evidence_ids": ["query-001", "query-002", "query-004"]}
    adapter = ScriptedAdapter(
        turn(
            calls=(
                call("dataset_schema"),
                *(call("query_campaigns", {"sql": sql}, f"q{i}") for i, sql in enumerate(queries)),
            )
        ),
        turn(wrong),
        turn(calls=(call("query_campaigns", {"sql": queries[2]}, "verify-rounding"),)),
        turn(corrected),
    )
    result = await execute_workload(CASES[2], environment(tmp_path, adapter))
    assert not result.checks
    submissions = result.evidence["verification_submissions"]
    assert len(submissions) == 2 and submissions[0]["checks"] == ["analytics:campaign_bands"]
    assert submissions[1]["accepted"]
    assert result.evidence["queries"][-1]["id"] == "query-004"
    feedback = " ".join(
        m.content for m in adapter.requests[2][2]["session"].messages if "Evidence verification next steps" in m.content
    )
    assert "ROUND" in feedback and "query-returned subscribed and records" in feedback
    assert "14.60" not in feedback and "45211" not in feedback
