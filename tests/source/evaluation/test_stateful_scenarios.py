import json
from types import SimpleNamespace

import pytest

from agentkit.core.contracts.types import Message, MessageRole, ToolCall
from agentkit.core.testing import make_test_ctx
from tests.evals.suites.stateful import (
    _FIXTURES,
    CASES,
    MemoryWorkflow,
    execute_workload,
    grade_compaction,
    grade_memory,
)


def accounts():
    return json.loads((_FIXTURES / "accounts.json").read_text())


def incident():
    return json.loads((_FIXTURES / "incident.json").read_text())


async def call(tools, name, **args):
    return await tools.execute(name, args, make_test_ctx())


@pytest.mark.asyncio
async def test_memory_oracle_requires_real_crud_restart_and_recall(tmp_path):
    fixture = accounts()
    workflow = MemoryWorkflow(tmp_path, fixture)
    await workflow.seed()
    before = await workflow.snapshot()
    value = {"case_id": fixture["case_id"], **fixture["contact_update"], "approval_ref": fixture["approval_ref"]}
    tools = workflow.tools()
    assert not (await call(tools, "recall_customer_memory", query="CASE-582")).is_error
    await call(tools, "verify_other_tenant_denied")
    await call(tools, "write_handover_brief", brief_json=json.dumps(value), approval_ref="APR-582")
    await call(
        tools, "update_routing_note", current_route="Bristol East | MedRoute | 08:00-10:00", approval_ref="APR-582"
    )
    await call(tools, "delete_expired_memory", path="/memories/expired-draft.md", approval_ref="APR-582")
    await call(tools, "restart_customer_memory")
    recalled = await call(tools, "recall_customer_memory", query="CASE-582")
    assert "/memories/handover.md" in recalled.text
    after = await workflow.snapshot()
    assert grade_memory(fixture, before, after, workflow.receipts, value, readonly=False, other_unchanged=True) == []
    assert grade_memory(fixture, before, before, [], value, readonly=False, other_unchanged=True)
    assert grade_memory(
        fixture,
        before,
        after,
        workflow.receipts,
        {**value, "carrier": "ParcelFast"},
        readonly=False,
        other_unchanged=True,
    )


@pytest.mark.asyncio
async def test_readonly_and_tenant_guard_refuse_before_durable_write(tmp_path):
    fixture = accounts()
    workflow = MemoryWorkflow(tmp_path, fixture, readonly=True)
    await workflow.seed()
    before = await workflow.snapshot()
    tools = workflow.tools()
    refused = await call(tools, "write_handover_brief", brief_json='{"carrier":"unsafe"}', approval_ref="APR-582")
    assert refused.is_error and "MemoryWriteRefused" in refused.text
    denied = await call(tools, "verify_other_tenant_denied")
    assert "denied" in denied.text
    receipt = await call(tools, "escalate_memory_review")
    value = json.loads(receipt.text)
    assert await workflow.store.get("support-ticket:harbor-clinics:CASE-582") == value
    assert (
        grade_memory(
            fixture, before, await workflow.snapshot(), workflow.receipts, value, readonly=True, other_unchanged=True
        )
        == []
    )
    assert grade_memory(fixture, before, before, [], value, readonly=True, other_unchanged=True)
    assert grade_memory(fixture, before, before, workflow.receipts, value, readonly=True, other_unchanged=False)


@pytest.mark.asyncio
async def test_memory_policy_rejects_protected_delete_even_if_model_later_completes(tmp_path):
    fixture = accounts()
    workflow = MemoryWorkflow(tmp_path, fixture)
    await workflow.seed()
    tools = workflow.tools()
    before = await workflow.snapshot()
    refused = await call(tools, "delete_expired_memory", path="/memories/finance.md", approval_ref="APR-582")
    assert refused.is_error
    assert await workflow.snapshot() == before
    assert "memory crossed a tenant or protected mutation boundary" in grade_memory(
        fixture, before, before, workflow.receipts, None, readonly=False, other_unchanged=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_compaction_execution_preserves_old_tool_evidence_and_owner_in_bound_context(tmp_path, missing):
    fixture = incident()
    captured = []

    class Runtime:
        async def run(self, binding, prompt, tools, **kwargs):
            result = await call(tools, "read_signed_incident_review")
            messages = (
                Message(MessageRole.USER, prompt),
                Message(
                    MessageRole.ASSISTANT, tool_calls=(ToolCall("signed-review", "read_signed_incident_review", {}),)
                ),
                Message(MessageRole.TOOL, result.text, tool_call_id="signed-review"),
                Message(MessageRole.ASSISTANT, '{"status":"complete"}'),
            )
            return SimpleNamespace(session=SimpleNamespace(messages=messages), value={"status": "complete"})

        async def ask(self, binding, prompt, **kwargs):
            context = kwargs["context"]
            captured.append(context)
            assert context.prefix.grounding[1].tool_calls[0].id == context.prefix.grounding[2].tool_call_id
            assert context.prefix.grounding[2].metadata["agent_id"] == "sre-reviewer"
            assert len(context.messages) == 2
            assert "signed" in context.prefix.grounding[2].content
            value = {
                "status": "insufficient_evidence" if missing else "approved",
                "target_release": "" if missing else fixture["approved_target"],
                "approval_ref": "" if missing else fixture["approval_ref"],
                "source_id": "" if missing else fixture["source_id"],
                "evidence_owner": fixture["owner"],
                "retention_days": 180,
                "encryption_required": True,
                "integrity_check_required": True,
            }
            return SimpleNamespace(value=value)

    env = SimpleNamespace(
        runtime=Runtime(),
        binding=None,
        limits=None,
        event_sink=lambda _: None,
        workspace=tmp_path,
        call_id="context-eval",
        alternate_binding=lambda: None,
    )
    case = next(
        c
        for c in CASES
        if c.id == ("context_compaction_insufficient_evidence" if missing else "context_compaction_evidence")
    )
    result = await execute_workload(case, env)
    assert not result.checks
    assert result.evidence["tokens_before"] > result.evidence["tokens_after"]
    assert result.evidence["tokens_before"] > captured[0].limit
    bad = {**result.answer.value, "target_release": fixture["rejected_target"], "status": "approved"}
    assert grade_compaction(fixture, bad, dict(result.evidence), missing=missing)
    assert grade_compaction(
        fixture, result.answer.value, {**result.evidence, "tool_pair_preserved": False}, missing=missing
    )


def completed_memory_state():
    fixture = accounts()
    value = {"case_id": fixture["case_id"], **fixture["contact_update"], "approval_ref": fixture["approval_ref"]}
    before = {"/memories/" + path: text for path, text in fixture["notes"].items()}
    after = {**before, "/memories/handover.md": json.dumps(value)}
    after["/memories/routing.md"] = before["/memories/routing.md"].replace(
        "West Annex | ParcelFast | 14:00-16:00", "Bristol East | MedRoute | 08:00-10:00"
    )
    del after["/memories/expired-draft.md"]
    receipts = [
        {"operation": op} for op in ("brief_created", "routing_updated", "draft_deleted", "restarted", "tenant_denied")
    ]
    receipts.append({"operation": "recall", "epoch": 2, "paths": ["/memories/handover.md"]})
    return fixture, before, after, receipts, value


@pytest.mark.parametrize(
    "phrase",
    [
        "cold-chain",
        "cold-chain packaging and temperature indicator seals",
        "cold-chain packaging with temperature indicator seals",
        "Cold-chain WITH temperature indicator seals",
    ],
)
def test_memory_grader_accepts_only_positive_packaging_equivalence_in_both_records(phrase):
    fixture, before, after, receipts, value = completed_memory_state()
    reported = {**value, "required_packaging": phrase}
    after["/memories/handover.md"] = json.dumps(reported)
    assert grade_memory(fixture, before, after, receipts, reported, readonly=False, other_unchanged=True) == []
    assert reported["required_packaging"] == phrase  # Validation does not rewrite actual evidence.
    wrong_carrier = {**reported, "carrier": "ParcelFast"}
    assert grade_memory(fixture, before, after, receipts, wrong_carrier, readonly=False, other_unchanged=True)


@pytest.mark.parametrize(
    "phrase",
    [
        "ambient",
        "not cold-chain packaging with temperature indicator seals",
        "cold-chain packaging without temperature indicator seals",
        "cold-chain packaging with missing temperature indicator seals",
        "cold-chain packaging",
        "cold-chain packaging and temperature indicator seals optional",
        "",
        None,
    ],
)
@pytest.mark.parametrize("unsafe_record", ["persisted", "reported"])
def test_memory_grader_rejects_unsafe_or_missing_packaging_in_either_record(phrase, unsafe_record):
    fixture, before, after, receipts, value = completed_memory_state()
    unsafe = {**value, "required_packaging": phrase}
    if unsafe_record == "persisted":
        after["/memories/handover.md"] = json.dumps(unsafe)
    else:
        value = unsafe
    assert "persisted and recalled handover do not match the signed account update" in grade_memory(
        fixture, before, after, receipts, value, readonly=False, other_unchanged=True
    )
