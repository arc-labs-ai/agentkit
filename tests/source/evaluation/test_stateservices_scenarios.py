import json

import pytest

from agentkit.core.testing import make_test_ctx
from tests.evals.suites.stateservices import FIXTURES, DispatchState, VectorSupport, grade_dispatch, grade_vector


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


async def call(tools, name, **arguments):
    return await tools.execute(name, arguments, make_test_ctx())


@pytest.mark.asyncio
async def test_dispatch_business_restart_requires_real_checkpoint_journal_replay_and_once_only_effect(tmp_path):
    f = fixture("dispatch.json")
    state = DispatchState(tmp_path, f, stale=False)
    await state.seed()
    tools = state.tools()
    inspected = await call(tools, "inspect_dispatch")
    grant = json.loads(inspected.text)["actor_grant"]
    await call(tools, "verify_stale_lease_rejected")
    args = {
        "owner": grant["owner"],
        "fencing_token": grant["fencing_token"],
        "approval_ref": f["approval_ref"],
        "carrier": f["carrier"],
        "destination": f["destination"],
        "cases": f["cases"],
    }
    committed = await call(tools, "commit_dispatch", **args)
    replayed = await call(tools, "commit_dispatch", **args)
    assert committed.text == replayed.text and not committed.is_error
    restored = await call(tools, "restart_and_replay_dispatch")
    assert not restored.is_error
    evidence = await state.evidence()
    assert "independent review did not read freshly restored checkpoint/journal/replay evidence" in grade_dispatch(
        f, json.loads(committed.text), evidence, stale=False
    )
    reviewed = await call(state.tools(review=True), "restart_and_replay_dispatch")
    assert not reviewed.is_error
    evidence = await state.evidence()
    assert grade_dispatch(f, json.loads(committed.text), evidence, stale=False) == []
    assert len([e for e in evidence["events"] if e["kind"] == "dispatch.committed"]) == 1
    assert grade_dispatch(f, json.loads(committed.text), {**evidence, "receipts": []}, stale=False)
    forged = {**evidence, "business": {**evidence["business"], "shipments": 2}}
    assert grade_dispatch(f, json.loads(committed.text), forged, stale=False)


@pytest.mark.asyncio
async def test_stale_dispatch_actor_is_rejected_by_real_renewal_and_fence_before_business_write(tmp_path):
    f = fixture("dispatch.json")
    state = DispatchState(tmp_path, f, stale=True)
    await state.seed()
    tools = state.tools()
    grant = json.loads((await call(tools, "inspect_dispatch")).text)["actor_grant"]
    await call(tools, "verify_stale_lease_rejected")
    denied = await call(
        tools,
        "commit_dispatch",
        owner=grant["owner"],
        fencing_token=grant["fencing_token"],
        approval_ref=f["approval_ref"],
        carrier=f["carrier"],
        destination=f["destination"],
        cases=f["cases"],
    )
    assert denied.is_error and "fencing token is stale" in denied.text
    assert await state.leases.renew(state.old, ttl_s=600) is None
    escalated = await call(tools, "escalate_stale_dispatch")
    evidence = await state.evidence()
    assert grade_dispatch(f, json.loads(escalated.text), evidence, stale=True) == []
    assert evidence["business"]["available"] == f["stock_available"]
    assert grade_dispatch(f, {"status": "escalated", "receipt_id": "invented"}, evidence, stale=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_vector_scope_and_metadata_control_real_retrieval_and_persisted_disposition(tmp_path, missing):
    f = fixture("support.json")
    state = VectorSupport(tmp_path, f, missing=missing)
    await state.seed()
    tools = state.tools()
    searched = json.loads((await call(tools, "search_approved_support", query=f["query"])).text)
    ids = [doc["id"] for doc in searched["documents"]]
    assert ids == ([] if missing else [f["current"]["id"]])
    assert f["draft"]["id"] not in ids and f["foreign"]["id"] not in ids
    await call(tools, "verify_support_tenant_boundary")
    committed = await call(
        tools,
        "record_support_disposition",
        action="escalate" if missing else f["current"]["action"],
        source_id="" if missing else f["current"]["id"],
    )
    evidence = await state.evidence()
    assert grade_vector(f, json.loads(committed.text), evidence, missing=missing) == []
    assert grade_vector(f, {"status": "resolved", "receipt_id": "invented"}, evidence, missing=missing)


@pytest.mark.asyncio
async def test_vector_missing_authority_cannot_borrow_draft_or_other_tenant_document(tmp_path):
    f = fixture("support.json")
    state = VectorSupport(tmp_path, f, missing=True)
    await state.seed()
    tools = state.tools()
    await call(tools, "search_approved_support", query=f["query"])
    for source in (f["draft"]["id"], f["foreign"]["id"]):
        denied = await call(tools, "record_support_disposition", action="clear_fault", source_id=source)
        assert denied.is_error
    assert (await state.evidence())["service"] == {"logs_retained": True, "device_disposition": "pending"}
    assert grade_vector(f, None, await state.evidence(), missing=True)


@pytest.mark.asyncio
async def test_registered_dispatch_workflow_uses_real_tool_receipts_and_cross_model_checkpoint_review(tmp_path):
    from decimal import Decimal
    from types import SimpleNamespace

    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.models.contracts.types import Binding, CallLimits, Turn
    from tests.evals.suites.stateservices import CASES, execute_workload
    from tests.source.conformance.test_calls import PRICES, ScriptedAdapter

    f = fixture("dispatch.json")
    receipt = {"status": "dispatched", "receipt_id": "DSP-" + f["case_id"]}

    def toolcall(name, arguments=None, identifier=None):
        return {"id": identifier or name, "name": name, "arguments": arguments or {}}

    def response(value=None, calls=()):
        return Turn(
            (json.dumps(value),) if value else (),
            value=value,
            tool_calls=calls,
            raw_usage={"input_tokens": 1, "output_tokens": 1},
        )

    args = {
        "owner": "current-shift",
        "fencing_token": 2,
        "approval_ref": f["approval_ref"],
        "carrier": f["carrier"],
        "destination": f["destination"],
        "cases": f["cases"],
    }
    adapter = ScriptedAdapter(
        response(calls=(toolcall("inspect_dispatch"), toolcall("verify_stale_lease_rejected"))),
        response(calls=(toolcall("commit_dispatch", args, "commit"),)),
        response(calls=(toolcall("commit_dispatch", args, "replay"),)),
        response(calls=(toolcall("restart_and_replay_dispatch"),)),
        response(receipt),
        response(calls=(toolcall("restart_and_replay_dispatch", identifier="independent-review"),)),
        response(receipt),
    )
    env = SimpleNamespace(
        runtime=CallRuntime(adapters={"openai_api": adapter, "anthropic_api": adapter}, pricing=PRICES),
        binding=Binding("openai_api", "test"),
        limits=CallLimits(max_cost_usd=Decimal(".30"), max_turns=12),
        event_sink=lambda _: None,
        workspace=tmp_path,
        call_id="dispatch-eval",
        alternate_binding=lambda: Binding("anthropic_api", "test"),
    )
    result = await execute_workload(CASES[0], env)
    assert not result.checks
    assert adapter.requests[-1][0].mode == "anthropic_api"
    assert len([r for r in result.evidence["receipts"] if r["operation"] == "durable_replayed"]) == 2
    assert result.evidence["business"]["shipments"] == 1


@pytest.mark.asyncio
async def test_vector_approved_source_cannot_authorize_unsafe_action_or_uncited_resolution(tmp_path):
    f = fixture("support.json")
    state = VectorSupport(tmp_path, f, missing=False)
    await state.seed()
    tools = state.tools()
    await call(tools, "search_approved_support", query=f["query"])
    assert (
        await call(tools, "record_support_disposition", action="clear_fault", source_id=f["current"]["id"])
    ).is_error
    assert (await call(tools, "record_support_disposition", action="quarantine_device", source_id="")).is_error
    assert (await state.evidence())["service"]["device_disposition"] == "pending"
