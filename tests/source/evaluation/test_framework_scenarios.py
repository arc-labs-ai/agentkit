from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from agentkit.core.contracts.types import Mode
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef, Turn
from agentkit.models.providers.frontiers.calls import ApiAdapter
from tests.evals.engine.contracts import ScenarioExpectation
from tests.evals.suites.framework import API_MODES, CASES, DiagnosticWorkflow, complete_pairs, execute_workload
from tests.source.conformance.test_calls import PRICES


class DiagnosticAdapter:
    """Offline model transport: all diagnostics and business tool work are real."""

    def __init__(self):
        self.requests = []
        self.next_id = 0

    async def turn(self, binding, prompt, **kwargs):
        self.requests.append(kwargs)
        self.next_id += 1
        data = []
        for message in kwargs["session"].messages if kwargs["session"] else ():
            if message.role == "tool":
                try:
                    data.append(json.loads(message.content))
                except ValueError:
                    pass
        receipts = [item for item in data if "recommendation_id" in item]
        telemetry = [item for item in data if "approval_ref" in item]
        if receipts:
            return Turn((json.dumps(receipts[-1]),), raw_usage={"input_tokens": 10, "output_tokens": 5})
        if telemetry:
            payload = {
                "incident_id": "INC-9012",
                "action": "rollback_canary",
                "approval_ref": telemetry[-1]["approval_ref"],
            }
            name = "record_recommendation"
        else:
            assert "diagnose_incident" in [schema.name for schema in kwargs["tools"].schemas()], (
                "diagnostic context was lost"
            )
            payload = {"incident_id": "INC-9012"}
            name = "diagnose_incident"
        return Turn(
            tool_calls=({"id": f"diagnostic-{self.next_id}", "name": name, "arguments": payload},),
            raw_usage={"input_tokens": 10, "output_tokens": 5},
        )


def environment(runtime, mode, workspace, case):
    return SimpleNamespace(
        runtime=runtime,
        binding=Binding(
            mode,
            "test",
            CredentialRef(
                lambda: {
                    "OPENAI_API_KEY": "should-not-be-used-for-invalid-auth",
                    "ANTHROPIC_API_KEY": "should-not-be-used-for-invalid-auth",
                }
            ),
        ),
        limits=CallLimits(max_turns=8, max_output_tokens=4000, wall_clock_s=5),
        event_sink=lambda event: None,
        workspace=workspace,
        call_id="framework-offline/" + case.id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", API_MODES)
@pytest.mark.parametrize("case", CASES[:3], ids=lambda case: case.id)
async def test_real_diagnostic_lifecycle_and_safe_continuation(mode, case, tmp_path):
    adapter = DiagnosticAdapter()
    runtime = CallRuntime(adapters={mode: adapter}, pricing=PRICES)
    result = await execute_workload(case, environment(runtime, mode, tmp_path, case))
    assert result.checks == (), result.evidence
    assert len(result.evidence["recommendations"]) == 1
    assert result.evidence["cleanup_observed"]
    if case.expectation != ScenarioExpectation.SUCCESS:
        assert result.answer.outcome == case.expectation.value
        assert result.evidence["canonical_pairs_complete"]
        assert result.evidence["resumed_outcome"] == "answered"
        assert complete_pairs(adapter.requests[-1]["session"].messages)
        if case.expectation == ScenarioExpectation.LIMIT_REACHED:
            assert result.evidence["diagnostics_completed"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", API_MODES)
async def test_synthetic_invalid_auth_reaches_api_boundary_before_business_tools(mode, tmp_path):
    def reject(request):
        header = (
            request.headers.get("x-api-key") if mode == Mode.ANTHROPIC_API else request.headers.get("authorization")
        )
        assert "invalid-eval-model-credential" in header
        assert "should-not-be-used" not in header
        return httpx.Response(401, json={"error": {"type": "authentication_error", "code": "invalid_api_key"}})

    case = CASES[-1]
    client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
    try:
        runtime = CallRuntime(adapters={mode: ApiAdapter(client=client)}, pricing=PRICES)
        result = await execute_workload(case, environment(runtime, mode, tmp_path, case))
        assert result.checks == ()
        assert result.answer.failure.classification == "auth"
        assert result.evidence["journal"] == []
        assert result.evidence["business_ledger_unchanged"]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_lifecycle_workloads_explicitly_refuse_unsupported_cli_equivalence(tmp_path):
    case = CASES[1]
    assert case.supported_modes == API_MODES
    with pytest.raises(ValueError, match="native CLI"):
        await execute_workload(case, environment(CallRuntime(), Mode.CODEX, tmp_path, case))


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES[:3], ids=lambda case: case.id)
async def test_approval_reference_cannot_be_substituted_for_a_committed_receipt(case, tmp_path):
    class InventedReceiptAdapter(DiagnosticAdapter):
        async def turn(self, binding, prompt, **kwargs):
            session = kwargs["session"]
            telemetry = []
            for message in session.messages if session else ():
                if message.role == "tool":
                    try:
                        value = json.loads(message.content)
                    except ValueError:
                        continue
                    if "approval_ref" in value:
                        telemetry.append(value)
            if telemetry:
                self.requests.append(kwargs)
                return Turn((json.dumps({
                    "incident_id": "INC-9012", "action": "rollback_canary",
                    "recommendation_id": telemetry[-1]["approval_ref"],
                }),), raw_usage={"input_tokens": 10, "output_tokens": 5})
            return await super().turn(binding, prompt, **kwargs)

    adapter = InventedReceiptAdapter()
    runtime = CallRuntime(adapters={Mode.ANTHROPIC_API: adapter}, pricing=PRICES)
    result = await execute_workload(case, environment(runtime, Mode.ANTHROPIC_API, tmp_path, case))
    assert "workflow did not return its trusted committed recommendation receipt" in result.checks
    assert "diagnostic or once-only recommendation commit was missing" in result.checks
    assert result.evidence["diagnostics_completed"] == 1
    assert result.evidence["recommendations"] == []


@pytest.mark.asyncio
async def test_diagnostic_next_action_is_authorized_guidance_not_a_commit_receipt(tmp_path):
    state = DiagnosticWorkflow(tmp_path)
    state.prepare()
    telemetry = await state.diagnose_incident("INC-9012")
    assert telemetry["recommendation_committed"] is False
    assert "recommendation_id" not in telemetry and not state.recommendations
    action = telemetry["next_action"]
    assert action["tool"] == "record_recommendation"
    receipt = state.record_recommendation(**action["arguments"])
    assert state.recommendations == [receipt]
    assert receipt["recommendation_id"] != telemetry["approval_ref"]
    assert state.record_recommendation(**action["arguments"]) == receipt
    assert len(state.recommendations) == 1
