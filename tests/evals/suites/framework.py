"""API lifecycle failures around a real scoped incident-diagnostic workflow.

Native CLI cancellation and turn accounting have different capabilities; these
scenarios explicitly run only through the two API transports.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentkit.core.contracts.types import Mode
from agentkit.core.execution.concurrency import CancellationToken
from agentkit.models.contracts.constants import FailureClass, Outcome
from agentkit.models.contracts.schema import OutputSpec
from agentkit.models.contracts.types import Binding, CredentialRef
from agentkit.tools import ToolRegistry
from agentkit.tools.declarations.function import FunctionTool
from agentkit.tools.execution.policy import ToolSet
from tests.evals.engine.acceptance import run_verified
from tests.evals.engine.contracts import (
    EvalDomain,
    ScenarioExpectation,
    WorkflowEnvironment,
    WorkflowResult,
    WorkloadCase,
)

API_MODES = (Mode.OPENAI_API, Mode.ANTHROPIC_API)
_SUCCESS = "framework_diagnostic_success"
CASES = (
    WorkloadCase(
        _SUCCESS,
        EvalDomain.OPERATIONS,
        "Diagnose a degraded canary from scoped telemetry and record one approved recommendation.",
        ("real_tool_execution", "trusted_evidence", "scoped_business_write"),
        max_calls=3,
        max_turns=8,
        max_output_tokens=4000,
        max_cost_usd=Decimal("0.30"),
        supported_modes=API_MODES,
    ),
    WorkloadCase(
        "framework_cancelled_diagnostic_resume",
        EvalDomain.OPERATIONS,
        "Cancel an in-flight diagnostic, verify cleanup and safely resume without dangling tool calls.",
        ("cancellation", "cleanup", "canonical_tool_pair", "safe_resume"),
        max_calls=2,
        max_turns=8,
        max_output_tokens=4000,
        max_cost_usd=Decimal("0.40"),
        expectation=ScenarioExpectation.CANCELLED,
        pair_id=_SUCCESS,
        supported_modes=API_MODES,
    ),
    WorkloadCase(
        "framework_turn_limited_diagnostic_resume",
        EvalDomain.OPERATIONS,
        "Stop after a diagnostic turn, retain its evidence and resume to record the scoped recommendation.",
        ("turn_limit", "context_retention", "safe_resume", "canonical_tool_pair"),
        max_calls=2,
        max_turns=8,
        max_output_tokens=4000,
        max_cost_usd=Decimal("0.40"),
        expectation=ScenarioExpectation.LIMIT_REACHED,
        pair_id=_SUCCESS,
        supported_modes=API_MODES,
    ),
    WorkloadCase(
        "framework_api_auth_rejected",
        EvalDomain.SECURITY,
        "Verify a real API rejects a synthetic invalid credential before any incident business write.",
        ("auth_failure", "prewrite_refusal", "credential_boundary"),
        max_turns=1,
        max_output_tokens=256,
        max_cost_usd=Decimal("0.10"),
        expectation=ScenarioExpectation.AUTH_FAILURE,
        pair_id=_SUCCESS,
        supported_modes=API_MODES,
    ),
)
_SCHEMA = OutputSpec(
    {
        "type": "object",
        "properties": {
            "incident_id": {"type": "string"},
            "recommendation_id": {"type": "string"},
            "action": {"type": "string", "enum": ["rollback_canary"]},
        },
        "required": ["incident_id", "recommendation_id", "action"],
        "additionalProperties": False,
    }
)


@dataclass
class DiagnosticWorkflow:
    workspace: Path
    blocking: bool = False
    started: asyncio.Event = field(default_factory=asyncio.Event)
    cleaned: asyncio.Event = field(default_factory=asyncio.Event)
    journal: list[dict[str, Any]] = field(default_factory=list)
    completed: int = 0
    recommendations: list[dict[str, str]] = field(default_factory=list)
    approval: str = field(default_factory=lambda: "APR-INC-" + secrets.token_hex(8))

    def prepare(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "incident.json").write_text(
            json.dumps(
                {
                    "incident_id": "INC-9012",
                    "service": "orders-api",
                    "canary_release": "r2026.10.13.1",
                    "stable_release": "r2026.10.12.4",
                    "canary_error_rate": 0.087,
                    "baseline_error_rate": 0.004,
                    "rollback_threshold": 0.01,
                    "approved_action": "rollback_canary",
                    "approval_ref": self.approval,
                }
            ),
            encoding="utf-8",
        )

    async def diagnose_incident(self, incident_id: str) -> dict[str, Any]:
        """Read scoped incident telemetry and the approved remediation reference. Complete diagnostic sampling
        before calling record_recommendation. approval_ref authorizes that tool; it is not a recommendation_id
        or evidence that the recommendation was committed. This tool can be interrupted while sampling.
        """
        if incident_id != "INC-9012":
            raise PermissionError("diagnostic request is outside the assigned incident")
        self.journal.append({"action": "diagnostic_started", "incident_id": incident_id})
        self.started.set()
        try:
            data: dict[str, Any] = json.loads(await asyncio.to_thread((self.workspace / "incident.json").read_text))
            if self.blocking:
                await asyncio.Event().wait()
            self.completed += 1
            self.journal.append({"action": "diagnostic_completed", "incident_id": incident_id})
            # Telemetry is evidence for the next authorized action, never a
            # fabricated receipt for an action that has not happened.
            data["recommendation_committed"] = False
            data["next_action"] = {
                "tool": "record_recommendation",
                "arguments": {"incident_id": incident_id, "action": data["approved_action"],
                              "approval_ref": data["approval_ref"]},
            }
            return data
        finally:
            self.cleaned.set()
            self.journal.append({"action": "diagnostic_cleanup", "incident_id": incident_id})

    def record_recommendation(self, incident_id: str, action: str, approval_ref: str) -> dict[str, str]:
        """Record one recommendation using the exact approval reference returned by the completed diagnostic.
        Only the approved incident and rollback_canary action are permitted. Identical repeats are idempotent.
        Return its actual committed receipt; recommendation_id exists only after this tool succeeds.
        """
        if (
            not self.completed
            or incident_id != "INC-9012"
            or action != "rollback_canary"
            or approval_ref != self.approval
        ):
            self.journal.append({"action": "recommendation_refused"})
            raise PermissionError("recommendation lacks completed scoped diagnostic and approved action/reference")
        if self.recommendations:
            return self.recommendations[0]
        receipt = {"incident_id": incident_id, "action": action, "recommendation_id": "REC-INC-9012"}
        self.recommendations.append(receipt)
        self.journal.append({"action": "recommendation_committed", **receipt})
        return receipt

    def tools(self, *, diagnosis: bool = True) -> ToolSet:
        tools = [FunctionTool.from_callable(self.record_recommendation, side_effecting=True)]
        if diagnosis:
            tools.append(FunctionTool.from_callable(self.diagnose_incident, side_effecting=False))
        return ToolSet(ToolRegistry.from_tools(tools))

    def evidence(self) -> dict[str, Any]:
        return {
            "diagnostics_completed": self.completed,
            "cleanup_observed": self.cleaned.is_set(),
            "recommendations": list(self.recommendations),
            "journal": list(self.journal),
        }


def complete_pairs(messages: Any) -> bool:
    calls = {call.id for message in messages for call in message.tool_calls}
    results = {message.tool_call_id for message in messages if message.role == "tool"}
    return bool(calls) and calls == results


def grade_completed(state: DiagnosticWorkflow, answer: Any) -> list[str]:
    checks = []
    if answer.outcome != Outcome.ANSWERED or answer.value != {
        "incident_id": "INC-9012",
        "action": "rollback_canary",
        "recommendation_id": "REC-INC-9012",
    }:
        checks.append("workflow did not return its trusted committed recommendation receipt")
    if state.completed < 1 or len(state.recommendations) != 1:
        checks.append("diagnostic or once-only recommendation commit was missing")
    if any(item["action"] == "recommendation_refused" for item in state.journal):
        checks.append("model attempted an out-of-scope or unverified recommendation")
    return checks


async def execute_workload(case: WorkloadCase, environment: WorkflowEnvironment) -> WorkflowResult:
    if environment.binding.mode not in API_MODES:
        raise ValueError("API lifecycle workloads do not claim native CLI cancellation/turn-limit equivalence")
    cancellation = case.expectation == ScenarioExpectation.CANCELLED
    state = DiagnosticWorkflow(environment.workspace, blocking=cancellation)
    state.prepare()
    prompt = (
        "Diagnose INC-9012 from the actual incident telemetry tool. Compare the canary error rate to its "
        "rollback threshold, then record the approved recommendation using the exact diagnostic approval "
        "reference by calling record_recommendation. Diagnosis alone does not commit a recommendation: "
        "approval_ref is not recommendation_id. Record it once and return only the receipt actually returned "
        "by the successful record_recommendation tool, never a fabricated or substituted identifier."
    )
    kwargs: dict[str, Any] = {
        "events": environment.event_sink,
        "call_id": environment.call_id,
        "output": _SCHEMA,
        "limits": environment.limits,
    }
    if case.expectation == ScenarioExpectation.AUTH_FAILURE:
        invalid = Binding(
            environment.binding.mode,
            environment.binding.model,
            CredentialRef(
                lambda: {
                    "OPENAI_API_KEY": "invalid-eval-model-credential",
                    "ANTHROPIC_API_KEY": "invalid-eval-model-credential",
                }
            ),
        )
        answer = await environment.runtime.run(invalid, prompt, state.tools(), **kwargs)
        checks = []
        if answer.outcome != Outcome.FAILED or not answer.failure or answer.failure.classification != FailureClass.AUTH:
            checks.append("provider did not classify the invalid credential as an authentication failure")
        if state.journal or state.recommendations:
            checks.append("business tool work occurred before authentication rejection")
        return WorkflowResult(
            answer,
            tuple(checks),
            {
                **state.evidence(),
                "synthetic_invalid_credential": True,
                "business_ledger_unchanged": not state.recommendations,
            },
        )
    if case.expectation == ScenarioExpectation.SUCCESS:
        answer, submissions = await run_verified(
            environment,
            case,
            prompt,
            state.tools(),
            lambda candidate: grade_completed(state, candidate),
            output=_SCHEMA,
            max_calls=case.max_calls,
        )
        return WorkflowResult(
            answer,
            tuple(grade_completed(state, answer)),
            {**state.evidence(), "verification_submissions": submissions},
        )
    if cancellation:
        token = CancellationToken()
        task = asyncio.create_task(
            environment.runtime.run(environment.binding, prompt, state.tools(), cancel=token, **kwargs)
        )
        entered = asyncio.create_task(state.started.wait())
        try:
            done, _ = await asyncio.wait(
                {task, entered}, timeout=environment.limits.wall_clock_s or 180, return_when=asyncio.FIRST_COMPLETED
            )
            token.cancel()
            interrupted = await task
            invoked = entered in done and state.started.is_set()
        finally:
            if not entered.done():
                entered.cancel()
                await asyncio.gather(entered, return_exceptions=True)
        expected = Outcome.CANCELLED
    else:
        interrupted = await environment.runtime.run(
            environment.binding, prompt, state.tools(), **{**kwargs, "limits": replace(environment.limits, max_turns=1)}
        )
        invoked = state.started.is_set()
        expected = Outcome.LIMIT_REACHED
    checks = []
    if interrupted.outcome != expected:
        checks.append("initial workflow did not reach the expected terminal interruption")
    if not invoked or not state.cleaned.is_set():
        checks.append("actual diagnostic was not invoked and cleaned up")
    if state.recommendations:
        checks.append("recommendation was committed before interruption recovery")
    if interrupted.session is None or not complete_pairs(interrupted.session.messages):
        checks.append("interrupted continuation contains an unpaired canonical tool call")
    if not cancellation and (
        interrupted.session is None
        or not any(m.role == "tool" and "approval_ref" in m.content for m in interrupted.session.messages)
    ):
        checks.append("turn-limit continuation lost the completed diagnostic evidence")
    if checks:
        return WorkflowResult(interrupted, tuple(checks), state.evidence())
    assert interrupted.session is not None
    state.blocking = False
    resumed = await environment.runtime.run(
        environment.binding,
        "Resume INC-9012 safely. "
        + (
            "The interrupted diagnostic returned no confirmed telemetry; rerun it before recording a recommendation."
            if cancellation
            else "Use the preserved diagnostic evidence. Diagnosis cannot be rerun; record its recommendation."
        )
        + " Call record_recommendation using the diagnostic approval_ref. That approval reference is not "
        "a recommendation receipt. Return the committed receipt fields only after the tool succeeds.",
        state.tools(diagnosis=cancellation),
        resume=interrupted.session,
        **{**kwargs, "call_id": environment.call_id + "/resume"},
    )
    checks.extend(grade_completed(state, resumed))
    return WorkflowResult(
        interrupted,
        tuple(checks),
        {
            **state.evidence(),
            "initial_outcome": interrupted.outcome,
            "initial_context": [
                dict(role=m.role, tool_call_id=m.tool_call_id, content=m.content, metadata=dict(m.metadata))
                for m in interrupted.session.messages
            ],
            "canonical_pairs_complete": True,
            "resumed_outcome": resumed.outcome,
            "resumed_value": resumed.value,
            "resumed_cost_usd": str(resumed.cost_usd),
        },
    )
