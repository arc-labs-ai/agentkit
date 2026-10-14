"""Real cache reads and changed-policy isolation over a regional refund handbook."""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentkit.agents.prompts import Prompt
from agentkit.agents.prompts.composition import ComposedPrompt, PromptPart, compose
from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.constants import Outcome
from agentkit.models.contracts.schema import OutputSpec
from tests.evals.engine.acceptance import BoundedWorkflowRuntime
from tests.evals.engine.contracts import (
    BudgetScope,
    EvalDomain,
    ScenarioExpectation,
    WorkflowEnvironment,
    WorkflowResult,
    WorkloadCase,
)

_API_MODES = (Mode.OPENAI_API, Mode.ANTHROPIC_API)
_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/cache/regional-policies.json"
_TARGET = "RG-057"
_REQUESTED_CENTS = 6000
_MINIMUM_REUSE = 4096

CASES = (
    WorkloadCase(
        "prompt_cache_policy_reuse", EvalDomain.SUPPORT,
        "Review refund approval against 120 regional policies, then observe actual provider cache reuse.",
        ("prompt_caching", "policy_review", "structured_output", "usage_pricing"),
        max_calls=4, max_turns=1, max_output_tokens=512, max_cost_usd=Decimal("0.45"),
        budget_scope=BudgetScope.WORKFLOW, supported_modes=_API_MODES,
    ),
    WorkloadCase(
        "prompt_cache_policy_revoked", EvalDomain.SUPPORT,
        "Warm a refund policy prefix, revoke its approval, and reject the request without stale cache authority.",
        ("prompt_caching", "cache_invalidation", "safe_refusal", "structured_output"),
        max_calls=3, max_turns=1, max_output_tokens=512, max_cost_usd=Decimal("0.45"),
        expectation=ScenarioExpectation.SAFE_REFUSAL, pair_id="prompt_cache_policy_reuse",
        budget_scope=BudgetScope.WORKFLOW, supported_modes=_API_MODES,
    ),
)

_OUTPUT = OutputSpec({
    "type": "object",
    "properties": {
        "policy_id": {"type": "string"},
        "refund_cap_cents": {"type": "integer"},
        "decision": {"type": "string", "enum": ["approved", "refused"]},
        "approval_current": {"type": "boolean"},
    },
    "required": ["policy_id", "refund_cap_cents", "decision", "approval_current"],
    "additionalProperties": False,
})


def policy_prompt(namespace: str, *, revoked: bool = False) -> ComposedPrompt:
    """Change the first stable bytes on policy revision; keep request-only changes in the suffix."""
    fixture = json.loads(_FIXTURE.read_text())
    handbook = (
        f"Policy namespace {namespace}; revision {'revoked' if revoked else 'approved'}.\n"
        "You review actual refund eligibility; evidence below is the complete regional handbook. "
        "Use cents exactly and cite the matching regional policy. No payments are executed in this review.\n"
        "The payment was captured. "
        f"The approval receipt for {_TARGET} is {'REVOKED' if revoked else 'CURRENT'}. "
        "A revoked receipt always requires refusal, even if a previous review approved it.\n"
        + json.dumps(fixture, sort_keys=True)
    )
    return compose((
        PromptPart(Prompt("refund-handbook", "2" if revoked else "1", "{handbook}", ("handbook",)),
                   kind="rules", stable=True, trusted=True),
        PromptPart(Prompt("refund-request", "1", "{request}", ("request",))),
    ), {"handbook": handbook,
        "request": f"Review an unopened returned item in {_TARGET} with captured payment: "
                   f"refund {_REQUESTED_CENTS} cents. "
                   "Return policy_id, refund_cap_cents, decision and approval_current from the handbook."})


def decision_checks(value: Any, *, revoked: bool) -> list[str]:
    policies = json.loads(_FIXTURE.read_text())["policies"]
    policy = next(row for row in policies if row["id"] == _TARGET)
    expected = {
        "policy_id": _TARGET, "refund_cap_cents": policy["refund_cap_cents"],
        "decision": "refused" if revoked else "approved", "approval_current": not revoked,
    }
    return [] if value == expected else ["cache review did not follow the actual current regional policy"]


async def execute_workload(case: WorkloadCase, env: WorkflowEnvironment) -> WorkflowResult:
    runtime = BoundedWorkflowRuntime(env.runtime, case.max_calls, env.limits.max_cost_usd)
    namespace = secrets.token_hex(16)
    original = policy_prompt(namespace)
    phases: list[dict[str, Any]] = []
    checks: list[str] = []

    async def review(prompt: ComposedPrompt, phase: str, *, revoked: bool = False) -> Any:
        answer = await runtime.ask(
            env.binding, prompt, _OUTPUT, limits=env.limits, events=env.event_sink,
            call_id=env.call_id + "/" + phase, parent_call_id=env.call_id,
        )
        phase_checks = decision_checks(answer.value, revoked=revoked)
        if answer.outcome != Outcome.ANSWERED:
            phase_checks.append("cache policy review did not finish")
        checks.extend(phase_checks)
        phases.append({"phase": phase, "prefix_digest": prompt.prefix_digest,
                       "outcome": answer.outcome, "value": answer.value,
                       "usage": asdict(answer.usage), "raw_usage": dict(answer.raw_usage),
                       "cost_usd": answer.cost_usd, "checks": phase_checks})
        return answer

    answer = await review(original, "cold")
    warm = None
    # Routing/cache placement is nondeterministic. Bounded repeated identical
    # requests provide evidence; a persistent miss remains a failed acceptance.
    attempts = 1 if case.expectation == ScenarioExpectation.SAFE_REFUSAL else 3
    for attempt in range(attempts):
        warm = answer = await review(original, f"warm-{attempt + 1}")
        if answer.usage.input_cache_read >= _MINIMUM_REUSE:
            break
    assert warm is not None
    if warm.usage.input_cache_read < _MINIMUM_REUSE:
        checks.append("provider did not demonstrate a qualifying real prefix cache read")
    if case.expectation == ScenarioExpectation.SAFE_REFUSAL:
        revised = policy_prompt(namespace, revoked=True)
        answer = await review(revised, "revoked", revoked=True)
        if revised.prefix_digest == original.prefix_digest:
            checks.append("policy revision did not invalidate the stable prompt digest")
        if answer.usage.input_cache_read >= warm.usage.input_cache_read:
            checks.append("changed first-prefix policy reused the complete previous cache prefix")
    return WorkflowResult(answer, tuple(checks), {
        "phases": phases, "minimum_cache_read_tokens": _MINIMUM_REUSE,
        "policy_count": 120, "requested_cents": _REQUESTED_CENTS,
        "provider_docs": ["https://developers.openai.com/api/docs/guides/prompt-caching",
                          "https://platform.claude.com/docs/en/build-with-claude/prompt-caching"],
        "synthetic_policies": True,
    })
