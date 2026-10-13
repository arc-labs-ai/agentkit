"""Typed workload contracts independent of model transport and application domain."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentkit.core.contracts.types import Mode
from agentkit.core.runtime.calls import CallRuntime
from agentkit.core.serialization._immutable import deep_freeze
from agentkit.models.contracts.types import Answer, Binding, CallLimits, EventSink

if TYPE_CHECKING:
    from tests.evals.sandbox.workspace import WorkspaceSandbox


class EvalDomain(StrEnum):
    CODING = "coding"
    RESEARCH = "research"
    ANALYTICS = "analytics"
    SUPPORT = "support"
    OPERATIONS = "operations"
    SECURITY = "security"


class ScenarioExpectation(StrEnum):
    SUCCESS = "success"
    SAFE_REFUSAL = "safe_refusal"
    RECOVERY = "recovery"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AUTH_FAILURE = "auth_failure"
    CANCELLED = "cancelled"
    LIMIT_REACHED = "limit_reached"


class BudgetScope(StrEnum):
    PER_CALL = "per_call"
    WORKFLOW = "workflow"


class EvalStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class WorkloadCase:
    id: str
    domain: EvalDomain
    description: str
    aspects: tuple[str, ...]
    max_calls: int = 1
    max_turns: int = 16
    max_output_tokens: int = 12000
    max_cost_usd: Decimal = Decimal("0.75")
    wall_clock_s: float = 300
    expectation: ScenarioExpectation = ScenarioExpectation.SUCCESS
    pair_id: str | None = None
    budget_scope: BudgetScope = BudgetScope.PER_CALL
    supported_modes: tuple[Mode, ...] = tuple(Mode)
    idle_s: float | None = None
    first_event_s: float | None = None
    suite: str = field(default="workload", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", EvalDomain(self.domain))
        object.__setattr__(self, "expectation", ScenarioExpectation(self.expectation))
        object.__setattr__(self, "budget_scope", BudgetScope(self.budget_scope))
        object.__setattr__(self, "supported_modes", tuple(Mode(mode) for mode in self.supported_modes))
        if not self.supported_modes or len(set(self.supported_modes)) != len(self.supported_modes):
            raise ValueError("workloads require distinct supported modes")
        if self.pair_id == self.id:
            raise ValueError("a scenario must pair with a distinct case")
        if not self.id or not self.description or not self.aspects:
            raise ValueError("workloads require identity, description and acceptance aspects")
        if min(self.max_calls, self.max_turns, self.max_output_tokens, self.wall_clock_s) <= 0:
            raise ValueError("workload bounds must be positive")
        if not self.max_cost_usd.is_finite() or self.max_cost_usd <= 0:
            raise ValueError("workload spend ceiling must be finite and positive")
        if any(value is not None and (not math.isfinite(value) or value <= 0)
               for value in (self.idle_s, self.first_event_s)):
            raise ValueError("workload liveness limits must be finite and positive")


@dataclass(frozen=True)
class WorkflowResult:
    answer: Answer[Any] | None
    checks: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", deep_freeze(dict(self.evidence)))


@dataclass
class WorkflowEnvironment:
    runtime: CallRuntime
    binding: Binding
    limits: CallLimits
    event_sink: EventSink
    workspace: Path
    call_id: str
    sandbox: WorkspaceSandbox

    def alternate_binding(self) -> Binding:
        from tests.evals.engine.config import MODES, EvalConfig

        mode = MODES[(MODES.index(self.binding.mode) + 1) % len(MODES)]
        return EvalConfig().binding(mode, self.binding.credential)
