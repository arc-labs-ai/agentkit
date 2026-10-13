"""Discoverable catalogue separating contract checks from application workloads."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from tests.evals.engine.contracts import WorkflowEnvironment, WorkflowResult, WorkloadCase
from tests.evals.suites import caching, coding, framework, knowledge, operations
from tests.evals.suites.conformance import CASES as CONFORMANCE_CASES
from tests.evals.suites.conformance import EvalCase

WorkloadExecutor = Callable[[WorkloadCase, WorkflowEnvironment], Awaitable[WorkflowResult]]
MODULES = (coding, knowledge, operations, framework, caching)
WORKLOAD_CASES = tuple(case for module in MODULES for case in module.CASES)
CASES: tuple[WorkloadCase | EvalCase, ...] = (*WORKLOAD_CASES, *CONFORMANCE_CASES)
WORKLOAD_EXECUTORS: dict[str, WorkloadExecutor] = {
    case.id: module.execute_workload for module in MODULES for case in module.CASES
}
if len({case.id for case in CASES}) != len(CASES):
    raise ValueError("evaluation catalogue contains duplicate case identities")


def describe_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": case.id,
            "suite": case.suite,
            "domain": case.domain if isinstance(case, WorkloadCase) else "conformance",
            "description": case.description if isinstance(case, WorkloadCase) else ", ".join(case.contract),
            "aspects": case.aspects if isinstance(case, WorkloadCase) else case.contract,
            "max_calls": case.max_calls,
            "expectation": case.expectation if isinstance(case, WorkloadCase) else "success",
            "pair_id": case.pair_id if isinstance(case, WorkloadCase) else None,
            "supported_modes": case.supported_modes,
        }
        for case in CASES
    ]
