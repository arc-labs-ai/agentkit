"""Bounded application verification using the public agent harness."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from typing import Any

from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet, ToolGrant
from agentkit.core.contracts.types import Message
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.constants import MindKind, Outcome
from agentkit.models.contracts.schema import OutputSpec
from agentkit.models.contracts.types import Answer, CallLimits, Usage
from agentkit.state.context import WorkingContext
from agentkit.tools.execution.policy import ToolSet
from tests.evals.engine.contracts import WorkflowEnvironment, WorkloadCase


class BoundedWorkflowRuntime(CallRuntime):
    """Share one model-invocation count and exact spend ceiling across workflow phases."""

    def __init__(self, delegate: CallRuntime, max_calls: int, ceiling: Decimal | None = None) -> None:
        super().__init__()
        self.delegate, self.max_calls, self.ceiling = delegate, max_calls, ceiling
        self.used_calls = 0
        self._spent = Decimal(0)
        self._unknown = False
        self._last: Answer[Any] | None = None

    async def run(self, *args: Any, **kwargs: Any) -> Answer[Any]:
        remaining = None if self.ceiling is None else self.ceiling - self._spent
        if self.used_calls >= self.max_calls or (
            self.ceiling is not None and (self._unknown or remaining is not None and remaining <= 0)
        ):
            assert self._last is not None
            return replace(self._last, outcome=Outcome.LIMIT_REACHED, text="", value=None, messages=(),
                           usage=Usage(), raw_usage={}, cost_usd=Decimal(0), attempts=0)
        limits = kwargs.get("limits", CallLimits())
        if remaining is not None:
            ceiling = remaining if limits.max_cost_usd is None else min(remaining, limits.max_cost_usd)
            kwargs["limits"] = replace(limits, max_cost_usd=ceiling)
        self.used_calls += 1
        self._last = await self.delegate.run(*args, **kwargs)
        if self._last.cost_usd is None:
            self._unknown = True
        else:
            self._spent += self._last.cost_usd
        return self._last


async def run_verified(
    env: WorkflowEnvironment,
    case: WorkloadCase,
    prompt: str,
    tools: ToolSet | None,
    checker: Callable[[Answer[Any]], Any],
    *,
    output: OutputSpec | None = None,
    context: WorkingContext | None = None,
    mind: MindKind = MindKind.REACT,
    max_calls: int = 2,
    resume: Any = None,
    parent_call_id: str | None = None,
) -> tuple[Answer[Any], list[dict[str, Any]]]:
    """Retain failed submissions and bounded feedback; never disclose expected oracle answers."""
    working = context if context is not None else WorkingContext()
    submissions: list[dict[str, Any]] = []

    async def verify(answer: Answer[Any]) -> bool:
        checks = checker(answer)
        if inspect.isawaitable(checks):
            checks = await checks
        failures = list(checks)
        submissions.append({"outcome": answer.outcome, "value": answer.value,
                            "checks": failures, "accepted": not failures})
        if failures:
            working.append(Message("user", "Host acceptance checks rejected this submission: "
                                   + "; ".join(failures)
                                   + ". Recheck the actual evidence and tool receipts. "
                                   "Do not invent results or claim uncommitted actions."))
        return not failures

    runtime = BoundedWorkflowRuntime(env.runtime, max_calls)
    definition = AgentDefinition(
        case.id, "2", case.description, tools=ToolGrant(allow=tuple(tools.registry.names()) if tools else ()),
        limits=env.limits, mind=mind,
    )
    runner = AgentRunner(runtime, tools=tools)
    answer = await runner.run_agent(
        definition,
        InstructionSet(prompt, definition_of_done="All declared factual and tool/state acceptance checks pass.",
                       verifier=verify), {}, output=output, binding=env.binding,
        context=working, resume=resume, events=env.event_sink, call_id=env.call_id,
        parent_call_id=parent_call_id,
    )
    return answer, submissions
