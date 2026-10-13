"""Policy — orchestration strategy for a coordinator Agent.

A ``Policy`` decides who speaks next, how termination is determined, and what the final
result aggregates. A coordinator ``Agent(children=..., policy=...)`` delegates its
``run`` to ``self.policy.execute(self, task, ctx, context)``; the Policy owns the loop
semantics while the Agent owns the lifecycle (the ``invoke_agent`` span, the
``AgentResult`` type).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentkit.agents.definitions.results import AgentResult
from agentkit.core.contracts.protocols import Ctx
from agentkit.state.context import WorkingContext

if TYPE_CHECKING:
    from agentkit.agents.execution.agent import Agent


@runtime_checkable
class Policy(Protocol):
    """Orchestration strategy for a coordinator Agent.

    The Policy decides who speaks next, how termination is determined, and what the
    final result aggregates. A coordinator Agent delegates its run loop to its Policy,
    which drives the multi-agent interaction over ``coordinator.children`` until the
    coordinator's termination condition fires or the policy's own ceiling is hit.
    """

    name: str

    async def execute(
        self,
        coordinator: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AgentResult: ...


__all__ = ["Policy"]
