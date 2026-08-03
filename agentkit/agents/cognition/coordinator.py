"""CoordinatorCognition — delegate the loop to a multi-agent Policy.

Holds the children roster + policy + (optional) coordinator-level
termination. ``drive`` calls ``policy.execute(coordinator, task, ctx,
context)`` and collapses the return into a single terminal ``final``
event.

The Policy owns who-speaks-next, termination, and aggregation; the
Agent owns the ``invoke_agent`` span; the CoordinatorCognition owns
the roster + the policy plug-in. There is no incremental stream from
the policy today — coordinators stream as one terminal event by
design.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import StreamEvent

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent
    from agentkit.agents.control.termination import TerminationCondition
    from agentkit.agents.policies.base import Policy
    from agentkit.capabilities.checkpointer import Checkpointer
    from agentkit.context import WorkingContext


@dataclass(slots=True)
class CoordinatorCognition:
    """Policy-driven multi-agent loop.

    Holds:
        children: ``{name: Agent}`` roster the policy dispatches to.
        policy: the ``Policy`` plug-in (RoundRobin / Selector / Plan /
            Ledger / custom).
        termination: coordinator-level smart stop applied by the policy
            to the multi-agent transcript.
        checkpointer: per-coordinator durable checkpoint store; ``None``
            falls back to ``ctx.checkpointer``. Used by policies that
            support coordinator-level suspend/resume (RoundRobin, Selector).

    ``drive`` yields exactly one terminal ``final`` event carrying the
    aggregated ``AgentResult``.
    """

    children: dict[str, Agent]
    policy: Policy
    termination: TerminationCondition | None = None
    checkpointer: Checkpointer | None = None
    name: str = field(default="coordinator")

    def child(self, name: str) -> Agent:
        if name not in self.children:
            raise KeyError(f"coordinator has no child {name!r}")
        return self.children[name]

    async def drive(
        self,
        agent: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AsyncIterator[StreamEvent]:
        result = await self.policy.execute(agent, task, ctx, context)
        yield StreamEvent("final", result=result, usage=result.usage)


# ``Any`` keeps the dataclass field annotation evaluation cheap when the type
# hints get resolved by tools that don't follow TYPE_CHECKING.
_ = Any


__all__ = ["CoordinatorCognition"]
