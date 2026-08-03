"""Cognition Protocol — a turn-taking strategy for an Agent.

The Strategy plug-in that decides what an agent does next and how
the loop terminates. The Agent owns lifecycle (the ``invoke_agent``
span, services, durable resume); the Cognition owns "what's the
next move and when do we stop."

Three first-party cognitions cover today's regimes:

- ``SingleCallCognition``  — one chat call + optional parse-and-repair.
- ``ReActCognition``       — chat ↔ tool-call loop with smart termination,
                             human-in-the-loop suspend/resume, and durable
                             checkpoint persistence.
- ``CoordinatorCognition`` — delegate the loop to a multi-agent ``Policy``
                             that drives ``agent.children``.

Adding a new cognition is implementing this Protocol; no Agent
subclassing, no parallel class hierarchy.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import StreamEvent

if TYPE_CHECKING:
    from agentkit.agents.agent import Agent
    from agentkit.context import WorkingContext


@runtime_checkable
class Cognition(Protocol):
    """A turn-taking strategy for an Agent.

    ``name`` is the stable diagnostic label stamped on the
    ``invoke_agent`` span (``agentkit.agent.cognition``) so an
    operator can see which strategy ran.

    ``drive`` yields ``StreamEvent``s in the same shape ``Agent.stream``
    has always produced: ``message_delta`` for token streaming,
    ``tool_call`` / ``tool_result`` / ``interrupt`` / ``step`` for
    loop progress, and exactly one terminal ``final`` event carrying
    the ``AgentResult``.

    The cognition reads configuration from the ``Agent`` it's driving
    — ``agent.model``, ``agent.tools``, ``agent.parse``,
    ``agent.termination``, etc. — rather than duplicating state.
    Cognitions are pure behavior; the Agent is the state holder.
    """

    name: str

    def drive(
        self,
        agent: Agent,
        task: str,
        ctx: Ctx,
        context: WorkingContext,
    ) -> AsyncIterator[StreamEvent]: ...


__all__ = ["Cognition"]
