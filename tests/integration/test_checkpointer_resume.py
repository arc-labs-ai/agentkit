"""E2E test for the Checkpointer capability + resume-across-processes.

Covers:
    - InMemoryCheckpointStore: HITL suspend produces a Checkpoint; the state
      round-trips through pickle; a fresh RunContext + a re-hydrated store
      completes agent.resume(...) correctly and returns a valid AgentResult.
    - Postgres path is DEFERRED (needs docker); flagged in the report.
"""

from __future__ import annotations

import asyncio
import pickle

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.adapters.llm.providers import claude as _claude
from agentkit.adapters.store import InMemoryStore
from agentkit.agents import Agent, Suspended
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities.checkpointer import Checkpointer
from agentkit.kernel.types import Scope
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.tools import ToolRegistry, tool

from .conftest import HAIKU_MODEL, MAX_TOKENS, requires_anthropic


def _run(coro):
    return asyncio.run(coro)


def _ctx(llm, *, correlation_id, checkpointer=None, store=None, autonomy="gated") -> RunContext:
    invoker = Invoker(
        llm=llm,
        chat_middleware=[tracing(), meter(), retry()],
        tool_middleware=[tracing(), meter(), retry()],
    )
    services = Services(invoker=invoker, checkpointer=checkpointer, store=store)
    return RunContext(
        correlation_id=correlation_id,
        scope=Scope(org_id=1, domain_id=1),
        budget=Budget(),
        services=services,
        autonomy=autonomy,
    )


@tool(side_effecting=True, requires_approval=True)
def send_notification(recipient: str, message: str) -> str:
    """Send a notification to the recipient. Side-effecting; requires approval."""
    return f"notified {recipient}: {message}"


@requires_anthropic
def test_checkpoint_state_survives_pickle_roundtrip_and_resumes(
    anthropic_key: str,
) -> None:
    """Serialize the checkpoint store's dict, restore it, and resume in a fresh
    RunContext. Verify the resumed run completes."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            store = InMemoryStore()
            checkpoint_port = InMemoryCheckpointStore()
            checkpointer = Checkpointer(port=checkpoint_port)
            ctx = _ctx(
                llm,
                correlation_id="ckpt-run-1",
                checkpointer=checkpointer,
                store=store,
            )
            registry = ToolRegistry.from_tools([send_notification])
            agent = Agent(
                name="notifier",
                model=HAIKU_MODEL,
                prompt=("You send notifications. When asked, call send_notification once and then reply 'ok'."),
                max_tokens=MAX_TOKENS,
                cognition=ReActCognition(tools=registry, max_iterations=3, checkpointer=checkpointer),
            )
            result = await agent.run("Notify test@example.com with the message 'ping'.", ctx)
            suspended = result.evals.get("suspended")
            if suspended is None:
                pytest.skip(f"model did not call the gated tool — evals={result.evals!r}")
            assert isinstance(suspended, Suspended)

            # Pickle the ENTIRE checkpoint dict → simulate cross-process transport
            raw = pickle.dumps(checkpoint_port._cps)
            restored_dict = pickle.loads(raw)
            assert restored_dict, "expected persisted checkpoint(s)"

            # Fresh checkpoint port + checkpointer in a FRESH RunContext
            new_port = InMemoryCheckpointStore()
            new_port._cps = restored_dict
            new_checkpointer = Checkpointer(port=new_port)
            fresh_ctx = _ctx(
                llm,
                correlation_id="ckpt-run-1",
                checkpointer=new_checkpointer,
                store=store,
            )
            # Rebuild the agent with the new checkpointer (cognition holds it)
            agent2 = Agent(
                name="notifier",
                model=HAIKU_MODEL,
                prompt=("You send notifications. When asked, call send_notification once and then reply 'ok'."),
                max_tokens=MAX_TOKENS,
                cognition=ReActCognition(
                    tools=registry,
                    max_iterations=3,
                    checkpointer=new_checkpointer,
                ),
            )
            decisions = {tc.id: "approve" for tc in suspended.pending}
            resumed = await agent2.resume("ckpt-run-1", decisions, fresh_ctx)
            assert isinstance(resumed.output, str)
            # Sanity: not still suspended after resume
            assert not resumed.evals.get("suspended")
        finally:
            await llm.aclose()

    _run(go())
