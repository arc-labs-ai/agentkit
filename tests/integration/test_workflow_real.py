"""E2E tests for `Workflow` composition against real Claude.

Covers:
    - Linear three-node workflow (agent_a → transform fn → agent_b): outputs accumulate,
      usage merges, stop_reason == "complete".
    - Middle node raises: WorkflowResult surfaces failure, downstream nodes don't run.
    - Human gate + resume: Suspended surfaces, resume() picks up cleanly.
"""

from __future__ import annotations

import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.adapters.llm.providers import claude as _claude
from agentkit.adapters.store import InMemoryStore
from agentkit.agents import Agent, Workflow
from agentkit.agents.result import Suspended
from agentkit.capabilities.checkpointer import Checkpointer
from agentkit.kernel.types import Scope
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services

from .conftest import HAIKU_MODEL, MAX_TOKENS, requires_anthropic


def _run(coro):
    return asyncio.run(coro)


def _ctx_with_llm(llm, *, correlation_id="wf-run", store=None, checkpointer=None) -> RunContext:
    invoker = Invoker(
        llm=llm,
        chat_middleware=[tracing(), meter(), retry()],
        tool_middleware=[tracing(), meter(), retry()],
    )
    services = Services(invoker=invoker, store=store, checkpointer=checkpointer)
    return RunContext(
        correlation_id=correlation_id,
        scope=Scope(org_id=1, domain_id=1),
        budget=Budget(),
        services=services,
    )


@requires_anthropic
def test_workflow_three_node_linear_happy_path(anthropic_key: str) -> None:
    """agent_a → fn (transform) → agent_b. Verify outputs dict + merged usage."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            ctx = _ctx_with_llm(llm)
            agent_a = Agent(
                name="pick-color",
                model=HAIKU_MODEL,
                prompt="Reply with a single color word. Nothing else.",
                max_tokens=MAX_TOKENS,
            )
            agent_b = Agent(
                name="describe-color",
                model=HAIKU_MODEL,
                prompt="Write ONE short sentence about the color you were given.",
                max_tokens=MAX_TOKENS,
            )
            wf = Workflow("color-flow")
            wf.agent("pick", agent_a)

            def transform(inputs: dict, goal: str) -> str:
                # Capitalize the picked color
                picked = str(inputs.get("pick", "")).strip()
                return picked.upper()

            wf.fn("upper", transform, after="pick")
            wf.agent("describe", agent_b, after="upper")

            result = await wf.run("Pick a color and describe it.", ctx)
            assert result.stop_reason == "complete"
            assert "pick" in result.outputs
            assert "upper" in result.outputs
            assert "describe" in result.outputs
            assert result.outputs["upper"] == str(result.outputs["pick"]).strip().upper()
            assert result.usage.output_tokens > 0
            assert result.steps >= 3
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_workflow_middle_node_raises_downstream_stops(anthropic_key: str) -> None:
    """A raising fn node aborts the workflow before downstream nodes run.

    Workflow does not currently swallow node exceptions (contract: fatal). Verify
    it propagates and downstream nodes are NOT invoked."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            ctx = _ctx_with_llm(llm)
            agent_a = Agent(
                name="pick",
                model=HAIKU_MODEL,
                prompt="Reply with a single color word.",
                max_tokens=MAX_TOKENS,
            )
            downstream_ran = {"v": False}

            def bad_transform(inputs: dict) -> str:
                raise ValueError("bad transform")

            def downstream(inputs: dict) -> str:
                downstream_ran["v"] = True
                return "downstream"

            wf = Workflow("bad-flow")
            wf.agent("pick", agent_a)
            wf.fn("bad", bad_transform, after="pick")
            wf.fn("down", downstream, after="bad")

            with pytest.raises(Exception) as excinfo:
                await wf.run("go", ctx)
            # ValueError propagates (possibly wrapped by the concurrency layer)
            assert (
                "bad transform" in str(excinfo.value).lower()
                or isinstance(excinfo.value, ValueError)
                or excinfo.type.__name__ in {"ExceptionGroup", "BaseExceptionGroup"}
            )
            assert downstream_ran["v"] is False
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_workflow_human_gate_suspends_and_resumes(anthropic_key: str) -> None:
    """Wire a human_gate between two agent nodes. Verify the workflow suspends,
    then resume(...) with a decision continues to the last node."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            store = InMemoryStore()
            checkpoint_port = InMemoryCheckpointStore()
            checkpointer = Checkpointer(port=checkpoint_port)
            ctx = _ctx_with_llm(
                llm,
                correlation_id="wf-hitl-1",
                store=store,
                checkpointer=checkpointer,
            )
            agent_a = Agent(
                name="ask",
                model=HAIKU_MODEL,
                prompt="Reply with the word 'apple'. Nothing else.",
                max_tokens=MAX_TOKENS,
            )
            agent_b = Agent(
                name="say",
                model=HAIKU_MODEL,
                prompt="Repeat the fruit you were given in one word.",
                max_tokens=MAX_TOKENS,
            )
            wf = Workflow("gated-flow")
            wf.agent("ask", agent_a)
            wf.human_gate("gate", after="ask")
            wf.agent("say", agent_b, after="gate")

            result = await wf.run("go", ctx)
            assert result.stop_reason == "suspended", (
                f"expected 'suspended' stop reason, got {result.stop_reason!r} (outputs={result.outputs!r})"
            )
            assert result.suspended is not None
            assert isinstance(result.suspended, Suspended)

            # Approve the gate — decisions are keyed by gate name for workflow gates
            ctx2 = _ctx_with_llm(
                llm,
                correlation_id="wf-hitl-1",
                store=store,
                checkpointer=checkpointer,
            )
            resumed = await wf.resume("wf-hitl-1", {"gate": "approve"}, ctx2)
            assert resumed.stop_reason == "complete", f"resume should complete, got {resumed.stop_reason!r}"
            assert "say" in resumed.outputs
        finally:
            await llm.aclose()

    _run(go())
