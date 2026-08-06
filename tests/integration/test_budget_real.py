"""E2E tests for Budget / Quota / MeterExceeded against real Claude.

Covers:
    - Budget(max_cost_usd=…) → MeterExceeded raises cleanly, agent unwinds.
    - Budget(max_calls=1) with a ReAct loop that would take multiple calls →
      the loop halts and returns partial=True on the AgentResult.
    - Quota(max_rpm=…) — exhaust a rolling per-tenant cap; subsequent calls fail.
"""

from __future__ import annotations

import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import claude as _claude
from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.types import Scope
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Budget, Invoker, MeterExceeded, Quota, RunContext, Services
from agentkit.tools import ToolRegistry, tool

from .conftest import HAIKU_MODEL, MAX_TOKENS, requires_anthropic


def _run(coro):
    return asyncio.run(coro)


def _ctx(llm, *, budget: Budget | None = None, meters=()) -> RunContext:
    invoker = Invoker(
        llm=llm,
        chat_middleware=[tracing(), meter(), retry()],
        tool_middleware=[tracing(), meter(), retry()],
    )
    return RunContext(
        correlation_id="budget-run",
        scope=Scope(org_id=1, domain_id=1),
        budget=budget or Budget(),
        services=Services(invoker=invoker),
        meters=list(meters),
    )


@requires_anthropic
def test_budget_max_cost_usd_triggers_meter_exceeded(anthropic_key: str) -> None:
    """A tiny cost cap (1e-12 USD) will trip after the first call charges. The
    agent's second attempt (or the initial guard) raises MeterExceeded cleanly.
    We use a ReAct loop to force at least 2 calls."""

    _CALLS: list[str] = []

    @tool(side_effecting=False)
    def note(text: str) -> str:
        """Record a short note to the persistent notepad. Used for testing budget guards."""
        _CALLS.append(text)
        return "ok"

    async def go() -> None:
        # We inject a `pricing=` callable that always returns a positive cost so
        # this test doesn't depend on the built-in pricing table matching the
        # exact served model name. (There IS a real bug where the served model
        # name includes a date suffix (`claude-haiku-4-5-20251001`) and the
        # bundled pricing table only has the bare name, so cost silently comes
        # out as 0 — flagged separately in the report. Injecting `pricing=`
        # here isolates the Budget guard test from that pricing lookup miss.)
        def _fixed_price(model, usage):
            return 0.001  # any positive cost > max_cost_usd

        from agentkit.adapters.llm.providers import AnthropicLLM

        llm = AnthropicLLM(api_key=anthropic_key, default_model=HAIKU_MODEL, pricing=_fixed_price)
        try:
            budget = Budget(max_cost_usd=1e-12)  # any real call blows past this
            ctx = _ctx(llm, budget=budget)
            agent = Agent(
                name="over-budget",
                model=HAIKU_MODEL,
                prompt="Reply with a single word.",
                max_tokens=MAX_TOKENS,
                cognition=ReActCognition(tools=ToolRegistry.from_tools([note]), max_iterations=3),
            )
            with pytest.raises(MeterExceeded):
                await agent.run("Please say hi.", ctx)
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_budget_max_calls_halts_loop_partial_true(anthropic_key: str) -> None:
    """max_calls=1 with a ReAct loop that expects multiple calls → the loop
    stops after the meter fires. Either raises MeterExceeded or completes
    with partial=True — both are acceptable per the meter contract."""

    _CALLS: list[str] = []

    @tool(side_effecting=False)
    def note(text: str) -> str:
        """Record a short note. Always call twice for testing."""
        _CALLS.append(text)
        return "ok"

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            budget = Budget(max_calls=1)
            ctx = _ctx(llm, budget=budget)
            agent = Agent(
                name="one-call",
                model=HAIKU_MODEL,
                prompt="You must call the note tool twice, then reply 'done'.",
                max_tokens=MAX_TOKENS,
                cognition=ReActCognition(tools=ToolRegistry.from_tools([note]), max_iterations=4),
            )
            try:
                result = await agent.run("Please note 'a' then note 'b'.", ctx)
                # If it completes, must be partial (loop halted)
                assert result.partial or budget.calls <= 1, (
                    f"expected partial=True or budget.calls≤1, got partial={result.partial} calls={budget.calls}"
                )
            except MeterExceeded:
                # Also acceptable — the meter fired before the 2nd call
                pass
            # Budget.calls should have moved off zero (at least one real call)
            assert budget.calls >= 1
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_quota_max_rpm_exhausted_raises(anthropic_key: str) -> None:
    """Set Quota(max_rpm=1). First call passes; second on the same scope in
    the same window raises MeterExceeded."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            quota = Quota(max_rpm=1)
            ctx = _ctx(llm, meters=[quota])
            agent = Agent(
                name="quota",
                model=HAIKU_MODEL,
                prompt="Reply with the word 'ok'. Nothing else.",
                max_tokens=MAX_TOKENS,
            )
            # First call — should succeed
            res = await agent.run("Say ok.", ctx)
            assert res.output
            # Second call — should trip the RPM cap
            with pytest.raises(MeterExceeded):
                await agent.run("Say ok again.", ctx)
        finally:
            await llm.aclose()

    _run(go())
