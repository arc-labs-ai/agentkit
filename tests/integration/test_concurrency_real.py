"""E2E concurrency + cancellation tests against real Claude.

Covers:
    - 8 concurrent agent.run(...) with tiny prompts → all complete + Budget
      metering is accurate (no double count).
    - CancellationToken cancels ONE in-flight run without disturbing the
      siblings.
"""

from __future__ import annotations

import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import claude as _claude
from agentkit.agents import Agent
from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.types import Scope
from agentkit.middlewares import meter, retry, tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services

from .conftest import HAIKU_MODEL, MAX_TOKENS, requires_anthropic


def _run(coro):
    return asyncio.run(coro)


@requires_anthropic
def test_eight_concurrent_agents_all_complete_with_shared_budget(anthropic_key: str) -> None:
    """Fan out 8 concurrent tiny requests against ONE shared LLM client + budget.
    Verify all complete, budget.calls == 8, no double counting."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            budget = Budget()
            invoker = Invoker(
                llm=llm,
                chat_middleware=[tracing(), meter(), retry()],
                tool_middleware=[tracing(), meter(), retry()],
            )
            services = Services(invoker=invoker)
            # ONE shared RunContext (budget is per-tree)
            ctx = RunContext(
                correlation_id="fanout",
                scope=Scope(org_id=1, domain_id=1),
                budget=budget,
                services=services,
            )
            agent = Agent(
                name="fan",
                model=HAIKU_MODEL,
                prompt="Reply with a single number.",
                max_tokens=50,
            )
            tasks = [agent.run(f"Return the number {i}. Just the digit.", ctx) for i in range(8)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 8
            for r in results:
                assert r.output, "expected a non-empty output"
            # Budget should have counted exactly 8 successful chat calls
            assert budget.calls == 8, f"expected 8 metered calls, got {budget.calls}"
        finally:
            await llm.aclose()

    _run(go())


@requires_anthropic
def test_cancellation_token_stops_one_run_without_disturbing_siblings(
    anthropic_key: str,
) -> None:
    """Two concurrent runs; cancel one via a private token; the other completes."""

    async def go() -> None:
        llm = _claude(api_key=anthropic_key, model=HAIKU_MODEL)
        try:
            invoker = Invoker(
                llm=llm,
                chat_middleware=[tracing(), meter(), retry()],
                tool_middleware=[tracing(), meter(), retry()],
            )

            def make_ctx(tok):
                return RunContext(
                    correlation_id="cx",
                    scope=Scope(org_id=1, domain_id=1),
                    budget=Budget(),
                    services=Services(invoker=invoker),
                    cancel=tok,
                )

            tok_a = CancellationToken()
            tok_b = CancellationToken()
            ctx_a = make_ctx(tok_a)
            ctx_b = make_ctx(tok_b)
            agent = Agent(
                name="two",
                model=HAIKU_MODEL,
                prompt="Reply with any short greeting.",
                max_tokens=MAX_TOKENS,
            )

            # Fire A, then flip A's cancel token immediately, then start B.
            # Cancellation is cooperative — the ReAct-loop-less SingleCall
            # cognition checks ``ctx.check_cancelled()`` before each chat
            # request. Cancelling before await gathers means A raises
            # Cancelled at its first check.
            tok_a.cancel()
            task_a = asyncio.create_task(agent.run("hi A", ctx_a))
            task_b = asyncio.create_task(agent.run("hi B", ctx_b))
            done_a, done_b = await asyncio.gather(task_a, task_b, return_exceptions=True)
            # A was cancelled — should have raised Cancelled (or CancelledError)
            assert isinstance(done_a, (Cancelled, asyncio.CancelledError)), (
                f"expected Cancelled for A, got {type(done_a).__name__}: {done_a!r}"
            )
            # B should have completed
            assert not isinstance(done_b, BaseException), f"B should not have raised, got {done_b!r}"
            assert done_b.output
        finally:
            await llm.aclose()

    _run(go())
