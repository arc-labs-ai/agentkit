"""Nested fan-out must not deadlock — the semaphore is per DEPTH, not per tree.

A single tree-wide `asyncio.Semaphore` cannot be both respected by nested
acquisition and deadlock-free, and the framework had the deadlocking half. A
parent's fan-out holds its permits for the ENTIRE duration of each child run,
so a nested fan-out draws from a pool its own ancestors have already drained.

Reproduced through the public API: an Agent dispatching two `as_tool`
sub-agents, each of which dispatches its own tools, hangs forever at
`max_concurrency=2`. Every nesting boundary in the framework goes through
`ctx.child()`, so keying the pool on depth breaks the cycle structurally — an
ancestor at depth d can only hold permits from pool d, and its children draw
from pool d+1.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit import Agent, Budget, Scope, as_tool
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.concurrency import gather_bounded
from agentkit.kernel.types import Delta, ToolCall, Usage
from agentkit.runtime.context import RunContext
from agentkit.testing import make_test_ctx
from agentkit.tools import tool


@tool(side_effecting=False)
def leaf(q: str) -> str:
    """A trivial read-only leaf tool that echoes its query back to the caller."""
    return "leaf:" + q


class _FanOutLLM:
    """Stateless by design: decides from the OFFERED TOOLS and the transcript,
    so every agent in the tree behaves the same regardless of call order. A
    stateful fake silently gives inner agents the 'done' turn and the
    deadlock never reproduces."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, *, messages, tools=None, **_kw):
        self.calls += 1
        offered = {t.name for t in (tools or ())}
        already_ran = any(getattr(m, "role", "") == "tool" for m in messages)
        pick = "sub" if "sub" in offered else ("leaf" if "leaf" in offered else None)
        if pick and not already_ran:
            args = {"task": "go"} if pick == "sub" else {"q": "x"}
            yield Delta(text="working", model="m", provider="f")
            yield Delta(
                tool_calls=(ToolCall("a", pick, args), ToolCall("b", pick, args)),
                usage=Usage(1, 1, 0.0),
                finish_reason="tool_calls",
                model="m",
                provider="f",
            )
        else:
            yield Delta(text="done", model="m", provider="f")
            yield Delta(usage=Usage(1, 1, 0.0), finish_reason="stop", model="m", provider="f")


def _nested_agent() -> Agent:
    inner = Agent("inner", "m", cognition=ReActCognition(tools=[leaf]))
    sub = as_tool(inner, name="sub", description="Delegate a sub-task to the inner agent.")
    return Agent("outer", "m", cognition=ReActCognition(tools=[sub]))


@pytest.mark.parametrize("cap", [1, 2, 3, 8])
def test_nested_fan_out_completes_at_every_concurrency_cap(cap: int) -> None:
    """The regression. At cap=2 this hung forever: the outer agent's two
    `as_tool` permits are held until the inner runs finish, and the inner runs
    cannot start without a permit from the same pool."""
    outer = _nested_agent()
    ctx = make_test_ctx(
        llm=_FanOutLLM(), budget=Budget(max_concurrency=cap, max_depth=8)
    )

    async def go():
        # A real deadlock hangs forever; the timeout turns that into a failure
        # rather than a stalled CI job.
        return await asyncio.wait_for(outer.run("go", ctx), timeout=10)

    assert asyncio.run(go()).stop_reason == "complete"


def test_the_per_level_cap_is_still_enforced() -> None:
    """Fixing the deadlock must not mean removing the bound. One depth's pool
    still admits at most `max_concurrency` at a time."""
    budget = Budget(max_concurrency=3)
    ctx = RunContext("r", Scope(), budget)
    live = peak = 0

    async def work() -> None:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1

    asyncio.run(gather_bounded([work() for _ in range(20)], sem=ctx.semaphore()))
    assert peak <= 3, f"cap 3 admitted {peak} concurrently"
    assert peak == 3, "the cap should actually be reached, or this proves nothing"


def test_each_depth_gets_its_own_pool() -> None:
    """The mechanism, asserted directly: distinct depths must not share a
    semaphore, and the same depth must share one (or the cap is per-agent,
    not per-level)."""
    budget = Budget(max_concurrency=4)
    root = RunContext("r", Scope(), budget)
    child = root.child()
    grandchild = child.child()

    assert root.semaphore() is not child.semaphore()
    assert child.semaphore() is not grandchild.semaphore()
    # Two contexts at the SAME depth share the pool.
    assert root.semaphore() is RunContext("other", Scope(), budget).semaphore()
    assert sorted(budget._sems) == [0, 1, 2]


def test_the_documented_worst_case_bound_holds() -> None:
    """The honest trade: the bound is per level, so worst-case in-flight work
    is `max_concurrency * (max_depth + 1)`. Pinned so the docstring's claim
    cannot drift from the code."""
    budget = Budget(max_concurrency=5, max_depth=3)
    ctx = RunContext("r", Scope(), budget)
    for _ in range(3):
        ctx = ctx.child()
    assert len(budget._sems) == 0  # nothing created until asked for
    root = RunContext("r", Scope(), budget)
    seen = root
    for _ in range(3):
        seen.semaphore()
        seen = seen.child()
    seen.semaphore()
    assert len(budget._sems) == budget.max_depth + 1


# ── cooperative cancellation must not be isolated into a failure slot ────────


def test_best_effort_propagates_cooperative_cancellation() -> None:
    """`gather_best_effort` re-raised `asyncio.CancelledError` but caught
    agentkit's own `Cancelled` under `except Exception`, turning it into a
    `Failure` slot.

    The token is SHARED across the run tree, so a tripped token means every
    sibling is about to raise it too — the caller received N independent
    "failures" with no way to distinguish an aborted run from a batch where
    everything happened to break at once. `Cancelled` is documented as an
    ABORT, distinct from a graceful TerminationCondition and from
    MeterExceeded, so it must abort.
    """
    from agentkit.kernel.concurrency import CancellationToken, Cancelled, gather_best_effort

    token = CancellationToken()
    token.cancel()

    async def child(i: int) -> int:
        token.raise_if_cancelled()  # what every pattern does at a safe point
        return i

    async def go():
        return await gather_best_effort(
            [child(0), child(1)], sem=asyncio.Semaphore(4)
        )

    with pytest.raises(Cancelled):
        asyncio.run(go())


def test_best_effort_still_isolates_an_ordinary_failure() -> None:
    """The behaviour that must NOT change: one child raising a normal
    exception is isolated into its slot as first-class data."""
    from agentkit.kernel.concurrency import gather_best_effort
    from agentkit.kernel.errors import Failure

    async def ok() -> str:
        return "fine"

    async def boom() -> str:
        raise ValueError("provider hiccup")

    async def go():
        return await gather_best_effort([ok(), boom()], sem=asyncio.Semaphore(4))

    results = asyncio.run(go())
    assert results[0] == "fine"
    assert isinstance(results[1], Failure)
    assert isinstance(results[1].cause, ValueError)
