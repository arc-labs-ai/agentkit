"""`run_agents`' ActorBudget slicing and settlement — previously untested.

This was the largest uncovered block in `kernel/concurrency.py` (~50 lines of
money handling at 57% module coverage), and it only started mattering once the
envelope was actually charged: while `ActorBudget` was inert, reserving and
releasing slices with zero usage was unobservable either way.

Two properties matter here and they pull in opposite directions:

* A fan-out must not silently produce no-op children. A zero-sized slice
  reserves nothing, "succeeds", and hands the child an already-exhausted
  envelope — a fan-out that appears to run and does nothing.
* A fan-out must not over-grant. Handing a child more than was reserved lets it
  spend on an axis the parent never committed, and `settle_child` caps usage at
  the reservation, so that spend is invisible on the parent's books.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from _assertions import assert_money

from agentkit import Agent, Budget
from agentkit.agents.control.budget import ActorBudget, BudgetExhausted
from agentkit.kernel.concurrency import run_agents
from agentkit.kernel.types import Delta, Usage
from agentkit.middlewares import meter
from agentkit.testing import make_test_ctx


class _Spend:
    """Charges a fixed amount per call, so slices and settlement are checkable
    against arithmetic rather than against a fake's mood."""

    def __init__(self, cost: float = 0.10, tokens: int = 100) -> None:
        self.cost = cost
        self.tokens = tokens
        self.calls = 0

    async def stream(self, **_kw):
        self.calls += 1
        yield Delta(text="ok", model="m", provider="f")
        yield Delta(
            usage=Usage(self.tokens, 10, self.cost),
            finish_reason="stop",
            model="m",
            provider="f",
        )


def _ctx_with(parent: ActorBudget, llm: _Spend):
    ctx = make_test_ctx(llm=llm, budget=Budget(), chat_middleware=[meter()])
    ctx.actor_budget = parent
    return ctx


def _pairs(n: int):
    return [(Agent(f"c{i}", "m"), "go") for i in range(n)]


# ── the happy round trip ─────────────────────────────────────────────────────


def test_child_spend_rolls_up_to_the_parent_envelope() -> None:
    """Each child charges its OWN slice (the child ctx carries its own
    ActorBudget), and `settle_child` rolls the actual usage onto the parent's
    books when the reservation is released."""
    parent = ActorBudget(
        max_tokens=100_000, max_cost_usd="1.00", max_steps=100, max_wall_seconds=1e9
    )
    llm = _Spend(cost=0.10, tokens=100)

    asyncio.run(run_agents(_pairs(3), _ctx_with(parent, llm)))

    assert llm.calls == 3
    assert_money(parent.used_cost(), "0.30", label="3 children x $0.10")
    assert_money(parent.reserved_cost(), "0", label="every reservation settled")
    assert parent.used_tokens == 330  # (100 in + 10 out) x 3
    assert parent.used_steps == 3, "one model call is one step"


def test_reservations_are_released_even_when_a_child_raises() -> None:
    """The settlement runs in a `finally`, so a failed fan-out cannot leave the
    parent's `reserved_*` permanently held — which would shrink every later
    slice for the rest of the run."""

    class _Boom:
        async def stream(self, **_kw):
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover — makes this an async generator

    parent = ActorBudget(
        max_tokens=100_000, max_cost_usd="1.00", max_steps=100, max_wall_seconds=1e9
    )
    ctx = make_test_ctx(llm=_Boom(), budget=Budget(), chat_middleware=[meter()])
    ctx.actor_budget = parent

    with pytest.raises(BaseException):  # noqa: B017 — TaskGroup raises an ExceptionGroup
        asyncio.run(run_agents(_pairs(2), ctx))

    assert_money(parent.reserved_cost(), "0", label="reservations released on failure")
    assert parent.reserved_tokens == 0 and parent.reserved_steps == 0


def test_best_effort_still_settles_every_slice() -> None:
    """Same guarantee on the isolating path, where failures become data."""
    from agentkit.kernel.errors import Failure

    class _HalfBroken:
        def __init__(self) -> None:
            self.n = 0

        async def stream(self, **_kw):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("first child fails")
            yield Delta(text="ok", model="m", provider="f")
            yield Delta(usage=Usage(100, 10, 0.10), finish_reason="stop", model="m", provider="f")

    parent = ActorBudget(
        max_tokens=100_000, max_cost_usd="1.00", max_steps=100, max_wall_seconds=1e9
    )
    ctx = make_test_ctx(llm=_HalfBroken(), budget=Budget(), chat_middleware=[meter()])
    ctx.actor_budget = parent

    out = asyncio.run(run_agents(_pairs(2), ctx, best_effort=True))
    assert any(isinstance(o, Failure) for o in out)
    assert_money(parent.reserved_cost(), "0", label="settled despite an isolated failure")


# ── starvation fails fast on EVERY axis ──────────────────────────────────────


@pytest.mark.parametrize(
    ("caps", "axis"),
    [
        ({"max_tokens": 100_000, "max_cost_usd": "1.00", "max_steps": 2}, "steps"),
        ({"max_tokens": 2, "max_cost_usd": "1.00", "max_steps": 100}, "tokens"),
        ({"max_tokens": 100_000, "max_cost_usd": "0.000001", "max_steps": 100}, "cost_usd"),
    ],
)
def test_a_starved_axis_fails_fast_before_any_child_runs(caps: dict, axis: str) -> None:
    """Only `steps` used to do this. `tokens` and `cost` floored their slice to
    zero, so a fan-out of 3 against 2 tokens produced three children that each
    stopped on their first pre-flight — a fan-out that looked like it ran and
    did nothing, on two of the three axes.

    Fail-fast is the documented intent, and now every axis honours it, naming
    the axis that ran out.
    """
    parent = ActorBudget(max_wall_seconds=1e9, **caps)
    llm = _Spend()

    with pytest.raises(BudgetExhausted) as exc:
        asyncio.run(run_agents(_pairs(3), _ctx_with(parent, llm)))

    assert exc.value.axis == axis
    assert llm.calls == 0, "a starved fan-out still made a provider call"
    # And the books are clean: nothing left reserved by the aborted carve.
    assert parent.reserved_tokens == 0 and parent.reserved_steps == 0
    assert_money(parent.reserved_cost(), "0", label="aborted carve released")


def test_an_already_exhausted_envelope_refuses_to_fan_out() -> None:
    """A spent envelope cannot be carved at all. Without this, every axis
    yields zero-sized slices, each reservation of zero "succeeds", and the
    caller gets N children that stop immediately — the silent no-op fan-out
    again, by a different route."""
    parent = ActorBudget(
        max_tokens=100, max_cost_usd="1.00", max_steps=10, max_wall_seconds=1e9
    )
    parent.charge(tokens=100)  # tokens axis spent
    assert parent.exhausted() is True

    llm = _Spend()
    with pytest.raises(BudgetExhausted) as exc:
        asyncio.run(run_agents(_pairs(2), _ctx_with(parent, llm)))
    assert exc.value.axis == "tokens"
    assert "already" in str(exc.value)
    assert llm.calls == 0


# ── the slice arithmetic ─────────────────────────────────────────────────────


class _RecordingActorBudget(ActorBudget):
    """Records what each reservation asked for.

    A subclass rather than a monkeypatched instance because `ActorBudget` uses
    `__slots__`, so instance attributes cannot be assigned — which is also why
    it needs its own slot for the log.
    """

    __slots__ = ("reservations",)

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.reservations: list[tuple[int, object, int]] = []

    def reserve_for_child(self, *, tokens, cost_usd, steps) -> None:  # type: ignore[override]
        self.reservations.append((tokens, cost_usd, steps))
        super().reserve_for_child(tokens=tokens, cost_usd=cost_usd, steps=steps)


def test_slices_are_equal_and_exact() -> None:
    """Equal shares, so the order reservations happen in cannot skew fairness,
    and computed in `Decimal` off `remaining_cost()` rather than through the
    float mirror."""
    parent = _RecordingActorBudget(
        max_tokens=99, max_cost_usd="0.90", max_steps=9, max_wall_seconds=1e9
    )
    asyncio.run(run_agents(_pairs(3), _ctx_with(parent, _Spend(cost=0.01, tokens=1))))

    assert len(parent.reservations) == 3
    assert len(set(parent.reservations)) == 1, f"slices differ: {parent.reservations}"
    tokens, cost, steps = parent.reservations[0]
    assert tokens == 33 and steps == 3
    assert isinstance(cost, Decimal), "the money slice must not round-trip through float"
    assert_money(cost, "0.30", label="0.90 / 3")


def test_a_child_is_granted_exactly_what_was_reserved() -> None:
    """The step axis used to be granted `max(slice_steps, 1)`, so a child could
    take a step the parent never committed. `settle_child` caps usage at the
    reservation, which made that spend invisible on the parent's books."""
    parent = _RecordingActorBudget(
        max_tokens=90, max_cost_usd="0.90", max_steps=9, max_wall_seconds=1e9
    )
    asyncio.run(run_agents(_pairs(3), _ctx_with(parent, _Spend(cost=0.01, tokens=1))))

    assert [r[2] for r in parent.reservations] == [3, 3, 3]
    # The parent committed 9 steps and the children used 3 (one call each).
    assert parent.used_steps == 3
    assert parent.reserved_steps == 0


# ── run_sync: the sync bridge's nested-loop branch ───────────────────────────


def test_run_sync_from_a_thread_with_no_running_loop() -> None:
    """The straightforward branch: no loop in this thread, so `asyncio.run`."""
    from agentkit.kernel.concurrency import run_sync

    async def work() -> str:
        await asyncio.sleep(0)
        return "done"

    assert run_sync(work()) == "done"


def test_run_sync_nested_inside_a_running_loop() -> None:
    """The branch that was untested, and the one that actually earns its keep.

    A sync host calling into agentkit from inside an async caller would hit
    "asyncio.run() cannot be called from a running event loop". `run_sync`
    detects the running loop and hands the coroutine to a fresh worker thread
    with its own loop, keeping a blocking signature. Untested, this is the kind
    of path that quietly regresses into a deadlock.
    """
    from agentkit.kernel.concurrency import run_sync

    async def inner() -> str:
        await asyncio.sleep(0)
        return "from a nested loop"

    async def outer() -> str:
        # A loop IS running here, so run_sync must take the worker-thread path.
        return await asyncio.to_thread(run_sync, inner())

    assert asyncio.run(outer()) == "from a nested loop"


def test_run_sync_propagates_the_exception_from_a_nested_loop() -> None:
    """A blocking bridge that swallowed failures would be worse than no
    bridge — the caller would see `None` instead of the error."""
    from agentkit.kernel.concurrency import run_sync

    async def boom() -> None:
        raise ValueError("inner failure")

    async def outer() -> None:
        await asyncio.to_thread(run_sync, boom())

    with pytest.raises(ValueError, match="inner failure"):
        asyncio.run(outer())


def test_a_single_child_slice_is_the_parents_exact_remaining() -> None:
    """Kills: computing the slice off the float MIRROR instead of the ledger.

    Dividing by N and quantizing at six decimals absorbs the float error, so
    even a large balance split three ways agrees. A fan-out of ONE leaves no
    division to absorb it:

        exact:      10000000000.000001
        via float:  10000000000.000002

    A fan-out of one is not a contrived case — it is what a coordinator does
    when its policy selects a single child for a turn.
    """
    parent = _RecordingActorBudget(
        max_tokens=10**12,
        max_cost_usd="10000000000.000001",
        max_steps=10**6,
        max_wall_seconds=1e9,
    )
    asyncio.run(run_agents(_pairs(1), _ctx_with(parent, _Spend(cost=0.01, tokens=1))))

    assert len(parent.reservations) == 1
    _, cost, _ = parent.reservations[0]
    assert_money(cost, "10000000000.000001", label="single-child slice")


def test_a_child_cannot_take_more_steps_than_its_slice() -> None:
    """Kills: over-granting the step axis.

    The child used to be handed `max(slice_steps, 1)`, so with a one-step slice
    it could take a step the parent never reserved — and `settle_child` caps
    usage at the reservation, making that spend invisible on the parent's
    books.

    A one-call child cannot show this: 3 granted vs 4 granted is
    indistinguishable when only 1 is used. So the child here is a tool loop
    that WANTS two calls, against a slice of exactly one step.
    """
    from agentkit.agents.cognition import ReActCognition
    from agentkit.kernel.types import ToolCall
    from agentkit.tools import tool

    @tool(side_effecting=False)
    def probe(q: str) -> str:
        """A read-only tool so the loop has a reason to take a second turn."""
        return "ok"

    class _WantsTwoTurns:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, *, messages, **_kw):
            self.calls += 1
            ran = any(getattr(m, "role", "") == "tool" for m in messages)
            if not ran:
                yield Delta(text="calling a tool", model="m", provider="f")
                yield Delta(
                    tool_calls=(ToolCall("t1", "probe", {"q": "x"}),),
                    usage=Usage(10, 1, 0.001),
                    finish_reason="tool_calls",
                    model="m",
                    provider="f",
                )
            else:
                yield Delta(text="done", model="m", provider="f")
                yield Delta(
                    usage=Usage(10, 1, 0.001), finish_reason="stop", model="m", provider="f"
                )

    # 1 step per child: enough for the first call, not the second.
    parent = ActorBudget(
        max_tokens=100_000, max_cost_usd="1.00", max_steps=1, max_wall_seconds=1e9
    )
    llm = _WantsTwoTurns()
    ctx = make_test_ctx(llm=llm, budget=Budget(), chat_middleware=[meter()])
    ctx.actor_budget = parent
    pairs = [(Agent("c0", "m", cognition=ReActCognition(tools=[probe])), "go")]

    out = asyncio.run(run_agents(pairs, ctx))

    assert llm.calls == 1, f"the child exceeded its one-step slice ({llm.calls} calls)"
    assert out[0].stop_reason == "budget_exhausted"
    assert "steps" in out[0].evals["error"]
