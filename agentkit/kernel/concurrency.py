"""Structured concurrency for the agent tree — bounded fan-out of sub-agents / tools / LLM calls.

Uses `asyncio.TaskGroup` (a failing child cancels its siblings and raises an ExceptionGroup) bounded
by a shared `asyncio.Semaphore`, and copies the current `contextvars` into each task so the active
trace span / correlation id propagate. `gather_best_effort` is the opt-in "one failure shouldn't sink
the batch" variant. Stdlib-only (asyncio + contextvars); needs Python ≥ 3.11 for TaskGroup.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
from collections.abc import Awaitable, Coroutine
from decimal import Decimal
from typing import Any, TypeVar

from agentkit.kernel.errors import Failure

R = TypeVar("R")

def _at_least_one(units: int) -> int:
    """Floor-divided slice, never zero.

    A zero-sized slice reserves nothing, so the reservation "succeeds" and the
    child is handed an already-exhausted envelope — a fan-out that appears to
    run and does nothing. Forcing one unit makes N children against fewer than
    N units over-commit, which is what surfaces as ``BudgetExhausted`` before
    any child starts.
    """
    return max(1, units)


class _NeverRaised(Exception):
    """Never raised anywhere. Exists so the mutation catalogue can neutralise
    an `except` clause by retargeting it, producing a real behavioural change
    rather than a NameError that any test would "catch" for the wrong reason."""


class Cancelled(RuntimeError):
    """Raised by a cooperative cancellation check when the run (or its subtree) has been cancelled."""


class CancellationToken:
    """A cooperative cancellation signal shared across a run tree (via `RunContext`).

    A caller/parent calls `cancel()`; patterns check the token at safe points (loop top, between steps
    or tool calls) and `raise_if_cancelled()` → `Cancelled`. This is an **abort** — distinct from a
    graceful `TerminationCondition` (which returns a `Stop`) and from a resource ceiling (`MeterExceeded`).
    Shared down the tree, so cancelling a parent cancels its children.
    """

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise Cancelled("run cancelled")


def run_sync(coro: Coroutine[Any, Any, R]) -> R:
    """The single sync bridge for sync hosts to drive async agentkit.

    If no event loop is running in this thread → `asyncio.run(coro)`. If a loop IS already running
    (e.g. nested inside an async caller), spin up a fresh worker thread with its own loop and run the
    coroutine there to completion — avoids "loop already running" while keeping a blocking signature.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


async def gather_bounded(coros: list[Awaitable[R]], *, sem: asyncio.Semaphore) -> list[R]:
    """Run coroutines concurrently, at most `sem._value` at once, preserving input order.
    A failure cancels the rest (structured concurrency) and surfaces as an ExceptionGroup."""

    async def _run(coro: Awaitable[R]) -> R:
        async with sem:
            return await coro

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_run(c), context=contextvars.copy_context()) for c in coros]
    return [t.result() for t in tasks]


async def gather_best_effort(
    coros: list[Awaitable[R]], *, sem: asyncio.Semaphore
) -> list[R | Failure]:
    """Best-effort fan-out: bound concurrency, isolate failures (each slot is a result OR a
    :class:`~agentkit.kernel.errors.Failure` wrapping the raised exception). Use when one
    sub-agent/tool failing must NOT cancel the others.

    Wrapping the exception in a ``Failure`` (rather than returning the raw exception object)
    lets a parent treat the failed slot as **first-class data**: it carries a classified
    category, a ``source`` naming the slot (``gather_best_effort[i]``), and the originating
    ``cause``, so a caller can retry / route around / escalate uniformly. It also disambiguates
    "the coroutine returned an ``Exception`` value" from "the coroutine raised" — the raw-
    exception design conflated the two.
    """

    async def _run(idx: int, coro: Awaitable[R]) -> R | Failure:
        async with sem:
            try:
                return await coro
            except asyncio.CancelledError:  # never swallow cancellation — let it unwind the gather
                raise
            except Cancelled:
                # agentkit's own COOPERATIVE cancellation. Isolating this into
                # a Failure slot was wrong: the token is shared across the run
                # tree, so a tripped token means every sibling is about to
                # raise it too, and the caller would receive N independent
                # "failures" with no way to tell an aborted run from a batch
                # where everything happened to break at once. An abort aborts
                # — that is the documented distinction between ``Cancelled``,
                # a graceful ``TerminationCondition``, and ``MeterExceeded``.
                raise
            except Exception as exc:  # noqa: BLE001 — a normal failure is isolated into the slot
                return Failure.of(exc, source=f"gather_best_effort[{idx}]")

    return await asyncio.gather(*(_run(i, c) for i, c in enumerate(coros)))


async def run_agents(
    pairs: list[tuple[Any, Any]], ctx: Any, *, best_effort: bool = False
) -> list[Any]:
    """Run `[(agent, task), …]` concurrently — each under `ctx.child()`, bounded by the tree semaphore.

    Returns AgentResults in input order. In ``best_effort`` mode a failed slot holds a
    :class:`~agentkit.kernel.errors.Failure` (with the raised exception on ``.cause``), not the
    raw exception, so callers can inspect it as first-class data. The shared budget accrues
    across all of them; depth/concurrency caps apply tree-wide.

    Two opt-in coordination seams engage automatically when the caller wired
    them onto ``ctx``:

    * **ActorBudget slicing** — when ``ctx.actor_budget`` is set, carve one
      slice per child (``1/N`` of each axis of the parent's remaining) via
      ``reserve_for_child`` BEFORE dispatch. Each child's ctx sees its OWN
      ``ActorBudget`` (with the slice as ``max_*``), and after every child
      returns we ``settle_child`` on the parent's book so the parent's
      ``used_*`` reflects what was actually spent (capped at reservation).
      If any reservation would exceed the parent's cap, we release the
      already-reserved siblings and re-raise ``BudgetExhausted`` before any
      child runs — fail-fast semantics.
    * **SignalChannel attach** — when the coordinator's ctx has
      ``signal_channel`` set AND the child ``Agent`` has its own ``channel``
      field, we call ``child.channel.attach_parent(ctx.signal_channel.merge_inbox)``
      so ``DataSignal``s emitted by the child fan up to the coordinator's
      merge inbox without polling.

    Both features are strict no-ops when the caller doesn't opt in. Missing
    fields on structural stubs (``NullCtx``) are tolerated via ``getattr``.
    """
    sem = ctx.semaphore()
    n = len(pairs)

    # ── ActorBudget wiring ─────────────────────────────────────────────
    # Compute per-child slices (fixed at fan-out time — every child sees the
    # SAME share of the parent's remaining at kickoff so the ordering of
    # reservations doesn't skew fairness) and reserve upfront so exhaustion
    # is fail-fast BEFORE we spawn any coroutines.
    parent_actor_budget: Any = getattr(ctx, "actor_budget", None)
    child_actor_budgets: list[Any] = [None] * n
    reserved_slices: list[tuple[int, Decimal, int] | None] = [None] * n

    if parent_actor_budget is not None and n > 0:
        # Late import — the runtime layer doesn't own ActorBudget, and eager
        # import at module load would drag the agents.control tree into the
        # kernel's import chain.
        from agentkit.agents.control.budget import ActorBudget, BudgetExhausted

        # Also lazy, and for the same reason: the KERNEL must not import the
        # runtime at module scope. ``kernel.concurrency`` -> ``runtime.meter``
        # pulls in ``runtime/__init__`` -> ``runtime.context`` ->
        # ``kernel.concurrency``, which is a genuine cycle
        # ("cannot import name 'CancellationToken' from partially initialized
        # module"). Inside the function the runtime is fully built.
        from agentkit.runtime.meter import MONEY_SCALE, to_money

        # The smallest slice worth carving on the money axis — one unit at the
        # ledger's scale. A slice rounding below this is starvation, not a
        # budget.
        money_unit = to_money(Decimal(1).scaleb(-MONEY_SCALE))

        # An already-spent envelope cannot be carved at all, and saying so is
        # better than the alternative. Without this, a fully-spent axis yields
        # zero-sized slices, every reservation of zero "succeeds", and the
        # caller gets N children that each stop on their first pre-flight —
        # a fan-out that looks like it ran and did nothing. ``exhausted()`` is
        # an OR across axes, so passing it guarantees EVERY axis has room and
        # every ``base_*`` below is positive.
        if parent_actor_budget.exhausted():
            from agentkit.agents._agent_helpers import _tightest_exhausted_axis

            raise BudgetExhausted(
                _tightest_exhausted_axis(parent_actor_budget),
                f"cannot fan out {n} child(ren): the parent actor budget is already "
                f"exhausted on {_tightest_exhausted_axis(parent_actor_budget)}",
            )

        base_tokens = parent_actor_budget.remaining_tokens()
        base_cost = parent_actor_budget.remaining_cost()  # exact, not the float mirror
        base_steps = parent_actor_budget.remaining_steps()
        base_wall = parent_actor_budget.remaining_wall_seconds()

        # Equal slices, so the order reservations happen in cannot skew
        # fairness. Floor division strands at most n-1 of the smallest unit on
        # each axis (a micro-dollar, a token, a step) until the next fan-out
        # re-reads ``remaining_*`` — accepted deliberately, because handing the
        # remainder to whichever child happens to be first is a worse property
        # than leaving a rounding crumb.
        #
        # ``_at_least_one`` is what makes a STARVED axis fail fast rather than
        # silently starve: N children against fewer than N units over-commits,
        # so the reservation loop below raises ``BudgetExhausted`` before any
        # child runs. Previously only ``steps`` did this — tokens and cost
        # floored to zero, producing exactly the no-op children described
        # above, on two of the three axes.
        slice_tokens = _at_least_one(base_tokens // n)
        slice_steps = _at_least_one(base_steps // n)
        slice_cost = max(money_unit, to_money(base_cost / n))

        try:
            for idx in range(n):
                parent_actor_budget.reserve_for_child(
                    tokens=slice_tokens,
                    cost_usd=slice_cost,
                    steps=slice_steps,
                )
                reserved_slices[idx] = (slice_tokens, slice_cost, slice_steps)
                child_actor_budgets[idx] = ActorBudget(
                    max_tokens=slice_tokens,
                    max_cost_usd=slice_cost,
                    # Exactly what was reserved — no ``max(slice_steps, 1)``
                    # floor. Granting a step that was never reserved let a
                    # child spend on an axis the parent had not committed, and
                    # ``settle_child`` caps usage at the reservation, so that
                    # spend was invisible on the parent's books. The slice is
                    # already >= 1 by construction.
                    max_steps=slice_steps,
                    # Wall isn't a reservation axis on the parent — pass the
                    # parent's remaining wall_seconds so the child's clock
                    # respects the same deadline without double-counting.
                    max_wall_seconds=base_wall,
                )
        except BudgetExhausted:
            # Release the already-reserved slices so the parent's books
            # don't leak — ``settle_child`` with zero usage releases the
            # reservation without charging. Then re-raise so the caller sees
            # the fail-fast contract.
            for slice_ in reserved_slices:
                if slice_ is None:
                    continue
                st, sc, ss = slice_
                parent_actor_budget.settle_child(
                    reserved_tokens=st,
                    reserved_cost_usd=sc,
                    reserved_steps=ss,
                    used_tokens=0,
                    used_cost_usd=0.0,
                    used_steps=0,
                )
            raise

    # ── Child ctx construction + SignalChannel wiring ──────────────────
    # Build child contexts (each with its own ``ctx.child()`` — depth+1,
    # shared budget/services/cancel) and attach the ActorBudget slice +
    # signal channel wiring per child.
    parent_channel: Any = getattr(ctx, "signal_channel", None)
    child_ctxs: list[Any] = []
    for idx, (agent, _task) in enumerate(pairs):
        cb = child_actor_budgets[idx]
        # Prefer an agent-specific channel over ctx propagation — a child
        # with its own ``channel`` field is the coordination story's
        # canonical shape (agent-identity attribution on the merge inbox).
        agent_channel = getattr(agent, "channel", None)
        child_kwargs: dict[str, Any] = {}
        if cb is not None:
            child_kwargs["actor_budget"] = cb
        if agent_channel is not None:
            child_kwargs["signal_channel"] = agent_channel
        child_ctx = ctx.child(**child_kwargs) if child_kwargs else ctx.child()
        # Attach the child's channel to the coordinator's merge_inbox so the
        # child's ``emit`` dual-writes upward. Guarded on BOTH sides — if
        # either the parent or the child hasn't opted in, we skip.
        if agent_channel is not None and parent_channel is not None:
            merge_inbox = getattr(parent_channel, "merge_inbox", None)
            if merge_inbox is not None:
                agent_channel.attach_parent(merge_inbox)
        child_ctxs.append(child_ctx)

    coros = [agent.run(task, child_ctxs[idx]) for idx, (agent, task) in enumerate(pairs)]
    try:
        results = await (
            gather_best_effort(coros, sem=sem) if best_effort else gather_bounded(coros, sem=sem)
        )
    finally:
        # ── ActorBudget settlement ─────────────────────────────────────
        # Settle every reservation even on cancel/raise so the parent's
        # ``reserved_*`` never leaks. ``settle_child`` is idempotent-safe
        # only if called once per reservation; ``finally`` guarantees the
        # single-call shape here.
        if parent_actor_budget is not None:
            for idx, slice_ in enumerate(reserved_slices):
                if slice_ is None:
                    continue
                cb = child_actor_budgets[idx]
                st, sc, ss = slice_
                parent_actor_budget.settle_child(
                    reserved_tokens=st,
                    reserved_cost_usd=sc,
                    reserved_steps=ss,
                    used_tokens=cb.used_tokens if cb is not None else 0,
                    used_cost_usd=cb.used_cost_usd if cb is not None else 0.0,
                    used_steps=cb.used_steps if cb is not None else 0,
                )
    return results
