"""Structured concurrency for the agent tree — bounded fan-out of sub-agents / tools / LLM calls.

Uses `asyncio.TaskGroup` (a failing child cancels its siblings and raises an ExceptionGroup) bounded
by a shared `asyncio.Semaphore`, and copies the current `contextvars` into each task so the active
trace span / correlation id propagate. `gather_best_effort` is the opt-in "one failure shouldn't sink
the batch" variant. Stdlib-only (asyncio + contextvars + logging); needs Python ≥ 3.11 for
TaskGroup.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
from collections.abc import Awaitable, Coroutine
from typing import Any, TypeVar

from agentkit.core.contracts.errors import Failure

R = TypeVar("R")

_LOGGER = logging.getLogger(__name__)

#: How long an abort will wait for already-cancelled siblings to run their
#: ``finally`` blocks before giving up on them and unwinding anyway.
#:
#: An abort must be BOUNDED. Awaiting the cancelled siblings unconditionally
#: hands the abort's liveness to the slowest — or most badly behaved — child:
#: measured with one sibling looping on ``except CancelledError: pass``, the
#: gather was still wedged at 130s with no output, and an outer
#: ``asyncio.wait_for(..., timeout=2)`` could NOT break it out, because the
#: await sat inside an ``except BaseException`` handler (a shielded region from
#: the caller's point of view). Five seconds is a graceful-shutdown grace, not a
#: deadline: the cleanup this exists to buy — releasing a semaphore, closing a
#: session, flushing a span — is sub-millisecond, and the worst well-behaved
#: case measured (1s of post-cancel work) still finishes comfortably inside it.
#: Module-level so a test can shrink it; deliberately NOT a parameter, because
#: the bound is a safety property of the abort, not a per-call tuning knob.
SIBLING_CLEANUP_GRACE_S = 5.0



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
    A failure cancels the rest (structured concurrency) and surfaces as an ExceptionGroup.

    **Pass ``sem=ctx.semaphore()``, not a semaphore of your own, whenever the
    coroutines may themselves fan out.** A permit is held for the WHOLE duration
    of each coroutine, so a nested call draws from a pool its own ancestors have
    already drained — and the result is a permanent hang, not a slowdown.
    Measured with one hand-rolled ``asyncio.Semaphore(2)`` shared across levels,
    a two-deep two-wide tree never returns; at ``Semaphore(4)``, three-deep does
    the same. The cliff moves with the cap, so a tree that works in testing can
    still wedge in production once it gets one level deeper.

    ``ctx.semaphore()`` returns a pool keyed on ``ctx.depth``, which makes the
    cycle structurally impossible: an ancestor at depth d only ever holds
    permits from pool d, and its children draw from pool d+1. See
    :meth:`agentkit.core.runtime.Budget.semaphore` for the full rationale and the
    bound this trades for (``max_concurrency`` per level, so worst-case
    in-flight work is ``max_concurrency * (max_depth + 1)``).

    A private semaphore is fine for a FLAT fan-out — a batch of independent
    tool calls or provider requests that do not fan out again."""

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
    :class:`~agentkit.core.contracts.errors.Failure` wrapping the raised exception). Use when one
    sub-agent/tool failing must NOT cancel the others.

    Wrapping the exception in a ``Failure`` (rather than returning the raw exception object)
    lets a parent treat the failed slot as **first-class data**: it carries a classified
    category, a ``source`` naming the slot (``gather_best_effort[i]``), and the originating
    ``cause``, so a caller can retry / route around / escalate uniformly. It also disambiguates
    "the coroutine returned an ``Exception`` value" from "the coroutine raised" — the raw-
    exception design conflated the two.

    The nesting rule from :func:`gather_bounded` applies here identically, and
    the isolation this function provides does NOT soften it: a deadlock is not a
    failure any slot can hold, so a nested fan-out over a private semaphore
    hangs exactly as it would there. Pass ``sem=ctx.semaphore()`` whenever the
    coroutines may fan out again.
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

    # Explicit tasks, not ``gather(*coros)``, so the escape hatches above can
    # take the siblings down with them. ``asyncio.gather`` propagates the first
    # exception IMMEDIATELY and leaves its remaining children running detached:
    # measured on a ``Cancelled`` abort, ``live tasks after gather returned: 2``
    # and the siblings' log kept filling AFTER the abort had already reached the
    # caller. That is precisely the shape ``Cancelled`` exists to prevent — an
    # abort that aborts nothing, with orphan work still spending the run's budget
    # and racing whatever the caller does next.
    tasks = [
        asyncio.create_task(_run(i, c), context=contextvars.copy_context())
        for i, c in enumerate(coros)
    ]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        # Only ``Cancelled`` / ``CancelledError`` can get here — every other
        # exception is isolated into a ``Failure`` slot by ``_run``. Cancel AND
        # await: without the await the tasks are merely *marked* cancelled and
        # have not yet run their ``finally`` blocks (releasing the semaphore,
        # closing resources) when the caller resumes.
        #
        # But the await must be BOUNDED. Awaiting them unconditionally made the
        # abort's liveness a function of the WORST-behaved child: a sibling
        # looping on ``except CancelledError: pass`` left the gather wedged at
        # 130s with no output, and an outer ``asyncio.wait_for(..., timeout=2)``
        # could not break out either, because this await sits inside an
        # ``except BaseException`` handler — a region the caller cannot
        # interrupt. Measured after the bound: that same swallower unwinds in
        # ~0.1s (the grace), while a sibling doing 1s of genuine post-cancel
        # cleanup still completes it (1.05s, ``finally`` ran) exactly as before.
        #
        # ``asyncio.wait`` rather than ``wait_for(gather(...))``: it returns
        # ``(done, pending)`` on timeout instead of raising, it does NOT cancel
        # or re-await the stragglers on the way out (so it cannot itself wedge),
        # and it stays interruptible — an outer cancel now lands here and
        # propagates instead of being absorbed.
        for t in tasks:
            t.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=SIBLING_CLEANUP_GRACE_S)
            if pending:
                # We are abandoning these. Whatever they hold — most importantly
                # the semaphore permit taken by ``async with sem`` in ``_run`` —
                # stays held, because a task that will not honour cancellation
                # cannot be made to release anything from out here. Nothing can
                # be done about it; what CAN be done is refuse to hang and say
                # so loudly, since a permanently shrunk semaphore is otherwise a
                # silent, drifting concurrency ceiling.
                _LOGGER.warning(
                    "gather_best_effort: abandoning %d sibling task(s) that did not "
                    "finish cancelling within %.1fs; their cleanup (semaphore release, "
                    "resource close) has NOT run: %s",
                    len(pending),
                    SIBLING_CLEANUP_GRACE_S,
                    [t.get_name() for t in pending],
                )
        raise
