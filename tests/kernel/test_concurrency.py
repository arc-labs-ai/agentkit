"""Kernel concurrency helpers — `gather_best_effort`'s failure-isolation contract.

Two invariants:

* A raised ``Exception`` is captured into a ``Failure`` (first-class error data), not
  returned as the raw exception object — so a caller can inspect ``.cause`` / ``.source`` /
  ``.category`` uniformly and never confuses "the coroutine returned an ``Exception``
  value" with "the coroutine raised".
* An ``asyncio.CancelledError`` is NEVER swallowed — cancellation is control flow that
  must unwind the whole gather, not a per-slot failure.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.kernel.concurrency import Cancelled, gather_best_effort
from agentkit.kernel.errors import Failure


def _run(coro):
    return asyncio.run(coro)


def test_gather_best_effort_wraps_exceptions_in_failure():
    """A worker raising ``Exception`` lands in the slot as a ``Failure`` carrying the
    originating exception on ``.cause`` and an indexed ``.source``."""

    async def ok(value):
        return value

    async def boom():
        raise ValueError("boom")

    sem = asyncio.Semaphore(2)
    results = _run(gather_best_effort([ok("a"), boom(), ok("c")], sem=sem))

    assert results[0] == "a"
    assert results[2] == "c"

    slot = results[1]
    assert isinstance(slot, Failure)
    # The raw exception must NOT be in the slot — the wrap is the whole point of the fix.
    assert not isinstance(slot, BaseException)
    assert isinstance(slot.cause, ValueError)
    assert str(slot.cause) == "boom"
    assert "1" in slot.source  # the failed index (1) is named in the source
    assert "gather_best_effort" in slot.source


def test_gather_best_effort_lets_cancelled_propagate():
    """``asyncio.CancelledError`` inside a worker is re-raised — NOT wrapped as a ``Failure``.
    Cancellation is control flow that must unwind the whole gather."""

    async def cancels():
        raise asyncio.CancelledError()

    sem = asyncio.Semaphore(1)
    with pytest.raises(asyncio.CancelledError):
        _run(gather_best_effort([cancels()], sem=sem))


# ── A propagating abort must take the siblings down with it ──────────────────
#
# ``asyncio.gather`` propagates the first exception IMMEDIATELY and leaves its
# remaining children running detached. Measured on a ``Cancelled`` abort:
# ``live tasks after gather returned: 2``, and the siblings' log kept filling
# AFTER the abort had already reached the caller — an abort that aborted
# nothing, with orphan work still spending the run's budget.


def test_cancelled_abort_cancels_and_awaits_the_siblings():
    """When ``Cancelled`` unwinds the gather, every sibling must be cancelled
    AND awaited: cancelled so no orphan keeps working, awaited so its
    ``finally`` (semaphore release, resource close) has actually run by the
    time the caller resumes."""
    log: list[str] = []
    cleaned: list[str] = []

    async def sibling(name):
        try:
            for i in range(100):
                await asyncio.sleep(0.01)
                log.append(f"{name}{i}")
        except asyncio.CancelledError:
            cleaned.append(name)  # the ``finally``-shaped work the await buys us
            raise

    async def aborts():
        await asyncio.sleep(0.02)
        raise Cancelled("run cancelled")

    async def go():
        sem = asyncio.Semaphore(10)
        with pytest.raises(Cancelled):
            await gather_best_effort([sibling("a"), sibling("b"), aborts()], sem=sem)
        live = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        at_abort = len(log)
        await asyncio.sleep(0.1)  # a detached sibling would keep logging here
        return live, at_abort, len(log)

    live, at_abort, later = _run(go())
    assert live == [], "siblings were left running detached after the abort"
    assert sorted(cleaned) == ["a", "b"], "siblings were cancelled but never awaited"
    assert later == at_abort, "a sibling kept producing work after the abort"


def test_asyncio_cancellederror_also_takes_the_siblings_down():
    """Edge case — the same guarantee for real ``asyncio.CancelledError``, which
    takes a different route out of ``_run`` than agentkit's cooperative
    ``Cancelled``."""
    cleaned: list[str] = []

    async def sibling(name):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cleaned.append(name)
            raise

    async def cancels():
        await asyncio.sleep(0.01)
        raise asyncio.CancelledError()

    async def go():
        sem = asyncio.Semaphore(10)
        with pytest.raises(asyncio.CancelledError):
            await gather_best_effort([sibling("a"), cancels()], sem=sem)
        return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    assert _run(asyncio.wait_for(go(), timeout=2.0)) == []
    assert cleaned == ["a"]


def test_positive_control_an_isolated_failure_does_not_cancel_siblings():
    """POSITIVE CONTROL: the whole point of ``gather_best_effort`` is that an
    ordinary ``Exception`` is isolated into a ``Failure`` slot and the siblings
    RUN TO COMPLETION. A "fix" that cancelled peers on any exception — or that
    turned the gather into fail-fast — fails here."""

    async def slow_ok(value):
        await asyncio.sleep(0.05)  # finishes well after the failure
        return value

    async def boom():
        raise ValueError("isolated")

    async def go():
        sem = asyncio.Semaphore(10)
        return await gather_best_effort([slow_ok("a"), boom(), slow_ok("c")], sem=sem)

    results = _run(asyncio.wait_for(go(), timeout=2.0))
    assert results[0] == "a" and results[2] == "c"  # siblings completed normally
    assert isinstance(results[1], Failure)
