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
import logging
import time

import pytest

from agentkit.kernel import concurrency
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


# ── …but the abort must be BOUNDED ───────────────────────────────────────────
#
# Cancelling AND awaiting the siblings (above) handed the abort's liveness to
# the worst-behaved child. Measured with one sibling looping on
# ``except CancelledError: pass``: the process was still wedged at 130s with no
# output, and an outer ``asyncio.wait_for(..., timeout=2)`` could not break out
# either — the await sat inside an ``except BaseException`` handler, which is a
# shielded region from the caller's point of view. After bounding the wait, the
# same swallower unwinds in ~grace (measured 3.05s at a 3s grace) and the outer
# ``wait_for`` regains its power to interrupt (measured 2.00s).
#
# ``SIBLING_CLEANUP_GRACE_S`` is shrunk in these tests so the suite doesn't pay
# the production grace; the tests assert BEHAVIOUR relative to it, not the value.


@pytest.fixture
def short_grace(monkeypatch):
    """Shrink the cleanup grace so a wedged-sibling test costs ~0.2s, not 5s."""
    monkeypatch.setattr(concurrency, "SIBLING_CLEANUP_GRACE_S", 0.2)
    return 0.2


def test_abort_is_bounded_when_a_sibling_swallows_cancellation(short_grace, caplog):
    """EDGE CASE — a sibling that swallows ``CancelledError`` entirely can never
    be stopped, so the abort must give up on it rather than wait forever.
    BEFORE: still hung at 130s. AFTER: unwinds at the grace, and says so."""
    # ``release`` is a TEST-HARNESS escape only: it stays False for the whole
    # window under test (so the sibling is genuinely unstoppable while the abort
    # runs) and is flipped afterwards purely so ``asyncio.run``'s own shutdown —
    # which cancels-and-awaits every leftover task — doesn't wedge the suite the
    # way the production bug wedged the process.
    release: list[bool] = [False]

    async def swallower():
        while True:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                if release[0]:
                    raise
                # never honours the cancel

    async def aborts():
        await asyncio.sleep(0.01)
        raise Cancelled("run cancelled")

    async def go():
        sem = asyncio.Semaphore(10)
        started = time.perf_counter()
        with pytest.raises(Cancelled):
            await gather_best_effort([swallower(), aborts()], sem=sem)
        elapsed = time.perf_counter() - started
        release[0] = True
        return elapsed

    with caplog.at_level(logging.WARNING, logger="agentkit.kernel.concurrency"):
        # The outer timeout is a TRIPWIRE, not the mechanism: if the bound
        # regressed this would hang forever rather than fail (the pre-fix
        # behaviour), so we keep it well above the grace.
        elapsed = _run(asyncio.wait_for(go(), timeout=5.0))

    assert elapsed < 2.0, f"the abort was not bounded by the grace (took {elapsed:.2f}s)"
    assert elapsed >= short_grace, "the grace was not honoured — did the await get deleted?"
    # Abandoning work silently would leave a permanently shrunk semaphore and no
    # trace of why, so the give-up is required to be loud.
    assert any("abandoning" in r.getMessage() for r in caplog.records), (
        f"no warning was logged for the abandoned sibling: {caplog.records}"
    )


def test_outer_wait_for_can_now_interrupt_the_abort(short_grace):
    """EDGE CASE — the caller's own timeout must be able to break out. BEFORE the
    bound, ``wait_for(timeout=2)`` around a gather with a swallowing sibling
    never fired, because the wait lived inside ``except BaseException``."""

    release: list[bool] = [False]  # see the note in the test above

    async def swallower():
        while True:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                if release[0]:
                    raise

    async def aborts():
        await asyncio.sleep(0.01)
        raise Cancelled("run cancelled")

    async def go():
        sem = asyncio.Semaphore(10)
        # A caller timeout SHORTER than the grace must win.
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await asyncio.wait_for(gather_best_effort([swallower(), aborts()], sem=sem), timeout=0.05)
        release[0] = True

    _run(asyncio.wait_for(go(), timeout=5.0))


def test_a_sibling_that_takes_a_moment_still_finishes_its_cleanup(short_grace):
    """POSITIVE CONTROL — the bound must not become "don't wait at all". A
    sibling whose post-cancel cleanup takes a moment (but less than the grace)
    still runs it to completion BEFORE the caller resumes.

    This is the assertion that fails a "fix" that merely deletes the await:
    without the await the sibling is only *marked* cancelled, so ``cleaned`` is
    still empty when ``pytest.raises`` returns and the semaphore is still held.
    """
    cleaned: list[str] = []

    async def polite():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # real post-cancel work, well inside the grace
            cleaned.append("released")
            raise

    async def aborts():
        await asyncio.sleep(0.01)
        raise Cancelled("run cancelled")

    async def go():
        sem = asyncio.Semaphore(2)
        with pytest.raises(Cancelled):
            await gather_best_effort([polite(), aborts()], sem=sem)
        # Sampled the instant the caller regains control — not after a sleep.
        return list(cleaned), sem._value, [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    done, permits, live = _run(asyncio.wait_for(go(), timeout=5.0))
    assert done == ["released"], "the sibling's cleanup had NOT run when the caller resumed"
    assert permits == 2, "the semaphore permit was still held when the caller resumed"
    assert live == []


def test_abort_with_an_already_finished_sibling(short_grace):
    """EDGE CASE — cancelling an already-done task is a no-op, and the bounded
    wait must return immediately rather than sit out the whole grace."""

    async def quick():
        return "done"

    async def aborts():
        await asyncio.sleep(0.05)  # long enough that ``quick`` has finished
        raise Cancelled("run cancelled")

    async def go():
        sem = asyncio.Semaphore(10)
        started = time.perf_counter()
        with pytest.raises(Cancelled):
            await gather_best_effort([quick(), aborts()], sem=sem)
        return time.perf_counter() - started

    elapsed = _run(asyncio.wait_for(go(), timeout=5.0))
    assert elapsed < short_grace + 0.05, "a finished sibling should not cost the grace"


def test_empty_task_list_is_a_no_op(short_grace):
    """EDGE CASE — no coroutines at all. ``asyncio.wait`` raises ``ValueError``
    on an empty set, so the bounded wait must be guarded; and with nothing to
    abort the gather simply returns an empty list."""
    assert _run(asyncio.wait_for(gather_best_effort([], sem=asyncio.Semaphore(4)), 5.0)) == []


def test_normal_path_nobody_is_cancelled(short_grace):
    """POSITIVE CONTROL — the abort path must stay entirely off the happy path:
    every coroutine completes, in input order, with no cancellation and no
    grace paid."""

    async def work(value, delay):
        await asyncio.sleep(delay)
        return value

    async def go():
        sem = asyncio.Semaphore(2)  # forces queueing, so the semaphore is exercised
        started = time.perf_counter()
        out = await gather_best_effort([work("a", 0.02), work("b", 0.01), work("c", 0.02)], sem=sem)
        return out, sem._value, time.perf_counter() - started

    out, permits, elapsed = _run(asyncio.wait_for(go(), timeout=5.0))
    assert out == ["a", "b", "c"]
    assert permits == 2, "the semaphore leaked a permit on the happy path"
    assert elapsed < short_grace, "the happy path paid the cleanup grace"
