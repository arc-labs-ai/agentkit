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

from agentkit.kernel.concurrency import gather_best_effort
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
