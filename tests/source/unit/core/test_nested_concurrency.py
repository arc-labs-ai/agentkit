"""Fan-out that fans out again — the two hazards a flat gather never shows.

WHY THIS FILE EXISTS
--------------------
``test_concurrency.py`` next door drives FLAT batches: one level of
``gather_best_effort``, one semaphore, one layer of slots. Both of this
module's documented production incidents need a SECOND level to reproduce.

* **An abort that gets isolated.** ``Cancelled`` is agentkit's COOPERATIVE
  cancellation and the token is shared down the whole run tree, so a tripped
  token means every coroutine at every level is about to raise it. Letting the
  ``except Exception`` arm catch it turns one abort into N independent
  ``Failure`` slots — and in a tree the count is multiplicative, so the caller
  receives something that looks like "everything broke at once" instead of
  "this run was cancelled". The two call for opposite handling: one is retried
  per slot, the other must not be retried at all, because the token is still
  tripped.

* **A permit cycle.** ``_run`` holds its permit for the whole duration of the
  coroutine, so a child drawing from a pool its own ancestors have already
  drained waits for a release only an ancestor can perform. The module's
  docstring records a two-deep two-wide tree on ``Semaphore(2)`` that never
  returns, and the cliff moves with the cap — which is why a tree that passes
  testing can still wedge in production one level deeper.

Written as SYNC tests driving an inner coroutine through ``asyncio.run``, the
established pattern in ``test_concurrency.py``: the assertions are about what
escaped the gather, and they read better outside the event loop.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.core.contracts.errors import Failure
from agentkit.core.execution.concurrency import (
    CancellationToken,
    Cancelled,
    gather_best_effort,
    gather_bounded,
)


def test_a_tripped_token_two_levels_down_aborts_the_whole_tree_instead_of_filling_slots() -> None:
    """A cooperative abort must unwind every level of the tree as ONE event.

    The token is shared across the run tree, so the inner leaf that trips it
    and every sibling that checks it next are all reporting the same abort.
    Isolated into slots, the caller gets a list of unrelated-looking failures
    with no way to tell an aborted run from a batch where everything happened
    to break at once — and in a tree the miscount is per-leaf, so the louder
    the fan-out the more convincing the lie. Worse, a caller that retries
    failed slots would then retry work the run has already been told to stop
    doing, against a token that is still tripped.
    """
    token = CancellationToken()
    completed: list[str] = []

    async def leaf(name: str, *, trips: bool = False) -> str:
        if trips:
            token.cancel()
        token.raise_if_cancelled()
        completed.append(name)
        return name

    async def branch(name: str, *, trips: bool = False) -> list[str | Failure]:
        inner = asyncio.Semaphore(2)
        return await gather_best_effort(
            [leaf(f"{name}.1", trips=trips), leaf(f"{name}.2")], sem=inner
        )

    async def go() -> None:
        outer = asyncio.Semaphore(2)
        await gather_best_effort([branch("a", trips=True), branch("b")], sem=outer)

    with pytest.raises(Cancelled):
        asyncio.run(go())

    assert "a.1" not in completed, "the leaf that tripped the token must not report success"


def test_an_abort_in_a_nested_batch_is_not_reachable_as_slot_data() -> None:
    """The negative half of the same guarantee, asserted on the VALUES rather
    than on the exception type.

    A ``Cancelled`` isolated by the ``except Exception`` arm does not look like
    a bug from outside: the gather returns normally and every slot is a
    well-formed ``Failure`` carrying an indexed source, which is precisely the
    shape a caller is told to inspect. So the test has to reach the returned
    lists and assert that nothing in them describes the abort — checking only
    that ``Cancelled`` propagated would still pass for an arrangement that
    isolated the abort at the inner level and re-raised something else at the
    outer one.
    """
    token = CancellationToken()
    slots: list[object] = []

    async def leaf(*, trips: bool = False) -> str:
        if trips:
            token.cancel()
        token.raise_if_cancelled()
        return "ok"

    async def branch(*, trips: bool = False) -> list[str | Failure]:
        inner = asyncio.Semaphore(2)
        results = await gather_best_effort([leaf(trips=trips), leaf()], sem=inner)
        slots.extend(results)
        return results

    async def go() -> None:
        outer = asyncio.Semaphore(2)
        slots.extend(await gather_best_effort([branch(trips=True), branch()], sem=outer))

    with pytest.raises(Cancelled):
        asyncio.run(go())

    isolated = [s for s in slots if isinstance(s, Failure) and isinstance(s.cause, Cancelled)]
    assert isolated == [], (
        "a cooperative abort was isolated into a Failure slot; the caller would "
        "read the run as a batch of independent failures and could retry them"
    )


def test_an_ordinary_failure_two_levels_down_still_stays_in_its_own_slot() -> None:
    """The positive control, and the reason the abort rule cannot simply be
    "stop isolating things in a tree". A ``ValueError`` from one leaf is a
    normal per-slot failure at every depth: its sibling finishes, its parent
    returns, and the outer batch is a success containing one inner batch that
    holds a ``Failure``. If this stopped holding, the isolation guarantee would
    exist only for flat batches and every nested caller would start receiving
    exception groups it has no handler for."""

    async def leaf(name: str, *, boom: bool = False) -> str:
        if boom:
            raise ValueError(f"{name} broke")
        return name

    async def branch(name: str, *, boom: bool = False) -> list[str | Failure]:
        inner = asyncio.Semaphore(2)
        return await gather_best_effort(
            [leaf(f"{name}.1", boom=boom), leaf(f"{name}.2")], sem=inner
        )

    async def go() -> list[list[str | Failure] | Failure]:
        outer = asyncio.Semaphore(2)
        return await gather_best_effort([branch("a", boom=True), branch("b")], sem=outer)

    outer_results = asyncio.run(go())

    assert all(not isinstance(r, Failure) for r in outer_results), "the tree itself did not fail"
    first, second = outer_results
    assert isinstance(first, list) and isinstance(second, list)
    assert isinstance(first[0], Failure)
    assert isinstance(first[0].cause, ValueError)
    assert first[1] == "a.2", "the sibling of a failed leaf still ran to completion"
    assert second == ["b.1", "b.2"], "the other branch was untouched"


def test_a_permit_is_held_for_the_whole_coroutine_so_a_nested_call_waits_on_its_parent() -> None:
    """The permit cycle, pinned at its cause rather than by reproducing a hang.

    ``_run`` wraps each coroutine in ``async with sem``, so the permit is held
    until that coroutine RETURNS — not until it next suspends. A child drawing
    from the same pool therefore queues behind its own ancestors, and with the
    pool drained by those ancestors the wait can never end: no amount of
    waiting releases a permit only a parent can release. Asserted as "the pool
    the parent is inside has nothing left to give, while the next depth's pool
    does", because that IS the deadlock — the hang itself cannot be left in a
    unit suite that must fail fast, and a timeout probe would abandon
    coroutines mid-acquire.
    """
    observed: dict[str, bool] = {}

    async def only_child() -> str:
        return "done"

    async def parent(same_pool: asyncio.Semaphore, deeper_pool: asyncio.Semaphore) -> str:
        # Inside the parent's own coroutine: its permit is still held.
        observed["same_pool_exhausted"] = same_pool.locked()
        observed["deeper_pool_available"] = not deeper_pool.locked()
        nested = await gather_bounded([only_child()], sem=deeper_pool)
        assert nested == ["done"], "the nested fan-out drew from the deeper pool and finished"
        return "ok"

    async def go() -> list[str]:
        depth_0 = asyncio.Semaphore(1)
        depth_1 = asyncio.Semaphore(1)
        return await gather_bounded([parent(depth_0, depth_1)], sem=depth_0)

    assert asyncio.run(go()) == ["ok"]
    assert observed["same_pool_exhausted"], (
        "the parent's permit was released before its coroutine returned, which "
        "would make the documented nesting hazard unreproducible"
    )
    assert observed["deeper_pool_available"], (
        "a pool keyed on depth is what makes the cycle structurally impossible"
    )
