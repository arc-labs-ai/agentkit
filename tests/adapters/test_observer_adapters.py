"""Observer adapters under load: the `QueueObserver` bound, the `RollupObserver` flush race, and
`close()` forwarding through a wrapper stack.

The happy paths live in ``tests/kernel/test_observation.py``. This module covers the failure modes an
observer is specifically forbidden to have — losing observations, and raising into the run it observes.
Offline and deterministic (handshake events, never wall-clock sleeps)."""

import asyncio

import pytest

from agentkit.adapters.observer import (
    CollectingObserver,
    Hooks,
    PolicyObserver,
    QueueObserver,
    RollupObserver,
)
from agentkit.kernel.observation import Observation, ObserverPort


def _run(coro, *, timeout: float = 5.0):
    """Every test here is a fixed amount of work with no I/O, so a wall-clock
    ceiling is free. It exists because each of these regressions fails as a
    HANG rather than an assertion: a dropped observation means the consumer's
    handshake never fires, and a re-entered ``_flush`` deadlocks on the gate.
    Without the ceiling the suite would stall instead of reporting."""
    return asyncio.run(asyncio.wait_for(coro, timeout))


def _p(i):
    return Observation(kind="progress", render=f"p{i}")


# ── QueueObserver: the bound is queue DEPTH, not lifetime emissions ──────────
#
# ``_noncritical`` was incremented on emit and never decremented when
# ``stream()`` popped, so it counted every non-critical observation the
# observer had EVER seen. Past ``maxsize`` cumulative emissions, each new
# observation was appended and immediately deleted — from an empty queue,
# with a consumer keeping up perfectly.


def test_queue_bound_is_depth_not_lifetime_and_a_live_consumer_loses_nothing():
    """Measured before the fix: ``maxsize=4``, 12 progress emitted with the
    consumer draining each one → the consumer saw ``['p0','p1','p2','p3']``
    and the observer ended at ``_noncritical=4, len(_items)=0``. 8 of 12
    silently lost with nothing backed up at all.

    The handshake (``consumed``) makes this deterministic: the producer only
    emits again once the consumer has taken the previous item, so the queue
    depth never exceeds 1 and the bound can never legitimately fire."""
    seen: list[str] = []

    async def go():
        q = QueueObserver(maxsize=4)
        consumed = asyncio.Event()

        async def consume():
            async for o in q.stream():
                seen.append(o.render)
                consumed.set()

        task = asyncio.create_task(consume())
        for i in range(12):
            await q.emit(_p(i))
            await consumed.wait()  # bounded by `_run`'s ceiling
            consumed.clear()
        await q.close()
        await task
        return q

    q = _run(go())
    assert seen == [f"p{i}" for i in range(12)]  # 12 emitted, 12 delivered
    assert q._noncritical == 0 and not q._items  # depth counter is back to empty


def test_queue_still_bounds_a_consumer_slower_than_the_producer():
    """POSITIVE CONTROL for the fix above. ``maxsize`` is a real backpressure
    bound, so a fix that merely stopped dropping would fail here: with nobody
    draining, 20 progress observations must coalesce down to the newest 3."""

    async def go():
        q = QueueObserver(maxsize=3)
        for i in range(20):
            await q.emit(_p(i))
        await q.emit(Observation(kind="result", render="RESULT"))
        await q.emit(Observation(kind="error", render="ERR"))
        await q.close()
        return [o.render async for o in q.stream()]

    got = _run(go())
    assert got == ["p17", "p18", "p19", "RESULT", "ERR"]  # oldest coalesced, newest kept


def test_queue_never_drops_a_critical_however_far_over_the_bound():
    """``result``/``error`` are never dropped and never count against the
    bound — so a run that overflows its progress budget by 100x still
    delivers every terminal observation."""

    async def go():
        q = QueueObserver(maxsize=2)
        for i in range(200):
            await q.emit(_p(i))
            if i % 20 == 0:
                await q.emit(Observation(kind="result", render=f"R{i}"))
        await q.close()
        return [o.render async for o in q.stream()]

    got = _run(go())
    assert [g for g in got if g.startswith("R")] == [f"R{i}" for i in range(0, 200, 20)]
    assert len([g for g in got if g.startswith("p")]) == 2  # bound applies to progress only


def test_queue_capacity_is_restored_by_draining():
    """The regression's signature: capacity must come back when items leave.
    Three bursts of 5 with a drain between them each keep their newest 2 — if
    the counter still accumulated, burst 2 and 3 would yield nothing."""

    async def go():
        q = QueueObserver(maxsize=2)
        it = q.stream().__aiter__()
        got = []
        for r in range(3):
            for i in range(5):
                await q.emit(Observation(kind="progress", render=f"r{r}p{i}"))
            got += [(await it.__anext__()).render for _ in range(2)]  # drain to empty
        return got

    assert _run(go()) == ["r0p3", "r0p4", "r1p3", "r1p4", "r2p3", "r2p4"]


def test_queue_maxsize_zero_drops_every_noncritical_without_spinning():
    """Edge: a zero budget is legal (results-only backpressure) and must not
    loop forever looking for a droppable item once only criticals remain."""

    async def go():
        q = QueueObserver(maxsize=0)
        await q.emit(Observation(kind="result", render="R"))
        for i in range(5):
            await q.emit(_p(i))
        await q.close()
        return [o.render async for o in q.stream()]

    assert _run(go()) == ["R"]


# ── RollupObserver: `_flush` is re-entered concurrently ──────────────────────
#
# ``_flush`` awaited ``summarize`` and only THEN did ``self._buf = []``. A
# concurrent ``emit`` re-entered it inside that window; once the first task
# cleared the buffer the second read ``self._buf[-1]`` on an empty list.


def _gathering_rollup(sink, *, every=2, delay=0.001):
    async def summarize(buf):
        await asyncio.sleep(delay)  # the async hook: an LLM/Compactor summariser
        return ",".join(o.render for o in buf)

    return RollupObserver(sink, every=every, summarize=summarize)


def test_rollup_concurrent_flush_never_raises_into_the_caller():
    """Measured before the fix with ``every=2`` and 6 concurrent
    observations: ``exceptions raised INTO the caller's emit():
    ["IndexError('list index out of range')", ...]``. An observer that breaks
    the run it observes violates the channel's core promise."""
    sink = CollectingObserver()
    roll = _gathering_rollup(sink)
    errors: list[str] = []

    async def one(i):
        try:
            await roll.emit(_p(i))
        except Exception as e:  # noqa: BLE001 — the point of the test is what escapes
            errors.append(repr(e))

    async def go():
        await asyncio.gather(*(one(i) for i in range(40)))
        await roll.close()

    _run(go())
    assert errors == []


def test_rollup_concurrent_flush_summarises_every_observation_exactly_once():
    """POSITIVE CONTROL. Swallowing the ``IndexError`` would also make the
    test above pass while still losing data — the same window discarded every
    observation appended during the await, because the post-await
    ``self._buf = []`` threw them away unsummarised. Detaching the buffer
    BEFORE the await is what makes the accounting exact."""
    sink = CollectingObserver()
    roll = _gathering_rollup(sink)

    async def go():
        await asyncio.gather(*(roll.emit(_p(i)) for i in range(40)))
        await roll.close()

    _run(go())
    rolled = [r for o in sink.items for r in o.render.split(",") if r]
    assert sorted(rolled) == sorted(f"p{i}" for i in range(40))  # none lost
    assert len(rolled) == len(set(rolled))  # and none summarised twice
    assert sum(o.payload["rolled"] for o in sink.items) == 40  # the count agrees


def test_rollup_items_emitted_during_a_flush_land_in_the_next_summary():
    """The race window, made deterministic with a gate instead of a sleep: an
    observation arriving while ``summarize`` is awaited belongs to the NEXT
    roll-up, never to the void."""
    sink = CollectingObserver()
    gate = asyncio.Event()

    async def summarize(buf):
        await gate.wait()
        return ",".join(o.render for o in buf)

    roll = RollupObserver(sink, every=2, summarize=summarize)

    async def go():
        await roll.emit(_p(0))
        flushing = asyncio.create_task(roll.emit(_p(1)))  # trips `every`, blocks in summarize
        for _ in range(3):
            await asyncio.sleep(0)  # park `flushing` on the gate
        await roll.emit(_p(2))  # appended DURING the await
        gate.set()
        await flushing
        await roll.close()

    _run(go())
    assert [o.render for o in sink.items] == ["p0,p1", "p2"]


def test_rollup_critical_flushes_the_tail_under_concurrency():
    """A ``result`` racing a flush must still land after the roll-up it
    triggered, with nothing buffered left behind."""
    sink = CollectingObserver()
    roll = _gathering_rollup(sink, every=8)

    async def go():
        await asyncio.gather(
            *(roll.emit(_p(i)) for i in range(5)),
            roll.emit(Observation(kind="result", render="DONE")),
        )
        await roll.close()

    _run(go())
    assert [o.kind for o in sink.items].count("result") == 1  # forwarded exactly once
    rolled = [r for o in sink.items if o.kind == "summary" for r in o.render.split(",") if r]
    assert sorted(rolled) == sorted(f"p{i}" for i in range(5))  # tail flushed, nothing stranded


# ── close() is part of ObserverPort — and must forward through wrappers ──────


class _ClosingSink(CollectingObserver):
    """A terminal sink that records whether it was actually closed. A
    ``close()`` added purely to satisfy ``isinstance`` would leave
    ``closed`` False and fail the forwarding tests below."""

    def __init__(self):
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "observer",
    [
        PolicyObserver(CollectingObserver()),
        Hooks(),
        RollupObserver(CollectingObserver()),
        QueueObserver(),
        CollectingObserver(),
    ],
    ids=["policy", "hooks", "rollup", "queue", "collecting"],
)
def test_every_observer_adapter_satisfies_observer_port(observer):
    """``ObserverPort`` declares ``close()``. `PolicyObserver` and `Hooks` had
    none, so ``isinstance(PolicyObserver(...), ObserverPort) -> False``
    (measured) — the cadence wrappers did not satisfy the Protocol they exist
    to compose."""
    assert isinstance(observer, ObserverPort)


def test_close_forwards_through_two_wrapper_layers_and_flushes_the_tail():
    """POSITIVE CONTROL for the two ``close()`` methods: a no-op stub would
    satisfy the isinstance check above but still strand the buffer. Through
    ``Hooks -> PolicyObserver -> RollupObserver`` the trailing summary must
    reach the sink and the sink must itself be closed. Measured before:
    ``('NO close() on Hooks', [])`` — the tail was lost."""
    sink = _ClosingSink()
    stack = Hooks(PolicyObserver.everything(RollupObserver(sink, every=100)))

    async def go():
        await stack.emit(_p(0))  # far below the roll-up threshold
        assert sink.items == [], "nothing should have been forwarded yet"
        await stack.close()

    _run(go())
    assert [o.kind for o in sink.items] == ["summary"]
    assert sink.items[0].payload["rolled"] == 1
    assert sink.closed is True  # shutdown reached the bottom of the stack


def test_close_is_a_noop_on_an_unwired_or_close_less_inner():
    """Edge: `Hooks` with no inner, and a wrapper over a minimal observer that
    implements only ``emit`` (a passthrough impl is allowed to ignore
    ``close``). Neither may raise on shutdown."""

    class _EmitOnly:
        async def emit(self, obs):
            return None

    async def go():
        await Hooks().close()
        await Hooks(_EmitOnly()).close()
        await PolicyObserver(_EmitOnly()).close()

    _run(go())  # no AttributeError / TypeError


def test_hooks_close_does_not_swallow_an_inner_flush_failure():
    """``emit`` deliberately swallows handler errors so a hook can't
    destabilise a run. ``close`` is the opposite case — an explicit shutdown
    whose whole job is flushing a tail. Hiding a failure there would lose the
    buffered data with no signal."""

    class _BadClose:
        async def emit(self, obs):
            return None

        async def close(self):
            raise RuntimeError("flush failed")

    with pytest.raises(RuntimeError, match="flush failed"):
        _run(Hooks(_BadClose()).close())
