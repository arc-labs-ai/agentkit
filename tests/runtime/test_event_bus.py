"""Unit tests for the generic ``EventBus`` mechanism.

The bus is in-process pub/sub, so every test runs inside a single event
loop with the bus + a handful of subscriber tasks. We use
``asyncio.Event`` for synchronisation instead of ``asyncio.sleep``
races; the bus's contract is "any publish during subscribe is visible
exactly once", and proving that requires deterministic hand-offs.

The bus is generic over the event type; here we parameterise with a
plain dataclass to keep the tests independent of any application's
event union. Application-level (typed) views are tested where the
typed alias lives — these tests cover only the mechanism.

Coverage:
- monotonic per-stream version stamping (independent counters)
- replay from version 0 + from an arbitrary version
- subscribe-during-publish race: every event visible exactly once
- slow-consumer drop (per-subscriber, doesn't affect siblings)
- two concurrent subscribers both see every event
- close_stream terminates ``async for`` loops cleanly
- close_stream is idempotent
- shutdown closes every stream
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import pytest

from agentkit.runtime.event_bus import EventBus, VersionedEvent


@dataclass(frozen=True)
class _TestEvent:
    """Minimal event type for parameterising ``EventBus``. The bus
    treats events opaquely; only the wrapping ``VersionedEvent.version``
    is bus-managed."""

    data: str


def _ev(data: str = "x") -> _TestEvent:
    return _TestEvent(data=data)


# ── version stamping ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_publish_stamps_monotonic_versions() -> None:
    bus: EventBus[_TestEvent] = EventBus()
    v1 = await bus.publish("stream_a", _ev("a"))
    v2 = await bus.publish("stream_a", _ev("b"))
    v3 = await bus.publish("stream_a", _ev("c"))

    assert v1.version == 1
    assert v2.version == 2
    assert v3.version == 3


@pytest.mark.asyncio
async def test_version_counters_are_per_stream() -> None:
    """Two streams never share a counter — concurrent streams each
    start at 1."""
    bus: EventBus[_TestEvent] = EventBus()
    a1 = await bus.publish("stream_a", _ev())
    b1 = await bus.publish("stream_b", _ev())
    a2 = await bus.publish("stream_a", _ev())
    b2 = await bus.publish("stream_b", _ev())

    assert (a1.version, a2.version) == (1, 2)
    assert (b1.version, b2.version) == (1, 2)


# ── replay semantics ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_from_zero_replays_then_lives() -> None:
    """A subscriber asking for from_version=0 sees the entire ring,
    then live events. The replay happens before any live event so the
    iterator's ordering is monotonic by version."""
    bus: EventBus[_TestEvent] = EventBus()
    await bus.publish("stream_a", _ev("1"))
    await bus.publish("stream_a", _ev("2"))

    received: list[VersionedEvent[_TestEvent]] = []
    done = asyncio.Event()

    async def consumer() -> None:
        async for ve in bus.subscribe("stream_a", name="t", from_version=0):
            received.append(ve)
            if ve.version >= 3:
                done.set()
                return

    task = asyncio.create_task(consumer())
    # Spin once so the consumer can attach. Then publish the live event.
    await asyncio.sleep(0)
    await bus.publish("stream_a", _ev("3"))
    await asyncio.wait_for(done.wait(), timeout=1.0)
    await task

    assert [ve.version for ve in received] == [1, 2, 3]


@pytest.mark.asyncio
async def test_subscribe_from_version_n_skips_earlier_events() -> None:
    bus: EventBus[_TestEvent] = EventBus()
    for i in range(1, 6):
        await bus.publish("stream_a", _ev(str(i)))

    received: list[int] = []

    async def consumer() -> None:
        async for ve in bus.subscribe("stream_a", name="t", from_version=4):
            received.append(ve.version)
            if len(received) == 2:
                return

    task = asyncio.create_task(consumer())
    await asyncio.wait_for(task, timeout=1.0)

    # from_version=4 means versions 4 and 5 only — first three are
    # before the cursor.
    assert received == [4, 5]


# ── live-only subscription (replay=False) ──────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_with_replay_false_skips_log() -> None:
    """``replay=False`` attaches the subscriber to live events only —
    the ring buffer's existing contents are skipped, but every publish
    *after* the attach point is delivered.

    This is the WebSocket fan-out shape: the api sends its own baseline
    snapshot first, then wants only the deltas that arrive from now on.
    """
    bus: EventBus[_TestEvent] = EventBus()
    # Three events already in the ring before anyone subscribes.
    await bus.publish("stream_a", _ev("1"))
    await bus.publish("stream_a", _ev("2"))
    await bus.publish("stream_a", _ev("3"))

    received: list[int] = []
    attached = asyncio.Event()
    done = asyncio.Event()

    async def consumer() -> None:
        async for ve in bus.subscribe("stream_a", name="live", replay=False):
            if not attached.is_set():
                attached.set()
            received.append(ve.version)
            done.set()
            return

    task = asyncio.create_task(consumer())
    # We can't observe "attached" until at least one event has been
    # delivered (the subscriber blocks on q.get()), so emit the live
    # event after a single event-loop spin to let the subscribe-coro
    # reach the lock-protected attach point.
    await asyncio.sleep(0)
    v4 = await bus.publish("stream_a", _ev("4"))

    await asyncio.wait_for(done.wait(), timeout=1.0)
    await task

    # Only version 4 — the three pre-existing events are absent because
    # the replay slice was suppressed.
    assert received == [v4.version] == [4]


# ── race: publish during subscribe ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_during_publish_loses_no_events() -> None:
    """The load-bearing race: a publisher hammering while a new
    subscriber joins must end up with every event visible exactly once
    (either via replay slice or via the live queue). The bus's
    per-channel lock makes this deterministic.

    We drive it by publishing a wave of events from one task while
    starting a subscriber in parallel; the subscriber must see every
    version 1..N in order."""
    bus: EventBus[_TestEvent] = EventBus()

    total = 50

    async def publisher() -> None:
        for _ in range(total):
            await bus.publish("stream_a", _ev())
            # Yield so the subscriber-attach can interleave.
            await asyncio.sleep(0)

    received: list[int] = []

    async def consumer() -> None:
        # We know the publisher will produce exactly `total` events;
        # exit as soon as we've seen them. Avoids the "is publisher
        # done?" handshake (which has a window where the consumer can
        # consume the last event before the done-flag is set).
        async for ve in bus.subscribe("stream_a", name="late", from_version=0):
            received.append(ve.version)
            if len(received) >= total:
                return

    pub = asyncio.create_task(publisher())
    # Start the consumer mid-publish — partway through, so the replay
    # slice + the live queue both have to contribute.
    await asyncio.sleep(0)  # let publisher emit a few
    cons = asyncio.create_task(consumer())

    await asyncio.wait_for(pub, timeout=2.0)
    await asyncio.wait_for(cons, timeout=2.0)

    assert received == list(range(1, total + 1)), (
        f"expected monotonic 1..{total}, got {received[:10]}…{received[-10:]}"
    )


# ── backpressure / slow consumer ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_slow_consumer_drops_without_blocking_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A subscriber that doesn't drain its queue overflows and the bus
    drops events FOR THAT SUBSCRIBER ONLY. A fast subscriber in
    parallel sees every event.

    We don't drain the slow one until the fast one is finished, so the
    test doesn't depend on timing — only on the bus refusing to block
    a publisher behind a full queue."""
    bus: EventBus[_TestEvent] = EventBus(queue_size=4)

    fast_received: list[int] = []
    fast_attached = asyncio.Event()
    slow_attached = asyncio.Event()
    fast_done = asyncio.Event()
    drain_slow = asyncio.Event()

    async def fast() -> None:
        # Subscribe with from_version > any future version so we don't
        # replay; we want to count live events only.
        async for ve in bus.subscribe("stream_a", name="fast", from_version=1):
            if not fast_attached.is_set():
                fast_attached.set()
            fast_received.append(ve.version)
            if ve.version >= 20:
                fast_done.set()
                return

    slow_received: list[int] = []

    async def slow() -> None:
        async for ve in bus.subscribe("stream_a", name="slow", from_version=1):
            if not slow_attached.is_set():
                slow_attached.set()
                # Block here until the test releases us — every event
                # published before then piles up in our bounded queue
                # and the overflow gets dropped.
                await drain_slow.wait()
            slow_received.append(ve.version)
            if len(slow_received) >= 4:
                return

    fast_task = asyncio.create_task(fast())
    slow_task = asyncio.create_task(slow())

    # Wait until both are attached so the publish loop hits both. We
    # need at least one publish to wake them; iterate until both
    # ``async for`` loops have actually entered.
    await bus.publish("stream_a", _ev())  # version 1 — wakes both
    await asyncio.wait_for(fast_attached.wait(), timeout=1.0)
    await asyncio.wait_for(slow_attached.wait(), timeout=1.0)

    # Now publish a flood — fast drains, slow doesn't. With queue_size=4
    # and slow blocked, every event past the 4th is dropped for slow.
    # ``sleep(0)`` after each publish gives the fast consumer a turn on
    # the event loop so its queue stays drained; without it the
    # publisher can win the loop and the fast queue can overflow too.
    with caplog.at_level(logging.WARNING, logger="agentkit.runtime.event_bus"):
        for _ in range(2, 21):
            await bus.publish("stream_a", _ev())
            await asyncio.sleep(0)

        await asyncio.wait_for(fast_done.wait(), timeout=1.0)

    # The fast consumer must have seen every event.
    assert fast_received == list(range(1, 21))

    # Drops were logged for "slow" — concrete proof of per-consumer
    # isolation. We don't pin the exact count (queue overflow timing is
    # implementation-dependent), only that at least one drop happened.
    assert any("slow" in rec.message for rec in caplog.records), (
        f"expected at least one drop log mentioning 'slow', got {caplog.records}"
    )

    # Release the slow consumer; it drains whatever's still in its
    # queue. The contents are a subset of all events; the test is that
    # nothing exploded and the fast peer was unaffected.
    drain_slow.set()
    await asyncio.wait_for(slow_task, timeout=1.0)
    await fast_task


# ── two concurrent subscribers ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_subscribers_both_see_every_event() -> None:
    bus: EventBus[_TestEvent] = EventBus()

    a_received: list[int] = []
    b_received: list[int] = []
    a_attached = asyncio.Event()
    b_attached = asyncio.Event()
    a_done = asyncio.Event()
    b_done = asyncio.Event()

    async def consumer_a() -> None:
        async for ve in bus.subscribe("stream_a", name="a", from_version=1):
            if not a_attached.is_set():
                a_attached.set()
            a_received.append(ve.version)
            if ve.version >= 5:
                a_done.set()
                return

    async def consumer_b() -> None:
        async for ve in bus.subscribe("stream_a", name="b", from_version=1):
            if not b_attached.is_set():
                b_attached.set()
            b_received.append(ve.version)
            if ve.version >= 5:
                b_done.set()
                return

    task_a = asyncio.create_task(consumer_a())
    task_b = asyncio.create_task(consumer_b())
    # Prime — first publish wakes both into their ``async for`` body.
    await bus.publish("stream_a", _ev())
    await asyncio.wait_for(a_attached.wait(), timeout=1.0)
    await asyncio.wait_for(b_attached.wait(), timeout=1.0)

    for _ in range(4):
        await bus.publish("stream_a", _ev())

    await asyncio.wait_for(a_done.wait(), timeout=1.0)
    await asyncio.wait_for(b_done.wait(), timeout=1.0)
    await task_a
    await task_b

    assert a_received == [1, 2, 3, 4, 5]
    assert b_received == [1, 2, 3, 4, 5]


# ── close_stream / shutdown ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_stream_terminates_subscribers_cleanly() -> None:
    """An in-flight ``async for`` exits on ``close_stream`` without
    raising — the bus pushes a sentinel and the iterator returns. No
    StopAsyncIteration leak; no CancelledError needed."""
    bus: EventBus[_TestEvent] = EventBus()
    attached = asyncio.Event()
    completed = asyncio.Event()

    async def consumer() -> None:
        async for _ in bus.subscribe("stream_a", name="c", from_version=0):
            attached.set()
        completed.set()

    task = asyncio.create_task(consumer())
    # Publish one event so the consumer enters its loop body (sets
    # ``attached``). Then close the stream; the sentinel ends the loop.
    await bus.publish("stream_a", _ev())
    await asyncio.wait_for(attached.wait(), timeout=1.0)
    await bus.close_stream("stream_a")
    await asyncio.wait_for(completed.wait(), timeout=1.0)
    await task


@pytest.mark.asyncio
async def test_close_stream_is_idempotent() -> None:
    bus: EventBus[_TestEvent] = EventBus()
    await bus.close_stream("never_existed")  # no error
    await bus.publish("stream_a", _ev())
    await bus.close_stream("stream_a")
    await bus.close_stream("stream_a")  # second close is a no-op


@pytest.mark.asyncio
async def test_shutdown_closes_all_streams() -> None:
    bus: EventBus[_TestEvent] = EventBus()
    closed_a = asyncio.Event()
    closed_b = asyncio.Event()
    attached_a = asyncio.Event()
    attached_b = asyncio.Event()

    async def consumer(stream_id: str, attached: asyncio.Event, closed: asyncio.Event) -> None:
        async for _ in bus.subscribe(stream_id, name=stream_id, from_version=0):
            attached.set()
        closed.set()

    ta = asyncio.create_task(consumer("stream_a", attached_a, closed_a))
    tb = asyncio.create_task(consumer("stream_b", attached_b, closed_b))
    await bus.publish("stream_a", _ev())
    await bus.publish("stream_b", _ev())
    await asyncio.wait_for(attached_a.wait(), timeout=1.0)
    await asyncio.wait_for(attached_b.wait(), timeout=1.0)

    await bus.shutdown()

    await asyncio.wait_for(closed_a.wait(), timeout=1.0)
    await asyncio.wait_for(closed_b.wait(), timeout=1.0)
    await ta
    await tb


# ── close_stream sentinel must arrive even on a full subscriber queue ────────
#
# If ``close_stream`` did ``put_nowait(_CLOSE_SENTINEL)`` and dropped the
# sentinel on ``QueueFull``, ``_iter_subscription`` would block forever on
# ``await q.get()`` and ``subscribers.clear()`` would strand the coroutine —
# graceful shutdown would leak one task per slow subscriber. Instead,
# close drops the oldest queued event to make room for the sentinel; the
# subscriber loses one live event but exits cleanly.


@pytest.mark.asyncio
async def test_close_stream_sentinel_lands_when_subscriber_queue_full() -> None:
    """A subscriber that NEVER pulls from its queue (simulating a wedged
    consumer) still receives the close sentinel — the sentinel triggers
    loop exit, the subscriber doesn't leak."""
    bus: EventBus[_TestEvent] = EventBus(queue_size=4)
    received: list[VersionedEvent[_TestEvent]] = []
    sub_attached = asyncio.Event()
    consumer_can_drain = asyncio.Event()
    consumer_exited = asyncio.Event()

    async def consumer() -> None:
        # Iterate, but block before pulling the first event so the
        # queue fills up.
        async for ve in bus.subscribe("s", name="wedge", from_version=0):
            if not sub_attached.is_set():
                sub_attached.set()
                # Wait until the test fires the gate; meanwhile the
                # queue fills.
                await consumer_can_drain.wait()
            received.append(ve)
        consumer_exited.set()

    # Subscribe with replay so a first publish lands in replay and the
    # consumer enters the live loop with the attached flag set.
    await bus.publish("s", _ev("seed"))
    task = asyncio.create_task(consumer())
    await asyncio.wait_for(sub_attached.wait(), timeout=1.0)

    # Now overflow the queue: queue_size=4, blast 12 publishes. The
    # slow-consumer log fires per drop; the queue settles at 4 items.
    for i in range(12):
        await bus.publish("s", _ev(f"overflow-{i}"))

    # Close the stream. The bus drops the oldest queued event to make
    # room, so the sentinel lands and the consumer exits cleanly.
    await bus.close_stream("s")

    # Release the consumer's gate so it can drain remaining items + the
    # sentinel.
    consumer_can_drain.set()
    await asyncio.wait_for(consumer_exited.wait(), timeout=2.0)
    await task  # no leak


@pytest.mark.asyncio
async def test_close_stream_sentinel_arrives_even_when_consumer_actively_drains() -> None:
    """Regression — happy path where the queue is empty at close time
    still works (sentinel goes via the fast put_nowait branch)."""
    bus: EventBus[_TestEvent] = EventBus()
    exited = asyncio.Event()
    attached = asyncio.Event()

    async def consumer() -> None:
        async for _ in bus.subscribe("s", name="fast", from_version=0):
            attached.set()
        exited.set()

    task = asyncio.create_task(consumer())
    await bus.publish("s", _ev())
    await asyncio.wait_for(attached.wait(), timeout=1.0)

    await bus.close_stream("s")
    await asyncio.wait_for(exited.wait(), timeout=1.0)
    await task


@pytest.mark.asyncio
async def test_close_stream_logs_warning_when_dropping_to_make_room(caplog) -> None:
    """When close_stream drops a queued event to make room for the
    sentinel, it must log a warning so operators can see slow
    subscribers in production."""
    bus: EventBus[_TestEvent] = EventBus(queue_size=2)
    sub_attached = asyncio.Event()
    can_drain = asyncio.Event()

    async def consumer() -> None:
        async for _ in bus.subscribe("s", name="slow", from_version=0):
            if not sub_attached.is_set():
                sub_attached.set()
                await can_drain.wait()

    await bus.publish("s", _ev("seed"))
    task = asyncio.create_task(consumer())
    await asyncio.wait_for(sub_attached.wait(), timeout=1.0)

    # Overflow.
    for i in range(8):
        await bus.publish("s", _ev(f"x-{i}"))

    with caplog.at_level(logging.WARNING, logger="agentkit.runtime.event_bus"):
        await bus.close_stream("s")
    # Some warning specifically about the slow subscriber dropped event
    # — pin the keyword so refactors keep operator messaging consistent.
    assert any(
        "slow subscriber" in r.message and "close sentinel" in r.message for r in caplog.records
    )

    can_drain.set()
    await task
