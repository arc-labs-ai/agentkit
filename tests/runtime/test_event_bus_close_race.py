"""Late and racing subscribers must never hang on a closed ``EventBus`` stream.

``close_stream`` pops the channel out of ``EventBus._channels``, pushes
the close sentinel to the subscribers it can see, and clears the list.
Two subscribers are invisible to that sweep:

- one that calls ``subscribe`` *after* the pop — it gets a brand-new
  channel that nobody will ever close;
- one caught *mid-attach* — it obtained the channel object before the
  pop and appends its queue after the sweep already cleared the list.

Both used to park on ``await q.get()`` forever: one leaked task per
late subscriber, which is precisely the leak ``close_stream``'s sentinel
guarantee was written to prevent, just moved one code path over.
Measured with a probe before the fix:

    subscribe then close      : ended cleanly    <- the covered path
    subscribe AFTER close     : HUNG (had to be cancelled)
    shutdown racing subscribe : HUNG (had to be cancelled)

Determinism
-----------
The failure mode is an infinite hang, so nothing here may rely on
``sleep`` to lose a race: a sleep-timed test passes on an idle laptop
and wedges CI under load. The mid-attach race is instead forced with
``_AttachGate``, which pauses a subscriber at the exact instruction
between "obtained the channel" and "took the channel lock" — the window
the bug lives in — and every wait is bounded by ``_TRIPWIRE`` so a
regression FAILS with a TimeoutError instead of hanging the suite.
``@pytest.mark.timeout`` is the outer net for the same reason.

The POSITIVE CONTROLS section at the bottom is load-bearing: those tests
pass both before and after the fix, and exist to prove the guard is
targeted rather than a blanket "subscribe returns immediately", which
would silently disable the bus while making every regression test green.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.runtime.event_bus import EventBus, VersionedEvent

# Every wait in this module is bounded. 2s is ~1000x the real hand-off
# time (these tests are pure event-loop work with no I/O), so it cannot
# flake under CI load, while still failing fast on a genuine hang.
_TRIPWIRE = 2.0


@dataclass(frozen=True)
class _TestEvent:
    """Minimal event type for parameterising ``EventBus``."""

    data: str


def _ev(data: str = "x") -> _TestEvent:
    return _TestEvent(data=data)


class _AttachGate:
    """Freeze a subscriber inside the mid-attach window.

    ``_iter_subscription`` does two separately-locked steps: fetch the
    channel (under the channel-dict lock), then attach its queue (under
    that channel's own lock). The bug lives strictly between them —
    ``close_stream`` can pop and drain the channel while the subscriber
    holds a reference to it and has not yet appended.

    Patching the *instance* (not the class) keeps the gate scoped to one
    bus, and ``*args/**kwargs`` passthrough keeps it agnostic to the
    method's signature, so the same gate drives the buggy and the fixed
    implementation identically.
    """

    def __init__(self, bus: EventBus[_TestEvent]) -> None:
        self._real = bus._get_or_create_channel
        self.armed = False
        self.paused = asyncio.Event()
        self.release = asyncio.Event()
        bus._get_or_create_channel = self._gated  # type: ignore[method-assign]

    async def _gated(self, stream_id: str, **kwargs: Any) -> Any:
        channel = await self._real(stream_id, **kwargs)
        if self.armed:
            # One-shot: only the subscriber we armed for gets frozen,
            # so a publish inside the gated window is not also stalled.
            self.armed = False
            self.paused.set()
            await self.release.wait()
        return channel


async def _drain(it: AsyncIterator[VersionedEvent[_TestEvent]]) -> list[int]:
    return [ve.version async for ve in it]


# ── REGRESSIONS: these fail (by TimeoutError) against the unfixed bus ───────


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_subscribe_after_close_terminates_instead_of_hanging() -> None:
    """The plain late attach. ``close_stream`` has already popped the
    channel, so ``subscribe`` manufactures a fresh one — with the bug,
    it then waits forever for a sentinel that was handed out before it
    existed."""
    bus: EventBus[_TestEvent] = EventBus()
    await bus.publish("s", _ev("1"))
    await bus.close_stream("s")

    versions = await asyncio.wait_for(_drain(bus.subscribe("s", name="late")), timeout=_TRIPWIRE)

    # Nothing to replay: the bus does not retain closed channels, so the
    # tombstone's ring is empty. The contract is "terminates", not
    # "resurrects history" — see the replay-slice comment in
    # ``_iter_subscription``.
    assert versions == []


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_subscribe_after_close_of_never_opened_stream_terminates() -> None:
    """``close_stream`` on an id nobody ever published to still means
    closed. Recording it unconditionally is what makes that true — the
    alternative ("a stream with no channel was never open, so ignore
    the close") leaves the same hang behind a different door."""
    bus: EventBus[_TestEvent] = EventBus()
    await bus.close_stream("never_opened")

    versions = await asyncio.wait_for(
        _drain(bus.subscribe("never_opened", name="late")), timeout=_TRIPWIRE
    )
    assert versions == []


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_subscribe_after_close_with_replay_false_terminates() -> None:
    """The live-only subscriber has no replay slice to fall back on, so
    its whole iteration is the part that used to block."""
    bus: EventBus[_TestEvent] = EventBus()
    await bus.publish("s", _ev("1"))
    await bus.close_stream("s")

    versions = await asyncio.wait_for(
        _drain(bus.subscribe("s", name="live-only", replay=False)), timeout=_TRIPWIRE
    )
    assert versions == []


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_close_racing_mid_attach_subscriber_terminates() -> None:
    """The nastier half of the bug: the subscriber holds the very
    channel ``close_stream`` pops. The dict lookup can no longer tell it
    anything — the entry is gone — so only the channel's own ``closed``
    flag can, which is exactly the state that was written and never
    read.

    The gate makes this exact, not probable: the subscriber is stopped
    with the channel in hand and the close runs underneath it."""
    bus: EventBus[_TestEvent] = EventBus()
    gate = _AttachGate(bus)
    await bus.publish("s", _ev("1"))
    await bus.publish("s", _ev("2"))

    received: list[int] = []

    async def consumer() -> None:
        async for ve in bus.subscribe("s", name="racer", from_version=0):
            received.append(ve.version)

    gate.armed = True
    task = asyncio.create_task(consumer())
    await asyncio.wait_for(gate.paused.wait(), timeout=_TRIPWIRE)

    # Subscriber is frozen holding the live channel; close it out from
    # under them. Its subscriber list is empty at this instant, so the
    # sentinel sweep reaches nobody.
    await bus.close_stream("s")
    gate.release.set()

    await asyncio.wait_for(task, timeout=_TRIPWIRE)

    # It still holds the real (popped) channel, so its replay slice is
    # the genuine history — the same thing it would have seen had it won
    # the lock by a microsecond. Losing the race is lossless, not lossy.
    assert received == [1, 2]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_shutdown_racing_mid_attach_subscriber_terminates() -> None:
    """Same race, reached through ``shutdown`` — the path that actually
    runs in production, from the application's lifespan exit. A leaked
    task here outlives the thing it was streaming for."""
    bus: EventBus[_TestEvent] = EventBus()
    gate = _AttachGate(bus)
    await bus.publish("s", _ev("1"))

    async def consumer() -> None:
        async for _ in bus.subscribe("s", name="racer", from_version=0):
            pass

    gate.armed = True
    task = asyncio.create_task(consumer())
    await asyncio.wait_for(gate.paused.wait(), timeout=_TRIPWIRE)

    await bus.shutdown()
    gate.release.set()

    await asyncio.wait_for(task, timeout=_TRIPWIRE)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_shutdown_with_many_streams_and_late_subscribers() -> None:
    """Shutdown at realistic shape: several streams, several subscribers
    each, plus a wave of subscribers that arrive after the whole bus is
    down. Every one of them must end; a single survivor is a leaked
    task holding the event loop open at process exit."""
    bus: EventBus[_TestEvent] = EventBus()
    streams = ["a", "b", "c"]
    attached = {s: asyncio.Event() for s in streams}
    exits: list[asyncio.Task[None]] = []

    async def consumer(stream_id: str) -> None:
        async for _ in bus.subscribe(stream_id, name=f"sub:{stream_id}", from_version=0):
            attached[stream_id].set()

    for stream_id in streams:
        exits.append(asyncio.create_task(consumer(stream_id)))
        exits.append(asyncio.create_task(consumer(stream_id)))
    for stream_id in streams:
        await bus.publish(stream_id, _ev())
        await asyncio.wait_for(attached[stream_id].wait(), timeout=_TRIPWIRE)

    await bus.shutdown()
    await asyncio.wait_for(asyncio.gather(*exits), timeout=_TRIPWIRE)

    # Now the late wave, after shutdown has completed.
    late = [_drain(bus.subscribe(s, name="late")) for s in streams]
    assert await asyncio.wait_for(asyncio.gather(*late), timeout=_TRIPWIRE) == [[], [], []]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_subscribe_does_not_reopen_a_closed_stream() -> None:
    """A read must not resurrect a stream. If ``subscribe`` cleared the
    closed record (or registered its channel), the *second* late
    subscriber would attach to a live channel with no publisher and no
    closer — the original hang, one call later."""
    bus: EventBus[_TestEvent] = EventBus()
    await bus.publish("s", _ev())
    await bus.close_stream("s")

    for attempt in range(3):
        versions = await asyncio.wait_for(
            _drain(bus.subscribe("s", name=f"late-{attempt}")), timeout=_TRIPWIRE
        )
        assert versions == []

    # And no channel was retained: the tombstone handed to each
    # subscriber is deliberately not stored, so late subscribers cannot
    # grow ``_channels`` without bound.
    assert "s" not in bus._channels
    assert "s" in bus._closed_ids


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_publish_reopens_a_closed_stream() -> None:
    """The other half of "don't poison a reused id": ``publish`` is an
    explicit statement that the id is live again, and the documented
    behaviour is that it creates a *fresh* channel (version counter back
    to 1). A subscriber joining afterwards must get real live delivery,
    not an instant EOF."""
    bus: EventBus[_TestEvent] = EventBus()
    await bus.publish("s", _ev("old"))
    await bus.close_stream("s")

    reopened = await bus.publish("s", _ev("b"))
    assert reopened.version == 1, "re-open must build a fresh channel, not resume the old one"

    got: list[str] = []
    first = asyncio.Event()
    second = asyncio.Event()

    async def consumer() -> None:
        async for ve in bus.subscribe("s", name="reopened", from_version=0):
            got.append(ve.event.data)
            (first if len(got) == 1 else second).set()

    task = asyncio.create_task(consumer())
    # "b" reaches it via replay of the fresh channel's ring.
    await asyncio.wait_for(first.wait(), timeout=_TRIPWIRE)
    # "c" reaches it live, which is only possible if it genuinely
    # attached rather than short-circuiting on a stale closed record.
    await bus.publish("s", _ev("c"))
    await asyncio.wait_for(second.wait(), timeout=_TRIPWIRE)

    # And the re-opened stream still closes normally.
    await bus.close_stream("s")
    await asyncio.wait_for(task, timeout=_TRIPWIRE)
    assert got == ["b", "c"]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_closed_stream_memory_is_bounded() -> None:
    """Stream ids are per-run, so remembering every close forever is
    just a slower leak. The record is ids only and FIFO-capped; here the
    cap is 4 so the eviction is checkable, in production it is 1024."""
    bus: EventBus[_TestEvent] = EventBus(closed_memory=4)
    for i in range(10):
        await bus.publish(f"s{i}", _ev())
        await bus.close_stream(f"s{i}")

    assert bus._closed_ids == {"s6", "s7", "s8", "s9"}
    assert set(bus._closed_order) == bus._closed_ids, "deque and set must not drift"
    assert len(bus._closed_order) == 4
    # No channel survived either — the whole point is that closing frees
    # the ring buffers and keeps only the ids.
    assert bus._channels == {}

    # Re-closing an id already recorded must not churn eviction order,
    # or a caller closing one stream repeatedly would evict every other
    # tombstone and reopen the hang for its neighbours.
    for _ in range(10):
        await bus.close_stream("s9")
    assert bus._closed_ids == {"s6", "s7", "s8", "s9"}

    # Re-opening drops the record from BOTH structures.
    await bus.publish("s7", _ev())
    assert bus._closed_ids == {"s6", "s8", "s9"}
    assert set(bus._closed_order) == bus._closed_ids


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_closed_memory_zero_disables_the_record_without_growing() -> None:
    """``closed_memory=0`` opts out of late-attach protection entirely
    (the mid-attach guard still applies, since that reads the channel's
    own flag). The set must stay empty rather than growing unbounded
    behind a zero-length deque."""
    bus: EventBus[_TestEvent] = EventBus(closed_memory=0)
    for i in range(5):
        await bus.publish(f"s{i}", _ev())
        await bus.close_stream(f"s{i}")

    assert bus._closed_ids == set()
    assert len(bus._closed_order) == 0


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_double_close_and_close_of_unknown_stream_stay_quiet() -> None:
    bus: EventBus[_TestEvent] = EventBus()
    await bus.close_stream("never_existed")
    await bus.close_stream("never_existed")
    await bus.publish("s", _ev())
    await bus.close_stream("s")
    await bus.close_stream("s")

    versions = await asyncio.wait_for(_drain(bus.subscribe("s", name="after")), timeout=_TRIPWIRE)
    assert versions == []


# ── POSITIVE CONTROLS: these pass BOTH before and after the fix ─────────────
#
# A guard that ends every subscription immediately would make everything
# above green while turning the bus into a no-op. These pin the
# behaviour that must NOT change.


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_control_subscribe_publish_close_still_works() -> None:
    """The ordinary lifecycle: attach, receive live events, exit on the
    close sentinel."""
    bus: EventBus[_TestEvent] = EventBus()
    got: list[int] = []
    ready = asyncio.Event()

    async def consumer() -> None:
        async for ve in bus.subscribe("s", name="normal", from_version=0):
            got.append(ve.version)
            if len(got) == 3:
                ready.set()

    task = asyncio.create_task(consumer())
    for _ in range(3):
        await bus.publish("s", _ev())
    await asyncio.wait_for(ready.wait(), timeout=_TRIPWIRE)
    await bus.close_stream("s")
    await asyncio.wait_for(task, timeout=_TRIPWIRE)

    assert got == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_control_race_free_subscribe_delivers_each_event_exactly_once() -> None:
    """The invariant the per-channel lock exists for: a publish
    concurrent with a subscribe is either in the replay slice or in the
    queue — never both, never neither. The close guard is read inside
    that same critical section, so it must not have widened it."""
    bus: EventBus[_TestEvent] = EventBus()
    total = 50

    async def publisher() -> None:
        for _ in range(total):
            await bus.publish("s", _ev())
            await asyncio.sleep(0)

    received: list[int] = []

    async def consumer() -> None:
        async for ve in bus.subscribe("s", name="racer", from_version=0):
            received.append(ve.version)
            if len(received) >= total:
                return

    pub = asyncio.create_task(publisher())
    await asyncio.sleep(0)
    cons = asyncio.create_task(consumer())
    await asyncio.wait_for(asyncio.gather(pub, cons), timeout=_TRIPWIRE)

    assert received == list(range(1, total + 1)), (
        f"expected monotonic 1..{total} exactly once, got {received[:10]}…{received[-10:]}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_control_from_version_and_replay_false() -> None:
    """Replay windowing is untouched: ``from_version=K`` starts at K,
    ``replay=False`` starts at the attach point."""
    bus: EventBus[_TestEvent] = EventBus()
    for _ in range(5):
        await bus.publish("s", _ev())

    # Pull the replay window by hand and close the iterator: the stream
    # is still live, so draining it to exhaustion would (correctly)
    # block on the next live event.
    resume_gen = bus.subscribe("s", name="resume", from_version=4)
    resumed = [(await asyncio.wait_for(resume_gen.__anext__(), timeout=_TRIPWIRE)).version
               for _ in range(2)]
    await resume_gen.aclose()
    assert resumed == [4, 5]

    got: list[int] = []
    seen = asyncio.Event()

    async def consumer() -> None:
        async for ve in bus.subscribe("s", name="live", replay=False):
            got.append(ve.version)
            seen.set()

    task = asyncio.create_task(consumer())
    # One loop turn is enough: attach takes two uncontended locks and an
    # empty replay walk, so the task reaches ``q.get()`` in its first
    # slice. Asserted rather than assumed — a scheduling surprise must
    # fail loudly here, not silently turn into a wrong ``got``.
    await asyncio.sleep(0)
    assert len(bus._channels["s"].subscribers) == 1, "live-only subscriber failed to attach"

    await bus.publish("s", _ev())
    await asyncio.wait_for(seen.wait(), timeout=_TRIPWIRE)
    await bus.close_stream("s")
    await asyncio.wait_for(task, timeout=_TRIPWIRE)

    assert got == [6], "replay=False must skip the ring and start at the attach point"


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_control_slow_subscriber_drops_only_its_own_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Per-subscriber backpressure: a full queue drops that
    subscriber's event, not the publish and not its siblings'."""
    bus: EventBus[_TestEvent] = EventBus(queue_size=4)
    fast_got: list[int] = []
    fast_attached = asyncio.Event()
    slow_attached = asyncio.Event()
    fast_done = asyncio.Event()
    drain_slow = asyncio.Event()

    async def fast() -> None:
        async for ve in bus.subscribe("s", name="fast", from_version=1):
            fast_attached.set()
            fast_got.append(ve.version)
            if ve.version >= 20:
                fast_done.set()
                return

    async def slow() -> None:
        async for _ in bus.subscribe("s", name="slow", from_version=1):
            if not slow_attached.is_set():
                slow_attached.set()
                await drain_slow.wait()

    fast_task = asyncio.create_task(fast())
    slow_task = asyncio.create_task(slow())
    await bus.publish("s", _ev())
    await asyncio.wait_for(fast_attached.wait(), timeout=_TRIPWIRE)
    await asyncio.wait_for(slow_attached.wait(), timeout=_TRIPWIRE)

    with caplog.at_level(logging.WARNING, logger="agentkit.runtime.event_bus"):
        for _ in range(2, 21):
            await bus.publish("s", _ev())
            await asyncio.sleep(0)
        await asyncio.wait_for(fast_done.wait(), timeout=_TRIPWIRE)

    assert fast_got == list(range(1, 21)), "the fast peer must be unaffected by the slow one"
    assert any("slow" in r.message for r in caplog.records)

    drain_slow.set()
    await bus.close_stream("s")
    await asyncio.wait_for(asyncio.gather(fast_task, slow_task), timeout=_TRIPWIRE)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_control_close_sentinel_still_lands_on_a_full_queue() -> None:
    """The pre-existing sentinel guarantee — close drops the oldest
    queued event to make room — must keep working. This is the path the
    original leak comment covers, and the fix must not have taken the
    subscriber off the list before it could be swept."""
    bus: EventBus[_TestEvent] = EventBus(queue_size=4)
    attached = asyncio.Event()
    can_drain = asyncio.Event()
    exited = asyncio.Event()

    async def consumer() -> None:
        async for _ in bus.subscribe("s", name="wedged", from_version=0):
            if not attached.is_set():
                attached.set()
                await can_drain.wait()
        exited.set()

    await bus.publish("s", _ev("seed"))
    task = asyncio.create_task(consumer())
    await asyncio.wait_for(attached.wait(), timeout=_TRIPWIRE)

    for i in range(12):
        await bus.publish("s", _ev(f"overflow-{i}"))

    await bus.close_stream("s")
    can_drain.set()
    await asyncio.wait_for(exited.wait(), timeout=_TRIPWIRE)
    await asyncio.wait_for(task, timeout=_TRIPWIRE)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_control_cancelled_subscriber_unsubscribes() -> None:
    """A subscriber cancelled mid-iteration must remove its queue from
    the channel, or every cancelled consumer leaves a queue the
    publisher keeps filling.

    Driven by hand rather than by cancelling a wrapping task: cancelling
    the task leaves the async generator to be finalised by the loop's
    asyncgen hook at some later tick, which is not something a
    deterministic test can wait on. Cancelling the ``__anext__`` raises
    inside the generator frame, so its ``finally`` runs synchronously as
    part of the unwind."""
    bus: EventBus[_TestEvent] = EventBus()
    agen = bus.subscribe("s", name="cancelme", from_version=0)

    await bus.publish("s", _ev())
    assert (await agen.__anext__()).version == 1
    assert len(bus._channels["s"].subscribers) == 1, "expected to be attached"

    pending = asyncio.create_task(agen.__anext__())
    await asyncio.sleep(0)  # park it on ``q.get()``
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=_TRIPWIRE)

    assert bus._channels["s"].subscribers == [], "cancellation must unsubscribe"

    # The stream is still usable by everyone else.
    later = await bus.publish("s", _ev())
    assert later.version == 2
    await bus.close_stream("s")
