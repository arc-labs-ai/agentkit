"""Generic in-process pub/sub for stream-scoped events.

The mechanism is framework-level. The TYPING (what counts as an "event")
is application-level — instantiate as ``EventBus[MyEventType]`` at the
consumption site. The bus itself doesn't care about event content.

The bus is a thin in-memory shim; the public surface is the upgrade
path. Swap this for Redis Streams / NATS / Kafka behind the same
``publish`` / ``subscribe`` / ``close_stream`` interface for distributed
deployments. Nothing about the call sites changes.

Backpressure model
------------------
Bounded per-subscriber queues (``queue_size``). When a subscriber's
queue overflows on publish, **the offending event is dropped FOR THAT
SUBSCRIBER ONLY**; other subscribers and the publisher continue. The
drop is logged with the subscriber's ``name`` so production can spot
slow consumers without needing a separate trace.

Replay model
------------
Each stream carries a ring buffer of the most recent ``ring_size``
events (default 1024). A new subscriber may request ``from_version=K``
to replay events ``K..now`` before joining the live stream. Beyond the
ring's window the caller must reconstruct from durable storage (out of
scope for the bus — this is "pragmatic decoupling", not event sourcing).

Race-free subscribe/replay
--------------------------
Replay-then-subscribe with concurrent publishers risks one of two
defects: lose events (replay sees N, queue attached after publish of
N+1) or duplicate events (replay sees N, queue receives N too). Both
are closed by serialising "stamp version + append to log + deliver to
live queues + (if we're in subscribe) snapshot the replay" under a
single ``asyncio.Lock`` per channel. Subscribe holds the lock long
enough to *copy* the replay slice and attach the queue, then releases;
further publishes are caught by the queue. Replay yields happen
*outside* the lock so a slow subscriber doesn't back-pressure
publishers.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

E = TypeVar("E")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionedEvent(Generic[E]):
    """An event with a monotonic version stamped by the bus at publish
    time. Subscribers use the version to dedupe + resume from a known
    point — the version is internal bus bookkeeping and does **not**
    cross any wire by itself (callers unwrap ``.event`` before sending).
    """

    version: int
    event: E


@dataclass
class _StreamChannel(Generic[E]):
    """Per-stream state inside the bus: the version counter, the
    in-memory ring buffer (capped at ``ring_size``), the live subscriber
    queues, and the serialising lock.

    One channel per ``stream_id``. The bus never shares state across
    streams — version counters and subscriber lists are scoped per
    channel.
    """

    versions: int = 0
    log: deque[VersionedEvent[E]] = field(default_factory=deque)
    subscribers: list[tuple[str, asyncio.Queue[VersionedEvent[E] | object]]] = field(
        default_factory=list
    )
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False


# Sentinel pushed into subscriber queues by ``close_stream`` so the
# ``async for`` loop exits cleanly. Using a singleton (not an exception)
# avoids stack-trace noise on the consumer side.
_CLOSE_SENTINEL: object = object()


class EventBus(Generic[E]):
    """In-process pub/sub bus, generic over the event type ``E``.

    One instance per app/host. Producers call ``publish``; consumers
    call ``subscribe`` and iterate. ``close_stream`` ends every
    in-flight subscriber's loop on terminal transitions / shutdown.

    Create with ``EventBus[MyEventType]()`` to type the publish /
    subscribe surface. The mechanism itself is fully content-agnostic.
    """

    def __init__(self, *, ring_size: int = 1024, queue_size: int = 256) -> None:
        self._channels: dict[str, _StreamChannel[E]] = {}
        self._ring_size = ring_size
        self._queue_size = queue_size
        # Guards channel-dict mutation. Per-channel locks guard the
        # publish-subscribe race; this one guards channel creation /
        # removal only. Two distinct concerns, two distinct locks.
        self._channels_lock = asyncio.Lock()

    # ── public surface ────────────────────────────────────────────────

    async def publish(self, stream_id: str, event: E) -> VersionedEvent[E]:
        """Stamp a version, append to the stream's ring buffer, and
        deliver to every live subscriber's queue. Returns the versioned
        event so callers can log or expose the version.

        Drops on a per-subscriber basis: a full subscriber queue causes
        *that* subscriber's event to be dropped (logged), not the
        publish itself. The publisher never blocks waiting for a slow
        downstream.
        """
        channel = await self._get_or_create_channel(stream_id)
        async with channel.lock:
            channel.versions += 1
            versioned: VersionedEvent[E] = VersionedEvent(version=channel.versions, event=event)
            channel.log.append(versioned)
            # Snapshot the subscribers list inside the lock so a
            # concurrent subscribe() can't see a half-distributed
            # publish. Iteration + put_nowait happen outside any await,
            # so we hold the lock briefly.
            for name, q in channel.subscribers:
                try:
                    q.put_nowait(versioned)
                except asyncio.QueueFull:
                    logger.warning(
                        "EventBus: dropping event v=%d for slow subscriber %r on stream %s",
                        versioned.version,
                        name,
                        stream_id,
                    )
        return versioned

    def subscribe(
        self,
        stream_id: str,
        *,
        name: str = "anon",
        from_version: int = 0,
        replay: bool = True,
    ) -> AsyncIterator[VersionedEvent[E]]:
        """Return an async iterator over events for ``stream_id``.

        Replay semantics:

        - ``replay=True`` (default) — yields the replay slice first
          (events in the ring with ``version >= from_version``), then
          switches to live events. ``from_version=0`` replays the entire
          available ring; ``from_version=K`` resumes from version K.
        - ``replay=False`` — skips the replay slice entirely and yields
          only events published *after* the subscriber attaches. The
          ``from_version`` argument is ignored in this mode. Use this
          for "live-only" subscribers (e.g. a fresh WebSocket fan-out
          that has its own baseline snapshot and doesn't want historical
          deltas behind it).

        The returned iterator's ``finally`` block unsubscribes; consumers
        just ``async for ve in bus.subscribe(...)`` and break out when
        done.

        Pass ``name`` something identifying ("checkpointer", "ws:<id>")
        — drop logs key off it.
        """
        return self._iter_subscription(
            stream_id, name=name, from_version=from_version, replay=replay
        )

    async def close_stream(self, stream_id: str) -> None:
        """Tear down a stream's channel: push the close sentinel into
        every subscriber queue (their ``async for`` loops exit) and
        drop the channel. Called from the application's terminal phase
        and from ``shutdown``.

        Idempotent — closing an already-closed (or never-opened) stream
        is a no-op. Subsequent ``publish`` calls create a fresh channel,
        which is harmless if the application is genuinely past the
        terminal phase (it shouldn't be).
        """
        async with self._channels_lock:
            channel = self._channels.pop(stream_id, None)
        if channel is None:
            return
        async with channel.lock:
            channel.closed = True
            for name, q in channel.subscribers:
                # Guarantee the sentinel lands. A best-effort
                # ``put_nowait`` + log on a full queue would leave
                # ``_iter_subscription`` blocked forever on
                # ``await q.get()`` while ``subscribers.clear()`` strands
                # the coroutine — one task leaked per slow subscriber on
                # graceful shutdown.
                #
                # Strategy: try put_nowait first (the fast path). On
                # ``QueueFull`` drop the OLDEST queued event to make room
                # for the sentinel — the subscriber loses one live event
                # but exits cleanly. If their queue was full at close
                # time they were already lossy; the sentinel is the
                # load-bearing message and must always arrive.
                while True:
                    try:
                        q.put_nowait(_CLOSE_SENTINEL)
                        break
                    except asyncio.QueueFull:
                        try:
                            dropped = q.get_nowait()
                        except asyncio.QueueEmpty:  # pragma: no cover — racy edge
                            # The queue went from full to empty between
                            # put and get; loop and retry the put.
                            continue
                        logger.warning(
                            "EventBus: dropped queued event for slow subscriber %r on stream %s "
                            "to make room for close sentinel (was %r)",
                            name,
                            stream_id,
                            type(dropped).__name__,
                        )
            channel.subscribers.clear()

    async def shutdown(self) -> None:
        """Close every channel. Called from the application's lifespan
        exit so any consumer tasks wrapped around ``subscribe()`` unwind
        cleanly before the host process terminates."""
        async with self._channels_lock:
            stream_ids = list(self._channels.keys())
        for stream_id in stream_ids:
            await self.close_stream(stream_id)

    # ── internals ─────────────────────────────────────────────────────

    async def _get_or_create_channel(self, stream_id: str) -> _StreamChannel[E]:
        async with self._channels_lock:
            channel = self._channels.get(stream_id)
            if channel is None:
                channel = _StreamChannel[E](
                    log=deque(maxlen=self._ring_size),
                )
                self._channels[stream_id] = channel
            return channel

    async def _iter_subscription(
        self, stream_id: str, *, name: str, from_version: int, replay: bool
    ) -> AsyncIterator[VersionedEvent[E]]:
        """The actual async generator behind ``subscribe``.

        Sequence:
        1. Take the channel lock.
        2. Copy the replay slice (events in the ring with
           ``version >= from_version``) — *or* an empty list when
           ``replay=False`` (live-only subscribers).
        3. Create a fresh queue and add it to ``subscribers`` — every
           publish from this point on is delivered to us.
        4. Release the lock.
        5. Yield the replay slice (outside the lock so a slow consumer
           can't back-pressure publishers).
        6. Yield live events from the queue until the sentinel arrives
           or the consumer cancels.

        Step 3 happening under the same lock as ``publish`` is the
        race-closer: any publish during steps 1–3 is either visible in
        the replay slice (already in the ring) or delivered to the
        freshly-attached queue. Never both, never neither. The
        ``replay=False`` path still snapshots-and-attaches inside the
        lock — we simply discard the snapshot — so the race-free
        attach-point is identical: any publish concurrent with subscribe
        ends up in the queue.
        """
        channel = await self._get_or_create_channel(stream_id)
        q: asyncio.Queue[VersionedEvent[E] | object] = asyncio.Queue(maxsize=self._queue_size)
        async with channel.lock:
            replay_slice: list[VersionedEvent[E]] = (
                [ve for ve in channel.log if ve.version >= from_version] if replay else []
            )
            channel.subscribers.append((name, q))

        try:
            for ve in replay_slice:
                yield ve
            while True:
                item = await q.get()
                if item is _CLOSE_SENTINEL:
                    return
                # Narrow: anything that's not the sentinel is a
                # VersionedEvent[E] we put in ourselves.
                assert isinstance(item, VersionedEvent)
                yield item
        finally:
            await self._unsubscribe(stream_id, q)

    async def _unsubscribe(
        self,
        stream_id: str,
        q: asyncio.Queue[VersionedEvent[E] | object],
    ) -> None:
        async with self._channels_lock:
            channel = self._channels.get(stream_id)
        if channel is None:
            return
        async with channel.lock:
            channel.subscribers = [(n, sq) for n, sq in channel.subscribers if sq is not q]


__all__ = ["EventBus", "VersionedEvent"]
