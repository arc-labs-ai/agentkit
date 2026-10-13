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

Close semantics for late subscribers
------------------------------------
``close_stream`` pops the channel and pushes the sentinel to the
subscribers it can see at that instant. A subscriber that arrives
*after* that — or one caught mid-attach, still holding the channel
object ``close_stream`` just popped — would otherwise park on
``await q.get()`` waiting for a sentinel nobody will ever send: one
leaked task per late subscriber. That is the very leak the sentinel
guarantee inside ``close_stream`` was written to prevent, simply moved
one code path over.

Two guards close it, because there are two distinct failure shapes:

- ``_StreamChannel.closed`` is *read* under the channel lock, in the
  same critical section that attaches the queue. This catches the
  mid-attach race, where the subscriber holds the popped channel and
  the dict no longer knows about it.
- The bus keeps a **bounded** memory of recently closed stream ids, so a
  channel manufactured *after* the close is born closed. This catches
  the plain late attach, where the subscriber's channel object is a
  brand-new one that ``close_stream`` never saw.

Either way the subscriber yields whatever replay slice it actually
holds and then terminates cleanly. ``publish`` — and only ``publish`` —
clears the closed record, preserving the documented "publish after
close opens a fresh channel" behaviour; ``subscribe`` never un-closes a
stream, so a reused stream id is never permanently poisoned and a dead
one is never silently resurrected.
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
    # Set by ``close_stream`` *after* the channel has been popped out of
    # ``EventBus._channels``, and read by ``_iter_subscription`` under
    # ``lock``. It is the only thing that tells a subscriber holding a
    # popped channel that its queue will never be drained — the dict
    # lookup can no longer tell it, because the entry is already gone.
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

    def __init__(
        self, *, ring_size: int = 1024, queue_size: int = 256, closed_memory: int = 1024
    ) -> None:
        self._channels: dict[str, _StreamChannel[E]] = {}
        self._ring_size = ring_size
        self._queue_size = queue_size
        # Guards channel-dict mutation. Per-channel locks guard the
        # publish-subscribe race; this one guards channel creation /
        # removal only. Two distinct concerns, two distinct locks. It
        # also guards the closed-stream memory below, which is mutated
        # in exactly the same create/remove critical sections and must
        # stay consistent with the dict.
        self._channels_lock = asyncio.Lock()
        # Bounded memory of recently closed stream ids, so a subscriber
        # arriving after ``close_stream`` terminates instead of blocking
        # forever on a channel nobody will ever close again.
        #
        # Ids ONLY. A tombstone costs one stream-id string plus a deque
        # slot and a set slot — order 100 bytes. Retaining the closed
        # *channel* instead would cost its whole ring: up to
        # ``ring_size`` events, 1024 by default, i.e. three orders of
        # magnitude more per stream. Stream ids are typically per-run,
        # so unbounded retention of either would be a genuine leak;
        # 1024 tombstones is tens of KB.
        #
        # Why 1024 is the right bound: a tombstone only has to survive
        # the window between ``close_stream`` and the last racing
        # subscriber attaching, which is microseconds — the racing
        # subscriber is by construction already in flight. 1024 is
        # therefore not a "how many runs will this host see" number (it
        # would be far too small for that) but a "how many runs can
        # possibly close inside one attach window" number, where it is
        # enormous headroom. Past that bound the oldest tombstones are
        # forgotten and a very late subscriber to one of those ids falls
        # back to waiting on a live re-created stream: bounded memory
        # buys bounded protection, deliberately.
        self._closed_capacity = max(0, closed_memory)
        self._closed_order: deque[str] = deque(maxlen=self._closed_capacity)
        self._closed_ids: set[str] = set()

    # ── public surface ────────────────────────────────────────────────

    async def publish(self, stream_id: str, event: E) -> VersionedEvent[E]:
        """Stamp a version, append to the stream's ring buffer, and
        deliver to every live subscriber's queue. Returns the versioned
        event so callers can log or expose the version.

        Drops on a per-subscriber basis: a full subscriber queue causes
        *that* subscriber's event to be dropped (logged), not the
        publish itself. The publisher never blocks waiting for a slow
        downstream.

        Publishing to a stream that was previously closed re-opens it
        (``reopen=True``): a write is an explicit statement that the id
        is live again, and this is the documented "subsequent publish
        creates a fresh channel" behaviour. Subscribe deliberately does
        NOT do this — see ``_get_or_create_channel``.
        """
        channel = await self._get_or_create_channel(stream_id, reopen=True)
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

        Subscribing to a stream that is already closed (or that closes
        while this call is attaching) does not block: the iterator
        yields whatever replay slice is still available and then ends,
        so the ``async for`` completes normally rather than leaking a
        task waiting for a sentinel that will never arrive. It does not
        re-open the stream either — a subsequent ``publish`` is what
        re-opens a stream id.

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
        is a no-op beyond recording the id as closed. That recording is
        unconditional: "closed" is a statement about the stream id, not
        about whether anyone happened to publish to it first, and a
        subscriber arriving afterwards must terminate either way.

        Subsequent ``publish`` calls create a fresh channel and clear the
        closed record, which is harmless if the application is genuinely
        past the terminal phase (it shouldn't be). Subsequent
        ``subscribe`` calls do *not* re-open the stream; they terminate
        immediately instead of blocking for a sentinel that will never
        come.
        """
        async with self._channels_lock:
            channel = self._channels.pop(stream_id, None)
            # Record the close atomically with the pop. Any
            # ``_get_or_create_channel`` that runs after this critical
            # section must see a closed id rather than manufacture a
            # live channel for a stream that just ended — that gap is
            # exactly how a late subscriber used to acquire a brand-new
            # channel and block on it forever.
            self._remember_closed(stream_id)
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

    def _remember_closed(self, stream_id: str) -> None:
        """Record ``stream_id`` as closed, evicting the oldest record
        once ``closed_memory`` is reached. Caller must hold
        ``_channels_lock``.

        Two structures, on purpose: the deque gives FIFO eviction order,
        the set gives O(1) membership for the subscribe path. A linear
        scan of up to 1024 ids on every ``subscribe`` would put the
        bound's cost on the hot path, which is the wrong trade.
        """
        if self._closed_capacity == 0 or stream_id in self._closed_ids:
            # Re-closing an already-recorded id must not churn eviction
            # order: otherwise a caller that closes one stream in a loop
            # would evict every other tombstone behind it.
            return
        if len(self._closed_order) == self._closed_capacity:
            # ``deque(maxlen=...)`` drops the leftmost entry on append;
            # mirror that in the set or it grows without bound.
            self._closed_ids.discard(self._closed_order[0])
        self._closed_order.append(stream_id)
        self._closed_ids.add(stream_id)

    def _forget_closed(self, stream_id: str) -> None:
        """Drop the closed record for ``stream_id`` — a live channel is
        being created for it. Caller must hold ``_channels_lock``.

        ``deque.remove`` is O(n) in the bound (1024), which is fine
        because this only runs when a closed stream id is genuinely
        re-opened, not on the steady-state publish path.
        """
        if stream_id not in self._closed_ids:
            return
        self._closed_ids.discard(stream_id)
        try:
            self._closed_order.remove(stream_id)
        except ValueError:  # pragma: no cover — deque and set are kept in sync
            pass

    async def _get_or_create_channel(self, stream_id: str, *, reopen: bool) -> _StreamChannel[E]:
        """Fetch the live channel for ``stream_id``, creating one if
        needed.

        ``reopen`` splits the two callers, and the split is the whole
        point of the flag:

        - ``publish`` passes ``True``. A write says this stream id is in
          use again, so any closed record is cleared and the caller gets
          a live channel. Without this, one ``close_stream`` would
          poison a legitimately reused stream id for the life of the
          process.
        - ``subscribe`` passes ``False``. A read must never un-close a
          stream, so a remembered-closed id yields a *tombstone*:
          closed, empty, and deliberately NOT stored in
          ``self._channels``. Storing it would retain one dead channel
          per late-subscribed closed id forever — reintroducing the
          retention leak the id-only memory exists to avoid. The
          tombstone is garbage-collected along with the subscriber's
          generator.

        Invariant: a channel present in ``self._channels`` is never
        closed, and its id is never in the closed memory. ``close_stream``
        pops and records under this same lock, and the only creator is
        right here, so the two states cannot overlap.
        """
        async with self._channels_lock:
            channel = self._channels.get(stream_id)
            if channel is not None:
                return channel
            if not reopen and stream_id in self._closed_ids:
                return _StreamChannel[E](log=deque(maxlen=self._ring_size), closed=True)
            channel = _StreamChannel[E](
                log=deque(maxlen=self._ring_size),
            )
            self._channels[stream_id] = channel
            self._forget_closed(stream_id)
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

        Step 3 is also where the close guard lives: ``channel.closed`` is
        read in the same critical section, so attaching to a stream that
        is closing is decided atomically with attaching at all.
        """
        channel = await self._get_or_create_channel(stream_id, reopen=False)
        q: asyncio.Queue[VersionedEvent[E] | object] = asyncio.Queue(maxsize=self._queue_size)
        async with channel.lock:
            replay_slice: list[VersionedEvent[E]] = (
                [ve for ve in channel.log if ve.version >= from_version] if replay else []
            )
            # Read ``closed`` inside the attach critical section. Two
            # ways to get here with it set: this is the very channel
            # ``close_stream`` popped moments ago (its subscriber list
            # was already drained and cleared, so appending would park
            # us on a sentinel that has already been handed out), or it
            # is a tombstone minted for a stream id the bus remembers
            # closing. Both must not attach.
            attached = not channel.closed
            if attached:
                channel.subscribers.append((name, q))

        if not attached:
            # Debug, not warning: a subscriber racing a graceful
            # shutdown is expected and benign, and warning here would
            # make every clean shutdown noisy. It is still worth
            # recording — "my subscriber attached and got nothing" is
            # otherwise a mystery.
            logger.debug(
                "EventBus: subscriber %r attached to closed stream %s; "
                "replaying %d event(s) then terminating",
                name,
                stream_id,
                len(replay_slice),
            )

        try:
            # Yield the replay slice even when the stream is closed. It
            # is exactly what a subscriber that won the lock race by a
            # microsecond would have seen, so honouring it makes losing
            # that race benign rather than silently lossy.
            #
            # The slice's CONTENT differs by path, and that is a
            # property of retention rather than of this guard: in the
            # mid-attach race it is the closed stream's real ring, while
            # a plain post-close attach gets a tombstone whose ring is
            # empty, because the bus deliberately does not retain closed
            # channels. That is the same "the history you asked for may
            # be gone" caveat the ring buffer already carries for a
            # ``from_version`` older than its window — never wrong data,
            # never duplicated, just possibly less of it.
            for ve in replay_slice:
                yield ve
            if not attached:
                return
            while True:
                item = await q.get()
                if item is _CLOSE_SENTINEL:
                    return
                # Narrow: anything that's not the sentinel is a
                # VersionedEvent[E] we put in ourselves.
                assert isinstance(item, VersionedEvent)
                yield item
        finally:
            if attached:
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
