"""SignalChannel + MergeInbox — typed per-actor communication seam.

Each actor owns one ``SignalChannel`` bundling three queues:
``inbox`` (parent's ControlSignals in), ``outbox`` (this actor's
DataSignals out — tests + replay read here), ``merge_inbox``
(children's DataSignals multiplexed in, so the merge loop reads
one queue instead of polling each child).

The load-bearing invariant is the **dual-write emit**:
``channel.emit(signal)`` lands on this outbox AND on the parent's
``merge_inbox`` (if attached). That's what makes child progress
stream upward without the parent polling every descendant.

I/O-free aside from ``asyncio.Queue``. Stamps ``sender_id`` +
``timestamp_us`` on emit — no domain coupling beyond the envelope.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from agentkit.agents.control.signals import ControlSignal, DataSignal
from agentkit.kernel.observation import Observation

logger = logging.getLogger(__name__)

ControlT = TypeVar("ControlT", bound=ControlSignal)
DataT = TypeVar("DataT", bound=DataSignal)


def _monotonic_us() -> int:
    """Default timestamp source: monotonic microseconds. Wall-clock
    independent so signals stay ordered correctly across NTP jumps."""
    return time.monotonic_ns() // 1000


# ── MergeInbox ──────────────────────────────────────────────────────


class MergeInbox(Generic[DataT]):
    """Multiplexed channel: many children → one read seam on the
    parent. Each enqueued item is ``(sender_id, signal)`` so the
    parent's dispatcher can attribute work back to a specific child
    without per-child polling.

    Bounded by ``buffer_size``. When full, producers (child emits)
    block via ``await put`` — that's the right backpressure shape
    (the parent is the bottleneck; making children wait for the
    parent to catch up keeps memory bounded).
    """

    __slots__ = ("_queue",)

    def __init__(self, *, buffer_size: int = 512) -> None:
        self._queue: asyncio.Queue[tuple[str, DataT]] = asyncio.Queue(maxsize=buffer_size)

    async def put(self, item: tuple[str, DataT]) -> None:
        """Block until there's room. Backpressure flows to the
        producing child."""
        await self._queue.put(item)

    def put_nowait(self, item: tuple[str, DataT]) -> None:
        """Best-effort put. Raises ``asyncio.QueueFull`` if full —
        caller decides whether to drop or block."""
        self._queue.put_nowait(item)

    def get_nowait(self) -> tuple[str, DataT]:
        """Pull one item without blocking. Raises
        ``asyncio.QueueEmpty`` if empty — pair with ``empty()``."""
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        """Approximate queue depth — diagnostic only (race-prone)."""
        return self._queue.qsize()


# ── SignalChannel ───────────────────────────────────────────────────


class SignalChannel(Generic[ControlT, DataT]):
    """Per-agent bidirectional channel. The single seam an agent uses
    to talk to its parent and to its children.

    Concurrency model:

    - The owning agent reads ``inbox`` and ``merge_inbox``; writes
      ``outbox`` via ``emit``. No locking needed — one consumer per
      queue.
    - The parent agent writes ``inbox`` via the channel's
      ``send_to`` method.
    - Children write to this channel's ``merge_inbox`` via THEIR
      own channel's ``emit`` — wired by ``attach_parent``.

    The dual-write on emit (outbox + parent's merge_inbox) is what
    lets tests + replay read from ``outbox`` without a parent
    context, while production agents see signals via the merge
    inbox without polling each child.
    """

    __slots__ = (
        "_clock",
        "_observer",
        "_parent_merge_inbox",
        "agent_id",
        "inbox",
        "merge_inbox",
        "outbox",
    )

    def __init__(
        self,
        *,
        agent_id: str,
        buffer_size: int = 256,
        merge_buffer_size: int | None = None,
        clock: Callable[[], int] = _monotonic_us,
        observer: Any = None,
    ) -> None:
        """Build a channel owned by ``agent_id``.

        ``merge_buffer_size`` defaults to ``2 * buffer_size`` because
        the merge inbox aggregates from N children — a single
        child's outbox depth is the right baseline, but the merge
        side needs headroom for fan-in.

        ``clock`` is injectable so tests can pin timestamps; default
        is monotonic microseconds.

        ``observer`` is an optional ``ObserverPort``-shaped sink
        (anything with an async ``emit(Observation)``). When wired,
        every ``emit`` stamps a ``signal.emitted`` observation so the
        run's audit timeline sees the coordination cascade — same
        shape as observations from the LLM / tool / memory seams.
        Kept optional so unit tests + orphan channels (no runner
        attached) still work at zero-config.
        """
        self.agent_id = agent_id
        self.inbox: asyncio.Queue[ControlT] = asyncio.Queue(maxsize=buffer_size)
        self.outbox: asyncio.Queue[DataT] = asyncio.Queue(maxsize=buffer_size)
        self.merge_inbox: MergeInbox[DataT] = MergeInbox(
            buffer_size=merge_buffer_size if merge_buffer_size is not None else buffer_size * 2,
        )
        self._parent_merge_inbox: MergeInbox[DataT] | None = None
        self._clock = clock
        self._observer = observer

    def attach_parent(self, parent_merge_inbox: MergeInbox[DataT]) -> None:
        """Wire this channel's emit to also fan up to the parent's
        merge inbox. Called by the parent's spawn handler after
        constructing the child."""
        self._parent_merge_inbox = parent_merge_inbox

    async def emit(self, signal: DataT) -> None:
        """Emit a data signal upward.

        Stamps ``sender_id`` and ``timestamp_us`` on the envelope at
        emit time. Puts the signal on this agent's outbox AND on the
        parent's merge inbox (if attached).

        Both writes ``await`` so backpressure flows correctly — if
        the parent is slow, the child blocks. Tests that read from
        a detached channel (no parent attached) only see the outbox
        write; production flows see both.
        """
        # Envelope is a frozen dataclass. Build a stamped copy via
        # ``dataclasses.replace``; sharing one mutable envelope across
        # outbox + parent's merge_inbox would race two consumers on any
        # later rewrite (audit trail vs live merge loop).
        stamped = dataclasses.replace(
            signal,
            sender_id=signal.sender_id if signal.sender_id is not None else self.agent_id,
            timestamp_us=signal.timestamp_us if signal.timestamp_us != 0 else self._clock(),
        )

        await self.outbox.put(stamped)
        if self._parent_merge_inbox is not None:
            await self._parent_merge_inbox.put((self.agent_id, stamped))

        # Best-effort observability. A misbehaving observer must never
        # break the coordination cascade, so wrap in ``suppress``; the
        # queue writes above have already succeeded regardless. Topic
        # is the signal's Python class name — same taxonomy replay
        # tools use to filter a cascade by kind.
        if self._observer is not None:
            with contextlib.suppress(Exception):
                topic = type(signal).__name__
                await self._observer.emit(
                    Observation(
                        kind="signal.emitted",
                        render=f"signal.emitted:{topic}",
                        payload={
                            "sender_id": stamped.sender_id,
                            "topic": topic,
                            "timestamp_us": stamped.timestamp_us,
                        },
                    )
                )

    async def send_to(self, signal: ControlT) -> None:
        """Deliver a control signal to THIS agent's inbox. Called by
        the parent (or the runner) to push down directives.

        ``sender_id`` + ``timestamp_us`` get stamped here too —
        symmetric with ``emit``, since the parent is the "sender"
        from the child's POV when it reads the inbox.
        """
        stamped = dataclasses.replace(
            signal,
            sender_id=signal.sender_id if signal.sender_id is not None else "parent",
            timestamp_us=signal.timestamp_us if signal.timestamp_us != 0 else self._clock(),
        )
        await self.inbox.put(stamped)

    def try_send_to(self, signal: ControlT) -> bool:
        """Non-blocking control-signal delivery. Returns False if
        the inbox is full — used by best-effort sibling broadcast
        where dropping a hint is preferable to blocking the parent.
        """
        stamped = dataclasses.replace(
            signal,
            sender_id=signal.sender_id if signal.sender_id is not None else "parent",
            timestamp_us=signal.timestamp_us if signal.timestamp_us != 0 else self._clock(),
        )
        try:
            self.inbox.put_nowait(stamped)
            return True
        except asyncio.QueueFull:
            logger.warning(
                "SignalChannel(%s).try_send_to: inbox full; dropping %s",
                self.agent_id,
                type(signal).__name__,
            )
            return False


__all__ = ["MergeInbox", "SignalChannel"]
