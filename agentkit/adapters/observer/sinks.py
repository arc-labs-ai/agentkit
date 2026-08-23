"""Terminal observation sinks: `CollectingObserver` (list capture) and `QueueObserver` (bounded async)."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator

from agentkit.kernel.observation import CRITICAL_KINDS, Observation


class CollectingObserver:
    """Records every observation into `.items` (no backpressure). For tests and small in-proc consumers."""

    def __init__(self) -> None:
        self.items: list[Observation] = []

    async def emit(self, obs: Observation) -> None:
        self.items.append(obs)

    async def close(self) -> None:
        # No buffered tail — everything landed on ``.items`` at emit
        # time. The no-op keeps the ``ObserverPort`` Protocol satisfied
        # so shutdown paths that call ``await observer.close()``
        # uniformly find the hook.
        return None


class QueueObserver:
    """Non-blocking, bounded async observer with the never-drop-results rule.

    `emit` never blocks and never raises into the run. Non-critical observations (progress/summary) are
    bounded to `maxsize`; when over, the **oldest non-critical** is dropped (coalesced). `result`/`error`
    are **never dropped** and never count against the bound. `stream()` async-iterates in insertion order
    until `close()`.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._items: deque[Observation] = deque()
        # QUEUE DEPTH, not a lifetime counter. It is incremented on
        # ``emit`` and decremented on every path that removes an item
        # (``_drop_oldest_noncritical`` and ``stream``'s pop) — the two
        # must stay paired. Counting lifetime emissions instead made
        # ``maxsize`` a lifetime budget: measured with ``maxsize=4`` and a
        # consumer draining each item, 12 progress observations emitted
        # → the consumer saw only ``['p0','p1','p2','p3']`` and the queue
        # ended at ``_noncritical=4, len(_items)=0`` — 8 silently lost
        # from an EMPTY queue, with nothing backed up at all.
        self._noncritical = 0
        self._max_noncritical = maxsize
        self._event = asyncio.Event()
        self._closed = False

    async def emit(self, obs: Observation) -> None:
        self._items.append(obs)
        if obs.kind not in CRITICAL_KINDS:
            self._noncritical += 1
            while self._noncritical > self._max_noncritical:  # coalesce: drop oldest non-critical
                if not self._drop_oldest_noncritical():
                    break
        self._event.set()

    def _drop_oldest_noncritical(self) -> bool:
        """Evict the oldest coalescable item; ``False`` when the queue holds
        only ``CRITICAL_KINDS`` (never dropped, and never counted)."""
        for i, it in enumerate(self._items):
            if it.kind not in CRITICAL_KINDS:
                del self._items[i]
                self._noncritical -= 1
                return True
        return False

    async def close(self) -> None:
        self._closed = True
        self._event.set()

    async def stream(self) -> AsyncIterator[Observation]:
        while True:
            while not self._items:
                if self._closed:
                    return
                self._event.clear()
                await self._event.wait()
            item = self._items.popleft()
            # Decrement BEFORE the yield: the item has left the queue, so
            # the depth must reflect that even if the consumer never
            # resumes. Omitting this is what turned the bound into a
            # lifetime budget.
            if item.kind not in CRITICAL_KINDS:
                self._noncritical -= 1
            yield item
