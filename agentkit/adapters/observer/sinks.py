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
        self._noncritical = 0
        self._max_noncritical = maxsize
        self._event = asyncio.Event()
        self._closed = False

    async def emit(self, obs: Observation) -> None:
        self._items.append(obs)
        if obs.kind not in CRITICAL_KINDS:
            self._noncritical += 1
            while self._noncritical > self._max_noncritical:  # coalesce: drop oldest non-critical
                for i, it in enumerate(self._items):
                    if it.kind not in CRITICAL_KINDS:
                        del self._items[i]
                        self._noncritical -= 1
                        break
                else:
                    break
        self._event.set()

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
            yield self._items.popleft()
