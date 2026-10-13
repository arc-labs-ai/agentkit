"""Per-key single-flight locks that reclaim themselves.

Every ``StorePort`` backend needs the same thing for ``get_or_set``: one lock
per key so concurrent callers run the producer once, and NO permanent entry
left behind for every key the process has ever touched. The in-memory store's
table was write-only — measured, 5,000 ``get_or_set`` + ``delete`` pairs left
``kv=0`` but ``locks=5000``, one ``asyncio.Lock`` per key forever — and the
Redis and Postgres backends still had the same shape.

Extracted here rather than copied a third time. Four copies of one token
estimator, and three of a module-vs-function name collision, are what the rest
of this release was spent undoing; a lock table is not worth repeating the
lesson on.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class KeyLock:
    """A per-key lock plus a live-user count.

    Reference counting, NOT eviction-on-release: ``asyncio.Lock.release()``
    clears ``locked()`` *before* the woken waiter resumes, so ``locked()`` is
    not a safe "nobody needs this" test. Dropping the lock in that window lets
    a queued contender build a second lock for the same key and run the
    producer twice, which is precisely the single-flight guarantee this exists
    to provide.

    ``users`` is incremented and decremented with no ``await`` in between, so
    it is exact under asyncio's cooperative scheduling.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@asynccontextmanager
async def key_lock(table: dict[str, KeyLock], key: str) -> AsyncIterator[None]:
    """Hold ``key``'s lock, creating it on demand and reclaiming it after.

    The reclaim runs in a ``finally``, so a producer that RAISES releases its
    entry too — otherwise a backend that mostly fails would leak faster than
    one that mostly works.

    The ``table.get(key) is entry`` guard matters: between this waiter's last
    decrement and the delete, another task may have created a fresh entry for
    the same key. Deleting unconditionally would drop the lock a live caller
    is holding.
    """
    entry = table.get(key)
    if entry is None:
        entry = table[key] = KeyLock()
    entry.users += 1  # no ``await`` between the lookup and this — atomic
    try:
        async with entry.lock:
            yield
    finally:
        entry.users -= 1
        if entry.users == 0 and table.get(key) is entry:
            del table[key]


__all__ = ["KeyLock", "key_lock"]
