"""`InMemoryStore` — the offline reference `StorePort` and the contract every durable backend matches.

TTL is honored: a caller passing a 300s idempotency entry must see
expiry in dev the same way it sees it in Redis. Keys expire via a lazy
deadline check on every read, plus an amortized sweep on the write path so a
key that is never read again is still reclaimed (`purge_expired`).

Every table here is bounded by live data, not by lifetime traffic: `_locks`
entries are reference-counted and dropped when the last waiter leaves, and
reads never materialize log buckets.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from agentkit.adapters.store._keylock import KeyLock, key_lock
from agentkit.adapters.store._primitives import (
    check_by,
    check_limit,
    is_counter,
    not_a_counter,
)

# Writes between amortized expiry sweeps. Purge-on-read alone never reclaims
# a TTL'd key that is never read again — the common shape for idempotency
# keys, which are written once and only ever *checked* under a different key.
_SWEEP_EVERY = 256


class InMemoryStore:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._kv: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}  # key → monotonic deadline (set only when ttl given)
        self._logs: dict[str, list[Any]] = defaultdict(list)
        self._locks: dict[str, KeyLock] = {}
        self._clock = clock
        self._writes_since_sweep = 0

    def _is_expired(self, key: str) -> bool:
        deadline = self._expiry.get(key)
        if deadline is None:
            return False
        if self._clock() < deadline:
            return False
        # Lazy purge — drop the entry on the read that observed the
        # expiry so subsequent gets / sets see a clean slate.
        self._kv.pop(key, None)
        self._expiry.pop(key, None)
        return True

    async def get(self, key: str) -> Any | None:
        if self._is_expired(key):
            return None
        return self._kv.get(key)

    def purge_expired(self) -> int:
        """Drop every key whose deadline has passed; returns how many went.

        Purge-on-read reclaimed a key only if something read it again, so
        TTL'd keys nobody revisits were retained forever — a store used purely
        for idempotency grew without bound while reporting the right answers.
        Public because a long-lived process may want to sweep on its own
        cadence; `set` also calls it every ``_SWEEP_EVERY`` writes, which keeps
        the table bounded by live data rather than lifetime traffic."""
        now = self._clock()
        dead = [k for k, deadline in self._expiry.items() if now >= deadline]
        for k in dead:
            self._kv.pop(k, None)
            self._expiry.pop(k, None)
        return len(dead)

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        self._kv[key] = value
        if ttl is not None:
            self._expiry[key] = self._clock() + ttl
        else:
            # An explicit ``ttl=None`` clears any prior expiry — match
            # Redis SET semantics (a SET without EX/PX removes the TTL).
            self._expiry.pop(key, None)
        # Amortized O(1): one full sweep per ``_SWEEP_EVERY`` writes. The
        # just-written key is never collected — its deadline is strictly in
        # the future for any ``ttl > 0``, and ``ttl=0`` already means
        # "expired now" on the read path.
        self._writes_since_sweep += 1
        if self._writes_since_sweep >= _SWEEP_EVERY:
            self._writes_since_sweep = 0
            self.purge_expired()

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        """Single-flight: a hit returns the stored value; a miss runs `fn` once and stores it. A raised
        `fn` propagates and is NOT stored, so a transient error retries clean (failures are never cached).
        ``ttl`` is applied on the stored result, matching the other backends."""
        if key in self._kv and not self._is_expired(key):
            return self._kv[key]
        async with key_lock(self._locks, key):
            if key in self._kv and not self._is_expired(key):
                return self._kv[key]
            result = await fn()  # a raised fn propagates here, unstored
            await self.set(key, result, ttl=ttl)
            return result

    async def delete(self, key: str) -> None:
        self._kv.pop(key, None)  # idempotent: missing key is a no-op
        self._expiry.pop(key, None)

    async def append(self, key: str, value: Any) -> None:
        self._logs[key].append(value)

    async def list(self, key: str) -> list[Any]:
        # ``self._logs[key]`` on a defaultdict CREATES the bucket, so reading a
        # never-appended key left a permanent empty list behind (measured: one
        # ``list()`` miss → ``len(_logs) == 1``). A read must not be a write.
        return list(self._logs.get(key, ()))

    async def compare_and_set(
        self, key: str, expected: Any, value: Any, *, ttl: int | None = None
    ) -> bool:
        """Deliberately NOT under ``self._locks``.

        There is no ``await`` between the read and the write below, and asyncio
        is cooperative, so no other task can observe or interleave with the
        intermediate state — the sequence is already atomic and a lock would
        add nothing. Taking `get_or_set`'s lock instead would be actively
        worse: a producer that itself compare-and-sets the key it is producing
        would deadlock against its own single-flight entry.

        ``==`` rather than ``is``: the durable backends compare JSON-decoded
        values, and an identity check here would make the reference store the
        only one where CAS on a dict could ever succeed.
        """
        current = None if self._is_expired(key) else self._kv.get(key)
        if current != expected:
            return False
        await self.set(key, value, ttl=ttl)
        return True

    async def increment(self, key: str, by: int = 1, *, ttl: int | None = None) -> int:
        """Same no-await-in-between argument as `compare_and_set`: read, add, write.

        ``ttl`` is applied only when the key has no deadline yet, which is
        Redis's ``EXPIRE NX`` and keeps a hot counter from sliding its own
        window forever.

        The deadline is captured and restored around the write because ``set``
        CLEARS any expiry when handed ``ttl=None`` (it mirrors Redis's SET,
        where a plain SET drops the TTL). Without that, the second increment of
        every windowed counter would silently make it permanent — the counter
        would keep counting and the window would never close.

        Keyed on PRESENCE, not on ``get() is None``. A key holding a stored
        ``null`` is a VALUE, and the durable backends cannot add to it — Redis
        INCRBY on the four bytes ``null`` is an error, and so is Postgres's
        ``::bigint`` cast. Treating it as zero here would make the offline
        reference the one backend that accepted it, which is the same drift
        `get_or_set` was already caught in.
        """
        check_by(key, by)
        present = key in self._kv and not self._is_expired(key)
        current = self._kv[key] if present else 0
        if present and not is_counter(current):
            raise not_a_counter(key, current)
        deadline = self._expiry.get(key)
        total = current + by
        await self.set(key, total, ttl=None)
        if deadline is not None:
            self._expiry[key] = deadline
        elif ttl is not None:
            self._expiry[key] = self._clock() + ttl
        return total

    async def scan(self, prefix: str, *, limit: int | None = None) -> AsyncIterator[str]:
        """Snapshot the key list BEFORE yielding any of it.

        ``yield`` is a suspension point, so a writer runs between two
        iterations of this loop; iterating ``self._kv`` live raised
        ``RuntimeError: dictionary changed size during iteration`` the first
        time an audit reader ran against a live run. The snapshot costs one
        list of keys — which the caller is about to hold anyway — and buys the
        only concurrency promise the port makes.
        """
        check_limit(limit)
        if limit == 0:
            return  # a real cap of zero, not "unset" — see `check_limit`
        sent = 0
        for key in list(self._kv):
            if not key.startswith(prefix) or self._is_expired(key):
                continue
            yield key
            sent += 1
            if limit is not None and sent >= limit:
                return
