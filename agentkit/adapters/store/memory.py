"""`InMemoryStore` — the offline reference `StorePort` and the contract every durable backend matches.

TTL is honored: a caller passing a 300s idempotency entry must see
expiry in dev the same way it sees it in Redis. Keys expire via a lazy
deadline check on every read — no background sweeper needed.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any


class InMemoryStore:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._kv: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}  # key → monotonic deadline (set only when ttl given)
        self._logs: dict[str, list[Any]] = defaultdict(list)
        self._locks: dict[str, asyncio.Lock] = {}
        self._clock = clock

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

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        self._kv[key] = value
        if ttl is not None:
            self._expiry[key] = self._clock() + ttl
        else:
            # An explicit ``ttl=None`` clears any prior expiry — match
            # Redis SET semantics (a SET without EX/PX removes the TTL).
            self._expiry.pop(key, None)

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        """Single-flight: a hit returns the stored value; a miss runs `fn` once and stores it. A raised
        `fn` propagates and is NOT stored, so a transient error retries clean (failures are never cached).
        ``ttl`` is applied on the stored result, matching the other backends."""
        if key in self._kv and not self._is_expired(key):
            return self._kv[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
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
        return list(self._logs[key])
