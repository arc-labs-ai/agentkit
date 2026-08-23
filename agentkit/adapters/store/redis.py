"""`RedisStore` — durable `StorePort` over Redis (extra: `arc-agentkit[redis]`)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

# Sentinel for "key absent", distinct from a stored ``None``. A module-level
# object() is unforgeable — no JSON value can ever compare identical to it.
_MISS = object()


class RedisStore:
    """Durable `StorePort` over Redis (extra: `arc-agentkit[redis]`). KV via GET/SET(EX), append-logs via
    RPUSH/LRANGE, values JSON-encoded. Single-flight is in-process (an `asyncio.Lock` per key); TTL is honored
    natively (SETEX). Inject a `client` (e.g. a fake) for tests; else built from `url`.
    """

    def __init__(
        self, url: str | None = None, *, client: Any = None, namespace: str = "agentkit"
    ) -> None:
        self._ns = namespace
        self._locks: dict[str, asyncio.Lock] = {}
        if client is not None:
            self._redis = client
        else:
            import redis.asyncio as aioredis  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]  — the [redis] extra

            self._redis = aioredis.from_url(url or "redis://localhost:6379/0")

    def _k(self, key: str) -> str:
        return f"{self._ns}:kv:{key}"

    def _l(self, key: str) -> str:
        return f"{self._ns}:log:{key}"

    @staticmethod
    def _loads(raw: Any) -> Any:
        return None if raw is None else json.loads(raw)

    async def _lookup(self, key: str) -> Any:
        """Return the stored value, or `_MISS` when the key is absent.

        ``get`` collapses both onto ``None``, which is right for the public
        `StorePort` signature but useless for `get_or_set`: a Redis miss and a
        stored JSON ``null`` are different facts. Redis itself keeps them
        apart (GET returns ``None`` vs the two bytes ``null``), so the
        distinction is recovered here rather than guessed at the call site."""
        raw = await self._redis.get(self._k(key))
        return _MISS if raw is None else json.loads(raw)

    async def get(self, key: str) -> Any | None:
        return self._loads(await self._redis.get(self._k(key)))

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        await self._redis.set(self._k(key), json.dumps(value), ex=ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(self._k(key))  # idempotent (DEL of a missing key is a no-op)

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        # Presence-based, not truthiness-based. ``existing is not None``
        # treated a legitimately cached ``None`` as a miss, so a producer
        # returning ``None`` re-ran on EVERY call and single-flight silently
        # stopped holding. Measured against the `InMemoryStore` reference: 3
        # calls with a ``None``-returning fn → InMemory ran it 1x, Redis 3x.
        existing = await self._lookup(key)
        if existing is not _MISS:
            return existing
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            existing = await self._lookup(key)
            if existing is not _MISS:
                return existing
            result = await fn()  # a raised fn propagates, unstored (never cache failure)
            await self.set(key, result, ttl=ttl)
            return result

    async def append(self, key: str, value: Any) -> None:
        await self._redis.rpush(self._l(key), json.dumps(value))

    async def list(self, key: str) -> list[Any]:
        return [json.loads(x) for x in await self._redis.lrange(self._l(key), 0, -1)]

    async def aclose(self) -> None:
        aclose = getattr(
            self._redis, "aclose", None
        )  # close the connection pool (no-op for an injected fake)
        if aclose is not None:
            await aclose()
