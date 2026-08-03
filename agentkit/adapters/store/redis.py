"""`RedisStore` — durable `StorePort` over Redis (extra: `arc-agentkit[redis]`)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any


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

    async def get(self, key: str) -> Any | None:
        return self._loads(await self._redis.get(self._k(key)))

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        await self._redis.set(self._k(key), json.dumps(value), ex=ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(self._k(key))  # idempotent (DEL of a missing key is a no-op)

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        existing = await self.get(key)
        if existing is not None:
            return existing
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            existing = await self.get(key)
            if existing is not None:
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
