"""`PostgresStore` — durable `StorePort` over Postgres (extra: `agentkit[postgres]`, asyncpg)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any


class PostgresStore:
    """Durable `StorePort` over Postgres (extra: `agentkit[postgres]`, via asyncpg). One KV table +
    one append-log table, JSON-text values. Call `await init()` once to create the tables. Single-flight is
    in-process; durability/atomicity is Postgres's. TTL is not enforced (no sweeper — use `RedisStore` for TTL).
    Inject a `pool` for tests; else an asyncpg pool is created from `dsn`.
    """

    _DDL = (
        "CREATE TABLE IF NOT EXISTS agentkit_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT now())",
        "CREATE TABLE IF NOT EXISTS agentkit_log (key TEXT NOT NULL, seq BIGSERIAL PRIMARY KEY, "
        "value TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now())",
        "CREATE INDEX IF NOT EXISTS agentkit_log_key_seq ON agentkit_log (key, seq)",
    )

    def __init__(self, dsn: str | None = None, *, pool: Any = None) -> None:
        self._dsn = dsn
        self._pool = pool
        self._locks: dict[str, asyncio.Lock] = {}

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]  — the [postgres] extra, no py.typed marker

            self._pool = await asyncpg.create_pool(self._dsn)
        return self._pool

    async def init(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as con:
            for ddl in self._DDL:
                await con.execute(ddl)

    async def aclose(self) -> None:
        if self._pool is not None:  # close the connection pool we created
            await self._pool.close()
            self._pool = None

    async def get(self, key: str) -> Any | None:
        pool = await self._get_pool()
        async with pool.acquire() as con:
            row = await con.fetchval("SELECT value FROM agentkit_kv WHERE key = $1", key)
        return None if row is None else json.loads(row)

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        # PostgresStore has no background sweeper for TTL — silently
        # dropping the kwarg would give the same signature as Redis with
        # a different durability outcome. Raise explicitly so callers
        # opt into RedisStore for that key space, or implement a sweeper
        # job and remove this guard.
        if ttl is not None:
            raise NotImplementedError(
                "PostgresStore.set does not support ttl — no background "
                "sweeper. Use RedisStore for TTL semantics, or implement "
                "expiry in your application layer."
            )
        pool = await self._get_pool()
        async with pool.acquire() as con:
            await con.execute(
                "INSERT INTO agentkit_kv (key, value) VALUES ($1, $2) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                key,
                json.dumps(value),
            )

    async def delete(self, key: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as con:
            await con.execute("DELETE FROM agentkit_kv WHERE key = $1", key)  # idempotent

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        # Refuse a TTL request the same way ``set`` does, so the
        # composite contract stays consistent on the Postgres backend.
        if ttl is not None:
            raise NotImplementedError(
                "PostgresStore.get_or_set does not support ttl — see "
                "PostgresStore.set for the rationale."
            )
        existing = await self.get(key)
        if existing is not None:
            return existing
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            existing = await self.get(key)
            if existing is not None:
                return existing
            result = await fn()
            await self.set(key, result)
            return result

    async def append(self, key: str, value: Any) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as con:
            await con.execute(
                "INSERT INTO agentkit_log (key, value) VALUES ($1, $2)", key, json.dumps(value)
            )

    async def list(self, key: str) -> list[Any]:
        pool = await self._get_pool()
        async with pool.acquire() as con:
            rows = await con.fetch(
                "SELECT value FROM agentkit_log WHERE key = $1 ORDER BY seq", key
            )
        return [json.loads(r["value"]) for r in rows]
