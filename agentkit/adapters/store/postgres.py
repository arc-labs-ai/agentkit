"""`PostgresStore` — durable `StorePort` over Postgres (extra: `arc-agentkit[postgres]`, asyncpg)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from agentkit.adapters.store._keylock import KeyLock, key_lock
from agentkit.adapters.store._primitives import check_by, check_limit, not_a_counter

# Sentinel for "row absent", distinct from a stored ``None``. ``value`` is
# ``TEXT NOT NULL``, so a ``None`` from ``fetchval`` can only mean "no row".
_MISS = object()


class PostgresStore:
    """Durable `StorePort` over Postgres (extra: `arc-agentkit[postgres]`, via asyncpg). One KV table +
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
        self._locks: dict[str, KeyLock] = {}

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

    async def _lookup(self, key: str) -> Any:
        """Return the stored value, or `_MISS` when the row is absent — the
        distinction `get` has to collapse to satisfy the `StorePort`
        signature, but which `get_or_set` needs to tell a cached ``None``
        from a cache miss."""
        pool = await self._get_pool()
        async with pool.acquire() as con:
            row = await con.fetchval("SELECT value FROM agentkit_kv WHERE key = $1", key)
        return _MISS if row is None else json.loads(row)

    async def get(self, key: str) -> Any | None:
        found = await self._lookup(key)
        return None if found is _MISS else found

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
        # Presence-based, not truthiness-based — see the same fix in
        # `RedisStore.get_or_set`. ``existing is not None`` re-ran a
        # ``None``-returning producer on every call (measured: 3 calls → 3
        # runs, against 1 for the `InMemoryStore` reference contract).
        existing = await self._lookup(key)
        if existing is not _MISS:
            return existing
        # Reference-counted so the table is not write-only: the in-memory
        # backend leaked one permanent ``asyncio.Lock`` per key ever touched
        # (5,000 get_or_set + delete pairs left kv=0 but locks=5000), and
        # these two had the identical shape.
        async with key_lock(self._locks, key):
            existing = await self._lookup(key)
            if existing is not _MISS:
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

    def _refuse_ttl(self, ttl: int | None, method: str) -> None:
        """Same refusal as `set`, so the whole surface is consistent on this backend.

        A method that quietly accepted ``ttl`` while its neighbours raised would
        be worse than either policy on its own: the caller would conclude the
        backend supports expiry.
        """
        if ttl is not None:
            raise NotImplementedError(
                f"PostgresStore.{method} does not support ttl — see "
                "PostgresStore.set for the rationale."
            )

    async def compare_and_set(
        self, key: str, expected: Any, value: Any, *, ttl: int | None = None
    ) -> bool:
        """ONE statement, so the compare and the set cannot be pulled apart.

        A ``SELECT`` followed by an ``UPDATE`` is two statements and, at the
        default READ COMMITTED isolation, another transaction commits between
        them — the lost update this primitive exists to prevent. The predicate
        rides inside the write instead, and the row count is the answer.

        ``value::jsonb = $3::jsonb`` rather than text equality: the column is
        TEXT holding ``json.dumps`` output, and Python dicts with the same
        contents in a different insertion order serialise to different text.
        Comparing as JSON makes the equality structural, which is what the port
        promises and what every other backend does.

        ``expected=None`` takes the INSERT branch because absent must compare
        equal to None — and the ``ON CONFLICT`` guard keeps it honest: if the
        row turns out to exist it is only overwritten when it holds JSON null.
        Without that guard the upsert would clobber any value at all, turning
        the CAS into an unconditional `set`.
        """
        self._refuse_ttl(ttl, "compare_and_set")
        pool = await self._get_pool()
        async with pool.acquire() as con:
            if expected is None:
                applied = await con.fetchval(
                    "INSERT INTO agentkit_kv (key, value) VALUES ($1, $2) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now() "
                    "WHERE agentkit_kv.value::jsonb = 'null'::jsonb "
                    "RETURNING 1",
                    key,
                    json.dumps(value),
                )
            else:
                applied = await con.fetchval(
                    "UPDATE agentkit_kv SET value = $2, updated_at = now() "
                    "WHERE key = $1 AND value::jsonb = $3::jsonb "
                    "RETURNING 1",
                    key,
                    json.dumps(value),
                    json.dumps(expected),
                )
        return applied is not None

    async def increment(self, key: str, by: int = 1, *, ttl: int | None = None) -> int:
        """One upsert that adds in place, guarded so the cast can never raise.

        ``kv.value ~ '^-?[0-9]+$'`` is not defensive noise: without it, a key
        holding ``{"a": 1}`` makes ``value::bigint`` raise
        ``InvalidTextRepresentationError`` — an asyncpg type escaping through a
        port whose whole purpose is that callers never see backend types. With
        it, the conflicting row simply fails the ``ON CONFLICT ... WHERE``, no
        row is returned, and the uniform `StoreValueError` is raised from here.
        (``ON CONFLICT ... WHERE`` filters rows before the SET expression is
        evaluated, so the cast never runs on a non-numeric row.)

        The regex is exactly the set of strings ``json.dumps`` produces for an
        ``int``, which is why a counter and an ordinary stored value can share
        the KV table: ``1.5``, ``"7"``, ``true`` and ``null`` all fail it, and
        all four are values Redis's INCRBY refuses too.
        """
        check_by(key, by)
        self._refuse_ttl(ttl, "increment")
        pool = await self._get_pool()
        async with pool.acquire() as con:
            total = await con.fetchval(
                "INSERT INTO agentkit_kv AS kv (key, value) VALUES ($1, $2) "
                "ON CONFLICT (key) DO UPDATE SET value = (kv.value::bigint + $3)::text, "
                "updated_at = now() WHERE kv.value ~ '^-?[0-9]+$' "
                "RETURNING kv.value::bigint",
                key,
                json.dumps(by),
                by,
            )
        if total is None:
            # The guard excluded the row, so something non-integer is there.
            # Re-read it to name the type in the error — but OUTSIDE the
            # ``acquire`` above, because ``self.get`` takes a connection of its
            # own. Asking the pool for a second connection while still holding
            # the first deadlocks outright on ``max_size=1`` (measured: the
            # call hangs forever), and on any pool size it doubles the
            # connections a concurrent burst of bad increments needs, so N
            # workers each holding one and waiting for another is a deadlock at
            # every size. The error path must not be the expensive one.
            raise not_a_counter(key, await self.get(key))
        return int(total)

    async def scan(self, prefix: str, *, limit: int | None = None) -> AsyncIterator[str]:
        """A prefix query over the KV table only — the log table has its own reader.

        ``starts_with(key, $1)`` rather than ``key LIKE $1 || '%'`` because the
        prefix is caller data: LIKE would read ``_`` as "any character" and
        ``%`` as "anything", so a scan for ``run_1:`` would also return
        ``run-1:``'s keys. Escaping for LIKE is possible but it is a
        correctness trap that has to be got right in every caller's head;
        ``starts_with`` has no pattern syntax to escape. The same class of bug
        is escaped explicitly on the Redis backend, where SCAN gives no
        alternative.

        The whole result is fetched, then yielded. `asyncpg` cursors are
        transaction-scoped, and holding a transaction open for as long as a
        caller takes to consume an audit scan is a far worse failure mode than
        materialising a list of keys — the caller is about to hold them anyway.
        """
        check_limit(limit)
        if limit == 0:
            return
        pool = await self._get_pool()
        async with pool.acquire() as con:
            if limit is None:
                rows = await con.fetch(
                    "SELECT key FROM agentkit_kv WHERE starts_with(key, $1)", prefix
                )
            else:
                rows = await con.fetch(
                    "SELECT key FROM agentkit_kv WHERE starts_with(key, $1) LIMIT $2",
                    prefix,
                    limit,
                )
        for row in rows:
            yield str(row["key"])
