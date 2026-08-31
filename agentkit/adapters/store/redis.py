"""`RedisStore` — durable `StorePort` over Redis (extra: `arc-agentkit[redis]`)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from agentkit.adapters.store._keylock import KeyLock, key_lock
from agentkit.adapters.store._primitives import check_by, check_limit, not_a_counter

# Sentinel for "key absent", distinct from a stored ``None``. A module-level
# object() is unforgeable — no JSON value can ever compare identical to it.
_MISS = object()

# Redis's own wording for INCRBY against something that is not a number.
_NOT_AN_INTEGER = "not an integer"

# Characters Redis's glob-style MATCH treats as pattern syntax. A key prefix is
# caller data (a `Scope.key()`, a tool name, a correlation id), never a
# pattern, so every one of these has to be escaped before it reaches SCAN.
_GLOB_META = "\\*?[]"

# SCAN's COUNT is a HINT about work per call, not a page size — the server may
# return more, fewer, or (on a sparse match) none at all while still handing
# back a non-zero cursor. So the loop below is driven by the cursor and never
# by how many keys came back; this number only trades round trips against
# per-call latency.
_SCAN_COUNT = 500


def _escape_glob(text: str) -> str:
    """Make ``text`` match itself literally under Redis's MATCH.

    Unescaped, ``scan("tenant-a*")`` would ask Redis for every key beginning
    ``tenant-a`` — including ``tenant-ab``'s. On a store whose entire job is
    scoping, that is a cross-tenant read, and it appears only on this backend
    because the other three compare prefixes with ``startswith``.

    The backslash goes first in ``_GLOB_META``; escaping it after the others
    would double-escape the backslashes just inserted.
    """
    for char in _GLOB_META:
        text = text.replace(char, "\\" + char)
    return text


def _is_watch_conflict(exc: BaseException) -> bool:
    """``redis.exceptions.WatchError``, matched by CLASS NAME rather than ``isinstance``.

    ``redis`` is an optional extra and this adapter is routinely constructed
    with an injected client and no ``redis`` package installed at all — the
    offline test path. A module-level ``from redis.exceptions import
    WatchError`` would make that path unimportable, and importing it lazily
    inside the ``except`` would resolve a class the raised exception is not an
    instance of, so a lost race would escape as an error instead of being
    reported as ``False``. A lost compare-and-set is an ordinary outcome; that
    is the whole reason the method returns a bool.
    """
    return type(exc).__name__ == "WatchError"


class RedisStore:
    """Durable `StorePort` over Redis (extra: `arc-agentkit[redis]`). KV via GET/SET(EX), append-logs via
    RPUSH/LRANGE, values JSON-encoded. Single-flight is in-process (an `asyncio.Lock` per key); TTL is honored
    natively (SETEX). Inject a `client` (e.g. a fake) for tests; else built from `url`.
    """

    def __init__(
        self, url: str | None = None, *, client: Any = None, namespace: str = "agentkit"
    ) -> None:
        self._ns = namespace
        self._locks: dict[str, KeyLock] = {}
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
        # Reference-counted so the table is not write-only: the in-memory
        # backend leaked one permanent ``asyncio.Lock`` per key ever touched
        # (5,000 get_or_set + delete pairs left kv=0 but locks=5000), and
        # these two had the identical shape.
        async with key_lock(self._locks, key):
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

    async def compare_and_set(
        self, key: str, expected: Any, value: Any, *, ttl: int | None = None
    ) -> bool:
        """WATCH / read / MULTI / SET — Redis's optimistic-locking idiom.

        WATCH arms the key BEFORE the read, so if anything writes it between
        the read and the EXEC, the EXEC is discarded and redis-py raises
        ``WatchError``. Without the WATCH this would be a plain
        read-then-write across two round trips, which is the lost update the
        primitive exists to prevent — and it would pass every single-threaded
        test.

        A ``WatchError`` is translated to ``False``, not re-raised: losing the
        race IS the answer the caller asked for. Any other exception
        propagates, because "the store is down" is not a lost race.
        """
        redis_key = self._k(key)
        async with self._redis.pipeline(transaction=True) as pipe:
            await pipe.watch(redis_key)
            raw = await pipe.get(redis_key)
            # Absent and stored-null both decode to None here, which is the
            # documented CAS semantics: ``expected`` came from `get`, which
            # cannot tell them apart either.
            if self._loads(raw) != expected:
                return False
            pipe.multi()
            pipe.set(redis_key, json.dumps(value), ex=ttl)
            try:
                await pipe.execute()
            except Exception as exc:
                if _is_watch_conflict(exc):
                    return False
                raise
            return True

    async def increment(self, key: str, by: int = 1, *, ttl: int | None = None) -> int:
        """INCRBY and ``EXPIRE ... NX`` inside ONE transaction.

        The two commands must be one unit. Issued separately, a process that
        dies between them leaves a counter with no window — a rate limit that
        is permanently exhausted for that key, with no expiry to ever clear it.
        MULTI/EXEC makes the pair atomic.

        ``NX`` is what makes the window fixed rather than sliding: it sets an
        expiry only when the key has none. A plain EXPIRE would push the
        deadline out on every hit, so a caller sending traffic faster than the
        window is long would keep the counter alive forever and the limit would
        never reset — the failure appears only under exactly the load the limit
        exists for. (``EXPIRE ... NX`` needs Redis 7.0+.)

        The JSON encoding lines up for free: ``json.dumps(5) == "5"`` is
        precisely the string Redis stores an integer as, so a counter written
        by INCRBY reads back through `get` as an ``int`` and a counter written
        by `set` can be INCRBY'd. Nothing else in this adapter would work if
        that were not true.

        The error translation matches on Redis's MESSAGE rather than importing
        ``ResponseError`` — same reason as `_is_watch_conflict`: the extra may
        not be installed at all on the path that constructs this with a client.
        """
        check_by(key, by)
        redis_key = self._k(key)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incrby(redis_key, by)
            if ttl is not None:
                pipe.expire(redis_key, ttl, nx=True)
            try:
                results = await pipe.execute()
            except Exception as exc:
                if _NOT_AN_INTEGER not in str(exc):
                    raise
                # One extra round trip, on the failure path only, so the error
                # can name what is actually in the key. A caller told "not an
                # integer" without being told what it IS has to go look.
                raise not_a_counter(key, await self.get(key)) from exc
            return int(results[0])

    async def scan(self, prefix: str, *, limit: int | None = None) -> AsyncIterator[str]:
        """SCAN with an escaped MATCH, looping on the cursor and de-duplicating.

        Three properties of SCAN that a naive translation gets wrong:

        * MATCH is a GLOB. The prefix is caller data, so it is escaped — see
          `_escape_glob` for the cross-tenant read that follows if it is not.
        * A non-zero cursor means "more to come" even when this call returned
          nothing. Stopping on an empty page silently truncates the scan.
        * The same key may be returned more than once (a rehash during the
          scan), so keys are de-duplicated. The ``seen`` set is bounded by the
          number of MATCHING keys, which is what the caller is about to hold
          anyway — the alternative is a caller that double-processes an audit
          record and cannot tell why.

        Keys come back as ``bytes``: redis-py only decodes when the client was
        built with ``decode_responses=True``, and this one is not. `get`
        survives that because ``json.loads`` accepts bytes; the string surgery
        here does not.
        """
        check_limit(limit)
        if limit == 0:
            return
        namespace = f"{self._ns}:kv:"
        pattern = _escape_glob(namespace + prefix) + "*"
        cursor = 0
        seen: set[str] = set()
        sent = 0
        while True:
            cursor, raw_keys = await self._redis.scan(cursor, match=pattern, count=_SCAN_COUNT)
            for raw in raw_keys:
                full = raw.decode() if isinstance(raw, bytes) else raw
                if full in seen:
                    continue
                seen.add(full)
                yield full[len(namespace) :]
                sent += 1
                if limit is not None and sent >= limit:
                    return
            if cursor == 0:
                return

    async def aclose(self) -> None:
        aclose = getattr(
            self._redis, "aclose", None
        )  # close the connection pool (no-op for an injected fake)
        if aclose is not None:
            await aclose()
