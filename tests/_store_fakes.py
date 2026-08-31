"""Faithful offline stand-ins for the Redis and Postgres surfaces the `StorePort` adapters drive.

Neither extra is installed in this repo's dev environment, and CI has no
service containers, so a fake is not a convenience here — it is the ONLY
coverage `RedisStore` and `PostgresStore` get. That raises the bar for what a
fake has to do, and these are built against three failure modes a naive
dict-backed double lets straight through:

1. **No scheduling points.** A fake whose methods are ``async def`` with no
   ``await`` inside never yields to the event loop, so a `compare_and_set`
   implemented as a non-atomic read-then-write passes every concurrency test.
   Every command here starts with ``await asyncio.sleep(0)``, which is what
   makes "exactly one racer wins" a real assertion rather than a restatement
   of "this test ran on one thread".

2. **Str-typed reads.** ``redis.asyncio`` returns ``bytes`` unless the client
   was built with ``decode_responses=True``, and `RedisStore` builds it
   without. A fake that hands back ``str`` hides every missing ``.decode()`` —
   ``json.loads`` accepts bytes, so `get` survives, but `scan`, which does
   string surgery on the key, does not.

3. **Single-page SCAN.** ``COUNT`` is a hint, not a promise; a real Redis
   returns whatever it returns and the caller must loop on the cursor. This
   fake pages in threes no matter what ``COUNT`` says, so an adapter that
   reads one page and stops is caught, and it can be told to repeat a key
   across pages (``scan_duplicates``) because a real SCAN may return the same
   key twice under rehashing.

The Postgres fake dispatches on the SQL text and **raises on anything it does
not model**. Returning ``None`` for an unrecognised statement is how a fake
silently starts testing nothing: the adapter's new query would read as "no
row" and half the assertions would still pass.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

try:  # pragma: no cover — depends on whether the [redis] extra is installed
    from redis.exceptions import ResponseError, WatchError  # pyright: ignore[reportMissingImports]
except ImportError:
    # The fake must raise what `RedisStore` recognises. It matches on the
    # exception's CLASS NAME and message rather than on ``isinstance`` (see the
    # rationale in ``agentkit/adapters/store/redis.py``), so a local stand-in
    # with the same name and the same server message is indistinguishable to
    # the adapter — which is the property that keeps this fake honest whether
    # or not the extra happens to be installed.
    class ResponseError(Exception):  # type: ignore[no-redef]
        pass

    class WatchError(Exception):  # type: ignore[no-redef]
        pass


# Redis's own wording, verbatim. `RedisStore` translates on this text, so a
# fake that paraphrased it would be testing the translation against itself.
NOT_AN_INTEGER = "value is not an integer or out of range"

# How many keys one SCAN page returns, regardless of COUNT. Small on purpose:
# with a page size at or above the number of keys a test writes, the adapter's
# cursor loop would never go round twice and a break-on-first-page bug would
# survive.
_SCAN_PAGE = 3


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a Redis glob (``*``, ``?``, ``[abc]``, ``\\``-escapes) to a regex.

    Written out rather than delegated to ``fnmatch`` because ``fnmatch`` has no
    escape character: a pattern that escapes a literal ``*`` — precisely what
    `RedisStore.scan` emits for a prefix containing one — would be read as
    "backslash, then anything" and match nothing. The bug this catches exists
    only BECAUSE the two sides are independent: the adapter writes the escapes,
    this reads them.
    """
    out: list[str] = ["(?s)\\A"]
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\" and i + 1 < len(pattern):
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        if char == "*":
            out.append(".*")
        elif char == "?":
            out.append(".")
        elif char == "[":
            end = pattern.find("]", i + 1)
            if end == -1:
                out.append(re.escape(char))  # unterminated class → a literal '['
            else:
                body = pattern[i + 1 : end]
                negate = body.startswith("^")
                out.append("[" + ("^" if negate else "") + re.escape(body.lstrip("^")) + "]")
                i = end + 1
                continue
        else:
            out.append(re.escape(char))
        i += 1
    out.append("\\Z")
    return re.compile("".join(out))


class FakeRedis:
    """The ``redis.asyncio`` commands `RedisStore` issues, over dicts.

    ``clock`` is injectable so TTL is testable without sleeping — the counter
    expiry the rate-limit shape depends on is a *deadline*, and a test that
    slept a real second to observe it would be both slow and flaky.
    """

    def __init__(self, *, clock: Any = None) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.deadlines: dict[str, float] = {}  # key → absolute expiry; absent when persistent
        self.versions: dict[str, int] = {}  # bumped on every write; WATCH compares these
        self.scan_duplicates = False
        self._clock = clock or (lambda: 0.0)

    # ── internals ────────────────────────────────────────────────────────────

    def _expire_due(self) -> None:
        now = self._clock()
        for key in [k for k, deadline in self.deadlines.items() if now >= deadline]:
            self.kv.pop(key, None)
            self.lists.pop(key, None)
            self.deadlines.pop(key, None)
            self.versions[key] = self.versions.get(key, 0) + 1

    def _touch(self, key: str) -> None:
        self.versions[key] = self.versions.get(key, 0) + 1

    async def _tick(self) -> None:
        """A real scheduling point before every command — see the module docstring."""
        await asyncio.sleep(0)
        self._expire_due()

    # ── string / key commands ────────────────────────────────────────────────

    async def get(self, key: str) -> bytes | None:
        await self._tick()
        raw = self.kv.get(key)
        return None if raw is None else raw.encode()

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        await self._tick()
        self.kv[key] = value
        # SET without EX clears any prior TTL — the behaviour `InMemoryStore`
        # deliberately mirrors, so the fake has to have it too.
        self.deadlines.pop(key, None)
        if ex is not None:
            self.deadlines[key] = self._clock() + ex
        self._touch(key)
        return True

    async def delete(self, key: str) -> int:
        await self._tick()
        existed = self.kv.pop(key, None) is not None
        self.deadlines.pop(key, None)
        self._touch(key)
        return int(existed)

    async def incrby(self, key: str, amount: int) -> int:
        await self._tick()
        return self._incrby(key, amount)

    def _incrby(self, key: str, amount: int) -> int:
        raw = self.kv.get(key, "0")
        try:
            current = int(raw)
        except ValueError:
            raise ResponseError(NOT_AN_INTEGER) from None
        new = current + amount
        self.kv[key] = str(new)
        self._touch(key)
        return new

    async def expire(self, key: str, seconds: int, nx: bool = False) -> bool:
        await self._tick()
        return self._expire(key, seconds, nx)

    def _expire(self, key: str, seconds: int, nx: bool) -> bool:
        if key not in self.kv:
            return False
        if nx and key in self.deadlines:
            return False  # EXPIRE NX never moves a window that already exists
        self.deadlines[key] = self._clock() + seconds
        return True

    async def scan(
        self, cursor: int = 0, match: str | None = None, count: int | None = None
    ) -> tuple[int, list[bytes]]:
        await self._tick()
        keys = sorted(self.kv)
        page = keys[cursor : cursor + _SCAN_PAGE]
        nxt = cursor + _SCAN_PAGE
        if self.scan_duplicates and page:
            page = [page[0], *page]  # a real SCAN may return a key more than once
        pattern = _glob_to_regex(match) if match is not None else None
        hits = [k.encode() for k in page if pattern is None or pattern.match(k)]
        return (0 if nxt >= len(keys) else nxt), hits

    # ── list commands ────────────────────────────────────────────────────────

    async def rpush(self, key: str, value: str) -> int:
        await self._tick()
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        await self._tick()
        items = self.lists.get(key, [])
        window = items if end == -1 else items[start : end + 1]
        return [v.encode() for v in window]

    # ── transactions ─────────────────────────────────────────────────────────

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        return _FakePipeline(self)


class _FakePipeline:
    """WATCH / MULTI / EXEC, with the one property the adapter's correctness rests on:
    EXEC raises ``WatchError`` when a watched key changed after it was watched.

    Optimistic locking is the entire mechanism behind
    `RedisStore.compare_and_set`. A pipeline fake that just replayed the queued
    commands would make the compare-and-set look atomic while the real one raced.
    """

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._watched: dict[str, int] = {}
        self._queue: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._buffering = False

    async def __aenter__(self) -> _FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        await self.reset()
        return False

    async def watch(self, *keys: str) -> bool:
        await self._redis._tick()
        for key in keys:
            self._watched[key] = self._redis.versions.get(key, 0)
        return True

    async def unwatch(self) -> bool:
        self._watched.clear()
        return True

    async def reset(self) -> None:
        self._watched.clear()
        self._queue.clear()
        self._buffering = False

    def multi(self) -> None:
        self._buffering = True

    async def get(self, key: str) -> bytes | None:
        # Immediate-execution mode: before MULTI, redis-py sends the command and
        # awaits the reply. That is what makes read-then-CAS possible at all.
        assert not self._buffering, "get() after multi() is queued, not immediate"
        return await self._redis.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> _FakePipeline:
        self._queue.append(("set", (key, value), {"ex": ex}))
        return self

    def incrby(self, key: str, amount: int) -> _FakePipeline:
        self._queue.append(("incrby", (key, amount), {}))
        return self

    def expire(self, key: str, seconds: int, nx: bool = False) -> _FakePipeline:
        self._queue.append(("expire", (key, seconds), {"nx": nx}))
        return self

    async def execute(self) -> list[Any]:
        await self._redis._tick()
        for key, version in self._watched.items():
            if self._redis.versions.get(key, 0) != version:
                self._watched.clear()
                self._queue.clear()
                raise WatchError("Watched variable changed.")
        results: list[Any] = []
        # EXEC is atomic: no scheduling point between queued commands, which is
        # why the synchronous ``_incrby`` / ``_expire`` are used here.
        for name, args, kwargs in self._queue:
            if name == "set":
                self._redis.kv[args[0]] = args[1]
                self._redis.deadlines.pop(args[0], None)
                if kwargs["ex"] is not None:
                    self._redis.deadlines[args[0]] = self._redis._clock() + kwargs["ex"]
                self._redis._touch(args[0])
                results.append(True)
            elif name == "incrby":
                results.append(self._redis._incrby(*args))
            elif name == "expire":
                results.append(self._redis._expire(args[0], args[1], kwargs["nx"]))
            else:  # pragma: no cover — a command the fake does not model
                raise AssertionError(f"unmodelled queued command: {name}")
        self._queue.clear()
        self._watched.clear()
        return results


class FakePgPool:
    """The asyncpg pool surface `PostgresStore` drives, dispatching on SQL text.

    ``agentkit_kv.value`` is ``TEXT NOT NULL``, so a missing row is the only way
    ``fetchval`` yields ``None`` — preserved here, because the adapter's
    miss-versus-stored-null distinction is built on exactly that.
    """

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.log: dict[str, list[str]] = {}

    def acquire(self) -> _FakePgAcquire:
        return _FakePgAcquire(self)

    async def close(self) -> None:
        return None


class _FakePgConnection:
    def __init__(self, pool: FakePgPool) -> None:
        self._pool = pool

    async def _tick(self) -> None:
        await asyncio.sleep(0)  # see the module docstring: real interleaving

    async def execute(self, sql: str, *args: Any) -> None:
        await self._tick()
        if sql.startswith(("CREATE TABLE", "CREATE INDEX")):
            return
        if sql.startswith("INSERT INTO agentkit_kv (key, value)"):
            self._pool.kv[args[0]] = args[1]
            return
        if sql.startswith("DELETE FROM agentkit_kv"):
            self._pool.kv.pop(args[0], None)
            return
        if sql.startswith("INSERT INTO agentkit_log"):
            self._pool.log.setdefault(args[0], []).append(args[1])
            return
        raise AssertionError(f"FakePgPool does not model this statement: {sql!r}")

    async def fetchval(self, sql: str, *args: Any) -> Any:
        await self._tick()
        if sql.startswith("SELECT value FROM agentkit_kv"):
            return self._pool.kv.get(args[0])
        if sql.startswith("UPDATE agentkit_kv"):
            # compare_and_set against a present, non-null expected value.
            # ``value::jsonb = $3::jsonb`` is JSON equality, not text equality —
            # emulated by decoding both sides rather than comparing the strings.
            key, new, expected = args
            stored = self._pool.kv.get(key)
            if stored is None or json.loads(stored) != json.loads(expected):
                return None
            self._pool.kv[key] = new
            return 1
        if sql.startswith("INSERT INTO agentkit_kv (key, value)"):
            # compare_and_set with expected=None: insert when absent, and on
            # conflict update only when the stored value is JSON null.
            #
            # The ``ON CONFLICT ... WHERE`` clause is read out of the SQL
            # rather than assumed. A fake that hard-coded the guard would keep
            # answering correctly after the guard was deleted from the query —
            # it would be testing this file's intentions instead of the
            # statement the adapter actually sends, which is the one thing a
            # SQL fake exists to check.
            key, new = args
            stored = self._pool.kv.get(key)
            guarded = "value::jsonb = 'null'::jsonb" in sql
            if stored is not None and guarded and json.loads(stored) is not None:
                return None
            self._pool.kv[key] = new
            return 1
        if sql.startswith("INSERT INTO agentkit_kv AS kv"):
            # increment: one statement, insert-or-add. Same rule — the
            # integer-shaped guard is honoured only if the SQL carries it, and
            # without it the ``::bigint`` cast raises, which is exactly what
            # Postgres does (InvalidTextRepresentationError, SQLSTATE 22P02).
            key, seed, by = args
            stored = self._pool.kv.get(key)
            if stored is None:
                self._pool.kv[key] = seed
                return int(seed)
            if "kv.value ~ '^-?[0-9]+$'" in sql and not re.fullmatch(r"-?[0-9]+", stored):
                return None  # ON CONFLICT ... WHERE excluded the row
            new_total = int(stored) + by
            self._pool.kv[key] = str(new_total)
            return new_total
        raise AssertionError(f"FakePgPool does not model this statement: {sql!r}")

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        await self._tick()
        if sql.startswith("SELECT value FROM agentkit_log"):
            return [{"value": v} for v in self._pool.log.get(args[0], [])]
        if sql.startswith("SELECT key FROM agentkit_kv"):
            prefix = args[0]
            hits = [{"key": k} for k in sorted(self._pool.kv) if k.startswith(prefix)]
            return hits if len(args) < 2 else hits[: args[1]]
        raise AssertionError(f"FakePgPool does not model this statement: {sql!r}")


class _FakePgAcquire:
    def __init__(self, pool: FakePgPool) -> None:
        self._pool = pool

    async def __aenter__(self) -> _FakePgConnection:
        return _FakePgConnection(self._pool)

    async def __aexit__(self, *exc: object) -> bool:
        return False


__all__ = ["NOT_AN_INTEGER", "FakePgPool", "FakeRedis", "ResponseError", "WatchError"]
