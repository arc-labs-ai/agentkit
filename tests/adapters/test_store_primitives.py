"""Backend-SPECIFIC edges of `compare_and_set` / `increment` / `scan`.

The shared contract lives in ``tests/meta/test_protocol_conformance.py`` and
runs against all four adapters. What is here is the part that cannot be
expressed as a shared contract because it is about HOW one backend keeps the
promise:

* Redis's SCAN may return a key twice and may hand back a non-zero cursor with
  an empty page; both are invisible from the port's surface.
* Redis's optimistic lock only works if a ``WatchError`` from EXEC becomes
  ``False`` rather than escaping.
* Postgres's atomicity is "one statement", which no black-box test can observe
  — so the statement count is asserted directly.
* FileStore's key space is a directory, so its scan has to exclude two kinds
  of file that are not keys.

The live-server tests at the end follow the pattern the other adapter modules
use (an env-var skipif plus a reachability probe, so a configured-but-down
server skips instead of failing).
"""

import asyncio
import json
import os
import pathlib
import tempfile

import pytest
from _store_fakes import FakePgPool, FakeRedis

from agentkit.adapters.store import FileStore, InMemoryStore, PostgresStore, RedisStore
from agentkit.kernel.errors import StoreValueError


def _run(coro):
    return asyncio.run(coro)


async def _drain(keys) -> list:
    return [k async for k in keys]


# ---- RedisStore ---------------------------------------------------------------------------------


def test_redis_scan_pages_until_the_cursor_comes_back_zero():
    """SCAN's COUNT is a hint; the server decides how much to return. An
    adapter that read one page and stopped would report a fraction of the key
    space and look correct on any test whose fixture fits in one page — which
    is why the fake pages in threes regardless of COUNT."""
    store = RedisStore(client=FakeRedis(), namespace="ak")

    async def go():
        for i in range(10):
            await store.set(f"run:{i:02d}", i)
        return await _drain(store.scan("run:"))

    assert sorted(_run(go())) == [f"run:{i:02d}" for i in range(10)]


def test_redis_scan_returns_each_key_once_even_when_the_server_repeats_it():
    """A real SCAN may return the same key more than once when the keyspace
    rehashes mid-scan — Redis guarantees "at least once", never "exactly
    once". A caller replaying an audit prefix would process the record twice
    and have no way to tell why, so the adapter de-duplicates."""
    fake = FakeRedis()
    fake.scan_duplicates = True
    store = RedisStore(client=fake, namespace="ak")

    async def go():
        for i in range(5):
            await store.set(f"run:{i}", i)
        return await _drain(store.scan("run:"))

    keys = _run(go())
    assert sorted(keys) == [f"run:{i}" for i in range(5)]
    assert len(keys) == len(set(keys))


def test_redis_scan_does_not_leak_the_log_namespace_or_other_namespaces():
    """Two stores over one Redis is the normal multi-tenant wiring. The MATCH
    is anchored at ``<namespace>:kv:``, so neither the sibling namespace nor
    this store's own append-log keys can appear."""
    fake = FakeRedis()
    mine = RedisStore(client=fake, namespace="mine")
    theirs = RedisStore(client=fake, namespace="theirs")

    async def go():
        await mine.set("run:1", "a")
        await mine.append("run:1", "log-entry")
        await theirs.set("run:1", "not mine")
        return await _drain(mine.scan("run:"))

    assert _run(go()) == ["run:1"]


def test_a_watch_conflict_is_recognised_by_class_name_not_by_isinstance():
    """``redis`` is an optional extra, and this adapter is routinely built with
    an injected client and no ``redis`` package installed. An ``isinstance``
    check would need the import; a lazy import inside the ``except`` would
    resolve a DIFFERENT class than the one raised, so a lost race would escape
    as an error instead of returning ``False``."""
    from agentkit.adapters.store.redis import _is_watch_conflict

    class WatchError(Exception):
        """Same name, unrelated class — exactly the offline situation."""

    assert _is_watch_conflict(WatchError("Watched variable changed."))
    assert not _is_watch_conflict(RuntimeError("connection reset"))
    assert not _is_watch_conflict(ValueError("Watched variable changed."))


def test_redis_compare_and_set_reports_false_when_a_competitor_commits_mid_transaction():
    """The WATCH is the whole mechanism. This client reads the expected value,
    another client writes, and the EXEC must be discarded — surfaced to the
    caller as ``False``, not as a ``WatchError`` escaping the port.

    Simulated by making the pipeline's read itself trigger a competing write,
    which is the one interleaving that a single-threaded test cannot otherwise
    schedule reliably."""

    class _RacingRedis(FakeRedis):
        def pipeline(self, transaction=True):
            pipe = super().pipeline(transaction)
            read = pipe.get

            async def get_then_lose_the_race(key):
                value = await read(key)
                self.kv[key] = json.dumps("stolen")  # a competitor commits
                self._touch(key)
                return value

            pipe.get = get_then_lose_the_race
            return pipe

    store = RedisStore(client=_RacingRedis(), namespace="ak")

    async def go():
        await store.set("k", 1)
        applied = await store.compare_and_set("k", 1, 2)
        return applied, await store.get("k")

    assert _run(go()) == (False, "stolen")


def test_redis_increment_keeps_the_counter_and_its_expiry_in_one_transaction():
    """INCRBY and EXPIRE must reach the server as one unit. Sent separately, a
    process that dies between them leaves a counter with no window — a rate
    limit permanently exhausted for that key, with nothing to ever clear it.

    Asserted on the wire: the pipeline carries both commands and executes them
    together, rather than the adapter issuing INCRBY and then EXPIRE."""
    fake = FakeRedis()
    executed: list[list] = []
    original = fake.pipeline

    def recording_pipeline(transaction=True):
        pipe = original(transaction)
        execute = pipe.execute

        async def record():
            executed.append([name for name, _a, _k in pipe._queue])
            return await execute()

        pipe.execute = record
        return pipe

    fake.pipeline = recording_pipeline
    store = RedisStore(client=fake, namespace="ak")

    assert _run(store.increment("hits", ttl=60)) == 1
    assert executed == [["incrby", "expire"]]


def test_redis_increment_names_what_is_actually_in_the_key():
    """Redis says "value is not an integer or out of range" and nothing about
    which key or what type. The translated error carries both, because the
    caller's next question is always "what IS in there?"."""
    store = RedisStore(client=FakeRedis(), namespace="ak")

    async def go():
        await store.set("cfg", {"a": 1})
        await store.increment("cfg")

    with pytest.raises(StoreValueError, match=r"'cfg'.*dict"):
        _run(go())


def test_redis_increment_does_not_swallow_an_unrelated_error():
    """POSITIVE CONTROL for the message-matching translation: only Redis's
    not-an-integer reply becomes a `StoreValueError`. A connection failure must
    still surface as itself, or an outage reads as bad data."""

    class _BrokenRedis(FakeRedis):
        def pipeline(self, transaction=True):
            pipe = super().pipeline(transaction)

            async def execute():
                raise ConnectionError("Error 111 connecting to localhost:6379")

            pipe.execute = execute
            return pipe

    store = RedisStore(client=_BrokenRedis(), namespace="ak")
    with pytest.raises(ConnectionError):
        _run(store.increment("hits"))


# ---- PostgresStore ------------------------------------------------------------------------------


class _RecordingPgPool(FakePgPool):
    """Remembers every statement, so "the compare and the write are ONE
    statement" — the only thing that makes this backend's CAS atomic, and the
    one thing a black-box test over a fake cannot observe — can be asserted."""

    def __init__(self):
        super().__init__()
        self.statements: list[str] = []

    def acquire(self):
        acquirer = super().acquire()
        recorder = self

        class _Recording:
            async def __aenter__(self):
                con = await acquirer.__aenter__()
                for name in ("execute", "fetchval", "fetch"):
                    original = getattr(con, name)

                    def wrap(original=original):
                        async def call(sql, *args):
                            recorder.statements.append(sql)
                            return await original(sql, *args)

                        return call

                    setattr(con, name, wrap())
                return con

            async def __aexit__(self, *exc):
                return await acquirer.__aexit__(*exc)

        return _Recording()


def test_postgres_compare_and_set_is_a_single_statement():
    """A ``SELECT`` then an ``UPDATE`` is two statements, and at READ COMMITTED
    another transaction commits between them — the lost update the primitive
    exists to prevent. The predicate has to ride inside the write."""
    pool = _RecordingPgPool()
    store = PostgresStore(pool=pool)

    async def go():
        await store.set("k", 1)
        pool.statements.clear()
        return await store.compare_and_set("k", 1, 2)

    assert _run(go()) is True
    assert len(pool.statements) == 1, f"compare_and_set issued {pool.statements}"
    assert "UPDATE" in pool.statements[0] and "RETURNING" in pool.statements[0]


def test_postgres_increment_is_a_single_statement():
    """Same reason: read-add-write across two statements loses updates under
    concurrency, and the fake cannot race, so the shape is pinned directly."""
    pool = _RecordingPgPool()
    store = PostgresStore(pool=pool)

    assert _run(store.increment("hits", 3)) == 3
    assert len(pool.statements) == 1, f"increment issued {pool.statements}"
    assert "ON CONFLICT" in pool.statements[0]


def test_postgres_scan_uses_starts_with_rather_than_like():
    """``LIKE`` reads ``_`` as "any character" and ``%`` as "anything", and the
    prefix is caller data — keys are built from scopes, tool names and
    correlation ids. ``scan("run_1:")`` under LIKE would also return
    ``run-1:``'s keys, which on a store whose job is scoping is a cross-tenant
    read."""
    pool = _RecordingPgPool()
    store = PostgresStore(pool=pool)

    async def go():
        await store.set("run_1:a", 1)
        await store.set("run-1:a", 2)
        pool.statements.clear()
        return await _drain(store.scan("run_1:"))

    assert _run(go()) == ["run_1:a"]
    assert "LIKE" not in pool.statements[0]
    assert "starts_with" in pool.statements[0]


def test_postgres_refuses_a_ttl_on_every_method_that_takes_one():
    """There is no sweeper, so silent acceptance on ONE method would be worse
    than the blanket refusal: a caller would conclude the backend supports
    expiry and use it where expiry is load-bearing."""
    store = PostgresStore(pool=FakePgPool())

    async def go():
        with pytest.raises(NotImplementedError, match=r"compare_and_set does not support ttl"):
            await store.compare_and_set("k", None, 1, ttl=30)
        with pytest.raises(NotImplementedError, match=r"increment does not support ttl"):
            await store.increment("k", ttl=30)

    _run(go())


def test_postgres_increment_guard_leaves_the_offending_value_alone():
    """The ``ON CONFLICT ... WHERE`` filter excludes the row, so the failed
    increment is not a partial write — the JSON document that was there is
    still there, byte for byte."""
    pool = FakePgPool()
    store = PostgresStore(pool=pool)

    async def go():
        await store.set("cfg", {"retries": 3})
        with pytest.raises(StoreValueError):
            await store.increment("cfg")
        return await store.get("cfg")

    assert _run(go()) == {"retries": 3}


class _BoundedPgPool(FakePgPool):
    """`FakePgPool` with asyncpg's `max_size`, which the plain fake does not have.

    ``asyncpg.create_pool`` is always bounded (``max_size`` defaults to 10, and
    1 is an ordinary setting for a small worker), so a fake that hands out
    unlimited connections cannot see a method asking for a second one while it
    still holds the first. That is the only reason the deadlock below survived
    review: it is invisible to every test that uses the unbounded fake.
    """

    def __init__(self, size: int = 1) -> None:
        super().__init__()
        self._sem = asyncio.Semaphore(size)
        self.high_water = 0
        self._live = 0

    def acquire(self):
        pool = self
        inner = FakePgPool.acquire(self)

        class _Bounded:
            async def __aenter__(self):
                await pool._sem.acquire()
                pool._live += 1
                pool.high_water = max(pool.high_water, pool._live)
                return await inner.__aenter__()

            async def __aexit__(self, *exc):
                pool._live -= 1
                pool._sem.release()
                return False

        return _Bounded()


def test_postgres_increment_holds_only_one_connection_even_when_it_fails():
    """The FAILING increment must not need a second connection.

    `increment` re-reads the key to name the offending type in its error. Doing
    that inside the ``async with pool.acquire()`` that ran the statement asks
    the pool for a second connection while still holding the first: against
    ``max_size=1`` the call hung forever (measured — it never returned), and at
    any pool size N concurrent bad increments each hold one connection and wait
    for another, which is a deadlock at every size. A store method that works
    on a healthy key and hangs on an unhealthy one is the worst shape available.
    """
    pool = _BoundedPgPool(size=1)
    store = PostgresStore(pool=pool)

    async def go():
        await store.set("cfg", {"retries": 3})
        with pytest.raises(StoreValueError):
            # The timeout is the assertion: without it a regression hangs the
            # suite instead of failing it.
            await asyncio.wait_for(store.increment("cfg"), timeout=5)
        return await store.get("cfg")

    assert _run(go()) == {"retries": 3}
    assert pool.high_water == 1, f"increment held {pool.high_water} connections at once"


# ---- FileStore ----------------------------------------------------------------------------------


def test_file_scan_ignores_log_files_and_in_flight_temp_files():
    """The directory holds three kinds of file and only one is a key. A
    half-written ``.tmp`` from `_write_atomic` is the dangerous one: surfacing
    it would report a key whose `get` raises, and it exists precisely while a
    concurrent writer is running — which is when scans happen."""
    directory = tempfile.mkdtemp()
    store = FileStore(directory)

    async def go():
        await store.set("run:1", "kv")
        await store.append("run:1", "log-entry")
        return await _drain(store.scan("run:"))

    # Exactly what `_write_atomic` leaves behind mid-write. Written out here
    # rather than inside the coroutine: blocking pathlib calls on the event
    # loop are what `asyncio.to_thread` exists to avoid, and ruff's ASYNC240
    # holds tests to that too.
    pathlib.Path(directory, "run%3A2.jsonabc123.tmp").write_text('{"half')
    assert _run(go()) == ["run:1"]


def test_file_scan_decodes_percent_encoded_filenames_back_to_keys():
    """Keys are percent-encoded to make them collision-free filenames, so the
    prefix has to be compared against the DECODED key. Comparing the encoded
    forms would make ``scan("checkpoint:org1/")`` match nothing at all — the
    stored name is ``checkpoint%3Aorg1%2F...``."""
    store = FileStore(tempfile.mkdtemp())

    async def go():
        await store.set("checkpoint:org1/run-1", "a")
        await store.set("checkpoint:org2/run-1", "b")
        return await _drain(store.scan("checkpoint:org1/"))

    assert _run(go()) == ["checkpoint:org1/run-1"]


def test_file_store_warns_once_across_all_three_ttl_taking_methods():
    """The latch is per store, not per method. Three copies of it would mean
    three warnings from one store — and a warning that repeats is a warning
    that gets filtered and stops being read."""
    import warnings

    store = FileStore(tempfile.mkdtemp())

    with pytest.warns(UserWarning, match="ignores ttl"):
        _run(store.set("k", 1, ttl=60))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(store.compare_and_set("k", 1, 2, ttl=60))
        _run(store.increment("c", ttl=60))
    assert not [w for w in caught if "ignores ttl" in str(w.message)]


def test_file_compare_and_set_and_increment_share_one_lock_table():
    """Two lock tables keyed on the same string give mutual exclusion within
    each operation and NONE between them — a `get_or_set` producer writing its
    result while a `compare_and_set` reads is exactly the lost update the
    primitive exists to prevent.

    Asserted by OBSERVING the shared entry while both callers are in flight,
    not by listing what the table still holds afterwards. The earlier version
    of this test ran the three operations to completion and asserted
    ``sorted(store._locks) == ["counter", "k"]`` — which passes only because
    the table never reclaimed, so it pinned the write-only lock table
    `_keylock` exists to prevent as if it were the intended design. A test
    whose assertion is satisfied by a leak cannot also detect one.
    """
    store = FileStore(tempfile.mkdtemp())
    observed: dict[str, object] = {}
    released = asyncio.Event()
    holding = asyncio.Event()

    async def slow_producer():
        # The producer body runs ONLY while `get_or_set` holds the lock, so
        # this event firing is proof the lock is held — waiting on a fixed
        # number of `sleep(0)`s instead is not, because `get_or_set` hops to a
        # thread for `_exists` before it ever reaches the lock.
        holding.set()
        await released.wait()
        return "made"

    async def go():
        producer = asyncio.create_task(store.get_or_set("k", slow_producer))
        await holding.wait()  # the producer now definitely holds "k"
        contender = asyncio.create_task(store.compare_and_set("k", "made", "swapped"))
        await asyncio.sleep(0)  # let the contender queue on the SAME entry

        # One entry for "k", two callers on it: `get_or_set` and
        # `compare_and_set` resolved the same string to the same lock. Separate
        # tables would show users == 1 and the contender running free.
        observed["entries"] = sorted(store._locks)
        observed["users"] = store._locks["k"].users
        observed["contender_done_early"] = contender.done()

        released.set()
        await producer
        assert await contender is True
        await store.increment("counter")
        observed["after"] = sorted(store._locks)

    _run(go())
    assert observed["entries"] == ["k"]
    assert observed["users"] == 2, "the two operations took different locks for one key"
    assert observed["contender_done_early"] is False, "the contender did not wait"
    assert observed["after"] == [], "the shared table is write-only again"


async def _made():
    return "made"


# ---- InMemoryStore ------------------------------------------------------------------------------


def test_memory_scan_skips_a_key_whose_ttl_has_passed():
    """Expiry is a lazy deadline check, so an expired key is still physically
    in the dict until something reads it. A scan that ignored the deadline
    would enumerate keys that `get` answers ``None`` for — and reclaim code
    built on scan would then "delete" entries that were already gone while
    missing the live ones."""
    now = {"t": 0.0}
    store = InMemoryStore(clock=lambda: now["t"])

    async def go():
        await store.set("run:keeps", 1)
        await store.set("run:expires", 1, ttl=10)
        now["t"] = 10.0
        return await _drain(store.scan("run:"))

    assert _run(go()) == ["run:keeps"]


def test_memory_compare_and_set_does_not_take_the_single_flight_lock():
    """Deliberate: there is no ``await`` between the read and the write, so the
    sequence is already atomic under asyncio's cooperative scheduling. Taking
    `get_or_set`'s lock would be strictly worse — a producer that
    compare-and-sets the key it is producing would deadlock against its own
    single-flight entry."""
    store = InMemoryStore()

    async def go():
        async def produce():
            # The producer CASes its own key from inside get_or_set.
            await store.compare_and_set("k", None, "staked")
            return "made"

        return await store.get_or_set("k", produce)

    assert _run(asyncio.wait_for(go(), timeout=1.0)) == "made"


# ---- LIVE backends (gated on reachability, same pattern as the other adapter modules) ------------


def _pg_dsn() -> str | None:
    if os.getenv("AGENTKIT_TEST_PG_DSN"):
        return os.environ["AGENTKIT_TEST_PG_DSN"]
    host = os.getenv("DB_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.getenv('DB_USER', 'agentkit')}:{os.getenv('DB_PASSWORD', 'agentkit')}"
        f"@{host}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'agentkit')}"
    )


def _redis_url() -> str | None:
    return os.getenv("AGENTKIT_TEST_REDIS_URL") or os.getenv("REDIS_URL")


async def _reachable(coro_factory) -> bool:
    try:
        await asyncio.wait_for(coro_factory(), timeout=3.0)
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(
    _pg_dsn() is None, reason="set AGENTKIT_TEST_PG_DSN (or DB_HOST) for a live Postgres"
)
def test_the_three_primitives_against_a_live_postgres():
    """The fake models the SQL; only a real server checks that the SQL is
    valid Postgres — ``ON CONFLICT ... WHERE``, the ``::jsonb`` comparison and
    ``starts_with`` all parse here or not at all."""
    pytest.importorskip("asyncpg")
    import asyncpg  # pyright: ignore[reportMissingImports]

    dsn = _pg_dsn()

    async def _probe():
        con = await asyncpg.connect(dsn)
        await con.close()

    if not _run(_reachable(_probe)):
        pytest.skip("Postgres not reachable at the configured DSN")

    store = PostgresStore(dsn)

    async def go():
        await store.init()
        try:
            keys = [f"ak:prim:{i}" for i in range(3)]
            for key in [*keys, "ak:prim-sibling"]:
                await store.delete(key)
            await store.delete("ak:prim:counter")

            assert await store.compare_and_set(keys[0], None, {"v": 1}) is True
            assert await store.compare_and_set(keys[0], {"v": 9}, "no") is False
            assert await store.compare_and_set(keys[0], {"v": 1}, [1, 2]) is True
            assert await store.get(keys[0]) == [1, 2]

            assert await store.increment("ak:prim:counter", 5) == 5
            assert await store.increment("ak:prim:counter", -2) == 3
            with pytest.raises(StoreValueError):
                await store.increment(keys[0])

            await store.set(keys[1], 1)
            await store.set("ak:prim-sibling", 1)
            found = set(await _drain(store.scan("ak:prim:")))
            assert {keys[0], keys[1], "ak:prim:counter"} <= found
            assert "ak:prim-sibling" not in found
        finally:
            await store.aclose()

    _run(go())


@pytest.mark.integration
@pytest.mark.skipif(
    _redis_url() is None, reason="set AGENTKIT_TEST_REDIS_URL (or REDIS_URL) for a live Redis"
)
def test_the_three_primitives_against_a_live_redis():
    """Only a real server checks the parts the fake defines rather than
    models: that ``EXPIRE ... NX`` exists (Redis 7.0+), that WATCH really does
    abort the EXEC, and that SCAN's MATCH reads our escapes the way Redis's
    own glob does."""
    pytest.importorskip("redis")
    import redis.asyncio as aioredis  # pyright: ignore[reportMissingImports]

    url = _redis_url()

    async def _probe():
        client = aioredis.from_url(url)
        await client.ping()
        await client.aclose()

    if not _run(_reachable(_probe)):
        pytest.skip("Redis not reachable at the configured URL")

    async def go():
        store = RedisStore(url, namespace="agentkit-test-prim")
        try:
            for key in ("a", "b", "counter", "cfg"):
                await store.delete(key)

            assert await store.compare_and_set("a", None, {"v": 1}) is True
            assert await store.compare_and_set("a", {"v": 9}, "no") is False
            assert await store.compare_and_set("a", {"v": 1}, [1, 2]) is True
            assert await store.get("a") == [1, 2]

            assert await store.increment("counter", 5, ttl=60) == 5
            assert await store.increment("counter", -2, ttl=60) == 3
            await store.set("cfg", {"a": 1})
            with pytest.raises(StoreValueError):
                await store.increment("cfg")

            await store.set("b", 1)
            found = set(await _drain(store.scan("")))
            assert {"a", "b", "counter", "cfg"} <= found
        finally:
            await store.aclose()

    _run(go())
