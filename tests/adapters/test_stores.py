"""Durable StorePort adapters. RedisStore is unit-tested against a faithful in-memory fake of the few
redis.asyncio methods it uses (verifiable offline). PostgresStore's integration test runs only against a
live Postgres (CI's `test` job has the service); it's skipped locally."""

import asyncio
import os

import pytest

from agentkit.adapters.store import PostgresStore, RedisStore


def _run(coro):
    return asyncio.run(coro)


class _FakeRedis:
    """In-memory stand-in for the redis.asyncio surface RedisStore uses: get/set/rpush/lrange."""

    def __init__(self):
        self.kv: dict = {}
        self.lists: dict = {}

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = v

    async def delete(self, k):
        self.kv.pop(k, None)

    async def rpush(self, k, v):
        self.lists.setdefault(k, []).append(v)

    async def lrange(self, k, start, end):
        items = self.lists.get(k, [])
        return items if end == -1 else items[start : end + 1]


# ---- RedisStore (offline, fake client) --------------------------------------------------------


def test_redis_store_kv_and_namespacing():
    fake = _FakeRedis()
    s = RedisStore(client=fake, namespace="ak")
    assert _run(s.get("k")) is None
    _run(s.set("k", {"v": 1}))
    assert _run(s.get("k")) == {"v": 1}
    assert "ak:kv:k" in fake.kv  # namespaced + JSON-encoded


def test_redis_store_get_or_set_single_flight():
    s = RedisStore(client=_FakeRedis())
    calls = {"n": 0}

    async def produce():
        calls["n"] += 1
        return {"made": True}

    async def go():
        a = await s.get_or_set("g", produce)
        b = await s.get_or_set("g", produce)  # hit — fn not re-run
        return a, b

    a, b = _run(go())
    assert a == b == {"made": True} and calls["n"] == 1


def test_redis_store_append_and_list():
    s = RedisStore(client=_FakeRedis())

    async def go():
        await s.append("log", {"a": 1})
        await s.append("log", {"b": 2})
        return await s.list("log")

    assert _run(go()) == [{"a": 1}, {"b": 2}]


def test_store_delete_is_idempotent():
    """`delete` removes a key and treats a missing key as a no-op — the shape checkpoint reclamation relies on."""
    from agentkit.adapters.store import InMemoryStore

    for s in (InMemoryStore(), RedisStore(client=_FakeRedis())):

        async def go(store=s):
            await store.set("k", {"v": 1})
            assert await store.get("k") == {"v": 1}
            await store.delete("k")
            assert await store.get("k") is None
            await store.delete("k")  # missing key → no error

        _run(go())


# ---- PostgresStore (live DB only — CI test job has Postgres) -----------------------------------


def _pg_dsn() -> str | None:
    """Resolve the live Postgres DSN from env, matching the other test
    modules' precedence: ``AGENTKIT_TEST_PG_DSN`` (full DSN) takes
    priority, then the ``DB_*`` decomposition with ``agentkit`` defaults
    that CI's test job uses."""
    if os.getenv("AGENTKIT_TEST_PG_DSN"):
        return os.environ["AGENTKIT_TEST_PG_DSN"]
    host = os.getenv("DB_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.getenv('DB_USER', 'agentkit')}:{os.getenv('DB_PASSWORD', 'agentkit')}"
        f"@{host}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'agentkit')}"
    )


async def _reachable(coro_factory, *, timeout: float = 3.0) -> bool:
    """Probe a connection: ``True`` if the coroutine completes within
    ``timeout``, ``False`` on any exception or timeout. Matches the
    pattern in ``test_postgres_checkpoint.py`` and
    ``test_durable_resume_backends.py``."""
    try:
        await asyncio.wait_for(coro_factory(), timeout=timeout)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    _pg_dsn() is None,
    reason="set AGENTKIT_TEST_PG_DSN (or DB_HOST) for a live Postgres",
)
def test_postgres_store_durable_roundtrip():
    """Regression for the dual-gate pattern: the env-var skipif fires
    when no DSN is configured; the reachability probe fires when a DSN
    is configured but the server is down (e.g., dev running tests with
    docker-compose stopped). Either path → ``pytest.skip``, never a
    spurious test failure."""
    pytest.importorskip("asyncpg")
    import asyncpg  # type: ignore[import-untyped]  # asyncpg has no py.typed marker

    dsn = _pg_dsn()

    async def _probe() -> None:
        con = await asyncpg.connect(dsn)
        await con.close()

    if not _run(_reachable(_probe)):
        pytest.skip("Postgres not reachable at the configured DSN")

    s = PostgresStore(dsn)

    async def go():
        await s.init()
        await s.set("ak:test:k", {"v": 1})
        assert await s.get("ak:test:k") == {"v": 1}
        got = await s.get_or_set("ak:test:g", lambda: _coro({"made": True}))
        assert got == {"made": True}
        await s.append("ak:test:log", "a")
        await s.append("ak:test:log", "b")
        assert (await s.list("ak:test:log"))[-2:] == ["a", "b"]

    _run(go())


async def _coro(v):
    return v


# ── ``ttl`` is honored uniformly or raises explicitly ────────────────────────
#
# The three ``StorePort`` backends share one signature but have very
# different durability semantics for ``ttl``. RedisStore honors it via
# SETEX; InMemoryStore expires keys via a lazy deadline check;
# PostgresStore has no sweeper and raises ``NotImplementedError`` on a
# non-None ``ttl`` so the contract gap is visible at the seam instead
# of surfacing as an indefinite-retention surprise in production.


def test_inmemory_ttl_expires_key_after_deadline():
    """A key set with ``ttl=10`` is gone once the clock crosses the
    deadline; a subsequent ``set`` without ttl clears the prior expiry."""
    from agentkit.adapters.store import InMemoryStore

    now = {"t": 0.0}

    def clk() -> float:
        return now["t"]

    s = InMemoryStore(clock=clk)

    async def go() -> None:
        await s.set("k", "v", ttl=10)
        assert await s.get("k") == "v"
        # Just before expiry.
        now["t"] = 9.9
        assert await s.get("k") == "v"
        # After deadline.
        now["t"] = 10.0
        assert await s.get("k") is None
        # Subsequent set without ttl clears the prior expiry.
        await s.set("k", "v2")
        now["t"] = 1_000.0
        assert await s.get("k") == "v2"

    _run(go())


def test_inmemory_get_or_set_applies_ttl():
    """``get_or_set`` plumbs ttl through to set so the contract matches
    Redis; if it were dropped, a cache key with an expected expiry
    would live forever."""
    from agentkit.adapters.store import InMemoryStore

    now = {"t": 0.0}

    def clk() -> float:
        return now["t"]

    s = InMemoryStore(clock=clk)

    async def go() -> None:
        async def producer():
            return "fresh"

        v1 = await s.get_or_set("k", producer, ttl=5)
        assert v1 == "fresh"
        # Within TTL — still cached.
        now["t"] = 4.0
        v2 = await s.get_or_set("k", producer, ttl=5)
        assert v2 == "fresh"
        # After TTL — producer re-runs.
        now["t"] = 5.0
        v3 = await s.get_or_set("k", lambda: _coro("regenerated"), ttl=5)
        assert v3 == "regenerated"

    _run(go())


def test_inmemory_no_ttl_means_no_expiry():
    """Regression — keys set without ttl don't accidentally expire."""
    from agentkit.adapters.store import InMemoryStore

    now = {"t": 0.0}
    s = InMemoryStore(clock=lambda: now["t"])

    async def go() -> None:
        await s.set("k", "forever")
        now["t"] = 1e9
        assert await s.get("k") == "forever"

    _run(go())


def test_postgres_set_with_ttl_raises_explicitly():
    """PostgresStore has no sweeper, so the only honest TTL outcome is
    refusal — it raises ``NotImplementedError`` pointing callers to
    Redis rather than silently dropping ``ttl`` and retaining the key
    indefinitely."""
    import pytest as _pytest

    s = PostgresStore("postgres://ignored")

    async def go() -> None:
        with _pytest.raises(NotImplementedError, match=r"PostgresStore.set does not support ttl"):
            await s.set("k", "v", ttl=300)
        with _pytest.raises(
            NotImplementedError, match=r"PostgresStore.get_or_set does not support ttl"
        ):
            await s.get_or_set("k", lambda: _coro("x"), ttl=300)

    _run(go())


# ── Single-flight under concurrent load ──────────────────────────────────────
#
# ``get_or_set`` is the seam for memoize / idempotency / caching. The
# single-flight promise — "on a cache miss, ``fn`` runs exactly once
# even under concurrent misses" — is what keeps the framework from
# stampeding a slow producer under load. These tests exercise the
# concurrent path directly.


def test_inmemory_get_or_set_runs_fn_once_under_concurrent_misses() -> None:
    """Twenty coroutines race the same cold key. The producer's
    counter proves ``fn`` fired exactly once; every coroutine sees
    the same value. Verifies the lock covers the miss path (not just
    the surface API)."""
    import asyncio as _aio

    from agentkit.adapters.store import InMemoryStore

    s = InMemoryStore()
    call_count = {"n": 0}

    async def slow_producer():
        # Yield to the loop so other coroutines get scheduled. Without
        # this, the producer completes before contenders even wake up
        # and the test would pass trivially.
        await _aio.sleep(0)
        call_count["n"] += 1
        return {"generated": True}

    async def go():
        results = await _aio.gather(*(s.get_or_set("cold-key", slow_producer) for _ in range(20)))
        return results

    results = _run(go())
    # Every caller sees the same value (single-flight semantics).
    assert all(r == {"generated": True} for r in results)
    # And the producer ran exactly once, not 20 times.
    assert call_count["n"] == 1


def test_inmemory_get_or_set_isolates_keys_under_concurrent_load() -> None:
    """Concurrent misses on DIFFERENT keys must NOT serialize — the
    lock is per-key, not global. Ten distinct keys' producers all
    fire; the lock table doesn't turn into a global bottleneck."""
    import asyncio as _aio

    from agentkit.adapters.store import InMemoryStore

    s = InMemoryStore()
    per_key_calls: dict[str, int] = {}

    async def go():
        async def produce(k: str):
            await _aio.sleep(0)
            per_key_calls[k] = per_key_calls.get(k, 0) + 1
            return f"value-{k}"

        # Ten distinct keys, each hit concurrently by three coroutines.
        keys = [f"key-{i}" for i in range(10)]
        return await _aio.gather(
            *(s.get_or_set(k, lambda k=k: produce(k)) for k in keys for _ in range(3))
        )

    results = _run(go())
    # 30 calls total (10 keys × 3 racers), 10 distinct values, each
    # producer fired exactly once for its key.
    assert len(results) == 30
    assert set(results) == {f"value-key-{i}" for i in range(10)}
    assert per_key_calls == {f"key-{i}": 1 for i in range(10)}


def test_inmemory_get_or_set_raising_fn_does_not_cache_and_next_call_retries() -> None:
    """A ``fn`` that raises must not poison the cache. The raise
    propagates to the caller (unstored), and a subsequent call gets
    a fresh attempt. This is the single-flight *failure* contract —
    transient errors retry clean."""

    from agentkit.adapters.store import InMemoryStore

    s = InMemoryStore()
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("temporary provider hiccup")
        return "eventually-ok"

    async def go():
        # First two attempts raise; the third succeeds.
        for expected_n in range(1, 3):
            try:
                await s.get_or_set("k", flaky)
                assert False, "should have raised"
            except RuntimeError:
                pass
            # Cache is still empty (the failure wasn't stored).
            assert await s.get("k") is None
            assert attempts["n"] == expected_n

        # Third attempt succeeds and IS stored.
        result = await s.get_or_set("k", flaky)
        assert result == "eventually-ok"
        assert attempts["n"] == 3
        # Fourth caller sees the stored success — no additional fn calls.
        again = await s.get_or_set("k", flaky)
        assert again == "eventually-ok"
        assert attempts["n"] == 3

    _run(go())


def test_inmemory_get_or_set_concurrent_racers_after_failure_still_single_flight() -> None:
    """The failure-recovery path stays single-flight under load. Two
    contenders race a cold key with a flaky producer; the first
    attempt raises, the SECOND attempt (also concurrent with a third
    racer) succeeds and both racers see the stored success."""
    import asyncio as _aio

    from agentkit.adapters.store import InMemoryStore

    s = InMemoryStore()
    state = {"attempts": 0}

    async def once_flaky():
        await _aio.sleep(0)
        state["attempts"] += 1
        if state["attempts"] == 1:
            raise RuntimeError("first attempt fails")
        return "stable"

    async def go():
        # Round 1: two racers, one fails, both raise (they share the
        # lock; the first through the door runs fn and raises; the
        # second wakes, re-checks — cache still empty — runs fn again).
        results = await _aio.gather(
            s.get_or_set("k", once_flaky),
            s.get_or_set("k", once_flaky),
            return_exceptions=True,
        )
        # One racer got the exception; the other got the recovery.
        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1 and successes[0] == "stable"
        assert len(failures) == 1
        # Producer ran exactly twice: once failed, once succeeded.
        assert state["attempts"] == 2

    _run(go())


# ── FileStore durability: the adapter whose whole job is surviving a crash ───


def test_filestore_writes_atomically() -> None:
    """A plain `path.write_text` is not atomic, and this adapter exists to
    "survive a process restart, so a human-gate suspend or a crashed run
    resumes from disk". A crash DURING the write left a truncated file and
    every later `get` raised — the checkpoint became permanently unreadable
    and the run could never resume. The failure mode the adapter is FOR was
    the one that broke it.

    Atomicity is asserted through its observable consequence: a failure at the
    rename must leave the previous value fully readable, and must not litter
    the directory with scratch files.
    """
    import os
    import pathlib
    import tempfile

    from agentkit.adapters.store import FileStore

    d = tempfile.mkdtemp()
    store = FileStore(d)

    async def go():
        await store.set("ckpt", {"v": "original"})
        real = os.replace
        os.replace = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            with pytest.raises(OSError):
                await store.set("ckpt", {"v": "new"})
        finally:
            os.replace = real
        return await store.get("ckpt")

    assert asyncio.run(go()) == {"v": "original"}, "a failed write destroyed the old value"
    leftovers = [f.name for f in pathlib.Path(d).iterdir() if f.name.endswith(".tmp")]
    assert leftovers == [], f"scratch files left behind: {leftovers}"


def test_filestore_reports_a_corrupt_entry_with_its_path() -> None:
    """Writes are atomic now, so an unparseable file means EXTERNAL corruption
    (a disk fault, a hand-edit, an older non-atomic build). Returning None
    would report "no checkpoint" and restart a run that has durable state, so
    it raises — but a bare JSONDecodeError from inside a `to_thread` frame
    gives an operator nothing to act on, so the message names the file."""
    import pathlib
    import tempfile

    from agentkit.adapters.store import FileStore
    from agentkit.kernel.errors import StoreUnavailable

    d = tempfile.mkdtemp()
    store = FileStore(d)
    asyncio.run(store.set("ckpt", {"goal": "build"}))
    path = next(p for p in pathlib.Path(d).iterdir() if p.suffix == ".json")
    path.write_text(path.read_text()[:8])  # external truncation

    with pytest.raises(StoreUnavailable) as exc:
        asyncio.run(store.get("ckpt"))
    assert "ckpt" in str(exc.value) and str(path) in str(exc.value)


def test_one_torn_log_line_does_not_destroy_the_audit_trail() -> None:
    """`list()` did `json.loads` over every line and raised on the first bad
    one — so a crash during an append made every EARLIER audit record
    unreadable too. An append-only log degrades to "the records that
    survived", which is what an audit trail is for."""
    import pathlib
    import tempfile

    from agentkit.adapters.store import FileStore

    d = tempfile.mkdtemp()
    store = FileStore(d)

    async def go():
        await store.append("audit", {"a": 1})
        await store.append("audit", {"b": 2})

    asyncio.run(go())
    log = next(p for p in pathlib.Path(d).iterdir() if p.suffix == ".log")
    log.write_text(log.read_text() + '{"c": ')  # a crash mid-append

    with pytest.warns(UserWarning, match="unparseable"):
        records = asyncio.run(store.list("audit"))
    assert records == [{"a": 1}, {"b": 2}], "surviving records were lost with the torn one"


def test_filestore_says_so_when_it_ignores_a_ttl() -> None:
    """FileStore has no expiry sweeper, so a ttl is silently permanent. That
    matters most for idempotency: a key that never expires dedupes a
    legitimate retry of the same operation forever. Documented in a docstring
    is not the same as visible at the call site."""
    import tempfile
    import warnings

    from agentkit.adapters.store import FileStore

    store = FileStore(tempfile.mkdtemp())

    with pytest.warns(UserWarning, match="ignores ttl"):
        asyncio.run(store.set("k", "v", ttl=60))

    # Once per store, not once per call — a per-call warning gets filtered.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(store.set("k2", "v", ttl=60))
    assert not [w for w in caught if "ignores ttl" in str(w.message)]

    # And no warning at all when no ttl is asked for.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(FileStore(tempfile.mkdtemp()).set("k", "v"))
    assert caught == []
