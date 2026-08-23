"""Durable resume against real StorePort backends.

Proves the production scenario — a workflow suspends at a human gate, its checkpoint is persisted to the
backend, and a brand-new workflow instance resumes it from that backend and completes — for each durable
store, not just FileStore:

  • RedisStore over a JSON-round-tripping in-memory fake — always runs (it exercises the real
    encode→store→fetch→decode path a live Redis takes, which is the actual serialization risk).
  • A LIVE Postgres — runs when `AGENTKIT_TEST_PG_DSN` (or the `DB_*` env) points at a reachable server.
  • A LIVE Redis — runs when `AGENTKIT_TEST_REDIS_URL` (or `REDIS_URL`) points at a reachable server.

Each live test runs the whole scenario in ONE event loop (a pooled connection is bound to its loop).
"""

import asyncio
import os

import pytest

from agentkit.adapters.store import PostgresStore, RedisStore
from agentkit.agents import Agent, Workflow

# Workflow persists through the shared ``resolve_checkpointer`` seam, so a
# store-only wiring is bridged by ``StoreBackedCheckpointStore`` and the record
# lands at ``ckpt_key`` (``checkpoint:<run_id>``). That is the ONLY key this
# test needs to touch: the workflow's old private ``workflow:<run_id>`` slot is
# gone, and asserting on the current key is what keeps this exercising the real
# encode -> store -> fetch -> decode path, which is the serialization risk the
# test exists for.
from agentkit.capabilities.checkpointer import ckpt_key
from agentkit.kernel.types import Scope
from agentkit.testing import FakeLLM, make_test_ctx

_RUN_ID = "wf-int-1"


def _run(coro):
    return asyncio.run(coro)


def _wf() -> Workflow:
    wf = Workflow()
    wf.agent("design", Agent("d", "m"))
    wf.human_gate("review", after="design")
    wf.fn("ship", lambda inp: f"shipping with {inp['review']}", after="review")
    return wf


async def _resume_cycle(store):
    """Suspend → assert the checkpoint is in the backend → resume from a fresh workflow → assert it
    completes and the checkpoint is reclaimed. `store` is the real durable backend under test."""
    await store.delete(ckpt_key(_RUN_ID))  # clean slate (idempotent)

    res = await _wf().run(
        "build",
        make_test_ctx(
            llm=FakeLLM("PLAN"),
            store=store,
            correlation_id=_RUN_ID,
            scope=Scope(1, 2),
        ),
    )
    assert res.stop_reason == "suspended" and res.suspended.pending == ("review",)
    assert "ship" not in res.outputs  # downstream did not run
    assert (
        await store.get(ckpt_key(_RUN_ID)) is not None
    )  # checkpoint durably persisted in the backend

    res2 = await _wf().resume(
        res.suspended.run_id,
        {"review": "approved"},
        make_test_ctx(
            llm=FakeLLM("PLAN"),
            store=store,
            correlation_id=_RUN_ID,
            scope=Scope(1, 2),
        ),
    )
    assert res2.stop_reason == "complete"
    assert res2.outputs["ship"] == "shipping with approved"  # resumed from the backend and finished
    assert await store.get(ckpt_key(_RUN_ID)) is None  # terminal completion reclaimed it


# ---- RedisStore over a JSON-round-tripping fake (always runs) ----------------------------------


class _FakeRedis:
    """The few redis.asyncio methods RedisStore needs for checkpoints — and it stores the JSON-encoded
    string RedisStore hands it, so this exercises the same serialization path a live Redis takes."""

    def __init__(self):
        self.kv: dict = {}

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = v

    async def delete(self, k):
        self.kv.pop(k, None)


def test_durable_resume_through_redis_store_serialization():
    _run(_resume_cycle(RedisStore(client=_FakeRedis())))


# ---- LIVE backends (gated on reachability) -----------------------------------------------------


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


@pytest.mark.skipif(
    _pg_dsn() is None, reason="set AGENTKIT_TEST_PG_DSN (or DB_HOST) for a live Postgres"
)
def test_durable_resume_against_live_postgres():
    pytest.importorskip("asyncpg")
    import asyncpg  # pyright: ignore[reportMissingImports]

    dsn = _pg_dsn()

    async def _probe():
        con = await asyncpg.connect(dsn)
        await con.close()

    if not _run(_reachable(_probe)):
        pytest.skip("Postgres not reachable at the configured DSN")

    async def go():
        store = PostgresStore(dsn)
        await store.init()
        try:
            await _resume_cycle(store)
        finally:
            await store.aclose()

    _run(go())


@pytest.mark.skipif(
    _redis_url() is None, reason="set AGENTKIT_TEST_REDIS_URL (or REDIS_URL) for a live Redis"
)
def test_durable_resume_against_live_redis():
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
        store = RedisStore(url, namespace="agentkit-test")
        try:
            await _resume_cycle(store)
        finally:
            await store.aclose()

    _run(go())
