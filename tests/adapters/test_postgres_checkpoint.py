"""Live-Postgres integration test for `PostgresCheckpointStore`.

Skipped unless a Postgres DSN is available (matches `test_durable_resume_backends.py`'s gating —
`AGENTKIT_TEST_PG_DSN` or the `DB_*` env vars). The whole round-trip runs in one event loop:
the asyncpg pool is bound to its loop, so save/latest/at_version/list_versions/delete must share it.
"""

import asyncio
import os

import pytest

from agentkit.adapters.checkpoint import PostgresCheckpointStore
from agentkit.kernel.ports import Checkpoint, CheckpointStatus

_RUN_ID = "pg-ckpt-int-1"


def _run(coro):
    return asyncio.run(coro)


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


async def _reachable(coro_factory) -> bool:
    try:
        await asyncio.wait_for(coro_factory(), timeout=3.0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    _pg_dsn() is None,
    reason="set AGENTKIT_TEST_PG_DSN (or DB_HOST) for a live Postgres checkpoint test",
)
def test_postgres_checkpoint_store_roundtrip():
    pytest.importorskip("asyncpg")
    import asyncpg  # pyright: ignore[reportMissingImports]

    dsn = _pg_dsn()

    async def _probe():
        con = await asyncpg.connect(dsn)
        await con.close()

    if not _run(_reachable(_probe)):
        pytest.skip("Postgres not reachable at the configured DSN")

    async def go():
        # Application-owned pool — adapter never owns connection lifetime.
        pool = await asyncpg.create_pool(dsn)
        assert pool is not None
        try:
            store = PostgresCheckpointStore(pool)
            await store.ensure_schema()
            await store.delete(_RUN_ID)  # clean slate (idempotent)

            # Empty-state baseline
            assert await store.latest(_RUN_ID) is None
            assert await store.at_version(_RUN_ID, 1) is None
            assert await store.list_versions(_RUN_ID) == []

            cp1 = Checkpoint(
                run_id=_RUN_ID,
                version=1,
                state={"phase": "planning", "items": [1, 2, 3]},
                created_at=1_700_000_000.5,
                status=CheckpointStatus.RUNNING,
                metadata={"actor": "planner"},
            )
            cp2 = Checkpoint(
                run_id=_RUN_ID,
                version=2,
                state={"phase": "review", "items": [1, 2, 3, 4]},
                created_at=1_700_000_100.25,
                status=CheckpointStatus.SUSPENDED,
                metadata={"actor": "planner", "gate": "evidence-review"},
            )
            cp3 = Checkpoint(
                run_id=_RUN_ID,
                version=3,
                state={"phase": "complete", "summary": "ok"},
                created_at=1_700_000_200.0,
                status=CheckpointStatus.DONE,
            )

            await store.save(cp1)
            await store.save(cp2)
            await store.save(cp3)

            # latest = highest version
            latest = await store.latest(_RUN_ID)
            assert latest is not None
            assert latest.version == 3
            assert latest.status is CheckpointStatus.DONE
            assert latest.state == {"phase": "complete", "summary": "ok"}
            assert latest.metadata == {}  # default field preserved through JSONB round-trip

            # at_version = time travel
            v2 = await store.at_version(_RUN_ID, 2)
            assert v2 is not None
            assert v2.version == 2
            assert v2.status is CheckpointStatus.SUSPENDED
            assert v2.state == {"phase": "review", "items": [1, 2, 3, 4]}
            assert v2.metadata == {"actor": "planner", "gate": "evidence-review"}
            assert v2.created_at == pytest.approx(1_700_000_100.25)

            assert await store.at_version(_RUN_ID, 999) is None

            # list_versions = ascending
            assert await store.list_versions(_RUN_ID) == [1, 2, 3]

            # save is idempotent on (run_id, version) — replaying cp2 overwrites in place
            cp2_overwrite = Checkpoint(
                run_id=_RUN_ID,
                version=2,
                state={"phase": "review", "items": [1, 2, 3, 4], "note": "retried"},
                created_at=1_700_000_150.0,
                status=CheckpointStatus.SUSPENDED,
                metadata={"actor": "planner", "gate": "evidence-review", "retry": True},
            )
            await store.save(cp2_overwrite)
            assert await store.list_versions(_RUN_ID) == [1, 2, 3]
            v2b = await store.at_version(_RUN_ID, 2)
            assert v2b is not None
            assert v2b.state["note"] == "retried"
            assert v2b.metadata["retry"] is True

            # delete = remove all rows for the run
            await store.delete(_RUN_ID)
            assert await store.latest(_RUN_ID) is None
            assert await store.list_versions(_RUN_ID) == []
            # second delete is a no-op
            await store.delete(_RUN_ID)
        finally:
            await pool.close()

    _run(go())
