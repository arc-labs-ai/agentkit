"""`PostgresCheckpointStore` — durable `CheckpointPort` over Postgres (extra:
`arc-agentkit[postgres]`, asyncpg).

One `agentkit_checkpoints` table keyed by `(run_id, version)`. `state` and `metadata` are JSONB
(opaque to the port — the producer's snapshot shape is its own concern). An index on
`(run_id, version DESC)` makes `latest()` and `list_versions()` cheap.

The pool is *application-owned*: it's opened in the FastAPI lifespan (or equivalent) and passed in.
The adapter never creates or closes connections — that's the calling layer's job. Apps call
`ensure_schema()` once at startup before the first save/load; this matches `PostgresStore.init()`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentkit.kernel.errors import CheckpointerError
from agentkit.kernel.ports import Checkpoint, CheckpointStatus

if TYPE_CHECKING:  # asyncpg is the [postgres] extra; never imported at module load
    import asyncpg  # type: ignore[import-not-found]  # asyncpg lacks a py.typed marker


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS agentkit_checkpoints ("
    "run_id      TEXT NOT NULL,"
    "version     INTEGER NOT NULL,"
    "state       JSONB NOT NULL,"
    "created_at  DOUBLE PRECISION NOT NULL,"
    "status      TEXT NOT NULL,"
    "metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,"
    "PRIMARY KEY (run_id, version)"
    ")",
    "CREATE INDEX IF NOT EXISTS agentkit_checkpoints_run_id_version "
    "ON agentkit_checkpoints (run_id, version DESC)",
)


class PostgresCheckpointStore:
    """A `CheckpointPort` over Postgres via asyncpg. Pool is application-owned (matches the
    explicit-lifecycle convention `PostgresStore` follows — see `store/postgres.py`). Call
    `await ensure_schema()` once at startup; the per-call methods then go through `pool.acquire()`
    like every other asyncpg adapter in this package.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        """Idempotent DDL — `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS`. Safe to
        call multiple times. Apps call this once at startup before any save/load (mirrors
        `PostgresStore.init()`)."""
        try:
            async with self._pool.acquire() as con:
                for ddl in _SCHEMA:
                    await con.execute(ddl)
        except CheckpointerError:
            raise
        except (
            Exception
        ) as exc:  # asyncpg.*, OSError, etc. — wrap so callers see the framework taxonomy
            raise CheckpointerError(f"checkpoint schema init failed: {exc}") from exc

    async def save(self, cp: Checkpoint) -> None:
        # ON CONFLICT keeps `save` idempotent on retry — the version is the producer's monotonic
        # numbering, so a retried save for the same `(run_id, version)` overwrites in place.
        try:
            async with self._pool.acquire() as con:
                await con.execute(
                    "INSERT INTO agentkit_checkpoints "
                    "(run_id, version, state, created_at, status, metadata) "
                    "VALUES ($1, $2, $3::jsonb, $4, $5, $6::jsonb) "
                    "ON CONFLICT (run_id, version) DO UPDATE SET "
                    "state = EXCLUDED.state, "
                    "created_at = EXCLUDED.created_at, "
                    "status = EXCLUDED.status, "
                    "metadata = EXCLUDED.metadata",
                    cp.run_id,
                    cp.version,
                    json.dumps(cp.state),
                    cp.created_at,
                    str(cp.status),
                    json.dumps(cp.metadata),
                )
        except CheckpointerError:
            raise
        except Exception as exc:
            raise CheckpointerError(
                f"failed to save checkpoint run_id={cp.run_id!r} version={cp.version}: {exc}"
            ) from exc

    async def latest(self, run_id: str) -> Checkpoint | None:
        try:
            async with self._pool.acquire() as con:
                row = await con.fetchrow(
                    "SELECT run_id, version, state, created_at, status, metadata "
                    "FROM agentkit_checkpoints WHERE run_id = $1 "
                    "ORDER BY version DESC LIMIT 1",
                    run_id,
                )
            return None if row is None else _row_to_checkpoint(row)
        except CheckpointerError:
            raise  # already-typed (from _row_to_checkpoint's JSON wrap) — don't double-wrap
        except Exception as exc:
            raise CheckpointerError(
                f"failed to load latest checkpoint for run_id={run_id!r}: {exc}"
            ) from exc

    async def at_version(self, run_id: str, version: int) -> Checkpoint | None:
        try:
            async with self._pool.acquire() as con:
                row = await con.fetchrow(
                    "SELECT run_id, version, state, created_at, status, metadata "
                    "FROM agentkit_checkpoints WHERE run_id = $1 AND version = $2",
                    run_id,
                    version,
                )
            return None if row is None else _row_to_checkpoint(row)
        except CheckpointerError:
            raise
        except Exception as exc:
            raise CheckpointerError(
                f"failed to load checkpoint run_id={run_id!r} version={version}: {exc}"
            ) from exc

    async def list_versions(self, run_id: str) -> list[int]:
        try:
            async with self._pool.acquire() as con:
                rows = await con.fetch(
                    "SELECT version FROM agentkit_checkpoints WHERE run_id = $1 ORDER BY version",
                    run_id,
                )
            return [r["version"] for r in rows]
        except CheckpointerError:
            raise
        except Exception as exc:
            raise CheckpointerError(
                f"failed to list checkpoint versions for run_id={run_id!r}: {exc}"
            ) from exc

    async def delete(self, run_id: str) -> None:
        # Idempotent — matches `PostgresStore.delete` (missing rows are a no-op).
        try:
            async with self._pool.acquire() as con:
                await con.execute("DELETE FROM agentkit_checkpoints WHERE run_id = $1", run_id)
        except CheckpointerError:
            raise
        except Exception as exc:
            raise CheckpointerError(
                f"failed to delete checkpoints for run_id={run_id!r}: {exc}"
            ) from exc


def _row_to_checkpoint(row: Any) -> Checkpoint:
    """Decode an asyncpg row into a `Checkpoint`. JSONB columns come back as `str` from asyncpg
    by default (no codec registered on the pool), so we `json.loads` them; if a future codec
    returns them pre-decoded, we pass through. `status` round-trips through `CheckpointStatus`.

    A malformed JSON payload surfaces as ``CheckpointerError`` with the original
    ``json.JSONDecodeError`` chained on ``__cause__`` — callers get a stable framework type
    instead of a stdlib decode error escaping the port."""
    try:
        state = row["state"]
        if isinstance(state, str):
            state = json.loads(state)
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise CheckpointerError(
            f"malformed checkpoint JSON (run_id={row['run_id']!r}, version={row['version']!r}): {exc}"
        ) from exc
    return Checkpoint(
        run_id=row["run_id"],
        version=row["version"],
        state=state,
        created_at=row["created_at"],
        status=CheckpointStatus(row["status"]),
        metadata=metadata,
    )


__all__ = ["PostgresCheckpointStore"]
