"""Local durable event journal; replay delivery and subscriber cursors are host-owned."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from agentkit.core.serialization._immutable import thaw
from agentkit.models.contracts.types import CallEvent


class SQLiteEventJournal:
    """Persist before acknowledgement on one SQLite database shared by local processes.

    WAL plus FULL synchronization supplies SQLite's local-filesystem durability.
    This is not a multi-host database or exactly-once subscriber delivery. Hosts
    replay by cursor and make effects idempotent using event IDs. After restart,
    publishers resume explicit sequences or use a fresh call ID.
    """

    def __init__(self, path: str | Path, *, busy_timeout_s: float = 5) -> None:
        if not math.isfinite(busy_timeout_s) or busy_timeout_s <= 0:
            raise ValueError("SQLite busy timeout must be positive")
        self.path, self.busy_timeout_s = str(path), busy_timeout_s
        if self.path == ":memory:":
            raise ValueError("durable journal requires a filesystem database")
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS agent_events (call_id TEXT NOT NULL, sequence INTEGER NOT NULL, "
                "payload TEXT NOT NULL, PRIMARY KEY (call_id, sequence))"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=self.busy_timeout_s)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    async def append(self, event: CallEvent) -> None:
        if (
            not event.call_id
            or not isinstance(event.sequence, int)
            or isinstance(event.sequence, bool)
            or event.sequence <= 0
        ):
            raise ValueError("journal events require a call ID and positive explicit sequence")
        payload = json.dumps(
            {
                "kind": str(event.kind),
                "call_id": event.call_id,
                "fields": thaw(event.fields),
                "sequence": event.sequence,
                "timestamp": event.timestamp,
                "parent_call_id": event.parent_call_id,
                "schema_version": event.schema_version,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        await asyncio.to_thread(self._append, event, payload)

    def _append(self, event: CallEvent, payload: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT payload FROM agent_events WHERE call_id=? AND sequence=?", (event.call_id, event.sequence)
            ).fetchone()
            if prior:
                if prior[0] != payload:
                    raise ValueError("event ID already exists with different content")
                return
            latest = db.execute("SELECT MAX(sequence) FROM agent_events WHERE call_id=?", (event.call_id,)).fetchone()[
                0
            ]
            if latest is not None and event.sequence <= latest:
                raise ValueError("journal sequence must increase within a call")
            db.execute("INSERT INTO agent_events VALUES (?, ?, ?)", (event.call_id, event.sequence, payload))

    async def read(self, *, call_id: str, after_sequence: int = 0) -> tuple[CallEvent, ...]:
        if after_sequence < 0:
            raise ValueError("journal cursor cannot be negative")
        return await asyncio.to_thread(self._read, call_id, after_sequence)

    def _read(self, call_id: str, after_sequence: int) -> tuple[CallEvent, ...]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM agent_events WHERE call_id=? AND sequence>? ORDER BY sequence",
                (call_id, after_sequence),
            ).fetchall()
        return tuple(CallEvent(**json.loads(row[0])) for row in rows)


__all__ = ["SQLiteEventJournal"]
