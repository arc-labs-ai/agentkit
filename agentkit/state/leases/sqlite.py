"""Cross-process admission on a shared local SQLite database."""

from __future__ import annotations

import asyncio
import math
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from agentkit.core.contracts.leases import Lease


class SQLiteLeaseStore:
    """Finite leases with monotonically increasing per-resource fencing tokens.

    Use on a local filesystem, not a network share. A crashed process retains
    its slot until expiry. Hosts renew long-running work and enforce fencing
    tokens in protected writes; a lease alone cannot stop an expired worker.
    AgentRunner's local concurrency is unchanged; harnesses explicitly bind
    this host admission port around work. Clock synchronization is host-owned.
    """

    def __init__(self, path: str | Path, *, busy_timeout_s: float = 5) -> None:
        if not math.isfinite(busy_timeout_s) or busy_timeout_s <= 0:
            raise ValueError("SQLite busy timeout must be positive")
        self.path, self.busy_timeout_s = str(path), busy_timeout_s
        if self.path == ":memory:":
            raise ValueError("cross-process leases require a filesystem database")
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS lease_resources (resource TEXT PRIMARY KEY, "
                "capacity INTEGER NOT NULL, next_token INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS admission_leases (resource TEXT NOT NULL, owner TEXT NOT NULL, "
                "lease_id TEXT PRIMARY KEY, fencing_token INTEGER NOT NULL, expires_at REAL NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS lease_resource ON admission_leases(resource)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=self.busy_timeout_s)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _ttl(ttl_s: float) -> None:
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("lease TTL must be finite and positive")

    async def acquire(self, resource: str, owner: str, *, ttl_s: float, capacity: int = 1) -> Lease | None:
        self._ttl(ttl_s)
        if not resource or not owner or not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("lease resource/owner must be nonempty and capacity a positive integer")
        return await asyncio.to_thread(self._acquire, resource, owner, ttl_s, capacity)

    def _acquire(self, resource: str, owner: str, ttl_s: float, capacity: int) -> Lease | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            db.execute("INSERT OR IGNORE INTO lease_resources VALUES (?, ?, 0)", (resource, capacity))
            stored_capacity, token = db.execute(
                "SELECT capacity,next_token FROM lease_resources WHERE resource=?", (resource,)
            ).fetchone()
            if stored_capacity != capacity:
                raise ValueError("resource capacity differs from its established admission policy")
            db.execute("DELETE FROM admission_leases WHERE resource=? AND expires_at<=?", (resource, now))
            held = db.execute("SELECT COUNT(*) FROM admission_leases WHERE resource=?", (resource,)).fetchone()[0]
            if held >= capacity:
                return None
            lease = Lease(resource, owner, uuid4().hex, token + 1, now + ttl_s)
            db.execute("UPDATE lease_resources SET next_token=? WHERE resource=?", (lease.fencing_token, resource))
            db.execute(
                "INSERT INTO admission_leases VALUES (?, ?, ?, ?, ?)",
                (lease.resource, lease.owner, lease.lease_id, lease.fencing_token, lease.expires_at),
            )
            return lease

    async def renew(self, lease: Lease, *, ttl_s: float) -> Lease | None:
        self._ttl(ttl_s)
        return await asyncio.to_thread(self._renew, lease, ttl_s)

    def _renew(self, lease: Lease, ttl_s: float) -> Lease | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            expires_at = now + ttl_s
            changed = db.execute(
                "UPDATE admission_leases SET expires_at=? WHERE resource=? AND owner=? AND lease_id=? "
                "AND fencing_token=? AND expires_at>?",
                (expires_at, lease.resource, lease.owner, lease.lease_id, lease.fencing_token, now),
            ).rowcount
            return (
                Lease(lease.resource, lease.owner, lease.lease_id, lease.fencing_token, expires_at) if changed else None
            )

    async def release(self, lease: Lease) -> bool:
        return await asyncio.to_thread(self._release, lease)

    def _release(self, lease: Lease) -> bool:
        with self._connect() as db:
            changed = db.execute(
                "DELETE FROM admission_leases WHERE resource=? AND owner=? AND lease_id=? AND fencing_token=?",
                (lease.resource, lease.owner, lease.lease_id, lease.fencing_token),
            ).rowcount
            return changed == 1


__all__ = ["SQLiteLeaseStore"]
