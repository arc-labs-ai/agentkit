"""`InMemoryCheckpointStore` — a `CheckpointPort` backed by a dict for tests and local dev.

NOT durable across process restarts; production wires a Postgres or Redis adapter (follow-up).
Versions per `run_id` are kept in insertion order; since `Checkpointer.snapshot` always picks
`max(versions) + 1`, that order coincides with ascending version order.
"""

from __future__ import annotations

from agentkit.kernel.ports import Checkpoint


class InMemoryCheckpointStore:
    """A `CheckpointPort` over an in-process dict. One list of `Checkpoint`s per `run_id`."""

    def __init__(self) -> None:
        self._cps: dict[str, list[Checkpoint]] = {}

    async def save(self, cp: Checkpoint) -> None:
        self._cps.setdefault(cp.run_id, []).append(cp)

    async def latest(self, run_id: str) -> Checkpoint | None:
        cps = self._cps.get(run_id)
        if not cps:
            return None
        # Highest version wins (the producer's monotonic numbering owns ordering, not
        # insertion order — a defensive lookup so a producer that re-saved an old version
        # by mistake doesn't suddenly become "latest").
        return max(cps, key=lambda c: c.version)

    async def at_version(self, run_id: str, version: int) -> Checkpoint | None:
        for cp in self._cps.get(run_id, ()):
            if cp.version == version:
                return cp
        return None

    async def list_versions(self, run_id: str) -> list[int]:
        return sorted(cp.version for cp in self._cps.get(run_id, ()))

    async def delete(self, run_id: str) -> None:
        self._cps.pop(run_id, None)


__all__ = ["InMemoryCheckpointStore"]
