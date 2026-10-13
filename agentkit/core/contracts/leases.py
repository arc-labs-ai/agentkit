"""Host-owned distributed admission; callers must fence protected side effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Lease:
    resource: str
    owner: str
    lease_id: str
    fencing_token: int
    expires_at: float


class LeaseStore(Protocol):
    async def acquire(self, resource: str, owner: str, *, ttl_s: float, capacity: int = 1) -> Lease | None:
        """Atomically admit a holder, or return None while all slots are held."""
        ...

    async def renew(self, lease: Lease, *, ttl_s: float) -> Lease | None:
        """Renew an unexpired holder; expired ownership cannot be resurrected."""
        ...

    async def release(self, lease: Lease) -> bool:
        """Release only the matching holder, idempotently."""
        ...


__all__ = ["Lease", "LeaseStore"]
