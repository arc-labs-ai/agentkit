"""Host-facing event ports; no broker dependency in the runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agentkit.models.contracts.types import CallEvent


class EventPublisher(Protocol):
    async def publish(self, event: CallEvent) -> None:
        """Acknowledge publication, or raise when the delivery contract fails."""
        ...


class EventJournal(Protocol):
    async def append(self, event: CallEvent) -> None:
        """Persist before delivery, idempotently keyed by event.event_id."""
        ...

    async def read(self, *, call_id: str, after_sequence: int = 0) -> Sequence[CallEvent]: ...
