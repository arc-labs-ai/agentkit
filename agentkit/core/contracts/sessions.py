"""Host-owned native transcript bundle storage, independent of a runtime process."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SessionBundleRef:
    """Opaque immutable reference; it contains no transcript or credential bytes."""

    key: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or re.fullmatch(r"[0-9a-f]{64}", self.key) is None:
            raise ValueError("session bundle reference must be a lowercase SHA-256 digest")


class SessionBundleStore(Protocol):
    async def save(self, home: Path, *, previous_ref: SessionBundleRef | None = None) -> SessionBundleRef:
        """Store only native transcript files, incrementally where possible."""
        ...

    async def restore(self, reference: SessionBundleRef, home: Path) -> None:
        """Validate and restore transcript files beneath the caller's isolated home."""
        ...


__all__ = ["SessionBundleRef", "SessionBundleStore"]
