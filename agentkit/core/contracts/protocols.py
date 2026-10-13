"""Structural context accepted by tools and provider transports.

The context is supplied by CallRuntime and carries only execution facts; it
is deliberately independent of any agent or middleware runtime.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

AutonomyLiteral = Literal["auto", "gated", "manual"]


class Ctx(Protocol):
    scope: Any
    correlation_id: str
    cancel: Any

    def check_cancelled(self) -> None: ...


__all__ = ["Ctx", "AutonomyLiteral"]
