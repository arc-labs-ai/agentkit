"""Task instructions and host-defined completion verification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentkit.models.contracts.types import Answer


@dataclass(frozen=True)
class InstructionSet:
    goal: str
    definition_of_done: str = "Return the requested answer."
    rules: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    escalation: str = "Return progress and the reason you cannot finish."
    orientation: str = ""
    verifier: Callable[[Answer[Any]], Any] | None = field(default=None, repr=False, compare=False)
    no_progress_slices: int = 2

    def __post_init__(self) -> None:
        if not self.goal or len(self.goal) > 50_000 or self.no_progress_slices < 1:
            raise ValueError("invalid instruction set")
