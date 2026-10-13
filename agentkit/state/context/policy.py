"""Composable context transformation seam used between model slices."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from agentkit.state.context.context import WorkingContext
from agentkit.state.context.scope import LastNTurns

type ContextPolicyResult = WorkingContext | Awaitable[WorkingContext]


class ContextPolicy(Protocol):
    """Deterministically transform a run's working context between slices.

    Policies should preserve source attribution and protected tool evidence.
    Summarizing policies are explicit host opt-ins; the default policy is absent.
    """

    def __call__(self, context: WorkingContext) -> ContextPolicyResult: ...


@dataclass(frozen=True)
class SlidingWindowContextPolicy:
    """Keep the latest complete user/assistant turns without generating a summary."""

    max_turns: int

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("context window must retain at least one turn")

    def __call__(self, context: WorkingContext) -> WorkingContext:
        return context.slice(LastNTurns(self.max_turns))
