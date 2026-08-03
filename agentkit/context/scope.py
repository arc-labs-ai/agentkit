"""Typed slice predicates over a message list.

A ``ContextScope`` is a tiny predicate ``(message, index) -> bool``
that ``WorkingContext.slice(scope)`` applies to materialize a new
view. The combinators (``AllOf`` / ``AnyOf`` / ``Not``) compose them
without forcing the caller to write closures.

Scopes are frozen dataclasses so they're hashable, pickleable, and
trivially equal — which matters because they're sometimes used as
cache keys for re-grounding decisions.

Note: system messages are special-cased by ``LastNTurns`` — they
always survive a turn-window slice. The other scopes are role-blind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentkit.kernel.types import Message


@runtime_checkable
class ContextScope(Protocol):
    """Predicate over messages — returns True when the message
    belongs in the sliced view. Composable via the obvious
    ``AllOf`` / ``AnyOf`` / ``Not`` combinators.

    ``index`` is the message's index in the *original* message list
    (not the sliced output). This is what makes ``Since`` cheap and
    ``LastNTurns`` implementable.
    """

    def matches(self, message: Message, index: int) -> bool: ...


@dataclass(frozen=True)
class LastNTurns:
    """Keeps the last ``n`` turns (a turn = consecutive user+assistant
    pair). System messages always survive — they're the cache-stable
    framing the loop relies on.

    Implementation note: this scope can't decide membership purely
    from ``(message, index)`` — it needs the full list to count back
    ``n`` turns. ``WorkingContext.slice()`` pre-computes the set of
    surviving indices for any scope that exposes ``_surviving_indices``,
    falling back to per-message ``matches()`` otherwise. The protocol
    surface stays simple; the windowing path is local to this class.
    """

    n: int

    def matches(self, message: Message, index: int) -> bool:
        # Without the full list we can't tell — fall back to True so
        # that a misuse (calling `.matches` directly) doesn't silently
        # drop messages. `WorkingContext.slice()` uses the more
        # specific `_surviving_indices` path for accuracy.
        return True

    def _surviving_indices(self, messages: list[Message]) -> set[int]:
        if self.n <= 0:
            # Keep only system messages — zero turns of history.
            return {i for i, m in enumerate(messages) if m.role == "system"}

        survivors: set[int] = set()
        # System messages always survive.
        for i, m in enumerate(messages):
            if m.role == "system":
                survivors.add(i)

        # Walk backwards collecting full turns. A "turn" here is a
        # contiguous run of non-system messages that contains at least
        # one user OR one assistant message — we count by the
        # transition into a user message walking forward, or
        # equivalently by counting assistant tails walking backward.
        # Simpler: count consecutive (user, assistant) flips.
        turns_collected = 0
        i = len(messages) - 1
        while i >= 0 and turns_collected < self.n:
            m = messages[i]
            if m.role == "system":
                i -= 1
                continue
            survivors.add(i)
            # A turn boundary is a transition from user -> earlier
            # (assistant/tool). When we hit a user message we count
            # one turn complete.
            if m.role == "user":
                turns_collected += 1
            i -= 1
        return survivors


@dataclass(frozen=True)
class RoleFilter:
    """Keeps only messages with role in ``roles``."""

    roles: frozenset[str]

    def matches(self, message: Message, index: int) -> bool:
        return message.role in self.roles


@dataclass(frozen=True)
class Tagged:
    """Keeps messages whose ``name`` field matches the tag. Useful
    for per-agent slicing in a team."""

    tag: str

    def matches(self, message: Message, index: int) -> bool:
        return message.name == self.tag


@dataclass(frozen=True)
class Since:
    """Keeps messages with index ≥ ``checkpoint_index``."""

    checkpoint_index: int

    def matches(self, message: Message, index: int) -> bool:
        return index >= self.checkpoint_index


# ── combinators ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AllOf:
    """Conjunction — keep messages every scope keeps."""

    scopes: tuple[ContextScope, ...]

    def matches(self, message: Message, index: int) -> bool:
        return all(s.matches(message, index) for s in self.scopes)


@dataclass(frozen=True)
class AnyOf:
    """Disjunction — keep messages any scope keeps."""

    scopes: tuple[ContextScope, ...]

    def matches(self, message: Message, index: int) -> bool:
        return any(s.matches(message, index) for s in self.scopes)


@dataclass(frozen=True)
class Not:
    """Inversion — keep messages the inner scope drops."""

    scope: ContextScope

    def matches(self, message: Message, index: int) -> bool:
        return not self.scope.matches(message, index)


__all__ = [
    "AllOf",
    "AnyOf",
    "ContextScope",
    "LastNTurns",
    "Not",
    "RoleFilter",
    "Since",
    "Tagged",
]
