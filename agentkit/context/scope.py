"""Typed slice predicates over a message list.

A ``ContextScope`` is a tiny predicate ``(message, index) -> bool``
that ``WorkingContext.slice(scope)`` applies to materialize a new
view. The combinators (``AllOf`` / ``AnyOf`` / ``Not``) compose them
without forcing the caller to write closures.

Scopes are frozen dataclasses so they're hashable, pickleable, and
trivially equal — which matters because they're sometimes used as
cache keys for re-grounding decisions.

Two kinds of scope live here, and the difference is the whole design:

* **pointwise** scopes (``RoleFilter``, ``Tagged``, ``Since``) decide
  membership from ``(message, index)`` alone. ``matches()`` is the
  complete truth about them.
* **positional** scopes (``LastNTurns``) cannot. "The last 2 turns" is
  a property of the LIST, not of any message in it — you have to see
  the tail to know where the window starts. They implement
  ``_surviving_indices(messages) -> set[int]`` instead, and
  ``WorkingContext.slice()`` prefers that hook when it's there.

Composition is done in INDEX SPACE, not in predicate space, precisely
so the two kinds mix. ``surviving_indices()`` below is the single
converter: it calls the hook when a scope has one and derives the set
from ``matches()`` when it doesn't, after which ``AllOf`` is ``&``,
``AnyOf`` is ``|``, and ``Not`` is a complement against
``range(len(messages))``.

Note: system messages are special-cased by ``LastNTurns`` — they
always survive a turn-window slice. The other scopes are role-blind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentkit.kernel.errors import AgentkitError
from agentkit.kernel.types import Message


class PositionalScopeError(AgentkitError):
    """Raised when a positional scope is asked to judge one message.

    A positional scope (``LastNTurns``) has no answer to "does message
    #3 belong?" without the rest of the list. Rather than guess, it
    says so. See ``LastNTurns.matches`` for why guessing was worse.
    """


@runtime_checkable
class ContextScope(Protocol):
    """Predicate over messages — returns True when the message
    belongs in the sliced view. Composable via the obvious
    ``AllOf`` / ``AnyOf`` / ``Not`` combinators.

    ``index`` is the message's index in the *original* message list
    (not the sliced output). This is what makes ``Since`` cheap.

    A scope that genuinely cannot answer from ``(message, index)``
    alone — one whose answer depends on where the END of the list is —
    should implement ``_surviving_indices`` as well; see
    ``PositionalScope``.
    """

    def matches(self, message: Message, index: int) -> bool: ...


@runtime_checkable
class PositionalScope(Protocol):
    """A scope that needs the whole list to decide.

    The hook is spelled with a leading underscore because it predates
    this protocol and ``WorkingContext.slice()`` already probes for
    that exact name; renaming it would be a silent break for anyone
    who wrote a scope against the old contract. It is a real extension
    point, so it is documented and exported rather than pretended away.

    Implement BOTH members: ``_surviving_indices`` for the truth, and
    ``matches`` to state — by raising ``PositionalScopeError`` — that
    the pointwise question has no answer.
    """

    def matches(self, message: Message, index: int) -> bool: ...

    def _surviving_indices(self, messages: list[Message]) -> set[int]: ...


def surviving_indices(scope: ContextScope, messages: list[Message]) -> set[int]:
    """The set of indices ``scope`` keeps out of ``messages``.

    The one place the two scope kinds are reconciled. Positional
    scopes answer directly; pointwise scopes get their ``matches()``
    run once per message and the result collected. Everything above —
    every combinator — works on the sets this returns, which is why a
    mixed ``AllOf((LastNTurns(2), RoleFilter(...)))`` needs no special
    case: the window contributes ``{4, 5}``, the role filter
    contributes whatever ``matches()`` says, and ``&`` does the rest.
    """
    if isinstance(scope, PositionalScope):
        return scope._surviving_indices(messages)
    return {i for i, m in enumerate(messages) if scope.matches(m, i)}


@dataclass(frozen=True)
class LastNTurns:
    """Keeps the last ``n`` turns (a turn = consecutive user+assistant
    pair). System messages always survive — they're the cache-stable
    framing the loop relies on.

    Implementation note: this scope can't decide membership purely
    from ``(message, index)`` — it needs the full list to count back
    ``n`` turns. ``WorkingContext.slice()`` pre-computes the set of
    surviving indices for any scope that exposes ``_surviving_indices``,
    falling back to per-message ``matches()`` otherwise.
    """

    n: int

    def matches(self, message: Message, index: int) -> bool:
        # This used to `return True` — "safe by default, so a misuse
        # doesn't silently drop messages". It had the failure mode
        # exactly backwards. Nothing calls `.matches()` on a bare
        # LastNTurns; what called it was `AllOf.matches`, once per
        # message, and `True` there means "this child has no opinion",
        # i.e. the window evaporates. Measured on a 6-message
        # transcript: `slice(LastNTurns(2))` -> 2 messages, while
        # `slice(AllOf((LastNTurns(2), RoleFilter(("user",)))))` -> 6.
        # No error, no warning — just a prompt three times the size it
        # was asked to be, and a context overflow some turns later.
        #
        # There is no honest return value here. `True` over-keeps,
        # `False` empties the window, and both are indistinguishable
        # from a correct answer at the call site. So it refuses. The
        # combinators no longer route through this path (they compose
        # via `_surviving_indices`), which means reaching this line at
        # all now signals real code that is about to be wrong.
        raise PositionalScopeError(
            f"LastNTurns({self.n}).matches() cannot answer for a single message — "
            "the window is a property of the whole list. Use "
            "surviving_indices(scope, messages), or WorkingContext.slice(scope), "
            "which does it for you."
        )

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
    """Keeps messages with index ≥ ``checkpoint_index``.

    Index-dependent but NOT positional: the checkpoint is an absolute
    coordinate the caller already knows, so ``matches()`` is the whole
    truth and this scope composes correctly through ``matches()``
    alone. Only a scope that measures from the END of the list needs
    the ``_surviving_indices`` hook.
    """

    checkpoint_index: int

    def matches(self, message: Message, index: int) -> bool:
        return index >= self.checkpoint_index


# ── combinators ────────────────────────────────────────────────────
#
# Every combinator implements `_surviving_indices` unconditionally,
# even when all of its children are pointwise. Two reasons:
#
#   1. Correctness has to be structural, not conditional. A combinator
#      that only grew the hook "when it detects a positional child"
#      would be one nesting level away from losing it again — which is
#      the bug this file just had.
#   2. It costs nothing. For pointwise children the set algebra is
#      provably the same answer as the boolean algebra: `&` over
#      per-message match sets IS `all()`, `|` IS `any()`, and the
#      complement IS `not`. Verified against the pre-existing
#      combinator tests, which pass unchanged.
#
# `matches()` is kept on the combinators for the pointwise case, where
# it remains exact.


@dataclass(frozen=True)
class AllOf:
    """Conjunction — keep messages every scope keeps."""

    scopes: tuple[ContextScope, ...]

    def matches(self, message: Message, index: int) -> bool:
        # Deliberately a materialized list, not a generator: `all()`
        # short-circuits on the first False, so with a generator
        # `AllOf((RoleFilter(...), LastNTurns(2)))` would raise or not
        # depending on whether the role filter happened to reject this
        # particular message first. An error that fires on message #3
        # but not message #2 is worse than one that always fires.
        return all([s.matches(message, index) for s in self.scopes])  # noqa: C419

    def _surviving_indices(self, messages: list[Message]) -> set[int]:
        # Intersection, seeded with everything so that `AllOf(())` is
        # vacuously true — the same answer `all([])` gives.
        keep = set(range(len(messages)))
        for s in self.scopes:
            keep &= surviving_indices(s, messages)
        return keep


@dataclass(frozen=True)
class AnyOf:
    """Disjunction — keep messages any scope keeps."""

    scopes: tuple[ContextScope, ...]

    def matches(self, message: Message, index: int) -> bool:
        # Materialized for the same reason as AllOf.matches: `any()`
        # short-circuits on the first True.
        return any([s.matches(message, index) for s in self.scopes])  # noqa: C419

    def _surviving_indices(self, messages: list[Message]) -> set[int]:
        # Union, seeded empty so that `AnyOf(())` is vacuously false —
        # the same answer `any([])` gives.
        keep: set[int] = set()
        for s in self.scopes:
            keep |= surviving_indices(s, messages)
        return keep


@dataclass(frozen=True)
class Not:
    """Inversion — keep messages the inner scope drops."""

    scope: ContextScope

    def matches(self, message: Message, index: int) -> bool:
        return not self.scope.matches(message, index)

    def _surviving_indices(self, messages: list[Message]) -> set[int]:
        # Complement against the FULL index range, which is the only
        # definition that keeps `Not` an involution:
        # `Not(Not(s)) == s` for every s, positional or not.
        #
        # Worth stating what this means for `Not(LastNTurns(2))`,
        # because it is easy to talk yourself into the opposite:
        # LastNTurns keeps the window AND every system message, so the
        # complement is "the older messages, system prompt excluded".
        # On a 6-user-message transcript that is 4 messages (indices
        # 0-3), not 6 and not 0. Pre-fix it returned 0 — `not True`
        # for every message — which is the same silent-window bug
        # wearing its opposite sign: the "drop the recent tail" scope
        # dropped the entire transcript.
        return set(range(len(messages))) - surviving_indices(self.scope, messages)


__all__ = [
    "AllOf",
    "AnyOf",
    "ContextScope",
    "LastNTurns",
    "Not",
    "PositionalScope",
    "PositionalScopeError",
    "RoleFilter",
    "Since",
    "Tagged",
    "surviving_indices",
]
