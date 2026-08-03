"""Scope predicates and combinators — tested in isolation.

The Scope module is a tiny library of frozen-dataclass predicates +
``AllOf`` / ``AnyOf`` / ``Not`` combinators. These tests pin each
predicate's behaviour without going through ``WorkingContext.slice``
so a regression points at the predicate itself, not at the call site.
"""

from __future__ import annotations

from agentkit.context.scope import (
    AllOf,
    AnyOf,
    LastNTurns,
    Not,
    RoleFilter,
    Since,
    Tagged,
)
from agentkit.kernel.types import Message


def _user(text: str) -> Message:
    return Message("user", text)


def _asst(text: str, *, name: str | None = None) -> Message:
    return Message("assistant", text, name=name)


def _system(text: str) -> Message:
    return Message("system", text)


# ── individual predicates ─────────────────────────────────────────


def test_role_filter_matches_only_listed_roles():
    rf = RoleFilter(frozenset({"user", "tool"}))
    assert rf.matches(_user("u"), 0) is True
    assert rf.matches(_asst("a"), 1) is False
    assert rf.matches(Message("tool", "t"), 2) is True


def test_tagged_matches_messages_by_name_field():
    t = Tagged("planner")
    assert t.matches(_asst("hello", name="planner"), 0) is True
    assert t.matches(_asst("hello", name="researcher"), 1) is False
    # A message with no name doesn't match a non-empty tag.
    assert t.matches(_asst("hello"), 2) is False


def test_since_matches_indices_at_or_after_checkpoint():
    s = Since(5)
    assert s.matches(_user("x"), 4) is False
    assert s.matches(_user("x"), 5) is True
    assert s.matches(_user("x"), 6) is True


def test_last_n_turns_uses_surviving_indices_path():
    """``LastNTurns`` is the only built-in that needs the full list to
    decide — exercised via its ``_surviving_indices`` accessor."""

    messages = [
        _system("seed"),  # 0 — system always survives
        _user("q1"),  # 1 — turn 1 (oldest)
        _asst("a1"),  # 2
        _user("q2"),  # 3 — turn 2
        _asst("a2"),  # 4
        _user("q3"),  # 5 — turn 3 (most recent)
        _asst("a3"),  # 6
    ]
    keep = LastNTurns(2)._surviving_indices(messages)
    # System (0) + last two user/asst pairs (3, 4, 5, 6).
    assert keep == {0, 3, 4, 5, 6}


def test_last_n_turns_zero_keeps_only_system():
    messages = [_system("seed"), _user("q"), _asst("a")]
    keep = LastNTurns(0)._surviving_indices(messages)
    assert keep == {0}


# ── combinators ───────────────────────────────────────────────────


def test_allof_requires_every_child_to_match():
    scope = AllOf((RoleFilter(frozenset({"user"})), Since(5)))
    assert scope.matches(_user("x"), 5) is True
    assert scope.matches(_user("x"), 4) is False  # since fails
    assert scope.matches(_asst("x"), 5) is False  # role fails


def test_anyof_requires_at_least_one_child_to_match():
    scope = AnyOf((RoleFilter(frozenset({"user"})), Tagged("planner")))
    assert scope.matches(_user("x"), 0) is True
    assert scope.matches(_asst("x", name="planner"), 1) is True
    assert scope.matches(_asst("x", name="other"), 2) is False


def test_not_inverts_inner_scope():
    scope = Not(RoleFilter(frozenset({"system"})))
    assert scope.matches(_system("x"), 0) is False
    assert scope.matches(_user("x"), 1) is True
