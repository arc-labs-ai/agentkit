"""Positional scopes must survive composition.

The defect this file exists for, measured on a six-message transcript
before the fix:

    slice(LastNTurns(2))                                 -> 2 messages
    slice(AllOf((LastNTurns(2), RoleFilter(("user",))))) -> 6 messages
    slice(AnyOf((LastNTurns(2), Tagged("pinned"))))      -> 6 messages
    slice(Not(LastNTurns(2)))                            -> 0 messages

``LastNTurns`` is positional — it answers via ``_surviving_indices``
because a turn window is a property of the list, not of a message.
``WorkingContext.slice()`` preferred that hook, the combinators did
not forward it, and the fallback ``LastNTurns.matches()`` returned
``True`` for everything. So wrapping the window in a combinator turned
a context-window control into a no-op: no error, no warning, a prompt
three times the requested size, a bill to match, and an overflow a few
turns later. ``Not`` failed the other direction and emptied the whole
transcript.

These tests are written against ``WorkingContext.slice`` rather than
against the predicates directly, because the slice call site is where
the damage was and a predicate-level test would have kept passing.

The imports of the new symbols (``surviving_indices``,
``PositionalScope``, ``PositionalScopeError``) are deliberately made
INSIDE the tests that need them. With them at module scope, reverting
the fix would break collection and every test in the file would "fail"
for the wrong reason — which proves nothing about whether the
assertions below actually catch the bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentkit.context.context import WorkingContext
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

USERS = RoleFilter(frozenset({"user"}))
ASSISTANTS = RoleFilter(frozenset({"assistant"}))
NOT_SYSTEM = Not(RoleFilter(frozenset({"system"})))


def _flat() -> WorkingContext:
    """Six user messages, no system prompt — the exact shape the bug
    was measured on."""
    wc = WorkingContext()
    for i in range(6):
        wc.append(Message("user", f"m{i}"))
    return wc


def _transcript() -> WorkingContext:
    """A realistic shape: cache-stable system prompt + three turns.

    Indices: 0 system, 1 q0, 2 a0, 3 q1, 4 a1, 5 q2, 6 a2.
    ``LastNTurns(2)`` keeps {0, 3, 4, 5, 6}.
    """
    wc = WorkingContext()
    wc.append(Message("system", "House rules."))
    for i in range(3):
        wc.append(Message("user", f"q{i}"), Message("assistant", f"a{i}"))
    return wc


def _contents(wc: WorkingContext) -> list[str]:
    return [m.content for m in wc.messages]


# ── the bug, one test per combinator shape ────────────────────────


def test_allof_keeps_the_window_a_positional_child_asks_for():
    # Was 6 (the whole transcript). The window contributes {4, 5};
    # RoleFilter contributes all six; the intersection is the window.
    assert _contents(_flat().slice(AllOf((LastNTurns(2), USERS)))) == ["m4", "m5"]


def test_anyof_unions_the_window_with_a_pointwise_child():
    # Was 5 (everything). A pinned note outside the window is exactly
    # why AnyOf exists — the union must keep it AND still bound the
    # tail, rather than degrading to "keep it all".
    wc = WorkingContext()
    wc.append(Message("assistant", "pinned note", name="pinned"))
    for i in range(2):
        wc.append(Message("user", f"q{i}"), Message("assistant", f"a{i}"))
    sliced = wc.slice(AnyOf((LastNTurns(1), Tagged("pinned"))))
    assert _contents(sliced) == ["pinned note", "q1", "a1"]


def test_not_of_a_window_keeps_the_older_messages():
    # Was 0 — `not True` for every message, so the scope that means
    # "everything except the recent tail" deleted the transcript. The
    # complement is taken against the full index range: {0..5} - {4,5}.
    assert _contents(_flat().slice(Not(LastNTurns(2)))) == ["m0", "m1", "m2", "m3"]


def test_not_of_a_window_also_drops_the_system_prompt():
    # LastNTurns keeps every system message, so its complement drops
    # them. Pinned here because the inverse ("system survives
    # everything") is the intuitive guess and it is wrong: `Not` must
    # complement what the inner scope actually returned, not what the
    # reader assumes about system messages.
    assert _contents(_transcript().slice(Not(LastNTurns(2)))) == ["q0", "a0"]


def test_not_stays_an_involution_over_a_positional_scope():
    double = _flat().slice(Not(Not(LastNTurns(2))))
    assert _contents(double) == _contents(_flat().slice(LastNTurns(2))) == ["m4", "m5"]


def test_nested_combinators_forward_the_window_through_every_level():
    # AllOf(AnyOf(positional, pointwise), pointwise) — the shape where
    # a conditional "only forward when a child is positional" fix
    # would still lose the window one level down. Was 6.
    scope = AllOf((AnyOf((LastNTurns(2), Tagged("pinned"))), USERS))
    assert _contents(_flat().slice(scope)) == ["m4", "m5"]


def test_pointwise_child_is_still_applied_per_message_inside_the_window():
    # The mixed case has to do BOTH jobs: bound the tail and filter
    # within it. Window is {0, 3, 4, 5, 6}; only 4 and 6 are assistant
    # messages. Pre-fix this returned all three assistant turns
    # including a0, which is outside the window.
    scope = AllOf((LastNTurns(2), ASSISTANTS))
    assert _contents(_transcript().slice(scope)) == ["a1", "a2"]


def test_window_then_filter_matches_filter_composed_into_the_window():
    # The documented workaround — slice twice — must now agree with
    # the single composed scope. When these two disagree, the
    # combinator has lost the window again.
    ctx = _transcript()
    chained = ctx.slice(LastNTurns(2)).slice(NOT_SYSTEM)
    composed = ctx.slice(AllOf((LastNTurns(2), NOT_SYSTEM)))
    assert _contents(composed) == _contents(chained) == ["q1", "a1", "q2", "a2"]


# ── the hook is a real extension point, not a LastNTurns special case ──


@dataclass(frozen=True)
class _FirstOnly:
    """A third-party positional scope: keeps only the first message.

    ``matches`` fails the test outright rather than returning a guess,
    so this doubles as proof that a composed positional scope is NEVER
    routed through the pointwise path.
    """

    def matches(self, message: Message, index: int) -> bool:
        raise AssertionError("a positional child must not be asked pointwise")

    def _surviving_indices(self, messages: list[Message]) -> set[int]:
        return {0} if messages else set()


def test_a_user_defined_positional_scope_composes_the_same_way():
    assert _contents(_flat().slice(AnyOf((_FirstOnly(), LastNTurns(2))))) == [
        "m0",
        "m4",
        "m5",
    ]
    assert _contents(_flat().slice(AllOf((_FirstOnly(), USERS)))) == ["m0"]


def test_surviving_indices_helper_answers_for_both_kinds_of_scope():
    from agentkit.context.scope import surviving_indices

    messages = _transcript().messages
    assert surviving_indices(LastNTurns(2), messages) == {0, 3, 4, 5, 6}
    assert surviving_indices(USERS, messages) == {1, 3, 5}
    assert surviving_indices(AllOf((LastNTurns(2), USERS)), messages) == {3, 5}


def test_positional_scope_protocol_recognises_the_hook():
    from agentkit.context.scope import PositionalScope

    assert isinstance(LastNTurns(2), PositionalScope)
    assert isinstance(_FirstOnly(), PositionalScope)
    # Combinators carry the hook unconditionally — see the comment in
    # scope.py: making it conditional is how the bug comes back.
    assert isinstance(AllOf((USERS, Tagged("x"))), PositionalScope)
    assert not isinstance(USERS, PositionalScope)


# ── the fallback that made the bug silent ─────────────────────────


def test_lastnturns_refuses_the_pointwise_question_instead_of_lying():
    from agentkit.context.scope import PositionalScopeError

    with pytest.raises(PositionalScopeError, match="whole list"):
        LastNTurns(2).matches(Message("user", "m"), 0)


def test_a_combinator_refuses_regardless_of_child_order():
    # `all()`/`any()` short-circuit, so a generator would make this
    # raise or not depending on which child ran first and what the
    # message happened to be. An error that fires on message #3 but
    # not message #2 is worse than one that always fires.
    from agentkit.context.scope import PositionalScopeError

    msg = Message("assistant", "m")  # RoleFilter(user) rejects it
    with pytest.raises(PositionalScopeError):
        AllOf((USERS, LastNTurns(2))).matches(msg, 0)
    with pytest.raises(PositionalScopeError):
        AllOf((LastNTurns(2), USERS)).matches(msg, 0)
    with pytest.raises(PositionalScopeError):
        AnyOf((USERS, LastNTurns(2))).matches(Message("user", "m"), 0)
    with pytest.raises(PositionalScopeError):
        Not(LastNTurns(2)).matches(msg, 0)


# ── positive controls: these pass with OR without the fix ─────────


def test_each_scope_alone_is_unchanged():
    flat = _flat()
    assert _contents(flat.slice(LastNTurns(2))) == ["m4", "m5"]
    assert _contents(flat.slice(USERS)) == [f"m{i}" for i in range(6)]
    assert _contents(flat.slice(ASSISTANTS)) == []
    assert _contents(flat.slice(Since(4))) == ["m4", "m5"]
    assert _contents(flat.slice(Tagged("nobody"))) == []
    assert _contents(_transcript().slice(LastNTurns(2))) == [
        "House rules.",
        "q1",
        "a1",
        "q2",
        "a2",
    ]


def test_pointwise_combinations_are_unchanged():
    ctx = _transcript()
    assert _contents(ctx.slice(AllOf((USERS, Since(3))))) == ["q1", "q2"]
    assert _contents(ctx.slice(AllOf((USERS, Tagged("x"))))) == []
    assert _contents(ctx.slice(AnyOf((USERS, RoleFilter(frozenset({"system"})))))) == [
        "House rules.",
        "q0",
        "q1",
        "q2",
    ]
    assert _contents(ctx.slice(NOT_SYSTEM)) == ["q0", "a0", "q1", "a1", "q2", "a2"]
    assert _contents(ctx.slice(Not(AllOf((USERS, Since(3)))))) == [
        "House rules.",
        "q0",
        "a0",
        "a1",
        "a2",
    ]


def test_empty_transcript_slices_to_empty_for_every_shape():
    empty = WorkingContext()
    for scope in (
        LastNTurns(2),
        USERS,
        Not(LastNTurns(2)),  # complement of {} against range(0) is {}
        AllOf((LastNTurns(2), USERS)),
        AnyOf((LastNTurns(2), Tagged("pinned"))),
        AllOf((AnyOf((LastNTurns(2), Tagged("pinned"))), USERS)),
    ):
        assert empty.slice(scope).messages == [], scope


def test_a_window_larger_than_the_transcript_keeps_everything():
    # A too-wide window must be a no-op, not an error or a truncation
    # — callers set `n` from config and the transcript is short early
    # in a run.
    everything = [f"m{i}" for i in range(6)]
    assert _contents(_flat().slice(LastNTurns(99))) == everything
    assert _contents(_flat().slice(AllOf((LastNTurns(99), USERS)))) == everything
    assert _contents(_flat().slice(Not(LastNTurns(99)))) == []


def test_empty_combinators_stay_vacuous():
    # `AllOf(())` is `all([])` is True; `AnyOf(())` is `any([])` is
    # False. The set algebra has to seed itself to match, which is why
    # AllOf starts from the full index range and AnyOf from empty.
    flat = _flat()
    assert _contents(flat.slice(AllOf(()))) == [f"m{i}" for i in range(6)]
    assert _contents(flat.slice(AnyOf(()))) == []


def test_zero_turns_keeps_only_system_even_inside_a_combinator():
    ctx = _transcript()
    assert _contents(ctx.slice(LastNTurns(0))) == ["House rules."]
    assert _contents(ctx.slice(AllOf((LastNTurns(0), NOT_SYSTEM)))) == []
    assert _contents(ctx.slice(AnyOf((LastNTurns(0), Tagged("x"))))) == ["House rules."]


def test_scopes_stay_hashable_frozen_values():
    # They are used as cache keys for re-grounding decisions; adding
    # methods must not have cost that.
    assert len({LastNTurns(2), LastNTurns(2), LastNTurns(3)}) == 2
    assert AllOf((LastNTurns(2), USERS)) == AllOf((LastNTurns(2), USERS))
    assert hash(Not(LastNTurns(2))) == hash(Not(LastNTurns(2)))
