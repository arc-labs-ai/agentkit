"""WorkingContext — the in-flight reasoning state after the context split.

These tests pin the new contract:

  1. ``append`` / ``extend`` / ``clear_messages`` / ``note`` /
     ``update_scratchpad`` touch the tail + scratchpad only — never
     the prefix (the cache-stable head must stay bit-identical
     across turns).
  2. ``fork()`` returns an independent copy whose mutations do not
     leak back.
  3. ``merge(other)`` concat-mode appends + updates scratchpad
     (legacy behaviour); union-mode dedupes structurally.
  4. ``slice(scope)`` returns a NEW context filtered by predicate,
     and the prefix + scratchpad are inherited unchanged.
  5. ``freeze()`` returns a fully-immutable snapshot that round-trips
     into an equivalent live ``WorkingContext``.
  6. ``diff(other)`` reports added/removed messages, scratchpad
     deltas, and prefix changes.
  7. ``assembled()`` = prefix + messages.
  8. ``tokens()`` with ``ApproxTokenCounter`` matches chars/4 of the
     assembled view.
  9. ``shared=True`` plus concurrent appends from two coroutines
     doesn't race (both messages survive).
"""

from __future__ import annotations

import asyncio

from agentkit.context import (
    AllOf,
    ApproxTokenCounter,
    FrozenContext,
    LastNTurns,
    PrefixContext,
    RoleFilter,
    Since,
    Tagged,
    WorkingContext,
)
from agentkit.kernel.types import Message


def _run(coro):
    return asyncio.run(coro)


def _user(text: str, *, name: str | None = None) -> Message:
    return Message("user", text, name=name)


def _asst(text: str, *, name: str | None = None) -> Message:
    return Message("assistant", text, name=name)


def _tool(text: str, *, name: str | None = None) -> Message:
    return Message("tool", text, name=name)


def _system(text: str) -> Message:
    return Message("system", text)


def _make(prefix: str = "sys", **kw) -> WorkingContext:
    return WorkingContext(prefix=PrefixContext(system_prompt=prefix), **kw)


# ── append + tail-only contract ───────────────────────────────────


def test_append_grows_tail_and_leaves_prefix_untouched():
    wc = _make()
    before = wc.prefix
    wc.append(_user("hi"))
    assert [m.content for m in wc.messages] == ["hi"]
    assert wc.prefix is before  # cache discipline: prefix is bit-stable


def test_extend_clear_note_update_scratchpad_round_trip():
    wc = _make()
    wc.extend([_user("a"), _asst("b")]).clear_messages()
    assert wc.messages == []
    wc.note("k", 1).update_scratchpad({"j": 2})
    assert wc.get("k") == 1
    assert wc.get("j") == 2
    assert wc.get("missing", "fallback") == "fallback"


# ── fork is independent ───────────────────────────────────────────


def test_fork_is_independent_of_source():
    base = _make().append(_user("a"))
    base.note("k", 1)
    forked = base.fork()
    forked.append(_user("b")).note("k", 2)
    # Source untouched on both axes.
    assert len(base.messages) == 1 and base.get("k") == 1
    # Fork carries its own mutations.
    assert len(forked.messages) == 2 and forked.get("k") == 2
    # Prefix is shared by reference (frozen, safe to share).
    assert forked.prefix is base.prefix


# ── merge: concat (legacy) + union (dedup) ────────────────────────


def test_merge_concat_appends_and_updates_scratchpad():
    parent = _make().append(_user("p")).note("a", 1)
    child = _make().append(_user("c")).note("b", 2)
    parent.merge(child)
    assert [m.content for m in parent.messages] == ["p", "c"]
    assert parent.scratchpad == {"a": 1, "b": 2}


def test_merge_union_dedupes_messages():
    parent = _make().append(_user("x"), _user("y"))
    sibling = _make().append(_user("y"), _user("z"))
    parent.merge(sibling, mode="union")
    # Only the NEW message from `sibling` lands; the shared "y" doesn't repeat.
    assert [m.content for m in parent.messages] == ["x", "y", "z"]


# ── slice: scoped views over the tail ─────────────────────────────


def test_slice_last_n_turns_keeps_system_and_recent_pair():
    wc = _make()
    wc.append(
        _user("q1"),
        _asst("a1"),
        _user("q2"),
        _asst("a2"),
        _user("q3"),
        _asst("a3"),
    )
    sliced = wc.slice(LastNTurns(2))
    # System messages always survive; the last two user/asst pairs survive.
    contents = [m.content for m in sliced.messages]
    assert "q3" in contents and "a3" in contents
    assert "q2" in contents and "a2" in contents
    assert "q1" not in contents and "a1" not in contents


def test_slice_role_filter_keeps_only_matching_role():
    wc = _make()
    wc.append(_user("u"), _asst("a"), _tool("t"))
    sliced = wc.slice(RoleFilter(frozenset({"tool"})))
    assert [m.role for m in sliced.messages] == ["tool"]


def test_slice_tagged_keeps_messages_by_name():
    wc = _make()
    wc.append(
        _asst("planner says", name="planner"),
        _asst("researcher replies", name="researcher"),
        _asst("planner again", name="planner"),
    )
    sliced = wc.slice(Tagged("planner"))
    assert all(m.name == "planner" for m in sliced.messages)
    assert len(sliced.messages) == 2


def test_slice_composes_allof_role_and_since():
    wc = _make()
    for i in range(10):
        # Alternate user/asst messages so the index-since predicate is meaningful.
        wc.append(_user(f"q{i}") if i % 2 == 0 else _asst(f"a{i}"))
    sliced = wc.slice(AllOf((RoleFilter(frozenset({"user"})), Since(5))))
    # Only user messages with index >= 5 survive.
    assert all(m.role == "user" for m in sliced.messages)
    assert [m.content for m in sliced.messages] == ["q6", "q8"]


def test_slice_inherits_prefix_and_scratchpad():
    wc = _make("sys")
    wc.append(_user("x")).note("k", "v")
    sliced = wc.slice(RoleFilter(frozenset({"user"})))
    assert sliced.prefix is wc.prefix
    assert sliced.scratchpad == {"k": "v"}


# ── freeze: immutable snapshot ────────────────────────────────────


def test_freeze_returns_frozen_with_tuple_messages():
    wc = _make().append(_user("a"), _asst("b")).note("k", 1)
    snap = wc.freeze()
    assert isinstance(snap, FrozenContext)
    assert isinstance(snap.messages, tuple)
    assert snap.prefix is wc.prefix


def test_freeze_round_trip_is_functionally_equivalent():
    wc = _make().append(_user("hello"))
    wc.note("topic", "redis")
    snap = wc.freeze()
    rehydrated = WorkingContext(
        prefix=snap.prefix,
        messages=list(snap.messages),
        scratchpad=dict(snap.scratchpad),
    )
    assert rehydrated.messages == wc.messages
    assert rehydrated.scratchpad == wc.scratchpad
    assert rehydrated.prefix == wc.prefix


# ── diff: structural comparison ───────────────────────────────────


def test_diff_reports_added_removed_and_scratchpad_changes():
    a = _make().append(_user("shared"), _user("only-a")).note("k", 1)
    b = _make().append(_user("shared"), _user("only-b")).note("k", 2).note("removed", 7)
    d = a.diff(b)
    # `a - b`: "only-a" is in a but not b → added; "only-b" reversed → removed.
    assert any(m.content == "only-a" for m in d.messages_added)
    assert any(m.content == "only-b" for m in d.messages_removed)
    # scratchpad: k differs (a says 1, b says 2 → a's view is 1); removed key
    # present in b but not a → None signals removal.
    assert d.scratchpad_changes.get("k") == 1
    assert d.scratchpad_changes.get("removed") is None


def test_diff_prefix_changed_flag_is_structural():
    a = _make("sys-A")
    b = _make("sys-B")
    assert a.diff(b).prefix_changed is True
    assert a.diff(_make("sys-A")).prefix_changed is False


# ── assembled = prefix + messages ─────────────────────────────────


def test_assembled_is_prefix_plus_tail():
    wc = WorkingContext(
        prefix=PrefixContext(
            system_prompt="seed",
            grounding=(Message("system", "context"),),
        ),
    )
    wc.append(_user("hi"))
    out = wc.assembled()
    assert [m.role for m in out] == ["system", "system", "user"]
    assert [m.content for m in out] == ["seed", "context", "hi"]


def test_assembled_returns_fresh_list_per_call():
    wc = _make().append(_user("hi"))
    first = wc.assembled()
    second = wc.assembled()
    assert first == second and first is not second  # mutation safety


# ── tokens via ApproxTokenCounter ───────────────────────────────────


def test_tokens_matches_chars_over_four_of_assembled():
    wc = WorkingContext(
        prefix=PrefixContext(system_prompt="abcd"),  # 4 chars
        messages=[Message("user", "wxyz")],  # 4 chars
        token_counter=ApproxTokenCounter(chars_per_token=4.0),
    )
    # 8 total chars / 4 = 2 tokens.
    assert _run(wc.tokens()) == 2


# ── shared=True: concurrent appends don't lose messages ───────────


def test_shared_concurrent_appends_serialise_into_the_tail():
    wc = _make(shared=True)

    async def go() -> None:
        async def writer(label: str, n: int) -> None:
            for i in range(n):
                wc.append(_user(f"{label}-{i}"))
                # Yield to the loop so the tasks really interleave.
                await asyncio.sleep(0)

        await asyncio.gather(writer("alice", 10), writer("bob", 10))

    _run(go())
    contents = [m.content for m in wc.messages]
    assert len(contents) == 20
    assert sum(1 for c in contents if c.startswith("alice-")) == 10
    assert sum(1 for c in contents if c.startswith("bob-")) == 10


# ── lock + apply_locked deliver real cross-coroutine atomicity ──────────────
#
# ``ctx.lock`` and ``ctx.apply_locked(fn)`` are explicit primitives for
# multi-step mutation sequences that need cross-coroutine atomicity.
# Mutating methods do NOT auto-lock; callers opt in via these primitives.


def test_lock_is_a_real_asyncio_lock_when_shared():
    """The ``lock`` property hands callers the same ``asyncio.Lock``
    instance the class holds privately, so callers can serialize
    multi-step mutations explicitly."""
    wc = _make(shared=True)
    assert isinstance(wc.lock, asyncio.Lock)
    # Same object across calls — not a fresh lock every time.
    assert wc.lock is wc.lock


def test_apply_locked_serializes_multi_step_mutations_with_awaits():
    """The canonical use-case: a sequence like ``append(x);
    await_something(); append(y)`` must observe AS A UNIT to other
    coroutines. Without the lock, two concurrent writers would
    interleave their pairs (x1, y2, y1, x2). With apply_locked the
    pairs are adjacent: (x1, y1, x2, y2) or (x2, y2, x1, y1)."""
    wc = _make(shared=True)

    async def go() -> None:
        async def add_pair(label: str) -> None:
            def _mutate(ctx):
                ctx.append(_user(f"{label}-A"))

            async def _mutate_with_await(ctx):
                ctx.append(_user(f"{label}-A"))
                await asyncio.sleep(0)  # force a yield mid-sequence
                ctx.append(_user(f"{label}-B"))

            await wc.apply_locked(_mutate_with_await)
            _ = _mutate  # keep the closure annotated; lint-friendly

        await asyncio.gather(*(add_pair(f"w{i}") for i in range(5)))

    _run(go())
    contents = [m.content for m in wc.messages]
    assert len(contents) == 10  # 5 writers × pair

    # The load-bearing invariant: every writer's A precedes its B
    # WITHOUT another writer's message between them.
    for i in range(5):
        label = f"w{i}"
        positions = [j for j, c in enumerate(contents) if c.startswith(f"{label}-")]
        assert len(positions) == 2, f"writer {label} should have exactly 2 messages"
        # Adjacent (no interleave).
        assert positions[1] == positions[0] + 1, (
            f"writer {label} pair was interleaved: positions {positions}"
        )


def test_apply_locked_supports_async_closures():
    """A closure that itself returns an awaitable is awaited inside the
    lock — so callers can compose async helpers under the same
    serialization without writing the ``async with`` boilerplate."""
    wc = _make(shared=True)

    async def go() -> None:
        async def _async_mutate(ctx):
            ctx.append(_user("step-1"))
            await asyncio.sleep(0)
            ctx.append(_user("step-2"))

        await wc.apply_locked(_async_mutate)

    _run(go())
    contents = [m.content for m in wc.messages]
    assert contents == ["step-1", "step-2"]


def test_manual_async_with_lock_blocks_concurrent_writers():
    """The other primitive: directly using ``async with ctx.lock`` for
    callers that want to inline the critical section. Confirms the
    lock is a real serialization barrier, not just exposed for show."""
    wc = _make(shared=True)
    order: list[str] = []

    async def go() -> None:
        async def writer(label: str) -> None:
            async with wc.lock:
                order.append(f"{label}-enter")
                await asyncio.sleep(0)
                order.append(f"{label}-exit")

        await asyncio.gather(writer("a"), writer("b"))

    _run(go())
    # Each writer's enter+exit must be adjacent (the lock prevented
    # interleave). The order between writers can vary by scheduler.
    enter_a = order.index("a-enter")
    enter_b = order.index("b-enter")
    assert order[enter_a + 1] == "a-exit"
    assert order[enter_b + 1] == "b-exit"
