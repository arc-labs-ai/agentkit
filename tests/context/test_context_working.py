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

import pytest

from agentkit.context import (
    AllOf,
    ApproxTokenCounter,
    ContextDiff,
    FrozenContext,
    LastNTurns,
    PrefixContext,
    RoleFilter,
    Since,
    Tagged,
    WorkingContext,
)
from agentkit.kernel._frozen import FrozenDict
from agentkit.kernel.types import Message, ToolCall


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


def _tool_call_turn(call_id: str = "c1", **arguments) -> Message:
    """An assistant turn that requested a tool — the shape EVERY message has on
    the coordinator fan-in path, and the shape none of the original merge tests
    used. ``arguments`` defaults to nested JSON because that is what a provider
    actually sends back."""
    return Message(
        "assistant",
        "",
        tool_calls=(ToolCall(call_id, "search", arguments or {"q": {"terms": ["a", "b"]}}),),
    )


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


def test_merge_union_dedupes_tool_call_messages():
    """The bug ``mode="union"`` shipped with. Dedup does ``set(self.messages)``,
    and a Message carrying tool calls was UNHASHABLE — ``ToolCall.arguments``
    was a mappingproxy then and is a ``FrozenDict`` now, and NEITHER is
    hashable — so the documented parent-merging-two-siblings path raised::

        parent.merge(sibling, mode="union")
        TypeError: unhashable type: 'dict'

    It passed review because every existing union test used plain messages;
    on a real tool-using agent every assistant turn looks like this."""
    parent = _make().append(_tool_call_turn("c1"))
    sibling = _make().append(_tool_call_turn("c1"), _tool_call_turn("c2"))
    parent.merge(sibling, mode="union")
    # The identical tool-call turn collapses; the distinct one lands.
    assert len(parent.messages) == 2
    assert [m.tool_calls[0].id for m in parent.messages] == ["c1", "c2"]


def test_merge_union_keeps_tool_calls_that_differ_only_in_arguments():
    """The counterpart, and the reason hashing a SUBSET of the fields is safe:
    these two share ``(id, name)`` so they land in the SAME hash bucket, and
    only ``__eq__`` can tell them apart. If dedup had leaned on the hash rather
    than equality, one of two genuinely different searches would vanish."""
    a = _tool_call_turn("c1", q="alpha")
    b = _tool_call_turn("c1", q="beta")
    assert hash(a) == hash(b) and a != b  # same bucket, different message
    parent = _make().append(a)
    parent.merge(_make().append(b), mode="union")
    assert [m.tool_calls[0].arguments["q"] for m in parent.messages] == ["alpha", "beta"]


def test_merge_union_on_plain_messages_is_unchanged():
    """POSITIVE CONTROL: the plain-message dedup that always worked keeps
    working, including the duplicate-within-`other` case."""
    parent = _make().append(_user("x"), _user("y"))
    parent.merge(_make().append(_user("y"), _user("z"), _user("z")), mode="union")
    assert [m.content for m in parent.messages] == ["x", "y", "z"]


def test_merge_concat_still_duplicates_tool_call_messages():
    """POSITIVE CONTROL: ``concat`` never hashed anything, so it worked before
    the fix and must behave identically after — no accidental dedup leaking
    into the default mode."""
    parent = _make().append(_tool_call_turn("c1"))
    parent.merge(_make().append(_tool_call_turn("c1")), mode="concat")
    assert len(parent.messages) == 2


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


# ── freeze: hashable enough to be the cache key it advertises ─────
#
# ``FrozenContext``'s own docstring promises a snapshot is safe "to use
# as a memoization-cache key". It wasn't. The dataclass hashed every
# field, and two of those fields are frozen CONTAINERS of objects the
# framework does not own. Measured before the fix::
#
#     hash(WorkingContext().note("k", "v").freeze())      1304396713497489601
#     hash(WorkingContext().note("plan", {}).freeze())    TypeError: unhashable type: 'dict'
#
# i.e. it broke by VALUE, not by type — on ``update_scratchpad({"plan":
# {...}})``, which is the documented API. The fix hashes the transcript
# axis plus the scratchpad KEYS; ``__eq__`` still compares everything.


def test_frozen_context_is_hashable_with_a_dict_scratchpad_value():
    """The documented API, verbatim from the ``update_scratchpad`` docstring —
    a plan object in the scratchpad — must not cost the snapshot its key."""
    wc = _make().update_scratchpad({"plan": {"steps": ["a", "b"], "done": False}})
    assert isinstance(hash(wc.freeze()), int)


def test_frozen_context_is_hashable_with_arbitrary_unhashable_values():
    """Lists, sets, nested dicts, and an object with no ``__hash__`` at all.
    A scratchpad value is any Python object; a snapshot that is hashable only
    when the agent stored scalars is not a cache key, it's a trap."""

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

    wc = _make().update_scratchpad(
        {"list": [1, 2], "set": {1, 2}, "deep": {"a": {"b": []}}, "obj": Unhashable()}
    )
    assert isinstance(hash(wc.freeze()), int)


def test_frozen_context_is_hashable_with_tool_call_messages():
    """The transcript half of the same problem: messages ARE hashed, so before
    ``ToolCall.__hash__`` a snapshot of any tool-using agent was unhashable
    even with an empty scratchpad."""
    wc = _make().append(_tool_call_turn("c1"), _user("q"))
    assert isinstance(hash(wc.freeze()), int)


def test_frozen_context_is_hashable_with_opaque_journal_entries():
    """``JournalEntryT`` is project-defined and very often a plain dict, so
    journal entries are excluded from the hash outright."""
    wc = _make()
    wc.journal.record({"kind": "note", "payload": {"x": [1, 2]}})
    assert isinstance(hash(wc.freeze()), int)


def test_frozen_context_hash_ignores_values_while_eq_does_not():
    """Same soundness argument as ``ToolCall``: hashing a subset only requires
    that EQUAL objects hash equally. Two snapshots differing in a scratchpad
    value share a bucket and ``__eq__`` separates them there — a correct cache
    with one extra comparison, not a wrong one."""
    a = _make().update_scratchpad({"plan": {"v": 1}}).freeze()
    b = _make().update_scratchpad({"plan": {"v": 2}}).freeze()
    assert hash(a) == hash(b)
    assert a != b
    assert len({a, b}) == 2


def test_frozen_context_hash_separates_transcript_and_scratchpad_keys():
    """The parts that ARE hashed have to earn their place: two snapshots that
    differ in messages, or in WHICH notes exist, must land in different buckets
    or a memo cache degrades to a linear scan."""
    base = _make().append(_user("a"))
    assert hash(base.freeze()) != hash(_make().append(_user("b")).freeze())
    assert hash(base.note("k", {}).freeze()) != hash(base.fork().note("j", {}).freeze())


def test_frozen_context_works_as_a_memoization_cache_key():
    """The promise the docstring makes, exercised the way a caller would: equal
    snapshots hit, an unequal one misses, and the value survives a lookup by a
    DIFFERENT but equal snapshot object."""
    wc = _make().append(_tool_call_turn("c1")).update_scratchpad({"plan": {"v": 1}})
    cache = {wc.freeze(): "cached-result"}
    assert cache[wc.freeze()] == "cached-result"  # equal snapshot, same entry
    assert wc.fork().append(_user("new")).freeze() not in cache


def test_frozen_context_survives_deepcopy_and_pickle_with_tool_calls():
    """Pickling a snapshot of a tool-using agent once raised ``TypeError:
    cannot pickle 'mappingproxy' object``, and ``ToolCall.__reduce__`` was
    written to fix it. ``FrozenDict`` now carries its own ``__reduce__``, so
    the payload pickles on its own account and ``ToolCall``'s hook survives
    only to keep already-written records readable. ``FrozenContext`` needs no
    hook at all: it pickles as soon as its contents do, which is what this
    test says."""
    import copy as _copy
    import pickle

    snap = _make().append(_tool_call_turn("c1")).update_scratchpad({"plan": {"v": [1]}}).freeze()

    assert _copy.deepcopy(snap) == snap
    revived = pickle.loads(pickle.dumps(snap))
    assert revived == snap
    assert hash(revived) == hash(snap)
    assert revived.messages[0].tool_calls[0].arguments["q"] == {"terms": ["a", "b"]}


def test_frozen_context_hashes_even_when_the_scratchpad_cannot_be_pickled():
    """Hashability and picklability are deliberately decoupled. A scratchpad
    holding an open socket / a lambda / a live client cannot be serialised —
    that is the CALLER's object failing, not this type's shape — but the
    snapshot must still work as an in-process cache key."""
    import pickle

    # Unhashable AND unpicklable: a dict (no ``__hash__``) wrapping a closure
    # (no pickle). Pre-fix this failed at ``hash`` before pickling was reached.
    snap = _make().note("client", {"callback": lambda x: x}).freeze()

    assert isinstance(hash(snap), int)
    with pytest.raises((TypeError, AttributeError, pickle.PicklingError)):
        pickle.dumps(snap)


def test_frozen_context_equality_and_shape_are_unchanged():
    """POSITIVE CONTROL: adding ``__hash__`` must not touch equality, field
    order, or the sorted-scratchpad shape. Passes before and after."""
    wc = _make().append(_user("hello")).update_scratchpad({"z": 1, "a": 2})
    snap = wc.freeze()
    assert snap == wc.freeze()
    assert snap != _make().append(_user("other")).update_scratchpad({"z": 1, "a": 2}).freeze()
    assert snap.scratchpad == (("a", 2), ("z", 1))  # sorted by key
    assert snap.messages == (_user("hello"),)


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


# ── diff: hashable, exactly as its own docstring says ─────────────
#
# ``ContextDiff``'s docstring states that its message tuples are frozen
# "so a ``ContextDiff`` is itself hashable / pickleable". Pickleable was
# true; hashable never was — ``scratchpad_changes`` is a plain dict, so
# the generated all-fields hash reached it on EVERY instance. Measured
# before the fix::
#
#     hash(ContextDiff((), (), {}, False))   TypeError: unhashable type: 'dict'
#     hash(a.diff(b))                        TypeError: unhashable type: 'dict'
#
# Even the empty diff, which is what makes this different from the
# ``FrozenContext`` bug above: that one broke by VALUE, this one by TYPE.
# The fix hashes the message tuples + the changed KEYS + prefix_changed.


def test_context_diff_is_hashable_when_it_is_empty():
    """The degenerate diff — no messages, no changes — was unhashable too, so
    the dict payload never even had to be reached to break it."""
    assert isinstance(hash(ContextDiff((), (), {}, False)), int)


def test_context_diff_from_the_documented_helper_is_hashable():
    """``WorkingContext.diff(other)`` is the only sanctioned producer, and the
    scratchpad values it copies in are whatever the agent noted — here a plan
    object and a tool-call transcript, i.e. the ordinary case."""
    a = _make().append(_tool_call_turn("c1")).update_scratchpad({"plan": {"steps": ["a"]}})
    b = _make().append(_user("other")).update_scratchpad({"plan": {"steps": ["b"]}})
    assert isinstance(hash(a.diff(b)), int)


def test_context_diff_is_hashable_with_arbitrary_unhashable_changed_values():
    """A changed value is any Python object — lists, sets, nested dicts, and a
    class that has opted out of hashing entirely. Same rule as the scratchpad:
    a diff that is hashable only when the agent stored scalars is a trap."""

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

    changes = {"list": [1, 2], "set": {1, 2}, "deep": {"a": {"b": []}}, "obj": Unhashable()}
    assert isinstance(hash(ContextDiff((), (), changes, False)), int)


def test_context_diff_hash_ignores_changed_values_while_eq_does_not():
    """The soundness argument, exercised: two diffs that touched the SAME key
    with different new values share a bucket, stay unequal, and both survive in
    a ``set``. Hashing a subset only requires that EQUAL objects hash equally."""
    a = ContextDiff((), (), {"plan": {"v": 1}}, False)
    b = ContextDiff((), (), {"plan": {"v": 2}}, False)
    assert hash(a) == hash(b)
    assert a != b
    assert len({a, b}) == 2


def test_context_diff_hash_is_o1_in_the_changed_payload():
    """Proven STRUCTURALLY, never by timing, so this cannot go flaky: a diff
    carrying a 100_000-key changed value hashes to the same number as one
    carrying a single key, which is only possible if the value is never read."""
    huge = {f"k{i}": i for i in range(100_000)}
    assert hash(ContextDiff((), (), {"plan": {"k0": 0}}, False)) == hash(
        ContextDiff((), (), {"plan": huge}, False)
    )


def test_context_diff_hash_does_not_depend_on_scratchpad_insertion_order():
    """Why the changed keys go in as a ``frozenset`` and not a key tuple. Dict
    equality ignores insertion order, so these two diffs are EQUAL — an
    order-sensitive key tuple would hash them differently and break the one
    invariant a hash has to keep."""
    a = ContextDiff((), (), {"a": 1, "b": 2}, False)
    b = ContextDiff((), (), {"b": 2, "a": 1}, False)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_context_diff_hash_separates_the_parts_it_keeps():
    """The hashed parts have to earn their place: diffs differing in messages,
    in WHICH keys changed, or in the prefix flag must land in different buckets
    or a set of diffs degrades to a linear scan."""
    base = ContextDiff((_user("a"),), (), {"k": 1}, False)
    assert hash(base) != hash(ContextDiff((_user("b"),), (), {"k": 1}, False))
    assert hash(base) != hash(ContextDiff((_user("a"),), (_user("z"),), {"k": 1}, False))
    assert hash(base) != hash(ContextDiff((_user("a"),), (), {"other": 1}, False))
    assert hash(base) != hash(ContextDiff((_user("a"),), (), {"k": 1}, True))


def test_context_diff_hash_survives_tool_call_messages():
    """The message half: ``Message`` is hashable only because ``ToolCall`` is,
    so a diff over a tool-using transcript was doubly broken before."""
    d = ContextDiff((_tool_call_turn("c1"),), (_tool_call_turn("c2"),), {}, False)
    assert isinstance(hash(d), int)


def test_context_diff_equality_and_shape_are_unchanged():
    """POSITIVE CONTROL: adding ``__hash__`` must not touch equality, field
    access, or the mutable-dict shape a debug caller reads. Passes before and
    after the fix."""
    d = ContextDiff((_user("a"),), (), {"k": 1, "removed": None}, False)
    assert d == ContextDiff((_user("a"),), (), {"k": 1, "removed": None}, False)
    assert d != ContextDiff((_user("a"),), (), {"k": 2, "removed": None}, False)
    assert d.scratchpad_changes["k"] == 1
    assert d.scratchpad_changes.get("removed") is None
    assert isinstance(d.scratchpad_changes, dict)  # NOT frozen into a proxy
    assert d.scratchpad_changes == {"k": 1, "removed": None}  # eq against a PLAIN dict
    assert d.messages_added[0].content == "a"


def test_context_diff_scratchpad_changes_refuse_post_construction_writes():
    """THE BREAKING CHANGE. The class docstring promises a ``ContextDiff`` is a
    value — "hashable / pickleable" — and the message tuples honoured that while
    ``scratchpad_changes`` did not: ``d.scratchpad_changes["k"] = 2`` edited a
    delta that had already been computed, reported and possibly logged. A diff
    that can be edited is not a diff.

    Migration::

        d = dataclasses.replace(d, scratchpad_changes={**d.scratchpad_changes, "k": 2})
    """
    import dataclasses

    d = ContextDiff((), (), {"plan": {"v": [1, 2]}}, False)

    with pytest.raises(TypeError, match="frozen value"):
        d.scratchpad_changes["plan"] = "rewritten"
    with pytest.raises(TypeError, match="frozen value"):
        d.scratchpad_changes["plan"]["v"].append(3)  # deep, not just the top level
    with pytest.raises(TypeError, match="frozen value"):
        d.scratchpad_changes.update({"extra": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.scratchpad_changes = {}  # unchanged — the reference was always frozen

    amended = dataclasses.replace(d, scratchpad_changes={**d.scratchpad_changes, "extra": 1})
    assert amended.scratchpad_changes == {"plan": {"v": [1, 2]}, "extra": 1}
    assert d.scratchpad_changes == {"plan": {"v": [1, 2]}}
    assert isinstance(amended.scratchpad_changes, FrozenDict)


def test_context_diff_empty_scratchpad_changes_are_frozen_too():
    """The empty diff is the common one — two contexts that only differ in
    messages — so a freeze that skipped ``{}`` would leave it writable."""
    d = ContextDiff((), (), {}, False)
    assert isinstance(d.scratchpad_changes, FrozenDict)
    assert d.scratchpad_changes == {}
    with pytest.raises(TypeError, match="frozen value"):
        d.scratchpad_changes["first"] = 1


def test_diff_does_not_alias_the_live_scratchpad_values():
    """``diff()`` builds ``changes`` by reading LIVE scratchpad values
    (``changes[k] = v``), so before the freeze a nested value in the diff WAS
    the context's object — and a later write through the context retroactively
    changed what the diff said."""
    a = WorkingContext()
    b = WorkingContext()
    plan = {"steps": ["draft"]}
    a.update_scratchpad({"plan": plan})

    d = a.diff(b)
    assert d.scratchpad_changes["plan"] == {"steps": ["draft"]}

    plan["steps"].append("review")  # the context's own value moves on
    a.update_scratchpad({"plan": plan})
    assert d.scratchpad_changes["plan"] == {"steps": ["draft"]}  # the diff does not


def test_context_diff_still_deepcopies_and_pickles():
    """POSITIVE CONTROL: the payload stays a ``dict`` (a subclass), so both
    paths keep working. Both are also where a frozen payload could plausibly
    break — each rebuilds a dict subclass through ``__setitem__`` unless
    ``__reduce__`` intervenes — and the copy must come back FROZEN, or the
    freeze leaks through ``copy``."""
    import copy as _copy
    import pickle

    d = ContextDiff((_tool_call_turn("c1"),), (), {"plan": {"v": [1, 2]}}, True)
    assert _copy.deepcopy(d) == d
    assert pickle.loads(pickle.dumps(d)) == d
    for clone in (_copy.deepcopy(d), pickle.loads(pickle.dumps(d))):
        assert isinstance(clone.scratchpad_changes, FrozenDict)
        with pytest.raises(TypeError, match="frozen value"):
            clone.scratchpad_changes["plan"]["v"][0] = 9


def test_context_diff_hash_survives_deepcopy_and_pickle():
    """A copied diff is an EQUAL diff, so it must hash equally — otherwise a
    diff that crossed a process boundary would miss in a set the original
    populated."""
    import copy as _copy
    import pickle

    d = ContextDiff((_tool_call_turn("c1"),), (), {"plan": {"v": [1, 2]}}, True)
    assert hash(_copy.deepcopy(d)) == hash(d)
    assert hash(pickle.loads(pickle.dumps(d))) == hash(d)


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
    # 8 total chars / 4 = 2 tokens, plus the 4-token-per-message structural
    # overhead every provider charges (2 assembled messages) = 10.
    assert _run(wc.tokens()) == 2 + 2 * 4


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
