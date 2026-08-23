"""MemoryItem — the value type every ``MemorySource`` returns.

The backends each have their own file; this one covers the shape they all
share. It exists because ``MemoryItem`` was a frozen dataclass that did not
behave like a value: ``metadata`` is ``field(default_factory=dict)``, so the
dataclass-generated all-fields hash reached a dict on EVERY item any backend
has ever returned. Measured before the fix::

    hash(MemoryItem(content="c", source="s"))                 TypeError: unhashable type: 'dict'
    hash(MemoryItem(content="c", source="s", metadata={"a": 1}))
    TypeError: unhashable type: 'dict'

Identical lines — unhashable by TYPE, not by value, so no caller ever saw a
working case to compare against. The caller this shape invites is the obvious
one: a composite memory pulls the same passage from two sources and wants
``set(items)`` to collapse the duplicate before the top-k cut.

The fix hashes ``(content, source, score)`` and leaves ``__eq__`` alone. That
is sound rather than a workaround: the hash invariant only requires EQUAL
objects to hash equally, never that unequal ones differ, so two items sharing
a bucket is what a bucket is for and ``__eq__`` separates them there.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle

from agentkit.memory import MemoryItem


def test_memory_item_is_hashable_with_the_default_empty_metadata() -> None:
    """The plainest item a backend can return — no metadata written at all —
    was unhashable. The bug needed no payload to reproduce."""
    assert isinstance(hash(MemoryItem(content="c", source="s")), int)


def test_memory_item_is_hashable_with_the_metadata_backends_actually_attach() -> None:
    """Representative payloads, not minimal ones: nested JSON straight out of a
    vector store, plus the ``None``/empty edges an unranked backend produces."""
    nested = {"chunk": {"id": 3, "offsets": [0, 120]}, "tags": ["a", "b"], "src": {"deep": {}}}
    assert isinstance(hash(MemoryItem("c", "vector", 0.91, nested)), int)
    assert isinstance(hash(MemoryItem("c", "tool", None, {})), int)
    assert isinstance(hash(MemoryItem("", "journal", None, {"ts": None})), int)


def test_memory_item_is_hashable_with_an_unhashable_metadata_value() -> None:
    """Metadata is whatever the store held — a live client handle, a set, an
    object that opted out of hashing. An item that is hashable only when the
    backend attached scalars is not a value type, it is a trap."""

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

    meta = {"list": [1, 2], "set": {1, 2}, "obj": Unhashable(), "deep": {"a": {"b": []}}}
    assert isinstance(hash(MemoryItem("c", "s", 0.5, meta)), int)


def test_memory_item_hash_ignores_metadata_while_eq_does_not() -> None:
    """The soundness argument, exercised. Two records of the same passage from
    the same source, differing only in chunk id, collide into one bucket, stay
    UNEQUAL, and both survive in a ``set`` — which is the honest answer for
    records that genuinely differ."""
    a = MemoryItem("the same passage", "vector", 0.9, {"chunk": 1})
    b = MemoryItem("the same passage", "vector", 0.9, {"chunk": 2})
    assert hash(a) == hash(b)
    assert a != b
    assert len({a, b}) == 2


def test_memory_item_hash_is_o1_in_the_metadata_payload() -> None:
    """Proven STRUCTURALLY rather than by timing, so it cannot go flaky: an
    item carrying 100_000 metadata keys hashes to the same number as one
    carrying a single key. Only possible if metadata is never read."""
    huge = {f"k{i}": {"nested": [i]} for i in range(100_000)}
    assert hash(MemoryItem("c", "s", 0.5, {"k0": {"nested": [0]}})) == hash(
        MemoryItem("c", "s", 0.5, huge)
    )


def test_memory_item_hash_separates_the_parts_it_keeps() -> None:
    """The hashed subset earns its place: content, the producing backend, and
    the relevance score all discriminate. A hash that dropped them would still
    be correct but would put every recall result in one bucket."""
    base = MemoryItem("c", "vector", 0.9)
    assert hash(base) != hash(MemoryItem("other", "vector", 0.9))
    assert hash(base) != hash(MemoryItem("c", "lexical", 0.9))
    assert hash(base) != hash(MemoryItem("c", "vector", 0.1))
    assert hash(base) != hash(MemoryItem("c", "vector", None))


def test_memory_items_dedupe_through_a_set() -> None:
    """The caller this unlocks: identical items collapse, so a composite memory
    can drop duplicates before reranking instead of scanning a list."""
    dup = (MemoryItem("passage", "vector", 0.9), MemoryItem("passage", "vector", 0.9))
    assert len(set(dup)) == 1
    assert len({*dup, MemoryItem("passage", "lexical", 0.9)}) == 2


def test_memory_item_metadata_stays_a_plain_mutable_dict() -> None:
    """POSITIVE CONTROL, and the constraint this commit is under: metadata is
    NOT frozen into a ``MappingProxyType``. It is serialised as JSON and read
    back through ``dataclasses.asdict``, neither of which a proxy survives.
    Passes before and after the fix."""
    item = MemoryItem("c", "vector", 0.9, {"chunk": 1})
    assert isinstance(item.metadata, dict)
    item.metadata["path"] = "docs/a.md"  # backends annotate after construction
    assert json.dumps(item.metadata)
    assert dataclasses.asdict(item)["metadata"] == {"chunk": 1, "path": "docs/a.md"}


def test_memory_item_field_access_and_equality_are_unchanged() -> None:
    """POSITIVE CONTROL: adding ``__hash__`` touches neither the fields nor
    ``__eq__``, which still compares metadata. Passes before and after."""
    item = MemoryItem(content="c", source="vector", score=0.9, metadata={"chunk": 1})
    assert (item.content, item.source, item.score) == ("c", "vector", 0.9)
    assert item == MemoryItem("c", "vector", 0.9, {"chunk": 1})
    assert item != MemoryItem("c", "vector", 0.9, {"chunk": 2})
    assert MemoryItem("c", "s").score is None and MemoryItem("c", "s").metadata == {}


def test_memory_item_still_deepcopies_and_pickles() -> None:
    """POSITIVE CONTROL: both already worked and must keep working — items
    cross process and storage boundaries through the cache decorator."""
    item = MemoryItem("c", "vector", 0.9, {"deep": {"a": [1, {"b": 2}]}})
    assert copy.deepcopy(item) == item
    assert pickle.loads(pickle.dumps(item)) == item


def test_memory_item_hash_survives_deepcopy_and_pickle() -> None:
    """A copied item is an EQUAL item, so it must hash equally — otherwise an
    item read back out of a cache would miss in a set the original populated."""
    item = MemoryItem("c", "vector", 0.9, {"deep": {"a": [1, {"b": 2}]}})
    assert hash(copy.deepcopy(item)) == hash(item)
    assert hash(pickle.loads(pickle.dumps(item))) == hash(item)
