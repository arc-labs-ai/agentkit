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

import pytest

from agentkit.kernel._frozen import FrozenDict
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


def test_memory_item_metadata_is_a_dict_for_every_consumer() -> None:
    """POSITIVE CONTROL, and the constraint the freeze is under: ``metadata`` is
    a ``dict`` SUBCLASS, not a ``MappingProxyType``. It is serialised as JSON
    and read back through ``dataclasses.asdict``, neither of which a proxy
    survives. Passes before and after the freeze.

    This test used to end with ``item.metadata["path"] = "docs/a.md"``, commented
    "backends annotate after construction", asserting that callers COULD write
    into ``metadata`` after the fact. That was the old contract; the write half
    now lives in ``test_memory_item_metadata_refuses_post_construction_writes``
    below, which pins the refusal and shows the migration."""
    item = MemoryItem("c", "vector", 0.9, {"chunk": 1, "offsets": [0, 12]})
    assert isinstance(item.metadata, dict)
    assert item.metadata["chunk"] == 1 and item.metadata["offsets"][1] == 12
    assert len(item.metadata) == 2 and set(item.metadata) == {"chunk", "offsets"}
    assert dict(item.metadata) == {"chunk": 1, "offsets": [0, 12]}
    assert item.metadata == {"chunk": 1, "offsets": [0, 12]}  # eq against a PLAIN dict
    assert json.loads(json.dumps(item.metadata))["offsets"] == [0, 12]
    assert dataclasses.asdict(item)["metadata"] == {"chunk": 1, "offsets": [0, 12]}


def test_memory_item_metadata_refuses_post_construction_writes() -> None:
    """THE BREAKING CHANGE, stated as a test — and it breaks a habit this file
    previously ENDORSED (see the test above). ``frozen=True`` protected only the
    field reference::

        item.metadata = {}                  # FrozenInstanceError, as intended
        item.metadata["path"] = "docs/a.md" # ...but this rewrote the record

    Migration — a backend that wants to annotate an item builds a new one::

        item = dataclasses.replace(item, metadata={**item.metadata, "path": p})

    That matters here more than it looks: ``decorators.py`` and ``tool.py`` pass
    ``metadata`` THROUGH into a new ``MemoryItem``, so an in-place annotation
    could land on an item another source still holds.
    """
    item = MemoryItem("c", "vector", 0.9, {"chunk": 1})

    with pytest.raises(TypeError, match="frozen value"):
        item.metadata["path"] = "docs/a.md"
    with pytest.raises(TypeError, match="frozen value"):
        item.metadata.update({"path": "docs/a.md"})
    with pytest.raises(TypeError, match="frozen value"):
        del item.metadata["chunk"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.metadata = {}  # type: ignore[misc]  # unchanged — always was frozen

    annotated = dataclasses.replace(item, metadata={**item.metadata, "path": "docs/a.md"})
    assert annotated.metadata == {"chunk": 1, "path": "docs/a.md"}
    assert item.metadata == {"chunk": 1}  # the original is untouched
    assert isinstance(annotated.metadata, FrozenDict)  # ``replace`` re-runs __post_init__


def test_memory_item_metadata_is_frozen_all_the_way_down() -> None:
    """``metadata`` is "whatever the store held" — nested JSON, chunk-offset
    lists, provider blobs — so a shallow freeze would leave the interesting
    paths writable."""
    item = MemoryItem("c", "vector", 0.9, {"src": {"page": {"n": 1}}, "offsets": [0, 12]})

    with pytest.raises(TypeError, match="frozen value"):
        item.metadata["src"]["page"]["n"] = 2
    with pytest.raises(TypeError, match="frozen value"):
        item.metadata["offsets"].append(99)
    assert item.metadata["src"]["page"]["n"] == 1


def test_memory_item_does_not_alias_the_backends_dict() -> None:
    """``vector.py`` builds a metadata dict per chunk and ``tool.py`` passes one
    through. If the item aliased it, a backend reusing its local buffer across
    a result page would rewrite items it had already yielded."""
    buf = {"chunk": 1}
    item = MemoryItem("c", "vector", 0.9, buf)

    buf["chunk"] = 2  # the backend keeps using its own dict — legal
    assert item.metadata == {"chunk": 1}


def test_memory_item_tolerates_a_none_metadata_the_way_the_backends_assume() -> None:
    """``metadata`` is annotated ``dict``, but ``file.py`` and ``vector.py`` both
    guard ``item.metadata.get(...) if item.metadata else None`` — the backends do
    not trust the annotation, and neither does the freeze. ``deep_freeze`` passes
    non-container leaves through, so a ``None`` payload stays ``None`` rather
    than raising inside ``__post_init__``."""
    item = MemoryItem("c", "vector", 0.9, None)  # type: ignore[arg-type]
    assert item.metadata is None
    assert dataclasses.replace(item, metadata={"path": "a.md"}).metadata == {"path": "a.md"}


def test_memory_item_empty_and_default_metadata_are_frozen_too() -> None:
    """The edge the default hides: ``field(default_factory=dict)`` means most
    items carry ``{}``, and a freeze that only ran on a non-empty payload would
    leave the COMMON case writable."""
    assert isinstance(MemoryItem("c", "s").metadata, FrozenDict)
    assert isinstance(MemoryItem("c", "s", None, {}).metadata, FrozenDict)
    assert MemoryItem("c", "s").metadata == {}

    with pytest.raises(TypeError, match="frozen value"):
        MemoryItem("c", "s").metadata["first"] = 1


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
    cross process and storage boundaries through the cache decorator. They are
    also the two paths a frozen payload could plausibly break, since both
    rebuild a dict subclass through ``__setitem__`` unless ``__reduce__``
    intervenes — and the copy must come back FROZEN, or the freeze leaks."""
    item = MemoryItem("c", "vector", 0.9, {"deep": {"a": [1, {"b": 2}]}})
    for clone in (copy.deepcopy(item), pickle.loads(pickle.dumps(item))):
        assert clone == item
        assert isinstance(clone.metadata, FrozenDict)
        with pytest.raises(TypeError, match="frozen value"):
            clone.metadata["deep"]["a"][1]["b"] = 3


def test_memory_item_hash_survives_deepcopy_and_pickle() -> None:
    """A copied item is an EQUAL item, so it must hash equally — otherwise an
    item read back out of a cache would miss in a set the original populated."""
    item = MemoryItem("c", "vector", 0.9, {"deep": {"a": [1, {"b": 2}]}})
    assert hash(copy.deepcopy(item)) == hash(item)
    assert hash(pickle.loads(pickle.dumps(item))) == hash(item)
