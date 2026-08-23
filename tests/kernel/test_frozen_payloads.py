"""`FrozenDict` / `FrozenList` / `deep_freeze` — the primitive behind making a
frozen dataclass's payload actually frozen.

The design constraint is not "refuse mutation" — that part is easy. It is
"refuse mutation WITHOUT breaking the four things these payloads have to do":
`json.dumps` (Checkpoint.state goes into a JSONB column), `dataclasses.asdict`
(AgentResult round-trips through it), `deepcopy` (the checkpointer snapshots
state on every save) and `pickle`. `MappingProxyType` — the obvious mechanism —
fails all four, which is why this module exists at all.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle

import pytest

from agentkit.kernel._frozen import FrozenDict, FrozenList, deep_freeze

# ── the mutation refusal itself ────────────────────────────────────────────


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.__setitem__("a", 2), id="setitem"),
        pytest.param(lambda d: d.__delitem__("a"), id="delitem"),
        pytest.param(lambda d: d.update({"b": 1}), id="update"),
        pytest.param(lambda d: d.pop("a"), id="pop"),
        pytest.param(lambda d: d.popitem(), id="popitem"),
        pytest.param(lambda d: d.clear(), id="clear"),
        pytest.param(lambda d: d.setdefault("b", 1), id="setdefault"),
        pytest.param(lambda d: d.__ior__({"b": 1}), id="ior"),
    ],
)
def test_every_dict_mutation_route_is_closed(mutate) -> None:
    """One blocked method is not a frozen dict. `update`, `setdefault` and `|=`
    are the ones an audit-rewriting caller reaches for after `[k] = v` fails."""
    with pytest.raises(TypeError, match="frozen value"):
        mutate(FrozenDict({"a": 1}))


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda x: x.append(1), id="append"),
        pytest.param(lambda x: x.extend([1]), id="extend"),
        pytest.param(lambda x: x.insert(0, 1), id="insert"),
        pytest.param(lambda x: x.remove("a"), id="remove"),
        pytest.param(lambda x: x.pop(), id="pop"),
        pytest.param(lambda x: x.clear(), id="clear"),
        pytest.param(lambda x: x.sort(), id="sort"),
        pytest.param(lambda x: x.reverse(), id="reverse"),
        pytest.param(lambda x: x.__setitem__(0, "z"), id="setitem"),
        pytest.param(lambda x: x.__delitem__(0), id="delitem"),
        pytest.param(lambda x: x.__iadd__(["b"]), id="iadd"),
        pytest.param(lambda x: x.__imul__(2), id="imul"),
    ],
)
def test_every_list_mutation_route_is_closed(mutate) -> None:
    with pytest.raises(TypeError, match="frozen value"):
        mutate(FrozenList(["a"]))


def test_the_error_says_how_to_do_it_properly() -> None:
    """A refusal that does not name the alternative just gets worked around.

    Asserts the EXACT text, not a substring. The loose version of this test
    matched only the `dataclasses.replace` substring and happily passed while the message
    shipped with doubled braces — `field={{**obj.field, ...}}` — because the
    constant had been written as if it were a format string that nothing ever
    formats. Every user hitting a frozen payload read that.
    """
    with pytest.raises(TypeError) as ei:
        FrozenDict({"a": 1})["a"] = 2
    assert str(ei.value) == (
        "this payload belongs to a frozen value and cannot be mutated in place. "
        "Build a new one instead: dataclasses.replace(obj, field={**obj.field, ...})"
    )
    assert "{{" not in str(ei.value) and "}}" not in str(ei.value)


# ── the constraints that ruled out MappingProxyType ────────────────────────


def test_a_frozen_payload_still_json_serialises() -> None:
    """`Checkpoint.state` is `json.dumps`'d straight into a JSONB column. A
    MappingProxyType raises `TypeError: not JSON serializable` here, which
    would have traded a mutability bug for a durable-write outage."""
    payload = deep_freeze({"turn": 3, "history": [{"role": "user"}], "tags": {"a": 1}})
    assert json.loads(json.dumps(payload)) == {
        "turn": 3,
        "history": [{"role": "user"}],
        "tags": {"a": 1},
    }


def test_a_frozen_payload_still_survives_dataclasses_asdict() -> None:
    """`AgentResult` round-trips through `asdict`. A proxy is not a `dict`
    subclass, so `asdict` never takes the mapping branch and raises."""

    @dataclasses.dataclass(frozen=True)
    class Rec:
        state: dict

    rec = Rec(state=deep_freeze({"a": [1, {"b": 2}]}))
    assert json.dumps(dataclasses.asdict(rec)) == '{"state": {"a": [1, {"b": 2}]}}'


def test_a_frozen_payload_deepcopies_and_pickles() -> None:
    """REGRESSION GUARD for the `__reduce__` hooks. copy and pickle rebuild a
    dict subclass by creating an empty one and repopulating it through
    `__setitem__` — the method being blocked — so without `__reduce__` both
    raise. The checkpointer deep-copies state on every snapshot, so that would
    have surfaced as a broken save, not a broken copy."""
    payload = deep_freeze({"a": [1, {"b": 2}]})
    assert copy.deepcopy(payload) == payload
    assert pickle.loads(pickle.dumps(payload)) == payload
    assert copy.copy(payload) == payload
    # ...and EVERY clone is still frozen, or the guarantee leaks through
    # whichever copy route a caller happens to use. Each route is asserted
    # separately on purpose: `deepcopy` goes through `__deepcopy__`, `copy`
    # through `__copy__`, and ONLY pickle reaches `__reduce__`. Asserting just
    # the deepcopy left `__reduce__` untested — a mutant that made it rebuild a
    # PLAIN dict survived, because a plain dict still compares equal.
    with pytest.raises(TypeError):
        copy.deepcopy(payload)["a"] = "evil"
    with pytest.raises(TypeError):
        copy.copy(payload)["a"] = "evil"
    with pytest.raises(TypeError):
        pickle.loads(pickle.dumps(payload))["a"] = "evil"
    # ...including the nested containers, which is what a shallow rebuild loses.
    with pytest.raises(TypeError):
        pickle.loads(pickle.dumps(payload))["a"][1]["b"] = "evil"


def test_a_frozen_payload_is_still_a_dict_and_still_compares_equal() -> None:
    """Consumers `isinstance`-check and compare against plain dicts. Both must
    keep working or every downstream branch changes shape."""
    frozen = deep_freeze({"a": 1})
    assert isinstance(frozen, dict)
    assert frozen == {"a": 1}
    assert {"a": 1} == frozen
    assert isinstance(deep_freeze([1]), list)
    assert deep_freeze([1]) == [1]


# ── depth ──────────────────────────────────────────────────────────────────


def test_freezing_reaches_all_the_way_down() -> None:
    """A shallow freeze leaves the same bug one level lower and harder to see."""
    payload = deep_freeze({"a": {"b": [1, {"c": 2}]}})
    with pytest.raises(TypeError):
        payload["a"]["b"][1]["c"] = 99
    with pytest.raises(TypeError):
        payload["a"]["b"].append(4)


def test_deep_freeze_is_idempotent() -> None:
    """A payload handed from one frozen value to another must not pay a second
    full walk — this is O(payload) work."""
    once = deep_freeze({"a": {"b": 1}})
    twice = deep_freeze(once)
    assert twice is once
    assert twice["a"] is once["a"]


def test_non_container_leaves_are_left_alone() -> None:
    """Recursively rewriting arbitrary user objects would mean guessing a
    reconstructor per type and silently swapping identities — the line
    `signals.py` draws too."""

    class Opaque:
        pass

    obj = Opaque()
    frozen = deep_freeze({"o": obj, "n": None, "i": 1, "s": "x", "t": (1, 2)})
    assert frozen["o"] is obj
    assert frozen["t"] == (1, 2)


def test_empty_containers_freeze_cleanly() -> None:
    assert deep_freeze({}) == {} and isinstance(deep_freeze({}), FrozenDict)
    assert deep_freeze([]) == [] and isinstance(deep_freeze([]), FrozenList)


def test_a_frozen_payload_does_not_alias_the_callers_object() -> None:
    """POSITIVE CONTROL for copy-on-freeze. Freezing the caller's own dict in
    place would not un-alias it — the caller would still hold a live handle and
    could keep editing what it already handed over."""
    mine = {"a": [1]}
    frozen = deep_freeze(mine)
    mine["a"].append(2)
    assert frozen["a"] == [1], "the frozen payload must not track the caller's object"


def test_a_mapping_proxy_is_normalised_not_passed_through() -> None:
    """A `MappingProxyType` handed in must not survive as one.

    It is already immutable, so passing it through looks harmless — and it
    silently defeats the reason this module exists. Measured while it WAS
    passed through: a caller who handed a proxy to `ToolCall(arguments=...)`
    got it stored verbatim, and `json.dumps(tc.arguments)` then raised
    `Object of type mappingproxy is not JSON serializable` on a payload the
    type advertises as serialisable. The defensive `dict(...)` unwraps around
    the codebase covered only the TOP level, so a nested proxy still broke.
    """
    from types import MappingProxyType

    frozen = deep_freeze(MappingProxyType({"q": "hi", "n": MappingProxyType({"a": 1})}))
    assert isinstance(frozen, FrozenDict)
    assert isinstance(frozen["n"], FrozenDict), "a NESTED proxy must be normalised too"
    assert json.loads(json.dumps(frozen)) == {"q": "hi", "n": {"a": 1}}
    with pytest.raises(TypeError):
        frozen["q"] = "x"


def test_other_mapping_subclasses_are_left_alone() -> None:
    """POSITIVE CONTROL for the narrowness of the rule above.

    Only the stdlib proxy is rewritten. A project's own `Mapping` type is
    returned by identity — swapping it for a `FrozenDict` would be the
    "silently reconstruct a user object" line this module refuses to cross
    everywhere else.
    """
    from collections.abc import Mapping

    class MyMapping(Mapping):
        def __init__(self, d): self._d = dict(d)
        def __getitem__(self, k): return self._d[k]
        def __iter__(self): return iter(self._d)
        def __len__(self): return len(self._d)

    mine = MyMapping({"a": 1})
    assert deep_freeze(mine) is mine
