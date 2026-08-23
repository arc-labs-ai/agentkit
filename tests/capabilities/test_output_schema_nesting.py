"""The round-trip invariant, held across every adapter and through nesting.

`PydanticAdapter.serialize` broke this three times in a row — first at the top
level, then one level down, then for tuple containers — each fix handling the
shape that happened to be in front of me. That prompted the question this file
answers: do `DataclassAdapter` and `AttrsAdapter` have the same gap?

They do not, and the reason is structural rather than lucky. Pydantic exposes
TWO external spellings for a field (validation alias and serialization alias),
so `serialize` and `json_schema()`/`parse()` could pick different ones. attrs
and dataclasses expose exactly ONE: `attrs.asdict` emits the attrs field name
(`_secret`), the schema advertises that same name, and the `_secret` → `secret`
constructor remap happens only at construction time, applied per level by the
recursive parse. There is no second name for the two sides to disagree about.

So rather than re-testing shapes per adapter, the test that matters is the
INVARIANT: whatever an adapter advertises in its JSON Schema is what its
`serialize` must key by, and a serialized value must parse back. Stated once,
checked for all three, and true at every level of nesting — which is what makes
a future adapter that grows a second spelling fail here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentkit.capabilities.output_schema import adapt

# ── the shapes, one per flavour, each carrying its awkward field ───────────


@dataclass
class DcLeaf:
    a: str
    derived: str = field(init=False, default="D")  # the shape that used to crash


@dataclass
class DcMid:
    leaf: DcLeaf
    tag: str = field(init=False, default="T")


@dataclass
class DcBag:
    items: list[DcLeaf]
    lookup: dict[str, DcLeaf]


# Defined at MODULE scope, not inside a helper. This file uses
# ``from __future__ import annotations``, so ``leaf: AtLeaf`` is the STRING
# "AtLeaf" and resolving it needs the name in module globals — a function-local
# class raises ``NameError`` from inside the schema builder. That is a property
# of deferred annotations, not of the adapter.
attrs = pytest.importorskip("attrs")


@attrs.define
class AtLeaf:
    _secret: str  # attrs name `_secret`, __init__ arg `secret`
    kept: str = "k"


@attrs.define
class AtMid:
    leaf: AtLeaf
    _tag: str = "t"


@attrs.define
class AtBag:
    items: list[AtLeaf]


def _round_trips(value: Any) -> bool:
    adapter = adapt(type(value))
    return adapter.parse(json.dumps(adapter.serialize(value))) == value


# ── 1. the invariant, stated once ──────────────────────────────────────────


def _assert_keys_match(cls: type, value: Any) -> None:
    """What the model is SHOWN must be what a dumped value is KEYED BY.

    This is the property Pydantic violated: `json_schema()` said `uname` while
    `serialize` said `userName`, so the model was told one contract and the
    durable store held another.
    """
    adapter = adapt(cls)
    schema_keys = set(adapter.json_schema().get("properties", {}))
    dumped_keys = set(adapter.serialize(value))
    assert dumped_keys == schema_keys, (
        f"{cls.__name__}: schema advertises {sorted(schema_keys)} but serialize "
        f"emitted {sorted(dumped_keys)}"
    )


def test_a_dataclass_advertises_what_it_serializes() -> None:
    _assert_keys_match(DcLeaf, DcLeaf(a="x"))
    _assert_keys_match(DcMid, DcMid(leaf=DcLeaf(a="x")))


def test_an_attrs_class_advertises_what_it_serializes() -> None:
    _assert_keys_match(AtLeaf, AtLeaf(secret="s"))
    _assert_keys_match(AtMid, AtMid(leaf=AtLeaf(secret="s")))


def test_a_pydantic_model_advertises_what_it_serializes() -> None:
    pydantic = pytest.importorskip("pydantic")

    class M(pydantic.BaseModel):
        user_name: str = pydantic.Field(
            validation_alias="uname", serialization_alias="userName"
        )

    _assert_keys_match(M, M(uname="bob"))


# ── 2. and it survives nesting, in every flavour ───────────────────────────


def test_a_nested_dataclass_round_trips() -> None:
    """Nested `init=False` at BOTH levels — the fix applied per level by the
    recursive parse rather than only at the top."""
    assert _round_trips(DcMid(leaf=DcLeaf(a="x")))


def test_dataclass_containers_round_trip() -> None:
    """Lists and dicts of nested dataclasses. (`tuple` is deliberately
    unsupported and refused at construction — see the test below.)"""
    assert _round_trips(DcBag(items=[DcLeaf(a="x")], lookup={"k": DcLeaf(a="y")}))


def test_a_nested_attrs_class_round_trips() -> None:
    """Private attrs at both levels: `_secret` and `_tag` each need the
    name→init-arg remap, and the nesting must not skip the inner one."""
    assert _round_trips(AtMid(leaf=AtLeaf(secret="s")))
    assert _round_trips(AtBag(items=[AtLeaf(secret="s")]))


def test_an_unsupported_dataclass_field_type_is_refused_at_construction() -> None:
    """POSITIVE CONTROL for the honesty of the dataclass adapter: `tuple[X, Y]`
    is not supported, and it says so with the supported set — rather than
    accepting the class and failing later on a value. That refusal is why the
    tuple gap that hit Pydantic cannot exist here."""

    @dataclass
    class HasTuple:
        pair: tuple[DcLeaf, DcLeaf]

    with pytest.raises(TypeError, match="unsupported field type"):
        adapt(HasTuple)
