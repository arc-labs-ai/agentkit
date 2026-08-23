"""DataclassAdapter — pin the contract for the stdlib-only default flavour:

1. Primitive fields and a ``required`` list map correctly.
2. Nested dataclass becomes an inline object schema.
3. ``Optional[str]`` widens ``type`` to allow ``"null"``.
4. ``Literal["a", "b"]`` becomes ``enum: ["a", "b"]``.
5. ``list[X]`` becomes ``type: array`` with the right ``items`` schema.
6. ``parse(valid)`` returns a typed instance.
7. ``parse(missing_required_field)`` raises :class:`OutputCoercionError`.
8. ``serialize`` round-trips.
9. Unsupported field type raises ``TypeError`` at ADAPTER CONSTRUCTION
   (not at parse) — fail-fast is the whole reason the adapter walks the
   type tree eagerly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import pytest

from agentkit.capabilities.output_schema import DataclassAdapter, OutputCoercionError


@dataclass
class Address:
    """Used to verify nested-dataclass schema inlining + nested parse."""

    street: str
    city: str


@dataclass
class Person:
    """Mixed shape that exercises every supported feature in one class."""

    name: str
    age: int
    tags: list[str]
    address: Address
    nickname: str | None = None
    role: Literal["admin", "user"] = "user"


def test_json_schema_primitives_and_required_list():
    adapter = DataclassAdapter(Person)
    schema = adapter.json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["name"] == {"type": "string"}
    assert props["age"] == {"type": "integer"}
    # ``name``, ``age``, ``tags``, ``address`` have no defaults → required.
    # ``nickname`` and ``role`` have defaults → not required.
    assert set(schema["required"]) == {"name", "age", "tags", "address"}


def test_nested_dataclass_inlined_as_object_schema():
    adapter = DataclassAdapter(Person)
    address_schema = adapter.json_schema()["properties"]["address"]
    assert address_schema["type"] == "object"
    assert address_schema["properties"]["street"] == {"type": "string"}
    assert address_schema["properties"]["city"] == {"type": "string"}
    assert set(address_schema["required"]) == {"street", "city"}


def test_optional_field_allows_null():
    schema = DataclassAdapter(Person).json_schema()
    nickname = schema["properties"]["nickname"]
    # We allow either the union-list form (``["string", "null"]``) or anyOf —
    # the adapter picks the simpler list form when possible.
    if isinstance(nickname.get("type"), list):
        assert "null" in nickname["type"] and "string" in nickname["type"]
    else:
        # anyOf branch
        types = {sub.get("type") for sub in nickname.get("anyOf", [])}
        assert "null" in types and "string" in types


def test_literal_becomes_enum():
    role = DataclassAdapter(Person).json_schema()["properties"]["role"]
    assert role["enum"] == ["admin", "user"]
    assert role.get("type") == "string"


def test_list_field_uses_items_schema():
    tags = DataclassAdapter(Person).json_schema()["properties"]["tags"]
    assert tags == {"type": "array", "items": {"type": "string"}}


def test_parse_valid_returns_typed_instance():
    adapter = DataclassAdapter(Person)
    payload = {
        "name": "alice",
        "age": 30,
        "tags": ["a", "b"],
        "address": {"street": "1 main", "city": "springfield"},
    }
    inst = adapter.parse(json.dumps(payload))
    assert isinstance(inst, Person)
    assert inst.name == "alice"
    assert inst.age == 30
    assert isinstance(inst.address, Address)
    assert inst.address.city == "springfield"
    # default-bearing fields are populated from the dataclass defaults
    assert inst.nickname is None
    assert inst.role == "user"


def test_parse_dict_input_also_works():
    """Provider structured-output modes hand us a dict; we must accept both."""
    adapter = DataclassAdapter(Address)
    inst = adapter.parse({"street": "5 elm", "city": "metropolis"})
    assert inst == Address(street="5 elm", city="metropolis")


def test_parse_missing_required_field_raises():
    adapter = DataclassAdapter(Person)
    payload = {
        "name": "bob",
        # age missing
        "tags": [],
        "address": {"street": "1 main", "city": "springfield"},
    }
    with pytest.raises(OutputCoercionError) as ei:
        adapter.parse(json.dumps(payload))
    err = ei.value
    assert any("age" in e and "missing" in e for e in err.errors)


def test_parse_wrong_type_raises_with_path():
    adapter = DataclassAdapter(Person)
    payload = {
        "name": "bob",
        "age": "not-a-number",
        "tags": [],
        "address": {"street": "1 main", "city": "springfield"},
    }
    with pytest.raises(OutputCoercionError) as ei:
        adapter.parse(json.dumps(payload))
    assert any("age" in e for e in ei.value.errors)


def test_parse_literal_outside_allowed_values_raises():
    adapter = DataclassAdapter(Person)
    payload = {
        "name": "bob",
        "age": 1,
        "tags": [],
        "address": {"street": "x", "city": "y"},
        "role": "not-a-valid-role",
    }
    with pytest.raises(OutputCoercionError) as ei:
        adapter.parse(json.dumps(payload))
    assert any("role" in e for e in ei.value.errors)


def test_parse_invalid_json_string_raises():
    adapter = DataclassAdapter(Person)
    with pytest.raises(OutputCoercionError):
        adapter.parse("not json")


def test_serialize_round_trips():
    adapter = DataclassAdapter(Person)
    inst = Person(
        name="alice",
        age=30,
        tags=["x"],
        address=Address(street="1 main", city="springfield"),
        nickname="al",
        role="admin",
    )
    dumped = adapter.serialize(inst)
    assert dumped == {
        "name": "alice",
        "age": 30,
        "tags": ["x"],
        "address": {"street": "1 main", "city": "springfield"},
        "nickname": "al",
        "role": "admin",
    }
    rebuilt = adapter.parse(dumped)
    assert rebuilt == inst


def test_unsupported_type_raises_at_construction_not_parse():
    """A field whose annotation we can't translate must fail when we build the
    adapter, not later when the LLM returns a value."""

    @dataclass
    class HasUnsupported:
        items: set[str]

    with pytest.raises(TypeError):
        DataclassAdapter(HasUnsupported)


def test_unsupported_union_with_two_non_none_types_rejected():
    """We only support ``Optional[X]``-shaped unions, not arbitrary ones."""

    @dataclass
    class BadUnion:
        either: int | str

    with pytest.raises(TypeError):
        DataclassAdapter(BadUnion)


def test_dict_field_uses_additional_properties():
    @dataclass
    class HasDict:
        counters: dict[str, int]

    schema = DataclassAdapter(HasDict).json_schema()["properties"]["counters"]
    assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}


def test_dict_with_non_string_keys_rejected():
    @dataclass
    class BadDict:
        m: dict[int, str]

    with pytest.raises(TypeError):
        DataclassAdapter(BadDict)


def test_default_factory_field_is_not_required():
    """A field with ``default_factory`` is optional from the caller's POV — the
    schema's ``required`` list must reflect that."""

    @dataclass
    class HasFactory:
        name: str
        items: list[str] = field(default_factory=list)

    schema = DataclassAdapter(HasFactory).json_schema()
    assert schema["required"] == ["name"]


def test_partial_parse_returns_partial_instance():
    """Streaming partial parse: tolerant JSON close + ``object.__new__``
    bypass yields a partial dataclass with only the streamed field set."""
    adapter = DataclassAdapter(Address)
    partial = adapter.partial_parse('{"street":"par')
    assert partial is not None
    assert isinstance(partial, Address)
    assert partial.street == "par"
    # ``city`` was never streamed — it stays unset (no default on the field).
    assert not hasattr(partial, "city")


def test_partial_parse_empty_is_none():
    adapter = DataclassAdapter(Address)
    assert adapter.partial_parse("") is None
    assert adapter.partial_parse("   ") is None


def test_name_defaults_to_class_name():
    assert DataclassAdapter(Person).name == "Person"


def test_python_type_is_the_class():
    assert DataclassAdapter(Person).python_type is Person


def test_constructor_rejects_non_dataclass():
    class NotADataclass:
        x: int

    with pytest.raises(TypeError):
        DataclassAdapter(NotADataclass)


# ---- field(init=False) ----------------------------------------------------
#
# REGRESSION. ``_dataclass_schema`` advertised every field as a property,
# then ``_coerce_dataclass`` fed them all to ``cls(**kwargs)`` — which
# rejects an ``init=False`` field. Measured: schema
# ``{"properties": {"a": ..., "b": {"type": "integer"}}}`` and
# ``parse('{"a":"x","b":5}')`` → ``TypeError: D.__init__() got an
# unexpected keyword argument 'b'``. A raw TypeError escapes the
# ``OutputCoercionError`` contract, so BOTH the repair loop and
# ``FunctionTool``'s ``ToolShapeError`` seam missed it, and
# ``parse(serialize(inst))`` was broken the same way.


@dataclass
class Derived:
    """``length`` is computed by the class, never supplied by the model."""

    text: str
    length: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.length = len(self.text)


def test_init_false_field_parses_instead_of_raising_typeerror():
    adapter = DataclassAdapter(Derived)
    out = adapter.parse('{"text":"xy","length":5}')
    assert isinstance(out, Derived)
    assert out.text == "xy"
    # The payload value wins over the recomputed one — that is what makes
    # a durable rehydrate restore what was stored.
    assert out.length == 5


def test_init_false_field_round_trips_both_ways():
    """``dataclasses.asdict`` emits the init=False field, so parse has to
    accept it; and the schema must still describe what serialize emits."""
    adapter = DataclassAdapter(Derived)
    inst = Derived("hello")
    assert adapter.parse(json.dumps(adapter.serialize(inst))) == inst
    assert "length" in adapter.json_schema()["properties"]


def test_init_false_field_is_not_required():
    """The model cannot know a computed value — demanding it would send the
    repair loop chasing a field it has no way to fix."""
    adapter = DataclassAdapter(Derived)
    assert adapter.json_schema().get("required") == ["text"]
    assert adapter.parse('{"text":"abc"}').length == 3  # default/__post_init__ still applies


def test_init_false_field_validates_from_a_plain_dict():
    """``validate`` shares the walker, so it carried the same TypeError."""
    adapter = DataclassAdapter(Derived)
    assert adapter.validate({"text": "xy", "length": 9}).length == 9


def test_init_false_on_a_frozen_dataclass_round_trips():
    """EDGE: a frozen dataclass rejects ``setattr``; the post-construction
    assignment goes through ``object.__setattr__``."""

    @dataclass(frozen=True)
    class Frozen:
        a: str
        tag: str = field(init=False, default="t")

    adapter = DataclassAdapter(Frozen)
    inst = Frozen("v")
    assert adapter.parse(json.dumps(adapter.serialize(inst))) == inst


def test_post_init_failure_surfaces_as_output_coercion_error():
    """EDGE: any construction failure must stay inside the coercion
    contract — a raw exception here is invisible to the repair loop."""

    @dataclass
    class Checked:
        a: int

        def __post_init__(self) -> None:
            if self.a < 0:
                raise ValueError("a must be >= 0")

    adapter = DataclassAdapter(Checked)
    with pytest.raises(OutputCoercionError) as exc:
        adapter.parse('{"a":-1}')
    assert "a must be >= 0" in str(exc.value.errors)


def test_ordinary_dataclass_still_parses_unchanged():
    """POSITIVE CONTROL — the init=False handling must not disturb the
    ordinary all-init-able shape."""
    adapter = DataclassAdapter(Address)
    inst = Address(street="1 Main", city="Springfield")
    assert adapter.parse('{"street":"1 Main","city":"Springfield"}') == inst
    assert adapter.json_schema()["required"] == ["street", "city"]
    with pytest.raises(OutputCoercionError):
        adapter.parse('{"street":"1 Main"}')
