"""PydanticAdapter — pin the contract end-to-end:

    1. ``json_schema()`` produces a schema that mirrors the model's field types
       (list[str], int, Optional[float] all show up correctly).
    2. ``parse(valid_json_str)`` returns a typed model instance.
    3. ``parse(invalid_json_str)`` raises :class:`OutputCoercionError` carrying
       the per-field validation errors so the retry middleware can reflect them
       back to the model.
    4. ``serialize(instance)`` round-trips back to the dict shape ``parse`` can
       reconsume.

The whole module is skipped when pydantic is not installed (it's an OPTIONAL
dep)."""

from __future__ import annotations

import json
from typing import Any

import pytest

pydantic = pytest.importorskip("pydantic")

from agentkit.capabilities.output_schema import (  # noqa: E402  — must follow importorskip
    OutputCoercionError,
    PydanticAdapter,
)


class Report(pydantic.BaseModel):
    """Mixed model that exercises list, primitive, and Optional[primitive]."""

    title: str
    tags: list[str]
    word_count: int
    confidence: float | None = None


def test_json_schema_reflects_field_types():
    adapter = PydanticAdapter(Report)
    schema = adapter.json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["title"]["type"] == "string"
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"
    assert props["word_count"]["type"] == "integer"
    # ``confidence`` is Optional[float] → either an anyOf with null or a typed
    # union list, depending on pydantic version. Both are acceptable.
    conf = props["confidence"]
    if "type" in conf:
        types = conf["type"] if isinstance(conf["type"], list) else [conf["type"]]
        assert "null" in types or "number" in types
    else:
        assert any("null" in str(v) or "number" in str(v) for v in conf.values())


def test_parse_returns_typed_instance():
    adapter = PydanticAdapter(Report)
    payload = {"title": "ok", "tags": ["a", "b"], "word_count": 42}
    inst = adapter.parse(json.dumps(payload))
    assert isinstance(inst, Report)
    assert inst.title == "ok"
    assert inst.tags == ["a", "b"]
    assert inst.word_count == 42
    assert inst.confidence is None


def test_parse_accepts_dict_too():
    """Provider structured-output modes return an already-parsed dict; the
    adapter must accept both string and dict inputs."""
    adapter = PydanticAdapter(Report)
    inst = adapter.parse({"title": "x", "tags": [], "word_count": 1})
    assert isinstance(inst, Report)
    assert inst.title == "x"


def test_parse_invalid_raises_with_errors():
    adapter = PydanticAdapter(Report)
    # word_count is an int — passing a string triggers Pydantic validation.
    bad = json.dumps({"title": "ok", "tags": [], "word_count": "not-a-number"})
    with pytest.raises(OutputCoercionError) as ei:
        adapter.parse(bad)
    err = ei.value
    assert err.raw == bad
    assert err.errors, "errors list should not be empty"
    assert any("word_count" in e for e in err.errors)


def test_parse_invalid_json_string_raises():
    adapter = PydanticAdapter(Report)
    with pytest.raises(OutputCoercionError):
        adapter.parse("not json at all")


def test_parse_strips_markdown_json_fences():
    """Claude routinely wraps JSON output in ```json … ``` even when asked
    for raw JSON. The adapter must peel the fence before parsing — otherwise
    every Reader / Synthesizer / Critic call would fail with InvalidJSON and
    burn retries until the model coincidentally stops fencing."""
    adapter = PydanticAdapter(Report)
    fenced = '```json\n{"title": "t", "tags": ["x"], "word_count": 3, "confidence": 0.5}\n```'
    result = adapter.parse(fenced)
    assert result == Report(title="t", tags=["x"], word_count=3, confidence=0.5)


def test_parse_strips_bare_triple_backtick_fences():
    """Some models emit ``` without the ``json`` language tag. Still safe to
    strip — valid JSON cannot start with a backtick."""
    adapter = PydanticAdapter(Report)
    fenced = '```\n{"title": "t", "tags": [], "word_count": 1, "confidence": 0.1}\n```'
    result = adapter.parse(fenced)
    assert result.title == "t"


def test_parse_passes_raw_json_through_unchanged():
    """The fence-strip helper must be a no-op for properly formatted raw
    JSON — no spurious transforms that could corrupt edge-case payloads."""
    adapter = PydanticAdapter(Report)
    raw = '{"title": "t", "tags": [], "word_count": 1, "confidence": 0.1}'
    result = adapter.parse(raw)
    assert result.title == "t"


def test_serialize_round_trips():
    adapter = PydanticAdapter(Report)
    inst = Report(title="t", tags=["x"], word_count=3, confidence=0.5)
    dumped = adapter.serialize(inst)
    assert dumped == {"title": "t", "tags": ["x"], "word_count": 3, "confidence": 0.5}
    # And parse(dumped) reconstructs the same instance.
    rebuilt = adapter.parse(dumped)
    assert rebuilt == inst


def test_name_defaults_to_class_name():
    adapter = PydanticAdapter(Report)
    assert adapter.name == "Report"


def test_name_override():
    adapter = PydanticAdapter(Report, name="my_custom_report")
    assert adapter.name == "my_custom_report"


def test_python_type_is_the_model_class():
    adapter = PydanticAdapter(Report)
    assert adapter.python_type is Report


def test_partial_parse_returns_partial_model():
    """Streaming partial parse: the tolerant JSON path closes the open
    string and ``model_construct`` builds the partial without validating
    missing required fields."""
    adapter = PydanticAdapter(Report)
    partial = adapter.partial_parse('{"title":"par')
    assert partial is not None
    assert isinstance(partial, Report)
    # ``model_dump(exclude_unset=True)`` shows only the fields the partial
    # stream actually populated — missing required ``tags`` / ``word_count``
    # are intentionally absent.
    assert partial.model_dump(exclude_unset=True) == {"title": "par"}


def test_partial_parse_empty_is_none():
    """Zero useful structure → ``None`` (caller pumps another delta)."""
    adapter = PydanticAdapter(Report)
    assert adapter.partial_parse("") is None
    assert adapter.partial_parse("   ") is None


def test_constructor_rejects_non_basemodel():
    with pytest.raises(TypeError):
        PydanticAdapter(dict)  # type: ignore[arg-type]


# ---- aliases -------------------------------------------------------------
#
# REGRESSION. ``serialize`` used a bare ``model_dump()`` (FIELD names)
# while ``json_schema()`` and ``parse()`` both speak ALIASES. Measured on
# ``user_name: str = Field(alias="userName")``: serialize gave
# ``{'user_name': 'bob'}`` and parsing that back raised
# ``OutputCoercionError: ['userName: Field required']`` — so the
# documented ``parse(json.dumps(serialize(v))) == v`` guarantee that
# durable rehydrate depends on was false for every aliased model.


class Aliased(pydantic.BaseModel):
    user_name: str = pydantic.Field(alias="userName")
    id_: int = pydantic.Field(alias="id")


class ByName(pydantic.BaseModel):
    """``populate_by_name`` accepts BOTH spellings on the way in."""

    model_config = pydantic.ConfigDict(populate_by_name=True)
    user_name: str = pydantic.Field(alias="userName")


def test_serialize_uses_aliases_so_parse_can_reconsume_it():
    adapter = PydanticAdapter(Aliased)
    value = Aliased(userName="bob", id=1)
    dumped = adapter.serialize(value)
    assert dumped == {"userName": "bob", "id": 1}
    assert adapter.parse(json.dumps(dumped)) == value


def test_serialize_keys_match_the_schema_shown_to_the_model():
    """The schema is the contract the model is held to; a serialized value
    that doesn't satisfy it is a transcript the model can't imitate."""
    adapter = PydanticAdapter(Aliased)
    dumped = adapter.serialize(Aliased(userName="bob", id=1))
    assert set(dumped) == set(adapter.json_schema()["properties"])


def test_alias_round_trip_survives_nesting():
    """EDGE: nested models dump through the same ``by_alias`` pass."""

    class Outer(pydantic.BaseModel):
        inner: Aliased
        top_level: int = pydantic.Field(alias="topLevel")

    adapter = PydanticAdapter(Outer)
    value = Outer(inner=Aliased(userName="b", id=2), topLevel=3)
    assert adapter.serialize(value) == {"inner": {"userName": "b", "id": 2}, "topLevel": 3}
    assert adapter.parse(json.dumps(adapter.serialize(value))) == value


def test_populate_by_name_model_round_trips():
    """EDGE: a ``populate_by_name`` model accepted its field name before the
    fix and must keep round-tripping now that we emit the alias."""
    adapter = PydanticAdapter(ByName)
    value = ByName(userName="bob")
    assert adapter.serialize(value) == {"userName": "bob"}
    assert adapter.parse(json.dumps(adapter.serialize(value))) == value
    assert adapter.parse('{"user_name":"bob"}') == value


def test_partial_parse_accepts_the_alias_keyed_stream():
    """EDGE: the model is shown the ALIASED schema, so a stream arrives
    alias-keyed — the name-only filter in ``partial_parse`` dropped every
    field and produced an empty partial."""
    adapter = PydanticAdapter(Aliased)
    partial = adapter.partial_parse('{"userName":"bo')
    assert partial is not None
    assert partial.user_name == "bo"


def test_unaliased_model_round_trips_unchanged():
    """POSITIVE CONTROL — ``by_alias`` is a no-op without aliases."""
    adapter = PydanticAdapter(Report)
    value = Report(title="t", tags=["a"], word_count=1, confidence=0.5)
    assert adapter.serialize(value) == {
        "title": "t",
        "tags": ["a"],
        "word_count": 1,
        "confidence": 0.5,
    }
    assert adapter.parse(json.dumps(adapter.serialize(value))) == value


# ── serialize speaks the language parse reads, for EVERY alias shape ───────
#
# Pydantic has two aliases, and `model_dump(by_alias=True)` picks the
# SERIALIZATION one — while `json_schema()` and `parse()` both speak the
# VALIDATION one. When they differ, the documented round trip
# `parse(dumps(serialize(v))) == v` breaks, and durable rehydrate with it:
#
#     Field(alias="userName")                    schema userName   dump userName   OK
#     Field(serialization_alias="userName")      schema user_name  dump userName   FAILS
#     Field(validation_alias="uname",
#           serialization_alias="userName")      schema uname      dump userName   FAILS
#
# This was previously written off as "a modelling choice" whose fix was
# `populate_by_name=True`. That is true of a single dump mode; the adapter knows
# both names and can just use the right one.


def _round_trips(model_cls: type, instance: Any) -> Any:
    adapter = PydanticAdapter(model_cls)
    return adapter.parse(json.dumps(adapter.serialize(instance)))


def test_a_serialization_only_alias_round_trips() -> None:
    """THE regression. `serialize` used to emit `userName` while the schema and
    `parse` both said `user_name`."""

    class M(pydantic.BaseModel):
        user_name: str = pydantic.Field(serialization_alias="userName")

    assert _round_trips(M, M(user_name="bob")).user_name == "bob"


def test_differing_validation_and_serialization_aliases_round_trip() -> None:
    """The worst shape: three different spellings in play at once."""

    class M(pydantic.BaseModel):
        user_name: str = pydantic.Field(
            validation_alias="uname", serialization_alias="userName"
        )

    assert _round_trips(M, M(uname="bob")).user_name == "bob"


def test_a_plain_alias_still_round_trips() -> None:
    """POSITIVE CONTROL — the shape that already worked must not regress."""

    class M(pydantic.BaseModel):
        user_name: str = pydantic.Field(alias="userName")

    assert _round_trips(M, M(userName="bob")).user_name == "bob"


def test_an_unaliased_model_is_untouched() -> None:
    """POSITIVE CONTROL — the overwhelmingly common case."""

    class M(pydantic.BaseModel):
        user_name: str
        count: int = 3

    dumped = PydanticAdapter(M).serialize(M(user_name="bob"))
    assert dumped == {"user_name": "bob", "count": 3}
    assert _round_trips(M, M(user_name="bob")).user_name == "bob"


@pytest.mark.parametrize(
    "field",
    [
        pydantic.Field(alias="userName"),
        pydantic.Field(serialization_alias="userName"),
        pydantic.Field(validation_alias="uname", serialization_alias="userName"),
    ],
    ids=["alias", "serialization_alias", "differing"],
)
def test_serialize_emits_exactly_the_keys_the_schema_advertises(field: Any) -> None:
    """The invariant underneath all of the above, stated directly: whatever the
    model is SHOWN is what a dumped value must be KEYED BY. Asserting the two
    agree catches a future change to either side, where the round-trip tests
    only catch a change that breaks both consistently."""

    class M(pydantic.BaseModel):
        user_name: str = field

    adapter = PydanticAdapter(M)
    schema_keys = set(adapter.json_schema()["properties"])
    dumped_keys = set(adapter.serialize(M.model_construct(user_name="bob")))

    assert dumped_keys == schema_keys


def test_extra_keys_are_not_dropped() -> None:
    """A model with `extra="allow"` keeps values the map has never heard of.
    Remapping must pass them through, not silently discard them."""

    class M(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")
        user_name: str = pydantic.Field(alias="userName")

    dumped = PydanticAdapter(M).serialize(M(userName="bob", spare=7))
    assert dumped["userName"] == "bob"
    assert dumped["spare"] == 7


def test_an_alias_choice_falls_back_to_the_field_name() -> None:
    """`AliasChoices` states several acceptable spellings, so there is no single
    "the" validation key. Guessing one branch would invent a contract the model
    never stated; the field name is what `populate_by_name` models accept."""

    class M(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(populate_by_name=True)
        user_name: str = pydantic.Field(
            validation_alias=pydantic.AliasChoices("uname", "userName")
        )

    assert PydanticAdapter(M).serialize(M(uname="bob")) == {"user_name": "bob"}
    assert _round_trips(M, M(uname="bob")).user_name == "bob"


def test_a_nested_model_with_its_own_differing_aliases_round_trips() -> None:
    """The regression my first attempt at this fix introduced, and which an
    existing nesting test caught: `model_dump()` emits FIELD names for nested
    models too, so remapping only the top level left the inner object keyed
    wrongly. The walk now follows the value tree, because only the value knows
    which model class produced each nested dict."""

    class Inner(pydantic.BaseModel):
        user_name: str = pydantic.Field(
            validation_alias="uname", serialization_alias="userName"
        )

    class Outer(pydantic.BaseModel):
        inner: Inner
        count: int = pydantic.Field(serialization_alias="theCount")

    adapter = PydanticAdapter(Outer)
    value = Outer(inner=Inner(uname="bob"), count=2)

    dumped = adapter.serialize(value)
    assert dumped == {"inner": {"uname": "bob"}, "count": 2}
    assert adapter.parse(json.dumps(dumped)) == value


def test_a_list_of_nested_models_round_trips() -> None:
    """Same walk, through a sequence — a shape `model_dump` flattens and a
    top-level-only remap would miss entirely."""

    class Item(pydantic.BaseModel):
        item_id: str = pydantic.Field(serialization_alias="itemId")

    class Bag(pydantic.BaseModel):
        items: list[Item]

    adapter = PydanticAdapter(Bag)
    value = Bag(items=[Item(item_id="a"), Item(item_id="b")])

    dumped = adapter.serialize(value)
    assert dumped == {"items": [{"item_id": "a"}, {"item_id": "b"}]}
    assert adapter.parse(json.dumps(dumped)) == value


def test_an_opaque_dict_field_is_not_rewritten() -> None:
    """A plain `dict` a model happens to hold is data, not a model. Remapping
    keys we do not own would corrupt it — so the walk passes it through."""

    class M(pydantic.BaseModel):
        payload: dict[str, Any]
        user_name: str = pydantic.Field(serialization_alias="userName")

    dumped = PydanticAdapter(M).serialize(M(payload={"user_name": 1, "odd": 2}, user_name="b"))
    assert dumped["payload"] == {"user_name": 1, "odd": 2}
    assert dumped["user_name"] == "b"
