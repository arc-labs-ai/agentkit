"""JsonSchemaAdapter — the raw-dict escape hatch:

1. A bare object-schema dict is accepted.
2. ``parse(valid_json)`` returns the parsed dict.
3. ``parse("not json")`` raises :class:`OutputCoercionError`.
4. ``serialize`` is identity (the value already IS a dict).
"""

from __future__ import annotations

import json

import pytest

from agentkit.capabilities.output_schema import (
    JsonSchemaAdapter,
    OutputCoercionError,
)

_SCHEMA = {
    "type": "object",
    "title": "PersonSchema",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_constructor_accepts_object_schema():
    adapter = JsonSchemaAdapter(_SCHEMA)
    assert adapter.json_schema() is _SCHEMA  # pass-through, no copy


def test_constructor_rejects_non_object_schema():
    with pytest.raises(TypeError):
        JsonSchemaAdapter({"type": "array", "items": {"type": "string"}})


def test_constructor_rejects_non_dict_input():
    with pytest.raises(TypeError):
        JsonSchemaAdapter("not a dict")  # type: ignore[arg-type]


def test_name_defaults_to_title_then_output():
    assert JsonSchemaAdapter(_SCHEMA).name == "PersonSchema"
    titleless = {"type": "object", "properties": {}}
    assert JsonSchemaAdapter(titleless).name == "Output"


def test_name_override():
    assert JsonSchemaAdapter(_SCHEMA, name="custom").name == "custom"


def test_python_type_is_dict():
    """Raw-schema flavour intentionally does not stamp a Python class — the
    type instances are coerced into is ``dict``."""
    assert JsonSchemaAdapter(_SCHEMA).python_type is dict


def test_parse_valid_returns_dict():
    adapter = JsonSchemaAdapter(_SCHEMA)
    payload = {"name": "alice", "age": 30}
    result = adapter.parse(json.dumps(payload))
    assert result == payload
    assert isinstance(result, dict)


def test_parse_accepts_dict_input():
    adapter = JsonSchemaAdapter(_SCHEMA)
    payload = {"name": "alice", "age": 30}
    assert adapter.parse(payload) == payload


def test_parse_invalid_json_raises_coercion_error():
    adapter = JsonSchemaAdapter(_SCHEMA)
    with pytest.raises(OutputCoercionError) as ei:
        adapter.parse("not json at all")
    assert ei.value.errors  # diagnostic should be populated


def test_parse_non_object_payload_raises():
    adapter = JsonSchemaAdapter(_SCHEMA)
    with pytest.raises(OutputCoercionError):
        adapter.parse(json.dumps([1, 2, 3]))


def test_serialize_is_identity():
    adapter = JsonSchemaAdapter(_SCHEMA)
    payload = {"name": "alice", "age": 30}
    assert adapter.serialize(payload) is payload


def test_partial_parse_returns_partial_dict():
    """Streaming partial parse: tolerant JSON close yields the dict as-is,
    no further schema validation (a half-streamed object would routinely
    violate ``required``)."""
    adapter = JsonSchemaAdapter(_SCHEMA)
    partial = adapter.partial_parse('{"name":"par')
    assert partial == {"name": "par"}


def test_partial_parse_empty_is_none():
    adapter = JsonSchemaAdapter(_SCHEMA)
    assert adapter.partial_parse("") is None
    assert adapter.partial_parse("   ") is None
