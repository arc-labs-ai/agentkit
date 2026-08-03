"""``adapt()`` — the single dispatcher that picks an adapter for any supported
output spec. Order matters: Pydantic and attrs come before the stdlib-dataclass
catch-all, raw JSON Schema is the last resort, everything else raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.capabilities.output_schema import (
    AttrsAdapter,
    DataclassAdapter,
    JsonSchemaAdapter,
    PydanticAdapter,
    adapt,
)


@dataclass
class _Plain:
    x: int


def test_dispatch_to_dataclass_adapter():
    a = adapt(_Plain)
    assert isinstance(a, DataclassAdapter)
    assert a.python_type is _Plain


def test_dispatch_to_json_schema_adapter():
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    a = adapt(schema)
    assert isinstance(a, JsonSchemaAdapter)


def test_dispatch_to_pydantic_adapter():
    pydantic = pytest.importorskip("pydantic")

    class M(pydantic.BaseModel):
        x: int

    a = adapt(M)
    assert isinstance(a, PydanticAdapter)
    assert a.python_type is M


def test_pydantic_beats_dataclass_when_both_apply():
    """Pydantic BaseModel uses @dataclass_transform; in some pydantic builds it
    would also pass ``dataclasses.is_dataclass``. The dispatcher MUST probe
    pydantic first so the more specific adapter wins."""
    pydantic = pytest.importorskip("pydantic")

    class M(pydantic.BaseModel):
        x: int

    a = adapt(M)
    assert isinstance(a, PydanticAdapter)
    assert not isinstance(a, DataclassAdapter)


def test_dispatch_to_attrs_adapter():
    attrs = pytest.importorskip("attrs")

    @attrs.define
    class A:
        x: int

    a = adapt(A)
    assert isinstance(a, AttrsAdapter)
    assert a.python_type is A


def test_dispatch_int_raises_type_error():
    with pytest.raises(TypeError):
        adapt(int)


def test_dispatch_string_raises_type_error():
    with pytest.raises(TypeError):
        adapt("string")


def test_dispatch_arbitrary_dict_without_object_type_raises():
    """A dict that doesn't look like a JSON Schema object must NOT be silently
    accepted — the user almost certainly made a mistake."""
    with pytest.raises(TypeError):
        adapt({"foo": "bar"})


def test_dispatch_array_top_level_schema_raises():
    """We only support top-level object schemas (provider structured-output
    modes expect that). An array-top schema should fall through to the
    catch-all TypeError."""
    with pytest.raises(TypeError):
        adapt({"type": "array", "items": {"type": "string"}})


def test_dispatch_passes_name_through():
    a = adapt(_Plain, name="my_plain")
    assert a.name == "my_plain"


def test_msgspec_struct_raises_not_implemented_when_installed():
    """The dispatcher detects msgspec.Struct and explicitly rejects it (rather
    than falling through to TypeError) so the deferred-support message is
    actionable. Skipped when msgspec isn't installed — the catch-all TypeError
    path is fine in that case."""
    msgspec = pytest.importorskip("msgspec")

    class S(msgspec.Struct):
        x: int

    with pytest.raises(NotImplementedError):
        adapt(S)
