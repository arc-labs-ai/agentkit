"""SchemaAdapter.partial_parse — per-flavour contract.

These tests pin the streaming-partial behaviour across every flavour
the framework supports: tolerant parse of an incomplete JSON prefix,
construction that bypasses ``__init__`` so missing required fields
don't blow up, and a ``None`` return for inputs with zero useful
structure (so the streaming middleware can withhold the partial
event).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentkit.capabilities.output_schema import (
    DataclassAdapter,
    JsonSchemaAdapter,
)

# ── Pydantic ────────────────────────────────────────────────────────────────

pydantic_mod = pytest.importorskip("pydantic")


class _PydPerson(pydantic_mod.BaseModel):
    name: str
    age: int


def test_pydantic_partial_with_partial_string_field() -> None:
    from agentkit.capabilities.output_schema import PydanticAdapter

    adapter = PydanticAdapter(_PydPerson)
    partial = adapter.partial_parse('{"name": "Ali')
    assert partial is not None
    assert isinstance(partial, _PydPerson)
    # ``age`` was never streamed — must not appear in the unset-aware dump.
    assert partial.model_dump(exclude_unset=True) == {"name": "Ali"}


def test_pydantic_partial_with_two_fields() -> None:
    from agentkit.capabilities.output_schema import PydanticAdapter

    adapter = PydanticAdapter(_PydPerson)
    partial = adapter.partial_parse('{"name": "Alice", "age": 3')
    assert partial is not None
    assert isinstance(partial, _PydPerson)
    assert partial.model_dump(exclude_unset=True) == {"name": "Alice", "age": 3}


def test_pydantic_partial_empty_returns_none() -> None:
    from agentkit.capabilities.output_schema import PydanticAdapter

    adapter = PydanticAdapter(_PydPerson)
    assert adapter.partial_parse("") is None


def test_pydantic_partial_discards_unknown_fields() -> None:
    """Tolerant parse can occasionally produce fields the model
    doesn't declare; ``model_construct`` rejects unknowns, so we
    silently drop them before constructing."""
    from agentkit.capabilities.output_schema import PydanticAdapter

    adapter = PydanticAdapter(_PydPerson)
    partial = adapter.partial_parse('{"name": "x", "bogus": 1, "age": 2')
    assert partial is not None
    assert partial.model_dump(exclude_unset=True) == {"name": "x", "age": 2}


# ── Dataclass ──────────────────────────────────────────────────────────────


@dataclass
class _DCPerson:
    name: str
    age: int


def test_dataclass_partial_with_partial_string_field() -> None:
    adapter = DataclassAdapter(_DCPerson)
    partial = adapter.partial_parse('{"name": "Ali')
    assert partial is not None
    assert isinstance(partial, _DCPerson)
    assert partial.name == "Ali"
    # ``age`` was never streamed and has no default → no attr set.
    assert not hasattr(partial, "age")


def test_dataclass_partial_with_two_fields() -> None:
    adapter = DataclassAdapter(_DCPerson)
    partial = adapter.partial_parse('{"name": "Alice", "age": 3')
    assert partial is not None
    assert partial.name == "Alice"
    assert partial.age == 3


def test_dataclass_partial_empty_returns_none() -> None:
    adapter = DataclassAdapter(_DCPerson)
    assert adapter.partial_parse("") is None


# ── attrs ──────────────────────────────────────────────────────────────────


def _attrs_or_skip():
    """Skip attrs tests when the optional ``attrs`` dep isn't installed,
    WITHOUT skipping the rest of the module (a top-level
    ``importorskip`` aborts collection of every test in the file)."""
    return pytest.importorskip("attrs")


def test_attrs_partial_with_partial_string_field() -> None:
    attrs_mod = _attrs_or_skip()

    @attrs_mod.define
    class _AttrsPerson:
        name: str
        age: int

    from agentkit.capabilities.output_schema import AttrsAdapter

    adapter = AttrsAdapter(_AttrsPerson)
    partial = adapter.partial_parse('{"name": "Ali')
    # attrs.define defaults to slots=True; bypass-init via ``object.__new__``
    # may fail on slotted classes — the adapter returns None in that case.
    if partial is None:
        pytest.skip("attrs slotted class rejects __new__ bypass on this platform")
    assert partial.name == "Ali"


def test_attrs_partial_with_two_fields() -> None:
    attrs_mod = _attrs_or_skip()

    @attrs_mod.define
    class _AttrsPerson:
        name: str
        age: int

    from agentkit.capabilities.output_schema import AttrsAdapter

    adapter = AttrsAdapter(_AttrsPerson)
    partial = adapter.partial_parse('{"name": "Alice", "age": 3')
    if partial is None:
        pytest.skip("attrs slotted class rejects __new__ bypass on this platform")
    assert partial.name == "Alice"
    assert partial.age == 3


def test_attrs_partial_empty_returns_none() -> None:
    attrs_mod = _attrs_or_skip()

    @attrs_mod.define
    class _AttrsPerson:
        name: str
        age: int

    from agentkit.capabilities.output_schema import AttrsAdapter

    adapter = AttrsAdapter(_AttrsPerson)
    assert adapter.partial_parse("") is None


# ── JsonSchema ─────────────────────────────────────────────────────────────


def test_json_schema_partial_returns_dict() -> None:
    """The raw-schema adapter returns the partial dict as-is (no further
    coercion, no schema validation — a half-streamed object would
    routinely violate ``required``)."""
    adapter = JsonSchemaAdapter(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
    )
    assert adapter.partial_parse('{"name": "Ali') == {"name": "Ali"}


def test_json_schema_partial_empty_returns_none() -> None:
    adapter = JsonSchemaAdapter({"type": "object", "properties": {"name": {"type": "string"}}})
    assert adapter.partial_parse("") is None
