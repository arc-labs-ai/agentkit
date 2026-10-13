"""Both agent interfaces use canonical schema rules and exact optional defaults."""

import builtins
from dataclasses import dataclass

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.schema import SchemaError, compile_schema
from agentkit.models.output import DataclassAdapter, JsonSchemaAdapter, OutputCoercionError, PydanticAdapter


def test_raw_schema_requires_validation_dependency_before_dispatch(monkeypatch):
    original = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError("validation deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match="Structured calls require"):
        JsonSchemaAdapter({"type": "object", "properties": {"count": {"type": "integer"}}})


def test_raw_schema_rejects_external_refs_before_any_model_or_network_request():
    with pytest.raises(SchemaError, match="only local"):
        JsonSchemaAdapter({"type": "object", "properties": {"private": {"$ref": "https://example.invalid/schema"}}})


def test_terminal_exclusive_union_retains_original_raw_diagnostics():
    adapter = JsonSchemaAdapter(
        {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [{"type": "integer"}, {"type": "number"}],
                }
            },
            "required": ["value"],
        }
    )
    raw = '{"value": 2}'
    with pytest.raises(OutputCoercionError) as failure:
        adapter.parse(raw)
    assert failure.value.raw == raw and failure.value.errors
    assert "oneOf" in adapter.json_schema()["properties"]["value"]
    assert "anyOf" in compile_schema(adapter.json_schema(), Mode.OPENAI_API, "test")["properties"]["value"]


@pytest.mark.parametrize("flavour", ["dataclass", "pydantic", "attrs"])
def test_strict_native_optional_null_placeholder_restores_default_without_weakening_nullable_fields(flavour):
    if flavour == "dataclass":

        @dataclass
        class Receipt:
            title: str
            label: str = "default-label"
            note: str | None = None

        adapter = DataclassAdapter(Receipt)
    elif flavour == "pydantic":
        pydantic = pytest.importorskip("pydantic")

        class Receipt(pydantic.BaseModel):
            title: str
            label: str = "default-label"
            note: str | None = None

        adapter = PydanticAdapter(Receipt)
    else:
        attrs = pytest.importorskip("attrs")
        from agentkit.models.output import AttrsAdapter

        @attrs.define
        class Receipt:
            title: str
            label: str = "default-label"
            note: str | None = None

        adapter = AttrsAdapter(Receipt)
    wire = compile_schema(adapter.json_schema(), Mode.OPENAI_API, "test")
    assert set(wire["required"]) == {"title", "label", "note"}
    parsed = adapter.parse({"title": "done", "label": None, "note": None})
    assert parsed.label == "default-label" and parsed.note is None
