import importlib

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.schema import compile_schema, validate_output

ValidationError: type[Exception] = importlib.import_module("jsonschema").ValidationError


@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.CODEX, Mode.ANTHROPIC_API])
def test_provider_compilation_lowers_nested_exclusive_union(mode: Mode) -> None:
    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {
                    "oneOf": [{"type": "integer"}, {"type": "string"}],
                },
            }
        },
        "required": ["values"],
        "additionalProperties": False,
    }
    wire = compile_schema(schema, mode)
    union = wire["properties"]["values"]["items"]
    assert "oneOf" not in union and len(union["anyOf"]) == 2
    assert "exactly one" in union["description"].lower()
    assert validate_output({"values": [4, "ready"]}, schema) == {"values": [4, "ready"]}


def test_claude_draft7_keeps_exclusivity_and_canonical_ambiguity_is_rejected() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [{"type": "integer"}, {"type": "number"}],
            }
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    wire = compile_schema(schema, Mode.CLAUDE_CODE)
    assert "oneOf" in wire["properties"]["value"]
    assert "anyOf" not in wire["properties"]["value"]
    with pytest.raises(ValidationError):
        validate_output({"value": 4}, schema)
    assert validate_output({"value": 4.5}, schema) == {"value": 4.5}
