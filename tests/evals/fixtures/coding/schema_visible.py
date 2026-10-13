import copy

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.schema import compile_schema


def test_exclusive_union_is_lowered_to_supported_provider_wire_dialect() -> None:
    canonical = {
        "type": "object",
        "properties": {
            "result": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "ok"}, "value": {"type": "integer"}},
                        "required": ["kind", "value"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "error"}, "reason": {"type": "string"}},
                        "required": ["kind", "reason"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
        "required": ["result"],
        "additionalProperties": False,
    }
    snapshot = copy.deepcopy(canonical)
    wire = compile_schema(canonical, Mode.OPENAI_API)
    assert "oneOf" not in wire["properties"]["result"]
    assert len(wire["properties"]["result"]["anyOf"]) == 2
    assert canonical == snapshot
