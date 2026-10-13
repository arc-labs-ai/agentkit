import copy

import pytest

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.schema import SchemaError, clean_nulls, compile_schema, validate_output


@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.CODEX, Mode.ANTHROPIC_API])
@pytest.mark.parametrize(
    "keyword,value",
    [
        ("allOf", [{"type": "object"}]),
        ("not", {"type": "null"}),
        ("patternProperties", {"x": {"type": "string"}}),
        ("additionalProperties", True),
    ],
)
def test_unrepresentable_schema_refused_before_dispatch(mode, keyword, value):
    with pytest.raises(SchemaError):
        compile_schema({"type": "object", keyword: value}, mode)


@pytest.mark.parametrize("ref", ["https://example.invalid/schema", "#/$defs/missing"])
def test_unresolved_or_remote_references_are_not_fetched(ref):
    with pytest.raises(SchemaError):
        compile_schema({"type": "object", "properties": {"v": {"$ref": ref}}}, Mode.OPENAI_API)


def test_recursive_anthropic_schema_refused_and_claude_draft_references_rewritten():
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/node"}},
        "$defs": {"node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/node"}}}},
    }
    with pytest.raises(SchemaError, match="recursion"):
        compile_schema(schema, Mode.ANTHROPIC_API)
    compiled = compile_schema(schema, Mode.CLAUDE_CODE)
    assert compiled["properties"]["node"]["$ref"] == "#/definitions/node"
    assert "definitions" in compiled and "$defs" not in compiled
    assert compiled["$schema"].endswith("draft-07/schema#")


def test_compilation_is_nonmutating_and_canonical_bounds_still_enforced():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3}}, "required": ["n"]}
    original = copy.deepcopy(schema)
    compiled = compile_schema(schema, Mode.ANTHROPIC_API)
    assert schema == original
    assert "minimum" not in compiled["properties"]["n"]
    assert "minimum: 3" in compiled["properties"]["n"]["description"]
    with pytest.raises(Exception):
        validate_output({"n": 2}, schema)
    assert validate_output({"n": 3}, schema) == {"n": 3}


def test_anthropic_optional_field_and_strict_depth_budgets():
    schema = {"type": "object", "properties": {f"field{i}": {"type": "string"} for i in range(25)}}
    with pytest.raises(SchemaError, match="optional/union"):
        compile_schema(schema, Mode.ANTHROPIC_API)
    schema = {"type": "string"}
    for _ in range(11):
        schema = {"type": "object", "properties": {"child": schema}}
    with pytest.raises(SchemaError, match="ten object levels"):
        compile_schema(schema, Mode.OPENAI_API)


def test_strict_enum_and_schema_name_size_budgets():
    schema = {"type": "object", "properties": {"v": {"enum": list(range(1001))}}}
    with pytest.raises(SchemaError, match="size budgets"):
        compile_schema(schema, Mode.CODEX)


def test_null_cleaning_selects_valid_union_branches_and_preserves_required_nulls():
    schema = {
        "type": "array",
        "items": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "tag": {"const": "a"},
                        "optional": {"type": "string"},
                        "required_null": {"type": ["string", "null"]},
                    },
                    "required": ["tag", "required_null"],
                },
                {
                    "type": "object",
                    "properties": {"tag": {"const": "b"}, "optional": {"type": "string"}},
                    "required": ["tag"],
                },
            ]
        },
    }
    value = [{"tag": "a", "optional": None, "required_null": None}, {"tag": "b", "optional": None}]
    assert clean_nulls(value, schema) == [{"tag": "a", "required_null": None}, {"tag": "b"}]
    assert value[0]["optional"] is None


@pytest.mark.parametrize("mode", [Mode.OPENAI_API, Mode.CODEX, Mode.ANTHROPIC_API])
def test_canonical_exclusive_unions_compile_to_supported_anyof_and_validate_exclusivity(mode):
    schema = {
        "type": "object",
        "properties": {"v": {"oneOf": [{"type": "integer"}, {"type": "number"}]}},
        "required": ["v"],
    }
    compiled = compile_schema(schema, mode)
    assert "oneOf" not in compiled["properties"]["v"]
    assert compiled["properties"]["v"]["anyOf"] == [{"type": "integer"}, {"type": "number"}]
    with pytest.raises(Exception):
        validate_output({"v": 1}, schema)
    assert validate_output({"v": 1.5}, schema) == {"v": 1.5}


def test_literal_constant_compiles_as_enum():
    schema = {"type": "object", "properties": {"tag": {"const": "example"}}}
    compiled = compile_schema(schema, Mode.OPENAI_API)
    assert compiled["properties"]["tag"]["anyOf"][0]["enum"] == ["example"]
