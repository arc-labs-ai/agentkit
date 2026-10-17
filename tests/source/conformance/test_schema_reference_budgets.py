"""Wire-schema budgets: references are locations, definitions are not instances."""

import copy
from typing import Literal

import pytest
from jsonschema import ValidationError
from pydantic import BaseModel, Field

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.schema import SchemaError, compile_schema, validate_output


class ProductRoute(BaseModel):
    route: str = ""
    role: Literal["core", "supporting", "optional", "uncertain"] = "uncertain"
    reason: str = ""


class RouteFamily(BaseModel):
    canonical: str = ""
    members: list[str] = Field(default_factory=list)
    rationale: str = ""


class Angle(BaseModel):
    topic: str = ""
    description: str = ""
    behind_auth: bool = False
    evidence: list[str] = Field(default_factory=list)
    intent_alignment: Literal["strong", "compatible", "conflicting"]
    alignment_reason: str
    route_claims: list[ProductRoute] = Field(default_factory=list)
    route_families: list[RouteFamily] = Field(default_factory=list)


class Answer(BaseModel):
    angles: list[Angle] = Field(default_factory=list)


def wire_counts(schema):
    """Independent oracle for the emitted object/array/union tree, no ref expansion.

    Count each emitted optional property and union-bearing schema location once.
    A definition is not a response location; these tests require a ref-free wire.
    """
    assert "$ref" not in schema and "$defs" not in schema and "definitions" not in schema
    properties = schema.get("properties", {})
    optional = len(set(properties) - set(schema.get("required", ())))
    unions = int("anyOf" in schema or isinstance(schema.get("type"), list))
    count = len(properties)
    children = list(properties.values())
    if "items" in schema:
        children.append(schema["items"])
    children.extend(schema.get("anyOf", ()))
    for child in children:
        child_optional, child_unions, child_count = wire_counts(child)
        optional += child_optional
        unions += child_unions
        count += child_count
    return optional, unions, count


def test_complete_reproduction_compiles_only_reachable_response_locations():
    wire = compile_schema(Answer.model_json_schema(), Mode.ANTHROPIC_API)
    # Answer: 1 optional; Angle: 6 optional + 2 required; each child object: 3 optional.
    assert wire_counts(wire) == (1 + 6 + 3 + 3, 0, 1 + 8 + 3 + 3)


def optional_object(count):
    return {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(count)}}


def union_object(count, *, both=False):
    field = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    if both:
        field["type"] = ["string", "null"]
    return {
        "type": "object",
        "properties": {f"f{i}": copy.deepcopy(field) for i in range(count)},
        "required": [f"f{i}" for i in range(count)],
    }


@pytest.mark.parametrize("definitions", ["$defs", "definitions"])
@pytest.mark.parametrize("reachable", [False, True])
def test_unused_definition_is_not_a_response_location_but_reachable_overflow_is(definitions, reachable):
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        definitions: {"large": optional_object(25)},
    }
    if reachable:
        schema["properties"]["value"] = {"$ref": f"#/{definitions}/large"}
        with pytest.raises(SchemaError, match="optional/union"):
            compile_schema(schema, Mode.ANTHROPIC_API)
    else:
        assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (0, 0, 1)


@pytest.mark.parametrize("width,allowed", [(12, True), (13, False)])
def test_shared_definition_is_counted_at_each_emitted_property(width, allowed):
    schema = {
        "type": "object",
        "properties": {name: {"$ref": "#/$defs/shared"} for name in ("left", "right")},
        "required": ["left", "right"],
        "$defs": {"shared": optional_object(width)},
    }
    if allowed:
        assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (24, 0, 2 + 24)
    else:
        with pytest.raises(SchemaError, match="optional/union"):
            compile_schema(schema, Mode.ANTHROPIC_API)


@pytest.mark.parametrize("width,allowed", [(8, True), (9, False)])
def test_shared_union_definition_is_counted_at_each_emitted_property(width, allowed):
    schema = {
        "type": "object",
        "properties": {name: {"$ref": "#/$defs/shared"} for name in ("left", "right")},
        "required": ["left", "right"],
        "$defs": {"shared": union_object(width)},
    }
    if allowed:
        assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (0, 16, 2 + 16)
    else:
        with pytest.raises(SchemaError, match="optional/union"):
            compile_schema(schema, Mode.ANTHROPIC_API)


@pytest.mark.parametrize("width,allowed", [(24, True), (25, False)])
def test_exact_optional_boundary_and_genuine_overflow(width, allowed):
    schema = optional_object(width)
    if allowed:
        assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (24, 0, 24)
    else:
        with pytest.raises(SchemaError, match="optional/union"):
            compile_schema(schema, Mode.ANTHROPIC_API)


@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize("width,allowed", [(16, True), (17, False)])
def test_union_parameter_counts_once_and_exact_boundary_is_preserved(both, width, allowed):
    schema = union_object(width, both=both)
    if allowed:
        assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (0, 16, 16)
    else:
        with pytest.raises(SchemaError, match="optional/union"):
            compile_schema(schema, Mode.ANTHROPIC_API)


@pytest.mark.parametrize("target", ["#/$defs/leaf", "#/$defs/missing", "#/$defs/mid"])
def test_nested_references_have_clean_unresolved_and_recursive_twins(target):
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/mid"}},
        "required": ["value"],
        "$defs": {"mid": {"type": "array", "items": {"$ref": target}}, "leaf": optional_object(2)},
    }
    original = copy.deepcopy(schema)
    if target == "#/$defs/leaf":
        assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (2, 0, 3)
    else:
        with pytest.raises(SchemaError):
            compile_schema(schema, Mode.ANTHROPIC_API)
    assert schema == original


@pytest.mark.parametrize("defect", [False, True])
def test_reproduction_keeps_canonical_constraints_and_does_not_mutate_input(defect):
    schema = Answer.model_json_schema()
    original = copy.deepcopy(schema)
    wire = compile_schema(schema, Mode.ANTHROPIC_API)
    assert wire_counts(wire) == (13, 0, 15)
    value = {"angles": [{"intent_alignment": "strong", "alignment_reason": "supported"}]}
    if defect:
        value["angles"][0]["intent_alignment"] = "invalid"
        with pytest.raises(ValidationError):
            validate_output(value, schema)
    else:
        assert validate_output(value, schema) == value
    assert schema == original


@pytest.mark.parametrize("mode", list(Mode))
def test_reproduction_compiles_each_transport_dialect_independently(mode):
    schema = Answer.model_json_schema()
    original = copy.deepcopy(schema)
    wire = compile_schema(schema, mode)
    if mode == Mode.ANTHROPIC_API:
        assert wire_counts(wire) == (13, 0, 15)
        assert "required" not in wire  # Never replace optional fields with required-nullable fields.
    elif mode == Mode.CLAUDE_CODE:
        assert "definitions" in wire and "$defs" not in wire
        assert wire["$schema"].endswith("draft-07/schema#")
    else:
        assert "$defs" in wire
        assert wire["required"] == ["angles"]
    assert schema == original


@pytest.mark.parametrize("width,allowed", [(24, True), (25, False)])
def test_anthropic_budget_does_not_apply_to_claude_code(width, allowed):
    schema = optional_object(width)
    wire = compile_schema(schema, Mode.CLAUDE_CODE)
    assert len(wire["properties"]) == width
    if not allowed:
        with pytest.raises(SchemaError):
            compile_schema(schema, Mode.ANTHROPIC_API)


@pytest.mark.parametrize("defect", [False, True])
def test_weakened_wire_bounds_still_fail_canonical_validation(defect):
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/bounded"}},
        "required": ["value"],
        "$defs": {"bounded": {"type": "integer", "minimum": 3}},
    }
    wire = compile_schema(schema, Mode.ANTHROPIC_API)
    assert "minimum" not in wire["properties"]["value"]
    if defect:
        with pytest.raises(ValidationError):
            validate_output({"value": 2}, schema)
    else:
        assert validate_output({"value": 3}, schema) == {"value": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["supported", "supported_model", "invalid_output", "overflow", "invalid", "unresolved"]
)
async def test_public_lifecycle_uses_scripted_transport_and_safe_local_failures(case):
    import json

    import httpx

    from agentkit.core.runtime import CallRuntime
    from agentkit.models import Binding, CallLimits, CredentialRef, OutputSpec
    from agentkit.models.providers.frontiers.calls import ApiAdapter

    requests = []
    resolutions = []
    events = []
    schema = Answer.model_json_schema()
    codes = {"overflow": "schema_budget_exceeded", "invalid": "invalid_schema", "unresolved": "unresolved_reference"}
    if case == "overflow":
        schema = optional_object(25)
    elif case == "invalid":
        schema = {"type": "planted-sensitive-schema-value"}
    elif case == "unresolved":
        schema = {"type": "object", "properties": {"value": {"$ref": "#/$defs/planted-sensitive-reference"}}}

    def credential():
        resolutions.append(True)
        return {"ANTHROPIC_API_KEY": "planted-credential-not-real"}

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["output_config"]["format"]["type"] == "json_schema"
        assert wire_counts(payload["output_config"]["format"]["schema"]) == (13, 0, 15)
        response_text = (
            '{"angles":[{"intent_alignment":"invalid","alignment_reason":"evidence"}]}'
            if case == "invalid_output"
            else '{"angles":[]}'
        )
        messages = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": response_text}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(event) + "\n\n" for event in messages),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with CallRuntime(
            adapters={Mode.ANTHROPIC_API: ApiAdapter(client=client)}, unpriced_policy="proceed_unpriced"
        ) as runtime:
            answer = await runtime.ask(
                Binding(Mode.ANTHROPIC_API, "test", CredentialRef(credential)),
                "planted-private-prompt",
                output=Answer if case == "supported_model" else OutputSpec(schema, max_corrections=0),
                limits=CallLimits(max_turns=1),
                call_id="schema-lifecycle",
                events=events.append,
            )
    if case in ("supported", "supported_model"):
        value = answer.value.model_dump() if case == "supported_model" else answer.value
        assert answer.outcome == "answered" and value == {"angles": []}
        assert len(requests) == len(resolutions) == answer.attempts == 1
    elif case == "invalid_output":
        assert answer.outcome == "failed" and answer.failure.classification == "invalid_request"
        assert answer.failure.mode_detail["reason"] == "canonical output validation failed"
        assert len(requests) == len(resolutions) == answer.attempts == 1
    else:
        assert answer.outcome == "failed" and answer.failure.classification == "invalid_request"
        assert answer.failure.retryable is False
        assert answer.failure.mode_detail["reason"] == "schema_compilation_failed"
        assert answer.failure.mode_detail["origin"] == "local"
        assert answer.failure.mode_detail["dispatch_state"] == "not_started"
        assert answer.failure.mode_detail["code"] == codes[case]
        assert not requests and not resolutions and answer.attempts == 0
        assert not any(event.kind == "attempt_started" for event in events)
        assert [event.kind for event in events] == ["call_started", "call_finished"]
        assert [event.sequence for event in events] == [1, 2]
        diagnostic = repr(answer.failure) + repr([event.fields for event in events])
        assert "planted-" not in diagnostic
        if case == "overflow":
            assert answer.failure.mode_detail["optional_count"] == 25
            assert answer.failure.mode_detail["optional_limit"] == 24


@pytest.mark.parametrize("defect", [False, True])
def test_reference_siblings_never_silently_replace_target_fields(defect):
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/value", "title": "A value"}},
        "required": ["value"],
        "$defs": {"value": {"type": "object", "properties": {"kept": {"type": "string"}}, "required": ["kept"]}},
    }
    original = copy.deepcopy(schema)
    if defect:
        schema["properties"]["value"]["properties"] = {"replacement": {"type": "string"}}
        original = copy.deepcopy(schema)
        with pytest.raises(SchemaError) as rejected:
            compile_schema(schema, Mode.ANTHROPIC_API)
        assert rejected.value.code == "unsupported_ref_siblings"
    else:
        wire = compile_schema(schema, Mode.ANTHROPIC_API)
        assert wire["properties"]["value"]["properties"] == {"kept": {"type": "string"}}
    assert schema == original


@pytest.mark.parametrize("defect", [False, True])
def test_definitions_are_not_removed_beneath_an_unexpanded_schema_reference(defect):
    schema = {
        "type": "object",
        "properties": {"values": {"type": "array", "items": {"type": "string"}}},
        "required": ["values"],
        "$defs": {"value": {"type": "string"}},
    }
    if defect:
        # `contains` is outside the compiler's supported reference expansion surface.
        schema["properties"]["values"]["contains"] = {"$ref": "#/$defs/value"}
        with pytest.raises(SchemaError) as rejected:
            compile_schema(schema, Mode.ANTHROPIC_API)
        assert rejected.value.code == "unsupported_reference_location"
    else:
        assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (0, 0, 1)


def test_unused_recursive_definition_is_not_expanded_or_counted():
    schema = {
        "type": "object",
        "properties": {},
        "$defs": {
            "unused": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/unused"}},
            }
        },
    }
    assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (0, 0, 0)


@pytest.mark.parametrize("annotation", ["default", "examples"])
def test_literal_annotation_data_is_neither_schema_budget_nor_a_reference(annotation):
    literal = {"$ref": "literal-not-a-reference", "type": "object", "properties": optional_object(25)["properties"]}
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "type": "object",
                "properties": {},
                annotation: [literal] if annotation == "examples" else literal,
            }
        },
        "required": ["value"],
    }
    original = copy.deepcopy(schema)
    assert wire_counts(compile_schema(schema, Mode.ANTHROPIC_API)) == (0, 0, 1)
    assert schema == original
