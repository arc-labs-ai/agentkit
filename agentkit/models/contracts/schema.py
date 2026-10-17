"""Canonical output schemas and mode-specific compilation.

Compilation never mutates the caller's schema. Constraints weakened on the wire
are still checked against the canonical schema on return.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentkit.core.contracts.types import Mode


class SchemaError(ValueError):
    """A schema cannot be safely represented on the requested mode."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "unsupported_schema",
        details: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _schema_nodes(schema: Any) -> Iterator[dict[str, Any]]:
    """Walk schema locations, never annotations or literal default/enum values."""
    if not isinstance(schema, dict):
        return
    yield schema
    for keyword in ("properties", "patternProperties", "dependentSchemas", "$defs", "definitions"):
        for child in schema.get(keyword, {}).values():
            yield from _schema_nodes(child)
    for keyword in (
        "items",
        "additionalProperties",
        "unevaluatedProperties",
        "additionalItems",
        "unevaluatedItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    ):
        child = schema.get(keyword)
        for item in child if isinstance(child, list) else (child,):
            yield from _schema_nodes(item)
    for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for child in schema.get(keyword, ()):
            yield from _schema_nodes(child)


@dataclass(frozen=True)
class OutputSpec:
    schema: dict[str, Any] = field(hash=False)
    name: str = "answer"
    max_corrections: int = 1
    validators: tuple[Callable[[Any], list[str]], ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        from agentkit.core.serialization._immutable import deep_freeze

        object.__setattr__(self, "schema", deep_freeze(self.schema))
        object.__setattr__(self, "validators", tuple(self.validators))
        if self.max_corrections < 0:
            raise ValueError("max_corrections must be nonnegative")
        if any(not callable(validator) for validator in self.validators):
            raise TypeError("output validators must be callable")


def tolerant_json_object(text: str) -> Any:
    """Extract one unambiguous complete top-level JSON object from a response."""
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate, flags=re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    starts: list[int] = []
    depth = 0
    quoted = escaped = False
    spans: list[tuple[int, int]] = []
    for index, char in enumerate(candidate):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            if depth == 0:
                starts.append(index)
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append((starts.pop(), index + 1))
    if depth == 1 and starts and candidate.endswith("}"):
        # A single stray close brace after the last complete object.
        candidate = candidate[:-1].rstrip()
        depth = 0
        spans = []
        starts = []
        quoted = escaped = False
        for index, char in enumerate(candidate):
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                if depth == 0:
                    starts.append(index)
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0:
                    spans.append((starts.pop(), index + 1))
    if len(spans) != 1:
        raise ValueError("response does not contain one unambiguous JSON object")
    start, end = spans[0]
    return json.loads(candidate[start:end])


def _validator(schema: Any) -> Any:
    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
        from jsonschema.exceptions import SchemaError as InvalidSchema  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError('Structured calls require pip install "arc-agentkit[calls]"') from exc
    try:
        Draft202012Validator.check_schema(schema)
    except InvalidSchema as exc:
        raise SchemaError("invalid canonical JSON Schema", code="invalid_schema") from exc

    for node in _schema_nodes(schema):
        if "$ref" in node:
            _resolve(node["$ref"], schema)
    return Draft202012Validator(schema)


def validate_schema(schema: dict[str, Any]) -> None:
    """Check the canonical schema before dispatch; validation dependencies are mandatory."""
    _validator(schema)


def _resolve(ref: str, root: dict[str, Any]) -> Any:
    if not ref.startswith("#/"):
        raise SchemaError("only local JSON Schema references are supported", code="unresolved_reference")
    node: Any = root
    try:
        for part in ref[2:].split("/"):
            node = node[part.replace("~1", "/").replace("~0", "~")]
    except (KeyError, TypeError) as exc:
        raise SchemaError(f"unresolved reference: {ref}", code="unresolved_reference") from exc
    return node


def compile_schema(schema: dict[str, Any], mode: Mode, model: str = "") -> dict[str, Any]:
    """Compile a canonical 2020-12 schema to a transport dialect.

    Model-specific schema extensions are deliberately conservative until a
    live probe establishes support. ``model`` is retained for future profiles.
    """
    try:
        schema = json.loads(json.dumps(schema))
    except (TypeError, ValueError) as exc:
        raise SchemaError("schema must be JSON serializable", code="invalid_schema") from exc
    _validator(schema)
    strict = mode in (Mode.CODEX, Mode.OPENAI_API)
    if strict and (schema.get("type") != "object" or "anyOf" in schema or "oneOf" in schema):
        raise SchemaError("OpenAI strict output requires an object root without unions")
    forbidden = {
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "unevaluatedProperties",
        "patternProperties",
        "prefixItems",
        "$dynamicRef",
    }
    bounds = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "format",
    }
    optional_count = union_count = property_count = enum_count = name_chars = 0

    def visit(node: Any, depth: int = 0, refs: tuple[str, ...] = ()) -> Any:
        nonlocal optional_count, union_count, property_count, enum_count, name_chars
        if not isinstance(node, dict):
            return copy.deepcopy(node)
        if mode != Mode.CLAUDE_CODE and forbidden.intersection(node):
            raise SchemaError(f"{mode} cannot express {sorted(forbidden.intersection(node))}")
        if "$ref" in node:
            ref = node["$ref"]
            target = _resolve(ref, schema)
            if mode == Mode.ANTHROPIC_API:
                if ref in refs:
                    raise SchemaError("Anthropic structured output cannot express recursion", code="recursive_schema")
                if not isinstance(target, dict):
                    raise SchemaError("reference target must be a schema object")
                annotations = {"title", "description", "default", "examples", "$comment", "$defs", "definitions"}
                if any(k in target and target[k] != v for k, v in node.items() if k not in annotations | {"$ref"}):
                    raise SchemaError(
                        "conflicting reference siblings require explicit constraint intersection",
                        code="unsupported_ref_siblings",
                    )
                return visit({**target, **{k: v for k, v in node.items() if k != "$ref"}}, depth, (*refs, ref))
        out = copy.deepcopy(node)
        notes: list[str] = []
        if mode != Mode.CLAUDE_CODE:
            if "oneOf" in out and "anyOf" in out:
                raise SchemaError("combined oneOf/anyOf composition cannot be represented safely")
            if "const" in out:
                out["enum"] = [out.pop("const")]
            if "oneOf" in out:
                out.pop("oneOf")
                notes.append("Match exactly one alternative; canonical validation enforces exclusivity")
            for key in bounds.intersection(out):
                if key == "minItems" and mode == Mode.ANTHROPIC_API and out[key] in (0, 1):
                    continue
                notes.append(f"{key}: {json.dumps(out.pop(key))}")
        if notes:
            out["description"] = (str(out.get("description", "")) + " Constraints: " + "; ".join(notes)).strip()
        properties = node.get("properties")
        if node.get("type") == "object" or properties is not None:
            if mode != Mode.CLAUDE_CODE and node.get("additionalProperties") not in (None, False):
                raise SchemaError(f"{mode} cannot express open object properties")
            props = properties or {}
            required = node.get("required", [])
            property_count += len(props)
            name_chars += sum(len(k) for k in props)
            optional_count += len(set(props) - set(required))
            if strict and depth >= 10:
                raise SchemaError("strict output exceeds ten object levels")
            compiled = {}
            for key, value in props.items():
                child = visit(value, depth + 1, refs)
                if strict and key not in required:
                    child = {"anyOf": [child, {"type": "null"}]}
                compiled[key] = child
            out["properties"] = compiled
            if mode != Mode.CLAUDE_CODE:
                out["additionalProperties"] = False
            if strict:
                out["required"] = list(props)
        for key in ("anyOf", "oneOf", "allOf"):
            if key in node:
                union_count += 1
                wire_key = "anyOf" if key == "oneOf" and mode != Mode.CLAUDE_CODE else key
                out[wire_key] = [visit(v, depth, refs) for v in node[key]]
        if isinstance(node.get("type"), list):
            union_count += 1
        if "items" in node:
            out["items"] = visit(node["items"], depth, refs)
        if mode == Mode.ANTHROPIC_API:
            # Every supported reference was expanded at its response location.
            # Retaining definitions would send and count redundant schema trees.
            out.pop("$defs", None)
            out.pop("definitions", None)
        elif "$defs" in node:
            out["$defs"] = {k: visit(v, depth, refs) for k, v in node["$defs"].items()}
        if "enum" in node:
            enum_count += len(node["enum"])
            name_chars += sum(len(str(v)) for v in node["enum"])
        return out

    result: dict[str, Any] = visit(schema)
    if strict and (property_count > 5000 or enum_count > 1000 or name_chars > 120000):
        raise SchemaError("strict output exceeds schema size budgets")
    if mode == Mode.ANTHROPIC_API:
        # Evaluate the actual wire tree, not canonical definitions or visit history.
        optional_count = union_count = property_count = 0
        for node in _schema_nodes(result):
            if {"$ref", "$defs", "definitions"}.intersection(node):
                raise SchemaError(
                    "reference or definition in an unsupported schema location cannot be expanded safely",
                    code="unsupported_reference_location",
                )
            properties = node.get("properties", {})
            property_count += len(properties)
            optional_count += len(set(properties) - set(node.get("required", ())))
            union_count += int("anyOf" in node or isinstance(node.get("type"), list))
        if optional_count > 24 or union_count > 16:
            raise SchemaError(
                "Anthropic structured output exceeds optional/union budgets",
                code="schema_budget_exceeded",
                details={
                    "optional_count": optional_count,
                    "optional_limit": 24,
                    "union_count": union_count,
                    "union_limit": 16,
                    "property_count": property_count,
                },
            )
    if mode == Mode.CLAUDE_CODE:
        # Reject 2020-12 constructs with no equivalent rather than mislabelling
        # them draft-07. Definitions and local references have a direct mapping.
        def draft(node: Any) -> Any:
            if isinstance(node, list):
                return [draft(x) for x in node]
            if not isinstance(node, dict):
                return node
            if {"prefixItems", "unevaluatedProperties", "$dynamicRef"}.intersection(node):
                raise SchemaError("schema uses constructs unavailable in draft-07")
            return {
                ("definitions" if k == "$defs" else k): v.replace("#/$defs/", "#/definitions/")
                if k == "$ref"
                else draft(v)
                for k, v in node.items()
                if k != "$schema"
            }

        result = draft(result)
        result["$schema"] = "http://json-schema.org/draft-07/schema#"
    return result


def clean_nulls(value: Any, schema: dict[str, Any]) -> Any:
    """Remove strict-mode placeholder nulls only for non-nullable optional fields.

    Union branches are chosen by canonical validation *after* cleaning. No
    matching branch leaves the value intact for the caller's validator.
    """
    validator = _validator(schema)

    def valid(v: Any, node: Any) -> bool:
        return bool(validator.evolve(schema=node).is_valid(v))

    def clean(v: Any, node: Any, depth: int = 0) -> Any:
        if depth > 256:
            raise SchemaError("structured value nesting exceeds 256 levels")
        if not isinstance(node, dict):
            return copy.deepcopy(v)
        if "$ref" in node:
            node = {**_resolve(node["$ref"], schema), **{k: x for k, x in node.items() if k != "$ref"}}
        for union in ("anyOf", "oneOf"):
            if union in node:
                matches = []
                base = {k: x for k, x in node.items() if k != union}
                for branch in node[union]:
                    candidate = clean(v, {**base, **branch}, depth + 1)
                    if valid(candidate, {**base, **branch}):
                        matches.append(candidate)
                if matches and (union == "anyOf" or len(matches) == 1):
                    return matches[0]
                return copy.deepcopy(v)
        if isinstance(v, dict):
            props = node.get("properties", {})
            required = node.get("required", [])
            out = {}
            for key, item in v.items():
                child = props.get(key, node.get("additionalProperties", {}))
                if item is None and key in props and key not in required and not valid(None, child):
                    continue
                out[key] = clean(item, child, depth + 1)
            return out
        if isinstance(v, list):
            return [clean(item, node.get("items", {}), depth + 1) for item in v]
        return copy.deepcopy(v)

    return clean(value, schema)


def validate_output(value: Any, schema: dict[str, Any]) -> Any:
    """Clean and validate a response against the canonical schema."""
    cleaned = clean_nulls(value, schema)
    _validator(schema).validate(cleaned)
    return cleaned
