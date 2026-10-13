"""Canonical output schemas and mode-specific compilation.

Compilation never mutates the caller's schema. Constraints weakened on the wire
are still checked against the canonical schema on return.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentkit.core.contracts.types import Mode


class SchemaError(ValueError):
    """A schema cannot be safely represented on the requested mode."""


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
    except ImportError as exc:
        raise ImportError('Structured calls require pip install "arc-agentkit[calls]"') from exc
    Draft202012Validator.check_schema(schema)

    def check_refs(node: Any) -> None:
        if isinstance(node, dict):
            if "$ref" in node:
                _resolve(node["$ref"], schema)
            for value in node.values():
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)

    check_refs(schema)
    return Draft202012Validator(schema)


def validate_schema(schema: dict[str, Any]) -> None:
    """Check the canonical schema before dispatch; validation dependencies are mandatory."""
    _validator(schema)


def _resolve(ref: str, root: dict[str, Any]) -> Any:
    if not ref.startswith("#/"):
        raise SchemaError("only local JSON Schema references are supported")
    node: Any = root
    try:
        for part in ref[2:].split("/"):
            node = node[part.replace("~1", "/").replace("~0", "~")]
    except (KeyError, TypeError) as exc:
        raise SchemaError(f"unresolved reference: {ref}") from exc
    return node


def compile_schema(schema: dict[str, Any], mode: Mode, model: str = "") -> dict[str, Any]:
    """Compile a canonical 2020-12 schema to a transport dialect.

    Model-specific schema extensions are deliberately conservative until a
    live probe establishes support. ``model`` is retained for future profiles.
    """
    schema = json.loads(json.dumps(schema))
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
                    raise SchemaError("Anthropic structured output cannot express recursion")
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
        if "$defs" in node:
            out["$defs"] = {k: visit(v, depth, refs) for k, v in node["$defs"].items()}
        if "enum" in node:
            enum_count += len(node["enum"])
            name_chars += sum(len(str(v)) for v in node["enum"])
        return out

    result: dict[str, Any] = visit(schema)
    if strict and (property_count > 5000 or enum_count > 1000 or name_chars > 120000):
        raise SchemaError("strict output exceeds schema size budgets")
    if mode == Mode.ANTHROPIC_API and (optional_count > 24 or union_count > 16):
        raise SchemaError("Anthropic structured output exceeds optional/union budgets")
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
