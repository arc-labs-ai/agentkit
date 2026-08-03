"""PydanticAdapter — adapter over a ``pydantic.BaseModel`` subclass.

Why a dedicated adapter even though Pydantic already speaks JSON Schema: provider
structured-output modes have *opinions* about the schema they accept (OpenAI's
strict mode rejects ``default`` on required fields, Anthropic's tool-schema mode
expects the top-level ``type: object`` and rejects ``$ref`` to definitions outside
the document, etc.). The adapter is the single place those provider-fitting tweaks
get applied, so the rest of the framework can treat *any* output flavour identically.

Pydantic is an OPTIONAL dependency. The import lives inside the methods so this
module loads cleanly in an environment without pydantic installed — the
``adapt()`` dispatcher will simply never construct one. The class-level
``__init__`` re-imports inside the constructor so the failure mode is a clean
``ImportError`` at the call site that needed it, not a confusing ``NameError``
deep inside ``parse()``.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

from agentkit.capabilities.output_schema.protocol import OutputCoercionError

T = TypeVar("T")


def _strip_markdown_fences(raw: str) -> str:
    """Strip a leading ``` ```json ``` (or bare ``` ``` ```) fence + matching
    closing fence from a string, if present.

    LLMs frequently wrap their structured output in markdown fences even when
    the prompt asks for raw JSON — Claude's instruction-following bias toward
    "render as code" wins over the schema hint at the API layer. Stripping at
    the adapter boundary is defense-in-depth: valid JSON cannot start with a
    backtick, so if the trimmed input does, it's a fence we can safely peel
    off. Falls through unchanged on any malformed/no-fence input — let the
    JSON parser surface the real error from the raw payload.
    """
    s = raw.strip()
    if not s.startswith("```"):
        return raw
    nl = s.find("\n")
    if nl == -1:
        return raw
    body = s[nl + 1 :]
    end = body.rfind("```")
    if end == -1:
        return raw
    return body[:end].rstrip()


class PydanticAdapter(Generic[T]):
    """Adapter over a ``pydantic.BaseModel`` subclass.

    Constructor takes the model class (not an instance). The adapter is stateless
    beyond a reference to that class — safe to share across runs and across
    threads/tasks (Pydantic's ``model_validate*`` are pure functions of input)."""

    def __init__(self, model: type[T], *, name: str | None = None) -> None:
        # Re-check pydantic is importable here so the failure mode is a clear
        # ImportError at construction, not later inside parse(). Keeping the import
        # local also means `import output_schema.pydantic_adapter` works on a box
        # without pydantic — only INSTANTIATING the adapter requires it.
        try:
            from pydantic import BaseModel
        except ImportError as e:  # pragma: no cover — exercised in environments without pydantic
            raise ImportError("PydanticAdapter requires `pydantic` to be installed") from e
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(f"PydanticAdapter expects a pydantic.BaseModel subclass, got {model!r}")
        self._model = model
        self.python_type: type[T] = model
        """The BaseModel subclass itself — used for ``AgentResult[T]`` stamping and
        ``isinstance(result.output, Model)`` checks at the boundary."""
        self.name: str = name or model.__name__
        """Defaults to the class ``__name__`` (e.g. ``"WeatherReport"``); the caller
        can override when the model name would clash with another tool name or when
        a more human-readable label is wanted in the system-prompt fallback."""

    def json_schema(self) -> dict[str, Any]:
        """Pydantic's own ``model_json_schema`` is the source of truth. We adjust for
        provider "strict" expectations: OpenAI's structured-output strict mode
        forbids ``default`` keys on required fields (it treats them as a hint that
        the field is optional). Dropping the key when the field is in ``required``
        keeps the schema valid for both strict and non-strict consumers."""
        # pydantic v2 BaseModel; not type-checked here since pydantic is an optional dep.
        schema: dict[str, Any] = cast(Any, self._model).model_json_schema()
        required = set(schema.get("required", []))
        for field_name, field_schema in (schema.get("properties") or {}).items():
            if field_name in required and isinstance(field_schema, dict):
                field_schema.pop("default", None)
        return schema

    def parse(self, raw: str | dict[str, Any]) -> T:
        """Dispatch on input type: a string goes through ``model_validate_json``
        (which parses + validates in one pass, surfacing JSON syntax errors with
        line/column context), a dict goes through ``model_validate`` (the
        provider already parsed the JSON for us in structured-output mode).

        String inputs are first passed through :func:`_strip_markdown_fences` —
        Claude and other instruction-tuned models routinely wrap JSON output in
        ``` ```json … ``` ``` fences even when the prompt asks for raw JSON.
        Valid JSON can never start with a backtick, so unconditional stripping
        is loss-less.

        Every Pydantic ``ValidationError`` is wrapped in :class:`OutputCoercionError`
        so the retry middleware has a single exception type to catch. The original
        per-field errors are flattened to strings — preserving the dotted-path
        location (``"address.city: field required"``) which is exactly what we want
        to reflect back to the model."""
        from pydantic import ValidationError

        try:
            model_any = cast(Any, self._model)
            if isinstance(raw, str):
                parsed: T = model_any.model_validate_json(_strip_markdown_fences(raw))
                return parsed
            validated: T = model_any.model_validate(raw)
            return validated
        except ValidationError as e:
            errors = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
            raise OutputCoercionError(
                f"failed to coerce response into {self.name}: {len(errors)} error(s)",
                raw=raw,
                errors=errors,
            ) from e

    def partial_parse(self, raw: str) -> T | None:
        """Tolerant parse → construct a partial model from whatever fields
        have already arrived.

        Uses ``model_construct(**data)`` rather than ``model_validate`` so
        missing required fields don't fail — the partial is intentionally
        under-specified. On any failure (including incomplete JSON our
        tolerant parser can't make sense of) returns ``None`` and the
        caller pumps another delta. Never raises.

        Side note on equality: Pydantic ``BaseModel`` instances compare
        by field values, so the streaming middleware can cheaply
        deduplicate identical partials with ``new == last``.
        """
        from agentkit.capabilities.output_schema._partial_json import parse_partial

        try:
            data = parse_partial(raw)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        try:
            # ``model_construct`` only accepts known field names; silently
            # discard any extras the partial parser might have produced from
            # a malformed prefix.
            model_any = cast(Any, self._model)
            field_names = set(model_any.model_fields)
            cleaned = {k: v for k, v in data.items() if k in field_names}
            constructed: T = model_any.model_construct(**cleaned)
            return constructed
        except Exception:
            return None

    def serialize(self, value: T) -> dict[str, Any]:
        """``model_dump`` returns plain Python types (dicts, lists, primitives) — no
        Pydantic-specific objects leak into the durable store or transcript. That's
        the round-trip guarantee: `parse(json.dumps(serialize(v))) == v`."""
        dumped: dict[str, Any] = value.model_dump()  # type: ignore[attr-defined]
        return dumped

    def validate(self, value: Any) -> T:
        """Validate an arbitrary Python value against the model.

        Fast-paths an already-typed instance (``isinstance(value, self._model)``)
        — the common case when a tool function literally constructs and
        returns a ``MyModel(...)``. Otherwise hands off to ``model_validate``
        (Pydantic accepts both dicts and dataclass-like / attrs-like objects),
        wrapping any ``ValidationError`` in :class:`OutputCoercionError` so the
        ``ToolShapeError`` fire site has a single exception type to catch."""
        from pydantic import ValidationError

        if isinstance(value, self._model):
            return value
        try:
            validated: T = cast(Any, self._model).model_validate(value)
            return validated
        except ValidationError as e:
            errors = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
            raw_for_error: Any = value if isinstance(value, dict | str) else repr(value)
            raise OutputCoercionError(
                f"failed to validate value into {self.name}: {len(errors)} error(s)",
                raw=raw_for_error,
                errors=errors,
            ) from e
        except (TypeError, AttributeError) as e:
            raw_for_error = value if isinstance(value, dict | str) else repr(value)
            raise OutputCoercionError(
                f"value of type {type(value).__name__} is not coercible into {self.name}",
                raw=raw_for_error,
                errors=[f"top-level: {type(value).__name__} not coercible into {self.name}"],
            ) from e
