"""SchemaAdapter — the uniform bridge between "user declared *this* output shape on
their Agent" and "the framework can ship the matching JSON Schema to the LLM and
coerce the response back into a typed instance."

Why a Protocol and not a base class: the four supported flavours (Pydantic BaseModel,
stdlib dataclass, attrs class, raw JSON Schema dict) have *nothing* in common at the
Python level — Pydantic owns `.model_validate`, dataclasses are walked via
`dataclasses.fields()`, attrs has `attrs.fields()`, and raw schemas are just dicts. A
nominal base class would force every adapter to inherit-and-stub, while structural
typing keeps each adapter self-contained and the surface area honest: the *only*
things the rest of the framework is allowed to lean on are `name`, `python_type`,
`json_schema()`, `parse()`, `partial_parse()`, and `serialize()`.

The methods are intentionally tiny. Anything beyond this (system-prompt rendering,
provider-specific structured-output mode, retry-on-coercion-failure, streaming
buffer management) lives at the Agent / middleware layer; the adapter is *just*
the bridge — the round trip "schema out, instance in" — and nothing else.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class SchemaAdapter(Protocol[T]):
    """Uniform interface over any output-schema flavour the user declares on an Agent.

    Implementations exist for Pydantic BaseModel, stdlib dataclass, attrs class, and
    raw JSON Schema dict. New flavours plug in by satisfying this protocol — the
    `adapt()` dispatcher is the single place that picks one for a given user input.
    """

    name: str
    """Used as the tool name on Anthropic structured-output mode and as the schema
    label inside the system-prompt fallback. Defaults to the class ``__name__`` for
    Python types; pass-through for raw JSON Schema dicts (caller can name them)."""

    python_type: type[T]
    """The concrete type instances are coerced INTO. ``dict`` for the raw-JsonSchema
    flavour. Used for ``AgentResult[T]`` generic stamping and ``isinstance`` checks
    at the boundary, so callers can write `assert isinstance(result.output, Schema)`
    without caring which adapter is in play."""

    def json_schema(self) -> dict[str, Any]:
        """Render the JSON Schema dict the LLM (or its structured-output mode) will
        see. Must be self-contained — no `$ref` to external documents — because some
        providers reject external refs."""
        ...

    def parse(self, raw: str | dict[str, Any]) -> T:
        """Coerce a raw model response into a typed instance. Accepts both:
        - ``str``: the model emitted a JSON string (text-mode fallback path).
        - ``dict``: a provider's structured-output mode already parsed it for us.

        Raises :class:`OutputCoercionError` (NOT the underlying ``ValidationError`` /
        ``json.JSONDecodeError`` / ``KeyError``) so the retry middleware has a single
        exception type to catch and a structured diagnostics list to reflect back to
        the model. Never returns ``None`` — partial / unfinished outputs go through
        :meth:`partial_parse` instead."""
        ...

    def partial_parse(self, raw: str) -> T | None:
        """Tolerant parse for streaming. Returns the best-effort partial object
        the so-far buffer permits — e.g. a Pydantic model with whatever fields
        have already arrived, missing required fields left unset. Returns
        ``None`` if the buffer has zero useful structure yet (empty, pure
        whitespace, a single open bracket the parser can't resolve to a
        useful prefix).

        Implementations build the partial OBJECT via the type's bypass-init
        path (``BaseModel.model_construct`` / ``object.__new__`` + setattr)
        so missing required fields don't raise. The streaming ``output_coerce``
        middleware calls this on each text delta and lifts the result onto
        ``Delta.partial`` when it changes — the consumer sees a steadily
        more-complete partial without waiting for the terminal delta. The
        strict :meth:`parse` still runs at end-of-stream and remains the
        source of truth for the final typed object."""
        ...

    def serialize(self, value: T) -> dict[str, Any]:
        """Round-trip the typed instance back to a JSON-serialisable dict. Used by
        the durable run-store (so a resumed run can read back the same output the
        original run produced) and by the transcript replay for evals."""
        ...

    def validate(self, value: Any) -> T:
        """Validate an arbitrary Python value against the schema.

        Distinct from :meth:`parse` (which expects raw JSON or dict from a model
        response). ``validate`` handles the tool-result case, where the value
        coming in is whatever Python object the tool's function produced —
        could be an already-instantiated model, a dict the tool built by hand,
        or a primitive when the schema wanted an object.

        - Already-typed instance (e.g. ``MyModel(...)`` passed when
          ``output_schema=MyModel``) — fast-path, returns as-is.
        - Dict → run through coercion (typically via ``parse``) to build the
          typed instance.
        - Anything else (wrong type entirely) → raise :class:`OutputCoercionError`.

        Used by the tool-result schema check on ``FunctionTool``, where the
        tool's return value must match its declared ``output_schema``."""
        ...


class OutputCoercionError(Exception):
    """Raised when a model response can't be coerced into the declared schema.

    Carries the validation diagnostics (`errors`) so the retry middleware can reflect
    them back to the model as a user message — "your previous response failed these
    checks: …, please try again" — which is the single most effective fix for the
    "almost-correct JSON" failure mode. The original raw payload (`raw`) is preserved
    for logging and for the transcript so a human reviewing a failed run can see
    exactly what the model emitted.

    Wrapping every flavour's native exception (Pydantic ``ValidationError``, stdlib
    ``json.JSONDecodeError``, our own missing-field errors) under a single type means
    callers don't have to know which adapter is in play to write a retry policy.
    """

    def __init__(
        self,
        message: str,
        *,
        raw: str | dict[str, Any],
        errors: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        """The exact payload (string or dict) the model produced — preserved verbatim
        for logging, transcripts, and the "show me the failure" UI."""
        self.errors = errors or []
        """Per-field validation diagnostics as plain strings. The retry middleware
        joins these into a user message; the eval harness asserts on them."""
