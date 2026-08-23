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
        # Built lazily and cached: reading ``model_fields`` per serialize call
        # would put reflection on the durable-write path.
        self._val_keys: dict[str, str] | None = None
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
            # ``model_construct`` only accepts known FIELD names; silently
            # discard any extras the partial parser might have produced from
            # a malformed prefix.
            #
            # The stream is alias-keyed — the model is shown
            # ``model_json_schema()``, which uses aliases — so an aliased field
            # arrived as ``{"userName": ...}`` and was dropped wholesale by the
            # name-only filter, making every partial of an aliased model empty.
            # Map alias → field name first; a field name that arrives verbatim
            # still matches (mirrors ``populate_by_name``).
            model_any = cast(Any, self._model)
            by_alias = {
                (info.alias or fname): fname for fname, info in model_any.model_fields.items()
            }
            field_names = set(model_any.model_fields)
            cleaned = {
                (k if k in field_names else by_alias[k]): v
                for k, v in data.items()
                if k in field_names or k in by_alias
            }
            constructed: T = model_any.model_construct(**cleaned)
            return constructed
        except Exception:
            return None

    def serialize(self, value: T) -> dict[str, Any]:
        """``model_dump`` returns plain Python types (dicts, lists, primitives) — no
        Pydantic-specific objects leak into the durable store or transcript. That's
        the round-trip guarantee: `parse(json.dumps(serialize(v))) == v`.

        ``by_alias=True`` is what actually makes that guarantee true. Both of the
        adapter's other surfaces speak ALIASES — ``model_json_schema()`` (validation
        mode, what the model is shown) and ``model_validate*`` (what ``parse``
        accepts) — while a bare ``model_dump()`` emits FIELD NAMES. Measured on
        ``user_name: str = Field(alias="userName")``: serialize gave
        ``{'user_name': 'bob'}`` and feeding that straight back to ``parse`` raised
        ``OutputCoercionError: ['userName: Field required']``, so every durable
        rehydrate of an aliased model failed. Aliasing is a no-op for models without
        aliases, and ``populate_by_name`` models (which accept both spellings)
        round-trip either way.

        ``by_alias=True`` is NOT enough on its own, because Pydantic has two
        aliases and that flag picks the wrong one when they differ. It emits the
        SERIALIZATION alias, while the schema and ``parse`` both speak the
        VALIDATION alias. Measured across the three shapes:

            Field(alias="userName")                       schema userName   dump userName   OK
            Field(serialization_alias="userName")         schema user_name  dump userName   FAILS
            Field(validation_alias="uname",
                  serialization_alias="userName")         schema uname      dump userName   FAILS

        So the keys are remapped to the validation-facing name — exactly what
        ``json_schema()`` advertises and ``parse`` accepts — rather than trusting
        a dump mode to pick it. The previous docstring called the mismatch "a
        modelling choice" whose fix was ``populate_by_name=True``; that is true
        of a single dump mode and not of the adapter, which knows both names and
        can simply use the right one.

        Keys the map does not know (a model with ``extra="allow"``) pass through
        untouched — dropping them would lose data the model deliberately kept."""
        # ``by_alias=True`` is the right BASE: it handles nesting and Pydantic's
        # own type conversion (datetime → str, and so on). It is only wrong
        # about WHICH alias, and only for fields whose two aliases differ — so
        # the walk below fixes exactly those keys and leaves everything else as
        # Pydantic produced it.
        dumped: dict[str, Any] = value.model_dump(by_alias=True)  # type: ignore[attr-defined]
        return cast(dict[str, Any], _to_validation_keys(value, dumped))



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


def _validation_key(field_info: Any, name: str) -> str:
    """The key ``parse`` and the JSON Schema use for this field.

    Mirrors Pydantic's own validation precedence: a plain-string
    ``validation_alias`` wins, then ``alias``, then the field name. The richer
    ``AliasChoices`` / ``AliasPath`` forms fall through to the field name, which
    ``populate_by_name`` models accept — guessing one branch of an alias CHOICE
    would invent a contract the model never stated.
    """
    alias = getattr(field_info, "validation_alias", None)
    if not isinstance(alias, str):
        alias = getattr(field_info, "alias", None)
    return alias if isinstance(alias, str) else name


def _serialization_key(field_info: Any, name: str) -> str:
    """The key ``model_dump(by_alias=True)`` produced for this field."""
    alias = getattr(field_info, "serialization_alias", None)
    if not isinstance(alias, str):
        alias = getattr(field_info, "alias", None)
    return alias if isinstance(alias, str) else name


def _to_validation_keys(value: Any, dumped: Any) -> Any:
    """Rewrite ``dumped``'s keys from serialization aliases to validation ones.

    Walks the VALUE tree alongside the dumped tree, because only the value knows
    which model class produced each dict — a nested model with its own differing
    aliases has to be fixed too. Measured when this only handled the top level:
    ``Outer(inner=Aliased(...))`` dumped its inner model under FIELD names while
    the schema advertised aliases, so the round trip failed one level down. An
    existing nesting test caught it.

    Anything that is not a model (a plain dict a model happens to hold, a
    scalar) is returned untouched — remapping keys we do not own would corrupt
    data the model deliberately kept opaque.
    """
    from pydantic import BaseModel

    if isinstance(value, BaseModel) and isinstance(dumped, dict):
        fields = type(value).model_fields
        rename = {}
        children = {}
        for name, info in fields.items():
            rename[_serialization_key(info, name)] = _validation_key(info, name)
            children[_serialization_key(info, name)] = getattr(value, name, None)
        return {
            rename.get(k, k): _to_validation_keys(children.get(k), v) for k, v in dumped.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)) and isinstance(
        dumped, (list, tuple)
    ):
        # ``dumped`` must be matched on BOTH shapes: ``model_dump`` preserves the
        # container, so a ``tuple[Leaf, Leaf]`` field comes back as a TUPLE while
        # a ``list[Leaf]`` comes back as a list. Guarding on ``list`` alone let
        # every tuple fall through unremapped — measured, a
        # ``tuple[Model, Model]`` serialized to ``[{"userName": ...}]`` (the
        # serialization alias) and failed to round-trip, while the list version
        # beside it worked. The container type is preserved on the way out so a
        # caller comparing against ``model_dump`` sees the same shape.
        rebuilt = [_to_validation_keys(v, d) for v, d in zip(value, dumped, strict=False)]
        return tuple(rebuilt) if isinstance(dumped, tuple) else rebuilt
    if isinstance(value, dict) and isinstance(dumped, dict):
        return {k: _to_validation_keys(value.get(k), d) for k, d in dumped.items()}
    return dumped
