"""AttrsAdapter — adapter over a class decorated with ``attrs.define`` / ``@attr.s``.

Why a separate adapter when dataclasses already work: attrs predates dataclasses,
has a wider feature surface (validators, converters, slots, frozen-by-default,
``Factory`` defaults that the stdlib only got later), and is the typed-struct
library that several existing codebases standardised on before dataclasses
shipped. Forcing those users to mirror their attrs classes into dataclasses just
for output schemas is exactly the kind of friction the adapter layer exists to
remove.

The shape mirrors :class:`DataclassAdapter` deliberately — same supported field
types, same fail-fast schema build, same batched error-collection on parse — so
maintainers who know one know both. attrs is OPTIONAL; the import lives inside
``__init__`` so this module loads on a box without it (the ``adapt()`` dispatcher
simply never picks this adapter)."""

from __future__ import annotations

import json
import types
import typing
from typing import Any, Generic, Literal, TypeVar, Union, get_args, get_origin

from agentkit.capabilities.output_schema.protocol import OutputCoercionError

T = TypeVar("T")

_PRIMITIVE_SCHEMAS: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}
"""Same conservative leaf set as the dataclass adapter — see that file's note
for the reasoning. Kept duplicated rather than imported to keep each adapter
file independently readable."""


def _is_attrs_class(cls: Any) -> bool:
    """Local probe (also used by the dispatcher). Guarded so this module loads
    cleanly without attrs installed."""
    if not isinstance(cls, type):
        return False
    try:
        import attrs  # type: ignore[import-not-found]
    except ImportError:
        return False
    result: bool = attrs.has(cls)
    return result


def _type_to_schema(tp: Any) -> dict[str, Any]:
    """Translate a Python type annotation into a JSON Schema fragment. Identical
    contract to the dataclass adapter's translator — see that file for the
    rationale on each branch."""
    if tp in _PRIMITIVE_SCHEMAS:
        return dict(_PRIMITIVE_SCHEMAS[tp])

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            inner = _type_to_schema(non_none[0])
            existing = inner.get("type")
            if isinstance(existing, str):
                inner["type"] = [existing, "null"]
            elif isinstance(existing, list) and "null" not in existing:
                inner["type"] = [*existing, "null"]
            else:
                return {"anyOf": [inner, {"type": "null"}]}
            return inner
        raise TypeError(f"unsupported union {tp!r}; only Optional[X] is supported")

    if origin is Literal:
        types_seen = {type(a) for a in args}
        schema: dict[str, Any] = {"enum": list(args)}
        if len(types_seen) == 1 and (only := next(iter(types_seen))) in _PRIMITIVE_SCHEMAS:
            schema["type"] = _PRIMITIVE_SCHEMAS[only]["type"]
        return schema

    if origin is list:
        if len(args) != 1:
            raise TypeError(f"unsupported list annotation {tp!r}; must be list[X]")
        return {"type": "array", "items": _type_to_schema(args[0])}

    if origin is dict:
        if len(args) != 2 or args[0] is not str:
            raise TypeError(f"unsupported dict annotation {tp!r}; must be dict[str, X]")
        return {"type": "object", "additionalProperties": _type_to_schema(args[1])}

    if _is_attrs_class(tp):
        return _attrs_schema(tp)

    raise TypeError(
        f"unsupported field type {tp!r}; supported: str, int, float, bool, "
        f"list[X], dict[str, X], Optional[X], Literal[...], nested attrs class"
    )


def _attrs_schema(cls: type) -> dict[str, Any]:
    """Walk ``attrs.fields(cls)`` and build a JSON Schema object."""
    import attrs

    hints = typing.get_type_hints(cls)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for f in attrs.fields(cls):
        ftype = hints.get(f.name, f.type)
        # Keyed by ``f.name`` (``_secret``, not the ``secret`` init alias) because
        # that is what ``attrs.asdict`` — i.e. ``serialize`` — emits; the schema and
        # the serialised form have to describe the same document for the round-trip
        # to hold under ``additionalProperties: false``.
        properties[f.name] = _type_to_schema(ftype)
        # attrs uses the sentinel ``attrs.NOTHING`` for "no default" — anything
        # else (including ``Factory(...)``) means the field is optional from the
        # caller's standpoint. ``init=False`` fields are computed by the class, never
        # supplied by the model, so they are never required either (see the
        # dataclass adapter for the same rule).
        if f.default is attrs.NOTHING and f.init:
            required.append(f.name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _coerce_value(tp: Any, value: Any, *, path: str) -> Any:
    """Coerce a JSON-parsed value into the Python type ``tp``. Mirrors the
    dataclass adapter's coercer — see that file for the per-branch reasoning."""
    if tp in _PRIMITIVE_SCHEMAS:
        if tp is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{path}: expected bool, got {type(value).__name__}")
            return value
        if tp is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{path}: expected int, got {type(value).__name__}")
            return value
        if tp is float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{path}: expected number, got {type(value).__name__}")
            return float(value)
        if tp is str:
            if not isinstance(value, str):
                raise ValueError(f"{path}: expected string, got {type(value).__name__}")
            return value

    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            if type(None) in args:
                return None
            raise ValueError(f"{path}: got null but field is not Optional")
        return _coerce_value(non_none[0], value, path=path)

    if origin is Literal:
        if value not in args:
            raise ValueError(f"{path}: {value!r} not in allowed values {list(args)!r}")
        return value

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array, got {type(value).__name__}")
        return [_coerce_value(args[0], v, path=f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object, got {type(value).__name__}")
        return {k: _coerce_value(args[1], v, path=f"{path}.{k}") for k, v in value.items()}

    if _is_attrs_class(tp):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object for {tp.__name__}, got {type(value).__name__}")
        return _coerce_attrs(tp, value, path=path)

    raise ValueError(f"{path}: cannot coerce into {tp!r}")


def _coerce_attrs(cls: type, payload: dict[str, Any], *, path: str) -> Any:
    """Construct an attrs instance from a JSON-parsed payload, batching all
    field errors. See the dataclass adapter for the why."""
    import attrs

    hints = typing.get_type_hints(cls)
    fields = attrs.fields(cls)
    kwargs: dict[str, Any] = {}
    # Two attrs facts the old ``cls(**kwargs)`` ignored, both of which escaped as a
    # raw ``TypeError`` (measured: ``A.__init__() got an unexpected keyword argument
    # '_secret'``) past the ``OutputCoercionError`` contract the repair loop and
    # ``FunctionTool``'s ``ToolShapeError`` seam both depend on:
    #   1. a private attribute is named ``_secret`` but its ``__init__`` parameter is
    #      ``secret`` — attrs exposes that mapping as ``Attribute.alias``;
    #   2. ``field(init=False)`` values are not accepted by ``__init__`` at all.
    # (1) is keyword-remapped, (2) is applied after construction so the value stored
    # by ``serialize`` survives ``parse(serialize(inst))``.
    post_init_values: dict[str, Any] = {}
    errors: list[str] = []
    seen: set[str] = set()
    field_names = {f.name for f in fields}
    for f in fields:
        ftype = hints.get(f.name, f.type)
        sub_path = f"{path}.{f.name}" if path else f.name
        if f.name in payload:
            seen.add(f.name)
            try:
                coerced_field = _coerce_value(ftype, payload[f.name], path=sub_path)
            except ValueError as e:
                errors.append(str(e))
            else:
                if f.init:
                    # ``alias`` is attrs >= 22.2; ``lstrip("_")`` reproduces attrs'
                    # own default rule on older releases.
                    kwargs[getattr(f, "alias", None) or f.name.lstrip("_")] = coerced_field
                else:
                    post_init_values[f.name] = coerced_field
        else:
            if f.default is attrs.NOTHING and f.init:
                errors.append(f"{sub_path}: missing required field")
    for key in payload:
        if key not in field_names:
            errors.append(f"{path + '.' if path else ''}{key}: unexpected field")
    if errors:
        raise ValueError("; ".join(errors))
    try:
        inst = cls(**kwargs)
    except Exception as e:
        # attrs validators/converters raise from ``__init__``; surface them as a
        # ValueError so the adapter wraps them in ``OutputCoercionError`` rather
        # than letting a raw exception escape the coercion boundary.
        raise ValueError(
            f"{path + ': ' if path else ''}could not construct {cls.__name__}: "
            f"{type(e).__name__}: {e}"
        ) from e
    for fname, fvalue in post_init_values.items():
        # ``object.__setattr__``: attrs classes are frozen by default under
        # ``@attrs.frozen`` and their ``__setattr__`` would raise.
        try:
            object.__setattr__(inst, fname, fvalue)
        except AttributeError as e:
            # Slotted class without that slot — nothing we can do; still an
            # OutputCoercionError rather than a raw AttributeError.
            raise ValueError(f"{path + '.' if path else ''}{fname}: cannot set ({e})") from e
    return inst


class AttrsAdapter(Generic[T]):
    """Adapter over an attrs-decorated class. Same fail-fast schema construction
    as :class:`DataclassAdapter`."""

    def __init__(self, cls: type[T], *, name: str | None = None) -> None:
        try:
            import attrs  # noqa: F401  — presence check; imports inside helpers do the actual work
        except ImportError as e:  # pragma: no cover
            raise ImportError("AttrsAdapter requires `attrs` to be installed") from e
        if not _is_attrs_class(cls):
            raise TypeError(f"AttrsAdapter expects an attrs class, got {cls!r}")
        self._schema = _attrs_schema(cls)
        self._cls = cls
        self.python_type: type[T] = cls
        """The attrs class itself — used for ``isinstance`` at the boundary."""
        self.name: str = name or cls.__name__
        """Defaults to the class ``__name__``; caller can override."""

    def json_schema(self) -> dict[str, Any]:
        """Deep-copy so a provider-specific massage can't mutate the cached schema."""
        schema: dict[str, Any] = json.loads(json.dumps(self._schema))
        return schema

    def parse(self, raw: str | dict[str, Any]) -> T:
        """Same shape as the dataclass adapter's parse: JSON-decode if needed,
        walk-and-construct, batch errors into one :class:`OutputCoercionError`."""
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                raise OutputCoercionError(
                    f"response is not valid JSON: {e.msg} (line {e.lineno}, col {e.colno})",
                    raw=raw,
                    errors=[f"json: {e.msg}"],
                ) from e
        else:
            payload = raw
        if not isinstance(payload, dict):
            raise OutputCoercionError(
                f"expected a JSON object for {self.name}, got {type(payload).__name__}",
                raw=raw,
                errors=[f"top-level: expected object, got {type(payload).__name__}"],
            )
        try:
            parsed: T = _coerce_attrs(self._cls, payload, path="")
            return parsed
        except ValueError as e:
            errors = str(e).split("; ")
            raise OutputCoercionError(
                f"failed to coerce response into {self.name}: {len(errors)} error(s)",
                raw=raw,
                errors=errors,
            ) from e

    def partial_parse(self, raw: str) -> T | None:
        """Tolerant parse → build a partial attrs instance.

        Mirrors the dataclass adapter. attrs's ``__init__`` rejects
        missing required positional args, so we bypass it: instantiate
        via ``object.__new__`` and ``setattr`` only the fields the
        stream has produced. Seeds defaults for the rest so attribute
        access doesn't raise on un-streamed fields.

        Caveat: attrs's slotted classes (``slots=True``) reject
        ``setattr`` for un-declared attributes. We catch and return
        ``None`` in that case — the streaming partial is best-effort,
        and the strict terminal parse remains the source of truth.
        """
        import attrs

        from agentkit.capabilities.output_schema._partial_json import parse_partial

        try:
            data = parse_partial(raw)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        try:
            inst = object.__new__(self._cls)
            field_names = {f.name for f in attrs.fields(self._cls)}  # type: ignore[arg-type]
            for f in attrs.fields(self._cls):  # type: ignore[arg-type]
                if f.default is attrs.NOTHING:
                    continue
                # ``attrs.Factory`` defaults are wrapped — invoke if so.
                default = f.default
                if isinstance(default, attrs.Factory):  # type: ignore[arg-type]
                    try:
                        object.__setattr__(inst, f.name, default.factory())
                    except Exception:
                        pass
                else:
                    try:
                        object.__setattr__(inst, f.name, default)
                    except Exception:
                        pass
            for key, value in data.items():
                if key in field_names:
                    try:
                        object.__setattr__(inst, key, value)
                    except Exception:
                        # Slotted attrs class rejecting an attr we couldn't
                        # populate — surrender the partial entirely.
                        return None
            return inst
        except Exception:
            return None

    def serialize(self, value: T) -> dict[str, Any]:
        """``attrs.asdict`` walks recursively into nested attrs classes, lists,
        and dicts."""
        import attrs

        dumped: dict[str, Any] = attrs.asdict(value)  # type: ignore[arg-type]
        return dumped

    def validate(self, value: Any) -> T:
        """Validate an arbitrary Python value against the attrs class.

        Mirrors :meth:`DataclassAdapter.validate` — fast-path an instance,
        coerce a dict through the type-tree walker, raise
        :class:`OutputCoercionError` for anything else."""
        if isinstance(value, self._cls):
            return value
        if isinstance(value, dict):
            try:
                coerced: T = _coerce_attrs(self._cls, value, path="")
                return coerced
            except ValueError as e:
                errors = str(e).split("; ")
                raise OutputCoercionError(
                    f"failed to validate value into {self.name}: {len(errors)} error(s)",
                    raw=value,
                    errors=errors,
                ) from e
        raise OutputCoercionError(
            f"value of type {type(value).__name__} is not coercible into {self.name}",
            raw=repr(value),
            errors=[f"top-level: expected {self.name}, got {type(value).__name__}"],
        )
