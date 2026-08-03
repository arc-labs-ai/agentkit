"""DataclassAdapter — adapter over a stdlib ``@dataclass`` class.

Why stdlib dataclasses are the *default* flavour: they have zero dependencies, are
in every modern Python install, are the most natural thing a Python developer
reaches for when they want a typed struct, and they round-trip cleanly through
``dataclasses.asdict`` / hand-rolled construction. The cost is that we have to
write our own JSON Schema generator and our own parser walker — there is no
"dataclasses.to_json_schema" in the stdlib. The cost is bounded: we support the
shapes we actually use (primitives, list, dict, Optional, Literal, nested
dataclass) and *refuse* unsupported shapes at adapter-construction time so a
typo can't ship a broken schema to the model.

The refusal-at-construction rule is the whole reason this module is more than a
hundred lines: every unsupported annotation (``set[str]``, ``Tuple[int, ...]``,
``Any``, an un-parameterised ``list``) raises ``TypeError`` *during ``__init__``*,
so the failure shows up at agent-build time instead of mid-run when the model
finally returns a value of the offending field.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from typing import Any, Generic, Literal, TypeVar, Union, cast, get_args, get_origin

from agentkit.capabilities.output_schema.protocol import OutputCoercionError

T = TypeVar("T")

_PRIMITIVE_SCHEMAS: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}
"""The leaf types we can translate verbatim. Deliberately conservative —
``bytes``/``bytearray``/``Decimal`` are absent because there is no canonical JSON
representation and forcing the model to emit base64 / a stringified decimal hides
intent. Add them only when there's a concrete use case."""


def _type_to_schema(tp: Any) -> dict[str, Any]:
    """Translate a Python type annotation into a JSON Schema fragment.

    Recursive: nested dataclasses become inline ``type: object`` blocks (we choose
    inline-everything over ``$defs`` because some provider structured-output modes
    reject internal refs, and inlined-twice is cheap for the schema sizes we see).

    Raises ``TypeError`` for any annotation we don't know how to translate — caller
    catches at adapter-construction time so the failure surfaces early."""
    if tp in _PRIMITIVE_SCHEMAS:
        return dict(_PRIMITIVE_SCHEMAS[tp])

    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[X] / X | None → allow null on the inner schema.
    # Note: ``Optional[X]`` is ``Union[X, None]`` and PEP 604 ``X | None`` is
    # ``types.UnionType`` — both expose ``get_origin``/``get_args``, so we treat
    # them uniformly.
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
                # No primitive `type` key (e.g. nested object) — fall back to
                # anyOf so JSON Schema validators still allow null.
                return {"anyOf": [inner, {"type": "null"}]}
            return inner
        raise TypeError(f"unsupported union {tp!r}; only Optional[X] (i.e. X | None) is supported")

    # Literal["a", "b"] → enum of those values. We keep the JSON-Schema-style
    # ``type`` hint when all literals share a primitive Python type, since some
    # providers reject a bare ``enum`` without a ``type``.
    if origin is Literal:
        types_seen = {type(a) for a in args}
        schema: dict[str, Any] = {"enum": list(args)}
        if len(types_seen) == 1 and (only := next(iter(types_seen))) in _PRIMITIVE_SCHEMAS:
            schema["type"] = _PRIMITIVE_SCHEMAS[only]["type"]
        return schema

    # list[X] → array of inner schema.
    if origin is list:
        if len(args) != 1:
            raise TypeError(
                f"unsupported list annotation {tp!r}; must be list[X] with one type arg"
            )
        return {"type": "array", "items": _type_to_schema(args[0])}

    # dict[str, X] → object with additionalProperties=<inner schema>. We do NOT
    # support non-string keys: JSON object keys are strings by spec, and silently
    # stringifying an int key would surprise both author and reader.
    if origin is dict:
        if len(args) != 2 or args[0] is not str:
            raise TypeError(f"unsupported dict annotation {tp!r}; must be dict[str, X]")
        return {"type": "object", "additionalProperties": _type_to_schema(args[1])}

    # Nested dataclass → recurse to an inlined object schema. Importing the
    # adapter here would be circular at module load (this file is what defines it),
    # so we build the nested schema by calling _dataclass_schema directly.
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _dataclass_schema(tp)

    raise TypeError(
        f"unsupported field type {tp!r}; supported: str, int, float, bool, "
        f"list[X], dict[str, X], Optional[X], Literal[...], nested dataclass"
    )


def _dataclass_schema(cls: type) -> dict[str, Any]:
    """Walk ``dataclasses.fields(cls)`` and build a JSON Schema object. Required =
    every field with no default and no ``default_factory`` — anything with a
    default is treated as optional, mirroring how a caller could legally omit it
    when constructing the instance."""
    hints = typing.get_type_hints(cls)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        ftype = hints.get(f.name, f.type)
        properties[f.name] = _type_to_schema(ftype)
        is_optional_value = (
            f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        )
        if not is_optional_value:
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
    """Coerce a JSON-parsed value into the Python type ``tp``. Walks the same
    annotations as ``_type_to_schema`` — the two functions are a pair, kept
    side-by-side so a change in supported types is impossible to forget on
    either side.

    Errors are raised as ``ValueError`` with a dotted path; the caller batches
    them into an :class:`OutputCoercionError`."""
    if tp in _PRIMITIVE_SCHEMAS:
        # bool is a subclass of int in Python — guard against True/False landing
        # in an int field (and vice versa).
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

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ValueError(
                f"{path}: expected object for {tp.__name__}, got {type(value).__name__}"
            )
        return _coerce_dataclass(tp, value, path=path)

    raise ValueError(f"{path}: cannot coerce into {tp!r}")


def _coerce_dataclass(cls: type, payload: dict[str, Any], *, path: str) -> Any:
    """Construct a dataclass instance from a JSON-parsed payload, collecting all
    field errors instead of raising on the first one. The collected errors are
    re-raised as a single ``ValueError`` whose message lists every problem — the
    adapter's ``parse`` upgrades that to :class:`OutputCoercionError`.

    Collecting all errors (rather than fail-fast) is deliberate: a model that
    botched two fields should hear about both in one retry, not have us yo-yo
    through "fix one, discover the next, fix the next" rounds."""
    hints = typing.get_type_hints(cls)
    fields = dataclasses.fields(cls)
    kwargs: dict[str, Any] = {}
    errors: list[str] = []
    seen: set[str] = set()
    field_names = {f.name for f in fields}
    for f in fields:
        ftype = hints.get(f.name, f.type)
        sub_path = f"{path}.{f.name}" if path else f.name
        if f.name in payload:
            seen.add(f.name)
            try:
                kwargs[f.name] = _coerce_value(ftype, payload[f.name], path=sub_path)
            except ValueError as e:
                errors.append(str(e))
        else:
            has_default = (
                f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
            )
            if not has_default:
                errors.append(f"{sub_path}: missing required field")
    # Surface extra keys: the schema we hand the model declares
    # ``additionalProperties: false``, so any extra is a model mistake and
    # silently dropping it would mask bugs.
    for key in payload:
        if key not in field_names:
            errors.append(f"{path + '.' if path else ''}{key}: unexpected field")
    if errors:
        raise ValueError("; ".join(errors))
    return cls(**kwargs)


class DataclassAdapter(Generic[T]):
    """Adapter over a stdlib ``@dataclass`` class. Constructor validates the
    *type-tree* up-front — every field annotation is run through ``_type_to_schema``
    so an unsupported shape (``set[X]``, ``Any``, bare ``list``) raises
    ``TypeError`` at agent-build time rather than at first model response."""

    def __init__(self, cls: type[T], *, name: str | None = None) -> None:
        if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
            raise TypeError(f"DataclassAdapter expects a stdlib dataclass type, got {cls!r}")
        # Fail-fast: build the schema NOW so an unsupported annotation surfaces
        # at construction time, not at first parse.
        self._schema = _dataclass_schema(cls)
        self._cls = cls
        self.python_type: type[T] = cls
        """The dataclass type itself — used for ``isinstance`` at the boundary."""
        self.name: str = name or cls.__name__
        """Defaults to the class ``__name__``; caller can override."""

    def json_schema(self) -> dict[str, Any]:
        """Return a deep-ish copy so a downstream mutation (provider-specific
        massaging) can't corrupt the cached schema."""
        schema: dict[str, Any] = json.loads(json.dumps(self._schema))
        return schema

    def parse(self, raw: str | dict[str, Any]) -> T:
        """Parse JSON if needed, then walk the type tree and construct the
        dataclass. All validation errors are batched into a single
        :class:`OutputCoercionError`."""
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
            parsed: T = _coerce_dataclass(self._cls, payload, path="")
            return parsed
        except ValueError as e:
            errors = str(e).split("; ")
            raise OutputCoercionError(
                f"failed to coerce response into {self.name}: {len(errors)} error(s)",
                raw=raw,
                errors=errors,
            ) from e

    def partial_parse(self, raw: str) -> T | None:
        """Tolerant parse → build a partial instance with only the fields
        the stream has supplied so far.

        Bypasses ``__init__`` (which would reject missing required fields)
        via ``object.__new__`` + ``setattr`` per present field. Fields
        absent from the partial JSON stay unset on the instance — the
        same convention Pydantic's ``model_construct`` uses. Returns
        ``None`` on any failure; never raises.

        Equality between partial instances goes through the dataclass's
        synthesised ``__eq__``, which compares every declared field.
        Unset fields can cause ``__eq__`` to raise ``AttributeError`` —
        which the streaming middleware swallows when it compares
        ``new != last``. To keep that comparison usable, we ALSO seed
        any declared default / default_factory so the instance is at
        least readable without ``AttributeError`` for absent fields
        that the dataclass itself defined a fallback for.
        """
        from agentkit.capabilities.output_schema._partial_json import parse_partial

        try:
            data = parse_partial(raw)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        try:
            inst = object.__new__(self._cls)
            field_names = {f.name for f in dataclasses.fields(cast(Any, self._cls))}
            # Seed defaults first so attribute access doesn't raise for
            # fields the stream hasn't reached yet but the dataclass
            # itself carries a default for.
            for f in dataclasses.fields(cast(Any, self._cls)):
                if f.default is not dataclasses.MISSING:
                    object.__setattr__(inst, f.name, f.default)
                elif f.default_factory is not dataclasses.MISSING:
                    try:
                        object.__setattr__(inst, f.name, f.default_factory())
                    except Exception:
                        pass
            for key, value in data.items():
                if key in field_names:
                    object.__setattr__(inst, key, value)
            return inst
        except Exception:
            return None

    def serialize(self, value: T) -> dict[str, Any]:
        """``dataclasses.asdict`` walks recursively, converting nested dataclasses,
        lists, and dicts into plain Python primitives. Round-trip guarantee holds
        for the supported type subset."""
        dumped: dict[str, Any] = dataclasses.asdict(cast(Any, value))
        return dumped

    def validate(self, value: Any) -> T:
        """Validate an arbitrary Python value against the dataclass.

        Fast-paths an already-typed instance. Coerces a dict via the same
        walker ``parse`` uses (so a tool that returns a hand-built dict
        gets validated against the type-tree). Anything else (wrong type
        entirely) raises :class:`OutputCoercionError` — the tool-shape
        check has a clear single point to catch."""
        if isinstance(value, self._cls):
            return value
        if isinstance(value, dict):
            try:
                coerced: T = _coerce_dataclass(self._cls, value, path="")
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
