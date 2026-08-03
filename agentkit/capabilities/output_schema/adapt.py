"""``adapt()`` — the single entry point that picks an adapter for a user-supplied
output spec.

The dispatcher is the *one* place that knows about every supported flavour, so
the rest of the framework can take whatever the user passed and call
``adapt(thing)`` without caring which adapter answers. Order of probing matters:

    1. Pydantic BaseModel  — most specific; BaseModel uses ``@dataclass_transform``,
       so a Pydantic class would also pass the stdlib-dataclass probe if we
       checked that one first.
    2. attrs class         — likewise specific, with its own ``attrs.has`` check.
    3. msgspec.Struct      — DEFERRED. Detected and rejected with a clear message
       rather than silently sliding past, so a user passing a msgspec.Struct gets
       "not yet" instead of a confusing TypeError from the final raise.
    4. stdlib dataclass    — the broad catch for "any typed struct"; must come
       after Pydantic.
    5. Raw JSON Schema dict — the escape hatch for non-Python schemas.

The probes for optional deps (pydantic, attrs, msgspec) live inside try/except
so this module loads cleanly in an environment where any subset is missing. An
absent dep just means the corresponding flavour is never picked — never a
``ModuleNotFoundError`` at import time."""

from __future__ import annotations

import dataclasses
from typing import Any

from agentkit.capabilities.output_schema.attrs_adapter import AttrsAdapter
from agentkit.capabilities.output_schema.dataclass_adapter import DataclassAdapter
from agentkit.capabilities.output_schema.json_schema_adapter import JsonSchemaAdapter
from agentkit.capabilities.output_schema.protocol import SchemaAdapter
from agentkit.capabilities.output_schema.pydantic_adapter import PydanticAdapter


def _is_pydantic_model(obj: Any) -> bool:
    """True if ``obj`` is a subclass of ``pydantic.BaseModel``. Returns False
    when pydantic is not installed — that's the whole point of the local
    import: load-time independence from the optional dep."""
    if not isinstance(obj, type):
        return False
    try:
        from pydantic import BaseModel
    except ImportError:
        return False
    return issubclass(obj, BaseModel)


def _is_attrs_class(obj: Any) -> bool:
    """True if ``obj`` is an attrs-decorated class. False when attrs isn't
    installed."""
    if not isinstance(obj, type):
        return False
    try:
        import attrs  # type: ignore[import-not-found]
    except ImportError:
        return False
    result: bool = attrs.has(obj)
    return result


def _is_msgspec_struct(obj: Any) -> bool:
    """True if ``obj`` is a ``msgspec.Struct`` subclass. False when msgspec
    isn't installed. Used only to give a precise NotImplementedError instead of
    falling through to the catch-all TypeError."""
    if not isinstance(obj, type):
        return False
    try:
        import msgspec  # type: ignore[import-not-found]
    except ImportError:
        return False
    return issubclass(obj, msgspec.Struct)


def adapt(output: Any, *, name: str | None = None) -> SchemaAdapter[Any]:
    """Build a :class:`SchemaAdapter` from a user-supplied output spec.

    Accepted shapes:
        * A ``pydantic.BaseModel`` subclass.
        * An ``attrs.define`` / ``@attr.s`` class.
        * A stdlib ``@dataclass`` class.
        * A raw JSON Schema ``dict`` (must have ``type: "object"`` at the top
          level).

    The ``name`` keyword overrides the default name (the Python ``__name__`` for
    classes, the schema ``title`` or ``"Output"`` for raw dicts). The name is
    surfaced to the LLM as the tool / schema label, so a caller can disambiguate
    two identically-named classes from different modules.

    Raises:
        TypeError: when ``output`` is none of the supported shapes.
        NotImplementedError: when ``output`` is a ``msgspec.Struct`` — that
            adapter is deferred to a later phase but the type is detected so
            the user gets a helpful message instead of a generic TypeError.
    """
    if _is_pydantic_model(output):
        return PydanticAdapter(output, name=name)
    if _is_attrs_class(output):
        return AttrsAdapter(output, name=name)
    if _is_msgspec_struct(output):
        raise NotImplementedError(
            "msgspec.Struct output schemas are not yet supported; "
            "use a pydantic BaseModel, attrs class, dataclass, or raw JSON Schema dict"
        )
    # Dataclass probe MUST come after pydantic — pydantic BaseModel uses
    # @dataclass_transform and would also pass dataclasses.is_dataclass() in
    # some pydantic v2 versions.
    if dataclasses.is_dataclass(output) and isinstance(output, type):
        return DataclassAdapter(output, name=name)
    if isinstance(output, dict) and output.get("type") == "object":
        return JsonSchemaAdapter(output, name=name)
    raise TypeError(
        f"unsupported output spec: {output!r}; accepted: "
        "Pydantic BaseModel, attrs class, dataclass, JSON Schema dict"
    )
