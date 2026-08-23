"""JSON-schema inference helpers for tools.

These translate a Python callable's signature + type hints into the LLM-advertised
``ToolSchema`` (input schema) and best-effort infer an output schema from the
return-type annotation. Module-private by convention — exported only via the
package ``__init__`` if something elsewhere needs them.
"""

from __future__ import annotations

import contextlib
import enum
import inspect
import types as _types
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from agentkit.capabilities.output_schema import adapt
from agentkit.kernel.types import ToolSchema
from agentkit.tools.errors import ToolDefinitionError

# Primitives and trivial annotations that should NOT trigger auto-inference of an
# output_schema from a function's return type. A tool typed ``-> str`` doesn't
# need (or want) the validation overhead; a tool typed ``-> dict[str, Any]`` has
# no schema-typed shape to enforce. We're conservative on purpose — auto-inference
# fires only when the return annotation is one of the four flavours ``adapt()``
# already supports (Pydantic BaseModel, dataclass, attrs class) AND the user can
# always opt out with ``output_schema=None``.
_OUTPUT_AUTO_SKIP_PRIMITIVES: tuple[type, ...] = (str, int, float, bool, bytes)


def _infer_output_schema(func: Callable[..., Any]) -> Any:
    """Best-effort: return the function's return-type annotation IF it's a shape
    ``adapt()`` knows how to enforce (Pydantic BaseModel, stdlib dataclass, attrs
    class). Return ``None`` for everything else (primitives, ``Any``, missing
    annotation, generic ``dict[str, X]`` / ``list[X]``, unions, etc.).

    Run inside a guarded ``contextlib.suppress`` block because ``get_type_hints``
    can fail on forward refs / circular imports — auto-inference is opportunistic
    by design and must NEVER crash tool construction. The user can always pass
    an explicit ``output_schema=`` (or ``output_schema=None`` to opt out)."""
    with contextlib.suppress(Exception):
        hints = get_type_hints(func)
        ann = hints.get("return")
        if ann is None or ann is type(None):
            return None
        if not isinstance(ann, type):
            # Generics (``list[X]``, ``dict[str, X]``), Union types, ``Any``, etc.
            # ``adapt()`` only enforces concrete typed-struct classes.
            return None
        if ann in _OUTPUT_AUTO_SKIP_PRIMITIVES:
            return None
        # Probe through adapt(): if it raises (TypeError / NotImplementedError /
        # ImportError), the annotation isn't a flavour we can enforce.
        with contextlib.suppress(Exception):
            adapt(ann)
            return ann
    return None


_JSON_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_CTX_PARAMS = ("ctx", "context")  # injected with the RunContext, not advertised to the model
_MIN_DESCRIPTION_LEN = 30  # the floor below which the model can't reliably understand the tool


def _is_ctx_param(p: inspect.Parameter) -> bool:
    """A `ctx`/`context` param is the injected context ONLY when it's unannotated or annotated with a
    context type — so a real data param like `def f(context: str)` is NOT hijacked (it stays advertised
    and is passed normally). Checked by annotation NAME rather than by importing the types, to keep the
    tools layer decoupled from runtime.

    ``Ctx`` is accepted alongside ``RunContext``, and that omission was a trap. ``Ctx`` is the structural
    Protocol the framework itself uses everywhere internally, so it is the natural annotation to reach
    for — and annotating with it used to silently demote the parameter to an ordinary one. Measured: the
    tool advertised ``{"ctx": {"type": "string"}}`` to the MODEL and listed it in ``required``, so every
    call failed with ``ToolArgumentError: missing required argument(s) ['ctx']``. Nothing pointed at the
    annotation; the author had written the most reasonable thing available.
    """
    if p.name not in _CTX_PARAMS:
        return False
    ann = p.annotation
    if ann is inspect.Parameter.empty:
        return True
    text = str(ann)
    name = getattr(ann, "__name__", "")
    return name in ("RunContext", "Ctx") or "RunContext" in text or text.endswith("Ctx")


def _enum_fragment(ann: type[enum.Enum]) -> dict[str, Any]:
    """An ``Enum`` class → ``{"enum": [...values...]}``, typed when the member
    values are homogeneous. Advertising a bare ``{"type": "string"}`` told the
    model nothing about which strings are legal, so it was free to invent one
    and the tool raised ``ValueError`` on a value the schema had implied was
    fine. Same treatment ``Literal`` already gets."""
    vals = [m.value for m in ann]
    frag: dict[str, Any] = {"enum": vals}
    kinds = {_JSON_PRIMITIVES.get(type(v)) for v in vals}
    if len(kinds) == 1 and (t := kinds.pop()) is not None:
        frag["type"] = t
    return frag


def _struct_fragment(ann: Any) -> dict[str, Any] | None:
    """A typed struct (Pydantic model / dataclass / attrs class) → its real
    object schema, via the same ``adapt()`` dispatcher ``output_schema`` uses.

    Without this a structured parameter advertised as ``{"type": "string"}``:
    the model dutifully sent a string, and the function received a ``str``
    where its annotation promised a ``Filter``. The schema was actively
    instructing the model to call the tool wrongly.

    Returns ``None`` for anything ``adapt()`` cannot describe, so the caller
    falls through to its existing best-effort mapping.
    """
    if not isinstance(ann, type) or issubclass(ann, enum.Enum):
        return None
    with contextlib.suppress(Exception):
        schema = adapt(ann).json_schema()
        if isinstance(schema, dict) and schema:
            # ``title``/``$defs`` are for humans and for $ref resolution; the
            # inline fragment keeps them, since a provider that rejects them
            # would equally reject them on ``Agent.output=``.
            return schema
    return None


def _json_type(ann: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON-schema type fragment (best-effort, dependency-free)."""
    if ann in _JSON_PRIMITIVES:
        return {"type": _JSON_PRIMITIVES[ann]}
    if ann in (list, tuple, set, frozenset):  # bare `list` / `tuple` (unsubscripted)
        return {"type": "array"}
    if ann is dict:
        return {"type": "object"}
    origin = get_origin(ann)
    if origin in (list, tuple, set, frozenset):  # list[X] / tuple[...] / set[X]
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if origin is Literal:  # Literal["a","b"] / Literal[1,2] → enum
        vals = list(get_args(ann))
        frag: dict[str, Any] = {"enum": vals}
        kinds = {_JSON_PRIMITIVES.get(type(v)) for v in vals}
        if (
            len(kinds) == 1 and (t := kinds.pop()) is not None
        ):  # homogeneous values → also pin the type
            frag["type"] = t
        return frag
    if origin is Union or origin is getattr(_types, "UnionType", None):  # Union[...] / X | Y
        real = [a for a in get_args(ann) if a is not type(None)]
        if len(real) == 1:  # Optional[X] / X | None → just X
            return _json_type(real[0])
        if real:  # genuine multi-type union → anyOf (don't pick one)
            return {"anyOf": [_json_type(a) for a in real]}
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return _enum_fragment(ann)
    struct = _struct_fragment(ann)
    if struct is not None:
        return struct
    return {"type": "string"}  # unknown / unannotated → string


def _resolved_description(func: Callable[..., Any], description: str | None) -> str:
    """Pick the description for `func`: explicit `description=` wins, else the first line of the docstring.
    Validates the chosen text meets the framework's tool-writing floor (>=30 chars after strip) and
    raises `ToolDefinitionError` otherwise — the model needs enough text to understand the tool."""
    if description is not None:
        desc = description.strip()
    else:
        doc = inspect.getdoc(func) or ""
        desc = doc.strip().split("\n", 1)[0].strip()
    if len(desc) < _MIN_DESCRIPTION_LEN:
        name = getattr(func, "__name__", "tool")
        raise ToolDefinitionError(
            f"tool {name!r} needs a docstring/description of at least "
            f"{_MIN_DESCRIPTION_LEN} characters so the model can understand it; "
            f"got {len(desc)} chars: {desc!r}"
        )
    return desc


def _build_schema(func: Callable[..., Any], name: str, description: str) -> ToolSchema:
    """Inspect a function's signature + type hints into a `ToolSchema` (LLM-advertised JSON schema).
    `description` is pre-validated by `_resolved_description`."""
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 — forward refs etc.; degrade to str
        hints = {}
    props: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for pname, p in inspect.signature(func).parameters.items():
        if pname in ("self", "cls") or _is_ctx_param(p):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[pname] = _json_type(hints.get(pname, str))
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return ToolSchema(
        name=name,
        description=description,
        parameters={"type": "object", "properties": props, "required": required},
    )
