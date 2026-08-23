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

from agentkit.capabilities.output_schema import OutputCoercionError, SchemaAdapter, adapt
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


def _struct_adapter(ann: Any) -> SchemaAdapter[Any] | None:
    """The ``SchemaAdapter`` for a typed struct (Pydantic model / dataclass /
    attrs class), or ``None`` when ``ann`` is not a shape ``adapt()`` can
    describe.

    This is the single gate both directions go through: ``_struct_fragment``
    asks it for the schema we ADVERTISE, ``_coercer_for`` asks it for the
    parser we APPLY. Deriving both from one predicate is the whole point —
    the pair used to disagree, and the disagreement was the bug (advertise an
    object schema, hand the body the raw ``dict``). Anything this returns
    ``None`` for is advertised by the fallback mapping and coerced by nobody,
    which is consistent in the other direction.

    The ``json_schema()`` probe (rather than a bare ``adapt()``) is deliberate:
    ``adapt()`` succeeding is not proof the flavour renders, and an adapter
    whose schema comes back empty is one we must not claim to describe."""
    if not isinstance(ann, type) or issubclass(ann, enum.Enum):
        return None
    with contextlib.suppress(Exception):
        adapter = adapt(ann)
        schema = adapter.json_schema()
        if isinstance(schema, dict) and schema:
            return adapter
    return None


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
    adapter = _struct_adapter(ann)
    if adapter is None:
        return None
    with contextlib.suppress(Exception):
        schema = adapter.json_schema()
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


# ---- argument coercion ------------------------------------------------------------------------
#
# Everything above turns an annotation into the schema the MODEL sees. Everything
# below turns the same annotation into the function that reconstitutes the value
# the BODY sees. They are two halves of one promise, and for a while the framework
# only kept the first half.
#
# Measured before this existed, on ``async def weather(city: str, unit: Unit)``:
#
#     schema advertises: {'enum': ['celsius', 'fahrenheit'], 'type': 'string'}
#     body receives    : str='celsius'
#
# — and the same shape of lie for every other rich annotation: a ``Filter``
# dataclass advertised as a full object schema arrived as ``dict={'field': 'name',
# 'limit': 2}``; ``Literal["a","b"]`` arrived as a bare ``str`` with nothing
# checking it was one of the two. An author who wrote ``unit.value`` or
# ``flt.field`` — the only reasonable thing to write, given the annotation they
# had just typed — got ``AttributeError`` from inside their own tool body, on a
# value the framework had told the model was that type.
#
# Worse than the AttributeError was the case that raised nothing at all. Measured:
# ``weather(city="SF", unit="kelvin")`` ran to completion and the body saw
# ``str='kelvin'``, a value the advertised enum excluded. ``search(flt={'limit':
# 2})`` ran with the schema's REQUIRED ``field`` key simply absent. The schema was
# decoration; nothing enforced it.


class _CoercionRejected(Exception):
    """Internal: a coercer could not turn the caller's value into the annotated type.

    Deliberately not a public error. ``schema.py`` knows the annotation and the
    offending value but not the tool name, and the tool name is what makes the
    message actionable — so this carries only ``reason`` and ``function.py``
    re-raises it as :class:`ToolArgumentError` with the call's identity attached.
    Keeping the public failure a ``ToolArgumentError`` is load-bearing: that is
    the type the retry/repair path already understands as "the caller sent
    something wrong, show it the message and let it try again", and a coercion
    failure is exactly that. A raw ``ValueError`` out of ``Unit('kelvin')`` would
    read as a tool crash instead."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_Coercer = Callable[[Any], Any]


def _coerce_enum(cls: type[enum.Enum], value: Any) -> Any:
    """Enum member VALUE → the member. ``cls(value)`` is the whole conversion; the
    surrounding code is about the two edges.

    The ``isinstance`` fast path is for idempotence. In-process callers (tests,
    ``ToolBackedMemory``, an agent calling another agent's tool directly) pass the
    real member, not the wire value, and must not be punished for it. ``Unit(Unit.C)``
    happens to return ``Unit.C`` so the constructor would survive it anyway — but
    that is an implementation detail of ``EnumMeta.__call__``, and one branch is
    cheaper than relying on it.

    Lookup is by VALUE only, never by member NAME. The model is shown
    ``{"enum": ["celsius", "fahrenheit"]}`` — the values — so the values are the
    contract, and accepting ``"C"`` as well would mean honouring input the schema
    never offered. The rejection message lists the legal values, which is what lets
    the repair turn fix it in one shot rather than guessing."""
    if isinstance(value, cls):
        return value
    try:
        return cls(value)
    except (ValueError, KeyError, TypeError) as exc:
        legal = [m.value for m in cls]
        raise _CoercionRejected(
            f"{value!r} is not a valid {cls.__name__}; expected one of {legal!r}"
        ) from exc


def _coerce_literal(allowed: tuple[Any, ...], value: Any) -> Any:
    """``Literal[...]`` is a membership CHECK, not a conversion — the permitted
    values are already JSON primitives, so a legal value arrives needing nothing
    done to it. What was missing was anyone checking.

    The match is type-exact on purpose. ``True == 1`` in Python, so a loose ``in``
    test would accept ``True`` for ``Literal[1, 2, 3]`` and hand the body a ``bool``
    where the annotation — and the advertised ``{"type": "integer"}`` — promised an
    ``int``. Measured: ``True in (1, 2, 3)`` is ``True``; the ``type(a) is type(value)``
    guard is what turns that into a rejection. ``a is value`` comes first so
    ``Literal[None]`` and singleton sentinels match without an ``==`` that a
    pathological ``__eq__`` could subvert."""
    for a in allowed:
        if a is value:
            return value
        if type(a) is type(value) and a == value:
            return value
    raise _CoercionRejected(f"{value!r} is not one of {list(allowed)!r}")


def _coerce_struct(adapter: SchemaAdapter[Any], value: Any) -> Any:
    """A decoded JSON object → the dataclass / Pydantic model / attrs instance the
    parameter was annotated with, through the very adapter whose ``json_schema()``
    we advertised for it.

    Reusing ``agentkit.capabilities.output_schema`` here was the one genuinely
    debatable call, and it is not a new dependency: ``schema.py`` already imports
    ``adapt`` to BUILD the advertised fragment, and ``function.py`` already imports
    it for ``output_schema=``. The tools layer is coupled to the adapter layer
    either way. Given that, the choice was between parsing the object with the
    adapter that produced the schema, or hand-rolling a second parser in the tools
    layer that would drift from it — two parsers for one schema, differing on
    nested structs, defaults and unions. One adapter, one truth.

    ``validate()`` rather than ``parse()``: it fast-paths an already-typed instance
    (idempotence for in-process callers, same as the enum branch) and only walks
    the type-tree for a ``dict``. ``FrozenDict`` is a ``dict`` subclass, so a
    deep-frozen payload takes the dict branch unchanged."""
    try:
        return adapter.validate(value)
    except OutputCoercionError as exc:
        detail = "; ".join(exc.errors) if exc.errors else str(exc)
        raise _CoercionRejected(f"not a valid {adapter.name}: {detail}") from exc


def _coercer_for(ann: Any) -> _Coercer | None:
    """An annotation → the function that reconstitutes its values, or ``None`` for
    "leave the value exactly as it arrived".

    ``None`` is the answer for everything the schema describes with a plain
    ``{"type": ...}``: ``str``/``int``/``float``/``bool``, ``Any``, an unannotated
    parameter, ``list``/``dict`` and their generics. Two reasons, and they are the
    same reason twice. First, those need no reconstitution — JSON already decodes
    to them. Second, coercing them would mean INVENTING a promise: nothing here
    turns ``"3"`` into ``3``, because a provider that sent a string for an
    ``{"type": "integer"}`` parameter has a bug the framework must surface, not
    launder. The mandate is to honour what the schema says, not to widen it.

    ``list[Unit]`` gets ``None`` for the same reason, and it is worth naming
    because it looks like an omission. ``_json_type`` advertises it as a bare
    ``{"type": "array"}`` with no ``items`` — the element type is never told to the
    model. Coercing the elements would be enforcing a contract the model was never
    shown. When the schema learns to describe ``items``, this is where the matching
    coercer goes."""
    origin = get_origin(ann)
    if origin is Union or origin is getattr(_types, "UnionType", None):
        real = [a for a in get_args(ann) if a is not type(None)]
        if len(real) == 1:
            # ``Optional[X]`` / ``X | None``. ``_json_type`` collapses this to
            # bare ``X``, so the coercer collapses identically — with ``None``
            # passed straight through, because ``None`` is what the annotation's
            # other half is FOR. ``unit: Unit | None = None`` called without
            # ``unit`` never reaches a coercer at all (the key is absent from
            # kwargs and the default applies untouched); called with an explicit
            # JSON ``null`` it reaches here and must not become a rejection.
            inner = _coercer_for(real[0])
            if inner is None:
                return None
            return lambda v: None if v is None else inner(v)
        # A genuine multi-member union is advertised as ``anyOf`` precisely
        # because the schema layer refuses to pick a member. Picking one here
        # would be the same guess, made silently and with the power to mutate
        # the value: for ``Unit | str``, "celsius" is a legal inhabitant of BOTH
        # arms, and there is no principled way to know which the author meant.
        # Declining is the honest answer, and it matches what we advertised.
        return None
    if origin is Literal:
        allowed = get_args(ann)
        return lambda v: _coerce_literal(allowed, v)
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return lambda v: _coerce_enum(ann, v)
    adapter = _struct_adapter(ann)
    if adapter is not None:
        return lambda v: _coerce_struct(adapter, v)
    return None


def _build_coercers(func: Callable[..., Any]) -> dict[str, _Coercer]:
    """Every parameter of ``func`` that needs reconstituting → its coercer, built
    ONCE at ``@tool`` decoration time.

    Build-time is where the expensive parts live — ``get_type_hints``, ``adapt()``,
    ``json_schema()`` — and they run ONCE per tool, here, alongside the pass that
    already builds the advertised schema. Measured at decoration: ~70-130us per
    tool, dominated by ``get_type_hints``. A process wiring fifty tools spends
    under 10ms on it, at import, once.

    What the call path inherits is a dict, and for the overwhelmingly common tool
    — every parameter a ``str``/``int``/``bool`` — that dict is EMPTY and the
    whole added cost is one falsy test. Measured over 20k calls (best of 7),
    ``run()`` end to end:

        primitives  2.554us -> 2.587us   (+1.3%, inside the run-to-run noise)
        Enum        2.594us -> 3.455us   (+0.86us)
        dataclass   2.585us -> 12.51us   (+9.9us)

    The dataclass row is the honest one and it is the only number worth a second
    look: ~10us is the adapter walking the type-tree, and the "before" figure it
    is measured against did not do the job at all — it handed the body a ``dict``.
    So it is not a regression from 2.6us, it is the price of the parameter being
    the type it was advertised as. Against a tool call that is about to touch the
    network or the disk — the reason tools exist — 10us is not material, and a
    tool that wants the raw ``dict`` says so by annotating ``dict``.

    Parameters skipped, and why each is not an oversight:
    - ``self``/``cls`` and ``ctx``/``context`` when ``_is_ctx_param`` says so —
      the same predicate ``_build_schema`` uses, so the two agree by construction.
      The ctx parameter is INJECTED, never supplied by the caller; giving it a
      coercer would be pointing a parser at a ``RunContext``. A real data
      parameter named ``context`` is not a ctx param, stays advertised, and gets
      a coercer like any other parameter.
    - ``*args``/``**kwargs`` — not advertised, so not coerced. A ``**kwargs`` tool
      still receives its extra keys exactly as they arrived; there is no
      annotation behind them to honour.
    - unannotated parameters — ``_build_schema`` defaults these to ``str`` for
      advertising purposes (``hints.get(pname, str)``), but this pass uses a bare
      ``hints.get(pname)`` and leaves them alone. Inventing a coercion from a
      default the author never wrote would be exactly the overreach the rest of
      this module refuses.

    ``get_type_hints`` failing (forward refs, circular imports) degrades to no
    coercion at all, mirroring ``_build_schema``'s degrade-to-``str``. A tool
    whose hints will not resolve keeps today's behaviour rather than half of it."""
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 — forward refs etc.; degrade to no coercion
        hints = {}
    coercers: dict[str, _Coercer] = {}
    for pname, p in inspect.signature(func).parameters.items():
        if pname in ("self", "cls") or _is_ctx_param(p):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = hints.get(pname)
        if ann is None:
            continue
        coercer = _coercer_for(ann)
        if coercer is not None:
            coercers[pname] = coercer
    return coercers
