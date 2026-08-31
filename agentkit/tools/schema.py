"""JSON-schema inference helpers for tools.

These translate a Python callable's signature + type hints into the LLM-advertised
``ToolSchema`` (input schema) and best-effort infer an output schema from the
return-type annotation. Module-private by convention — exported only via the
package ``__init__`` if something elsewhere needs them.
"""

from __future__ import annotations

import collections.abc as _abc
import contextlib
import enum
import inspect
import re
import types as _types
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, is_typeddict

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

# Annotations that describe a JSON array / object, matched BY IDENTITY against
# either the annotation itself (bare ``list`` / ``Sequence``) or its
# ``get_origin`` (``list[int]`` / ``Sequence[int]``).
#
# The ``collections.abc`` half is the part that was missing, and its absence was
# not cosmetic: ``get_origin(Sequence[int])`` is ``collections.abc.Sequence``,
# which matched nothing and fell through to the ``{"type": "string"}`` fallback
# at the bottom of ``_json_type``. Since ``Sequence``/``Mapping`` are the
# idiomatic way to annotate a parameter a tool only reads, the schema told the
# model to send a JSON string for exactly the parameters written most carefully.
#
# Identity, never ``isinstance``/``issubclass``: ``str`` and ``bytes`` are both
# registered ``Sequence`` subclasses, so a subclass test here would advertise
# every string parameter in the library as an array.
_ARRAY_ANNOTATIONS: tuple[Any, ...] = (
    list,
    tuple,
    set,
    frozenset,
    _abc.Sequence,
    _abc.MutableSequence,
    _abc.Iterable,
    _abc.Collection,
    _abc.Set,
    _abc.MutableSet,
)
_OBJECT_ANNOTATIONS: tuple[Any, ...] = (dict, _abc.Mapping, _abc.MutableMapping)
_CTX_PARAMS = ("ctx", "context")  # injected with the RunContext, not advertised to the model
_MIN_DESCRIPTION_LEN = 30  # the floor below which the model can't reliably understand the tool

# A docstring line that is exactly ``---`` ends the model-facing part; everything
# below it is for whoever is editing the source. This is the escape hatch that makes
# "the whole docstring ships" safe to turn on under existing code: an author with a
# human-only tail keeps it, and says so in one line rather than restructuring the
# tool. Measured across ``agentkit/``, ``tests/``, ``examples/`` and ``docs/``: 139
# tools, and not one of them already contains a bare ``---`` line, so switching this
# on changed no existing description by a single byte.
_HUMAN_TAIL_MARKER = "---"

# Notes addressed to a developer, recognised where a note is actually written: at the
# start of a line, allowing for indentation and an optional list bullet.
#
# Column 0 alone is NOT enough, and the gap is not hypothetical. ``inspect.getdoc``
# dedents by the COMMON leading whitespace, so a note written one level in keeps its
# extra indent all the way through the dedent — and the most ordinary way to write one
# is as a bullet under a heading:
#
#     Notes on behaviour:
#       - returns the vendor's status verbatim
#       - TODO: this is O(n^2), rewrite before the Q3 launch
#
# Under a column-0-only pattern both the indent and the ``- `` stand between the marker
# and the anchor, the check stays silent, and the note ships to the model as an
# instruction. That is precisely the failure this check exists to stop, arriving by the
# route authors actually take.
#
# What is still deliberately NOT a note is a marker used as a word inside a sentence, so
# a to-do-list tool can say "Append a TODO: item to the checklist" — there the marker is
# preceded by prose, and the pattern admits only whitespace and a bullet. ``\b`` keeps
# "XXXL is a size" and "HACKS are documented below" out of it as well.
_DEV_NOTE_RE = re.compile(r"^[ \t]*(?:[-*+][ \t]+)?(TODO|FIXME|XXX|HACK)\b", re.MULTILINE)


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


def _element_annotation(ann: Any) -> Any | None:
    """The single element type of a homogeneous container annotation, or
    ``None`` when there isn't one to describe.

    ``None`` is returned — deliberately, not as a fallback — for the two cases
    where naming an element type would be a false claim:

    * **Unparameterised** (`list`, `Sequence`). It says nothing about its
      elements, so neither do we.
    * **A heterogeneous tuple** (`tuple[int, str]`). ``items`` means "every
      element is this", which is simply untrue of a fixed-shape pair. Only
      ``tuple[X, ...]`` — the "any number of X" spelling — has one element type.

    ``Any`` is also refused, since `{"items": {"type": "string"}}` derived from
    `list[Any]` would advertise a constraint the annotation explicitly declines
    to make.
    """
    args = get_args(ann)
    if not args:
        return None
    if get_origin(ann) is tuple:
        # `tuple[X, ...]` is homogeneous; `tuple[X, Y]` is not.
        if len(args) == 2 and args[1] is Ellipsis:
            return None if args[0] is Any else args[0]
        return None
    if len(args) != 1:
        return None
    return None if args[0] is Any else args[0]


def _mapping_value_annotation(ann: Any) -> Any | None:
    """The value type of a `dict[K, V]` / `Mapping[K, V]`, or ``None``."""
    args = get_args(ann)
    if len(args) != 2 or args[1] is Any:
        return None
    return args[1]


def _typeddict_fragment(ann: Any) -> dict[str, Any] | None:
    """A ``TypedDict`` → its real object schema, in the same shape
    ``_struct_fragment`` produces for a dataclass / Pydantic model / attrs class.

    ``TypedDict`` is the stdlib way to name a JSON object's shape, and it used to
    be advertised as ``{"type": "string"}``: the model sent a JSON string and the
    body's ``q["field"]`` raised ``TypeError: string indices must be integers``.

    It belongs on the schema side of the line and NOT the coercion side, which is
    the whole reason it can be handled here rather than through
    ``_struct_adapter``: a ``TypedDict`` *is* a ``dict`` at runtime, so the value
    ``json.loads`` produced is already the annotated type and there is nothing to
    reconstitute. ``__required_keys__`` carries ``total=`` and per-key
    ``Required``/``NotRequired``, so the ``required`` list stays honest in both
    directions — claiming a key is mandatory when the tool treats it as optional
    would be a worse lie than the one being fixed.
    """
    if not is_typeddict(ann):
        return None
    with contextlib.suppress(Exception):
        hints = get_type_hints(ann)
        required = getattr(ann, "__required_keys__", frozenset(hints))
        return {
            "type": "object",
            "properties": {key: _json_type(hint) for key, hint in hints.items()},
            "additionalProperties": False,
            "required": [key for key in hints if key in required],
        }
    return None


def _json_type(ann: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON-schema type fragment (best-effort, dependency-free)."""
    if ann in _JSON_PRIMITIVES:
        return {"type": _JSON_PRIMITIVES[ann]}
    # Bare `list` / `dict` / `Sequence` / `Mapping` (unsubscripted).
    if ann in _ARRAY_ANNOTATIONS:
        return {"type": "array"}
    if ann in _OBJECT_ANNOTATIONS:
        return {"type": "object"}
    typed_dict = _typeddict_fragment(ann)  # before the origin checks: it has none
    if typed_dict is not None:
        return typed_dict
    origin = get_origin(ann)
    # `list[X]` / `tuple[...]` / `set[X]` / `Sequence[X]` / `Iterable[X]` / …
    if origin in _ARRAY_ANNOTATIONS:
        element = _element_annotation(ann)
        if element is None:
            return {"type": "array"}
        return {"type": "array", "items": _json_type(element)}
    if origin in _OBJECT_ANNOTATIONS:  # `dict[K, V]` / `Mapping[K, V]`
        value = _mapping_value_annotation(ann)
        if value is None:
            return {"type": "object"}
        # Only the VALUE type is described. A JSON object's keys are strings by
        # definition, so `dict[int, X]` cannot be honoured on the wire and
        # claiming otherwise in the schema would be a lie the transport
        # guarantees to break.
        return {"type": "object", "additionalProperties": _json_type(value)}
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
        args = get_args(ann)
        real = [a for a in args if a is not type(None)]
        # ``X | None`` used to collapse to bare ``X``, silently dropping the
        # null arm. On a parameter WITH a default that was merely incomplete —
        # the model can omit the key. On a REQUIRED one it was a dead end: the
        # parameter is in ``required``, so the model must send something, and
        # the only thing the schema permitted was an ``X``. A tool meaning
        # "a category, or null for all of them" could not express the second
        # half, and the model could not choose it.
        #
        # ``anyOf`` rather than ``{"type": ["string", "null"]}`` because the arms
        # here are not always bare types — an enum arm is ``{"enum": [...],
        # "type": "string"}`` and a struct arm is a whole object schema, neither
        # of which fits in a type list.
        nullable = len(real) != len(args)
        if len(real) == 1 and not nullable:
            return _json_type(real[0])
        if real:
            frags = [_json_type(a) for a in real]
            if nullable:
                frags.append({"type": "null"})
            return {"anyOf": frags}
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return _enum_fragment(ann)
    struct = _struct_fragment(ann)
    if struct is not None:
        return struct
    return {"type": "string"}  # unknown / unannotated → string


def _model_facing_docstring(func: Callable[..., Any]) -> tuple[str, bool]:
    """``func``'s docstring as the model will actually see it, plus whether a human-only
    tail was cut off it.

    The whole docstring, not its first line. The first-line rule was the framework's
    single most expensive silent decision: everything below line one was read by people
    editing the source and by nobody else, while still sitting in the docstring looking
    as though it had been delivered. It cost a real defect — the sentence a model most
    needed, *use this only when the information genuinely cannot be obtained any other
    way*, was in ``ask_human``'s second paragraph and never shipped — and it also cut
    ORDINARY docstrings mid-clause, because a single paragraph wrapped over two source
    lines is two lines. Two tools in this repo's own suite were advertised to the model
    as ``"A tool, so the ReAct loop has something to run and a reason to"`` and
    ``"Look up a Hit for query `q`. (Will return a malformed payload to"``.

    The alternative considered was refusing a multi-paragraph docstring at decoration
    time so the author had to choose out loud, which is how this module treats a missing
    ``side_effecting=`` and a thin docstring. The measurement killed it. Across
    ``agentkit/``, ``tests/``, ``examples/`` and ``docs/`` there are 139 tools; 6 have a
    multi-paragraph docstring and in ALL SIX the second paragraph is model-facing
    guidance ("Use this whenever the user mentions an order number."), not an
    implementation note. Refusing them would have forced agentkit's own ``ask_human`` to
    delete the exact sentence this change exists to deliver, and would still have left
    the two mid-clause truncations in place, since those docstrings are single
    paragraphs. Taking everything costs +7% description bytes repo-wide (8307 -> 8892
    chars over all 139 tools) and 0 of them flip across the 30-char floor.

    That leaves one real risk, and it is downstream rather than here: an author who did
    write notes for humans now has them read as instructions by a model. ``---`` on a
    line of its own is the answer, and it is deliberately an opt-OUT rather than the
    opt-in it is tempting to reach for (``long_description=True``, say). An opt-in would
    make all 139 tools do paperwork to keep prose they already meant for the model; the
    opt-out is one line for the few who need it, and its absence is the common case.
    ``description=`` remains available for authors who would rather say it all at the
    decorator, and unlike an opt-in flag it duplicates nothing.

    Returns ``(text, tail_was_cut)``; the caller needs the second value only to explain
    an empty result.

    The CRLF re-clean is not defensive tidying, it is load-bearing. ``inspect.getdoc``
    dedents by finding common leading whitespace, and a ``\\r\\n`` docstring defeats it
    completely: measured, ``"Do the thing.\\r\\n\\r\\n    Indented second paragraph.\\r\\n"``
    comes back out of ``getdoc`` with its four-space indent intact and a trailing
    ``\\r``. Under the first-line rule that never showed, because the split threw the
    tail away. Now the whole text ships, so a docstring authored on Windows would put
    raw ``\\r`` bytes and phantom indentation into the JSON we hand the provider."""
    doc = inspect.getdoc(func) or ""
    if "\r" in doc:
        # Normalise, then re-run the dedent that the \r defeated the first time.
        # cleandoc on already-clean text is a no-op, so this costs nothing on the
        # overwhelmingly common path — which is why it is guarded by the `in` test.
        doc = inspect.cleandoc(doc.replace("\r\n", "\n").replace("\r", "\n"))
    lines = doc.split("\n")
    cut = False
    for i, line in enumerate(lines):
        if line.strip() == _HUMAN_TAIL_MARKER:
            lines = lines[:i]
            cut = True
            break
    # rstrip per line so trailing whitespace an editor left behind does not ride to
    # the provider; the overall strip drops the blank line the marker was sitting on.
    return "\n".join(line.rstrip() for line in lines).strip(), cut


def _resolved_description(func: Callable[..., Any], description: str | None) -> str:
    """Pick the description for `func`: explicit `description=` wins, else the WHOLE
    docstring down to a ``---`` line (see :func:`_model_facing_docstring` for why the
    whole thing, and for the measurement behind it).

    Explicit ``description=`` is passed through with nothing but a ``strip()`` — no
    marker hunt, no developer-note check. That asymmetry is the point: a docstring is
    text this module INTERPRETS, addressed to two audiences at once, and the rules here
    are about splitting it. ``description=`` is text the author handed the framework
    directly and meant every byte of; going looking for markers in it would be the
    framework second-guessing an explicit instruction.

    Two decoration-time refusals, both free to fix where they fire and neither of them
    reachable at call time:

    - The >=30-char floor, unchanged in spirit, but now measured on the whole chosen
      text rather than on line one. The total is what the model is shown, so the total
      is what has to clear the bar. This only ever widens what passes (a docstring whose
      first line is thin but whose body is substantial used to be rejected); measured,
      0 of this repo's 139 tools change verdict.
    - A developer note above the ``---`` line. This one is genuinely new and it is the
      other half of the whole-docstring decision: ``TODO: this is O(n^2)`` used to be
      invisible to the model and is now an instruction to it, so the moment it becomes
      shippable text it has to be either shipped deliberately or moved. Refusing is the
      same call the module already makes for a missing ``side_effecting=`` — the author
      knows which audience they were writing for and the framework cannot.

    Both messages name the tool and state the remedy, because a ``ToolDefinitionError``
    that fires at import in someone else's application is only useful if it can be acted
    on without reading this file."""
    if description is not None:
        desc = description.strip()
    else:
        desc, tail_cut = _model_facing_docstring(func)
        if (note := _DEV_NOTE_RE.search(desc)) is not None:
            name = getattr(func, "__name__", "tool")
            raise ToolDefinitionError(
                f"tool {name!r} has a developer note ({note.group(1)}) in the part of its "
                f"docstring that is sent to the model, where it reads as an instruction; "
                f"move it below a line containing only {_HUMAN_TAIL_MARKER!r} (everything "
                f"after that line is for humans) or pass an explicit description="
            )
    if len(desc) < _MIN_DESCRIPTION_LEN:
        name = getattr(func, "__name__", "tool")
        # Naming the cut is the difference between "got 0 chars" and a fixable report:
        # the author put the whole docstring on the human side of the marker.
        cut_note = (
            f" (everything below the {_HUMAN_TAIL_MARKER!r} line in its docstring is for"
            " humans and was not counted)"
            if description is None and tail_cut
            else ""
        )
        raise ToolDefinitionError(
            f"tool {name!r} needs a docstring/description of at least "
            f"{_MIN_DESCRIPTION_LEN} characters so the model can understand it; "
            f"got {len(desc)} chars: {desc!r}{cut_note}"
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
        if p.kind is p.VAR_POSITIONAL:
            # A tool is invoked with a JSON OBJECT, so every argument arrives by
            # name. There is no wire spelling for a positional one, which makes
            # ``*args`` permanently empty — not usually, always. Dropping it from
            # the schema in silence (the old behaviour) left a parameter that
            # reads as meaningful and can never receive anything.
            #
            # ``**kwargs`` is deliberately NOT covered by this: the model CAN
            # send keys the signature does not name, and that is the documented
            # way to accept them.
            raise ToolDefinitionError(
                f"tool {name!r} declares `*{pname}`, which can never be filled: a tool is "
                "called with a JSON object, so every argument arrives by keyword and no "
                "positional one can be sent. Name the parameters you expect, or accept "
                "**kwargs if the tool genuinely takes arbitrary keys."
            )
        if p.kind is p.VAR_KEYWORD:
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


#: Container origins whose annotation names a type JSON cannot produce, so the
#: decoded list has to be rebuilt into it. `list` is absent because JSON already
#: decodes to one, and the abstract ABCs are absent because a `list` already
#: satisfies `Sequence` / `Iterable` / `Collection` — rebuilding those would be
#: inventing a concrete choice the author deliberately did not make.
_REBUILT_CONTAINERS: dict[Any, Any] = {set: set, frozenset: frozenset, tuple: tuple}


def _sequence_coercer(ann: Any, origin: Any) -> _Coercer | None:
    """Coerce a sequence's ELEMENTS, and rebuild the container the annotation
    actually asked for.

    Two independent reasons this can be needed, and either alone is enough:

    * the element type needs reconstituting (`list[Unit]` — JSON hands us
      strings, the body annotated members);
    * the container type is one JSON has no spelling for (`set[str]` — JSON
      only has arrays, so without this the body annotated `set` and received
      `list`).
    """
    element = _element_annotation(ann)
    inner = _coercer_for(element) if element is not None else None
    rebuild = _REBUILT_CONTAINERS.get(origin)
    if inner is None and rebuild is None:
        return None

    def coerce(value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            # Not a sequence at all. Leave it exactly as it arrived — the same
            # rule the scalar coercers follow, so a provider that ignored the
            # schema produces a visible failure rather than a laundered value.
            return value
        items = [inner(v) for v in value] if inner is not None else list(value)
        return rebuild(items) if rebuild is not None else items

    return coerce


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

    ``list[Unit]`` is the exception, and it is the one the schema now earns.
    ``_json_type`` describes the element type as ``items``, so the model HAS been
    shown the contract and honouring it is owed rather than invented. The two
    halves move together — element coercion without ``items`` would enforce a
    promise never made, and ``items`` without coercion would make a promise never
    kept."""
    origin = get_origin(ann)
    if origin in _ARRAY_ANNOTATIONS:
        return _sequence_coercer(ann, origin)
    if origin in _OBJECT_ANNOTATIONS:
        value = _mapping_value_annotation(ann)
        coerce_value = _coercer_for(value) if value is not None else None
        if coerce_value is None:
            return None
        each: _Coercer = coerce_value
        return lambda v: {k: each(x) for k, x in v.items()} if isinstance(v, dict) else v
    if origin is Union or origin is getattr(_types, "UnionType", None):
        real = [a for a in get_args(ann) if a is not type(None)]
        if len(real) == 1:
            # ``Optional[X]`` / ``X | None``. ``_json_type`` now advertises this
            # as ``anyOf: [X, null]``, so both arms are honoured here: an ``X``
            # is coerced as an ``X``, and ``None`` passes straight through
            # because ``None`` is what the annotation's other half is FOR and
            # the schema explicitly permits it. ``unit: Unit | None = None`` called without
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
