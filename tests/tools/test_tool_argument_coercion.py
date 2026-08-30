"""The types a tool ADVERTISES are the types its body RECEIVES.

``schema.py`` does real work to describe a parameter honestly: an ``Enum``
becomes an ``enum`` fragment listing its members, a ``Literal`` becomes its
values, a dataclass / Pydantic model / attrs class becomes a full object schema.
Nothing then reconstituted the incoming value, so the body got whatever
``json.loads`` had produced. Measured on the canonical case::

    schema advertises: {'enum': ['celsius', 'fahrenheit'], 'type': 'string'}
    body receives    : str='celsius'

An author who wrote ``unit.value`` — the only reasonable thing to write, given
the annotation they had just typed — got ``AttributeError`` from inside their own
tool. And the quieter half was worse: ``unit="kelvin"`` ran to completion, with
the body seeing a string the advertised enum excluded, because nothing checked
either.

These tests pin both halves of the fix — the value arrives as the annotated type,
and a value that cannot become that type is refused as a ``ToolArgumentError``
(the "the caller sent something wrong" error the retry/repair path already
understands) rather than crashing the body or slipping through.

The controls matter as much as the regressions: this changes the CALL path and
must not touch the SCHEMA path, must not touch primitives, and must not disturb
``ctx`` injection or ``**kwargs``.
"""

# NOTE: deliberately NO ``from __future__ import annotations`` here. Several
# cases below define their struct/enum inside the test function, and under PEP
# 563 the annotation is stored as the STRING ``"Span"``, which
# ``get_type_hints`` cannot resolve from a function-local scope. That degrade is
# real and consistent — the schema falls back to ``{"type": "string"}`` and
# coercion correctly declines, both halves agreeing — and it is pinned by
# ``test_an_unresolvable_annotation_degrades_on_both_sides`` at the bottom.
# Keeping the rest of the file on real annotation objects is what lets those
# cases test coercion rather than re-test the degrade.

import asyncio
import enum
import json
from collections.abc import (
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import pytest

from agentkit.kernel.protocols import Ctx
from agentkit.runtime import RunContext
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import ToolArgumentError, ToolDefinitionError, tool

CTX = make_test_ctx(llm=FakeLLM("x"))


def _run(t: Any, args: Any) -> Any:
    return asyncio.run(t.run(args, CTX))


class Unit(enum.Enum):
    C = "celsius"
    F = "fahrenheit"


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 2


@dataclass
class Filter:
    field: str
    limit: int = 5


@tool(side_effecting=False)
async def weather(city: str, unit: Unit) -> str:
    """Report the weather for a city in the requested unit of measure."""
    return f"{type(unit).__name__}:{unit.value}"


# ── 1. the advertised type is the type the body gets ────────────────────────


def test_an_enum_parameter_arrives_as_the_enum_member() -> None:
    """THE regression. Before: ``str:celsius`` was impossible — ``unit.value``
    raised ``AttributeError: 'str' object has no attribute 'value'`` inside the
    tool body, on a value the schema had told the model was a ``Unit``."""
    assert _run(weather, {"city": "SF", "unit": "celsius"}) == "Unit:celsius"


def test_an_int_enum_parameter_arrives_as_the_member_not_the_int() -> None:
    """The member value can be any JSON primitive; the schema advertises
    ``{"enum": [1, 2], "type": "integer"}`` and the body still gets the member."""

    @tool(side_effecting=False)
    def escalate(level: Level) -> str:
        """Escalate the incident to the on-call rota at the given level."""
        return f"{type(level).__name__}:{level.name}"

    assert _run(escalate, {"level": 2}) == "Level:HIGH"


def test_a_structured_parameter_arrives_as_the_struct() -> None:
    """Before: ``dict:{'field': 'name', 'limit': 2}``. The schema advertised a
    full object with ``properties`` and ``required``; the body got the raw dict
    and ``flt.field`` raised."""

    @tool(side_effecting=False)
    def find(q: str, flt: Filter) -> str:
        """Find records matching the query, narrowed by the structured filter."""
        return f"{type(flt).__name__}:{flt.field}/{flt.limit}"

    assert _run(find, {"q": "x", "flt": {"field": "name", "limit": 2}}) == "Filter:name/2"


def test_a_struct_parameter_fills_its_own_defaults() -> None:
    """The adapter builds a real instance, so a field the model omitted takes the
    struct's default — not a ``KeyError`` the first time the body reads it."""

    @tool(side_effecting=False)
    def find(flt: Filter) -> str:
        """Find records matching only the structured filter supplied here."""
        return f"{flt.field}/{flt.limit}"

    assert _run(find, {"flt": {"field": "name"}}) == "name/5"


def test_a_pydantic_parameter_arrives_as_the_model() -> None:
    pydantic = pytest.importorskip("pydantic")

    class Query(pydantic.BaseModel):
        text: str
        top_k: int = 3

    @tool(side_effecting=False)
    def ask(query: Query) -> str:
        """Ask the retrieval index using the structured query object given."""
        return f"{type(query).__name__}:{query.text}/{query.top_k}"

    assert _run(ask, {"query": {"text": "hi"}}) == "Query:hi/3"


def test_an_attrs_parameter_arrives_as_the_class() -> None:
    attrs = pytest.importorskip("attrs")

    @attrs.define
    class Span:
        start: int
        end: int

    @tool(side_effecting=False)
    def slice_it(span: Span) -> str:
        """Slice the document down to the span of offsets supplied here."""
        return f"{type(span).__name__}:{span.start}-{span.end}"

    assert _run(slice_it, {"span": {"start": 1, "end": 4}}) == "Span:1-4"


# ── 2. the failure mode is ToolArgumentError, never a raw ValueError ────────


def test_an_off_schema_enum_value_is_refused_by_name() -> None:
    """The other half of the gap, and the quieter one. Before, this RAN: the
    body received ``str='kelvin'``, a value the advertised enum excluded, and
    nothing anywhere reported it."""
    with pytest.raises(ToolArgumentError) as exc:
        _run(weather, {"city": "SF", "unit": "kelvin"})
    msg = str(exc.value)
    assert "weather" in msg  # the TOOL, not the Python function
    assert "unit" in msg  # which argument
    assert "celsius" in msg and "fahrenheit" in msg  # what would have worked


def test_the_rejection_is_a_tool_argument_error_not_a_value_error_from_the_enum() -> None:
    """``Unit('kelvin')`` raises ``ValueError: 'kelvin' is not a valid Unit`` —
    which reads as the tool crashing. ``ToolArgumentError`` is the type the
    retry/repair path already routes back to the model as a fixable bad call.
    It still subclasses ``ValueError``, so the surrounding isolation is intact."""
    with pytest.raises(ToolArgumentError) as exc:
        _run(weather, {"city": "SF", "unit": "kelvin"})
    assert isinstance(exc.value, ValueError)
    assert exc.value.tool_name == "weather"
    assert [k for k, _ in exc.value.invalid] == ["unit"]


def test_an_unparseable_struct_is_refused_with_the_field_diagnostics() -> None:
    """A dict missing a field the object schema marked ``required`` used to pass
    straight through; the body then raised on the first attribute access."""

    @tool(side_effecting=False)
    def find(flt: Filter) -> str:
        """Find records matching only the structured filter supplied here."""
        return flt.field

    with pytest.raises(ToolArgumentError) as exc:
        _run(find, {"flt": {"limit": 2}})
    assert "field" in str(exc.value)  # names the offending FIELD, not just the arg


def test_every_bad_argument_is_reported_in_one_go() -> None:
    """A repair loop that learns one mistake per round trip burns a turn and a
    model call per mistake. Both bad values come back in the same rejection."""

    @tool(side_effecting=False)
    def pair(unit: Unit, level: Level) -> str:
        """Report a reading in the given unit at the given escalation level."""
        return "ok"

    with pytest.raises(ToolArgumentError) as exc:
        _run(pair, {"unit": "kelvin", "level": 99})
    assert [k for k, _ in exc.value.invalid] == ["unit", "level"]


def test_a_missing_required_argument_still_wins_over_a_coercion_complaint() -> None:
    """Precedence: a call missing ``city`` should be told it is missing ``city``.
    Diagnosing the ``unit`` it also got wrong first would bury the lede."""
    with pytest.raises(ToolArgumentError) as exc:
        _run(weather, {"unit": "kelvin"})
    assert exc.value.missing == ("city",)
    assert exc.value.invalid == ()


# ── 3. Literal is checked, and checked type-exactly ─────────────────────────


def test_a_literal_value_passes_through_unchanged() -> None:
    """``Literal`` members are already JSON primitives — there is nothing to
    convert. What was missing was anyone checking."""

    @tool(side_effecting=False)
    def render(mode: Literal["fast", "pretty"]) -> str:
        """Render the report in one of the two supported output modes."""
        return f"{type(mode).__name__}:{mode}"

    assert _run(render, {"mode": "fast"}) == "str:fast"


def test_an_off_schema_literal_is_refused() -> None:
    @tool(side_effecting=False)
    def render(mode: Literal["fast", "pretty"]) -> str:
        """Render the report in one of the two supported output modes."""
        return mode

    with pytest.raises(ToolArgumentError) as exc:
        _run(render, {"mode": "sloppy"})
    assert "fast" in str(exc.value) and "pretty" in str(exc.value)


def test_a_bool_is_not_accepted_for_an_integer_literal() -> None:
    """``True == 1`` in Python, so a loose ``in`` test would admit ``True`` for
    ``Literal[1, 2, 3]`` and hand the body a ``bool`` where the annotation — and
    the advertised ``{"type": "integer"}`` — promised an ``int``."""
    assert True in (1, 2, 3)  # the trap, stated

    @tool(side_effecting=False)
    def retry(attempts: Literal[1, 2, 3]) -> str:
        """Retry the failed operation the given number of times at most."""
        return f"{type(attempts).__name__}"

    assert _run(retry, {"attempts": 2}) == "int"
    with pytest.raises(ToolArgumentError):
        _run(retry, {"attempts": True})


# ── 4. optional, defaulted, and genuinely-union parameters ─────────────────


def test_an_optional_enum_coerces_its_value_and_passes_none_through() -> None:
    """``_json_type`` collapses ``X | None`` to bare ``X``, so coercion collapses
    identically — with ``None`` untouched, because ``None`` is what the other
    half of the annotation is FOR."""

    @tool(side_effecting=False)
    def report(unit: Unit | None = None) -> str:
        """Report a reading, optionally converted into the given unit first."""
        return f"{type(unit).__name__}"

    assert _run(report, {"unit": "fahrenheit"}) == "Unit"
    assert _run(report, {"unit": None}) == "NoneType"  # explicit JSON null
    assert _run(report, {}) == "NoneType"  # omitted → the default


def test_a_defaulted_enum_keeps_its_default_untouched_when_omitted() -> None:
    """The default is a real ``Unit`` member the author wrote. An omitted
    argument never reaches a coercer, so it cannot be mangled into one."""

    @tool(side_effecting=False)
    def report(unit: Unit = Unit.C) -> str:
        """Report a reading in the given unit, defaulting to celsius always."""
        return f"{type(unit).__name__}:{unit.value}"

    assert _run(report, {}) == "Unit:celsius"
    assert _run(report, {"unit": "fahrenheit"}) == "Unit:fahrenheit"


def test_a_genuine_multi_member_union_is_left_alone() -> None:
    """``_json_type`` advertises ``anyOf`` precisely because it refuses to pick a
    member. Picking one here would be the same guess made silently — "celsius"
    inhabits both arms of ``Unit | str`` and nothing knows which was meant."""

    @tool(side_effecting=False)
    def loose(u: Unit | str) -> str:
        """Accept either a unit member or a free-form unit string here."""
        return f"{type(u).__name__}"

    assert _run(loose, {"u": "celsius"}) == "str"  # unchanged, as advertised


# ── 5. idempotence: an in-process caller passing the real thing ────────────


def test_passing_a_real_enum_member_is_a_no_op() -> None:
    """Not every caller is a model. Tests, ``ToolBackedMemory`` and one agent
    calling another's tool pass the typed value directly, and must not be
    punished for it with a double-conversion error."""
    assert _run(weather, {"city": "SF", "unit": Unit.F}) == "Unit:fahrenheit"


def test_passing_a_real_struct_instance_is_a_no_op() -> None:
    @tool(side_effecting=False)
    def find(flt: Filter) -> str:
        """Find records matching only the structured filter supplied here."""
        return f"{flt.field}/{flt.limit}"

    built = Filter(field="name", limit=9)
    assert _run(find, {"flt": built}) == "name/9"


# ── 6. controls: what must NOT have changed ────────────────────────────────


def test_primitive_and_unannotated_parameters_are_untouched() -> None:
    """No coercion is invented for anything the schema describes with a plain
    ``{"type": ...}``. In particular nothing turns ``"3"`` into ``3`` — a
    provider that sent a string for an integer parameter has a bug the framework
    must surface, not launder."""

    @tool(side_effecting=False)
    def kitchen(s: str, i: int, f: float, b: bool, x: Any, u=None) -> str:  # noqa: ANN001
        """Accept one of every primitive parameter kind plus an unannotated one."""
        return ",".join(type(v).__name__ for v in (s, i, f, b, x, u))

    out = _run(kitchen, {"s": "1", "i": 2, "f": 3.5, "b": True, "x": {"k": 1}, "u": "raw"})
    assert out == "str,int,float,bool,dict,str"
    assert _run(kitchen, {"s": "1", "i": "3", "f": 1.0, "b": False, "x": None, "u": 7}) == (
        "str,str,float,bool,NoneType,int"  # "3" stays a str; nothing is laundered
    )


def test_kwargs_passthrough_still_works_and_extras_are_not_coerced() -> None:
    """Extras admitted by ``**kwargs`` have no annotation behind them, so there
    is nothing to honour — they arrive exactly as they were sent."""

    @tool(side_effecting=False)
    def flexible(unit: Unit, **extra: object) -> str:
        """Accept arbitrary extra keyword arguments alongside the unit given."""
        return f"{type(unit).__name__}+{sorted(extra)}+{type(extra['b']).__name__}"

    assert _run(flexible, {"unit": "celsius", "b": "celsius"}) == "Unit+['b']+str"


def test_ctx_injection_survives_all_three_spellings() -> None:
    """The ctx path was fixed separately to accept the ``Ctx`` protocol. Coercion
    runs before injection and ``_build_coercers`` skips ctx parameters via the
    same ``_is_ctx_param`` predicate that keeps them out of the schema, so none
    of the three spellings can acquire a coercer."""
    seen: dict[str, Any] = {}

    @tool(side_effecting=False)
    def bare(unit: Unit, ctx=None) -> str:  # noqa: ANN001
        """Report a reading using an unannotated injected context parameter."""
        seen["bare"] = ctx
        return "ok"

    @tool(side_effecting=False)
    def runctx(unit: Unit, ctx: RunContext) -> str:
        """Report a reading using a RunContext annotated context parameter."""
        seen["runctx"] = ctx
        return "ok"

    @tool(side_effecting=False)
    def protocol(unit: Unit, ctx: Ctx) -> str:
        """Report a reading using the Ctx protocol annotated context parameter."""
        seen["protocol"] = ctx
        return "ok"

    for t in (bare, runctx, protocol):
        assert "ctx" not in t.schema.parameters["properties"]
        assert "ctx" not in t.schema.parameters["required"]
        assert _run(t, {"unit": "celsius"}) == "ok"
    assert seen == {"bare": CTX, "runctx": CTX, "protocol": CTX}


def test_a_real_data_param_named_context_is_still_advertised_and_passed() -> None:
    """``context: str`` is a data parameter, not the injected one — it stays in
    the schema, is passed normally, and (being a ``str``) is not coerced."""

    @tool(side_effecting=False)
    def summarise(text: str, context: str) -> str:
        """Summarise the text, taking the surrounding context into account."""
        return f"{type(context).__name__}:{context}"

    assert summarise.schema.parameters["properties"]["context"] == {"type": "string"}
    assert "context" in summarise.schema.parameters["required"]
    assert _run(summarise, {"text": "t", "context": "c"}) == "str:c"


def test_the_advertised_schema_is_byte_identical() -> None:
    """This changes the CALL path, not the ADVERTISED one. The exact JSON the
    model is shown for every branch of ``_json_type`` is pinned here, so a future
    edit to the coercion side cannot quietly move the schema side with it."""

    @tool(side_effecting=False)
    def everything(
        s: str,
        i: int,
        u: Unit,
        lit: Literal["a", "b"],
        flt: Filter,
        opt: Unit | None = None,
        mix: int | str = 1,
    ) -> str:
        """Exercise every annotation branch the schema inference layer knows."""
        return s

    assert json.dumps(everything.schema.parameters, sort_keys=True) == json.dumps(
        {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "u": {"enum": ["celsius", "fahrenheit"], "type": "string"},
                "lit": {"enum": ["a", "b"], "type": "string"},
                "flt": {
                    "type": "object",
                    "properties": {"field": {"type": "string"}, "limit": {"type": "integer"}},
                    "additionalProperties": False,
                    "required": ["field"],
                },
                "opt": {
                    "anyOf": [
                        {"enum": ["celsius", "fahrenheit"], "type": "string"},
                        {"type": "null"},
                    ]
                },
                "mix": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            },
            "required": ["s", "i", "u", "lit", "flt"],
        },
        sort_keys=True,
    )


def test_an_unresolvable_annotation_degrades_on_both_sides() -> None:
    """``get_type_hints`` failing (a forward ref to a name that never lands, a
    circular import) makes ``_build_schema`` degrade the parameter to
    ``{"type": "string"}``. Coercion degrades in lockstep — no coercer at all —
    so such a tool keeps exactly today's behaviour rather than half of it. The
    alternative, guessing from the raw string annotation, would be the two sides
    disagreeing again, which is the bug this file exists to close."""

    @tool(side_effecting=False)
    def ghost(x: "NeverDefinedAnywhere") -> str:  # type: ignore[name-defined] # noqa: F821
        """Accept a parameter whose annotation can never be resolved at all."""
        return type(x).__name__

    assert ghost.schema.parameters["properties"]["x"] == {"type": "string"}
    assert _run(ghost, {"x": {"still": "a dict"}}) == "dict"


# ── 6. abstract collections and TypedDict describe their real shape ─────────


def test_abstract_sequence_and_mapping_are_not_advertised_as_strings() -> None:
    """The same bug this file exists to close, one type-family over.

    ``_json_type`` recognised the CONCRETE generics — ``list[int]`` became
    ``{"type": "array"}``, ``dict[str, int]`` became ``{"type": "object"}`` —
    by matching ``get_origin(ann)`` against ``(list, tuple, set, frozenset)``
    and ``dict``. But ``get_origin(Sequence[int])`` is
    ``collections.abc.Sequence``, which matched nothing and fell through to the
    ``{"type": "string"}`` fallback at the bottom of the function.

    ``Sequence``/``Mapping`` are the idiomatic annotations for a parameter the
    tool only reads, so the failure landed on careful code. Measured before the
    fix, on a tool whose body was ``sum(values)``::

        advertises: {"values": {"type": "string"}}
        model sends: {"values": "[1, 2, 3]"}
        body raises: TypeError: unsupported operand type(s) for +: 'int' and 'str'

    Not even a ``ToolArgumentError`` — a raw ``TypeError`` from inside the
    author's own function, on a value the schema had instructed the model to
    send."""

    @tool(side_effecting=False)
    def totals(values: Sequence[int], weights: Mapping[str, int]) -> str:
        """Total the values and the weights, returning both sums as a string."""
        return f"{sum(values)}/{sum(weights.values())}"

    props = totals.schema.parameters["properties"]
    assert props["values"] == {"type": "array", "items": {"type": "integer"}}
    assert props["weights"] == {"type": "object", "additionalProperties": {"type": "integer"}}

    # And the body receives what the schema promised.
    assert _run(totals, {"values": [1, 2, 3], "weights": {"a": 4}}) == "6/4"


def test_other_abstract_collection_origins_describe_their_json_kind() -> None:
    """The rest of the ``collections.abc`` family a tool author reaches for.
    ``Iterable``/``Collection``/``AbstractSet``/``MutableSequence`` are arrays;
    ``MutableMapping`` is an object. All seven used to be ``"string"``."""

    @tool(side_effecting=False)
    def shapes(
        a: Iterable[int],
        b: Collection[str],
        c: AbstractSet[int],
        d: MutableSequence[int],
        e: MutableMapping[str, int],
    ) -> str:
        """Accept the abstract collection flavours and report how many arrived."""
        return "ok"

    props = shapes.schema.parameters["properties"]
    assert props["a"] == {"type": "array", "items": {"type": "integer"}}
    assert props["b"] == {"type": "array", "items": {"type": "string"}}
    assert props["c"] == {"type": "array", "items": {"type": "integer"}}
    assert props["d"] == {"type": "array", "items": {"type": "integer"}}
    assert props["e"] == {"type": "object", "additionalProperties": {"type": "integer"}}


def test_a_typeddict_parameter_advertises_its_properties() -> None:
    """A ``TypedDict`` is the stdlib way to name a JSON object's shape, and it
    was advertised as ``{"type": "string"}`` — so the model sent a JSON string
    and ``n["values"]`` raised ``TypeError: string indices must be integers``.

    It gets the same treatment a dataclass already got: the real object schema.
    No coercer is needed on the way back, because a ``TypedDict`` IS a ``dict``
    at runtime — which is exactly why it belongs on the schema side of the line
    and not the coercion side."""

    class Query(TypedDict):
        field: str
        limit: int

    @tool(side_effecting=False)
    def search(q: Query) -> str:
        """Search records using the structured query object supplied by caller."""
        return f"{type(q).__name__}:{q['field']}/{q['limit']}"

    assert search.schema.parameters["properties"]["q"] == {
        "type": "object",
        "properties": {"field": {"type": "string"}, "limit": {"type": "integer"}},
        "additionalProperties": False,
        "required": ["field", "limit"],
    }
    assert _run(search, {"q": {"field": "name", "limit": 2}}) == "dict:name/2"


def test_a_total_false_typeddict_marks_its_keys_optional() -> None:
    """``total=False`` means every key may be absent, so none are ``required``.
    Getting this wrong in the other direction would be worse than the original
    bug: the model would be told a key is mandatory when the tool treats it as
    optional."""

    class Partial(TypedDict, total=False):
        field: str
        limit: int

    @tool(side_effecting=False)
    def maybe(p: Partial) -> str:
        """Accept a partial query object where every single key is optional."""
        return str(sorted(p))

    assert maybe.schema.parameters["properties"]["p"]["required"] == []


def test_bare_abstract_collections_still_describe_their_kind() -> None:
    """Unsubscripted ``Sequence`` / ``Mapping`` have no ``get_origin``, so they
    take the identity branch alongside bare ``list`` / ``dict``."""

    @tool(side_effecting=False)
    def bare(a: Sequence, b: Mapping) -> str:
        """Accept unparameterised abstract collections and report their arrival."""
        return "ok"

    props = bare.schema.parameters["properties"]
    assert props["a"] == {"type": "array"}
    assert props["b"] == {"type": "object"}


def test_str_and_bytes_are_not_mistaken_for_sequences() -> None:
    """The control that keeps the fix honest. ``str`` and ``bytes`` are both
    registered ``Sequence`` subclasses, and a fix written as an ``isinstance``
    check against the abc would have turned every string parameter in the
    library into an array."""

    @tool(side_effecting=False)
    def text(a: str, b: bytes) -> str:
        """Accept a string and a bytes parameter and report what came through."""
        return "ok"

    props = text.schema.parameters["properties"]
    assert props["a"] == {"type": "string"}
    assert props["b"] == {"type": "string"}


# ── 7. containers describe their element type, and honour it ────────────────


def test_a_list_of_enums_advertises_items_and_arrives_coerced() -> None:
    """The other half of the promise this file is built on, finally kept for
    containers.

    ``list[Unit]`` used to advertise a bare ``{"type": "array"}``. The element
    type was never shown to the model, so coercing the elements would have been
    enforcing a contract the model was never given — which is why the coercer
    correctly declined, and why the old behaviour was consistent rather than
    merely incomplete.

    Now the schema names the element type, so the coercion is owed. Both halves
    move together; neither is correct alone."""

    @tool(side_effecting=False)
    def many(units: list[Unit]) -> str:
        """Report readings for each of the units of measure supplied here."""
        return f"{type(units).__name__}[{type(units[0]).__name__}]"

    assert many.schema.parameters["properties"]["units"] == {
        "type": "array",
        "items": {"enum": ["celsius", "fahrenheit"], "type": "string"},
    }
    assert _run(many, {"units": ["celsius"]}) == "list[Unit]"


def test_a_bad_element_is_refused_as_a_tool_argument_error() -> None:
    """A value the advertised ``items`` excludes must fail the same way a bad
    scalar does — as `ToolArgumentError`, which the retry/repair path
    understands — not as a raw exception from inside the tool body."""

    @tool(side_effecting=False)
    def many(units: list[Unit]) -> str:
        """Report readings for each of the units of measure supplied here."""
        return "unreachable"

    with pytest.raises(ToolArgumentError):
        _run(many, {"units": ["kelvin"]})


def test_scalar_element_types_are_described_but_not_laundered() -> None:
    """``items`` on a scalar element type is advertisement only. Nothing turns
    ``"3"`` into ``3`` — a provider that ignores the schema has a bug the
    framework surfaces rather than hides, exactly as for a top-level scalar."""

    @tool(side_effecting=False)
    def totals(values: list[int]) -> str:
        """Total the integer values supplied and report the element types."""
        return ",".join(type(v).__name__ for v in values)

    assert totals.schema.parameters["properties"]["values"] == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert _run(totals, {"values": [1, "2"]}) == "int,str"


def test_a_dict_describes_its_value_type() -> None:
    """The mapping equivalent of ``items``. The key type is not described
    because JSON object keys are always strings."""

    @tool(side_effecting=False)
    def readings(by_city: dict[str, Unit]) -> str:
        """Report the unit of measure recorded against each city named here."""
        return type(by_city["sf"]).__name__

    assert readings.schema.parameters["properties"]["by_city"] == {
        "type": "object",
        "additionalProperties": {"enum": ["celsius", "fahrenheit"], "type": "string"},
    }
    assert _run(readings, {"by_city": {"sf": "celsius"}}) == "Unit"


def test_the_container_type_matches_the_annotation() -> None:
    """JSON has one sequence type. The annotation may ask for a different one,
    and the body should get what it annotated — the same rule the top-level
    coercers already follow."""

    @tool(side_effecting=False)
    def shapes(a: set[Unit], b: frozenset[Unit], c: tuple[Unit, ...]) -> str:
        """Accept the same units through three different container flavours."""
        return f"{type(a).__name__},{type(b).__name__},{type(c).__name__}"

    got = _run(shapes, {"a": ["celsius"], "b": ["celsius"], "c": ["celsius"]})
    assert got == "set,frozenset,tuple"


def test_abstract_containers_stay_lists() -> None:
    """A `list` already satisfies `Sequence` / `Iterable`, so there is nothing
    to rebuild and rebuilding anyway would be inventing a choice the author
    did not make."""

    @tool(side_effecting=False)
    def readings(units: Sequence[Unit]) -> str:
        """Report the readings for the sequence of units of measure given."""
        return f"{type(units).__name__}[{type(units[0]).__name__}]"

    assert readings.schema.parameters["properties"]["units"] == {
        "type": "array",
        "items": {"enum": ["celsius", "fahrenheit"], "type": "string"},
    }
    assert _run(readings, {"units": ["celsius"]}) == "list[Unit]"


def test_a_heterogeneous_tuple_declines_to_describe_items() -> None:
    """``items`` says "every element is this". That is false for
    ``tuple[int, str]``, so nothing is claimed rather than something wrong."""

    @tool(side_effecting=False)
    def pair(p: tuple[int, str]) -> str:
        """Accept a fixed two-element pair of an integer and a string value."""
        return "ok"

    assert pair.schema.parameters["properties"]["p"] == {"type": "array"}


def test_an_unparameterised_container_describes_no_items() -> None:
    """Bare `list` says nothing about its elements, so neither do we."""

    @tool(side_effecting=False)
    def anything(xs: list, ys: dict) -> str:
        """Accept an unparameterised list and dict of entirely unknown shape."""
        return "ok"

    props = anything.schema.parameters["properties"]
    assert props["xs"] == {"type": "array"}
    assert props["ys"] == {"type": "object"}


def test_nested_containers_describe_all_the_way_down() -> None:
    """`items` recurses, because `_json_type` is what builds it."""

    @tool(side_effecting=False)
    def grid(rows: list[list[Unit]]) -> str:
        """Accept a grid of unit-of-measure values arranged as nested rows."""
        return f"{type(rows[0][0]).__name__}"

    assert grid.schema.parameters["properties"]["rows"] == {
        "type": "array",
        "items": {"type": "array", "items": {"enum": ["celsius", "fahrenheit"], "type": "string"}},
    }
    assert _run(grid, {"rows": [["celsius"]]}) == "Unit"


# ── 8. Optional says so ─────────────────────────────────────────────────────


def test_an_optional_parameter_advertises_null() -> None:
    """``X | None`` used to be advertised as bare ``X``, dropping the null arm.

    For a parameter with a default that was merely incomplete — the model can
    omit the key and the default applies. For a REQUIRED one it was a genuine
    dead end: `category: str | None` with no default is in `required`, so the
    model must send something, and the only thing the schema permitted was a
    string. A tool meaning "a category, or null for all of them" had no way to
    say the second half, and the model had no way to choose it."""

    @tool(side_effecting=False)
    def search(q: str, category: str | None) -> str:
        """Search records for the query, narrowed to a category or all of them."""
        return f"{q}/{category}"

    props = search.schema.parameters["properties"]
    assert props["category"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert "category" in search.schema.parameters["required"]
    assert _run(search, {"q": "x", "category": None}) == "x/None"


def test_a_multi_member_optional_union_keeps_every_arm() -> None:
    """The null arm is appended to the existing `anyOf`, not substituted for
    the members already there."""

    @tool(side_effecting=False)
    def mixed(v: int | str | None) -> str:
        """Accept an integer, a string, or nothing at all for this parameter."""
        return type(v).__name__

    assert mixed.schema.parameters["properties"]["v"] == {
        "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]
    }


def test_an_optional_enum_still_coerces_and_still_accepts_null() -> None:
    """Both halves survive: a member arrives as the member, and an explicit
    null arrives as `None` rather than being refused."""

    @tool(side_effecting=False)
    def reading(unit: Unit | None = None) -> str:
        """Report a reading in the unit of measure given, or the default one."""
        return type(unit).__name__

    assert reading.schema.parameters["properties"]["unit"] == {
        "anyOf": [{"enum": ["celsius", "fahrenheit"], "type": "string"}, {"type": "null"}]
    }
    assert _run(reading, {"unit": "celsius"}) == "Unit"
    assert _run(reading, {"unit": None}) == "NoneType"
    assert _run(reading, {}) == "NoneType"


def test_a_non_optional_union_gains_no_null_arm() -> None:
    """The control. `int | str` does not accept None, and must not claim to."""

    @tool(side_effecting=False)
    def strict(v: int | str) -> str:
        """Accept either an integer or a string, but never a null value here."""
        return type(v).__name__

    assert strict.schema.parameters["properties"]["v"] == {
        "anyOf": [{"type": "integer"}, {"type": "string"}]
    }


# ── 9. `*args` cannot be filled, so it is refused ───────────────────────────


def test_a_var_positional_parameter_is_refused_at_definition() -> None:
    """A tool is called with a JSON object, so every argument arrives by NAME.
    There is no wire spelling for a positional one, which makes ``*args``
    permanently empty — not usually, always.

    It used to be dropped from the schema in silence, leaving a parameter that
    reads as meaningful and can never receive anything. This class already
    refuses a tool with no docstring and one without an explicit
    ``side_effecting=``; a parameter that cannot exist belongs in the same
    bucket, and decoration time is where it is free to fix."""

    with pytest.raises(ToolDefinitionError, match=r"\*rest"):

        @tool(side_effecting=False)
        def variadic(a: int, *rest: int) -> str:
            """Total the first integer with any additional ones supplied."""
            return "unreachable"


def test_var_keyword_is_still_allowed() -> None:
    """The control, and the reason this is not a blanket ban on variadics.
    ``**kwargs`` IS reachable — the model can send keys the signature does not
    name — and the framework documents it as the way to accept them."""

    @tool(side_effecting=False)
    def flexible(unit: Unit, **extra: object) -> str:
        """Accept arbitrary extra keyword arguments alongside the unit given."""
        return f"{unit.value}+{sorted(extra)}"

    assert _run(flexible, {"unit": "celsius", "zzz": 1}) == "celsius+['zzz']"


def test_the_var_positional_error_names_the_tool_and_the_fix() -> None:
    """An error that says only "unsupported" makes the author guess. This one
    has to name which parameter and what to do instead."""

    with pytest.raises(ToolDefinitionError) as exc:

        @tool(side_effecting=False)
        def variadic(*rest: int) -> str:
            """Total up however many integer values happen to be supplied."""
            return "unreachable"

    message = str(exc.value)
    assert "variadic" in message
    assert "rest" in message
    assert "**kwargs" in message
