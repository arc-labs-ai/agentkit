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
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from agentkit.kernel.protocols import Ctx
from agentkit.runtime import RunContext
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import ToolArgumentError, tool

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


def test_a_list_of_enums_is_not_coerced_because_it_is_not_advertised() -> None:
    """Deliberately out of scope, and it looks like an omission so it is pinned:
    ``_json_type`` advertises ``list[Unit]`` as a bare ``{"type": "array"}`` with
    no ``items``. The element type is never shown to the model, so coercing the
    elements would enforce a contract the model was never given. When the schema
    learns ``items``, this test is the one that should change."""

    @tool(side_effecting=False)
    def many(units: list[Unit]) -> str:
        """Report readings for each of the units of measure supplied here."""
        return f"{type(units).__name__}[{type(units[0]).__name__}]"

    assert many.schema.parameters["properties"]["units"] == {"type": "array"}
    assert _run(many, {"units": ["celsius"]}) == "list[str]"


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
                "opt": {"enum": ["celsius", "fahrenheit"], "type": "string"},
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
