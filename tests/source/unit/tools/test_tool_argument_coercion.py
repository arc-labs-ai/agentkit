"""Tool schema inference and argument coercion.

Everything here goes through the public `@tool` decorator rather than the
private helpers, because the contract being tested is what the MODEL is shown
and what the BODY receives — and those are the two ends of the public path.

The theme, and the reason nine catalogue entries were parked on this file: the
schema and the coercion are two halves of one promise. `items` without element
coercion is a promise never kept; element coercion without `items` enforces a
promise never made. Several of the tests below exist to hold both halves
together.

The other theme is that an inferred schema must never say more than the
annotation does. `tuple[int, str]` has no single element type, `list[Any]`
declines to constrain its elements, and a `total=False` TypedDict key is not
required — advertising otherwise is a lie the transport then guarantees to
break.
"""

from __future__ import annotations

import collections.abc as abc
import enum
from typing import Any, Literal, NotRequired, TypedDict

import pytest

from agentkit.tools import tool


def _params(func: Any) -> dict[str, Any]:
    return dict(func.schema.parameters)


def _prop(func: Any, name: str) -> dict[str, Any]:
    return dict(_params(func)["properties"][name])


_DOC = "A tool with enough description that the inference layer is satisfied."


# ── primitives and bare containers ───────────────────────────────────────────


def test_primitives_map_to_their_json_types() -> None:
    @tool(side_effecting=False)
    def t(a: str, b: int, c: float, d: bool) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"type": "string"}
    assert _prop(t, "b") == {"type": "integer"}
    assert _prop(t, "c") == {"type": "number"}
    assert _prop(t, "d") == {"type": "boolean"}


def test_an_unsubscripted_container_says_nothing_about_its_elements() -> None:
    @tool(side_effecting=False)
    def t(a: list, b: dict) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"type": "array"}
    assert _prop(t, "b") == {"type": "object"}


def test_an_unannotated_parameter_degrades_to_string() -> None:
    @tool(side_effecting=False)
    def t(a) -> str:  # type: ignore[no-untyped-def]
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"type": "string"}


# ── containers: items, and when there is no single element type ──────────────


def test_abstract_collections_describe_items_like_concrete_ones() -> None:
    """`Sequence[int]` is as informative as `list[int]`, and a table listing
    only the concrete types would send the abstract ones to the
    `{"type": "string"}` fallback — telling the model to pass a JSON string
    where the body annotated a collection.

    Written as one tool with five parameters rather than a parametrized
    annotation on purpose: `from __future__ import annotations` makes every
    annotation a STRING resolved from the function's globals, so a
    parametrized `a: annotation` resolves the local name "annotation",
    fails, and falls back to "string" — a green-looking test that proves
    nothing about the table.
    """
    @tool(side_effecting=False)
    def t(
        concrete: list[int],
        sequence: abc.Sequence[int],
        iterable: abc.Iterable[int],
        collection: abc.Collection[int],
        abstract_set: abc.Set[int],
    ) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    expected = {"type": "array", "items": {"type": "integer"}}
    for name in ("concrete", "sequence", "iterable", "collection", "abstract_set"):
        assert _prop(t, name) == expected, name


def test_a_homogeneous_tuple_has_items_and_a_fixed_shape_one_does_not() -> None:
    """`items` means "every element is this". For `tuple[int, str]` that is
    simply untrue, so the schema stays silent rather than advertising a
    constraint the second element violates."""
    @tool(side_effecting=False)
    def t(homogeneous: tuple[int, ...], fixed: tuple[int, str]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "homogeneous") == {"type": "array", "items": {"type": "integer"}}
    assert _prop(t, "fixed") == {"type": "array"}


def test_any_declines_to_constrain_so_the_schema_does_too() -> None:
    @tool(side_effecting=False)
    def t(a: list[Any]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"type": "array"}


def test_nested_containers_describe_their_depth() -> None:
    @tool(side_effecting=False)
    def t(a: list[list[str]]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {
        "type": "array",
        "items": {"type": "array", "items": {"type": "string"}},
    }


# ── mappings ─────────────────────────────────────────────────────────────────


def test_a_mapping_describes_only_its_value_type() -> None:
    """A JSON object's keys are strings by definition, so `dict[int, X]` cannot
    be honoured on the wire and claiming otherwise would be a lie the transport
    guarantees to break."""
    @tool(side_effecting=False)
    def t(a: dict[str, int], b: dict[int, str], c: dict[str, Any]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"type": "object", "additionalProperties": {"type": "integer"}}
    assert _prop(t, "b") == {"type": "object", "additionalProperties": {"type": "string"}}
    assert _prop(t, "c") == {"type": "object"}


# ── TypedDict ────────────────────────────────────────────────────────────────


class _Query(TypedDict):
    field: str
    limit: int


class _Partial(TypedDict, total=False):
    a: str
    b: int


# Functional syntax, deliberately. This module uses
# `from __future__ import annotations`, which makes every class-body annotation
# a STRING — and CPython computes `__required_keys__` at class creation, so it
# cannot see a `NotRequired` it has not evaluated. A class-syntax TypedDict here
# would report `optional_key` as REQUIRED, and a test asserting that would be
# pinning a Python interaction rather than anything agentkit does. The
# functional form passes real objects, so requiredness is computed correctly.
# ruff's UP013 wants class syntax here, and class syntax is precisely what
# breaks under this module's `from __future__ import annotations`: CPython
# computes `__required_keys__` at class creation and cannot see a `NotRequired`
# it has not evaluated, so `optional_key` would be reported REQUIRED. A test
# asserting that would pin a Python interaction rather than anything agentkit
# does — the functional form passes real objects and is correct.
_Mixed = TypedDict(  # noqa: UP013
    "_Mixed", {"required_key": str, "optional_key": NotRequired[int]}
)


def test_a_typeddict_is_an_object_not_a_string() -> None:
    """It used to be advertised as `{"type": "string"}`: the model sent a JSON
    string and the body's `q["field"]` raised "string indices must be
    integers"."""
    @tool(side_effecting=False)
    def t(q: _Query) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "q") == {
        "type": "object",
        "properties": {"field": {"type": "string"}, "limit": {"type": "integer"}},
        "additionalProperties": False,
        "required": ["field", "limit"],
    }


def test_typeddict_requiredness_follows_the_declaration_both_ways() -> None:
    """`__required_keys__` carries `total=` and per-key Required/NotRequired.
    Advertising every key as required would be a worse lie than the string one
    this replaced — the model would be forced to invent values for keys the
    tool treats as optional."""
    @tool(side_effecting=False)
    def t(full: _Query, partial: _Partial, mixed: _Mixed) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "full")["required"] == ["field", "limit"]
    assert _prop(t, "partial")["required"] == []
    assert _prop(t, "mixed")["required"] == ["required_key"]


# ── Literal, enum, unions ────────────────────────────────────────────────────


def test_a_literal_becomes_an_enum_and_pins_the_type_when_homogeneous() -> None:
    @tool(side_effecting=False)
    def t(a: Literal["x", "y"], b: Literal[1, 2], c: Literal["x", 1]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"enum": ["x", "y"], "type": "string"}
    assert _prop(t, "b") == {"enum": [1, 2], "type": "integer"}
    assert _prop(t, "c") == {"enum": ["x", 1]}, "a mixed Literal cannot pin one type"


class _Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


def test_an_enum_class_becomes_its_values() -> None:
    @tool(side_effecting=False)
    def t(a: _Colour) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a")["enum"] == ["red", "blue"]


def test_optional_stays_nullable_so_a_required_parameter_can_be_sent_as_null() -> None:
    """`X | None` collapsing to bare `X` is a dead end on a REQUIRED parameter:
    it is in `required`, so the model must send something, and the only thing
    permitted was an `X`. A tool meaning "a category, or null for all of them"
    could not express the second half."""
    @tool(side_effecting=False)
    def t(a: str | None) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert "a" in _params(t)["required"]


def test_a_real_union_becomes_anyof() -> None:
    @tool(side_effecting=False)
    def t(a: str | int) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"anyOf": [{"type": "string"}, {"type": "integer"}]}


# ── parameter collection ─────────────────────────────────────────────────────


def test_var_positional_is_refused_rather_than_silently_dropped() -> None:
    """A tool is called with a JSON object, so every argument arrives by
    keyword and no positional one can ever be sent. Dropping `*args` silently
    would leave the author believing it was wired; refusing at DECLARATION time
    tells them at the only moment they can still fix it."""
    from agentkit.tools.declarations.errors import ToolDefinitionError

    with pytest.raises(ToolDefinitionError, match="can never be filled"):

        @tool(side_effecting=False)
        def t(a: str, *rest: int) -> str:
            """A tool with enough description that the inference layer is satisfied."""
            return "ok"


def test_a_context_parameter_is_injected_only_when_annotated_as_one() -> None:
    """Detection is by annotation, not by name alone, so a real data parameter
    called `context` is not hijacked. `Ctx` is accepted alongside `RunContext`
    because it is the Protocol the framework itself uses everywhere — omitting
    it used to advertise `{"ctx": {"type": "string"}}` to the model and then
    fail every call with a missing-argument error, with nothing pointing at the
    annotation."""
    from agentkit.core.contracts.protocols import Ctx

    @tool(side_effecting=False)
    def plain(a: str, ctx: Ctx = None) -> str:  # type: ignore[assignment]
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    @tool(side_effecting=False)
    def optional(a: str, ctx: Ctx | None = None) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    @tool(side_effecting=False)
    def bare(a: str, ctx=None) -> str:  # type: ignore[no-untyped-def]
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    @tool(side_effecting=False)
    def data_param(a: str, context: str = "") -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert "ctx" not in _params(plain)["properties"]
    assert "ctx" not in _params(optional)["properties"], (
        "`Ctx | None` is the more natural spelling and is what agentkit's own "
        "tools use in their run signatures; it was being advertised to the model"
    )
    assert "ctx" not in _params(bare)["properties"]
    assert "context" in _params(data_param)["properties"], (
        "a real data parameter called `context` must not be hijacked"
    )


def test_a_defaulted_parameter_is_not_required() -> None:
    @tool(side_effecting=False)
    def t(a: str, b: int = 1) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _params(t)["required"] == ["a"]


# ── coercion: the other half of the promise ──────────────────────────────────


class _Unit(enum.Enum):
    METRIC = "metric"
    IMPERIAL = "imperial"


@pytest.mark.asyncio
async def test_sequence_elements_are_coerced_because_items_promised_them() -> None:
    """`items` showed the model the element contract, so honouring it is owed
    rather than invented. Without this the body annotated `list[_Unit]` and
    received a list of strings."""
    seen: list[Any] = []

    @tool(side_effecting=False)
    def t(units: list[_Unit]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        seen.append(units)
        return "ok"

    await t.run({"units": ["metric", "imperial"]}, None)
    assert seen == [[_Unit.METRIC, _Unit.IMPERIAL]]


@pytest.mark.asyncio
async def test_the_annotated_container_is_rebuilt_not_left_as_a_list() -> None:
    """JSON only has arrays. Without rebuilding, a body annotated `set[str]`
    receives a `list` — and `.add()` raises on it."""
    seen: dict[str, Any] = {}

    @tool(side_effecting=False)
    def t(as_set: set[str], as_frozen: frozenset[str], as_tuple: tuple[str, ...]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        seen.update({"set": as_set, "frozen": as_frozen, "tuple": as_tuple})
        return "ok"

    await t.run({"as_set": ["x", "y"], "as_frozen": ["x"], "as_tuple": ["x", "y"]}, None)

    assert isinstance(seen["set"], set) and seen["set"] == {"x", "y"}
    assert isinstance(seen["frozen"], frozenset)
    assert isinstance(seen["tuple"], tuple) and seen["tuple"] == ("x", "y")


@pytest.mark.asyncio
async def test_a_scalar_is_never_widened_to_match_its_schema() -> None:
    """The mandate is to honour what the schema says, not to widen it. A
    provider that sent "3" for an `{"type": "integer"}` parameter has a bug the
    framework must SURFACE, not launder into a 3."""
    seen: list[Any] = []

    @tool(side_effecting=False)
    def t(n: int) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        seen.append(n)
        return "ok"

    await t.run({"n": "3"}, None)
    assert seen == ["3"], "the string must reach the body unchanged"


@pytest.mark.asyncio
async def test_a_non_sequence_value_is_left_exactly_as_it_arrived() -> None:
    """Same rule as the scalar coercers: a provider that ignored the schema
    produces a visible failure rather than a laundered value."""
    seen: list[Any] = []

    @tool(side_effecting=False)
    def t(a: set[str]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        seen.append(a)
        return "ok"

    await t.run({"a": "not-a-list"}, None)
    assert seen == ["not-a-list"]


@pytest.mark.asyncio
async def test_mapping_values_are_coerced() -> None:
    seen: list[Any] = []

    @tool(side_effecting=False)
    def t(by_name: dict[str, _Unit]) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        seen.append(by_name)
        return "ok"

    await t.run({"by_name": {"a": "metric"}}, None)
    assert seen == [{"a": _Unit.METRIC}]


@pytest.mark.asyncio
async def test_an_enum_value_the_schema_never_offered_is_refused() -> None:
    @tool(side_effecting=False)
    def t(unit: _Unit) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    with pytest.raises(Exception):  # noqa: B017 — the type is the module's own
        await t.run({"unit": "furlongs"}, None)
