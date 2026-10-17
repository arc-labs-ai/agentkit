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
from typing import Any, Literal, NotRequired, Required, TypedDict

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


# Class syntax, deliberately, and this is the whole point of the block below.
# This module uses `from __future__ import annotations`, so every class-body
# annotation is a STRING, and CPython computes `__required_keys__` at class
# creation — before anything has evaluated those strings. It therefore cannot
# see this `NotRequired` and reports `optional_key` as REQUIRED.
#
# That is the shape nearly all modern code has (PEP 563 is file-wide and common),
# so it is squarely agentkit's job to get right rather than a Python quirk to
# write around. These tests used to use the functional `TypedDict(...)` form
# specifically to dodge it; the dodge is gone.
class _Mixed(TypedDict):
    required_key: str
    optional_key: NotRequired[int]


class _Strict(TypedDict, total=False):
    loose: str
    pinned: Required[int]


class _OptionalBase(TypedDict, total=False):
    inherited: str


class _Child(_OptionalBase):
    own: str


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
    """Requiredness carries `total=` and per-key Required/NotRequired. Advertising
    every key as required would be a worse lie than the string one this replaced —
    the model would be forced to invent values for keys the tool treats as
    optional, and there is no value it can invent that means "I am not sending
    this"."""
    @tool(side_effecting=False)
    def t(full: _Query, partial: _Partial, mixed: _Mixed) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "full")["required"] == ["field", "limit"]
    assert _prop(t, "partial")["required"] == []
    assert _prop(t, "mixed")["required"] == ["required_key"]


def test_a_notrequired_key_is_optional_even_under_postponed_annotations() -> None:
    """`__required_keys__` is computed at class creation, so under PEP 563 — which
    this module and most modern code enable file-wide — it is reading the string
    "NotRequired[int]" and cannot tell it names a marker. It reports BOTH keys as
    required. Trusting it advertised a mandatory `optional_key` that the body
    treats as optional, and the model had to invent a number for it on every call.

    The marker has to be read from resolved hints instead, which is the one place
    it is visible for what it is."""
    assert "optional_key" in _Mixed.__required_keys__, (
        "if CPython ever resolves these at class creation this test is moot — but "
        "until then the wrong answer is what the schema layer has to correct"
    )

    @tool(side_effecting=False)
    def t(q: _Mixed) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "q")["required"] == ["required_key"]
    assert _prop(t, "q")["properties"]["optional_key"] == {"type": "integer"}, (
        "requiredness is read from the extras-bearing hints, but the property "
        "schema must still come from the plain ones — otherwise the `NotRequired` "
        "wrapper survives into `_json_type` and the key is advertised as a string"
    )


def test_an_explicit_required_key_in_a_total_false_typeddict_stays_required() -> None:
    """The mirror of the case above, and the reason the marker is an OVERRIDE rather
    than a recomputation: under PEP 563 `__required_keys__` cannot see this
    `Required` either, so it reports the key as optional. A tool that needs the key
    would then be called without it."""
    @tool(side_effecting=False)
    def t(q: _Strict) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "q")["required"] == ["pinned"]


def test_a_key_inherited_from_a_total_false_base_stays_optional() -> None:
    """Why `__required_keys__` is still the baseline rather than being replaced by
    `__total__` plus markers: `total=` is per-class, so a key inherited from a
    `total=False` base is optional even though the subclass is `total=True`. Only
    `__required_keys__` sees that, and it gets this case right — so markers override
    it key by key and leave everything they do not name alone."""
    @tool(side_effecting=False)
    def t(q: _Child) -> str:
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "q")["required"] == ["own"]


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


# ── annotations that will not resolve ────────────────────────────────────────
#
# `get_type_hints` is all-or-nothing, and that is the trap these three hold shut.
# One annotation it cannot evaluate raises for the WHOLE function, and the old
# code caught that and fell back to `str` for every parameter — the tool kept
# working, the model was told each parameter was a string, and the body got a
# string where it expected its annotated type.


def test_a_parameter_whose_annotation_cannot_be_resolved_is_refused() -> None:
    """Advertising it as a string is the worst of the three available outcomes: the
    tool ships, the model is told something false, and the failure surfaces later as
    a `TypeError` inside the body with nothing pointing back at the annotation.
    Refusing at decoration time costs the author one import and is the same call the
    module already makes for `*args`."""
    from agentkit.tools.declarations.errors import ToolDefinitionError

    with pytest.raises(ToolDefinitionError) as excinfo:

        @tool(side_effecting=False)
        def t(a: NoSuchTypeAnywhere) -> str:  # noqa: F821
            """A tool with enough description that the inference layer is satisfied."""
            return "ok"

    message = str(excinfo.value)
    assert "'a'" in message, "the message has to name the parameter to be actionable"
    assert "NoSuchTypeAnywhere" in message
    assert "NameError" in message, "and the underlying reason, or it is a guessing game"


def test_an_unannotated_parameter_is_still_a_string_and_not_a_refusal() -> None:
    """The refusal above is narrow on purpose. An unannotated parameter means
    "free-form string" and always has — it is a choice the author can read straight
    off the signature, unlike an annotation that was written and then silently
    discarded."""
    @tool(side_effecting=False)
    def t(a) -> str:  # noqa: ANN001
        """A tool with enough description that the inference layer is satisfied."""
        return "ok"

    assert _prop(t, "a") == {"type": "string"}


@pytest.mark.asyncio
async def test_one_unresolvable_annotation_does_not_disable_its_neighbours() -> None:
    """The silent half of the same defect, and the one that shipped. A ctx parameter
    annotated with a name that is only reachable inside an enclosing function cannot
    be resolved from module globals under PEP 563 — and it used to take every
    parameter beside it down, because the single `get_type_hints` call raised.

    So `unit: _Unit` was advertised as a bare string AND got no coercer: the enum
    values were never offered to the model, and the body received whatever string
    arrived. The ctx parameter is skipped either way, so nothing about it was ever
    visible — only the neighbours broke."""
    seen: list[Any] = []

    def declare() -> Any:
        from agentkit.core.contracts.protocols import Ctx  # deliberately function-local

        @tool(side_effecting=False)
        def t(unit: _Unit, ctx: Ctx = None) -> str:
            """A tool with enough description that the inference layer is satisfied."""
            seen.append(unit)
            return "ok"

        return t

    t = declare()
    assert "ctx" not in _params(t)["properties"]
    assert _prop(t, "unit") == {"enum": ["metric", "imperial"], "type": "string"}

    await t.run({"unit": "metric"}, None)
    assert seen == [_Unit.METRIC], "the body must receive the enum, not the raw string"
