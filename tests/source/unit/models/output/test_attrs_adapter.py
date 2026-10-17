"""AttrsAdapter — the whole translation, not just the two shapes that broke.

``test_output_schema_attrs.py`` next door is a regression suite: private
attributes, ``init=False``, validators raising out of ``__init__``. It says
nothing about the part of the adapter a caller actually leans on — the type
tree. This file is that part, and the decisions it pins are the ones the module
docstring claims to mirror from ``DataclassAdapter``:

* the supported leaf set is deliberately CONSERVATIVE, and an annotation
  outside it fails when the adapter is BUILT. A schema built lazily at parse
  time would turn a wiring mistake into a mid-run coercion failure, which the
  repair loop would then try to repair by asking the model to fix something the
  model did not do;
* JSON has one number type and ``True`` is an ``int`` in Python, so the coercer
  refuses the crossings the language would wave through. ``flag=1`` arriving as
  ``True`` is a model's guess silently promoted to a value;
* errors are BATCHED. One ``OutputCoercionError`` naming every bad field gives
  the repair turn one round trip; the first error alone gives it N;
* a streaming partial is best-effort and the strict terminal parse is the source
  of truth, so every failure in ``partial_parse`` withholds the partial rather
  than propagating — a broken partial must not break the stream.

Every type-driven test uses a real attrs class with its annotation written out
at module level. Parametrising over an annotation would not work here at all:
this file carries ``from __future__ import annotations``, so a parametrised
``ann`` reaches ``typing.get_type_hints`` as the unresolvable string ``"ann"``,
schema inference falls back to ``{"type": "string"}``, and the test passes
while asserting nothing about the type it named.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Literal

import pytest

attrs = pytest.importorskip("attrs")

from agentkit.models.output import AttrsAdapter, OutputCoercionError
from agentkit.models.output.attrs_adapter import _coerce_value, _is_attrs_class


@attrs.define
class Leaf:
    """A nested attrs class — the one non-scalar leaf the translator walks into."""

    label: str


@attrs.define
class Wide:
    """One field per supported shape, so the schema assertion is exhaustive
    rather than a sample."""

    name: str
    count: int
    ratio: float
    flag: bool
    tags: list[str]
    meta: dict[str, int]
    note: str | None = None
    child: Leaf | None = None
    mode: Literal["fast", "slow"] = "fast"
    extras: list[str] = attrs.Factory(list)


@attrs.define
class MixedLiteral:
    """A ``Literal`` whose members are not all one type — JSON Schema can express
    the enum but not a single ``type``."""

    choice: Literal[1, "a"] | None = None


@attrs.define
class AllOptional:
    """Every field defaulted, including a factory one."""

    count: int = 0
    tags: list[str] = attrs.Factory(list)


@attrs.define
class BadUnion:
    """A union that is not ``Optional[X]``."""

    either: int | str


@attrs.define
class BadDictKey:
    """A mapping JSON cannot express — its keys are not strings."""

    payload: dict[int, str]


@attrs.define
class BadList:
    """A list annotation with more than one element type."""

    pair: list[int, str]


@attrs.define
class BadLeaf:
    """A leaf JSON has no name for."""

    when: complex


@attrs.define
class BadTuple:
    """A container the translator does not describe at all."""

    pair: tuple[int, ...]


def _wide(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": "x", "count": 1, "ratio": 1.5, "flag": True, "tags": [], "meta": {}}
    payload.update(overrides)
    return payload


# ── the schema is built once, eagerly, and is exhaustive ───────────────────


def test_every_supported_shape_becomes_the_json_schema_the_model_is_shown() -> None:
    """Asserted as one whole document rather than key by key, because the thing
    that matters is what the PROVIDER receives: a field quietly translated to
    the wrong fragment is a field the model is told to fill differently from
    how the class will accept it, and the rejection that follows names the
    model rather than the mistranslation.
    """
    assert AttrsAdapter(Wide).json_schema() == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {"type": "object", "additionalProperties": {"type": "integer"}},
            "note": {"type": ["string", "null"]},
            "child": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
            "mode": {"enum": ["fast", "slow"], "type": "string"},
            "extras": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "count", "ratio", "flag", "tags", "meta"],
    }


def test_an_optional_field_widens_its_own_type_instead_of_becoming_an_anyof() -> None:
    """``{"type": ["string", "null"]}`` rather than ``anyOf`` because several
    providers' structured-output modes accept the former and reject or ignore
    the latter, and the nested case is the one that shows the difference: the
    object's ``properties`` and ``required`` have to stay ALONGSIDE the widened
    type, not get buried in a branch the provider drops.
    """
    properties = AttrsAdapter(Wide).json_schema()["properties"]

    assert properties["note"]["type"] == ["string", "null"]
    assert properties["child"]["type"] == ["object", "null"]
    assert properties["child"]["properties"] == {"label": {"type": "string"}}
    assert "anyOf" not in properties["child"]


def test_an_optional_with_no_single_type_to_widen_falls_back_to_anyof() -> None:
    """A mixed ``Literal`` has an ``enum`` and no ``type``, so there is nothing
    to append ``"null"`` to. Inventing one would describe a value the class
    will not accept; ``anyOf`` is the honest spelling, and the fallback has to
    exist or this annotation would produce a schema missing its nullability
    entirely."""
    assert AttrsAdapter(MixedLiteral).json_schema()["properties"]["choice"] == {
        "anyOf": [{"enum": [1, "a"]}, {"type": "null"}]
    }


def test_a_literal_carries_a_type_only_when_its_members_agree_on_one() -> None:
    """The ``type`` beside the ``enum`` is redundant for a validator and load
    bearing for a model: it is the hint that says "a string from this list",
    not "one of these tokens". It can only be stated when every member really
    is that type, which is why the mixed case drops it rather than guessing
    from the first member."""
    assert AttrsAdapter(Wide).json_schema()["properties"]["mode"] == {"enum": ["fast", "slow"], "type": "string"}
    assert "type" not in AttrsAdapter(MixedLiteral).json_schema()["properties"]["choice"]["anyOf"][0]


def test_a_class_with_a_default_everywhere_declares_nothing_required() -> None:
    """``required: []`` is not the same document as no ``required`` key, and
    some validators treat an empty list as "nothing may be supplied". A class
    where the model may legitimately send ``{}`` has to produce a schema that
    says so, and a ``Factory`` default counts — it is still a value the caller
    does not have to provide."""
    schema = AttrsAdapter(AllOptional).json_schema()

    assert "required" not in schema
    assert set(schema["properties"]) == {"count", "tags"}
    assert AttrsAdapter(AllOptional).parse("{}") == AllOptional()


def test_the_schema_handed_out_is_a_copy_so_a_provider_massage_cannot_poison_it() -> None:
    """One adapter serves every request for the life of the process, and
    provider adapters really do rewrite fragments in place to suit their own
    structured-output dialect. Handing out the cached object would let one
    provider's fix-up silently change what every other provider is shown.
    """
    adapter = AttrsAdapter(Wide)

    first = adapter.json_schema()
    first["properties"]["name"]["type"] = "number"
    first["properties"].pop("count")

    assert adapter.json_schema()["properties"]["name"] == {"type": "string"}
    assert "count" in adapter.json_schema()["properties"]


def test_a_union_that_is_not_optional_is_refused_when_the_adapter_is_built() -> None:
    """``int | str`` has no honest JSON Schema in this translator's vocabulary,
    and the refusal has to land at BUILD time: a schema assembled lazily at
    parse time turns a wiring mistake into a mid-run coercion failure, which
    the repair loop then tries to fix by asking the model to correct something
    the model did not do.
    """
    with pytest.raises(TypeError, match="only Optional") as exc:
        AttrsAdapter(BadUnion)

    assert "int" in str(exc.value) and "str" in str(exc.value), "the message must name the offending annotation"


def test_a_container_the_translator_cannot_describe_is_refused_with_the_rule() -> None:
    """``dict[int, str]`` cannot be a JSON object — JSON keys are strings — and
    ``list[int, str]`` has no single ``items``. Each message states the shape
    that IS accepted, because the fix is a one-line annotation change and the
    author should not have to find this file to learn it.
    """
    with pytest.raises(TypeError, match=r"must be dict\[str, X\]"):
        AttrsAdapter(BadDictKey)

    with pytest.raises(TypeError, match=r"must be list\[X\]"):
        AttrsAdapter(BadList)


def test_a_leaf_outside_the_supported_set_is_refused_and_the_set_is_listed() -> None:
    """The leaf set is deliberately narrow — the adapter would rather refuse a
    type than invent a lossy encoding for it — so the error carries the whole
    supported list. ``tuple`` also proves the probe for a nested attrs class
    tolerates a non-class annotation instead of raising on it.
    """
    for broken in (BadLeaf, BadTuple):
        with pytest.raises(TypeError, match="unsupported field type") as exc:
            AttrsAdapter(broken)
        assert "nested attrs class" in str(exc.value)
        assert "Literal" in str(exc.value)


# ── the coercer refuses what Python would wave through ─────────────────────


def test_a_json_true_is_never_accepted_as_a_number() -> None:
    """``isinstance(True, int)`` is ``True`` in Python, so the obvious check
    accepts ``{"count": true}`` and stores ``1``. The model guessed and the
    guess became a value nothing downstream can question — which is the whole
    reason both numeric branches test for ``bool`` first.
    """
    adapter = AttrsAdapter(Wide)

    with pytest.raises(OutputCoercionError) as integral:
        adapter.parse(json.dumps(_wide(count=True)))
    assert integral.value.errors == ["count: expected int, got bool"]

    with pytest.raises(OutputCoercionError) as fractional:
        adapter.parse(json.dumps(_wide(ratio=True)))
    assert fractional.value.errors == ["ratio: expected number, got bool"]


def test_a_json_number_is_never_accepted_as_a_boolean() -> None:
    """The other direction, and the one a model reaches for constantly: ``1``
    and ``0`` for a flag. Accepting them would make ``flag=0`` indistinguishable
    from a model that meant to send ``false`` and from one that sent a count by
    mistake."""
    with pytest.raises(OutputCoercionError) as exc:
        AttrsAdapter(Wide).parse(json.dumps(_wide(flag=1)))

    assert exc.value.errors == ["flag: expected bool, got int"]


def test_an_integer_widens_into_a_float_field_and_arrives_as_a_float() -> None:
    """JSON has one number type, so a model that means ``2.0`` writes ``2``.
    Refusing that would fail honest responses. The value is CONVERTED rather
    than passed through, so the field holds the type the class annotated and
    arithmetic downstream does not change behaviour with the response's
    spelling."""
    parsed = AttrsAdapter(Wide).parse(json.dumps(_wide(ratio=2)))

    assert parsed.ratio == 2.0
    assert isinstance(parsed.ratio, float), "an int stored in a float field changes // and repr downstream"


def test_a_string_field_refuses_every_other_json_scalar_including_null() -> None:
    """``str(value)`` would accept all of them and produce ``"7"`` or
    ``"None"`` — a value that reads as deliberate forever after. A required
    field arriving as ``null`` is the most common of these and still an error,
    because the annotation did not admit it."""
    adapter = AttrsAdapter(Wide)

    with pytest.raises(OutputCoercionError) as numeric:
        adapter.parse(json.dumps(_wide(name=7)))
    assert numeric.value.errors == ["name: expected string, got int"]

    with pytest.raises(OutputCoercionError) as nulled:
        adapter.validate(_wide(name=None))
    assert nulled.value.errors == ["name: expected string, got NoneType"]


def test_an_optional_field_does_accept_null_because_its_annotation_says_so() -> None:
    """POSITIVE CONTROL for the test above — the refusal has to be about the
    ANNOTATION, not about nulls being inconvenient."""
    parsed = AttrsAdapter(Wide).validate(_wide(note=None, child=None))

    assert parsed.note is None and parsed.child is None


def test_a_literal_field_refuses_a_value_outside_its_enum_and_lists_them() -> None:
    """The enum is advertised, so a value outside it is a model ignoring the
    schema — and the only useful next turn is one that shows the legal values
    again, in the error, rather than making the model re-derive them."""
    adapter = AttrsAdapter(Wide)

    with pytest.raises(OutputCoercionError) as exc:
        adapter.parse(json.dumps(_wide(mode="turbo")))
    assert exc.value.errors == ["mode: 'turbo' not in allowed values ['fast', 'slow']"]

    # POSITIVE CONTROL: a member of the enum passes through as the value the
    # model sent, not as an index into it or a coerced copy.
    assert adapter.parse(json.dumps(_wide(mode="slow"))).mode == "slow"


def test_a_container_field_refuses_a_scalar_rather_than_wrapping_it() -> None:
    """A string is iterable and a dict is not a list, so a tolerant
    implementation would turn ``"a"`` into ``["a"]`` — or, worse, into
    ``["a", "b"]`` for ``"ab"``. The field says array; anything else is the
    model's mistake and has to be named as one."""
    adapter = AttrsAdapter(Wide)

    with pytest.raises(OutputCoercionError) as listed:
        adapter.parse(json.dumps(_wide(tags="ab")))
    assert listed.value.errors == ["tags: expected array, got str"]

    with pytest.raises(OutputCoercionError) as mapped:
        adapter.parse(json.dumps(_wide(meta=[1, 2])))
    assert mapped.value.errors == ["meta: expected object, got list"]


def test_a_nested_attrs_field_names_the_class_it_wanted_an_object_for() -> None:
    """The nested class name is the only thing that tells the model WHICH
    object it failed to produce — "expected object" alone, in a response with
    several nested fields, does not."""
    with pytest.raises(OutputCoercionError) as exc:
        AttrsAdapter(Wide).parse(json.dumps(_wide(child="Leaf")))

    assert exc.value.errors == ["child: expected object for Leaf, got str"]


def test_an_error_inside_a_container_carries_the_path_to_the_bad_element() -> None:
    """``tags: expected string`` for a list of forty entries is not actionable.
    The index and the key are built on the way down for exactly this, and they
    are the difference between a repair turn and a re-roll."""
    adapter = AttrsAdapter(Wide)

    with pytest.raises(OutputCoercionError) as indexed:
        adapter.parse(json.dumps(_wide(tags=["a", 2])))
    assert indexed.value.errors == ["tags[1]: expected string, got int"]

    with pytest.raises(OutputCoercionError) as keyed:
        adapter.parse(json.dumps(_wide(meta={"ok": 1, "bad": "two"})))
    assert keyed.value.errors == ["meta.bad: expected int, got str"]


# ── errors are batched, and the boundary holds ─────────────────────────────


def test_every_bad_field_is_reported_at_once_rather_than_just_the_first() -> None:
    """One round trip instead of N. A model that got four fields wrong and is
    told about one will fix that one and resubmit three errors, and the repair
    loop's attempt budget is spent walking a list it could have been handed in
    full. Missing and unexpected fields are part of the same list.
    """
    with pytest.raises(OutputCoercionError) as exc:
        AttrsAdapter(Wide).parse('{"name": 1, "count": "two", "bogus": true}')

    assert exc.value.errors == [
        "name: expected string, got int",
        "count: expected int, got str",
        "ratio: missing required field",
        "flag: missing required field",
        "tags: missing required field",
        "meta: missing required field",
        "bogus: unexpected field",
    ]
    assert "7 error(s)" in str(exc.value), "the summary has to say how many, or the list reads as the whole story"


def test_an_undeclared_field_is_an_error_rather_than_being_dropped() -> None:
    """``additionalProperties: false`` is advertised, so a key outside the class
    is the model inventing one — and silently dropping it is how a response
    that looks accepted loses the half the model considered important.
    """
    assert AttrsAdapter(Wide).json_schema()["additionalProperties"] is False

    with pytest.raises(OutputCoercionError) as exc:
        AttrsAdapter(Wide).parse(json.dumps(_wide(surprise="!")))
    assert exc.value.errors == ["surprise: unexpected field"]


def test_a_response_that_is_not_json_names_where_the_decoder_stopped() -> None:
    """A truncated or prose-wrapped response is the single most common failure
    on this path, and the line and column are what let a human tell "the model
    wrote an apology" from "the stream was cut". The raw text is kept on the
    error for the same reason."""
    with pytest.raises(OutputCoercionError) as exc:
        AttrsAdapter(Wide).parse('{"name": "x"')

    assert "not valid JSON" in str(exc.value)
    assert "line 1" in str(exc.value) and "col" in str(exc.value)
    assert exc.value.errors and exc.value.errors[0].startswith("json: ")
    assert exc.value.raw == '{"name": "x"'


def test_valid_json_of_the_wrong_shape_is_refused_before_any_field_is_walked() -> None:
    """A top-level array decodes perfectly and has no fields, so walking it
    would produce a pile of "missing required field" errors that describe the
    symptom and not the cause. One error naming the type is the diagnosis."""
    with pytest.raises(OutputCoercionError) as exc:
        AttrsAdapter(Wide).parse("[1, 2]")

    assert exc.value.errors == ["top-level: expected object, got list"]
    assert "expected a JSON object for Wide" in str(exc.value)


def test_the_coercer_is_fail_closed_in_its_own_right_not_only_by_schema_filtering() -> None:
    """The schema builder currently refuses non-``Optional`` unions and unknown
    leaves, so these branches are unreachable through ``parse`` today. They are
    still the right behaviour to pin: the coercer is a separate function with
    its own callers, and "the other half happens to filter this" is how a
    fail-open default survives a refactor of the half that was filtering.
    """
    with pytest.raises(ValueError, match="not Optional"):
        _coerce_value(int | str, None, path="either")

    with pytest.raises(ValueError, match="cannot coerce into"):
        _coerce_value(object, 1, path="when")


# ── the streaming partial is best-effort and never breaks the stream ───────


def test_a_partial_seeds_declared_defaults_so_attribute_access_does_not_raise() -> None:
    """``partial_parse`` bypasses ``__init__`` — attrs would reject the missing
    required args — which means nothing runs the defaults either. A streaming
    consumer rendering the partial would then hit ``AttributeError`` on every
    field the model has not reached yet, so the defaults are applied by hand.
    """
    partial = AttrsAdapter(Wide).partial_parse('{"name": "Ali')

    assert partial is not None
    assert partial.name == "Ali"
    assert partial.mode == "fast"
    assert partial.note is None
    assert not hasattr(partial, "count"), "a field with no default must stay absent, not be invented"


def test_a_factory_default_is_invoked_per_partial_rather_than_shared() -> None:
    """``attrs.Factory`` exists precisely because a mutable default must not be
    shared. Storing the ``Factory`` object itself would hand the consumer
    something that is not a list at all; storing one call's result for every
    partial would alias two streams' fields together.
    """
    adapter = AttrsAdapter(AllOptional)

    first = adapter.partial_parse('{"count": 1')
    second = adapter.partial_parse('{"count": 2')

    assert first is not None and second is not None
    assert first.tags == [] and second.tags == []
    assert first.tags is not second.tags, "two partials share one mutable default"


def test_a_key_the_class_does_not_declare_is_skipped_rather_than_set() -> None:
    """The tolerant parser occasionally produces a key the class never declared
    — a half-written field name, a hallucinated one. Setting it would put an
    attribute on the instance that no schema describes; dropping the whole
    partial over it would withhold the fields that ARE good."""
    partial = AttrsAdapter(AllOptional).partial_parse('{"count": 1, "bogus": 2')

    assert partial is not None
    assert partial.count == 1
    assert not hasattr(partial, "bogus")


def test_a_partial_with_no_object_structure_is_withheld() -> None:
    """The streaming middleware emits a partial event only when there is one to
    emit. An array prefix, or a stream that has produced nothing yet, has no
    fields to show and ``None`` is how the adapter says so."""
    adapter = AttrsAdapter(AllOptional)

    assert adapter.partial_parse("[1, 2") is None
    assert adapter.partial_parse("") is None


def test_a_parser_failure_withholds_the_partial_instead_of_breaking_the_stream() -> None:
    """The tolerant parser is documented never to raise on malformed TEXT, which
    is not the same as never raising: it is handed whatever the streaming path
    accumulated, and ``bytes`` out of an SSE frame reaches it as happily as a
    ``str``. The terminal parse remains the source of truth, so a partial that
    cannot be built is one event not emitted — never an exception out of a
    stream the caller is iterating.
    """
    assert AttrsAdapter(AllOptional).partial_parse(b'{"count": 1') is None  # type: ignore[arg-type]


def test_a_class_the_init_bypass_cannot_build_withholds_the_partial() -> None:
    """``object.__new__`` is refused outright for a class whose layout needs its
    own — an attrs class over a ``str`` is the ordinary example. Streaming such
    an output type then yields no partials, which is a degradation; an exception
    out of the stream would be an outage. The strict path refuses it too, and
    the shape of THAT refusal is what matters: an ``OutputCoercionError``
    carrying the construction failure, not a raw ``TypeError`` escaping past
    the contract the repair loop keys off.
    """

    @attrs.define
    class Stringy(str):
        n: int = 0

    adapter = AttrsAdapter(Stringy)

    assert adapter.partial_parse('{"n": 1') is None

    with pytest.raises(OutputCoercionError) as exc:
        adapter.parse('{"n": 1}')
    assert exc.value.errors[0].startswith("could not construct Stringy: TypeError")


# ── validate(), and the attrs probe ────────────────────────────────────────


def test_validate_hands_back_the_very_instance_it_was_given() -> None:
    """The fast path is identity, not a copy. A tool returning an instance it
    also holds a reference to must get that object back — re-constructing it
    would re-run converters and validators on already-validated data and break
    identity comparisons at the boundary.
    """
    inst = Wide(name="x", count=1, ratio=1.0, flag=True, tags=[], meta={})

    assert AttrsAdapter(Wide).validate(inst) is inst


def test_validate_reports_its_failures_as_a_coercion_error_carrying_the_value() -> None:
    """``validate`` shares the walker with ``parse`` and must share the error
    contract too: ``FunctionTool``'s ``ToolShapeError`` seam and the repair loop
    both key off ``OutputCoercionError``, and a raw ``ValueError`` escaping here
    surfaces as a crash rather than a retry. The offending value is attached,
    because there is no raw response text on this path to fall back on.
    """
    with pytest.raises(OutputCoercionError) as exc:
        AttrsAdapter(Wide).validate({"count": "two"})

    assert "count: expected int, got str" in exc.value.errors
    assert exc.value.raw == {"count": "two"}
    assert "failed to validate value into Wide" in str(exc.value)


def test_a_value_that_is_not_a_mapping_at_all_names_both_types() -> None:
    """Nothing can be done with an ``int`` where an object was expected, so the
    error exists to tell a human which boundary produced it — the name the
    adapter was given and the type that arrived."""
    with pytest.raises(OutputCoercionError) as exc:
        AttrsAdapter(Wide).validate(42)

    assert exc.value.errors == ["top-level: expected Wide, got int"]
    assert "not coercible into Wide" in str(exc.value)


def test_the_adapter_name_defaults_to_the_class_and_can_be_overridden() -> None:
    """The name appears in every error the model is shown, so it has to be the
    class's own by default — and overridable, because the class name is an
    implementation detail a caller may not want to expose to a model."""
    assert AttrsAdapter(Wide).name == "Wide"
    assert AttrsAdapter(Wide, name="Report").name == "Report"
    assert AttrsAdapter(Wide).python_type is Wide, "the boundary isinstance check needs the real class"


def test_a_class_that_is_not_an_attrs_class_is_refused_at_construction() -> None:
    """The dispatcher picks an adapter by probing the class, so a direct
    construction is the one place a plain class can reach this adapter. It
    refuses rather than producing an empty schema, which would advertise an
    output type with no fields and coerce every response into nothing."""
    with pytest.raises(TypeError, match="expects an attrs class"):
        AttrsAdapter(dict)


def test_the_attrs_probe_answers_false_on_a_box_without_attrs() -> None:
    """attrs is OPTIONAL and the import lives inside the helper so this module
    loads without it — the ``adapt()`` dispatcher then simply never picks this
    adapter. A probe that raised instead would make an unrelated output type
    fail to wire at all on a bare install.
    """
    assert _is_attrs_class(Wide) is True
    assert _is_attrs_class("not even a class") is False

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "attrs", None)
        assert _is_attrs_class(Wide) is False, "the probe raised where it has to answer"
