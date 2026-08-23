"""AttrsAdapter — parse/serialize contract for the attrs flavour.

REGRESSION SUITE. ``_coerce_attrs`` built its kwargs from
``attrs.fields(cls)`` names and called ``cls(**kwargs)``, which breaks on
two ordinary attrs shapes:

1. A private attribute is NAMED ``_secret`` but its ``__init__``
   parameter is ``secret`` (attrs exposes the mapping as
   ``Attribute.alias``). Measured: ``@attrs.define class A: _secret: str``
   → ``serialize`` gives ``{'_secret': 's'}`` → ``parse``/``validate`` →
   ``TypeError: A.__init__() got an unexpected keyword argument
   '_secret'``.
2. ``attrs.field(init=False)`` is not accepted by ``__init__`` at all.

A raw ``TypeError`` escapes the ``OutputCoercionError`` contract that the
repair loop and ``FunctionTool``'s ``ToolShapeError`` seam both key off,
so the failure surfaced as an unhandled crash rather than a retry.
"""

from __future__ import annotations

import json

import pytest

from agentkit.capabilities.output_schema import AttrsAdapter, OutputCoercionError

attrs = pytest.importorskip("attrs")


@attrs.define
class Secretive:
    """Private attributes — ``_secret`` on the class, ``secret`` in ``__init__``."""

    _secret: str
    _count: int = 0


@attrs.define
class Computed:
    """``derived`` is the class's business, never the model's."""

    text: str
    derived: int = attrs.field(init=False, default=0)


def test_private_attribute_parses_instead_of_raising_typeerror():
    adapter = AttrsAdapter(Secretive)
    out = adapter.parse('{"_secret":"s","_count":3}')
    assert out == Secretive("s", 3)


def test_private_attribute_round_trips_both_ways():
    """``attrs.asdict`` emits ``_secret``, and the schema is keyed the same
    way — serialize output has to survive parse unchanged."""
    adapter = AttrsAdapter(Secretive)
    inst = Secretive("s", 3)
    assert adapter.serialize(inst) == {"_secret": "s", "_count": 3}
    assert adapter.parse(json.dumps(adapter.serialize(inst))) == inst
    assert set(adapter.json_schema()["properties"]) == {"_secret", "_count"}


def test_private_attribute_validates_from_a_plain_dict():
    """``validate`` shares the walker, so it carried the same TypeError."""
    adapter = AttrsAdapter(Secretive)
    assert adapter.validate({"_secret": "s", "_count": 3}) == Secretive("s", 3)


def test_private_attribute_on_a_frozen_class_round_trips():
    """EDGE: ``@attrs.frozen`` is slotted AND rejects ``setattr``."""

    @attrs.frozen
    class FrozenSecret:
        _token: str

    adapter = AttrsAdapter(FrozenSecret)
    inst = FrozenSecret("v")
    assert adapter.parse(json.dumps(adapter.serialize(inst))) == inst


def test_init_false_field_is_optional_and_round_trips():
    """EDGE: init=False AND a default — never ``required`` (the model can't
    know it), still restored from the payload so rehydrate is loss-less."""
    adapter = AttrsAdapter(Computed)
    assert adapter.json_schema().get("required") == ["text"]

    inst = Computed("hey")
    object.__setattr__(inst, "derived", 7)
    assert adapter.parse(json.dumps(adapter.serialize(inst))) == inst
    # Omitted entirely → the class's own default applies.
    assert adapter.parse('{"text":"hey"}').derived == 0


def test_validator_failure_surfaces_as_output_coercion_error():
    """EDGE: attrs validators raise from ``__init__``; that must arrive as
    an OutputCoercionError, not a raw exception, or the repair loop never
    sees it."""

    @attrs.define
    class Positive:
        n: int = attrs.field()

        @n.validator
        def _check(self, _attribute, value):
            if value < 0:
                raise ValueError("n must be >= 0")

    adapter = AttrsAdapter(Positive)
    with pytest.raises(OutputCoercionError) as exc:
        adapter.parse('{"n":-1}')
    assert "n must be >= 0" in str(exc.value.errors)


def test_ordinary_attrs_class_still_parses_unchanged():
    """POSITIVE CONTROL — plain public fields keep working, including the
    missing-required-field and unexpected-field errors."""

    @attrs.define
    class Plain:
        name: str
        age: int = 0

    adapter = AttrsAdapter(Plain)
    assert adapter.json_schema()["required"] == ["name"]
    assert adapter.parse('{"name":"bo","age":4}') == Plain("bo", 4)
    assert adapter.parse(json.dumps(adapter.serialize(Plain("bo")))) == Plain("bo")
    with pytest.raises(OutputCoercionError):
        adapter.parse('{"age":4}')
    with pytest.raises(OutputCoercionError):
        adapter.parse('{"name":"bo","nope":1}')
