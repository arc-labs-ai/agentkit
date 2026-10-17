"""Package-wide sweep over agentkit's public value types.

The point of a sweep rather than per-type tests: these invariants are properties
of the whole public surface, and the failure mode is a NEW type shipping without
them. A per-type test can only protect the types someone remembered to write one
for, and the fixtures in `value_factories` are deliberately built to be hostile —
nested mutable containers inside otherwise-frozen values, and a credential that
must never reach a log.

`value_factories.FACTORIES` was stranded when its consumer was deleted in the
test reorganisation: the fixtures survived, the sweep did not. This is the sweep.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from tests.source.conformance.value_factories import FACTORIES

_SECRET = "fake-test-credential"


def _make(name: str) -> Any:
    return FACTORIES[name]()


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_every_public_value_is_a_frozen_dataclass(name: str) -> None:
    """These travel across every boundary the framework has — into results, onto
    sessions, through journals — and several are cached or hashed. A mutable one is
    a value that can be edited after the decision it informed was already made, and
    `frozen=True` is the only thing that makes "value" mean anything here."""
    value = _make(name)
    assert dataclasses.is_dataclass(value), f"{name} is not a dataclass"
    fields = dataclasses.fields(value)
    assert fields, f"{name} has no fields to protect"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, fields[0].name, getattr(value, fields[0].name))


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_no_public_value_leaks_a_credential_through_its_repr(name: str) -> None:
    """`repr` is where secrets escape: it is what a traceback prints, what a logger
    renders for an un-formatted argument, and what a debugger shows. `CredentialRef`
    holds a CALLABLE precisely so the secret is fetched on demand and never sits in
    a field — which only pays off if nothing in the tree renders it eagerly."""
    value = _make(name)
    assert _SECRET not in repr(value), f"{name} renders its credential in repr()"
    assert _SECRET not in str(value), f"{name} renders its credential in str()"


def test_a_credential_ref_still_resolves_when_it_is_actually_asked() -> None:
    """The counterpart of the test above, and the reason it is not satisfied by a
    `CredentialRef` that simply lost the secret: the indirection has to still work."""
    assert FACTORIES["CredentialRef"]().resolve() == {"ANTHROPIC_API_KEY": _SECRET}


def test_reported_usage_is_deep_frozen_not_merely_wrapped() -> None:
    """`raw_usage` is the provider's own accounting, kept verbatim for audit. A
    shallow freeze looks identical in every assertion that reads it and still lets
    `answer.raw_usage["detail"]["tokens"] = 0` rewrite a billing record in place —
    which is why `Answer.__post_init__` calls `deep_freeze` rather than `dict`."""
    answer = _make("Answer")
    with pytest.raises(Exception):  # noqa: B017 — the module's own frozen-mapping type
        answer.raw_usage["input_tokens"] = 0

    nested = _make("SkillDocument").metadata
    assert nested["nested"] == [1]
    with pytest.raises(Exception):  # noqa: B017
        nested["nested"] = []


def test_the_answer_payload_is_deliberately_left_mutable() -> None:
    """The one documented exception, pinned so it reads as a decision rather than an
    oversight. `Answer.value` is the caller's own parsed output, typed `T` — freezing
    it would hand back a `FrozenDict` where the caller's annotation promised their
    model, so the freeze stops at `raw_usage`. If that ever changes, this test is the
    place that says it was intentional."""
    answer = _make("Answer")
    answer.value["added"] = 1
    assert answer.value["added"] == 1
