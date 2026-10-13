"""Sharp assertions for things Python's ``==`` is too generous about.

Every helper here exists because a plain ``assert a == b`` was found to pass
against a deliberately broken implementation. They are the difference between
a test that describes the intent and a test that enforces it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

__all__ = ["assert_money", "assert_no_float_money", "assert_frozen"]


def assert_money(actual: Any, expected: str, *, label: str = "value") -> None:
    """Assert an exact monetary amount AND that it is genuinely a ``Decimal``.

    ``assert budget.spent() == Decimal("1.00")`` looks airtight and is not:

        >>> Decimal("1.00") == 1.0
        True

    So a ``spent()`` that regressed to returning a float sails through. Worse,
    the leniency is inconsistent — ``Decimal("0.1") == 0.1`` is ``False``,
    because 0.1 has no exact binary representation while 1.00 does. That means
    whether a float regression is caught depends on which number the test
    happened to pick, which is not a property anyone should rely on.

    Discovered by mutation-testing this suite: replacing ``Budget.spent()``'s
    body with ``float(self._spent)`` left all 78 conformance tests green.

    ``expected`` is a STRING, not a ``Decimal``, so the literal in the test
    reads as the money it represents and cannot itself be built from a lossy
    float.
    """
    assert isinstance(actual, Decimal), (
        f"{label} must be a Decimal, got {type(actual).__name__} ({actual!r}). "
        f"Equality alone would not have caught this: Decimal('1.00') == 1.0 is True."
    )
    assert actual == Decimal(expected), f"{label}: expected {expected}, got {actual}"


def assert_no_float_money(*values: Any, label: str = "money") -> None:
    """Assert none of these monetary values is a float.

    ``bool`` is a subclass of ``int`` and ``int`` is fine for money, but a
    float anywhere in a ledger chain is how exactness is lost — one
    ``float(...)`` in an accumulator and a hundred cents stops being a dollar.
    """
    for value in values:
        assert not isinstance(value, float), (
            f"{label}: {value!r} is a float; monetary values must be Decimal or int"
        )


def assert_frozen(instance: Any, field: str) -> None:
    """Assert a value type really refuses mutation.

    ``@dataclass(frozen=True)`` is easy to drop in a refactor, and nothing
    else notices until two call sites start sharing a mutated object. The
    kernel's "values are immutable" invariant is only real if something checks.
    """
    import dataclasses

    try:
        setattr(instance, field, object())
    except (dataclasses.FrozenInstanceError, AttributeError):
        return
    raise AssertionError(
        f"{type(instance).__name__}.{field} accepted a write — the type is not frozen"
    )
