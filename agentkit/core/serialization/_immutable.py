"""Frozen payload containers — immutability that survives serialization.

WHY THIS EXISTS
---------------
Twelve public types are ``@dataclass(frozen=True)`` while carrying a mutable
``dict``/``list`` payload, so ``frozen`` stopped at the field reference::

    cp = Checkpoint(...)        # a DURABLE record
    cp.state = {}               # FrozenInstanceError, as intended
    cp.state["turn"] = 99       # ...but this rewrote it, silently

The obvious fix is ``MappingProxyType``, and it is the wrong one here. Measured
against the four things these payloads actually have to do:

                     json.dumps  asdict     deepcopy   pickle   isinstance(dict)
    MappingProxyType TypeError   TypeError  TypeError  TypeError  False
    FrozenDict       ok          ok         ok         ok         True

``Checkpoint.state`` is ``json.dumps``'d into a JSONB column and
``AgentResult`` round-trips through ``dataclasses.asdict``, so a proxy would
have traded a mutability bug for a data-path outage. A ``dict`` SUBCLASS keeps
every consumer working — serialisers, ``isinstance`` checks, equality against
plain dicts — while refusing mutation.

``__reduce__`` is load-bearing rather than decorative. ``copy`` and ``pickle``
rebuild a dict subclass by creating an empty one and repopulating it through
``__setitem__`` — the exact method being blocked — so without it a frozen
payload raises on ``deepcopy``, which the checkpointer does on every snapshot.
That is the same hazard that ``Prompt`` and ``ToolCall`` each had to solve.

Freezing is DEEP. A shallow freeze leaves ``cp.state["a"]["b"] = 1`` working,
which is the same bug one level down and harder to see. Cost is O(payload),
paid once at construction and measured BELOW the ``deepcopy`` these seams
already perform: 3.0 us for a small dict, 148 us for 50 nested entries, and
2.9 ms for 10,000 keys, against 3.7 us / 205 us / 5.0 ms for ``deepcopy``.

Non-container leaves are returned untouched. Recursively rewriting arbitrary
user objects would mean guessing a reconstructor per type and silently swapping
identities — the same line ``signals.py`` draws.
"""

from __future__ import annotations

import copy
from types import MappingProxyType
from typing import Any

# NOT a format string — nothing calls `.format()` on it, so doubled braces
# would reach the user verbatim. They did: the message shipped reading
# `field={{**obj.field, ...}}`, and the test that was supposed to guard it
# matched only on `dataclasses.replace`, which is loose enough to pass either
# way. The test now asserts the exact text.
_FROZEN_MSG = (
    "this payload belongs to a frozen value and cannot be mutated in place. "
    "Build a new one instead: dataclasses.replace(obj, field={**obj.field, ...})"
)


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise TypeError(_FROZEN_MSG)


class FrozenDict(dict[Any, Any]):
    """A ``dict`` that refuses mutation but IS a ``dict``.

    Subclassing rather than wrapping is the whole point: ``json.dumps``,
    ``dataclasses.asdict``, ``isinstance(x, dict)`` and ``x == {"a": 1}`` all
    keep working, which is what ``MappingProxyType`` gives up.

    Note it cannot be used as a dataclass FIELD DEFAULT: ``@dataclass`` rejects
    any unhashable default, and a ``FrozenDict`` is unhashable exactly like the
    ``dict`` it subclasses. Use ``field(default_factory=FrozenDict)``.
    """

    __slots__ = ()

    __setitem__ = _refuse
    __delitem__ = _refuse
    __ior__ = _refuse
    update = _refuse
    pop = _refuse
    popitem = _refuse
    clear = _refuse
    setdefault = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuild through the CONSTRUCTOR. copy/pickle otherwise repopulate via
        # __setitem__, which we just blocked — measured: deepcopy and pickle
        # both raised TypeError before this existed.
        return (FrozenDict, (dict(self),))

    def __copy__(self) -> FrozenDict:
        return FrozenDict(dict(self))

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return FrozenDict(copy.deepcopy(dict(self), memo))


class FrozenList(list[Any]):
    """A ``list`` that refuses mutation but IS a ``list``.

    Same reasoning as :class:`FrozenDict`. A ``tuple`` would have been simpler
    and is wrong for the same reason a proxy is: it breaks ``isinstance(x,
    list)`` and changes what callers get back from a field they already read.
    """

    __slots__ = ()

    __setitem__ = _refuse
    __delitem__ = _refuse
    __iadd__ = _refuse
    __imul__ = _refuse
    append = _refuse
    extend = _refuse
    insert = _refuse
    remove = _refuse
    pop = _refuse
    clear = _refuse
    sort = _refuse
    reverse = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenList, (list(self),))

    def __copy__(self) -> FrozenList:
        return FrozenList(list(self))

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenList:
        return FrozenList(copy.deepcopy(list(self), memo))


def thaw(value: Any) -> Any:
    """The inverse of :func:`deep_freeze` — return plain, writable containers.

    Needed because freezing is a property of a SNAPSHOT, not of the data. A
    durable record is rightly frozen; a context RESUMED from that record is a
    live working context and has to be writable again.

    Measured before this existed: ``Checkpoint.state`` is deep-frozen, and
    ``persistence.rehydrate`` passed the stored scratchpad straight through, so
    a resumed run raised ``TypeError: this payload belongs to a frozen
    value...`` on its FIRST ``ctx.note(...)``. A top-level ``dict(...)`` — the
    obvious fix, and what a sibling decoder already did — only moves the
    failure one level down, to the first nested write.

    Like ``deep_freeze``, this rebuilds containers and returns everything else
    by identity: a caller's own object is not reconstructed.
    """
    if isinstance(value, dict):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, list):
        return [thaw(v) for v in value]
    return value


def deep_freeze(value: Any) -> Any:
    """Return ``value`` with every nested dict/list replaced by a frozen one.

    Idempotent: an already-frozen container is returned as-is rather than
    rebuilt, so re-freezing a payload that was passed between two frozen values
    costs one isinstance check instead of another full walk.

    Sets and tuples are left alone deliberately. A ``tuple`` is already
    immutable at the container level, and a ``set`` of mutable members cannot
    exist (its members must be hashable), so neither is a way to smuggle a
    mutation back in through this door.
    """
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, MappingProxyType):
        # Normalised, not passed through. A ``MappingProxyType`` is already
        # immutable, so leaving it alone looks harmless — and it silently
        # defeats the reason this module exists. Measured with it passed
        # through: a caller who handed a proxy to ``ToolCall(arguments=...)``
        # got it stored VERBATIM, and then
        # ``json.dumps(tc.arguments)`` raised
        # ``Object of type mappingproxy is not JSON serializable`` on a payload
        # the type advertises as serialisable. The defensive ``dict(...)``
        # unwraps scattered around the codebase covered only the TOP level of
        # that, so a nested proxy still broke.
        #
        # Only the stdlib proxy is rewritten. Other ``Mapping`` subclasses are
        # left alone: they are the project's own types, and swapping them for a
        # FrozenDict would be the "silently reconstruct a user object" line
        # this module draws everywhere else.
        return FrozenDict({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, dict):
        return FrozenDict({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList([deep_freeze(v) for v in value])
    return value


# ── why a frozen payload still needs a hand-written __hash__ ────────────────
#
# `FrozenDict` and `FrozenList` are immutable, but they are `dict`/`list`
# SUBCLASSES, and those are unhashable. So freezing a payload does not make its
# owner hashable — every frozen dataclass carrying one defines `__hash__` over
# an identity SUBSET and leaves `__eq__` untouched.
#
# That is sound rather than a shortcut: the hash invariant only requires EQUAL
# objects to hash equally, never that unequal ones differ. Two values differing
# only in payload share a bucket, and `__eq__` separates them there. Each such
# `__hash__` says only what its OWN subset is and why; this is the shared half.


__all__ = ["FrozenDict", "FrozenList", "deep_freeze", "thaw"]
