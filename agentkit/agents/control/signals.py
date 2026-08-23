"""Typed inter-actor signals — the framework's coordination protocol.

Pair with ``SignalChannel`` (``agentkit.agents.control.channel``). Two
disjoint hierarchies enforce direction at the type level: a parent
can't emit a ``DoneSignal``, a child can't emit a ``CancelSignal``.

- ``ControlSignal``  — parent → child (cancel, retask, reduce
                       budget, broadcast context).
- ``DataSignal``     — child → parent (progress, done, escalate,
                       blocked).

The common envelope (``correlation_id``, ``causation_id``,
``sender_id``, ``timestamp_us``) lets a signal stream be replayed
or audited as a cascade graph.

A signal is a VALUE, and that has to hold for its payload too, not
just its fields. ``frozen=True`` alone only blocks reassignment: it
left ``done.metrics["used_cost"] = 999.0`` and
``done.final_delta.append(...)`` working on an already-emitted
signal, and left the sender's own list aliased into it, so appending
to that list after emit retroactively rewrote a signal the parent
had already absorbed. Either one makes the audit trail above a
claim rather than a guarantee. So every collection payload is
COPIED and FROZEN at construction — see
``SignalEnvelope.__post_init__``. Passing a plain ``list`` / ``dict``
is still the ergonomic; you just get back a ``Sequence`` / ``Mapping``
that nobody downstream can edit.

Projects subclass ``ControlSignal`` / ``DataSignal`` for
domain-specific findings and parameterise the generics
(``RedirectSignal[MySnap]``, ``ContextUpdateSignal[MyMutation]``).
Mark your own collection fields with ``metadata=FROZEN_PAYLOAD`` to
get the same treatment.

Stdlib-only (no Pydantic — agentkit is zero-dep). Validation lives
at the agent boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Final, Generic, TypeVar

from agentkit.kernel._frozen import deep_freeze

# ── Type variables ──────────────────────────────────────────────────

StateT = TypeVar("StateT")
"""The snapshot / state type a Redirect signal carries. Defined by
the agent project (e.g. a Pydantic model OR a dataclass describing
the new context to retask a child with)."""

MutationT = TypeVar("MutationT")
"""The mutation type a Progress/Done/ContextUpdate signal carries.
Defined by the agent project (typically a discriminated union of
domain mutation kinds, each stamped with ``agent_id`` for
attribution that survives merges)."""


# ── Payload freezing ────────────────────────────────────────────────

_PAYLOAD_KEY: Final = "agentkit.frozen_payload"

# Deliberately still a ``MappingProxyType`` while the PAYLOADS moved to
# ``FrozenDict``: this is dataclass field METADATA, not a value. It never
# reaches ``json.dumps`` or ``dataclasses.asdict``, so none of the reasons that
# ruled the proxy out for payloads apply — and ``dataclasses`` wraps field
# metadata in a proxy of its own regardless.
FROZEN_PAYLOAD: Final[Mapping[str, bool]] = MappingProxyType({_PAYLOAD_KEY: True})
"""Field metadata marking a field as a COLLECTION payload that the
envelope copies and freezes at construction. Subclasses opt their own
payload fields in the same way the built-ins do::

    @dataclass(slots=True, frozen=True)
    class FindingSignal(DataSignal):
        urls: Sequence[str] = field(
            default_factory=tuple, hash=False, metadata=FROZEN_PAYLOAD
        )

Opt-in per field, rather than "freeze every list/dict-valued field
found", because ``StateT`` and ``MutationT`` are OPAQUE user types and
are very often plain dicts. Freezing ``RedirectSignal.new_state``
because it happened to be a dict would hand the child a
``mappingproxy`` where it passed a dict — ``json.dumps`` and
``**state`` both stop working — in exchange for a guarantee the caller
never asked for. The framework freezes the containers it defines and
leaves the user's own objects alone."""

_PAYLOAD_FIELDS: dict[type[Any], tuple[str, ...]] = {}
"""Per-class cache of the marked field names. ``fields()`` rebuilds a
tuple and re-filters metadata on every call, and signals are
constructed in the emit path of an ACK-less stream: on the 5-field
envelope that cost 3.00 µs/construction vs 1.89 µs cached (1.6×).
Keyed by the concrete class, so subclasses each get their own entry."""


def _freeze_payload(owner: str, name: str, value: Any) -> Any:
    """Copy ``value``, then return a read-only view of the copy.

    The copy is the half that is easy to skip and the half that
    matters most: freezing the caller's own object in place would not
    un-alias it, and measured, the aliasing bug is the nastier of the
    two — a sender that keeps appending to the list it emitted edits
    signals the receiver already processed, with no traceable moment
    where that happened.

    Shallow by design. ``tuple(value)`` / ``dict(value)`` freeze the
    CONTAINER; a ``list[dict]`` payload still hands out mutable dicts,
    and a mutation object with a mutable attribute is still mutable.
    Going deeper means recursively rewriting ``MutationT`` instances
    the framework knows nothing about — it would have to guess a
    reconstructor for every user type, and would silently swap
    dataclasses for copies, breaking identity comparisons and any
    ``__post_init__`` invariants those types maintain. The container is
    what the protocol owns and what the audit trail depends on; the
    contents belong to the project, which is where a deep-immutability
    policy can actually be enforced (frozen mutation dataclasses).
    """
    if isinstance(value, Mapping):
        # ``FrozenDict``, not ``MappingProxyType``. Both refuse mutation; only
        # one is still a ``dict``. This module's whole justification is that a
        # signal stream can be "replayed or audited as a cascade graph", and a
        # proxy is invisible to ``json.dumps`` and ``dataclasses.asdict`` — the
        # two things an audit trail is most likely to be run through. It also
        # cost a ``__reduce__`` on the envelope purely to make deepcopy and
        # pickle work at all; that hook, and the module-level rebuild factory
        # it had to name, are both gone now that the payload pickles itself.
        return deep_freeze(dict(value))
    if isinstance(value, (str, bytes)):
        # Under the old ``options: list[str]`` annotation, ``options="retry"``
        # was a type error the checker caught. ``Sequence[str]`` accepts a bare
        # ``str``, and tuple() would silently explode it into
        # ``('r','e','t','r','y')`` — five options, none of them real. Refusing
        # keeps the widened annotation from costing a check we used to get.
        raise TypeError(
            f"{owner}.{name} takes a sequence of items, not a bare "
            f"{type(value).__name__} ({value!r:.40}) — wrap it: [{value!r:.30}]"
        )
    # A tuple is already immutable at the container level AND serialises to a
    # JSON array, so it needs no replacement — but its ELEMENTS may be dicts or
    # lists, and those were left mutable. ``deep_freeze`` reaches them without
    # touching the project's own ``MutationT`` objects, which is exactly the
    # line this module already drew.
    return tuple(deep_freeze(v) for v in value)


# ── Common envelope ─────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class SignalEnvelope:
    """Common stamp every signal carries.

    Fields:

    - ``correlation_id`` — the originating signal that started this
      cascade. Lets the audit timeline collapse a fan-out cascade
      back to its trigger.
    - ``causation_id``   — the immediate predecessor of THIS signal
      in the cascade graph. Distinct from the agent's ``parent_id``
      (which is structural) — causation tracks the message DAG.
    - ``sender_id``      — id of the agent that emitted the signal.
      Stamped at emit time by the channel; user code typically
      leaves it unset at construction.
    - ``timestamp_us``   — monotonic microseconds since channel
      start. The channel stamps this at emit time too.

    Also carries the payload-freezing machinery every signal in the
    hierarchy inherits (``__post_init__``). There is no ``__reduce__``
    here and no rebuild factory: a mapping payload becomes a
    ``FrozenDict``, which pickles and deep-copies itself, and a sequence
    payload becomes a plain ``tuple`` (with any nested containers frozen
    inside it), which always could — so the envelope needs no hook of
    its own. Note the freeze is NOT re-applied on the way back: pickle
    takes the default protocol, which restores state directly and never
    calls ``__post_init__``. Verified for a user subclass with extra
    fields, which round-trips by both routes with its payload still
    refusing mutation at every nested level.
    """

    correlation_id: str | None = None
    causation_id: str | None = None
    sender_id: str | None = None
    timestamp_us: int = 0

    def __post_init__(self) -> None:
        """Copy-then-freeze every field marked ``FROZEN_PAYLOAD``.

        Lives on the envelope rather than on each payload-carrying
        signal so a subclass that adds a payload field inherits the
        guarantee by marking it — and so there is exactly one place
        that decides what "frozen" means. A subclass defining its own
        ``__post_init__`` REPLACES this one and must chain, which is
        why this exists as a callable no-op even on the payload-free
        signals. Chain by NAMING the base, not with a bare ``super()``::

            def __post_init__(self) -> None:
                ...                                # your normalisation
                SignalEnvelope.__post_init__(self)

        ``super().__post_init__()`` is the obvious spelling, and whether
        it works depends on your interpreter — which is the reason to
        avoid it rather than a reason to shrug. ``@dataclass(slots=True)``
        cannot add slots in place, so it builds a REPLACEMENT class, and
        the zero-argument ``super()`` closes over the original, which the
        instance is not an instance of. CPython fixed that, mid-3.13:

            3.12      TypeError: ... must be an instance or subtype
            3.13.2    TypeError: ... is not an instance or subtype
            3.13.14   works

        So on a supported interpreter the bare spelling may raise or may
        not, and non-slotted subclasses were never affected at all — the
        code works right up until someone adds ``slots=True`` on a
        Python where it still bites. Naming the base is correct on every
        version and needs no table to explain, which is why it is what
        this prescribes.

        Otherwise compatible with ``slots=True``: slotted frozen
        dataclasses still route writes through ``object.__setattr__``,
        and the fields already exist as slot descriptors by the time
        ``__post_init__`` runs.
        """
        own = type(self)
        names = _PAYLOAD_FIELDS.get(own)
        if names is None:
            names = _PAYLOAD_FIELDS.setdefault(
                own,
                tuple(f.name for f in fields(self) if f.metadata.get(_PAYLOAD_KEY)),
            )
        for name in names:
            object.__setattr__(
                self, name, _freeze_payload(own.__name__, name, getattr(self, name))
            )


@dataclass(slots=True, frozen=True)
class ControlSignal(SignalEnvelope):
    """Parent → child directive base. Subclass to participate in
    a project's control-plane dispatch. The framework dispatches by
    ``isinstance``; no discriminator field is required."""


@dataclass(slots=True, frozen=True)
class DataSignal(SignalEnvelope):
    """Child → parent result base. Subclass to participate in a
    project's data-plane dispatch."""


# ── Control plane (parent → child) — ready-made generic signals ────


@dataclass(slots=True, frozen=True)
class CancelSignal(ControlSignal):
    """Hard stop. The child finishes its current tool call, ships
    its final delta upward, and exits. Distinct from
    ``MergeWithPeerSignal`` in that no successor takes over — the
    work is just terminated."""

    reason: str = ""


@dataclass(slots=True, frozen=True)
class BudgetReducedSignal(ControlSignal):
    """Parent shrunk the child's remaining budget envelope.

    ``constraints`` is an open dict-of-numerics rather than a fixed
    set of axes because different projects use different budget
    shapes (some care about tokens + cost only, some add wall-clock
    + steps). The child applies whichever keys it recognises and
    ignores the rest.

    Pass a plain dict; you get back a read-only ``Mapping`` that
    still compares equal to that dict.
    """

    constraints: Mapping[str, float] = field(
        default_factory=dict, hash=False, metadata=FROZEN_PAYLOAD
    )


@dataclass(slots=True, frozen=True)
class RedirectSignal(ControlSignal, Generic[StateT]):
    """Parent retasked the child mid-flight — typically the parent
    refined a sub-task and is handing the child a new context.

    The child replaces its read-only snapshot with ``new_state`` and
    typically clears any role-cached state derived from the old
    snapshot. Local journal stays — the child's authored history
    survives the retask.

    ``new_state`` is passed through untouched: it is an opaque
    ``StateT``, so the framework neither copies nor freezes it (see
    ``FROZEN_PAYLOAD``). It is excluded from the hash for the same
    reason — a project's snapshot type is under no obligation to be
    hashable, and ``RedirectSignal`` should not be the one signal in
    the protocol whose hashability depends on a user type.
    """

    new_state: StateT | None = field(default=None, hash=False)
    reason: str = ""


@dataclass(slots=True, frozen=True)
class ContextUpdateSignal(ControlSignal, Generic[MutationT]):
    """Parent broadcasts a useful new piece of context to siblings —
    typically a finding from sibling A that siblings B + C should
    fold into their own awareness (e.g. "A rejected
    contentfarm.com/x, don't waste a fetch on it").

    The child applies ``mutations`` to its own context without
    claiming authorship — the originating ``agent_id`` on each
    mutation preserves the audit trail.
    """

    mutations: Sequence[MutationT] = field(
        default_factory=tuple, hash=False, metadata=FROZEN_PAYLOAD
    )


@dataclass(slots=True, frozen=True)
class MergeWithPeerSignal(ControlSignal):
    """Parent decided this child's task overlaps with a peer's;
    hand control over to the survivor. The child ships its final
    delta upward and exits. The parent typically renders a
    "merged_into" relationship on its visualisation surface."""

    survivor_agent_id: str = ""
    reason: str = ""


# ── Data plane (child → parent) — ready-made generic signals ────────


@dataclass(slots=True, frozen=True)
class ProgressSignal(DataSignal, Generic[MutationT]):
    """Streamed delta — child made progress, here's what changed.

    Used as an ACK-less stream: the child emits at every meaningful
    boundary and the parent absorbs without replying. The child
    advances its journal watermark only after the parent has
    absorbed (the runner orchestrates this via the merge loop).

    ``mutations`` is frozen at construction, which matters more here
    than anywhere else: the child typically emits a slice of a
    journal list it goes on appending to.
    """

    mutations: Sequence[MutationT] = field(
        default_factory=tuple, hash=False, metadata=FROZEN_PAYLOAD
    )
    confidence: float | None = None


@dataclass(slots=True, frozen=True)
class DoneSignal(DataSignal, Generic[MutationT]):
    """Terminal signal — the child finished its work.

    Ships the FINAL delta (whatever the journal accumulated past
    the last watermark) plus the final confidence + optional
    metrics rollup (used_tokens, used_cost, used_steps — projects
    pick the keys that matter to them).

    The parent absorbs ``final_delta`` into its own journal, drops
    the child registry entry, and refunds the unused budget slice.
    Both payloads are frozen because this is the signal the budget
    ledger and the audit trail are reconciled from: a receiver that
    could still write ``metrics["used_cost"] = 999.0`` after the fact
    makes the reported spend unverifiable.
    """

    final_delta: Sequence[MutationT] = field(
        default_factory=tuple, hash=False, metadata=FROZEN_PAYLOAD
    )
    confidence: float = 0.0
    metrics: Mapping[str, float] = field(
        default_factory=dict, hash=False, metadata=FROZEN_PAYLOAD
    )


@dataclass(slots=True, frozen=True)
class EscalateSignal(DataSignal):
    """Child can't make progress and is asking the parent to pick a
    recovery. ``options`` is a structured menu so the parent (or
    its LLM planner) can choose deterministically — typical entries
    are project-defined verbs like ``"retry"``, ``"retire"``,
    ``"reformulate"``, ``"spawn_helper"``.

    The child typically blocks (no new work) until the parent
    replies with a ``RedirectSignal`` / ``BudgetReducedSignal`` /
    ``CancelSignal`` / ``ContextUpdateSignal``.
    """

    reason: str = ""
    options: Sequence[str] = field(
        default_factory=tuple, hash=False, metadata=FROZEN_PAYLOAD
    )


@dataclass(slots=True, frozen=True)
class BlockedSignal(DataSignal):
    """Child has hit a wall with no proposed recovery. Same effect
    as ``EscalateSignal`` from the parent's POV, but the absence of
    ``options`` signals "I genuinely don't know what to do next" —
    typically leads to the parent retiring the child's task."""

    reason: str = ""


__all__ = [
    "FROZEN_PAYLOAD",
    "BlockedSignal",
    "BudgetReducedSignal",
    "CancelSignal",
    "ContextUpdateSignal",
    "ControlSignal",
    "DataSignal",
    "DoneSignal",
    "EscalateSignal",
    "MergeWithPeerSignal",
    "MutationT",
    "ProgressSignal",
    "RedirectSignal",
    "SignalEnvelope",
    "StateT",
]
