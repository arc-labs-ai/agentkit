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

Projects subclass ``ControlSignal`` / ``DataSignal`` for
domain-specific findings and parameterise the generics
(``RedirectSignal[MySnap]``, ``ContextUpdateSignal[MyMutation]``).

Stdlib-only (no Pydantic — agentkit is zero-dep). Validation lives
at the agent boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

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
    """

    correlation_id: str | None = None
    causation_id: str | None = None
    sender_id: str | None = None
    timestamp_us: int = 0


# ── Direction-typed bases ───────────────────────────────────────────


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
    """

    constraints: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RedirectSignal(ControlSignal, Generic[StateT]):
    """Parent retasked the child mid-flight — typically the parent
    refined a sub-task and is handing the child a new context.

    The child replaces its read-only snapshot with ``new_state`` and
    typically clears any role-cached state derived from the old
    snapshot. Local journal stays — the child's authored history
    survives the retask.
    """

    new_state: StateT | None = None
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

    mutations: list[MutationT] = field(default_factory=list)


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
    """

    mutations: list[MutationT] = field(default_factory=list)
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
    """

    final_delta: list[MutationT] = field(default_factory=list)
    confidence: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


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
    options: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class BlockedSignal(DataSignal):
    """Child has hit a wall with no proposed recovery. Same effect
    as ``EscalateSignal`` from the parent's POV, but the absence of
    ``options`` signals "I genuinely don't know what to do next" —
    typically leads to the parent retiring the child's task."""

    reason: str = ""


__all__ = [
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
