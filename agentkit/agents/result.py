"""Shared agent / workflow result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agentkit.kernel.types import Usage

if TYPE_CHECKING:
    from agentkit.kernel.types import ToolCall

# The workflow's terminal reason is closed — a ``Literal`` catches a typo
# at the type-check seam instead of at read time.
WorkflowStopReason = Literal["complete", "suspended", "max_steps", "deadlock"]

# The agent's terminal reason, as a CLOSED taxonomy. This is the field a
# reader branches on, and the distinctions are behavioural, not cosmetic:
#
#   complete          the model produced a final answer — nothing to do
#   suspended         WAITING ON A PERSON. Resumable via ``Agent.resume``.
#                     NOT a failure: the run is intact and parked.
#   expired           a human-gate deadline passed with no answer. The run
#                     DEGRADED and finished rather than hanging — an
#                     ordinary recorded outcome, not an error.
#   budget_exhausted  a meter ceiling was reached. A checkpoint was written
#                     BEFORE stopping, so the spend is recoverable.
#   max_iterations    the tool-loop ceiling was hit with no final answer
#   invalid_output    parse-and-repair exhausted; ``output`` is the last
#                     (unparseable) text
#   terminated        a ``TerminationCondition`` fired. The specific
#                     condition's own reason string is in
#                     ``evals["stop_reason"]``.
#
# A run that FAILED does not produce an ``AgentResult`` at all — the
# exception propagates out of ``Agent.run`` / ``Agent.stream``. So
# "suspended vs failed" is the difference between getting a result whose
# ``stop_reason == "suspended"`` and getting an exception. That is the
# distinction Brief 4 asks a reader to be able to act on.
AgentStopReason = Literal[
    "complete",
    "suspended",
    "expired",
    "budget_exhausted",
    "max_iterations",
    "invalid_output",
    "terminated",
]

# Terminal reasons that leave the run RESUMABLE — a durable checkpoint
# exists and ``Agent.resume`` (or a fresh ``drive`` over the same
# ``correlation_id``) can pick it back up. Exposed as a frozenset so
# callers branch on membership rather than re-listing the strings.
RESUMABLE_STOP_REASONS: frozenset[str] = frozenset({"suspended", "budget_exhausted"})


@dataclass(frozen=True)
class AgentResult:
    output: str  # the raw model text
    usage: Usage
    partial: bool = False
    evals: dict[str, Any] = field(default_factory=dict)
    parsed: Any = None  # the validated/typed object when an output parser is set
    # (None if no parser, or if repair was exhausted → partial)
    prompt_version: str = ""  # the RequestBuilder's prompt_version for this run — empty if no
    # RequestBuilder/Prompt was used. Lets the caller attribute the
    # output to a specific template without poking at traces.
    stop_reason: AgentStopReason = "complete"  # WHY the run ended, as a closed taxonomy —
    # see ``AgentStopReason`` above. ``evals["stop_reason"]``
    # still carries the free-form detail string (a
    # TerminationCondition's own reason, for instance) and is
    # kept verbatim for back-compat; THIS field is the one to
    # branch on, because it type-checks.

    @property
    def is_suspended(self) -> bool:
        """True when the run is parked waiting on a person. Distinct from failure:
        a failed run raises instead of returning an ``AgentResult`` at all."""
        return self.stop_reason == "suspended"

    @property
    def is_resumable(self) -> bool:
        """True when a durable checkpoint exists and the run can be continued —
        ``suspended`` (waiting on a human) or ``budget_exhausted`` (waiting on a
        raised ceiling). See :data:`RESUMABLE_STOP_REASONS`."""
        return self.stop_reason in RESUMABLE_STOP_REASONS


@dataclass(frozen=True)
class Suspended:
    """Carried in `AgentResult.evals['suspended']` when the loop pauses for human approval.

    ``pending`` is a tuple, not a list — the operator UI renders the
    pending items and the resume path threads them back verbatim; a
    mutable list inside a frozen shell would let a stray
    ``suspended.pending.append(...)`` desync the operator's rendered
    view from what actually resumes. The frozen tuple pins the
    handshake at both ends.

    ``pending`` is narrowed to a tuple of ``ToolCall`` OR a tuple of
    ``str`` (gate-name identifiers, emitted by ``Workflow`` when a
    ``human_gate`` node suspends) — the two suspend surfaces produce
    different-shaped identifiers, and the union catches drift from a
    third caller passing arbitrary objects.
    """

    run_id: str
    pending: tuple[ToolCall, ...] | tuple[str, ...]
    reason: str = "awaiting_approval"
    elicitations: tuple[Any, ...] = ()  # tuple[Elicitation, ...] — what the run is asking a
    # person FOR, when the suspend is a value request rather
    # than a plain approve/deny on a tool call. Empty for the
    # classic approval suspend, so existing readers are
    # unaffected. Typed ``Any`` to keep ``result.py`` free of an
    # import from ``agents.control`` (which imports this module).
    deadline_at: float | None = None  # wall-clock epoch seconds after which this suspend is
    # considered abandoned. ``None`` (default) = no deadline,
    # exactly today's behaviour. An operator UI renders the
    # countdown; the resume path refuses a decision that
    # arrives late. See ``agents.control.elicitation``.


@dataclass(frozen=True)
class WorkflowResult:
    """Terminal result of a `Workflow` run. Carries every node's latest output, the merged
    usage, the number of node executions (incl. re-runs from loop-back), the stop reason
    (``complete`` | ``suspended`` | ``max_steps`` | ``deadlock``), and a ``Suspended`` when
    the run paused on a human-gate."""

    outputs: dict[str, Any]
    usage: Usage
    steps: int
    stop_reason: WorkflowStopReason
    suspended: Suspended | None = None
