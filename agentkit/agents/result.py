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
#   terminated        a ``TerminationCondition`` fired, or the run was
#                     stopped deliberately from outside (a cancel, a person
#                     declining at a gate). The specific reason string is in
#                     ``evals["stop_reason"]``.
#   failed            the run ended in an ERROR that the cognition chose to
#                     report as data instead of raising. Rare and deliberate:
#                     ``ClaudeCLICognition`` guarantees a terminal event even
#                     when the subprocess never starts, so a spawn failure or
#                     a non-zero exit arrives as a result rather than an
#                     exception. ``output`` holds whatever text was produced
#                     before the failure and ``evals["stop_reason"]`` names
#                     the mode (``spawn_failed``, ``cli_exit_2``, ...).
#
# A run that fails usually does not produce an ``AgentResult`` at all — the
# exception propagates out of ``Agent.run`` / ``Agent.stream``. ``failed``
# exists for the one cognition that deliberately does otherwise; mapping its
# spawn failures onto ``terminated`` would have read as "something stopped this
# on purpose", which is the opposite of what happened. So "suspended vs failed"
# is the difference between ``stop_reason == "suspended"`` and either an
# exception or ``stop_reason == "failed"``. That is the distinction Brief 4
# asks a reader to be able to act on.
AgentStopReason = Literal[
    "complete",
    "suspended",
    "expired",
    "budget_exhausted",
    "max_iterations",
    "invalid_output",
    "terminated",
    "failed",
]

# Terminal reasons that leave the run RESUMABLE — a durable checkpoint
# exists and ``Agent.resume`` (or a fresh ``drive`` over the same
# ``correlation_id``) can pick it back up. Exposed as a frozenset so
# callers branch on membership rather than re-listing the strings.
RESUMABLE_STOP_REASONS: frozenset[str] = frozenset({"suspended", "budget_exhausted"})


# Free-form ``reason`` → closed ``AgentStopReason``. Lives beside the taxonomy
# because EVERY producer needs it: the tool loop, the coordinator policies
# (round-robin, selector, ledger, plan) and any pattern a user writes. Two
# copies of this table is how a suspended plan came to report itself
# ``complete`` — the policies simply never mapped, so the typed field kept its
# default while ``evals["stop_reason"]`` said ``"awaiting_decision"``.
#
# Only the strings the framework passes ITSELF are listed. Anything else — a
# custom ``TerminationCondition``'s own wording — resolves to ``"terminated"``,
# which is the honest answer: something stopped the run deliberately and we are
# not going to invent a more specific category for it. That fallback is what
# makes the mapping TOTAL, so a producer can always set the field and no reader
# ever sees a default that outlived its meaning.
_REASON_TO_STOP: dict[str, AgentStopReason] = {
    # ── the tool loop ───────────────────────────────────────────────────────
    "complete": "complete",
    "awaiting_approval": "suspended",
    "awaiting_input": "suspended",
    "expired": "expired",
    "budget_exhausted": "budget_exhausted",
    "max_iterations": "max_iterations",
    "invalid_output": "invalid_output",
    # ── coordinator policies ────────────────────────────────────────────────
    # A plan parked on a human gate is the same KIND of stop as a tool call
    # parked on an approval: resumable, not failed. Anything else here is
    # terminal.
    "awaiting_decision": "suspended",
    "plan_complete": "complete",
    "satisfied": "complete",  # the ledger assessor judged the goal met
    "rejected": "terminated",  # a person declined at a gate
    "stalled": "terminated",  # replans exhausted without progress
    "no_children": "terminated",  # nothing to dispatch to
    # Turn/round ceilings are the coordinator's spelling of "iteration
    # ceiling". ``MaxTurns``/``MaxMessages`` emit the same strings, so a
    # condition firing and a policy ceiling tripping agree — which is correct,
    # they mean the same thing to a reader deciding whether to raise the limit.
    "max_turns": "max_iterations",
    "max_rounds": "max_iterations",
    "max_messages": "max_iterations",
    # ── the Claude CLI cognition ────────────────────────────────────────────
    # Its FAILURE reasons are mapped locally in ``cognition/claude_cli.py``,
    # because ``cli_exit_<n>`` is dynamic and cannot be enumerated here.
    "success": "complete",
    "cancelled": "terminated",
    # A person pressed stop. Deliberate, so ``terminated`` — and distinct from
    # ``cancelled``, which in that cognition means the process was killed and
    # the conversation is gone.
    "interrupted": "terminated",
}


def stop_reason_for(reason: str | None) -> AgentStopReason:
    """Map a producer's free-form stop reason onto the closed taxonomy.

    Total by construction: an unrecognised reason becomes ``"terminated"``.
    ``None`` (a producer that finished without recording a reason) becomes
    ``"complete"`` — reaching the end of the work IS completion.

    Keep the free-form string in ``evals["stop_reason"]`` as well; it carries
    the detail this field deliberately drops.
    """
    if reason is None:
        return "complete"
    return _REASON_TO_STOP.get(reason, "terminated")


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

    def __hash__(self) -> int:
        """Hash the RUN's identity — ``(output, usage, partial, prompt_version,
        stop_reason)`` — never ``evals``, never ``parsed``.

        This is the type every ``agent.run()`` returns, and it was unhashable
        for EVERY instance rather than for awkward ones. ``evals`` is declared
        ``field(default_factory=dict)``, so the dataclass-generated all-fields
        hash reached a dict on the plainest result the framework can build.
        Measured before this fix::

            hash(AgentResult(output="hi", usage=Usage()))   TypeError: unhashable type: 'dict'
            hash(AgentResult(output="hi", usage=Usage(), evals={"a": 1}))
            TypeError: unhashable type: 'dict'

        Both lines, identically — so it broke by TYPE, not by value, and no
        caller could ever have hit a working case to compare against. Nothing
        inside the framework hashes a result, which is why it survived: the
        first caller to put results in a ``set`` (deduping a fan-out's
        per-child results is the obvious one) or to memoize on one would have
        been the one to find it.

        ``evals`` is excluded because it cannot be hashed at all: it is
        free-form, nested application JSON — the tool loop writes
        ``{"stop_reason": ..., "suspended": ...}``, the coordinator policies
        write whole per-child result lists — and it is handed to callers to
        write into as they like. A hash that read it would make hashability
        depend on what a policy happened to record.

        ``parsed`` is excluded for that reason and one more: it is whatever an
        output parser produced — typically a Pydantic model, which is
        unhashable unless the application declared it frozen, or a plain
        dict/list for the JSON adapters. Hashability of the framework's most
        public type would then depend on a caller's schema.

        What IS hashed is the part a reader identifies a run BY: the text, the
        spend, whether it is partial, which prompt version produced it, and how
        it ended. ``usage`` is a frozen dataclass of ints/floats, so it is
        hashable by construction and costs nothing.

        Cost is O(1) in the payload — the excluded fields are never read.
        Measured with ``evals`` holding one key vs 100_000: 0.62 µs and
        0.60 µs, and ``hash(small) == hash(huge)`` proves it structurally
        rather than by timing, so the test cannot go flaky. ``output`` is a
        ``str`` of unbounded length, but CPython caches a string's hash on the
        object, so a long output is walked at most once.

        Excluding all of that is sound rather than a workaround: ``__eq__``
        still compares every field, and the hash invariant only requires EQUAL
        objects to hash equally — never that unequal ones differ. Two results
        with the same text and spend but different ``evals`` collide into one
        bucket and ``__eq__`` separates them there; that is what a bucket is
        for, and it keeps a ``set`` of results EXACT.
        """
        return hash((self.output, self.usage, self.partial, self.prompt_version, self.stop_reason))

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

    def __hash__(self) -> int:
        """Hash the run's SHAPE — which nodes ran, the spend, the step count and
        the stop reason — never the per-node output VALUES.

        ``outputs`` is a required ``dict``, so a ``WorkflowResult`` was
        unhashable for every instance, exactly like ``AgentResult``. Measured
        before this fix::

            hash(WorkflowResult(outputs={"n": "o"}, usage=Usage(), steps=1,
                                stop_reason="complete"))
            TypeError: unhashable type: 'dict'

        The node NAMES are kept and the values dropped. Keys are safe to hash
        unconditionally — anything that is already a dict key is hashable by
        definition, so no annotation has to be trusted for this to hold — and
        they are the part that actually varies between the workflows a cache
        or a dedup sees: a run that reached ``summarise`` differs from one that
        stopped at ``draft`` even when both outputs are opaque. The values are
        whatever the nodes returned: node outputs are ``Any`` and are commonly
        dicts or ``AgentResult``-shaped payloads.

        They go in as a ``frozenset``, NOT a tuple of keys, and that is
        load-bearing rather than taste. Dict equality ignores insertion order,
        so ``{"a": 1, "b": 2} == {"b": 2, "a": 1}`` — two ``WorkflowResult``s
        that compare EQUAL. A key tuple would hash those two differently and
        break the one invariant a hash has to keep. ``frozenset`` is
        order-independent, and it does not require the keys to be mutually
        COMPARABLE the way ``sorted()`` would.

        ``suspended`` is excluded. It is decoration on top of ``stop_reason``,
        which already records that the run parked, and ``Suspended.elicitations``
        is a ``tuple[Any, ...]`` deliberately typed loosely to keep this module
        free of an import cycle — so folding it in would reintroduce
        "hashable, depending on what the caller put in there".

        Cost is O(#nodes), never O(payload): measured at 0.62 µs whether a
        node's output holds one key or 100_000. Soundness is the same argument
        as everywhere else here — ``__eq__`` still compares every field, and
        only EQUAL objects are required to hash equally, so two results that
        differ only in node outputs share a bucket and ``__eq__`` splits them.
        """
        return hash((frozenset(self.outputs), self.usage, self.steps, self.stop_reason))
