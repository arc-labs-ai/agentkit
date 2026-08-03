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
