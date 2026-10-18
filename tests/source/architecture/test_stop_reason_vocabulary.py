"""The stop-reason vocabulary is CLOSED, and this is what closes it.

WHY THIS IS AN ARCHITECTURE TEST
--------------------------------
`stop_reason` was typed `AgentStopReason = str`. Several docstrings described
the vocabulary as a closed taxonomy, and nothing anywhere made that true, so
the two ways it can rot both happened:

* **A reason the enum does not know.** Each provider driver keeps its own
  `frozenset` of reason strings and maps them through the shared terminal
  mapper. A set member that is not in the vocabulary is a reason an operator
  sees and no caller can branch on — and a typo in one of those frozensets is
  invisible, because an unrecognised reason passes straight through.

* **A reason that does not exist.** `ClaudeCliSession.interrupt`'s docstring
  stated that an interrupted turn reports `terminated` "in the closed taxonomy".
  There is no `terminated` anywhere in agentkit — it was never produced and
  never consumed. A reader following that docstring would have written a branch
  that can never match, and the only thing that would have told them otherwise
  is a test that refuses a member nothing emits.

So this file checks BOTH directions. One direction alone is half a guard: the
first catches a forgotten reason, the second catches an invented one, and the
invented one is the failure that reads as documentation.
"""

from __future__ import annotations

import pathlib
from typing import get_args

from agentkit.core.contracts.types import StreamEventType
from agentkit.models.contracts.constants import CLI_EXIT_PREFIX, StopReason, is_cli_exit
from agentkit.models.providers.cli.claude import driver as claude_driver
from agentkit.models.providers.cli.codex import driver as codex_driver

# Every reason set a driver classifies against, by the name it has in source so
# a failure names the set to go and look at.
PACKAGE = pathlib.Path(__file__).resolve().parents[3] / "agentkit"

DRIVER_REASON_SETS = {
    "claude._CLI_FAILURE_REASONS": claude_driver._CLI_FAILURE_REASONS,
    "claude._CLI_INVALID_OUTPUT_REASONS": claude_driver._CLI_INVALID_OUTPUT_REASONS,
    "codex._CODEX_FAILURE_REASONS": codex_driver._CODEX_FAILURE_REASONS,
    "codex._CODEX_INVALID_OUTPUT_REASONS": codex_driver._CODEX_INVALID_OUTPUT_REASONS,
}

# Members that are terminal verdicts or deliberate pass-throughs rather than
# vendor reasons, so they correctly appear in no driver set. Spelled out, not
# derived, because the whole point is that adding a member here is a decision
# somebody has to make on purpose.
NOT_A_VENDOR_REASON = frozenset(
    {
        StopReason.COMPLETE,
        StopReason.FAILED,
        StopReason.INVALID_OUTPUT,
        StopReason.CANCELLED,
        StopReason.INTERRUPTED,
        StopReason.BUDGET_EXHAUSTED,
    }
)


def test_every_reason_a_driver_can_emit_is_in_the_vocabulary() -> None:
    """A reason outside the enum is one an operator sees and no caller can match.
    It is also how a typo hides: an unrecognised reason passes through the
    terminal mapper untouched, so nothing fails and the label is simply wrong."""
    known = {member.value for member in StopReason}
    unknown = {
        f"{name}: {reason}"
        for name, reasons in DRIVER_REASON_SETS.items()
        for reason in reasons
        if reason not in known and not is_cli_exit(reason)
    }
    assert not unknown, (
        "these reasons are classified by a driver but are not StopReason members, "
        "so they reach `evals['stop_reason']` as values nothing can branch on:\n  "
        + "\n  ".join(sorted(unknown))
    )


def test_every_vocabulary_member_is_actually_reachable() -> None:
    """The direction that catches fiction.

    A member nothing emits is worse than a missing one: it reads as documentation,
    so somebody writes a branch for it and the branch never fires. `terminated`
    was exactly this — asserted by a docstring, produced by nothing.
    """
    emitted = {reason for reasons in DRIVER_REASON_SETS.values() for reason in reasons}
    orphans = sorted(
        member.value for member in StopReason if member not in NOT_A_VENDOR_REASON and member.value not in emitted
    )
    assert not orphans, (
        "these StopReason members are emitted by no driver and are not listed as "
        "terminal verdicts or pass-throughs, so nothing can ever produce them:\n  "
        + "\n  ".join(orphans)
    )


def test_the_one_open_point_in_the_vocabulary_stays_recognisable() -> None:
    """`cli_exit_<n>` carries the exit code in its name, so it cannot be a member.
    That makes it the single hole in a closed set, and the only thing keeping the
    hole from swallowing typos is that one predicate recognises it — three modules
    build or match this prefix and a fourth documented it with a typo."""
    assert is_cli_exit(f"{CLI_EXIT_PREFIX}137")
    assert is_cli_exit(f"{CLI_EXIT_PREFIX}0")
    assert not is_cli_exit(CLI_EXIT_PREFIX.rstrip("_")), "the bare stem is not a reason"
    assert not is_cli_exit(StopReason.MALFORMED_OUTPUT)
    assert not is_cli_exit(None) and not is_cli_exit("")


def test_the_vocabulary_is_a_strenum_so_existing_string_comparisons_still_hold() -> None:
    """The reason introducing this enum was safe to do at all. `StrEnum` members
    ARE their string values, so every `== "complete"` in the codebase and in the
    tests keeps working and no call site had to change at once.

    The hazard is `is`, not `==` — identity against a `StrEnum` member is False
    for an equal string, which is precisely the bug found in `resilience.py`
    today (`cls is ErrorClass.PERMANENT` read a caller's own `"permanent"` as
    not-permanent and retried a 401 three times). Audited at introduction: zero
    identity comparisons against string literals exist in `agentkit/`.
    """
    assert StopReason.COMPLETE == "complete"
    assert "malformed_output" == StopReason.MALFORMED_OUTPUT
    assert StopReason.FAILED in {"failed", "other"}
    assert f"{StopReason.COMPLETE}" == "complete", "interpolation must not leak the member repr"


# ─────────────────────────────────────────────────────────────────────────────
# The same two directions, applied to the other closed vocabulary in the stack.
# ─────────────────────────────────────────────────────────────────────────────


def _emitted_stream_event_types() -> set[str]:
    """Every literal a `StreamEvent(...)` is actually constructed with."""
    import re

    emitted: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        for match in re.finditer(r"""StreamEvent\(\s*["']([a-z_]+)["']""", path.read_text()):
            emitted.add(match.group(1))
    return emitted


def test_every_stream_event_type_emitted_is_declared() -> None:
    """`StreamEventType` is a `Literal` precisely so a consumer can exhaust the
    branch table with mypy. A producer emitting a tag outside it breaks that
    promise quietly: the branch table still type-checks as exhaustive while a
    real event falls through every arm."""
    declared = set(get_args(StreamEventType))
    undeclared = sorted(_emitted_stream_event_types() - declared)
    assert not undeclared, (
        "these tags are constructed but are not in the StreamEventType Literal, so an "
        f"exhaustive consumer silently drops them: {undeclared}"
    )


def test_every_declared_stream_event_type_has_a_producer() -> None:
    """The fiction direction, and it caught one: `"interrupt"` sat in this closed
    set and was emitted by nothing. A consumer could write that branch, mypy
    would accept it, and it could never fire.

    This is the same failure shape as the `terminated` stop reason above. Two
    closed vocabularies, two members that existed only as documentation — which
    is the argument for checking closure mechanically rather than declaring it
    in a comment.
    """
    declared = set(get_args(StreamEventType))
    orphans = sorted(declared - _emitted_stream_event_types())
    assert not orphans, (
        "these StreamEventType members are declared but emitted by nothing, so a "
        f"consumer branch for them can never fire: {orphans}"
    )
