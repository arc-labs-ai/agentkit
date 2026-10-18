"""The CLI reason sets, mapped onto the CLOSED agent stop-reason taxonomy.

WHY THIS FILE EXISTS
--------------------
``stop_reason`` is a BRANCH TARGET, not a label. A caller writes
``if result.stop_reason == "failed": escalate()`` — so a vendor reason that
reaches them un-mapped is not a cosmetic wart, it is a branch that never
matches and an escalation that never happens, on a run that already failed.

``_cli_terminal_stop_reason`` is the one place both CLI cognitions turn their
own private vocabularies into that closed set. Two of its three rules are
enumerations (``invalid_output_reasons``, ``failure_reasons``), which a reader
can audit by eye; the third is a PREFIX test on ``cli_exit_<n>``, and that one
cannot be enumerated because ``n`` is whatever the subprocess chose to exit
with. Drop it and nothing breaks loudly: ``cli_exit_7`` is a truthy string, it
travels all the way to the caller, and the run reads as a deliberate stop with
a funny name. That is the shape these tests exist to close — the
``_process_exit_stop_reason`` tests next door prove the reason is MINTED
correctly and say nothing about whether it is then CLASSIFIED.

Both cognitions' reason sets are driven through the same assertions on purpose:
the mapping is shared code, and the only input either one contributes is its
own frozenset, so a rule that holds for one and not the other is a bug in the
set rather than in the mapping.
"""

from __future__ import annotations

import pytest

from agentkit.models.providers.cli._common import (
    _cli_terminal_stop_reason,
    _process_exit_stop_reason,
    stop_reason_for,
)
from agentkit.models.providers.cli.claude.driver import (
    _CLI_FAILURE_REASONS,
    _CLI_INVALID_OUTPUT_REASONS,
)
from agentkit.models.providers.cli.codex.driver import (
    _CODEX_FAILURE_REASONS,
    _CODEX_INVALID_OUTPUT_REASONS,
)

# The two shipped vocabularies, as ``(name, invalid_output, failure)``. Named so
# a failure report says WHICH cognition's set broke the rule.
_VOCABULARIES = (
    ("claude", _CLI_INVALID_OUTPUT_REASONS, _CLI_FAILURE_REASONS),
    ("codex", _CODEX_INVALID_OUTPUT_REASONS, _CODEX_FAILURE_REASONS),
)


def _classify(reason: str | None, vocabulary: tuple[str, frozenset[str], frozenset[str]]) -> str:
    _, invalid_output, failure = vocabulary
    return _cli_terminal_stop_reason(
        reason, invalid_output_reasons=invalid_output, failure_reasons=failure
    )


@pytest.mark.parametrize("vocabulary", _VOCABULARIES, ids=lambda v: v[0])
@pytest.mark.parametrize(
    ("return_code", "why"),
    [
        (1, "the ordinary 'it went wrong' exit"),
        (2, "a usage error from the binary's own argument parser"),
        (7, "an arbitrary application code, which is the point: n is unbounded"),
        (127, "command not found — the shell's own code, seen when a wrapper script is missing"),
    ],
)
def test_a_subprocess_that_exited_non_zero_is_a_failure_and_not_a_deliberate_stop(
    return_code: int, why: str, vocabulary: tuple[str, frozenset[str], frozenset[str]]
) -> None:
    """``cli_exit_<n>`` is MINTED with the exit code inside it, so it can never
    be a member of a frozenset — ``n`` is whatever the process chose. Classify
    it by enumeration alone and every non-zero exit escapes as its own reason:
    truthy, caller-visible, and read as a deliberate stop by anything branching
    on the closed set. A service that escalates on ``"failed"`` would silently
    stop escalating for the single commonest way a CLI run dies.

    Driven through the real minting helper rather than a hand-written string, so
    a change to the ``cli_exit_`` spelling cannot leave this test passing
    against a reason the drivers no longer produce.
    """
    reason = _process_exit_stop_reason(return_code, b"")

    assert reason == f"cli_exit_{return_code}", why
    assert _classify(reason, vocabulary) == "failed", why


@pytest.mark.parametrize("vocabulary", _VOCABULARIES, ids=lambda v: v[0])
def test_the_two_refinements_of_a_non_zero_exit_keep_the_same_verdict(
    vocabulary: tuple[str, frozenset[str], frozenset[str]],
) -> None:
    """``process_crashed`` and ``schema_rejected`` replace a ``cli_exit_<n>``
    with a more specific name, and both must land on the same verdict the code
    they replaced would have. A refinement that changed the CATEGORY as well as
    the name would mean a caller's branch depended on whether stderr happened
    to mention a schema — the diagnosis would move the escalation."""
    for reason in ("process_crashed", "schema_rejected"):
        assert _classify(reason, vocabulary) == "failed", reason


@pytest.mark.parametrize("vocabulary", _VOCABULARIES, ids=lambda v: v[0])
def test_every_enumerated_failure_reason_reaches_the_caller_as_failed(
    vocabulary: tuple[str, frozenset[str], frozenset[str]],
) -> None:
    """The enumerations are the auditable half of the mapping, and an entry
    that is in the set but does not map is unreachable protection: the comment
    beside it says ``failed`` and the caller sees something else. Asserted over
    the WHOLE frozenset rather than a sample, because the set is edited by hand
    every time a cognition learns a new way to fail."""
    name, _, failure = vocabulary
    assert failure, f"{name} declares no failure reasons at all"
    for reason in sorted(failure):
        assert _classify(reason, vocabulary) == "failed", f"{name}: {reason}"


@pytest.mark.parametrize("vocabulary", _VOCABULARIES, ids=lambda v: v[0])
def test_a_shape_failure_is_invalid_output_and_not_a_failed_run(
    vocabulary: tuple[str, frozenset[str], frozenset[str]],
) -> None:
    """The run worked and the SHAPE did not, which is a different repair: re-ask
    with a repaired schema rather than retry the run or page somebody. Folding
    these into ``failed`` would lose the only distinction that tells a caller
    the text in hand is real."""
    name, invalid_output, _ = vocabulary
    assert invalid_output, f"{name} declares no invalid-output reasons at all"
    for reason in sorted(invalid_output):
        assert _classify(reason, vocabulary) == "invalid_output", f"{name}: {reason}"


def test_invalid_output_outranks_failure_when_a_reason_is_in_both_sets() -> None:
    """Precedence has to be decided here rather than by set-membership order,
    because the two sets are edited independently and nothing stops a reason
    landing in both. ``invalid_output`` wins: it is the more specific
    diagnosis, and it carries the fact that the run itself completed."""
    both = frozenset({"structured_output_missing"})

    assert (
        _cli_terminal_stop_reason(
            "structured_output_missing", invalid_output_reasons=both, failure_reasons=both
        )
        == "invalid_output"
    )


def test_a_run_that_stopped_for_no_reason_at_all_is_complete() -> None:
    """A clean CLI run leaves ``state.stop_reason`` as ``None`` — there is no
    terminal subtype to read because nothing went wrong. ``None`` therefore has
    to mean success, not "unknown": a falsy reason propagated verbatim would
    reach ``AgentResult.stop_reason`` as ``None`` and make every consumer that
    compares against a string fall through its success branch."""
    assert stop_reason_for(None) == "complete"
    assert _classify(None, _VOCABULARIES[0]) == "complete"


@pytest.mark.parametrize(
    "reason",
    ["cancelled", "interrupted", "budget_exhausted"],
)
def test_a_lifecycle_reason_the_framework_already_owns_passes_through_unchanged(
    reason: str,
) -> None:
    """Three reasons are deliberately NOT in either vendor set, because they are
    not vendor facts: a cancelled run, an interrupted session turn and a budget
    refused before the spawn are framework lifecycle states that non-CLI agents
    report with the same words. Re-mapping them to ``failed`` would erase the
    distinction a caller needs most — a budget refusal is resumable and a
    failure is not — which is exactly the collapse
    ``agentkit/models/providers/cli/claude/driver.py`` documents for
    ``interrupted``."""
    assert _classify(reason, _VOCABULARIES[0]) == reason


@pytest.mark.parametrize("reason", ["error_max_turns", "error_during_execution", "api_error"])
def test_a_vendor_error_subtype_is_mapped_instead_of_leaking_through(reason: str) -> None:
    """These arrive on a run the CLI has flagged ``is_error``, so the run
    FAILED and the caller has to be able to see that with the same comparison
    every other failure answers. A reason nobody mapped is a reason nobody can
    branch on: the escalation path, the retry policy and the dashboard filter
    all key off the closed set, and all three go quiet at once."""
    assert _classify(reason, _VOCABULARIES[0]) == "failed"
