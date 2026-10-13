"""Failure classification parity through shared process and vendor reason sets."""

from __future__ import annotations

import pytest

from agentkit.models.providers.cli._common import (
    _SCHEMA_REJECTED_STDERR,
    _cli_terminal_stop_reason,
    _process_exit_stop_reason,
)
from agentkit.models.providers.cli.claude import driver as claude
from agentkit.models.providers.cli.codex import driver as codex


@pytest.mark.parametrize(
    "return_code,stderr,expected",
    [
        (-9, b"", "process_crashed"),
        (-11, b"invalid json schema", "process_crashed"),
        (2, b"INVALID JSON SCHEMA", "schema_rejected"),
        (2, b"\xffoutput schema is invalid\xfe", "schema_rejected"),
        (7, b"network failed", "cli_exit_7"),
        (1, b"", "cli_exit_1"),
    ],
)
def test_process_signal_schema_and_ordinary_failure_keep_identical_precedence(return_code, stderr, expected):
    assert _process_exit_stop_reason(return_code, stderr) == expected


@pytest.mark.parametrize("marker", _SCHEMA_REJECTED_STDERR)
def test_both_cli_wrappers_share_the_complete_schema_failure_vocabulary(marker):
    message = ("Startup diagnostic: " + marker.upper()).encode()
    assert _process_exit_stop_reason(2, message) == "schema_rejected"


def test_process_classifier_accepts_vendor_extensions_without_global_state_changes():
    assert (
        _process_exit_stop_reason(2, b"vendor-specific schema refusal", ("vendor-specific schema",))
        == "schema_rejected"
    )
    assert _process_exit_stop_reason(2, b"vendor-specific schema refusal") == "cli_exit_2"


@pytest.mark.parametrize(
    "invalid,failure",
    [
        (claude._CLI_INVALID_OUTPUT_REASONS, claude._CLI_FAILURE_REASONS),
        (codex._CODEX_INVALID_OUTPUT_REASONS, codex._CODEX_FAILURE_REASONS),
    ],
    ids=["claude", "codex"],
)
@pytest.mark.parametrize(
    "reason,expected",
    [
        (None, "complete"),
        ("success", "complete"),
        ("structured_output_missing", "invalid_output"),
        ("structured_output_mismatch", "invalid_output"),
        ("process_crashed", "failed"),
        ("schema_rejected", "failed"),
        ("cli_exit_127", "failed"),
        ("cancelled", "terminated"),
        ("budget_exhausted", "budget_exhausted"),
        ("awaiting_approval", "suspended"),
        ("new_vendor_reason", "terminated"),
    ],
)
def test_terminal_mapping_preserves_recovery_and_cancellation_distinctions(invalid, failure, reason, expected):
    assert _cli_terminal_stop_reason(reason, invalid_output_reasons=invalid, failure_reasons=failure) == expected


def test_invalid_output_wins_when_a_vendor_reason_is_in_both_sets():
    assert (
        _cli_terminal_stop_reason("schema", invalid_output_reasons={"schema"}, failure_reasons={"schema"})
        == "invalid_output"
    )


def test_reason_sets_are_call_scoped_and_do_not_leak_between_backends():
    assert (
        _cli_terminal_stop_reason("vendor_error", invalid_output_reasons=(), failure_reasons={"vendor_error"})
        == "failed"
    )
    assert _cli_terminal_stop_reason("vendor_error", invalid_output_reasons=(), failure_reasons=()) == "terminated"
