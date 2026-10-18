"""Closed domain vocabularies shared by callers, adapters and event consumers."""

from enum import StrEnum

from agentkit.core.contracts.types import CostSource as CostSource
from agentkit.models.providers.auth import AuthKind


class NativeToolPolicy(StrEnum):
    DENY = "deny"
    WARN = "warn"
    ALLOW = "allow"


class CacheTtl(StrEnum):
    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"
    THIRTY_MINUTES = "30m"


class UnpricedPolicy(StrEnum):
    REFUSE = "refuse"
    PROCEED_UNPRICED = "proceed_unpriced"


class ClaudePermissionMode(StrEnum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS_PERMISSIONS = "bypassPermissions"
    PLAN = "plan"
    AUTO = "auto"
    MANUAL = "manual"
    DONT_ASK = "dontAsk"


class CodexSandboxMode(StrEnum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


class ModelSelection(StrEnum):
    INHERIT = "inherit"


class Outcome(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"
    TRUNCATED = "truncated"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LIMIT_REACHED = "limit_reached"


class StopReason(StrEnum):
    """Why a provider turn stopped — the SPECIFIC reason, not the closed outcome.

    Two vocabularies live side by side and conflating them has already cost two
    defects, so the split is worth stating. ``Answer.stop_reason`` /
    ``CliResult.stop_reason`` carry the CLOSED terminal set a caller branches on
    (``complete`` / ``failed`` / ``invalid_output``, plus the deliberate
    pass-throughs). ``evals["stop_reason"]`` carries the member below that says
    WHICH failure it was, which is what an operator reads.

    This existed only as ``AgentStopReason = str``. The cost of that was not
    hypothetical:

    * a docstring described an interrupted turn as reporting ``terminated``
      "in the closed taxonomy"; there is no such value anywhere in agentkit, and
      nothing could have told the author otherwise;
    * ``interrupted`` is in neither driver's failure set, so it passes through
      the mapper unmapped — correct, as it happens, but indistinguishable from
      a reason that was forgotten.

    ``cli_exit_<n>`` is deliberately NOT a member: the exit code is part of the
    name, so the vocabulary is open at that one point. ``is_cli_exit`` recognises
    it, and ``test_stop_reason_vocabulary.py`` asserts every reason the drivers
    can emit is either a member here or that one dynamic form.
    """

    # ── terminal, closed set a caller branches on ───────────────────────────
    COMPLETE = "complete"
    FAILED = "failed"
    INVALID_OUTPUT = "invalid_output"

    # ── deliberate pass-throughs: somebody stopped it, nothing failed ───────
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    BUDGET_EXHAUSTED = "budget_exhausted"

    # ── the run never started, or never reached the provider ────────────────
    SPAWN_FAILED = "spawn_failed"
    WORKING_DIR_MISSING = "working_dir_missing"
    CLI_VERSION_UNSUPPORTED = "cli_version_unsupported"
    AUTHENTICATION_FAILED = "authentication_failed"

    # ── a bound was crossed (see CliTimeouts) ───────────────────────────────
    STARTUP_TIMEOUT = "startup_timeout"
    FIRST_EVENT_TIMEOUT = "first_event_timeout"
    IDLE_TIMEOUT = "idle_timeout"
    TOTAL_TIMEOUT = "total_timeout"

    # ── the stream or the process broke ─────────────────────────────────────
    MALFORMED_OUTPUT = "malformed_output"
    PARSE_FAILED = "parse_failed"
    OUTPUT_LINE_TOO_LONG = "output_line_too_long"
    PROCESS_CRASHED = "process_crashed"

    # ── the CLI said so itself ──────────────────────────────────────────────
    #
    # The first three are the Claude CLI's own ``result.subtype`` values when it
    # exits 0 with ``is_error: true``. They were lifted straight off the payload
    # and passed through the terminal mapper UNRECOGNISED, so a caller branching
    # on ``stop_reason == "failed"`` never matched a run the CLI had itself
    # flagged as an error — escalation, retry policy and dashboard filters all
    # went quiet at once.
    #
    # ``ERROR_MAX_TURNS`` is the debatable one and is filed as a failure on
    # purpose: the CLI sets ``is_error`` for it, and the alternative is a third
    # terminal bucket this mapper does not have. The specific name survives in
    # ``evals["stop_reason"]`` either way, so nothing is lost by bucketing it
    # here — but if a ``limit_reached`` terminal is ever added, this is the
    # member to move.
    ERROR_MAX_TURNS = "error_max_turns"
    ERROR_DURING_EXECUTION = "error_during_execution"
    API_ERROR = "api_error"
    CLI_REPORTED_ERROR = "cli_reported_error"
    TURN_FAILED = "turn_failed"
    SCHEMA_REJECTED = "schema_rejected"

    # ── session-scoped: the conversation, or just this turn ─────────────────
    SESSION_CLOSED = "session_closed"
    TURN_REFUSED = "turn_refused"

    # ── structured output ───────────────────────────────────────────────────
    STRUCTURED_OUTPUT_MISSING = "structured_output_missing"
    STRUCTURED_OUTPUT_MISMATCH = "structured_output_mismatch"
    ERROR_MAX_STRUCTURED_OUTPUT_RETRIES = "error_max_structured_output_retries"


CLI_EXIT_PREFIX = "cli_exit_"
"""The one OPEN point in :class:`StopReason`: ``cli_exit_<n>`` carries the exit
code in its name, so it cannot be an enum member. Kept as a constant because
three modules build or match it and a fourth documented it with a typo."""


def is_cli_exit(reason: str | None) -> bool:
    """Whether ``reason`` is the dynamic ``cli_exit_<n>`` form."""
    return bool(reason) and str(reason).startswith(CLI_EXIT_PREFIX)


class FailureClass(StrEnum):
    EXECUTION_INCOMPATIBLE = "execution_incompatible"
    EXECUTION_UNCERTAIN = "execution_uncertain"
    TRANSPORT = "transport"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    TRANSIENT_NETWORK = "transient_network"
    TIMEOUT = "timeout"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    REFUSED = "refused"
    CONTEXT_EXCEEDED = "context_exceeded"
    TOOL_ERROR = "tool_error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class EventKind(StrEnum):
    CALL_STARTED = "call_started"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_DONE = "message_done"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USAGE = "usage"
    ATTEMPT_STARTED = "attempt_started"
    ATTEMPT_FINISHED = "attempt_finished"
    RETRY = "retry"
    DEGRADED = "degraded"
    CALL_FINISHED = "call_finished"
    CACHE_INVALIDATED = "cache_invalidated"
    SKILL_LOADED = "skill_loaded"
    UNKNOWN_MODE_EVENT = "unknown_mode_event"
    TRANSPORT = "transport"


class Effort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"
    ULTRA = "ultra"


class ReasoningSummary(StrEnum):
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"
    NONE = "none"


class Verbosity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ServiceTier(StrEnum):
    AUTO = "auto"
    DEFAULT = "default"
    FLEX = "flex"
    PRIORITY = "priority"


class MindKind(StrEnum):
    SINGLE_CALL = "single_call"
    REACT = "react"
    CHAIN_OF_THOUGHT = "chain_of_thought"


CALL_EVENT_SCHEMA_VERSION = "1"

__all__ = [
    "CALL_EVENT_SCHEMA_VERSION",
    "AuthKind",
    "ModelSelection",
    "NativeToolPolicy",
    "CacheTtl",
    "UnpricedPolicy",
    "ClaudePermissionMode",
    "CodexSandboxMode",
    "CostSource",
    "Effort",
    "ReasoningSummary",
    "Verbosity",
    "ServiceTier",
    "EventKind",
    "FailureClass",
    "MindKind",
    "Outcome",
]
