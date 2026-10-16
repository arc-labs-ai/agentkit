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


class FailureClass(StrEnum):
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
