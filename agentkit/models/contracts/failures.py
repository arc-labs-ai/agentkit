"""Failure classification from structured transport facts, never prose."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from agentkit.models.contracts.constants import FailureClass, Outcome
from agentkit.models.contracts.types import Failure


def retry_after(value: str | None) -> float | None:
    """Parse both Retry-After formats."""
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except (ValueError, TypeError):
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError):
            return None


# ── conditions a provider reports ONLY in prose ─────────────────────────────
#
# Some account-level conditions arrive with no machine-readable `code` at all,
# stated only in the human-readable `message`. Measured against Anthropic, a
# reached spend cap returns:
#
#   status 400, type "invalid_request_error", code None, message
#   "You have reached your specified API usage limits. You will regain access
#    on 2026-11-01 at 00:00 UTC."
#
# Without reading that message the 400 fell through to INVALID_REQUEST — a
# client bug — so an operator went looking for a malformed request when the
# fix was billing. The information was never missing; it was in a field the
# classifier did not inspect.
#
# These are PATTERNS against a lowercased message, not exact substrings,
# because the wording belongs to the provider and changes without notice. They
# are matched only AFTER the structured `code` checks, so a provider that does
# send a code always wins, and they are deliberately specific: a bare "quota"
# or "billing" would misfile a genuine invalid request whose message happens
# to mention either.
_QUOTA_MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"usage\s+limits?",
        r"spend\s+limits?",
        r"insufficient\s+quota",
        # Up to two intervening words, so "quota was exceeded" and "quota has
        # now been reached" match while "the `quota` parameter is invalid"
        # still does not.
        r"quota\s+(?:\w+\s+){0,2}(?:exceeded|exhausted|reached)",
        r"(?:exceeded|exhausted)\s+(?:your\s+)?(?:current\s+)?quota",
        r"credit\s+balance\s+is\s+too\s+low",
        r"billing\s+(?:not\s+active|hard\s+limit)",
    )
)


def classify(exc_or_event: Any) -> Failure | None:
    if isinstance(exc_or_event, Failure):
        return exc_or_event
    if isinstance(exc_or_event, Mapping):
        facts = dict(exc_or_event)
    else:
        facts = {"status": getattr(exc_or_event, "status_code", None)}
        body = getattr(exc_or_event, "body", None)
        if body:
            try:
                decoded = json.loads(body)
                if isinstance(decoded, Mapping):
                    facts.update(decoded)
            except (ValueError, TypeError):
                pass
        facts["headers"] = getattr(exc_or_event, "headers", {})
        if isinstance(exc_or_event, TimeoutError):
            return Failure(FailureClass.TIMEOUT, True)
        if isinstance(exc_or_event, (ConnectionError, OSError)):
            return Failure(FailureClass.TRANSIENT_NETWORK, True)
        # Optional transport classes expose no HTTP status on network errors.
        if type(exc_or_event).__name__ in ("ConnectError", "ReadError", "RemoteProtocolError"):
            return Failure(FailureClass.TRANSIENT_NETWORK, True)
        if type(exc_or_event).__name__ in ("ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"):
            return Failure(FailureClass.TIMEOUT, True)
    error = facts.get("error") or facts
    if not isinstance(error, Mapping):
        error = {}
    details = error.get("details")
    code = error.get("code") or (details.get("error_code") if isinstance(details, Mapping) else None)
    kind = error.get("type") or facts.get("subtype")
    status = facts.get("status", facts.get("status_code"))
    code = code if isinstance(code, str) else None
    kind = kind if isinstance(kind, str) else None
    status = status if isinstance(status, int) else None
    raw_message = error.get("message")
    message = raw_message.lower() if isinstance(raw_message, str) else ""
    category: FailureClass = FailureClass.UNKNOWN
    if code in {
        "insufficient_quota",
        "billing_not_active",
        "enforced_spend_limit_reached",
        "usage_limit_reached",
        "spend_limit_reached",
    } or any(pattern.search(message) for pattern in _QUOTA_MESSAGE_PATTERNS):
        # Ordered ahead of the 400 branch deliberately: a reached spend cap is
        # an account condition, not a malformed request, and filing it as the
        # latter sends the reader to debug a request that was perfectly valid.
        category = FailureClass.QUOTA_EXHAUSTED
    elif code == "context_length_exceeded":
        category = FailureClass.CONTEXT_EXCEEDED
    elif status in (401, 403) or kind == "authentication_error" or code == "invalid_api_key":
        category = FailureClass.AUTH
    elif status == 429 or kind == "rate_limit_error" or code == "slow_down":
        category = FailureClass.RATE_LIMITED
    elif status in (503, 529) or kind == "overloaded_error" or code == "server_is_overloaded":
        category = FailureClass.OVERLOADED
    elif status in (408, 504):
        category = FailureClass.TIMEOUT
    elif status in (400, 402, 404, 413, 422) or kind == "invalid_request_error":
        # A structured client status outranks a generic provider type such as
        # ``api_error``. In particular, HTTP 400 must never become a retryable
        # transient merely because the CLI omitted a more specific subtype.
        category = FailureClass.INVALID_REQUEST
    elif status in (500, 502) or kind in ("api_error", "server_error"):
        category = FailureClass.TRANSIENT_NETWORK
    elif kind == "refusal":
        category = FailureClass.REFUSED
    elif kind == Outcome.CANCELLED:
        category = FailureClass.CANCELLED
    elif facts.get("type") == "error" and not error:
        return None
    headers = facts.get("headers") or {}
    if not isinstance(headers, Mapping):
        headers = {}
    wait = retry_after(headers.get("retry-after") or headers.get("Retry-After"))
    # Never capture exception messages, response bodies or credentials by default.
    return Failure(
        category,
        category
        in (FailureClass.RATE_LIMITED, FailureClass.OVERLOADED, FailureClass.TIMEOUT, FailureClass.TRANSIENT_NETWORK),
        wait,
        {"status": status, "type": kind, "code": code},
    )
