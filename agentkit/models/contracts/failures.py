"""Failure classification from structured transport facts, never prose."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from agentkit.models.contracts.constants import FailureClass, Outcome
from agentkit.models.contracts.execution import ExecutionCompatibility
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


# ── the providers' own error vocabularies ───────────────────────────────────
#
# Introspected from the official SDKs (anthropic 1.15.0, openai 3.33.0) rather
# than transcribed from prose, because that is the only list that is actually
# authoritative. agentkit speaks HTTP directly and has no SDK dependency, so
# nothing keeps this mapping in step with the providers except this table —
# which is why it lives in one place with the source named.
#
# Anthropic's canonical `type` values: api_error, authentication_error,
# billing_error, invalid_request_error, not_found_error, overloaded_error,
# permission_error, rate_limit_error, timeout_error.
#
# The TYPES matter independently of the statuses, and not only for tidiness: an
# in-band SSE error frame carries a `type` with NO status at all, so a
# condition recognised only by status is unclassifiable mid-stream.
_QUOTA_CODES = frozenset({
    "insufficient_quota",
    "billing_not_active",
    "enforced_spend_limit_reached",
    "usage_limit_reached",
    "spend_limit_reached",
})
# The same quota conditions arrive under `error.type` on some endpoints and
# `error.code` on others, so one rule has to cover both fields. Reading them
# only as codes is what let a 429 carrying `{"type": "insufficient_quota"}`
# and no `code` fall through to RATE_LIMITED — retryable — which is exactly
# the budget-burning retry that `_PERMANENT_429_TYPES` upstream exists to
# prevent. Two modules, one rule, and they disagreed.
_QUOTA_TYPES = frozenset({"billing_error", *_QUOTA_CODES})
# 402 Payment Required is a billing condition, not a malformed request. It used
# to sit in the INVALID_REQUEST set, which told an operator to go and debug a
# request that was perfectly well formed.
_QUOTA_STATUSES = frozenset({402})

_AUTH_TYPES = frozenset({"authentication_error", "permission_error"})
_AUTH_STATUSES = frozenset({401, 403})

_RATE_LIMIT_TYPES = frozenset({"rate_limit_error"})
_RATE_LIMIT_STATUSES = frozenset({429})

_OVERLOADED_TYPES = frozenset({"overloaded_error"})
_OVERLOADED_STATUSES = frozenset({503, 529})

_TIMEOUT_TYPES = frozenset({"timeout_error"})
_TIMEOUT_STATUSES = frozenset({408, 504})

_INVALID_REQUEST_TYPES = frozenset({"invalid_request_error", "not_found_error"})
# 409 is defined by BOTH SDKs (`ConflictError`) and was mapped nowhere, so a
# conflict arrived as UNKNOWN — indistinguishable from a failure agentkit has
# never seen. It is a 4xx about the caller's own request, so it belongs here
# and stays non-retryable: a conflict that resolves itself on an immediate
# retry is the exception, and guessing retryable on a write is the dangerous
# direction.
_INVALID_REQUEST_STATUSES = frozenset({400, 404, 409, 413, 422})

_TRANSIENT_TYPES = frozenset({"api_error", "server_error"})
_TRANSIENT_STATUSES = frozenset({500, 502})


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
    if (
        code in _QUOTA_CODES
        or kind in _QUOTA_TYPES
        or status in _QUOTA_STATUSES
        or any(pattern.search(message) for pattern in _QUOTA_MESSAGE_PATTERNS)
    ):
        # Ordered ahead of the 400 branch deliberately: a reached spend cap is
        # an account condition, not a malformed request, and filing it as the
        # latter sends the reader to debug a request that was perfectly valid.
        category = FailureClass.QUOTA_EXHAUSTED
    elif code == "context_length_exceeded":
        category = FailureClass.CONTEXT_EXCEEDED
    elif status in _AUTH_STATUSES or kind in _AUTH_TYPES or code == "invalid_api_key":
        category = FailureClass.AUTH
    elif status in _RATE_LIMIT_STATUSES or kind in _RATE_LIMIT_TYPES or code == "slow_down":
        category = FailureClass.RATE_LIMITED
    elif (
        status in _OVERLOADED_STATUSES
        or kind in _OVERLOADED_TYPES
        or code == "server_is_overloaded"
    ):
        category = FailureClass.OVERLOADED
    elif status in _TIMEOUT_STATUSES or kind in _TIMEOUT_TYPES:
        category = FailureClass.TIMEOUT
    elif status in _INVALID_REQUEST_STATUSES or kind in _INVALID_REQUEST_TYPES:
        # A structured client status outranks a generic provider type such as
        # ``api_error``. In particular, HTTP 400 must never become a retryable
        # transient merely because the CLI omitted a more specific subtype.
        category = FailureClass.INVALID_REQUEST
    elif status in _TRANSIENT_STATUSES or kind in _TRANSIENT_TYPES:
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
    if category is FailureClass.UNKNOWN and not isinstance(exc_or_event, Mapping):
        # Nothing on the exception itself said anything, so follow the chain.
        #
        # The transport branches above key on ``type(exc).__name__``, which only
        # works while the httpx error IS the exception. Every wrapper defeats
        # them: ``HttpLLM._post`` re-raises a plain ``ProviderError`` with no
        # status and no body, keeping the ``ConnectError`` on ``__cause__``
        # alone. Measured, before this: a refused connection through the API
        # runtime classified UNKNOWN / retryable=False, while the SAME error
        # handed here directly gave TRANSIENT_NETWORK / retryable=True. A real
        # connection failure was therefore never retried.
        #
        # Only ever consulted when this call produced UNKNOWN, so a wrapper that
        # deliberately reinterprets its cause keeps the last word, and the walk
        # is bounded because a long chain means the answer is not here.
        cause: BaseException | None = getattr(exc_or_event, "__cause__", None)
        for _ in range(4):
            if cause is None:
                break
            inherited = classify(cause)
            if inherited is not None and inherited.classification is not FailureClass.UNKNOWN:
                return inherited if wait is None else replace(inherited, retry_after_s=wait)
            cause = getattr(cause, "__cause__", None)
    # Never capture exception messages, response bodies or credentials by default.
    return Failure(
        category,
        category
        in (FailureClass.RATE_LIMITED, FailureClass.OVERLOADED, FailureClass.TIMEOUT, FailureClass.TRANSIENT_NETWORK),
        wait,
        {"status": status, "type": kind, "code": code},
    )


def execution_failure(compatibility: ExecutionCompatibility) -> Failure:
    """Safe public refusal, without a raw command, prompt, credential or CLI output.

    Lives beside :func:`classify` because it is the same kind of thing — the one
    sanctioned way to turn structured facts into a public ``Failure`` — and in
    ``contracts`` rather than either provider package because both families
    refuse in this shape and neither should import the other to do it. It
    consumes only contract types, so nothing here knows what a transport is.
    """
    return Failure(
        FailureClass.EXECUTION_INCOMPATIBLE,
        mode_detail={
            "reason": compatibility.incompatibilities[0].explanation,
            "dispatch_state": "not_started",
            "backend": compatibility.backend,
            "backend_version": compatibility.backend_version,
            "binary_version": compatibility.binary_version,
            "incompatibilities": tuple(asdict(item) for item in compatibility.incompatibilities),
        },
    )
