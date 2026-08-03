"""Resilience chokepoint: classify → backoff(jitter) → circuit breaker → idempotency.

`run_with_resilience` is async-first (the `retry` middleware's engine); the pure helpers
(classify/backoff/CircuitBreaker/idempotency_key) are sync and have no I/O. Clock/sleep/rng are
injectable so tests are deterministic and fast.
"""

from __future__ import annotations

import asyncio
import dataclasses as _dc
import datetime as _dt
import decimal as _decimal
import enum
import hashlib
import json
import pathlib as _pl
import random
import time
import types as _types
import uuid as _uuid
from collections.abc import Callable
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass
from typing import Any, Literal

# The breaker's state machine has three states; typing as ``Literal``
# catches a typo (``"half-open"`` with a hyphen) at assignment time
# instead of leaving the breaker stuck in a state where every branch
# comparison is False.
BreakerState = Literal["closed", "open", "half_open"]


class ErrorClass(str, enum.Enum):
    TRANSIENT = "transient"  # retry
    PERMANENT = "permanent"  # fail fast
    UNKNOWN = "unknown"  # conservative


_TRANSIENT = (
    "rate limit",
    "429",
    "timeout",
    "timed out",
    "503",
    "502",
    "504",
    "temporarily",
    "connection reset",
    "connection aborted",
    "circuit open",
    "overloaded",
)
_PERMANENT = (
    "400",
    "401",
    "403",
    "404",
    "422",
    "invalid",
    "unauthorized",
    "forbidden",
    "unsafeurl",
    "not allowed",
    "content filter",
    "permission denied",
    "validation",
)


def classify(exc: BaseException) -> ErrorClass:
    """Substring-classify an exception as TRANSIENT / PERMANENT / UNKNOWN.

    TRANSIENT wins on collision. Alternative orderings that checked PERMANENT
    first, so ``ValidationError("request timed out")`` matched ``"validation"``
    in the PERMANENT list and never got retried — exactly the failure mode
    the resilience layer exists to handle. Two reasons TRANSIENT must win:

    1. TRANSIENT signals (``timeout``, ``5xx``, ``rate limit``) are inherently
       more specific than generic PERMANENT substrings (``"invalid"``,
       ``"validation"``) that frequently appear in transient infra errors.
    2. The conservative choice on collision is to retry — wasting one retry
       on a permanent error costs the backoff delay; failing fast on a
       transient error costs the whole call.

    Substring matching is brittle by nature; for stronger guarantees,
    callers should pass a custom ``classify_fn`` to ``run_with_resilience``."""
    msg = f"{type(exc).__name__} {exc}".lower()
    if any(t in msg for t in _TRANSIENT):
        return ErrorClass.TRANSIENT
    if any(t in msg for t in _PERMANENT):
        return ErrorClass.PERMANENT
    return ErrorClass.UNKNOWN


def backoff_delay(
    attempt: int, *, base: float = 0.5, cap: float = 30.0, rng: Any = random
) -> float:
    """Full-jitter exponential backoff (prevents retry storms across workers)."""
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    delay: float = rng.uniform(0, ceiling)
    return delay


class CircuitOpen(RuntimeError):
    """Raised when a per-dependency breaker is OPEN (classified TRANSIENT → caller may degrade)."""


@dataclass
class CircuitBreaker:
    """Per-dependency breaker. CLOSED → (fail_threshold consecutive fails) → OPEN →
    (cooldown) → HALF_OPEN → one probe → CLOSED on success / OPEN on failure."""

    name: str
    fail_threshold: int = 5
    cooldown: float = 15.0
    clock: Callable[[], float] = time.monotonic
    state: BreakerState = "closed"
    _fails: int = 0
    _opened_at: float = 0.0

    def allow(self) -> bool:
        # No await inside → atomic within one event loop. The `half_open` state is itself the single-probe
        # gate: the caller that flips open→half_open is the one probe; every other caller sees `half_open`
        # and is refused until that probe resolves (success→closed / failure→open). (For cross-thread
        # sharing, serialize the breaker externally.)
        if self.state == "closed":
            return True
        if self.state == "half_open":
            return False  # a probe is already in flight → admit no others
        if (
            self.clock() - self._opened_at >= self.cooldown
        ):  # open + cooled down → admit exactly ONE probe
            self.state = "half_open"
            return True
        return False  # open, still cooling down

    def record_success(self) -> None:
        self._fails = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self._fails += 1
        if self.state == "half_open" or self._fails >= self.fail_threshold:
            self.state = "open"
            self._opened_at = self.clock()


def _stable_default(o: Any) -> Any:
    """Deterministic JSON fallback. A naive ``default=str`` would emit
    ``<Foo object at 0x...>`` for plain classes — non-deterministic
    hashes across processes / runs, breaking memoize / idempotency /
    audit fingerprints.

    The encoder handles the wire-relevant types explicitly. Anything that
    doesn't match falls back to ``{__type__, sorted-vars}`` (or just the
    type name) — still deterministic, no memory addresses."""
    # Datetime-like — ISO 8601.
    if isinstance(o, _dt.datetime | _dt.date | _dt.time):
        return o.isoformat()
    if isinstance(o, _dt.timedelta):
        return o.total_seconds()
    # Identifier-like — their str() is canonical.
    if isinstance(o, _uuid.UUID):
        return str(o)
    if isinstance(o, _pl.PurePath):
        return str(o)
    if isinstance(o, _decimal.Decimal):
        return str(o)
    # Bytes — stable hex (str() prefixes b'...' which is fine but hex is
    # shorter and excludes the quoting/repr layer).
    if isinstance(o, bytes | bytearray):
        return o.hex()
    # Set / frozenset — sort for stable iteration. The elements still go
    # through ``default`` recursively, so nested non-primitives are fine.
    if isinstance(o, set | frozenset):
        try:
            return sorted(o)
        except TypeError:
            # Heterogeneous unsortable contents — fall through to a stable
            # repr via list(_) + sort-by-_stable_repr.
            return sorted(o, key=lambda x: _stable_repr(x))
    # Mappings that ``json`` can't natively encode (``MappingProxyType``
    # from ``ToolCall.arguments`` is the load-bearing case — the type
    # is used as a read-only view over the caller-supplied dict).
    if isinstance(o, _types.MappingProxyType) or (
        isinstance(o, _Mapping) and not isinstance(o, dict)
    ):
        return dict(o)
    # Enums — value (already JSON-friendly if str/int).
    if isinstance(o, enum.Enum):
        return {"__enum__": type(o).__qualname__, "value": o.value}
    # Pydantic v2: prefer ``model_dump(mode="json")`` so nested datetimes /
    # UUIDs serialize the same way they would on the wire.
    model_dump = getattr(o, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    # Dataclass instance — walk fields manually rather than
    # ``dataclasses.asdict``, which deep-copies every value via
    # ``copy.deepcopy`` under the hood. Deep-copy fails on
    # ``MappingProxyType`` (the immutable view over ``ToolCall.arguments``)
    # because stdlib has no pickle protocol for the type. The manual
    # walk hands each field back to ``_stable_default`` on the next
    # recursion, which routes ``MappingProxyType`` through the mapping
    # branch above.
    #
    # Only ``init=True`` fields feed the hash. ``init=False`` fields
    # hold derived / cached scaffolding (adapter caches, TTL cursors,
    # session tokens); folding them in would pollute
    # idempotency / memoize / audit fingerprints with non-deterministic
    # state that has nothing to do with the value's identity.
    if _dc.is_dataclass(o) and not isinstance(o, type):
        return {f.name: getattr(o, f.name) for f in _dc.fields(o) if f.init}
    # Generic fallback — type + sorted __dict__ (no memory address, no
    # str() of opaque objects). Still deterministic for any plain class
    # that holds its state in __dict__.
    if hasattr(o, "__dict__") and o.__dict__:
        return {"__type__": type(o).__qualname__, **{k: o.__dict__[k] for k in sorted(o.__dict__)}}
    # Last resort: bare type name. Two distinct opaque objects of the same
    # type collide here — which is still better than two identical-looking
    # objects producing different hashes.
    return f"<{type(o).__qualname__}>"


def _stable_repr(o: Any) -> str:
    """Stable string key used to sort heterogeneous unsortable collections.
    Routes through ``_stable_default`` for type-name fallback so memory
    addresses can't leak into the sort key either."""
    try:
        return json.dumps(o, sort_keys=True, default=_stable_default)
    except (TypeError, ValueError):
        return f"<{type(o).__qualname__}>"


def stable_hash(obj: Any, *, length: int = 16) -> str:
    """Deterministic short hash of any JSON-ish value — the shared basis for
    cache keys and audit fingerprints, so they can't drift apart.

    Handles datetime / UUID / Path / Decimal / bytes / set / frozenset /
    Enum / Pydantic / dataclass / plain-class explicitly. The fallback is
    type-name + sorted-vars rather than ``str(o)`` so two semantically
    identical instances always hash to the same value across processes,
    avoiding silent cache-key drift."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=_stable_default).encode()
    ).hexdigest()[:length]


def idempotency_key(*parts: Any) -> str:
    return "idem:" + stable_hash(parts, length=24)


async def run_with_resilience(
    fn: Callable[[], Any],
    *,
    breaker: CircuitBreaker | None = None,
    max_attempts: int = 3,
    classify_fn: Callable[[BaseException], ErrorClass] = classify,
    sleep: Callable[[float], Any] | None = None,
    rng: Any = random,
) -> Any:
    """Run async `fn` with classification + jittered retry + optional circuit breaker (async-first;
    the one resilience entry point). `await fn()`; backoff via `await asyncio.sleep` (injectable
    async `sleep` for deterministic tests).

    PERMANENT errors fail fast; TRANSIENT/UNKNOWN retry up to max_attempts; an OPEN breaker
    raises CircuitOpen (itself TRANSIENT — the caller decides degrade vs propagate).
    """
    asleep = sleep or asyncio.sleep
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        if breaker is not None and not breaker.allow():
            # Preserve the original transient cause via ``from last`` so a
            # postmortem can reach the exception that opened the breaker —
            # a bare ``CircuitOpen`` would discard exactly the failure
            # class the breaker exists to surface.
            raise CircuitOpen(f"circuit open: {breaker.name}") from last
        try:
            result = await fn()
            if breaker is not None:
                breaker.record_success()
            return result
        except CircuitOpen:
            raise
        except Exception as exc:  # noqa: BLE001 — classify decides
            last = exc
            cls = classify_fn(exc)
            # Only TRANSIENT / UNKNOWN errors count toward the breaker's
            # fail threshold. A 401 / 403 / content-filter rejection is a
            # contract failure — not an upstream health signal — so folding
            # it into the counter would open the breaker for reasons
            # unrelated to dependency health.
            if breaker is not None and cls is not ErrorClass.PERMANENT:
                breaker.record_failure()
            if cls is ErrorClass.PERMANENT or attempt == max_attempts:
                raise
            await asleep(backoff_delay(attempt, rng=rng))
    raise (
        last if last is not None else RuntimeError("run_with_resilience: no attempt ran")
    )  # pragma: no cover
