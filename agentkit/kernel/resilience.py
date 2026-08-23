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
import threading
import time
import types as _types
import uuid as _uuid
from collections.abc import Callable
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

# The breaker's state machine has three states; typing as ``Literal``
# catches a typo (``"half-open"`` with a hyphen) at assignment time
# instead of leaving the breaker stuck in a state where every branch
# comparison is False.
BreakerState = Literal["closed", "open", "half_open"]


class ErrorClass(enum.StrEnum):
    TRANSIENT = "transient"  # retry
    PERMANENT = "permanent"  # fail fast
    UNKNOWN = "unknown"  # conservative


_TRANSIENT = (
    "rate limit",
    # Providers name their error TYPES with underscores, and those names are
    # what now reaches this classifier from an in-band SSE error frame
    # (`raise_if_error_frame`). Without these, `rate_limit_error` and
    # `server_error` fell to UNKNOWN — still retried, since only PERMANENT
    # fails fast, but classified on nothing.
    "rate_limit",
    "server_error",
    "overloaded",
    # 529 is Anthropic's overload status. Bare "500" is deliberately ABSENT:
    # it is a substring of "5000", which appears in ordinary text like
    # "max_tokens 5000", and a false TRANSIENT there would retry a request
    # that can never succeed. The existing 502/503/504 entries carry the same
    # risk in principle and are kept for continuity; this is not a reason to
    # add more.
    "529",
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


def backoff_delay(attempt: int, *, base: float = 0.5, cap: float = 30.0, rng: Any = random) -> float:
    """Full-jitter exponential backoff (prevents retry storms across workers)."""
    # ``2 ** (attempt - 1)`` is an arbitrary-precision INT, so the multiplication
    # by the float ``base`` has to convert it — and above 2**1024 that conversion
    # raises (measured: ``backoff_delay(1025)`` →
    # ``OverflowError: int too large to convert to float``). A retry helper
    # crashing on its own arithmetic turns a recoverable error into an
    # unrecoverable one, so the exponent is clamped and computed in float.
    #
    # The clamp is exact, not approximate: ``base * 2**1023`` (~9e307 * base)
    # already exceeds any sane ``cap`` — a cap that large would be measured in
    # more seconds than the age of the universe — so ``min`` picks ``cap`` for
    # every attempt at or beyond the clamp anyway. Float multiplication that
    # overflows yields ``inf`` rather than raising, and ``min(cap, inf)`` is
    # ``cap`` too. Only the upper end is clamped: ``attempt <= 0`` still yields
    # the same sub-``base`` ceiling it always did.
    ceiling = min(cap, base * (2.0 ** min(attempt - 1, 1023)))
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
    _probe_started_at: float = 0.0
    # A real lock, not a documented caveat. Every transition below is a
    # read-modify-write on shared fields, and the old note ("for cross-thread
    # sharing, serialize the breaker externally") pushed that onto every caller
    # who shares one breaker across a thread pool — the obvious way to deploy
    # it. Two threads could both flip open→half_open and send two probes at the
    # dependency this gate exists to protect from exactly that.
    #
    # ``threading.Lock``, not ``asyncio.Lock``: none of these methods await, so
    # there is nothing to yield to and no deadlock to create, and an asyncio
    # lock would force all four to become coroutines. The cost is one
    # uncontended acquire against an outbound network call.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def allow(self) -> bool:
        """Admit this caller, or refuse. The `half_open` state is itself the
        single-probe gate: the caller that flips open→half_open IS the probe;
        every other caller sees `half_open` and is refused until that probe
        resolves. The lock is what makes "the caller that flips it" exactly one
        caller, across threads as well as within an event loop."""
        with self._lock:
            return self._allow_locked()

    def _allow_locked(self) -> bool:
        if self.state == "closed":
            return True
        now = self.clock()
        if self.state == "half_open":
            # A probe is in flight → admit no others. But the gate is
            # time-bounded, because "the probe always reports back" is an
            # assumption the callers cannot honour: ``run_with_resilience``
            # deliberately skips ``record_failure`` on PERMANENT errors, and a
            # ``BaseException`` (``CancelledError``, ``SystemExit``) escapes its
            # ``except Exception`` entirely. Measured before this guard: one 401
            # on the post-cooldown probe left ``state == "half_open"`` and every
            # later call — against a fully healthy dependency, 10_000 simulated
            # seconds later — raised ``CircuitOpen`` for the rest of the process
            # lifetime.
            #
            # After one cooldown with no verdict the probe is presumed lost and
            # a fresh one is admitted. Still exactly ONE at a time, so the
            # no-stampede property the half-open state exists for is intact.
            if now - self._probe_started_at >= self.cooldown:
                self._probe_started_at = now
                return True
            return False
        if now - self._opened_at >= self.cooldown:  # open + cooled down → admit exactly ONE probe
            self.state = "half_open"
            self._probe_started_at = now
            return True
        return False  # open, still cooling down

    def record_success(self) -> None:
        """A success closes the breaker from CLOSED or HALF_OPEN — never from OPEN.

        The class docstring describes the state machine as
        ``CLOSED -> OPEN -> (cooldown) -> HALF_OPEN -> CLOSED``. There is no
        ``OPEN -> CLOSED`` edge, and there was one in the code: this method
        set ``state = "closed"`` unconditionally.

        That edge is reachable at any concurrency above one, which is the
        normal case for the documented pattern of ONE breaker shared per
        dependency (``client.py`` builds ``CircuitBreaker("agentkit.llm")``
        once). Several calls are in flight, enough fail to trip the breaker,
        and then a straggler that started BEFORE the trip finishes
        successfully and reports it — reopening the gate and sending the herd
        straight back at the failing provider. Measured: a 300-second cooldown
        skipped entirely by one late success.

        Ignoring the report while OPEN is the honest reading of it: that call
        was admitted under the old state, so it is evidence about the past,
        not about whether the dependency has recovered. Only the single probe
        admitted after the cooldown speaks to that.
        """
        with self._lock:
            if self.state == "open":
                return
            self._fails = 0
            self.state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._fails += 1
            if self.state == "half_open" or self._fails >= self.fail_threshold:
                self.state = "open"
                self._opened_at = self.clock()

    def release_probe(self) -> None:
        """Hand the HALF_OPEN probe slot back after an outcome that says NOTHING
        about dependency health — a PERMANENT contract failure (401 / 403 /
        content filter) or a ``BaseException`` such as cancellation.

        Neither ``record_success`` nor ``record_failure`` fits. ``record_failure``
        would fold the outcome into ``_fails``, breaking the deliberate rule that
        a contract failure is not an upstream health signal; ``record_success``
        would grant health credit for a call that never proved the dependency
        recovered. Doing NEITHER — which is what happened — wedged the breaker in
        ``half_open`` forever, because the only exits were those two methods.

        So this is the third, neutral edge: consume nothing, grant nothing. The
        state returns to OPEN and the cooldown clock is **not** restarted, so the
        already-elapsed cooldown still holds and the very next caller is admitted
        as a fresh probe. A dependency that only ever answers with PERMANENT
        errors therefore keeps failing fast (the correct, un-masked error reaches
        the caller) instead of being hidden behind a ``CircuitOpen`` that
        classifies as TRANSIENT and invites a retry that can never succeed.
        """
        with self._lock:
            if self.state != "half_open":
                return  # nothing in flight — CLOSED / OPEN are unaffected
            self.state = "open"


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
    if isinstance(o, _types.MappingProxyType) or (isinstance(o, _Mapping) and not isinstance(o, dict)):
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
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_stable_default).encode()).hexdigest()[:length]


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
            if breaker is not None:
                if cls is not ErrorClass.PERMANENT:
                    breaker.record_failure()
                else:
                    # ...but the call still CONSUMED the half-open probe slot,
                    # and skipping the report left the breaker wedged in
                    # ``half_open`` with no exit (measured: one 401 on the
                    # post-cooldown probe refused a healthy dependency for the
                    # rest of the process lifetime). ``release_probe`` returns
                    # the slot without touching ``_fails``, so the rule above
                    # is preserved exactly.
                    breaker.release_probe()
            if cls is ErrorClass.PERMANENT or attempt == max_attempts:
                raise
        except BaseException:
            # ``CancelledError`` / ``SystemExit`` never reach the ``except
            # Exception`` above, so a probe that ended this way was the other
            # half of the same wedge. Cancellation is not a health signal
            # either — release the slot and let it unwind untouched.
            if breaker is not None:
                breaker.release_probe()
            raise
        else:
            if breaker is not None:
                breaker.record_success()
            return result
            await asleep(backoff_delay(attempt, rng=rng))
    raise (last if last is not None else RuntimeError("run_with_resilience: no attempt ran"))  # pragma: no cover
