"""Failure — errors as first-class DATA, plus the typed error taxonomy raised at framework boundaries.

A `Failure` is a *value* a parent can reason about (retry / route around / escalate), distinct from an
exception (control flow / fault). It is categorised (via `classify`), names its `source`, and can carry a
`partial_output` and the originating `cause`. Failures **compose** (a parent aggregates child failures).

The exception classes below are the *control-flow* companion — a typed taxonomy that adapter and port
boundaries raise so a caller can pattern-match on cause without inspecting backend-specific types
(`asyncpg.PostgresError`, `httpx.HTTPError`, `redis.RedisError`). Every framework-raised exception is a
subclass of ``AgentkitError`` so a defensive ``except AgentkitError:`` catches the whole surface.

This module is dependency-free; built on the kernel's `classify`/`ErrorClass`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentkit.kernel.resilience import ErrorClass, classify


@dataclass(frozen=True)
class Failure:
    """A failure as data. `category` is TRANSIENT/PERMANENT/UNKNOWN; `retriable` defaults from it.

    Frozen — the docstring promises "a *value* a parent can reason
    about" and ``compose_failures`` aggregates children into an
    immutable tuple. Mutation of a child ``Failure`` observed by two
    parents would corrupt one parent's view under the other's edits."""

    category: ErrorClass
    source: str
    message: str
    retriable: bool = False
    cause: BaseException | None = None
    partial_output: Any = None
    children: tuple[Failure, ...] = ()  # set when this aggregates child failures

    @classmethod
    def of(cls, exc: BaseException, *, source: str, partial_output: Any = None) -> Failure:
        """Build a `Failure` from an exception, classifying it.

        ``retriable`` is True for TRANSIENT **and UNKNOWN**.
        ``run_with_resilience`` retries both classes; keeping
        ``Failure.retriable`` aligned means a higher layer that reads
        it directly makes the same decision the kernel's resilience
        loop would. Conservative on UNKNOWN means "try again" (matches
        ``classify``'s prose: 'UNKNOWN  # conservative')."""
        category = classify(exc)
        return cls(
            category=category,
            source=source,
            message=str(exc) or exc.__class__.__name__,
            retriable=(category != ErrorClass.PERMANENT),
            cause=exc,
            partial_output=partial_output,
        )


def compose_failures(
    failures: Sequence[Failure | None], *, source: str = "composite"
) -> Failure | None:
    """Aggregate child failures into one (or `None` if there are none; passthrough a single one).

    Category rule: PERMANENT if any child is permanent; TRANSIENT only if
    *all* are transient; else UNKNOWN. The aggregate is retriable for any
    non-PERMANENT category, matching ``Failure.of`` and
    ``run_with_resilience``. Children are preserved.
    """
    items = [f for f in failures if f is not None]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    cats = {f.category for f in items}
    if ErrorClass.PERMANENT in cats:
        category = ErrorClass.PERMANENT
    elif cats == {ErrorClass.TRANSIENT}:
        category = ErrorClass.TRANSIENT
    else:
        category = ErrorClass.UNKNOWN
    message = f"{len(items)} failures: " + "; ".join(f"{f.source}: {f.message}" for f in items)
    return Failure(
        category=category,
        source=source,
        message=message,
        retriable=(category != ErrorClass.PERMANENT),
        children=tuple(items),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Typed error taxonomy raised at adapter/port boundaries.
#
# Adapters catch their backend's raw exception type (asyncpg.PostgresError,
# httpx.HTTPError, redis.RedisError, json.JSONDecodeError, …) and wrap it in
# one of the classes below with ``raise …Error(…) from exc`` so the original
# cause is preserved on ``__cause__`` while callers get a stable, backend-
# agnostic type to except on.
#
# Rule of thumb for adapter authors: catch narrowly around the I/O boundary,
# wrap in the closest ``AgentkitError`` subclass, and chain the cause. Never
# let a backend type leak past the port.
# ─────────────────────────────────────────────────────────────────────────────


class AgentkitError(Exception):
    """Base for all typed agentkit framework errors."""


class CheckpointerError(AgentkitError):
    """Raised when a checkpoint store operation fails or a saved state is malformed."""


class StoreUnavailable(AgentkitError):
    """Raised when a StorePort operation cannot reach the backing store."""


class StoreValueError(AgentkitError):
    """Raised when a StorePort operation cannot be applied to what the key holds.

    Distinct from `StoreUnavailable`: the store is reachable and answered, and
    retrying will produce the same answer. ``increment`` on a key holding
    ``{"a": 1}`` is the whole reason this exists — the three durable backends
    each fail that differently (Redis ``ResponseError``, Postgres
    ``InvalidTextRepresentationError`` from the ``::bigint`` cast, a dict-backed
    store ``TypeError``), which is the backend-type leak this module's taxonomy
    exists to close. A caller writing ``except StoreValueError`` must not have
    to know which store is wired underneath.
    """


class ProviderAuthError(AgentkitError):
    """Raised when an LLM provider returns 401/403 (invalid credentials or forbidden).

    The concrete instance raised by provider adapters (see
    ``agentkit.adapters.llm.providers.base``) additionally mixes in
    ``ProviderError`` at that layer, so raised instances satisfy both
    ``isinstance(exc, ProviderAuthError)`` (kernel taxonomy) AND
    ``isinstance(exc, ProviderError)`` (existing transport-level catches).
    """


__all__ = [
    "Failure",
    "compose_failures",
    "AgentkitError",
    "CheckpointerError",
    "StoreUnavailable",
    "StoreValueError",
    "ProviderAuthError",
]
