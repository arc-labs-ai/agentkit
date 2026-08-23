"""resilience middlewares — `retry` and `fallback`.

`retry` classifies a failure, backs off with jitter, and re-invokes through an optional circuit breaker.
`fallback` walks an ordered list of models: on a transient failure or an OPEN breaker it rewrites
`request.model` to the next model and re-invokes `next`; a PERMANENT error raises immediately (walking
the chain on a 4xx just burns money). Composing `fallback` (outer) over `retry` (inner) gives "try model
A with retries, then model B with retries".
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

from agentkit.kernel.middleware import Call, Handler, Middleware
from agentkit.kernel.resilience import (
    CircuitOpen,
    ErrorClass,
    backoff_delay,
    classify,
)


def _emit_retry_event(call: Call, *, attempt: int, exc: BaseException, delay_s: float) -> None:
    """Drop a ``retry.attempt`` event on the currently-open span — best-effort.

    Fired BEFORE each retry (i.e. after a failure when another attempt is queued) so the
    span timeline carries the narrative reason for the back-off without needing to
    cross-reference structured logs. A misbehaving tracer must never break the run, so
    the call is wrapped in ``contextlib.suppress(Exception)``.

    Carries the classified error CLASS name plus the truncated error MESSAGE (capped at
    500 chars). A full request-diff between attempts would require snapshotting the
    previous attempt's request; that's a future addition once there's a use case —
    today we lean on the error message as the bigger signal (e.g. the provider's
    rate-limit detail or 5xx body)."""
    trace = getattr(call.ctx, "trace", None)
    if trace is None:
        return
    with contextlib.suppress(Exception):
        trace.add_event_to_current_span(
            "retry.attempt",
            attempt=attempt,
            last_error_type=type(exc).__name__,
            last_error_message=str(exc)[:500],
            backoff_ms=int(delay_s * 1000),
        )


async def _first(gen: AsyncIterator[Any]) -> tuple[Any, bool]:
    """Pull the first item of a stream — surfacing a *pre-stream* failure (the retry/fallover trigger).
    Returns (first_item, True) or (None, False) on an immediately-empty stream."""
    try:
        return await gen.__anext__(), True
    except StopAsyncIteration:
        return None, False


def retry(
    *,
    breaker: Any = None,
    max_attempts: int = 3,
    classify_fn: Callable[[BaseException], ErrorClass] = classify,
    sleep: Callable[[float], Any] | None = None,
) -> Middleware:
    """Re-invoke on a PRE-stream failure (before the first delta — the common case: rate-limit/connect/OPEN
    breaker); once the first delta is yielded, commit and stream through (a mid-stream failure propagates —
    you can't un-send tokens). PERMANENT errors fail fast."""
    asleep = sleep or asyncio.sleep

    async def mw(call: Call, nxt: Handler) -> AsyncIterator[Any]:
        last: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            # Publish the attempt count so an OUTER middleware can report it.
            # ``Audit`` sits outside ``retry()`` in the documented tool chain
            # and gets ONE phase pair per chain invocation, so three executions
            # folded into a single record that read as one execution —
            # measured: a side-effecting tool ran 3x and the trail said
            # ``executed`` once, with nothing to distinguish it from a clean
            # single call. It still cannot write three records from one
            # invocation, but it can stop implying one.
            call.meta["attempts"] = attempt
            if breaker is not None and not breaker.allow():
                raise CircuitOpen(f"circuit open: {getattr(breaker, 'name', '?')}")
            gen = nxt(call)
            try:
                first, ok = await _first(gen)
            except Exception as exc:  # noqa: BLE001 — pre-stream failure → classify decides retry vs fail
                last = exc
                if breaker is not None:
                    breaker.record_failure()
                if classify_fn(exc) is ErrorClass.PERMANENT or attempt == max_attempts:
                    raise
                delay = backoff_delay(attempt, rng=random)
                # Narrative span event: record the attempt + classified
                # reason on the currently-open (chat/tool) span so the
                # timeline shows the back-off.
                _emit_retry_event(call, attempt=attempt + 1, exc=exc, delay_s=delay)
                await asleep(delay)
                continue
            if not ok:  # empty stream → nothing to retry
                if breaker is not None:
                    breaker.record_success()
                return
            yield first  # committed to this attempt — stream the rest
            try:
                async for d in gen:
                    yield d
            except Exception:  # noqa: BLE001 — a mid-stream fault still trips the breaker
                if breaker is not None:
                    breaker.record_failure()
                raise  # not retried (tokens already shown), but recorded
            if breaker is not None:
                breaker.record_success()
            return
        raise last if last is not None else RuntimeError("retry: no attempt ran")

    return mw


def fallback(
    models: list[str],
    *,
    classify_fn: Callable[[BaseException], ErrorClass] = classify,
    breakers: dict[str, Any] | None = None,
) -> Middleware:
    """`models` is the ordered fallover chain; the served model is left on `request.model`. Falls over on a
    PRE-stream transient failure / OPEN breaker; commits to a model once its first delta flows."""

    async def mw(call: Call, nxt: Handler) -> AsyncIterator[Any]:
        last: BaseException | None = None
        for model in models:
            br = breakers.get(model) if breakers else None
            if br is not None and not br.allow():
                last = CircuitOpen(f"circuit open: {getattr(br, 'name', model)}")
                continue
            # ChatRequest is frozen — rewrite via replace + reassign the
            # mutable Call.request reference. The served model remains
            # readable on the request after the chain returns.
            call.request = replace(call.request, model=model)
            gen = nxt(call)
            try:
                first, ok = await _first(gen)
            except Exception as exc:  # noqa: BLE001 — classify decides fall-over vs fail-fast
                if br is not None:
                    br.record_failure()
                if classify_fn(exc) is ErrorClass.PERMANENT:
                    raise
                last = exc
                continue
            if ok:
                yield first  # committed to this model — stream the rest
                try:
                    async for d in gen:
                        yield d
                except Exception:  # noqa: BLE001 — mid-stream fault trips the breaker
                    if br is not None:
                        br.record_failure()
                    raise  # not failed-over (tokens already shown), but recorded
            if br is not None:
                br.record_success()
            return
        raise last if last is not None else RuntimeError("fallback: no models")

    return mw
