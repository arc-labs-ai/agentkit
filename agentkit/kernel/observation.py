"""Observation — the product-facing progress/result channel, and `TracePort` — the
operational tracing seam.

A run emits **structured observations** (progress / rolling summary / partial / final result / error /
interrupt) to an attached observer. This is distinct from operational tracing (`TracePort`): observations
are always-on, ordered, and consumed by a parent agent and/or the UI. Emit is async and **must never
break the run** — the default is a no-op, and concrete observers (see `adapters/observer.py`) bound their
buffers and never drop a `result`/`error`.

This module holds the value types (`Observation`, `TraceContext`), both seams (`ObserverPort` and
`TracePort`), and the no-op observer default (kernel = opinion-free, no dependency). The no-op
trace default (`NoopTrace`) lives in `runtime/context.py` because it ships with `Services`; both
seams live here so callers have a single import for the observability surfaces the kernel exposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

# ``kind`` is a closed ``Literal`` so mypy can exhaust the branch table
# on any consumer of ``obs.kind`` — a typo becomes a type error instead
# of a silent missed-branch.
ObservationKind = Literal[
    "progress",
    "summary",
    "partial_result",
    "result",
    "error",
    "interrupt",
    "signal.emitted",
    "gate.check",
    "memory.written",
]

# Kinds that must never be dropped under backpressure (vs progress/summary, which may be coalesced).
CRITICAL_KINDS: tuple[ObservationKind, ...] = ("result", "error")


@dataclass(frozen=True)
class TraceContext:
    """W3C-style trace identifiers. Frozen so an emitted Observation
    carries a stable reference to the span that was open when it
    fired — even if subsequent spans open/close.

    ``trace_id`` is the run-wide identifier (hex string, OTel-compatible
    32 chars). ``span_id`` is the specific span that was open at emit
    time (hex string, 16 chars)."""

    trace_id: str
    span_id: str


@dataclass(frozen=True)
class Observation:
    """A structured, product-facing record. `payload` is machine-readable; `render` is a short human line.

    Frozen: an Observation is fanned out to observers (audit sinks,
    WebSocket forwarders, rollup buffers) that may retain the ref
    alongside the run's live emit loop. Post-emit mutation of
    ``render``/``payload`` would corrupt the record every subsequent
    consumer sees — the same reason ``TraceContext`` is frozen."""

    kind: ObservationKind
    seq: int = 0
    ts: float = 0.0
    agent: str = ""
    render: str = ""
    run_id: str = ""
    payload: Any = None
    parent_id: str | None = None
    # Populated by ``RunContext.emit`` from the currently-open span (via
    # ``TracePort.current_span_id``) so a UI-visible observation can deep-link into the span tree
    # that produced it. None when no tracer is configured or no span is open.
    trace_context: TraceContext | None = None


@runtime_checkable
class ObserverPort(Protocol):
    """Producer side of the channel. Concrete observers add a consumer surface (e.g. `stream()`).

    ``close()`` is part of the Protocol. Rollup/buffering observers
    need a shutdown hook to flush their buffered tail; without it the
    trailing summary is silently dropped at process exit.
    ``NoopObserver`` implements a no-op — passthrough impls can ignore
    it entirely and Protocol satisfaction is unchanged."""

    async def emit(self, obs: Observation) -> None: ...
    async def close(self) -> None: ...


class NoopObserver:
    """Default observer: drops everything. Lets a run emit freely with nothing attached (zero-config)."""

    async def emit(self, obs: Observation) -> None:
        return None

    async def close(self) -> None:
        # No buffers to flush; explicit no-op keeps the Protocol
        # satisfied and unblocks shutdown paths that call
        # ``await observer.close()`` uniformly.
        return None


@runtime_checkable
class TracePort(Protocol):
    """Operational tracing seam. Distinct from `ObserverPort` (the
    product-facing observation channel) — tracing records what the
    framework's middleware/capabilities did, with spans + structured
    attributes, for debugging and performance analysis. Implementations
    bridge to OpenTelemetry or structured logging.

    `span(name, kind, **attrs)` is a context manager (sync or async —
    both forms are supported) that yields a span object with `.set(key,
    value)` and `.add_event(name, **fields)`. The default `NoopTrace`
    in `runtime/context.py` does nothing; production wires an adapter.

    `current_span_id()` returns the currently-open span's
    `(trace_id, span_id)` for in-process consumers (e.g. `ctx.emit`) to
    attach as ``Observation.trace_context``. Returns None when no span
    is open (cold emit / no tracer configured).

    `add_event_to_current_span(name, **fields)` drops a span event on
    whatever span is currently open, without the caller having to hold
    a reference to it. Used by `ctx.emit` to mirror CRITICAL observations
    into the trace timeline. No-op when no span is open."""

    def span(self, name: str, kind: str, **attrs: Any) -> Any: ...
    def current_span_id(self) -> TraceContext | None: ...
    def add_event_to_current_span(self, name: str, **fields: Any) -> None: ...


__all__ = [
    "CRITICAL_KINDS",
    "NoopObserver",
    "Observation",
    "ObservationKind",
    "ObserverPort",
    "TraceContext",
    "TracePort",
]
