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

from agentkit.kernel._frozen import deep_freeze

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
    consumer sees — the same reason ``TraceContext`` is frozen.

    NO ``seq`` / ``ts``, deliberately — they used to be here and were
    removed rather than populated. Measured before the removal:
    ``Observation(kind="tool_result", payload={}).seq`` was ``0`` and
    ``.ts`` was ``0.0``, and NOTHING in the package ever set them — not
    ``RunContext.emit``, not the signal channel, not ``RollupObserver``,
    not any adapter. ``__hash__`` nonetheless folded both into the stream
    key and its docstring described ``seq`` as "that emitter's monotonic
    counter", so the type promised an audit ordering it never delivered.

    Populating them was the other option and was rejected on evidence,
    not taste:

    * **The channel already orders itself, more strongly.** ``emit`` is
      awaited, so a sink observes emissions in emission order:
      ``CollectingObserver.items`` is append-ordered and
      ``QueueObserver.stream()`` yields in insertion order. A ``seq``
      would be a second, weaker source of truth for something the
      structure guarantees — and it would be wrong in the one case a
      field could beat arrival order (cross-process replay), because a
      per-``RunContext`` counter restarts at 0 in every ``child()`` while
      a counter on the shared ``Services`` would be shared by any two
      runs an application wires to one ``Services``.
    * **Durable ordering is already owned by the store.** The Postgres
      log adapter allocates ``seq BIGSERIAL PRIMARY KEY`` per row and
      reads back ``ORDER BY seq``. An audit stream that outlives the
      process gets its sequence there, from the only component that can
      make it total.
    * **One truthful seam is not enough.** Observations are constructed
      at four sites (``RunContext.emit``, ``SignalChannel``'s
      ``signal.emitted``, ``RollupObserver``'s roll-up, and the testing
      fake ctx). Stamping only the emit seam re-tells the same lie in
      three smaller places.
    * **``ts`` had no owner either.** Wall-clock display and operational
      timing both already live on the span ``trace_context`` deep-links
      into; a second, unpopulated timestamp on the record only invites
      the monotonic-vs-wall-clock question with no consumer to answer it.

    A consumer that genuinely needs a per-sink sequence or arrival stamp
    should keep it in the SINK, beside the record — the sink is what
    knows when it saw the observation, and one shared record cannot carry
    a different number for each of the three fan-out targets anyway."""

    kind: ObservationKind
    agent: str = ""
    render: str = ""
    run_id: str = ""
    payload: Any = None
    parent_id: str | None = None
    # Populated by ``RunContext.emit`` from the currently-open span (via
    # ``TracePort.current_span_id``) so a UI-visible observation can deep-link into the span tree
    # that produced it. None when no tracer is configured or no span is open.
    trace_context: TraceContext | None = None

    def __post_init__(self) -> None:
        """Freeze ``payload`` if it is a container; pass it through if not.

        An Observation is FANNED OUT — one record reaches an audit sink, a
        WebSocket forwarder and a rollup buffer, all of which retain the ref
        alongside the run's live emit loop. The class docstring already claims
        post-emit mutation would corrupt every downstream consumer; before
        this, ``frozen=True`` only enforced that for ``render``, while
        ``obs.payload["status"] = "done"`` — on the field that IS the
        machine-readable record — rewrote what the other two sinks had already
        queued.

        ``payload`` is annotated ``Any`` and genuinely is: `ctx.emit` is
        documented as ``emit("summary", "wrote intro", payload={"words": 120})``
        but callers also pass a bare string, an int, or nothing. ``deep_freeze``
        returns non-containers untouched, so a ``str``/``int``/``None`` payload
        is bit-for-bit the object the caller passed and only dict/list payloads
        are walked. Both directions are asserted in
        ``tests/kernel/test_frozen_value_payloads.py``.

        COST, stated plainly because this is the one hot path in the set: an
        Observation is constructed once per emitted event, and freezing is
        O(payload). Measured on this machine, per construction —

            payload                       before      after
            None / str / int              1.84 µs     2.30 µs   (+0.46)
            {"step": 3, "of": 10}         1.87 µs     3.23 µs   (+1.36)
            4-key summary                 1.84 µs     3.84 µs   (+2.00)
            nested signal payload         1.84 µs     5.20 µs   (+3.36)
            full agent result (~40 keys
            + 10-message transcript)      1.84 µs     37.9 µs    (+36)

        End to end on the emit path — construct, ``contextlib.suppress``, and
        ``await`` into a ``NoopObserver``, i.e. the cheapest emit that exists —
        a 4-key summary went from 2.62 µs to 4.60 µs. That is +75%, and it is
        the honest headline number rather than a footnote. What it buys back:
        the same 2 µs sits in front of an ``await`` into an adapter that
        buffers, forwards to a socket or writes an audit row, so it does not
        show up against a real observer; the +36 µs case is a ``result``,
        emitted once per run; and scalar payloads — the ones on the tightest
        progress loops — pay 0.46 µs. Judged worth it, because the alternative
        is a record three sinks share and any one of them can rewrite. A caller
        who profiles this as their bottleneck has a legitimate complaint and
        the fix is a shallower payload, not an unfrozen one.

        Frozen dataclass — assign through ``object.__setattr__``.
        """
        object.__setattr__(self, "payload", deep_freeze(self.payload))

    def __hash__(self) -> int:
        """Hash on the STREAM key — ``(run_id, agent, kind)`` — never on
        ``payload``.

        A frozen dataclass derives ``__hash__`` from every compared field, and
        ``payload: Any`` is in practice always a dict. That annotation is why
        this one hid: the type is hashable when a caller drops a string in and
        stops being hashable the moment it holds what observations actually
        carry. Measured before this fix::

            hash(Observation(kind="progress", payload="tick"))
            4380108333848103356                       # (salted; varies per run)
            hash(Observation(kind="result", payload={"k": "v"}))
            TypeError: unhashable type: 'dict'

        The second line is the real one — `RunContext.emit` is documented and
        used as ``emit("summary", "wrote intro", payload={"words": 120})``, so
        every observation a real run produces was unhashable, while a test that
        passed a string proved the opposite. It was found by the value-type
        ratchet rather than by a caller, precisely because a ratchet that
        builds MINIMAL instances would have passed it too.

        The stream key is what the ATTRIBUTION fields actually say: ``run_id``
        says which run, ``agent`` says who inside it, ``kind`` says what sort of
        record it is. None of it has anything to do with the payload, which is
        the point.

        It used to read ``(run_id, agent, seq, ts, kind)`` and describe ``seq``
        as "that emitter's monotonic counter". Nothing in the package ever set
        either field (see the class docstring for the removal and its
        rationale), so that sentence was false and the two extra tuple slots
        were constants. **Dropping them cost no discrimination at all**, which
        is the whole argument — measured over 1000 progress observations from
        one agent in one run::

                                          before   after
            distinct hashes                    1       1
            distinct records (``set``)      1000    1000
            hash, 1-key payload           0.372   0.258 µs
            hash, 100_000-key payload     0.434   0.252 µs

        Both hash counts are 1 because ``seq=0`` / ``ts=0.0`` on every record
        already made those two slots constant in every real run. The key
        discriminates exactly as well as it did before; it just no longer
        claims otherwise — and hashing got slightly cheaper for hashing three
        things instead of five. The payload is still never read, which is why
        the 1-key and 100_000-key rows agree.

        The 1000-vs-1 rows are the invariant that makes the collapse safe:
        ``__eq__`` still compares every field, so a ``set`` of a replayed
        stream keeps all 1000 records.

        The flip side is worth stating plainly rather than hiding: a run that
        dedups N same-``kind`` observations from one agent through a ``set``
        puts all N in one bucket and pays O(N²) in ``__eq__``. That is a real
        cost — and it is TODAY's cost, not a new one, for the same reason the
        table above shows 1 and 1. If it ever bites, the fix is a key that
        includes something genuinely varying (the sink's own arrival index),
        not a field the framework leaves at zero.

        ``payload`` is excluded because it cannot be hashed — it is
        machine-readable application JSON, nested by design — and because the
        hash must stay O(1) in it: an observation is fanned out to every
        attached observer on the hot emit path, and a ``result`` payload is a
        whole agent output. Measured: 0.258 µs for a 1-key payload and 0.252 µs
        for a 100_000-key one, because the payload is never read. (Those two
        numbers were 0.32/0.32 when the key still carried ``seq``/``ts``; the
        table above has the before/after. The point of the pair is that they
        agree with each other, not their absolute value.)

        ``render`` is excluded as well, though ``str`` is hashable: it is a
        human line DERIVED from the same information as the payload, it varies
        freely (a summarizer rewrites it), and it adds no discrimination the
        stream key does not already have. ``parent_id`` and ``trace_context``
        are correlation decoration attached after the fact by the emitter — an
        observation is the same record whether or not a tracer happened to be
        wired in, so folding them into the hash would split one record across
        two buckets depending on configuration.

        Excluding all of that is sound rather than a workaround: ``__eq__``
        still compares every field, and the hash invariant only requires EQUAL
        objects to hash equally — never that unequal ones differ. Two
        observations sharing a stream key but differing in payload collide into
        one bucket and ``__eq__`` separates them there; that is what a bucket
        is for, and it is what keeps a ``set``-based dedup of a replayed
        observation stream exact.
        """
        return hash((self.run_id, self.agent, self.kind))


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
