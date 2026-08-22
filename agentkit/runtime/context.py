"""RunContext — the run-scoped state threaded through every unit of work.

It carries three things: IDENTITY (`correlation_id`, `scope`, `depth`), METERS (`budget` always + any
extra meters like a tenant `Quota`), and SERVICES (`invoker`, `store`, `vector`, `trace` — the
app-shared collaborators). Patterns read `ctx.invoker` / `ctx.trace` / `ctx.store`; cross-cutting
concerns live in the invoker's middleware chain, not as fields here.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.kernel.concurrency import CancellationToken
from agentkit.kernel.metrics import MetricsPort, NoopMetrics
from agentkit.kernel.observation import (
    CRITICAL_KINDS,
    NoopObserver,
    Observation,
    ObservationKind,
    ObserverPort,
    TraceContext,
    TracePort,
)
from agentkit.kernel.protocols import AutonomyLiteral
from agentkit.kernel.replay import NoopReplayStore, ReplayStore
from agentkit.kernel.sampling import AlwaysOnSampler, SamplerPort
from agentkit.kernel.types import Scope
from agentkit.runtime.meter import Budget

if TYPE_CHECKING:
    from agentkit.agents.control.budget import ActorBudget
    from agentkit.agents.control.channel import SignalChannel


# Sentinel for RunContext.child() kwargs that need to distinguish
# "unspecified — propagate the parent's value" from "explicitly set to None".
# ``None`` is a legal value for both ``actor_budget`` and ``signal_channel``
# (opting out at this level of the tree), so we can't use ``None`` as the
# default. A private module-level sentinel disambiguates.
_UNSET: Any = object()


class NoopSpan:
    def set(self, key: str, value: Any) -> None: ...
    def add_event(self, name: str, **fields: Any) -> None: ...


class NoopTrace:
    """Default trace so a RunContext works without an injected exporter (tests, lean paths).

    Structurally satisfies `TracePort` from `agentkit.kernel.observation`: `span(name, kind, **attrs)`
    returns a sync context manager yielding a span with `.set` + `.add_event`; `current_span_id()`
    returns None (no real tracer attached, so no link); `add_event_to_current_span` is a no-op."""

    @contextmanager
    def span(self, name: str, kind: str, **attrs: Any) -> Any:
        yield NoopSpan()

    def current_span_id(self) -> TraceContext | None:
        return None

    def add_event_to_current_span(self, name: str, **fields: Any) -> None:
        return None


@dataclass
class Services:
    """App/process-shared collaborators, injected once and shared across runs."""

    invoker: Any = None  # Invoker (runs units of work through the middleware chains)
    store: Any = None  # StorePort (generic KV: idempotency, audit, cache)
    checkpointer: Any = None  # Checkpointer over CheckpointPort — durable run state
    vector: Any = None  # VectorPort (RAG / semantic cache)
    # Both seams default to no-op so a `Services()` with no args is fully usable.
    trace: TracePort = field(default_factory=NoopTrace)
    observer: ObserverPort = field(default_factory=NoopObserver)
    # Side-channel store for full LLM call replay payloads. Defaults to a
    # no-op so a `Services()` with no args is fully usable; production wires
    # a concrete adapter (Langfuse / Phoenix / S3 / disk). Spans carry the
    # index (span_id); the store carries the bulk so attribute cardinality
    # stays under the OTel SDK budget.
    replay: ReplayStore = field(default_factory=NoopReplayStore)
    # Metrics seam — counters + histograms for the time-series rollup
    # (token usage, latency, errors). Defaults to ``NoopMetrics`` so a
    # ``Services()`` with no args is fully usable. Production wires the
    # OTel ``OtelMetricsPort`` via ``adapters.observability.otel_meter``.
    metrics: MetricsPort = field(default_factory=NoopMetrics)
    # Sampler seam — head-sampling decision consulted at span open.
    # Defaults to ``AlwaysOnSampler`` (record every span) so existing
    # tests stay green; production overrides with ``TraceIdRatioSampler``
    # once traffic outpaces the trace backend.
    sampler: SamplerPort = field(default_factory=AlwaysOnSampler)
    # Human-in-the-loop transport (``agents.control.elicitation.Asker``). The one
    # seam that turns a suspend into a PARK: when set, a cognition awaits the
    # asker in place — the coroutine keeps its live, unserialisable state and
    # nothing unwinds. When unset (the default), the classic
    # checkpoint-and-return-and-resume path runs exactly as before, so this is
    # purely additive. Terminal / HTTP / queue / Slack are all the same to the
    # runtime: it never branches on transport, it only awaits ``ask``.
    # Declared ``Any`` to keep the runtime free of an import from the agents
    # layer (which imports the runtime).
    asker: Any = None


@dataclass
class RunContext:
    correlation_id: str
    scope: Scope
    budget: Budget = field(
        default_factory=Budget
    )  # the always-on run meter + depth/concurrency authority
    services: Services = field(default_factory=Services)
    meters: list[Any] = field(
        default_factory=list
    )  # EXTRA meters beyond budget (e.g. a tenant Quota)
    depth: int = 0
    cancel: CancellationToken | None = None  # cooperative cancellation, SHARED across the tree
    autonomy: AutonomyLiteral = "auto"  # "auto" | "gated" | "manual" — run-wide HITL level
    # Per-actor budget (``agentkit.agents.control.budget.ActorBudget``). Opt-in:
    # when set on the parent's ctx, ``run_agents`` carves a per-child slice via
    # ``reserve_for_child`` before dispatch and settles it after. Propagated by
    # reference through ``child()`` (like ``budget``) so the ENTIRE subtree sees
    # the same actor budget unless a caller explicitly overrides in ``child(actor_budget=...)``.
    # Framework code that touches this field reads via ``getattr(ctx, "actor_budget", None)``
    # so a NullCtx / structural stub without the field still works.
    actor_budget: ActorBudget | None = None
    # Per-actor signal channel (``agentkit.agents.control.channel.SignalChannel``).
    # Opt-in: when a coordinator ctx has ``signal_channel`` set and a spawned
    # child ``Agent`` has its own ``channel``, ``run_agents`` calls
    # ``child.channel.attach_parent(ctx.signal_channel.merge_inbox)`` so
    # ``DataSignal``s emitted by the child fan up to the coordinator's merge
    # inbox. Propagated through ``child()`` unless overridden.
    signal_channel: SignalChannel[Any, Any] | None = None

    # convenience accessors so patterns read ctx.invoker / ctx.trace / ctx.store / ctx.vector
    @property
    def invoker(self) -> Any:
        return self.services.invoker

    @property
    def trace(self) -> TracePort:
        return self.services.trace

    @property
    def store(self) -> Any:
        return self.services.store

    @property
    def checkpointer(self) -> Any:
        return self.services.checkpointer

    @property
    def vector(self) -> Any:
        return self.services.vector

    @property
    def asker(self) -> Any:
        """The run's human transport, or ``None``. See ``Services.asker``."""
        return self.services.asker

    @property
    def observer(self) -> ObserverPort:
        return self.services.observer

    @property
    def replay(self) -> ReplayStore:
        return self.services.replay

    @property
    def metrics(self) -> MetricsPort:
        return self.services.metrics

    @property
    def sampler(self) -> SamplerPort:
        return self.services.sampler

    @property
    def all_meters(self) -> list[Any]:
        return [self.budget, *self.meters]  # budget is always metered

    def child(
        self,
        *,
        actor_budget: Any = _UNSET,
        signal_channel: Any = _UNSET,
    ) -> RunContext:
        """Return a child context — depth+1, sharing budget/services/cancel by reference.

        Kwargs are opt-in overrides for the multi-agent coordination seams. Both
        default to the sentinel ``_UNSET`` (propagate the parent's value); pass
        ``None`` to explicitly UNSET at this level of the tree, or a concrete
        value to override. Existing callers (`ctx.child()` with no args) keep
        the unchanged propagation-by-reference behavior.
        """
        if self.depth + 1 > self.budget.max_depth:
            from agentkit.runtime.meter import MeterExceeded

            raise MeterExceeded(f"agent depth {self.depth + 1} > max_depth {self.budget.max_depth}")
        return RunContext(
            correlation_id=self.correlation_id,
            scope=self.scope,  # inherited read-only
            budget=self.budget,  # SHARED across the tree
            services=self.services,  # SHARED
            meters=self.meters,  # SHARED
            depth=self.depth + 1,
            cancel=self.cancel,  # SHARED — cancelling a parent cancels its children
            autonomy=self.autonomy,  # inherited run-wide HITL level
            # SHARED by reference (like ``budget``) unless the caller overrode.
            actor_budget=self.actor_budget if actor_budget is _UNSET else actor_budget,
            signal_channel=self.signal_channel if signal_channel is _UNSET else signal_channel,
        )

    def semaphore(self) -> Any:
        """This context's concurrency pool — scoped to ``depth``.

        Passing ``self.depth`` is what makes nested fan-out deadlock-free; see
        ``Budget.semaphore`` for why a single tree-wide pool cannot work.
        """
        return self.budget.semaphore(self.depth)

    def check_cancelled(self) -> None:
        """Cooperative cancellation check for patterns (no-op if no token is attached)."""
        if self.cancel is not None:
            self.cancel.raise_if_cancelled()

    async def emit(
        self,
        kind: ObservationKind,
        render: str = "",
        *,
        payload: Any = None,
        agent: str = "",
        parent_id: str | None = None,
    ) -> None:
        """Emit a structured observation on the run's observer (no-op default; never raises).

        If a tracer is attached and a span is currently open, the resulting Observation also carries
        a ``trace_context`` so UI consumers can deep-link from the observation to its span tree.
        For CRITICAL kinds (result/error) we additionally drop an ``observation.emitted`` event on
        the currently-open span so the trace timeline is complete without cross-referencing streams.
        Both trace-side effects are best-effort — they must never break the run."""
        trace_ctx: TraceContext | None = None
        trace = self.services.trace
        if trace is not None:
            with contextlib.suppress(Exception):
                trace_ctx = trace.current_span_id()
        # The observer.emit call must NEVER break the run. A misbehaving
        # adapter (Redis hiccup, kafka push error, slow subscriber
        # timeout) would otherwise propagate straight into the agent
        # loop. Every middleware and pattern in the framework emits
        # through this seam expecting it to be inert; wrap unconditionally.
        with contextlib.suppress(Exception):
            await self.observer.emit(
                Observation(
                    kind=kind,
                    render=render,
                    payload=payload,
                    run_id=self.correlation_id,
                    agent=agent,
                    parent_id=parent_id,
                    trace_context=trace_ctx,
                )
            )
        if kind in CRITICAL_KINDS and trace is not None:
            with contextlib.suppress(Exception):
                trace.add_event_to_current_span("observation.emitted", kind=kind)
