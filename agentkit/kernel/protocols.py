"""Structural protocols the kernel exposes to higher layers.

Why this module exists. `RunContext` lives in `agentkit.runtime.context`, which sits ABOVE the
kernel and the capabilities/patterns layers. Capabilities and patterns need to *type-annotate* the
context they receive without importing it (that import would be a circle). The historical
workaround was `ctx: Any` everywhere — type-checker silence + a worse story for callers.

`Ctx` is the **structural** surface a `RunContext` exposes to capabilities and patterns: just the
attributes/methods they actually read. `RunContext` satisfies it via duck-typing — no inheritance
needed. Test stubs satisfy it the same way.

`.trace` is pinned to `TracePort` (the kernel-owned structural shape from
`agentkit.kernel.observation`); its `span` returns `Any` so the span object itself stays
heterogeneous. `.invoker`, `.store`, `.checkpointer` remain `Any` — they have their own
structural shapes that don't yet have canonical Protocols in the kernel surface, and pinning them
here would force every test stub / adapter to implement those shapes, which is invasive for a
refactor whose point is to *remove* `Any`, not to spread Protocol scaffolding across every
collaborator. Each sub-object has its own real type at runtime (`Invoker`, `StorePort`) —
callers reach those through this façade.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, Self

from agentkit.kernel.observation import ObservationKind, TracePort

# The autonomy seam stays kernel-local; typing it as ``Literal``
# instead of pulling in the agents-layer ``Autonomy`` enum avoids a
# circular import while still catching typos (``"gaged"``, ``"Manual"``)
# at the write site. The ``Autonomy`` enum in
# ``agents.control.gate`` inherits from ``str`` — its values ARE
# these literals — so an ``Autonomy`` instance satisfies this type
# at runtime with no cast.
AutonomyLiteral = Literal["auto", "gated", "manual"]


class Ctx(Protocol):
    """The slice of `RunContext` that capabilities and agents actually use.

    Captured by going through every `ctx.<name>` call site in `agentkit/capabilities/*` and
    `agentkit/agents/*`. Anything an internal runtime path touches but no capability/agent
    reads is deliberately omitted — keeping this surface narrow keeps it cheap to satisfy from
    tests and adapters.
    """

    # IDENTITY ------------------------------------------------------------------------------
    correlation_id: str  # the run id (durable checkpoint keys, span attributes)
    scope: Any  # tenant Scope — passed to VectorPort.upsert/search
    autonomy: AutonomyLiteral  # "auto" | "gated" | "manual" — gating policy input

    # SERVICES (façades; each has its own structural shape) ---------------------------------
    @property
    def trace(self) -> TracePort:
        """Tracing facade; supports ``with ctx.trace.span(...)``."""
        ...

    @property
    def invoker(self) -> Any:
        """Model/tool invocation facade."""
        ...

    @property
    def store(self) -> Any | None:
        """Generic durable store, when configured."""
        ...

    @property
    def checkpointer(self) -> Any | None:
        """Durable run-state checkpointer, when configured."""
        ...

    @property
    def asker(self) -> Any | None:
        """Human transport, when configured."""
        ...

    # ``agents.control.elicitation.Asker`` is intentionally kept out of
    # this kernel protocol to avoid an upward import. When present a
    # cognition parks on it (awaiting in place with live state); when
    # absent it falls back to checkpoint-and-suspend.
    # Framework code reads it via ``getattr(ctx, "asker", None)`` so a
    # NullCtx / structural stub predating this field still works.

    # COOPERATIVE CONTROL -------------------------------------------------------------------
    def check_cancelled(self) -> None:
        """Raise `Cancelled` if the cooperative cancellation token has been tripped."""
        ...

    def semaphore(self) -> Any:
        """Return the budget-derived concurrency semaphore for `gather_bounded`."""
        ...

    def child(self) -> Self:
        """Return a child context with `depth+1`, sharing the budget/services/cancel token."""
        ...

    async def emit(
        self,
        kind: ObservationKind,
        render: str = "",
        *,
        payload: Any = None,
        agent: str = "",
        parent_id: str | None = None,
    ) -> None:
        """Publish a product-facing observation on the run's observer (never raises into the run)."""
        ...


__all__ = ["Ctx", "AutonomyLiteral"]
