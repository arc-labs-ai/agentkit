"""NullCtx — minimum-viable Ctx for single-shot agents.

`RequestBuilder.build()` (and any other capability that takes a `Ctx`)
expects a structural surface — `.trace.span(...)`, `.scope`, `.invoker`,
`.store`, `.correlation_id`, `.check_cancelled()`, etc. (see
`kernel/protocols.py::Ctx` for the canonical list). When a caller
hasn't yet built a full `RunContext` (e.g., a single-shot Planner that
only needs the RequestBuilder's prompt assembly, not the Invoker chain),
they need a Ctx that satisfies the Protocol without doing anything.

That's `NullCtx`. Every field is a sensible no-op:

  - `.trace.span(...)` is a context manager that records nothing
  - `.invoker` raises if used (the whole point — no LLM calls go
    through it)
  - `.store` and `.checkpointer` are None
  - `.check_cancelled()` is a no-op
  - `.scope` is a tenant-zero placeholder (`Scope()`)
  - `.semaphore()` returns a context manager that grants immediately
  - `.child()` returns self (no per-task isolation)
  - `.emit(...)` is a no-op coroutine

When the caller later threads a real `RunContext` through, this stub
goes away and behavior is unchanged at the call site. The Protocol's
structural typing guarantees that.

This is NOT for production use INSIDE the Invoker pipeline — there's no
budgeting, no tracing, no cancellation. It's specifically for callers
who use a *subset* of the Ctx surface (e.g., `RequestBuilder.build` only
touches `.trace.span`). If your code needs the Invoker, build a real
RunContext.
"""

from __future__ import annotations

from typing import Any

from agentkit.kernel.types import Scope


class _NullSpan:
    """A trace span that records nothing. Acts as both a sync and async
    context manager so it satisfies `with` and `async with` patterns."""

    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *_a: object) -> None:
        return None

    async def __aenter__(self) -> _NullSpan:
        return self

    async def __aexit__(self, *_a: object) -> None:
        return None

    def set(self, _key: str, _value: Any) -> None:
        return None

    def add_event(self, _name: str, **_fields: Any) -> None:
        return None


class _NullTracer:
    """A tracer that hands back `_NullSpan` for every span call. Accepts
    positional `(name, kind)` and arbitrary attribute kwargs, matching
    the `TracePort.span(name, kind, **attrs)` shape RequestBuilder uses."""

    def span(self, *_args: Any, **_attrs: Any) -> _NullSpan:
        return _NullSpan()


class _NullInvoker:
    """The point of `NullCtx` is callers that DON'T use the invoker. If
    anyone reaches for `.chat` / `.stream` / `.invoke_tool` here, they
    have the wrong context — fail loudly with a directive to wire a
    real RunContext."""

    async def chat(self, *_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("NullCtx has no invoker; thread a real RunContext for LLM calls")

    async def stream(self, *_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("NullCtx has no invoker; thread a real RunContext for streaming")

    async def invoke_tool(self, *_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("NullCtx has no invoker; thread a real RunContext for tool calls")


class _NullSemaphore:
    """An always-available semaphore-like. `gather_bounded` uses
    `async with sem:` — this grants immediately and tracks nothing.
    Calling `.acquire()` / `.release()` directly is also a no-op so
    legacy paths don't crash."""

    async def __aenter__(self) -> _NullSemaphore:
        return self

    async def __aexit__(self, *_a: object) -> None:
        return None

    async def acquire(self) -> bool:
        return True

    def release(self) -> None:
        return None


class NullCtx:
    """Minimum-viable Ctx. Structurally satisfies the `Ctx` Protocol
    in `agentkit.kernel.protocols`; every operation is a no-op or
    raises if it would require real behavior (only the invoker raises).

    Use this when the calling code touches a *subset* of the Ctx
    surface — e.g., `RequestBuilder.build()` only reads `.trace.span(...)`.
    When the caller later threads a real `RunContext`, behavior at the
    call site is unchanged thanks to structural typing.
    """

    correlation_id: str
    scope: Scope
    autonomy: str  # AutonomyLiteral at runtime; kept `str` here to stay Ctx-compatible via structural typing
    trace: _NullTracer
    invoker: _NullInvoker
    store: None
    checkpointer: None

    def __init__(self, scope: Scope | None = None) -> None:
        self.correlation_id = "null-ctx"
        # `Scope()` defaults to `(org_id=None, domain_id=None)` — tenant-zero
        # for callers that aren't tenant-scoped (single-shot, offline demo).
        self.scope = scope if scope is not None else Scope()
        self.autonomy = "manual"
        self.trace = _NullTracer()
        self.invoker = _NullInvoker()
        self.store = None
        self.checkpointer = None

    def check_cancelled(self) -> None:
        """No cancellation token attached — always a no-op."""
        return None

    def semaphore(self) -> _NullSemaphore:
        """Return an always-available semaphore-like. `gather_bounded`
        uses `async with`, which this satisfies without serialising."""
        return _NullSemaphore()

    def child(self) -> NullCtx:
        """No per-task isolation in a null context — the same instance
        is reusable as its own child. Real `RunContext.child()` bumps
        depth and shares services; here there is nothing to bump or
        share."""
        return self

    async def emit(
        self,
        kind: str,
        render: str = "",
        *,
        payload: Any = None,
        agent: str = "",
        parent_id: str | None = None,
    ) -> None:
        """No observer attached — drop the observation on the floor.
        Matches `RunContext.emit`'s contract of never raising into the
        run."""
        del kind, render, payload, agent, parent_id
        return None


__all__ = ["NullCtx"]
