"""NullCtx — the minimum-viable Ctx for single-shot agents.

These tests pin three things:

  1. `NullCtx()` constructs and structurally satisfies the `Ctx`
     Protocol (annotate-and-assign to a `Ctx`-typed local; mypy gates
     the structural fit, the runtime test gates the surface).
  2. The no-op surface behaves correctly — `.trace.span(...)` works
     as both sync and async context manager, `.invoker.chat(...)`
     raises a directive RuntimeError, `.check_cancelled()` and
     `.emit(...)` are no-ops, `.child()` returns self, and
     `.semaphore()` is async-with-able.
  3. Most importantly: a real `RequestBuilder.build()` call accepts
     `NullCtx()` *without* a `# type: ignore[arg-type]` at the call
     site. That's the load-bearing outcome of this module — the
     planner's stale `_NoOpCtx` + ignore goes away.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.capabilities.request_builder import RequestBuilder
from agentkit.context import WorkingContext
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Scope
from agentkit.prompts.prompt import Prompt
from agentkit.runtime.null_ctx import NullCtx


def _run(coro):
    return asyncio.run(coro)


# ── structural Ctx satisfaction ────────────────────────────────────────────


def test_null_ctx_constructs_with_defaults() -> None:
    """No args needed: scope defaults to a tenant-zero Scope, identity
    fields are set, trace + invoker are bound to the no-op
    collaborators. This is the wire-up callers rely on."""
    nc = NullCtx()
    assert nc.correlation_id == "null-ctx"
    assert isinstance(nc.scope, Scope)
    assert nc.autonomy == "manual"
    assert nc.store is None
    assert nc.checkpointer is None


def test_null_ctx_accepts_an_explicit_scope() -> None:
    """A caller can pass a tenant-scoped `Scope` so any downstream that
    keys off `scope.key()` (caching, quotas) still works coherently."""
    scope = Scope(org_id=7, domain_id=42)
    nc = NullCtx(scope=scope)
    assert nc.scope is scope
    assert nc.scope.key() == "org7:dom42"


def test_null_ctx_satisfies_ctx_protocol_structurally() -> None:
    """Assigning a `NullCtx` to a `Ctx`-typed local is the structural
    check mypy enforces; at runtime we simply assert the fields the
    Protocol declares are present. If a `Ctx` field is added later and
    `NullCtx` doesn't grow with it, mypy on the planner's call site
    catches it — this test catches it too."""
    nc: Ctx = NullCtx()
    # IDENTITY
    assert hasattr(nc, "correlation_id")
    assert hasattr(nc, "scope")
    assert hasattr(nc, "autonomy")
    # SERVICES
    assert hasattr(nc, "trace")
    assert hasattr(nc, "invoker")
    assert hasattr(nc, "store")
    assert hasattr(nc, "checkpointer")
    assert hasattr(nc, "asker")
    # COOPERATIVE CONTROL
    assert callable(nc.check_cancelled)
    assert callable(nc.semaphore)
    assert callable(nc.child)
    assert callable(nc.emit)


# ── trace span: sync + async context-manager surfaces ─────────────────────


def test_trace_span_is_a_sync_context_manager() -> None:
    """`RequestBuilder.build()` enters the span with `with ctx.trace.span(...)`.
    The span must accept positional name/kind and arbitrary kwargs, and
    enter/exit cleanly without side effects."""
    nc = NullCtx()
    with nc.trace.span("compose", "compose", **{"agentkit.prompt.version": "v1"}) as span:
        # Spans expose .set / .add_event in the real TracePort; the null
        # span accepts them and silently drops the call.
        span.set("k", "v")
        span.add_event("evt", k="v")


def test_trace_span_is_also_an_async_context_manager() -> None:
    """Some callers will `async with` the span. The null span supports
    both so it can't surprise downstream patterns by being one-or-the-other."""

    async def go() -> None:
        nc = NullCtx()
        async with nc.trace.span("any", "kind"):
            pass

    _run(go())


# ── invoker: explicit-failure on misuse ────────────────────────────────────


def test_invoker_chat_raises_with_directive_message() -> None:
    """If a caller hands `NullCtx` to something that calls
    `ctx.invoker.chat(...)`, they have the wrong context. The error
    message must point them at the fix — wire a real RunContext."""

    async def go() -> None:
        nc = NullCtx()
        with pytest.raises(RuntimeError, match="thread a real RunContext"):
            await nc.invoker.chat()

    _run(go())


def test_invoker_stream_and_invoke_tool_also_raise() -> None:
    """Same defensive failure mode on the other invoker entry points —
    callers that wander into the middleware-chain seam without a real
    context get a clear, directive error instead of a `None` attribute
    error several frames down."""

    async def go() -> None:
        nc = NullCtx()
        with pytest.raises(RuntimeError):
            await nc.invoker.stream()
        with pytest.raises(RuntimeError):
            await nc.invoker.invoke_tool()

    _run(go())


# ── cooperative control: cancellation, semaphore, child, emit ─────────────


def test_check_cancelled_is_a_noop() -> None:
    """No token attached — cancellation can never trip on a NullCtx."""
    nc = NullCtx()
    assert nc.check_cancelled() is None


def test_semaphore_grants_immediately() -> None:
    """`gather_bounded` uses `async with ctx.semaphore():` — the null
    semaphore enters and exits without serialising; multiple concurrent
    acquires succeed simultaneously."""

    async def go() -> None:
        nc = NullCtx()
        sem = nc.semaphore()
        # Two concurrent holders both succeed — there's no real bound.
        async with sem:
            async with sem:
                pass

    _run(go())


def test_child_returns_self_no_isolation() -> None:
    """Real `RunContext.child()` bumps depth + shares services. Here
    there's nothing to bump and no per-task state, so the same instance
    is its own child — saving an allocation and making depth tracking
    moot."""
    nc = NullCtx()
    assert nc.child() is nc


def test_emit_drops_the_observation_on_the_floor() -> None:
    """No observer is wired in. `emit(...)` must be a no-op coroutine —
    never raise into the run, regardless of the kind/payload/agent
    passed."""

    async def go() -> None:
        nc = NullCtx()
        assert await nc.emit("progress") is None
        assert await nc.emit(
            "summary", "render text", payload={"x": 1}, agent="a", parent_id="p"
        ) is None

    _run(go())


# ── the load-bearing integration test ──────────────────────────────────────


def test_request_builder_build_accepts_null_ctx_without_type_ignore() -> None:
    """The whole point of this module: a `RequestBuilder.build()` call site
    can pass `NullCtx()` *without* a `# type: ignore[arg-type]`. This
    test exercises the runtime; the static-type guarantee comes from
    mypy reading the planner.py call site (which no longer carries the
    ignore comment after this refactor).

    If this test ever needs to suppress a typing warning, NullCtx no
    longer satisfies the Ctx Protocol and the refactor has regressed."""

    async def go() -> None:
        prompt = Prompt(id="test.seed", version="v-null-ctx", template="You are a test agent.")
        wc = WorkingContext()
        req = await RequestBuilder(prompt=prompt).build("hello", wc, NullCtx())
        # RequestBuilder recorded a span under "compose" and laid down the
        # expected system + user transcript — proving the null trace
        # and the null ctx threaded through correctly.
        assert [m.role for m in req.messages] == ["system", "user"]
        assert req.prompt_version == "v-null-ctx"

    _run(go())
