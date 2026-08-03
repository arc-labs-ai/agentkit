"""The ONE interception model — every cross-cutting concern is a middleware around a unit of work.

Both core operations (a chat turn, a tool execution) are a `Call`; the `Invoker` runs each through a
composed chain whose terminal is the actual seam call. A middleware wraps `next`, so it can act
before, after, around, or instead of the rest of the chain (retry re-invokes next; fallback rewrites
the request and re-invokes; memoize may skip next entirely). Cross-cutting concerns like quota,
tracing, retry, budgeting, egress checks, idempotency, and audit are each their own middleware, and
their order is an explicit list the app owns rather than a single hardcoded method.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from agentkit.kernel.observation import Observation
from agentkit.kernel.protocols import AutonomyLiteral
from agentkit.kernel.types import Delta, Operation, assemble_deltas

_KIND_TO_OP = {"chat": Operation.MODEL_CALL, "tool": Operation.TOOL_CALL}


@dataclass
class Call:
    kind: str  # "chat" | "tool"
    request: Any  # ChatRequest | ToolRequest (mutable — middlewares may rewrite it)
    ctx: Any  # RunContext
    meta: dict[str, Any] = field(
        default_factory=dict
    )  # per-call signals between middlewares (e.g. memoize→audit hit)


# ONE stream-shaped contract: a handler yields the operation's items — `Delta`s for a chat,
# exactly one result item for a tool. A middleware is an async generator wrapping `next`.
Handler = Callable[[Call], "AsyncIterator[Any]"]
Middleware = Callable[[Call, Handler], "AsyncIterator[Any]"]


async def collect(stream: AsyncIterator[Any], kind: str = "chat") -> Any:
    """Reduce a stream to the operation's result — a chat result is just its collected stream. A chat
    assembles its `Delta`s into an `LLMResult`; any other op returns its single (last) yielded item."""
    if kind == "chat":
        deltas: list[Delta] = [d async for d in stream]
        return assemble_deltas(deltas)
    last = None
    async for item in stream:
        last = item
    return last


async def collect_one(stream: AsyncIterator[Any]) -> Any:
    """The single item of a one-item (tool) stream."""
    return await collect(stream, kind="tool")


def _assemble(call: Call, items: list[Any]) -> Any:
    return assemble_deltas(items) if call.kind == "chat" else (items[-1] if items else None)


def _result_to_stream(call: Call, result: Any) -> list[Any]:
    """Re-express an already-assembled result as stream items (for cache hits / error recovery), so a
    one-shot value re-enters the single streaming contract. A chat result → one terminal `Delta`."""
    if call.kind != "chat" or result is None:
        return [result]
    return [
        Delta(
            text=getattr(result, "content", "") or "",
            tool_calls=getattr(result, "tool_calls", ()),
            usage=getattr(result, "usage", None),
            finish_reason=getattr(result, "finish_reason", None),
            model=getattr(result, "model", None),
            provider=getattr(result, "provider", None),
        )
    ]


class MiddlewareContext:
    """The rich, ergonomic view a middleware sees — a thin facade over the `Call` + its `RunContext`, so a
    middleware reads `ctx.messages` / `ctx.prompt` / `ctx.budget` / `ctx.vector` rather than digging through
    `call.request` and `call.ctx`. The request is mutable (`ctx.request = …` rewrites it before the op)."""

    def __init__(self, call: Call) -> None:
        self._call = call

    @property
    def call(self) -> Call:  # the underlying unit of work (for collaborators that need it)
        return self._call

    @property
    def kind(self) -> str:  # raw internal kind: "chat" | "tool"
        return self._call.kind

    @property
    def operation(self) -> Operation:  # the typed operation to match on (MODEL_CALL | TOOL_CALL)
        return _KIND_TO_OP.get(self._call.kind, Operation.MODEL_CALL)

    @property
    def data(self) -> Any:  # the operation's payload: messages (model) | arguments (tool)
        if self.operation == Operation.MODEL_CALL:
            return self.messages
        return getattr(self._call.request, "arguments", None)

    @property
    def request(self) -> Any:
        return self._call.request

    @request.setter
    def request(self, value: Any) -> None:  # rewrite the unit of work before it runs
        self._call.request = value

    @property
    def messages(self) -> list[Any]:  # the chat transcript (empty for a tool call) — READ-only copy
        # A copy: read it freely, but to REWRITE the transcript assign a new request via `ctx.request = …`
        # (the writable seam). Mutating this list in place does not change the unit of work.
        return list(getattr(self._call.request, "messages", []) or [])

    @property
    def prompt(self) -> str:  # the latest user/task prompt
        for m in reversed(self.messages):
            if getattr(m, "role", None) == "user":
                return m.content or ""
        return ""

    @property
    def model(self) -> Any:
        return getattr(self._call.request, "model", None)

    @property
    def run(self) -> Any:  # the RunContext (identity + meters + services)
        return self._call.ctx

    @property
    def budget(self) -> Any:
        return self._call.ctx.budget

    @property
    def scope(self) -> Any:
        return self._call.ctx.scope

    @property
    def autonomy(self) -> AutonomyLiteral:
        result: AutonomyLiteral = self._call.ctx.autonomy
        return result

    @property
    def vector(self) -> Any:  # the memory/RAG seam (VectorPort), tenant-scoped
        return self._call.ctx.vector

    @property
    def store(self) -> Any:
        return self._call.ctx.store

    async def emit(self, kind: str, render: str = "", **payload: Any) -> None:
        """Emit a product-facing event on the run's observer stream (never raises into the run)."""
        await self._call.ctx.emit(kind, render, payload=payload or None)


class Blocked(Exception):
    """Raised by a middleware's `on_request` to refuse a unit of work before it runs (a guard/policy stop) —
    e.g. detected prompt injection. Propagates as a typed refusal, distinct from a model/tool fault."""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


class BaseMiddleware:
    """The ergonomic, class-based middleware — the **transform / guard / observe** layer.

    Override only the phases you need. Each phase is `async` and may be a plain coroutine OR an async
    generator — both are supported:

        on_request(ctx)          # before: mutate ctx.request; raise Blocked to refuse; (yield/emit events)
        on_response(ctx, result) # after success: return (or yield) a transformed result; default passes through
        on_error(ctx, exc)       # on failure: return/yield a value to recover, else raise (default re-raises)

    Emit a product-facing event either by `await ctx.emit(...)` or by `yield`-ing an `Observation` (the
    async-generator form). `on_response` returning/yielding a non-`Observation` value sets the result.

    `as_middleware()` compiles these to the `(call, next)` primitive, wrapping `next` in one try/except so
    `on_request` is always paired with exactly one `on_response` OR `on_error` — a balanced invariant that
    standalone phase hooks tend to break. This layer **cannot** re-invoke or rewrite-and-re-run `next`;
    retry / fallback / memoize / circuit-breaking are a separate *resilience/caching* concern written as
    raw `(call, next)` middlewares, not a `BaseMiddleware`.
    """

    buffers: bool = (
        False  # False → pass deltas through + OBSERVE the result (incremental; on_response's
    )
    #                          return is ignored). True → collect the inner stream so on_response can
    #                          TRANSFORM the result, then re-emit it (not incremental).

    async def on_request(self, ctx: MiddlewareContext) -> Any: ...  # default: proceed, no change

    async def on_response(self, ctx: MiddlewareContext, result: Any) -> Any:
        return result  # default: pass the result through unchanged

    async def on_error(self, ctx: MiddlewareContext, exc: Exception) -> Any:
        raise exc  # default: propagate

    def as_middleware(self) -> Middleware:
        async def mw(call: Call, nxt: Handler) -> AsyncIterator[Any]:
            mctx = MiddlewareContext(call)
            await _drive(
                call, self.on_request(mctx), default=None
            )  # before the stream; raise Blocked to refuse
            items: list[Any] = []
            try:
                if self.buffers:  # collect first (no incremental streaming)
                    async for item in nxt(call):
                        items.append(item)
                else:
                    async for item in nxt(call):  # PASS THROUGH incrementally (observe)
                        items.append(item)
                        yield item
            except Exception as exc:  # noqa: BLE001 — on_error decides
                recovered = await _drive(call, self.on_error(mctx, exc), default=_MISSING)
                if recovered is _MISSING:
                    raise  # (a raising on_error already propagated)
                for it in _result_to_stream(
                    call, recovered
                ):  # recover a PRE-stream failure cleanly
                    yield it
                return
            assembled = _assemble(call, items)
            transformed = await _drive(call, self.on_response(mctx, assembled), default=assembled)
            if self.buffers:  # buffered → on_response may have transformed
                for it in _result_to_stream(call, transformed):
                    yield it
            # else: deltas already streamed; on_response was observe-only (return ignored, side-effects applied)

        return mw


_MISSING = object()


async def _drive(call: Call, phase: Any, *, default: Any) -> Any:
    """Run a phase that is either a coroutine or an async generator. `Observation`s are emitted on the run's
    observer; the last non-`Observation` value produced becomes the return (else `default`)."""
    value = default
    if inspect.isasyncgen(phase):
        async for item in phase:
            if isinstance(item, Observation):
                await call.ctx.observer.emit(item)
            else:
                value = item
    else:
        out = await phase
        if isinstance(out, Observation):
            await call.ctx.observer.emit(out)
        elif out is not None:
            value = out
    return value


def chain(middlewares: list[Middleware | BaseMiddleware], terminal: Handler) -> Handler:
    """Fold the chain right so `middlewares[0]` is outermost: m0(m1(… terminal)).

    Accepts raw `Middleware` functions and/or `BaseMiddleware` instances (adapted via `as_middleware()`), so
    the two styles mix freely in one chain. Each is an async generator over the single streaming contract;
    the terminal is the seam stream (`llm.stream(**req)` / a one-item tool stream). An empty list returns
    the terminal unchanged.
    """
    handler = terminal
    for mw in reversed(list(middlewares)):
        fn = mw.as_middleware() if isinstance(mw, BaseMiddleware) else mw
        handler = _bind(fn, handler)
    return handler


def _bind(mw: Middleware, nxt: Handler) -> Handler:
    def handler(call: Call) -> AsyncIterator[Any]:
        return mw(call, nxt)  # mw is an async-gen fn → returns the async generator

    return handler
