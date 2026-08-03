"""``output_coerce`` middleware — attach the typed parsed output to ``LLMResult``.

The Agent stamps the active ``SchemaAdapter`` for the run on the
``RunContext`` (``ctx._output_adapter``) before invoking the chat,
and on the per-call ``Call.meta`` carrier under
``"output_adapter"``. This middleware sits in the chat chain after
the streamed deltas, collects the assembled content, runs
``adapter.parse(content)``, and emits a synthetic terminal ``Delta``
carrying the typed object — ``assemble_deltas`` lifts it onto
``LLMResult.parsed``.

On parse failure the middleware re-raises ``OutputCoercionError``
unmodified so the surrounding retry / parse-and-repair logic (the
Agent's loop, or a ``retry()`` middleware sat further out) can
reflect the validation diagnostics back to the model. Wrapping every
adapter's native exception under a single type means the retry
policy doesn't have to know which flavour is in play.

When no adapter is set on the call, this middleware is a strict no-
op: deltas flow through unchanged and ``LLMResult.parsed`` stays
``None``. The intended chain placement is just outside ``tracing()``
so the span timing covers the parse cost too:

    chat_middleware=[tracing(), output_coerce(), retry(...), ...]

Tool calls (``Operation.TOOL_CALL``) are passed through verbatim — a
tool result is opaque to the schema adapter.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from agentkit.capabilities.output_schema import OutputCoercionError, SchemaAdapter
from agentkit.kernel.middleware import Call, Handler, Middleware
from agentkit.kernel.types import Delta, assemble_deltas


def _emit_coercion_failed(call: Call, exc: OutputCoercionError) -> None:
    """Drop an ``output.coercion_failed`` event on the currently-open span — best-effort.

    Fired before re-raising so the trace timeline carries the parse failure even when the
    outer retry/repair loop catches the exception. A misbehaving tracer must never break
    the run, so the call is wrapped in ``contextlib.suppress(Exception)``."""
    trace = getattr(call.ctx, "trace", None)
    if trace is None:
        return
    with contextlib.suppress(Exception):
        trace.add_event_to_current_span(
            "output.coercion_failed",
            error_type=type(exc).__name__,
            errors_count=len(getattr(exc, "errors", ()) or ()),
        )


def _resolve_adapter(call: Call) -> SchemaAdapter[Any] | None:
    """Pick up the adapter the cognition threaded onto ``call.meta``.

    Single source of truth. The cognition passes
    ``meta={"output_adapter": adapter}`` to ``invoker.stream`` /
    ``invoker.chat`` — the Invoker deposits it on ``Call.meta`` and this
    middleware reads it here. Smuggling the adapter via a private
    attribute on the shared ``RunContext`` would let two agents sharing
    a ctx stomp on each other's adapter, and the ``Ctx`` Protocol never
    declared the field."""
    return call.meta.get("output_adapter") if call.meta else None


def output_coerce() -> Middleware:
    """Build the response-coercion middleware.

    Returns a stream-shaped middleware suitable for the chat chain. A
    factory (rather than a class) keeps the wiring shape uniform with
    the rest of ``agentkit.middlewares`` — every other raw
    ``(call, next)`` middleware is also a function-returning-coroutine.
    """

    async def mw(call: Call, nxt: Handler) -> AsyncIterator[Any]:
        # Tool calls: opaque payload, nothing to coerce — stream through
        # untouched. The kind check is cheap and keeps a misconfigured
        # tool chain from accidentally crashing on a missing ``content``.
        if call.kind != "chat":
            async for item in nxt(call):
                yield item
            return

        adapter = _resolve_adapter(call)
        if adapter is None:
            # Strict no-op fast path — no buffering, deltas pass straight
            # through. Preserves the streaming UX for unstructured calls.
            async for item in nxt(call):
                yield item
            return

        # Adapter is wired: capture deltas as they flow so we can
        # strict-parse the assembled content at end-of-stream, AND
        # opportunistically lift a partial typed object onto each
        # delta as the content accumulates. The streaming UX is
        # otherwise unchanged — we yield every delta verbatim.
        deltas: list[Delta] = []
        accumulated: list[str] = []
        last_partial: Any = None
        async for item in nxt(call):
            # Lift a partial typed object onto the delta when the
            # tolerant parse produces something new. Comparison goes
            # through the type's natural ``__eq__`` (BaseModel,
            # dataclass, attrs, dict — all field-by-field); a
            # comparison that itself raises (e.g. dataclass with
            # un-seeded required fields) we suppress and treat as
            # "different" so the consumer still sees the progress.
            if getattr(item, "text", ""):
                accumulated.append(item.text)
                current = "".join(accumulated)
                try:
                    partial = adapter.partial_parse(current)
                except Exception:
                    partial = None
                if partial is not None:
                    changed = True
                    try:
                        changed = partial != last_partial
                    except Exception:
                        changed = True
                    if changed:
                        # Delta is frozen — rebuild with the partial
                        # attached instead of in-place mutation.
                        item = replace(item, partial=partial)
                        last_partial = partial
            deltas.append(item)
            yield item

        if not deltas:
            return

        assembled = assemble_deltas(deltas)
        try:
            parsed: Any = adapter.parse(assembled.content)
        except OutputCoercionError as exc:
            # Narrative event: pin the parse failure on the trace
            # timeline so the outer retry/repair reaction is legible
            # without cross-referencing structured logs.
            _emit_coercion_failed(call, exc)
            # The retry middleware (or the Agent's parse-and-repair
            # loop) is the right home for the "reflect-and-retry"
            # reaction — re-raise verbatim so neither has to guess
            # whether the failure was a coercion or a transport fault.
            raise

        # Emit a synthetic post-terminal delta carrying ``parsed``.
        # ``assemble_deltas`` propagates it onto ``LLMResult.parsed``
        # without disturbing the streamed content. Also stamp the
        # per-call carrier so any outer middleware (audit, eval) can
        # read the typed result without re-running the adapter.
        call.meta["output_parsed"] = parsed
        yield Delta(parsed=parsed)

    return mw


__all__ = ["output_coerce"]
