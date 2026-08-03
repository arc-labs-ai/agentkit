"""compaction middleware — bound a chat request's transcript before it hits the model.

A ``BaseMiddleware`` (transform): ``on_request`` applies a ``Compactor``
to a chat call's messages. The *policy* (when to fire, what to keep,
how to summarize) lives in the compactor; this just applies it. Tool
calls pass through untouched.

Defensive fallback. A compactor is an optimization; if it produces an
empty message list, the request would hit the provider with zero
messages — 400 / hallucinated response depending on the provider. The
middleware refuses to install a rewrite that would break the run:
when ``compact(...)`` returns an empty list, the original messages
survive and a ``context.compaction_rejected`` event lands on the
surrounding span so the operator sees WHY the reduction didn't apply.

Observability. Each pass is wrapped in a ``context.compact`` span (OTel
GenAI-aligned name) carrying token-budget attributes; a
``context.compacted`` span event lands on the surrounding span with
``messages_before`` / ``messages_after`` so the trace timeline shows
when context was reduced. Both side-effects are best-effort — a
misbehaving tracer can never break the run.

Cache discipline. After the ``agentkit.context`` split, the leading
system messages on a chat call are the *prefix* portion of a
``WorkingContext`` — the cache-stable head. Every built-in
``Compactor`` already preserves a leading system message verbatim, so
this middleware-level call is safe: the prefix is not rewritten in
place. Custom compactors are expected to honour the same rule — the
prefix is read, never compacted, never replaced. Mutating it would
invalidate the provider's KV cache from that token forward.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any

from agentkit.capabilities.compaction import Compactor
from agentkit.kernel.middleware import BaseMiddleware, MiddlewareContext
from agentkit.kernel.types import Operation


def _approx_tokens(messages: list[Any]) -> int:
    """Same heuristic as ``capabilities.compaction._approx_tokens`` — duplicated here
    so the middleware can compute before/after counts for the observability event
    without reaching into the capability's private helper."""
    return sum(len(getattr(m, "content", "") or "") for m in messages) // 4


class Compaction(BaseMiddleware):
    def __init__(self, compactor: Compactor) -> None:
        self._compactor = compactor

    async def on_request(self, ctx: MiddlewareContext) -> None:
        if ctx.operation == Operation.MODEL_CALL:
            # The compactor receives the assembled message list (prefix
            # + tail). Built-in compactors keep the leading system
            # message untouched; custom compactors must do the same to
            # preserve the cache-stable prefix.
            messages_before = list(ctx.request.messages)
            tokens_before = _approx_tokens(messages_before)
            strategy = type(self._compactor).__name__

            run = ctx.run
            trace = getattr(run, "trace", None)
            span_cm = (
                trace.span(
                    "context.compact",
                    "internal",
                    **{
                        "agentkit.context.strategy": strategy,
                        "agentkit.context.tokens_before": tokens_before,
                        "agentkit.context.messages_before": len(messages_before),
                    },
                )
                if trace is not None
                else None
            )

            if span_cm is not None:
                with span_cm as span:
                    compacted = await self._compactor.compact(messages_before, run)
                    # A compactor that returns an empty list would hand
                    # the provider a zero-message request — 400 or
                    # hallucination depending on the vendor. Fall back
                    # to the original messages and stamp the trace so
                    # the operator sees the rejection.
                    if not compacted:
                        compacted = messages_before
                        with contextlib.suppress(Exception):
                            span.set("agentkit.context.compaction_rejected", "empty_result")
                    # ChatRequest is frozen — rewrite via replace,
                    # reassign the mutable request slot on the underlying
                    # Call. The setter on MiddlewareContext.request writes
                    # into ``self._call.request`` for exactly this path.
                    ctx.request = replace(ctx.request, messages=compacted)
                    tokens_after = _approx_tokens(compacted)
                    with contextlib.suppress(Exception):
                        span.set("agentkit.context.tokens_after", tokens_after)
                        span.set("agentkit.context.messages_after", len(compacted))
                        span.set(
                            "agentkit.context.messages_dropped",
                            max(0, len(messages_before) - len(compacted)),
                        )
            else:
                compacted = await self._compactor.compact(messages_before, run)
                if not compacted:
                    # Defensive fallback outside the traced path — same
                    # rationale as the traced branch above: never install
                    # an empty message rewrite that breaks the run.
                    compacted = messages_before
                ctx.request = replace(ctx.request, messages=compacted)
                tokens_after = _approx_tokens(compacted)

            # Drop a span event on the SURROUNDING span (typically `chat`)
            # so the chat timeline reflects the reduction. The dedicated
            # `context.compact` span above is the structural record; this
            # event is the narrative pointer for human readers scanning
            # the chat span's events.
            changed = len(compacted) != len(messages_before) or tokens_after != tokens_before
            if trace is not None and changed:
                with contextlib.suppress(Exception):
                    trace.add_event_to_current_span(
                        "context.compacted",
                        messages_before=len(messages_before),
                        messages_after=len(compacted),
                        tokens_saved=max(0, tokens_before - tokens_after),
                    )


def compaction(compactor: Compactor) -> Compaction:
    return Compaction(compactor)
