"""tracing middleware — opens the span for a unit of work and stamps gen_ai.* from the result.

A single place that owns span lifecycle for every chat/tool call, so individual patterns never open
spans by hand. Span NAMES align with the OTel GenAI v1.41 operation taxonomy (``chat`` /
``execute_tool``); the kernel stays OTel-free — only the names + attribute keys are aligned so an
adapter can lift them into a real OTel exporter without translation.

Replay side channel. When a ``ReplayStore`` is wired on the run's Services, the chat path writes a
``ReplayRecord`` keyed by the chat span's id around each call. Attribute cardinality stays under
the OTel SDK budget; the full prompt + assembled result land in the side store. The default
``NoopReplayStore`` makes this a zero-cost no-op when nothing is configured.

Metrics + sampling. Before opening the chat / execute_tool span we consult ``ctx.sampler`` —
when ``should_sample`` returns False the span (and its replay write) is skipped entirely; the
underlying handler still runs so the result keeps flowing through. After stream-end on a chat
call we record OTel-aligned histograms on ``ctx.metrics`` (input/output token usage, operation
duration); on the error path we increment ``gen_ai.client.error.count``. All metrics + sampler
calls are wrapped in ``contextlib.suppress(Exception)`` so a misbehaving impl never breaks the run.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agentkit.kernel.middleware import Call, Handler, Middleware
from agentkit.kernel.replay import ReplayRecord
from agentkit.kernel.types import assemble_deltas

_LOGGER = logging.getLogger(__name__)
# One-shot log flag — a misbehaving ``TracePort.span`` should never spam
# the log per unit of work. First failure gets the traceback; subsequent
# ones are silently absorbed so the tracer stays inert as far as the run
# is concerned.
_span_error_logged = False


def _log_span_error_once(where: str) -> None:
    global _span_error_logged
    if not _span_error_logged:
        _span_error_logged = True
        _LOGGER.warning(
            "tracing: TracePort.span raised in %s; continuing without a span", where, exc_info=True
        )


@contextlib.contextmanager
def _safe_trace_span(trace: Any, name: str, kind: str, attrs: dict[str, Any]) -> Iterator[Any]:
    """Best-effort span open + close. On any exception from
    ``trace.span(...)`` (open, ``__enter__``, or ``__exit__``) yield a
    ``NoopSpan`` and continue — the tracer is observability only and
    must never break the run. Every other observability seam in this
    middleware (metrics, sampler, replay, per-message events) is already
    wrapped in ``contextlib.suppress``; this closes the last gap.
    ``ExitStack`` gives us a single ``close()`` that either runs the
    real context manager's cleanup or is a no-op, so the exit path
    stays uniform whether we opened a real span or a noop.

    **The exception is forwarded to the span.** ``ExitStack.close()``
    is defined as ``__exit__(None, None, None)`` — it tells the wrapped
    context manager the block succeeded, whatever actually happened. A
    span CM is precisely the collaborator that needs the truth:
    ``adapters/observability/otel.py`` opens ``start_as_current_span``
    and relies on OTel's own ``__exit__`` to call ``record_exception``
    and set the ERROR status. Closing with a clean exit meant a
    provider failure was exported as a SUCCESSFUL span. Measured: the
    provider raised ``RuntimeError("provider 500")`` and the span came
    out with ``exception=None, status=None``. So the error path exits
    the stack with the live ``(type, value, traceback)`` instead.

    Two deliberate asymmetries:

    * The original exception is ALWAYS re-raised, even if the span CM's
      ``__exit__`` returns True. A tracer is observability only; letting
      it swallow a provider fault would be a far worse bug than the one
      being fixed here.
    * Only ``Exception`` is reported as a span error. ``GeneratorExit``
      / ``CancelledError`` reach this frame when a consumer abandons or
      cancels the stream, which is not a provider fault — those close
      the span cleanly.
    """
    from agentkit.runtime.context import NoopSpan

    stack = contextlib.ExitStack()
    span: Any
    if trace is None:
        # Not a misbehaving tracer — just no tracer wired (a caller reading
        # ``getattr(run, "trace", None)``). Skip the warning path entirely so
        # an untraced run doesn't log as if something broke.
        span = NoopSpan()
    else:
        try:
            span = stack.enter_context(trace.span(name, kind, **attrs))
        except Exception:
            _log_span_error_once("open")
            span = NoopSpan()
    closed = False
    try:
        yield span
    except Exception as exc:
        closed = True
        try:
            stack.__exit__(type(exc), exc, exc.__traceback__)
        except BaseException as close_exc:  # noqa: BLE001 — the tracer must stay inert
            # A ``@contextmanager``-based span re-raises the exception we
            # threw in; that is normal CM behaviour, not a tracer fault, so
            # only a DIFFERENT exception counts as the tracer misbehaving.
            if close_exc is not exc:
                _log_span_error_once("close")
        raise
    finally:
        if not closed:
            try:
                stack.close()
            except Exception:
                _log_span_error_once("close")


# Cap the cardinality of the ``gen_ai.response.tool_calls`` attribute — a tool-heavy
# turn with N calls would otherwise pin N tool names into a single span attribute,
# blowing past the OTel SDK's attribute-length budget on adversarial loops. 16 names
# covers the 99th percentile of real turns; beyond that we suffix ``,+K more`` so the
# attribute remains useful for at-a-glance reads and the raw count lives on the
# separate integer attribute ``gen_ai.response.tool_calls_count`` for histograms.
MAX_TOOL_NAMES_IN_ATTR = 16

# OTel GenAI semconv recommends per-event content caps — keep messages readable
# in the trace UI while staying inside the SDK's per-attribute budget. 8 KB is the
# upstream guidance for ``gen_ai.{system,user,assistant,tool}.message`` content.
MAX_MESSAGE_CONTENT_BYTES = 8 * 1024
# Tool returns can be HTML/JSON blobs — keep them readable but tighter than chat
# messages so a verbose scraper doesn't dominate the span's attribute budget.
MAX_TOOL_RESULT_CONTENT_BYTES = 4 * 1024
# Provider-specific cached-input price ratio. Both OpenAI + Anthropic bill cached
# prompt-cache reads at ~10% of the full input rate; the savings attribute is the
# delta against the un-cached price.
CACHED_INPUT_PRICE_RATIO = 0.1


def _truncate_content(text: str, max_bytes: int = MAX_MESSAGE_CONTENT_BYTES) -> str:
    """Truncate to ``max_bytes`` UTF-8 bytes. Adds ``[truncated]`` suffix when cut.

    OTel collectors enforce per-attribute length caps; over-long values are
    silently dropped. Truncating in our code lets us preserve the leading
    content (which is what an operator wants to see) and signal the cut.

    PII / redaction is NOT done here — operators configure collector-side
    redaction (e.g. OTel collector's ``processor/attributes`` regex rules)."""
    if text is None:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    suffix = b" [truncated]"
    head = encoded[: max_bytes - len(suffix)].decode("utf-8", errors="ignore")
    return head + " [truncated]"


def _estimate_message_tokens(content: str) -> int:
    """Approx token count for one message. chars/4 heuristic — keep cheap.

    Per-message tokens are useful for finding the heavy-weight turn in a
    long conversation without summing the whole transcript. The exact
    provider count lands on the chat span as ``gen_ai.usage.input_tokens``;
    this is the breakdown."""
    if not content:
        return 0
    return max(1, len(content) // 4)


def _serialize_tool_result(value: Any) -> str:
    """Best-effort string of a tool's return value. Prefer JSON when
    serializable (operator-readable diff), str() otherwise. Never
    raises — a tool returning a non-serializable object is normal."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    try:
        from agentkit.kernel._json import dumps as _json_dumps

        if isinstance(value, dict | list | tuple):
            return _json_dumps(value, default=str)
        return str(value)
    except Exception:  # noqa: BLE001 — never raise on serialization
        return str(value)


def _format_tool_calls(names: list[str]) -> str:
    if len(names) <= MAX_TOOL_NAMES_IN_ATTR:
        return ",".join(names)
    head = ",".join(names[:MAX_TOOL_NAMES_IN_ATTR])
    return f"{head},+{len(names) - MAX_TOOL_NAMES_IN_ATTR} more"


def _sampler_keeps(ctx: Any, *, operation: str, attrs: dict[str, Any]) -> bool:
    """Best-effort sampler check. A misbehaving sampler must never break
    the run — on exception we default to KEEPING the span (the safer
    failure mode: more spans, not silent loss)."""
    sampler = getattr(ctx, "sampler", None)
    if sampler is None:
        return True
    try:
        return bool(
            sampler.should_sample(
                operation=operation,
                correlation_id=ctx.correlation_id,
                attrs=attrs or None,
            )
        )
    except Exception:
        return True


def _safe_metric(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Wrap a metrics call so a misbehaving impl never breaks the run."""
    with contextlib.suppress(Exception):
        fn(*args, **kwargs)


def _emit_chat_metrics(ctx: Any, tags: dict[str, str], usage: Any, duration_ms: float) -> None:
    """Record the chat-path OTel GenAI histograms (token usage in/out and
    operation duration). Extracted so both the kept-span path and the
    sampler-rejected fast path emit the SAME metrics. Sampling controls
    trace fidelity; metrics roll up unconditionally."""
    metrics = getattr(ctx, "metrics", None)
    if metrics is None or usage is None:
        return
    if getattr(usage, "input_tokens", 0):
        _safe_metric(
            metrics.record_histogram,
            "gen_ai.client.token.usage",
            usage.input_tokens,
            tags={**tags, "gen_ai.token.type": "input"},
        )
    if getattr(usage, "output_tokens", 0):
        _safe_metric(
            metrics.record_histogram,
            "gen_ai.client.token.usage",
            usage.output_tokens,
            tags={**tags, "gen_ai.token.type": "output"},
        )
    _safe_metric(
        metrics.record_histogram,
        "gen_ai.client.operation.duration",
        duration_ms / 1000.0,
        tags=tags,
    )


def _emit_error_metric(ctx: Any, tags: dict[str, str], exc: BaseException) -> None:
    """Record the error counter. Same extraction rationale as
    ``_emit_chat_metrics``: both span paths share the metric work."""
    metrics = getattr(ctx, "metrics", None)
    if metrics is None:
        return
    _safe_metric(
        metrics.add_counter,
        "gen_ai.client.error.count",
        1,
        tags={**tags, "error.type": type(exc).__name__},
    )


def tracing() -> Middleware:
    async def mw(call: Call, nxt: Handler) -> AsyncIterator[Any]:
        ctx = call.ctx
        if call.kind == "chat":
            name, kind, attrs = "chat", "chat", {"gen_ai.request.model": call.request.model}
        else:
            name, kind, attrs = (
                "execute_tool",
                "execute_tool",
                {"gen_ai.tool.name": call.request.name},
            )

        # Build the low-cardinality tag set used by every metric this
        # middleware records. Kept tiny on purpose — model + (small,
        # known) org_id; everything else explodes the time-series count.
        # Built BEFORE the sampler check so the rejected-span fast path
        # can still emit metrics with the same tags. A sampler short-circuit
        # that skipped this block AND the metrics block would scale all
        # histograms by the sample rate.
        scope = getattr(ctx, "scope", None)
        org_id = getattr(scope, "org_id", None) if scope is not None else None
        if call.kind == "chat":
            metric_tags: dict[str, str] = {
                "gen_ai.request.model": str(call.request.model),
                "agentkit.scope.org_id": str(org_id) if org_id is not None else "none",
            }
        else:
            metric_tags = {
                "gen_ai.tool.name": str(call.request.name),
                "agentkit.scope.org_id": str(org_id) if org_id is not None else "none",
            }

        # Sampler short-circuit: when the head sampler drops this span,
        # skip the span / replay / per-message events for this call. The
        # METRICS layer still emits — sampling controls trace fidelity,
        # not aggregate observability. Without this split, dashboards
        # would under-count token cost / duration / error rate by exactly
        # the sample rate.
        if not _sampler_keeps(ctx, operation=name, attrs=attrs):
            start = time.perf_counter()
            deltas: list[Any] = []
            try:
                async for item in nxt(call):
                    if call.kind == "chat":
                        deltas.append(item)
                    yield item
            except Exception as exc:  # noqa: BLE001 — record-then-rethrow
                _emit_error_metric(ctx, metric_tags, exc)
                raise
            if call.kind == "chat" and deltas:
                result = assemble_deltas(deltas)
                duration_ms = (time.perf_counter() - start) * 1000
                _emit_chat_metrics(ctx, metric_tags, result.usage, duration_ms)
            return

        with _safe_trace_span(ctx.trace, name, kind, attrs) as span:
            deltas = []
            # Capture the span id eagerly — replay writes after the stream
            # need it to key the record, even if the tracer is a no-op
            # (current_span_id returns None then; the suppress block below
            # still skips the write cleanly).
            span_ctx = None
            with contextlib.suppress(Exception):
                span_ctx = ctx.trace.current_span_id()
            # Per-message events on chat spans (OTel GenAI v1.41 ``gen_ai.{role}.message``).
            # Drop one event per request message BEFORE the LLM call so the trace
            # shows what we sent — including the system prompt and any tool-result
            # turns that re-entered. Wrapped in suppress so a misbehaving tracer
            # never breaks the run.
            if call.kind == "chat":
                with contextlib.suppress(Exception):
                    for msg in getattr(call.request, "messages", None) or ():
                        role = getattr(msg, "role", "user") or "user"
                        content = getattr(msg, "content", "") or ""
                        span.add_event(
                            f"gen_ai.{role}.message",
                            **{
                                "gen_ai.message.role": role,
                                "gen_ai.message.content": _truncate_content(content),
                                "gen_ai.message.tokens": _estimate_message_tokens(content),
                            },
                        )
            # Monotonic clock for span timing — perf_counter() is immune to wall-clock
            # adjustments (NTP slews, DST) that would otherwise produce negative or
            # absurd durations on long-running streams.
            start = time.perf_counter()
            first_token_stamped = False
            # Capture the tool result for the post-stream ``gen_ai.tool.result`` event
            # (D2). The tool path is a one-item stream — yield identity-through but
            # remember the value so we can serialize + size it after stream-end.
            last_tool_result: Any = None
            try:
                async for item in nxt(call):  # span stays open across the stream; pass through
                    if call.kind == "chat":
                        deltas.append(item)
                        # First non-empty content delta marks the model's first token.
                        # ``gen_ai.response.first_token_latency_ms`` is undefined for
                        # streams that never produce content (tool-only turns, empty
                        # responses) — leave the attribute absent so backends show
                        # "no value" rather than a misleading 0.
                        if not first_token_stamped and getattr(item, "text", False):
                            span.set(
                                "gen_ai.response.first_token_latency_ms",
                                (time.perf_counter() - start) * 1000,
                            )
                            first_token_stamped = True
                    else:
                        last_tool_result = item
                    yield item
            except Exception as exc:  # noqa: BLE001 — record-then-rethrow
                # Error counter: low-cardinality (model + exception class
                # name). Production dashboards use this to alert on a
                # provider 5xx storm without joining traces. Funnelled
                # through the shared helper so kept-span and rejected-span
                # paths emit identically.
                _emit_error_metric(ctx, metric_tags, exc)
                raise
            if (
                call.kind == "chat" and deltas
            ):  # stamp gen_ai.* from the assembled result at stream-end
                result = assemble_deltas(deltas)
                usage = result.usage
                span.set("gen_ai.usage.input_tokens", usage.input_tokens)
                span.set("gen_ai.usage.output_tokens", usage.output_tokens)
                span.set("gen_ai.usage.cost_usd", usage.cost_usd)
                # Prompt-cache visibility — stamp only when the provider reported a
                # non-zero value so the 95% of calls without caching don't carry
                # noisy zeros.
                if usage.cache_read_tokens:
                    span.set("gen_ai.usage.cache_read_tokens", usage.cache_read_tokens)
                    # Cache savings: cached input is billed at ~10% of full input
                    # rate, so the absolute saved-USD is the cached tokens times
                    # the per-token full price times the 1-ratio gap. Only stamp
                    # when we can derive a per-token rate (input_tokens > 0); a
                    # zero-input call would divide by zero.
                    if usage.input_tokens > 0 and usage.cost_usd > 0:
                        per_token = usage.cost_usd / usage.input_tokens
                        saved = usage.cache_read_tokens * per_token * (1 - CACHED_INPUT_PRICE_RATIO)
                        span.set("gen_ai.usage.cost_saved_by_cache_usd", saved)
                if usage.cache_write_tokens:
                    span.set("gen_ai.usage.cache_write_tokens", usage.cache_write_tokens)
                span.set("gen_ai.response.model", result.model)
                span.set("gen_ai.response.finish_reasons", result.finish_reason)
                if result.tool_calls:
                    names = [t.name for t in result.tool_calls]
                    span.set("gen_ai.response.tool_calls", _format_tool_calls(names))
                    span.set("gen_ai.response.tool_calls_count", len(names))
                # Total turn duration (monotonic) — independent of first-token
                # latency so a backend can derive token-to-token throughput.
                duration_ms = (time.perf_counter() - start) * 1000
                span.set("gen_ai.response.duration_ms", duration_ms)

                # Assistant turn event (OTel GenAI v1.41 ``gen_ai.choice``).
                # One event per chat call, summarising the assembled reply. Carries
                # the (truncated) content, the finish reason, and the per-turn
                # output token count — distinct from the input-side per-message
                # events emitted before the LLM call.
                with contextlib.suppress(Exception):
                    span.add_event(
                        "gen_ai.choice",
                        **{
                            "gen_ai.choice.role": "assistant",
                            "gen_ai.choice.content": _truncate_content(result.content or ""),
                            "gen_ai.choice.finish_reason": result.finish_reason or "",
                            "gen_ai.choice.tokens": (
                                usage.output_tokens if usage is not None else 0
                            ),
                        },
                    )

                # Metrics — OTel GenAI semconv histograms. ``token.usage``
                # split into input + output series via the
                # ``gen_ai.token.type`` tag so a single histogram name
                # serves both. Duration in SECONDS per OTel convention.
                # Shared helper so kept-span and rejected-span
                # paths emit IDENTICAL metrics.
                _emit_chat_metrics(ctx, metric_tags, usage, duration_ms)

            elif call.kind == "tool":
                # Tool-result event (D2): one event per execute_tool span carrying
                # the (truncated) result body, its un-truncated byte size, and the
                # tool's wall-clock duration. Truncation cap is tighter than chat
                # messages (4 KB vs 8 KB) — tool outputs can be HTML/JSON blobs
                # that would otherwise dominate the span's attribute budget. The
                # raw size is kept un-capped so an operator can still see the
                # actual blob size even when content is truncated.
                duration_ms = (time.perf_counter() - start) * 1000
                with contextlib.suppress(Exception):
                    result_repr = _serialize_tool_result(last_tool_result)
                    span.add_event(
                        "gen_ai.tool.result",
                        **{
                            "gen_ai.tool.name": call.request.name,
                            "gen_ai.tool.result_content": _truncate_content(
                                result_repr, max_bytes=MAX_TOOL_RESULT_CONTENT_BYTES
                            ),
                            "gen_ai.tool.result_size_bytes": len(result_repr.encode("utf-8")),
                            "gen_ai.tool.duration_ms": duration_ms,
                        },
                    )

            if call.kind == "chat" and deltas:
                # Replay side channel: write the full request/response keyed
                # by the chat span id so downstream tooling can rebuild the
                # call without re-running it. Best-effort — a failed write
                # must NEVER break the run, and a no-op store costs nothing.
                if span_ctx is not None:
                    with contextlib.suppress(Exception):
                        await ctx.replay.put(
                            ReplayRecord(
                                span_id=span_ctx.span_id,
                                operation="chat",
                                request={
                                    "messages": list(call.request.messages),
                                    "model": call.request.model,
                                    "tools": call.request.tools,
                                    "response_format": call.request.response_format,
                                    "temperature": call.request.temperature,
                                    "max_tokens": call.request.max_tokens,
                                },
                                response=result,
                            )
                        )

    return mw
