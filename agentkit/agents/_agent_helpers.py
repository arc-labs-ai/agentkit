"""Stateless helpers for the agent run loop."""

from __future__ import annotations

from typing import Any

from agentkit.agents.result import AgentResult
from agentkit.capabilities.output_schema import SchemaAdapter
from agentkit.kernel._json import JSONDecodeError as _JSONDecodeError
from agentkit.kernel._json import dumps as _json_dumps
from agentkit.kernel._json import loads as _json_loads
from agentkit.kernel.types import Message, StreamEvent, Usage


def _assistant(res: Any) -> Message:
    return Message("assistant", content=res.content, tool_calls=res.tool_calls)


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return _json_dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _parse_args(decision: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Returns ``fallback`` on parse failure or non-dict result so a
    malformed model call doesn't crash the dispatcher."""
    try:
        parsed = _json_loads(decision)
        return parsed if isinstance(parsed, dict) else fallback
    except (_JSONDecodeError, TypeError):
        return fallback


def _stamp_terminal_span_attrs(span: Any, result: AgentResult) -> None:
    """Stamp per-agent rollups on the ``invoke_agent`` span just before close.

    ``gen_ai.usage.*_subtree`` aggregates the AgentResult's cumulative ``Usage``
    so an operator can answer "what did this agent + descendants cost?" without
    summing nested ``chat`` spans by hand. ``_subtree`` distinguishes it from
    per-call ``gen_ai.usage.*`` written by the tracing middleware. Cache fields
    only land when non-zero to keep low-signal attributes off the wire.

    ``gen_ai.system.prompt_version`` mirrors ``AgentResult.prompt_version``
    (set by the RequestBuilder) so a trace shows which template produced the
    output without inspecting the result object.

    ``span`` may be a ``NoopSpan`` (no-tracer default) — ``.set`` is a no-op
    there, so guarding on truthiness is enough.
    """
    if span is None:
        return
    if result.usage is not None:
        span.set("gen_ai.usage.input_tokens_subtree", result.usage.input_tokens)
        span.set("gen_ai.usage.output_tokens_subtree", result.usage.output_tokens)
        span.set("gen_ai.usage.cost_usd_subtree", result.usage.cost_usd)
        if result.usage.cache_read_tokens:
            span.set("gen_ai.usage.cache_read_tokens_subtree", result.usage.cache_read_tokens)
        if result.usage.cache_write_tokens:
            span.set("gen_ai.usage.cache_write_tokens_subtree", result.usage.cache_write_tokens)
    if result.prompt_version:
        span.set("gen_ai.system.prompt_version", result.prompt_version)


def _final_events(
    content: str,
    usage: Usage,
    *,
    partial: bool,
    reason: str,
    prompt_version: str,
    error: str | None = None,
) -> list[StreamEvent]:
    evals: dict[str, Any] = {"stop_reason": reason}
    if error is not None:
        evals["error"] = error
    result = AgentResult(
        output=content,
        usage=usage,
        partial=partial,
        evals=evals,
        prompt_version=prompt_version,
    )
    return [StreamEvent("final", result=result, usage=usage)]


def _infer_response_format(model: str | None, adapter: SchemaAdapter[Any]) -> dict[str, Any] | None:
    """Best-effort provider-native structured-output wiring from the model name.

    Conservative — sets ``response_format`` only when we're confident the
    provider adapter forwards it without surprises. Everything else falls
    back to prompt-injection (the schema block in ``PrefixContext.schema_block``
    is already present), which is provider-agnostic and never wrong.

    ``gpt-*`` → OpenAI ``json_schema`` mode with ``strict=True``. Anthropic
    ``tool_use`` wiring is deferred (invasive on the provider adapter side).
    """
    if not model:
        return None
    schema = adapter.json_schema()
    if model.startswith("gpt-"):
        return {
            "type": "json_schema",
            "json_schema": {"name": adapter.name, "schema": schema, "strict": True},
        }
    return None
