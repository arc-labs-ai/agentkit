"""Stateless helpers for the agent run loop."""

from __future__ import annotations

from typing import Any

from agentkit.agents.result import AgentResult, AgentStopReason
from agentkit.capabilities.output_schema import SchemaAdapter
from agentkit.kernel._json import JSONDecodeError as _JSONDecodeError
from agentkit.kernel._json import dumps as _json_dumps
from agentkit.kernel._json import loads as _json_loads
from agentkit.kernel.types import Message, StreamEvent, Usage


def _assistant(res: Any) -> Message:
    return Message("assistant", content=res.content, tool_calls=res.tool_calls)


def _last_assistant(context: Any) -> str:
    """The most recent assistant text in a working context, or ``""``.

    Used by the terminal paths that stop the loop WITHOUT a fresh model
    response in hand (iteration ceiling reached, budget already exhausted at
    loop entry). Returning the last thing the model actually said beats
    returning an empty string — a partial answer is still an answer.
    """
    text: str = next((m.content for m in reversed(context.messages) if m.role == "assistant"), "")
    return text


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
    stop_reason: AgentStopReason | None = None,
    evals_extra: dict[str, Any] | None = None,
) -> list[StreamEvent]:
    """Build the single terminal ``final`` event.

    ``reason`` is the FREE-FORM detail string and keeps landing in
    ``evals["stop_reason"]`` verbatim — a ``TerminationCondition`` names
    itself there, and existing readers/tests depend on it.

    ``stop_reason`` is the CLOSED taxonomy stamped on
    ``AgentResult.stop_reason``. When omitted it is derived from
    ``reason`` via :data:`_REASON_TO_STOP`, which maps every string the
    framework itself passes; anything unrecognised (i.e. a custom
    termination condition's own wording) becomes ``"terminated"``
    rather than guessing. The two never disagree because the derivation
    is total.
    """
    evals: dict[str, Any] = {"stop_reason": reason}
    if error is not None:
        evals["error"] = error
    if evals_extra:
        evals.update(evals_extra)
    result = AgentResult(
        output=content,
        usage=usage,
        partial=partial,
        evals=evals,
        prompt_version=prompt_version,
        stop_reason=stop_reason if stop_reason is not None else _REASON_TO_STOP.get(reason, "terminated"),
    )
    return [StreamEvent("final", result=result, usage=usage)]


# Free-form ``reason`` → closed ``AgentStopReason``. Only the strings the
# framework passes itself are listed; every other value (a custom
# ``TerminationCondition``'s wording) falls through to ``"terminated"``,
# which is the honest answer — *something* stopped the loop deliberately
# and we are not going to invent a more specific category for it.
_REASON_TO_STOP: dict[str, AgentStopReason] = {
    "complete": "complete",
    "awaiting_approval": "suspended",
    "awaiting_input": "suspended",
    "expired": "expired",
    "budget_exhausted": "budget_exhausted",
    "max_iterations": "max_iterations",
    "invalid_output": "invalid_output",
}


_OUTPUT_COERCE_MODULE = "agentkit.middlewares.output_coerce"


def _chain_has_output_coerce(ctx: Any) -> bool:
    """Is ``output_coerce()`` wired into this run's chat middleware chain?

    ``output_coerce()`` returns a closure, so identity comparison against
    the factory is useless; we match on the closure's DEFINING MODULE,
    which is stable across renames of the inner function and cannot
    collide with an application middleware. ``Invoker`` exposes the
    composed list as ``chat_middleware`` precisely so this is answerable.

    Returns ``True`` when we cannot tell (no invoker, or an invoker that
    doesn't expose its chain — a hand-rolled test double). Silence is the
    right default for an unknown wiring: a false "you forgot the
    middleware" warning on a legitimate custom setup is worse than no
    warning at all.
    """
    invoker = getattr(ctx, "invoker", None)
    chain_mws = getattr(invoker, "chat_middleware", None)
    if chain_mws is None:
        return True
    return any(getattr(mw, "__module__", "") == _OUTPUT_COERCE_MODULE for mw in chain_mws)


def _warn_missing_output_coerce(agent_name: str) -> None:
    """Warn that a declared output schema will not produce streamed partials.

    The failure this catches is silent and well-formed: with ``output=``
    set but ``output_coerce()`` absent from the chain, ``AgentResult.parsed``
    STILL works — the cognition calls ``agent.parse`` itself — so nothing
    looks broken. Only ``StreamEvent.partial_output`` is quietly ``None``
    forever, and a UI built to render in-progress objects just never
    updates. That is exactly the class of failure the framework is
    supposed to make loud.

    ``stacklevel=2`` points the warning at the caller's ``stream()`` /
    ``run()`` call rather than at this helper.
    """
    import warnings

    warnings.warn(
        f"agent {agent_name!r} declares an output schema but its chat middleware chain "
        "has no output_coerce() — AgentResult.parsed will still be populated, but "
        "StreamEvent.partial_output will always be None (no in-progress typed object "
        "will be streamed). Add output_coerce() to the chat chain: "
        "Invoker(llm=..., chat_middleware=[tracing(), output_coerce(), retry(...)]).",
        UserWarning,
        stacklevel=2,
    )


def _infer_response_format(model: str | None, adapter: SchemaAdapter[Any]) -> dict[str, Any] | None:
    """Provider-native structured-output wiring, from the model REGISTRY.

    Conservative — sets ``response_format`` only for models declared to
    support the strict server-side mode. Everything else falls back to
    prompt-injection (the schema block in ``PrefixContext.schema_block`` is
    already present), which is provider-agnostic and never wrong.

    This used to be ``model.startswith("gpt-")`` — a name-prefix guess, and
    the precedent Brief 5 exists to remove. It now reads
    ``ModelCapabilities.native_json_schema`` off the registry, so declaring a
    new model with that capability wires it automatically and a model
    declared without it is never sent a ``response_format`` its provider
    would reject.

    The ``gpt-`` prefix survives as a LAST-RESORT fallback for a model the
    registry has never heard of. That is deliberate, and it is the safe
    direction: an unregistered ``gpt-`` name is overwhelmingly an OpenAI
    model, and the cost of a wrong guess here is a provider 400 at wiring
    time, not a plausible empty answer. ``UNKNOWN`` for a non-``gpt-`` name
    yields ``None`` — prompt injection — which always works.
    """
    if not model:
        return None
    schema = adapter.json_schema()
    native = {
        "type": "json_schema",
        "json_schema": {"name": adapter.name, "schema": schema, "strict": True},
    }
    from agentkit.adapters.llm.model_registry import Capability, model_capabilities

    declared = model_capabilities(model).native_json_schema
    if declared is Capability.YES:
        return native
    if declared is Capability.NO:
        return None
    return native if model.startswith("gpt-") else None


def _derived_capabilities(agent: Any) -> tuple[str, ...]:
    """Capabilities this agent's WIRING implies, as opposed to ones it declared.

    Exactly one today: a cognition holding a non-empty tool registry needs
    ``tools``. Deriving it is safe to do automatically precisely because
    ``ModelRegistry.check`` treats derived requirements as NO-only — it raises
    against a model declared incapable of tools, and stays silent for every
    model it has simply never heard of. Without that asymmetry this inference
    would put an UNKNOWN warning on essentially every development wiring.
    """
    from agentkit.agents.cognition import ReActCognition

    cognition = getattr(agent, "cognition", None)
    tools = getattr(cognition, "tools", None)
    if isinstance(cognition, ReActCognition) and tools is not None and bool(tools.names()):
        return ("tools",)
    return ()
