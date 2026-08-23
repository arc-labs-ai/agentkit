"""Raw↔`LLMResult` translation shared by the LLM adapters: message flattening, tool-call parsing,
duck-typed result coercion, and the dependency-free zero-cost default."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentkit.kernel._json import JSONDecodeError
from agentkit.kernel._json import loads as _json_loads
from agentkit.kernel.types import Delta, LLMResult, Message, ToolCall, Usage


def coerce_tool_args(raw: Any) -> dict[str, Any]:
    """Parse a provider's tool-call `arguments`/`input` payload (a JSON string, or already a dict) into a
    dict, preserving the original text under `_raw` when it isn't valid JSON. Shared by every provider's
    tool-call parsing so the fallback behaviour is identical."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = _json_loads(raw or "{}")
    except (JSONDecodeError, TypeError):
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}


def result_to_delta(res: LLMResult) -> Delta:
    """Wrap a complete `LLMResult` as a single terminal `Delta` — the one-item stream a non-incremental
    adapter yields so it satisfies the streaming primitive; `collect` reproduces the result."""
    return Delta(
        text=res.content or "",
        tool_calls=res.tool_calls,
        usage=res.usage,
        finish_reason=res.finish_reason,
        model=res.model,
        provider=res.provider,
    )


def _flatten(messages: list[Message]) -> tuple[str, str]:
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    rest = [m for m in messages if m.role != "system"]
    user = (
        "\n\n".join(f"[{m.role}] {m.content}" for m in rest)
        if len(rest) > 1
        else (rest[-1].content if rest else "")
    )
    return system, user


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _extract_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    """Parse the OpenAI/OpenRouter tool-call shape into `ToolCall`s (duck-typed, best-effort)."""
    tcs = _get(raw, "tool_calls")
    out: list[ToolCall] = []
    for tc in tcs or ():
        fn = _get(tc, "function") or tc
        name = _get(fn, "name") or ""
        out.append(
            ToolCall(
                id=str(_get(tc, "id") or name),
                name=name,
                arguments=coerce_tool_args(_get(fn, "arguments")),
            )
        )
    return tuple(out)


def _zero_cost(model: Any, in_tok: int, out_tok: int) -> float:
    """Dependency-free default: cost is unknown without a pricing source. Inject `cost_fn` on
    `CallableLLM` (a provider price table or whatever the host provides) to populate it —
    agentkit's core never imports an observability/cost library."""
    return 0.0


def _first_int(obj: Any, *keys: str) -> int:
    """First truthy value among `keys`, read dict- OR attribute-wise, as an int.

    `getattr(raw, "usage", {})` followed by `usage.get(...)` had two failure modes, both
    measured: a plain-dict `usage` was never even reached (see `_duck_to_result`), and a real
    SDK response object CRASHED with `AttributeError: 'SdkUsage' object has no attribute 'get'`
    — a hard failure in a coercion helper whose whole contract is best-effort duck typing.
    Reading each key through `_get` degrades to 0 for both shapes instead."""
    for key in keys:
        value = _get(obj, key)
        if value:
            return int(value)
    return 0


def _duck_to_result(raw: Any, cost_fn: Callable[[Any, int, int], float] = _zero_cost) -> LLMResult:
    """Coerce a duck-typed provider response into an `LLMResult`.

    Every field is read with `_get`, which handles BOTH a mapping and an object. It used to use
    bare `getattr`, so a dict-shaped response — the single most common shape for an injected
    `CallableLLM` fn — matched nothing and coerced to an empty result: measured, a fn returning
    `{"content": "the answer is 42", "usage": {...}}` produced `content='' usage=Usage(0, 0, 0.0)`.
    Content, tool calls, tokens and cost were all lost with no exception, so the agent read the
    empty string as the model's answer and the call was billed as free."""
    if isinstance(raw, LLMResult):
        return raw
    usage = _get(raw, "usage")
    in_tok = _first_int(usage, "input_tokens", "prompt_tokens", "prompt_token_count")
    out_tok = _first_int(usage, "output_tokens", "completion_tokens", "candidates_token_count")
    model = _get(raw, "model")
    return LLMResult(
        content=_get(raw, "content") or "",
        model=model,
        provider=_get(raw, "provider"),
        finish_reason=_get(raw, "finish_reason"),
        usage=Usage(in_tok, out_tok, cost_fn(model, in_tok, out_tok)),
        # `CallableLLM.chat` back-fills these too; doing it here means the `complete()` path
        # (which never called `_extract_tool_calls`) stops dropping a tool-calling response.
        tool_calls=_extract_tool_calls(raw),
    )
