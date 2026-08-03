"""L0 value types — the vocabulary every layer speaks. Pure stdlib dataclasses, no opinions.

These never change behaviour; they only carry data. `ChatRequest`/`ToolRequest` are the two units of
work the `Invoker` runs through a middleware chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

# Wire-level discriminator tag — a closed set typed as ``Literal``
# so consumers can exhaust the branch table with mypy.
StreamEventType = Literal["message_delta", "tool_call", "tool_result", "step", "interrupt", "final"]


class Operation(str, Enum):
    """The kind of unit of work a middleware is intercepting — match on it in a middleware
    (`if ctx.operation == Operation.MODEL_CALL: …`). A `str` enum, so `== "model_call"` also holds."""

    TOOL_CALL = "tool_call"  # a tool execution   (request is a ToolRequest; data = arguments)
    MODEL_CALL = "model_call"  # a chat / LLM call (request is a ChatRequest; data = messages)


@dataclass(frozen=True)
class Scope:
    """Tenant-isolation key. Threaded through every memory recall / cache key / meter / callback."""

    org_id: int | None = None
    domain_id: int | None = None

    def key(self) -> str:
        return f"org{self.org_id}:dom{self.domain_id}"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_read_tokens: int = 0  # prompt-cache hits (billed cheaper) — for cache-aware costing
    cache_write_tokens: int = 0  # prompt-cache writes (billed once, sometimes a premium)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            round(self.cost_usd + other.cost_usd, 6),
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation the model requested. `id` is echoed back on the matching tool-result message.

    ``arguments`` is exposed as an immutable ``MappingProxyType``: the
    ToolCall flows into the ReAct approval snapshot AND into the
    idempotency-key hash AND into the audit trail — a tool impl that
    did ``args.pop("token")`` would have desynced all three. Callers
    that need to mutate copy first: ``dict(tc.arguments)``.

    ``MappingProxyType`` is not natively ``copy.deepcopy``-safe
    (stdlib limitation — the view can't be pickled), which matters
    because ``Checkpointer.snapshot`` deep-copies state that
    contains ``ToolCall``s. ``__deepcopy__`` / ``__copy__`` unwrap
    the view into a plain dict at copy time; the fresh ``ToolCall``
    re-wraps it via ``__post_init__``."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclass — assign via object.__setattr__ to bypass the freeze.
        if not isinstance(self.arguments, MappingProxyType):
            object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    def __deepcopy__(self, memo: dict[int, Any]) -> ToolCall:
        import copy as _copy

        return ToolCall(
            id=self.id, name=self.name, arguments=_copy.deepcopy(dict(self.arguments), memo)
        )

    def __copy__(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments=dict(self.arguments))


@dataclass(frozen=True)
class ToolSchema:
    """JSON-schema advertisement of a tool — what the provider needs to *see* to call it."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """One turn in a chat transcript."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    name: str | None = None  # tool name on a result message
    tool_call_id: str | None = None  # set on a role="tool" result message
    tool_calls: tuple[ToolCall, ...] = ()  # set on an assistant turn that requested tools


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()  # calls the model requested this turn (empty = done)
    parsed: Any = None  # set by the response-coerce middleware when an output schema adapter
    # is wired on the run; stays None for unstructured calls. Kept on
    # LLMResult so the Agent's loop can lift it onto AgentResult.parsed
    # without re-running the adapter at the pattern layer.


@dataclass(frozen=True)
class StreamEvent:
    """A pattern-level run event emitted by `Agent.stream`/`*.stream` — a higher-level signal than a
    `Delta`: where `Delta` is one transport increment of a single LLM response, a `StreamEvent` marks a
    step in the *run* (a token `message_delta`, a `tool_call`/`tool_result`, an `interrupt`, the `final`
    result). `type` discriminates the payload."""

    type: StreamEventType
    text: str = ""
    tool_call: Any = None
    tool_result: Any = None
    result: Any = None  # AgentResult on a "final" event
    usage: Usage | None = None


@dataclass(frozen=True)
class Delta:
    """One streamed increment of an LLM response — the item the single `stream()` primitive yields.
    A `text` fragment and/or `tool_calls`; the **terminal** delta carries `usage`/`finish_reason`/
    `model`/`provider`. `assemble_deltas()` reduces a sequence back to an `LLMResult`, so a full
    chat result is just the collected stream."""

    text: str = ""
    model: str | None = None
    usage: Usage | None = None  # set on the terminal delta
    provider: str | None = None
    finish_reason: str | None = None  # set on the terminal delta
    tool_calls: tuple[
        ToolCall, ...
    ] = ()  # tool calls revealed this delta (providers send them whole)
    parsed: Any = None  # set by the response-coerce middleware on a synthetic post-terminal
    # delta when an output schema adapter is wired; carries the typed
    # object so ``assemble_deltas`` can lift it onto ``LLMResult.parsed``.
    partial: Any = None  # set by the streaming response-coerce middleware on each text
    # delta when an output schema adapter is wired AND the adapter's
    # tolerant ``partial_parse`` produces a partial typed object that
    # differs from the previously-emitted partial. Distinct from
    # ``parsed`` (the strict-validated terminal result): ``partial``
    # may have unset required fields, refreshes per delta, and is a
    # per-delta concept — ``assemble_deltas`` does NOT lift it onto
    # ``LLMResult`` (only the final ``parsed`` survives assembly).


def assemble_deltas(deltas: list[Delta]) -> LLMResult:
    """Pure reducer: fold streamed `Delta`s into the final `LLMResult` (content concatenated; usage /
    finish_reason / model / provider / tool_calls / parsed taken from the deltas that carry them)."""
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage, finish, model, provider = Usage(), None, None, None
    parsed: Any = None
    for d in deltas:
        content += d.text
        if d.tool_calls:
            tool_calls = d.tool_calls
        if d.usage is not None:
            usage = d.usage
        if d.finish_reason is not None:
            finish = d.finish_reason
        model = d.model or model
        provider = d.provider or provider
        if d.parsed is not None:
            parsed = d.parsed
    return LLMResult(
        content=content,
        model=model,
        provider=provider,
        finish_reason=finish,
        usage=usage,
        tool_calls=tool_calls,
        parsed=parsed,
    )


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit for semantic memory / RAG (metadata carries source + scope tags)."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- the two units of work the Invoker runs through a middleware chain --------------------


@dataclass(frozen=True)
class ChatRequest:
    """Frozen value type. To rewrite a field mid-chain (fallback swaps
    the model, compaction rewrites messages), build a new instance
    via ``dataclasses.replace`` and reassign ``call.request`` — the
    ``Call`` reference is mutable; the ``ChatRequest`` it points at
    is not."""

    messages: list[Message]
    model: str
    tools: list[ToolSchema] | None = None
    response_format: dict[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    cache_hint: Any = None  # provider prompt-cache hint (stable prefix)


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict[str, Any]
    tool: Any  # the resolved ToolPort (terminal calls tool.run)
    side_effecting: bool = False
    url_arg: str | None = None  # egress middleware checks this arg if set
