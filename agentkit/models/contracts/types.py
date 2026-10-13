"""Immutable vocabulary for model-neutral calls."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Generic, Protocol, TypeVar

from agentkit.core.contracts.types import Message, Mode
from agentkit.core.serialization._immutable import deep_freeze
from agentkit.models.contracts.constants import CALL_EVENT_SCHEMA_VERSION, CostSource, EventKind, FailureClass, Outcome
from agentkit.state.context import PrefixContext

T = TypeVar("T")


@dataclass(frozen=True)
class CredentialRef:
    """Host-owned credential resolver; secrets are excluded from repr."""

    resolve: Callable[[], Mapping[str, str]] = field(repr=False, compare=False)


@dataclass(frozen=True)
class Binding:
    mode: Mode
    model: str
    credential: CredentialRef | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode(self.mode))
        if not self.model:
            raise ValueError("binding requires a model")


@dataclass(frozen=True)
class ModelSettings:
    effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    tier: str | None = None


@dataclass(frozen=True)
class CallLimits:
    max_cost_usd: Decimal | None = None
    max_turns: int | None = None
    max_output_tokens: int | None = None
    wall_clock_s: float | None = None
    idle_s: float | None = None
    first_event_s: float | None = None

    def __post_init__(self) -> None:
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value is not None:
                finite = value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)
                if not finite or value <= 0:
                    raise ValueError(f"{key} must be finite and positive")


@dataclass(frozen=True)
class Usage:
    """Disjoint input counts; reasoning is a subset of output, or unknown."""

    input_uncached: int = 0
    input_cache_read: int = 0
    input_cache_write_5m: int = 0
    input_cache_write_1h: int = 0
    output: int = 0
    reasoning: int | None = None

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_uncached + other.input_uncached,
            self.input_cache_read + other.input_cache_read,
            self.input_cache_write_5m + other.input_cache_write_5m,
            self.input_cache_write_1h + other.input_cache_write_1h,
            self.output + other.output,
            None if self.reasoning is None or other.reasoning is None else self.reasoning + other.reasoning,
        )


@dataclass(frozen=True)
class Failure:
    classification: FailureClass
    retryable: bool = False
    retry_after_s: float | None = None
    mode_detail: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "classification", FailureClass(self.classification))
        object.__setattr__(self, "mode_detail", deep_freeze(dict(self.mode_detail)))


@dataclass(frozen=True)
class Degradation:
    setting: str
    requested: Any
    applied: Any
    reason: str


@dataclass(frozen=True)
class EffectiveSettings:
    settings: ModelSettings
    degradations: tuple[Degradation, ...] = ()


@dataclass(frozen=True)
class CallEvent:
    kind: EventKind | str
    call_id: str
    fields: Mapping[str, Any] = field(default_factory=dict, hash=False)
    sequence: int = 0
    timestamp: str = ""
    parent_call_id: str | None = None
    schema_version: str = CALL_EVENT_SCHEMA_VERSION

    @property
    def event_id(self) -> str:
        return f"{self.call_id}:{self.sequence}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", deep_freeze(dict(self.fields)))


EventSink = Callable[[CallEvent], Awaitable[None] | None]


@dataclass(frozen=True)
class SessionHandle:
    mode: Mode
    run_id: str
    cli_session_id: str | None = None
    history: tuple[Mapping[str, Any], ...] = field(default=(), hash=False)
    prefix_digest: str = ""
    raw_usage: Mapping[str, Any] = field(default_factory=dict, hash=False)
    messages: tuple[Message, ...] = ()
    model: str = ""
    home_bundle: Mapping[str, str] = field(default_factory=dict, repr=False, hash=False)
    reported_cost: Decimal | None = None
    prefix: PrefixContext = field(default_factory=PrefixContext)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode(self.mode))
        object.__setattr__(self, "home_bundle", deep_freeze(dict(self.home_bundle)))
        object.__setattr__(self, "history", tuple(deep_freeze(dict(x)) for x in self.history))
        object.__setattr__(self, "raw_usage", deep_freeze(dict(self.raw_usage)))


@dataclass(frozen=True)
class Answer(Generic[T]):
    value: T | None = field(hash=False)
    text: str
    messages: tuple[str, ...]
    outcome: Outcome
    usage: Usage
    raw_usage: Mapping[str, Any] = field(hash=False)
    cost_usd: Decimal | None
    cost_source: CostSource
    effective: EffectiveSettings
    attempts: int = 1
    failure: Failure | None = None
    session: SessionHandle | None = None
    prompt_manifest: Any = field(default=None, hash=False)
    context: Any = field(default=None, hash=False)

    @property
    def degradations(self) -> tuple[Degradation, ...]:
        return self.effective.degradations

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        object.__setattr__(self, "cost_source", CostSource(self.cost_source))
        object.__setattr__(self, "raw_usage", deep_freeze(dict(self.raw_usage)))


@dataclass
class Turn:
    """Adapter facts. Interpretation, validation and retries belong to the core."""

    messages: tuple[str, ...] = ()
    value: Any = None
    raw_usage: dict[str, Any] = field(default_factory=dict)
    reported_cost: Decimal | None = None
    reported_cost_cumulative: bool = False
    failure: Failure | None = None
    refused: bool = False
    truncated: bool = False
    tool_calls: tuple[dict[str, Any], ...] = ()
    history: tuple[Mapping[str, Any], ...] = ()
    session_id: str | None = None
    usage_cumulative: bool = False
    usage_complete: bool = True
    home_bundle: dict[str, str] = field(default_factory=dict, repr=False)
    canonical_messages: tuple[Message, ...] = ()


class ModeAdapter(Protocol):
    """Transport-only seam, injectable for deterministic conformance tests."""

    async def turn(
        self,
        binding: Binding,
        prompt: Any,
        *,
        settings: ModelSettings,
        limits: CallLimits,
        schema: dict[str, Any] | None,
        tools: Any,
        session: SessionHandle | None,
        cancel: Any,
        emit: EventSink,
    ) -> Turn: ...
