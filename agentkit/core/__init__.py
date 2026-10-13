"""L0 kernel — opinion-free primitives: value types, the 4 infra seams, resilience combinators,
structured concurrency, and the one middleware contract. Nothing here encodes a policy about how to
use an LLM or tool; higher layers (runtime, middlewares, patterns) supply the opinions."""

from agentkit.core.contracts.errors import Failure, compose_failures
from agentkit.core.contracts.middleware import (
    BaseMiddleware,
    Blocked,
    Call,
    Handler,
    Middleware,
    MiddlewareContext,
    chain,
    collect,
    collect_one,
)
from agentkit.core.contracts.ports import (
    Checkpoint,
    CheckpointPort,
    CheckpointStatus,
    ClockPort,
    FetchPort,
    FetchResponse,
    LLMPort,
    SearchHit,
    SearchPort,
    StorePort,
    ToolPort,
    VectorPort,
)
from agentkit.core.contracts.replay import ReplayRecord, ReplayStore
from agentkit.core.contracts.types import (
    ChatRequest,
    Chunk,
    CostSource,
    Delta,
    LLMResult,
    Message,
    MessageRole,
    Mode,
    Operation,
    Scope,
    StreamEvent,
    ToolCall,
    ToolRequest,
    ToolSchema,
    Usage,
    assemble_deltas,
)
from agentkit.core.execution import streams
from agentkit.core.execution.concurrency import (
    CancellationToken,
    Cancelled,
    gather_best_effort,
    gather_bounded,
    run_agents,
    run_sync,
)
from agentkit.core.execution.recurrence import OnRepeat, Stuck, attempt_until_stuck
from agentkit.core.execution.resilience import (
    CircuitBreaker,
    CircuitOpen,
    ErrorClass,
    backoff_delay,
    classify,
    idempotency_key,
    run_with_resilience,
)
from agentkit.core.telemetry.metrics import MetricsPort
from agentkit.core.telemetry.observation import CRITICAL_KINDS, NoopObserver, Observation, ObserverPort, TracePort
from agentkit.core.telemetry.sampling import SamplerPort, TraceIdRatioSampler

__all__ = [
    "Mode",
    "CostSource",
    # types
    "Scope",
    "Usage",
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolSchema",
    "LLMResult",
    "StreamEvent",
    "Chunk",
    "ChatRequest",
    "ToolRequest",
    "Operation",
    "Delta",
    "assemble_deltas",
    # seams — model / tool / vector / store
    "LLMPort",
    "ToolPort",
    "VectorPort",
    "StorePort",
    # seams — web / wall-clock / durability
    "SearchPort",
    "SearchHit",
    "FetchPort",
    "FetchResponse",
    "ClockPort",
    "CheckpointPort",
    "Checkpoint",
    "CheckpointStatus",
    # middleware contract (the single streaming contract + its reducers)
    "Call",
    "Handler",
    "Middleware",
    "chain",
    "collect",
    "collect_one",
    "BaseMiddleware",
    "MiddlewareContext",
    "Blocked",
    # observation channel + tracing seam
    "Observation",
    "ObserverPort",
    "NoopObserver",
    "TracePort",
    "CRITICAL_KINDS",
    # replay store (high-cardinality side channel keyed by span_id)
    "ReplayRecord",
    "ReplayStore",
    # metrics seam (counters + histograms; time-series rollup)
    "MetricsPort",
    # sampling seam (head-sampling decision at span open)
    "SamplerPort",
    "TraceIdRatioSampler",
    # resilience
    "classify",
    "ErrorClass",
    "backoff_delay",
    "CircuitBreaker",
    "CircuitOpen",
    "idempotency_key",
    "run_with_resilience",
    # recurrence — the semantic companion to run_with_resilience: bounded by a
    # repeated OUTCOME rather than by an attempt count
    "attempt_until_stuck",
    "Stuck",
    "OnRepeat",
    # concurrency
    "gather_bounded",
    "gather_best_effort",
    "run_agents",
    "run_sync",
    "Cancelled",
    "CancellationToken",
    # typed failures + reactive stream operators
    "Failure",
    "compose_failures",
    "streams",
]
