"""agentkit — a complete framework for agentic-AI features.

The framework is organised into peer top-level packages, each owning ONE
architectural concept. Adding a new concept means adding a new peer
package, not bloating an existing one. See ARCHITECTURE.md for the full
discipline.

  kernel/       opinion-free primitives — value types, ports, middleware
                contract, resilience, concurrency, observation
  runtime/      RunContext · Invoker · Budget · Quota · EventBus · NullCtx
  middlewares/  the chat/tool middleware chain — tracing · meter · retry ·
                fallback · memoize · output_coerce · compaction · guard
  context/      WorkingContext — in-flight reasoning state (prefix +
                messages + scratchpad + journal) + TokenCounter
  memory/       external-reach: MemorySource Protocol + Composite/Sequential
                + VectorMemory / FileMemory / JournalMemory / ScratchpadMemory /
                ToolMemory + Scoped/Compacted/Cached decorators
  tools/        Tool Protocol + FunctionTool + ToolRegistry + @tool
                decorator + as_tool adapter (Agent → Tool)
  prompts/      versioned Prompt (validator / evaluator skeletons live
                elsewhere — this package is prompt-shape only)
  skills/       Skill Facade composing (prompt + cognition + tools + memory)
                as one wirable unit; ``skill.as_agent()`` / ``skill.as_tool()``
  agents/       Agent + Workflow + Cognition Protocol (SingleCall / ReAct /
                Coordinator) + control primitives (signals, channel,
                ActorBudget, handoff, gate, termination) + policies
                (RoundRobin / Selector / Plan / Ledger)
  capabilities/ optional cross-cutting collaborators — RequestBuilder +
                Grounder · Compactor (4 impls) · Guardrail · Evaluator ·
                Checkpointer · output_schema (multi-shape SchemaAdapter)
  adapters/     concrete Port impls (LLM providers, vector, store,
                checkpoint, observer, replay, OTel)
  testing/      shared test doubles + fixtures (FakeLLM, FakeFetch,
                FakeSearch, FakeMemory, FakeTool, FakeCompactor, FakeGrounder,
                FakeCtx, make_test_ctx)

The framework's bet: every cross-cutting concern is a typed Protocol injected
at wire-up, not buried in the loop. Composition over inheritance throughout —
Strategy (Cognition, Policy, MemorySource, Tool), Composite (CompositeMemory,
ToolRegistry), Facade (Skill), Adapter (ToolMemory, as_tool), Decorator
(ScopedMemory, CompactedMemory, CachedMemory). No agent subclassing; new
shapes are new Protocol impls.

Zero core runtime dependencies (ADR-0005); extras (`http`, `postgres`,
`redis`, `observability`) are opt-in seams.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arc-agentkit")
except PackageNotFoundError:  # pragma: no cover — running from a source tree, not an install
    __version__ = "0.0.0+unknown"

from agentkit import middlewares  # noqa: E402
from agentkit.agents import (
    ActorBudget,
    Agent,
    AgentResult,
    Autonomy,
    BlockedSignal,
    BudgetExhausted,
    BudgetReducedSignal,
    CancelSignal,
    ContextUpdateSignal,
    ControlSignal,
    DataSignal,
    DoneSignal,
    EscalateSignal,
    Handoff,
    MergeInbox,
    MergeWithPeerSignal,
    ProgressSignal,
    RedirectSignal,
    RunPolicy,
    SignalChannel,
    SignalEnvelope,
    Suspended,
    Workflow,
    WorkflowResult,
)

# Batteries-included client. Safe to import on the zero-dep core: client.py pulls only kernel/runtime/
# middlewares; the httpx provider is imported lazily when a preset is *called*, not here.
from agentkit.capabilities import (
    Checkpointer,
    Compactor,
    Evaluator,
    Grounder,
    Guardrail,
    ImportanceFilteringCompactor,
    RequestBuilder,
    SlidingWindowCompactor,
    SummarizationCompactor,
    TruncationCompactor,
)
from agentkit.capabilities.output_schema import (
    OutputCoercionError,
    SchemaAdapter,
    adapt,
)
from agentkit.client import Chat, claude, deepseek, openai, openrouter
from agentkit.context import (
    AllOf,
    AnyOf,
    ApproxTokenCounter,
    ContextDiff,
    ContextScope,
    FrozenContext,
    LastNTurns,
    MutationJournal,
    Not,
    PrefixContext,
    RoleFilter,
    Since,
    Tagged,
    TiktokenCounter,
    TokenCounter,
    WorkingContext,
)
from agentkit.kernel import streams
from agentkit.kernel.concurrency import (
    CancellationToken,
    Cancelled,
    gather_best_effort,
    gather_bounded,
    run_agents,
    run_sync,
)
from agentkit.kernel.errors import (
    AgentkitError,
    CheckpointerError,
    Failure,
    ProviderAuthError,
    StoreUnavailable,
    compose_failures,
)
from agentkit.kernel.metrics import MetricsPort, NoopMetrics
from agentkit.kernel.middleware import (
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
from agentkit.kernel.observation import NoopObserver, Observation, ObserverPort, TracePort
from agentkit.kernel.ports import (
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
from agentkit.kernel.replay import NoopReplayStore, ReplayRecord, ReplayStore
from agentkit.kernel.resilience import (
    CircuitBreaker,
    CircuitOpen,
    ErrorClass,
    classify,
    idempotency_key,
    run_with_resilience,
)
from agentkit.kernel.sampling import AlwaysOnSampler, SamplerPort, TraceIdRatioSampler
from agentkit.kernel.types import (
    ChatRequest,
    LLMResult,
    Message,
    Operation,
    Scope,
    StreamEvent,
    ToolCall,
    ToolRequest,
    ToolSchema,
    Usage,
)
from agentkit.memory import (
    CompositeMemory,
    MemoryItem,
    MemorySource,
    Reranker,
    ScopedMemory,
    SequentialMemory,
    VectorMemory,
    score_sort_rerank,
)
from agentkit.prompts import Prompt
from agentkit.runtime import (
    Budget,
    EventBus,
    Invoker,
    Meter,
    MeterExceeded,
    NullCtx,
    Quota,
    RunContext,
    Services,
    VersionedEvent,
)
from agentkit.skills import Skill

# Test doubles are NOT re-exported from the top-level package: a
# ``from agentkit import FakeLLM`` shape would let production code
# accidentally pin a test double. They live at ``from agentkit.testing
# import FakeLLM`` — the boundary that distinguishes test infra from
# production wiring.
from agentkit.tools import (
    FileTool,
    FunctionTool,
    InMemoryFiles,
    Tool,
    ToolDefinitionError,
    ToolRegistry,
    ToolShapeError,
    as_tool,
    render_result,
    tool,
)

__all__ = [
    "__version__",
    # ---- kernel: value types -------------------------------------------------
    "Scope",
    "Usage",
    "Message",
    "ToolCall",
    "ToolSchema",
    "LLMResult",
    "StreamEvent",
    "ChatRequest",
    "ToolRequest",
    "Operation",
    # ---- kernel: ports (the seams the core owns) -----------------------------
    "LLMPort",
    "ToolPort",
    "VectorPort",
    "StorePort",
    "SearchPort",
    "SearchHit",
    "FetchPort",
    "FetchResponse",
    "ClockPort",
    "CheckpointPort",
    "Checkpoint",
    "CheckpointStatus",
    # ---- kernel: middleware contract -----------------------------------------
    "Call",
    "Handler",
    "Middleware",
    "chain",
    "collect",
    "collect_one",
    "BaseMiddleware",
    "MiddlewareContext",
    "Blocked",
    # ---- kernel: concurrency + resilience + errors ---------------------------
    "run_sync",
    "run_agents",
    "gather_bounded",
    "gather_best_effort",
    "CancellationToken",
    "Cancelled",
    "classify",
    "ErrorClass",
    "CircuitBreaker",
    "CircuitOpen",
    "idempotency_key",
    "run_with_resilience",
    "Failure",
    "compose_failures",
    "AgentkitError",
    "CheckpointerError",
    "StoreUnavailable",
    "ProviderAuthError",
    # ---- kernel: observation + tracing seam ----------------------------------
    "Observation",
    "ObserverPort",
    "NoopObserver",
    "TracePort",
    "middlewares",
    "streams",
    # ---- kernel: observability v2 (metrics + sampling + replay) -------------
    "AlwaysOnSampler",
    "MetricsPort",
    "NoopMetrics",
    "NoopReplayStore",
    "ReplayRecord",
    "ReplayStore",
    "SamplerPort",
    "TraceIdRatioSampler",
    # ---- runtime -------------------------------------------------------------
    "RunContext",
    "Services",
    "Invoker",
    "Budget",
    "Quota",
    "Meter",
    "MeterExceeded",
    "EventBus",
    "VersionedEvent",
    "NullCtx",
    # ---- capabilities --------------------------------------------------------
    "RequestBuilder",
    "Grounder",
    "Compactor",
    "ImportanceFilteringCompactor",
    "SlidingWindowCompactor",
    "SummarizationCompactor",
    "TruncationCompactor",
    "Guardrail",
    "Evaluator",
    "Tool",
    "ToolRegistry",
    "FunctionTool",
    "ToolDefinitionError",
    "ToolShapeError",
    "tool",
    "FileTool",
    "InMemoryFiles",
    "as_tool",
    "render_result",
    "Checkpointer",
    # ---- memory (unified MemorySource Protocol + first-party sources) -------
    "MemoryItem",
    "MemorySource",
    "Reranker",
    "score_sort_rerank",
    "CompositeMemory",
    "SequentialMemory",
    "VectorMemory",
    "ScopedMemory",
    # ---- output schema (structured outputs) ---------------------------------
    "OutputCoercionError",
    "SchemaAdapter",
    "adapt",
    # ---- context (in-flight reasoning state) ---------------------------------
    "WorkingContext",
    "PrefixContext",
    "FrozenContext",
    "ContextDiff",
    "MutationJournal",
    "TokenCounter",
    "ApproxTokenCounter",
    "TiktokenCounter",
    "ContextScope",
    "LastNTurns",
    "RoleFilter",
    "Tagged",
    "Since",
    "AllOf",
    "AnyOf",
    "Not",
    # ---- agents --------------------------------------------------------------
    "Agent",
    "AgentResult",
    "Suspended",
    "Workflow",
    "WorkflowResult",
    "Handoff",
    "RunPolicy",
    "Autonomy",
    # ---- agents: per-actor budget (companion to runtime.Budget) -------------
    "ActorBudget",
    "BudgetExhausted",
    # ---- agents: signal protocol (typed parent <-> child messaging) ---------
    "ControlSignal",
    "DataSignal",
    "SignalEnvelope",
    "SignalChannel",
    "MergeInbox",
    "CancelSignal",
    "BudgetReducedSignal",
    "RedirectSignal",
    "ContextUpdateSignal",
    "MergeWithPeerSignal",
    "ProgressSignal",
    "DoneSignal",
    "EscalateSignal",
    "BlockedSignal",
    # ---- prompts -------------------------------------------------------------
    "Prompt",
    # ---- skills (Facade: prompt + cognition + memory packaged as a unit) ----
    "Skill",
    # ---- adapters: batteries-included client ---------------------------------
    "Chat",
    "claude",
    "openai",
    "deepseek",
    "openrouter",
]
