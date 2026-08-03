"""L0 seams — the infra ports the core owns. The tracing seam is an injected `TracePort`
(no-op by default; observability is never a required dependency).

A seam is an *external system agentkit cannot implement itself* (a model, the world, a vector index,
a durable store, web search, the network, the wall clock). Features (idempotency, audit, checkpoint,
cache, quota) are NOT ports — they are middlewares/meters backed by `StorePort`. Pure `Protocol`s,
no third-party imports. A signature change here is a deliberate, reviewed event.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid import cycles at runtime
    from agentkit.kernel.types import Chunk, Delta, LLMResult, Message, Scope, ToolSchema


@runtime_checkable
class LLMPort(Protocol):
    def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        cache_hint: Any = None,
    ) -> AsyncIterator[Delta]: ...  # THE primitive — the Invoker's chat terminal iterates it

    async def chat(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        cache_hint: Any = None,
    ) -> LLMResult: ...  # derived: ≡ collect(stream(...)); kept for direct callers

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult: ...  # single-call convenience; a thin shim over chat() on real adapters


@runtime_checkable
class ToolPort(Protocol):
    name: str
    schema: Any  # optional ToolSchema; None → loop-invisible

    async def run(self, args: Any, ctx: Any) -> Any: ...  # typed I/O via Pydantic models on impls


@runtime_checkable
class VectorPort(Protocol):
    # search returns (score, chunk) so callers can threshold (RAG ignores, semantic-cache uses it).
    async def upsert(self, scope: Scope, chunks: list[Chunk]) -> None: ...
    async def search(
        self, scope: Scope, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[tuple[float, Chunk]]: ...


@runtime_checkable
class StorePort(Protocol):
    """The single KV seam that backs the former cache / idempotency / checkpoint / audit ports.
    `get_or_set` is single-flight; a producer that raises is NOT stored (failures are never cached)."""

    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None: ...
    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any: ...
    async def delete(self, key: str) -> None: ...  # idempotent remove (e.g. clear a checkpoint)
    async def append(self, key: str, value: Any) -> None: ...  # append-only log (audit)
    async def list(self, key: str) -> list[Any]: ...  # read an appended log back


@dataclass(frozen=True)
class SearchHit:
    """One web-search result — every `SearchPort` adapter normalizes to this shape so callers
    (Researcher agents, retrieval middlewares) are provider-agnostic."""

    url: str
    title: str
    snippet: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SearchPort(Protocol):
    # `where` is a metadata filter (date range, domain allowlist) — same pattern as VectorPort.
    async def search(
        self, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchHit]: ...


@dataclass(frozen=True)
class FetchResponse:
    """A text HTTP response. Binary fetch is out of scope — callers that need bytes use a different
    port. `fetched_at` is a unix timestamp from the implementation's `ClockPort` (real or fake)."""

    url: str
    status: int
    headers: dict[str, str]
    body: str
    content_type: str
    fetched_at: float


@runtime_checkable
class FetchPort(Protocol):
    # The implementation enforces the URL allowlist + caching + rate limits; callers don't pass them.
    async def fetch(self, url: str, *, timeout_s: float = 10.0) -> FetchResponse: ...


@runtime_checkable
class ClockPort(Protocol):
    """Deterministic time seam — tests and replay need a fakeable wall clock. `now()` returns a
    unix timestamp; `sleep(s)` waits without blocking the loop. Use this instead of `time.time()` /
    `datetime.now()` / `asyncio.sleep` inside middlewares + retry/backoff loops."""

    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class CheckpointStatus(StrEnum):
    """Coarse durability gate for a `Checkpoint`. Auto-resume keys off
    this — a `suspended` checkpoint says "waiting on a human";
    `running` says "engine is in motion"; `done` / `failed` are
    terminal. Producers map their domain phases (e.g., the api's
    `RunPhase`) onto this taxonomy at snapshot time.

    Emission map (grep-able index — keep aligned with call sites):

    * ``RUNNING`` — the default snapshot status at every mid-run
      durable transition (``Checkpointer.snapshot(...)`` with no
      explicit status). Set by the ReAct cognition after each
      tool-loop iteration and by coordinator policies at each turn.
    * ``SUSPENDED`` — emitted by the ReAct cognition's human-approval
      path when a gated tool call needs a human decision
      (``agentkit/agents/cognition/react.py`` — the ``_save(...,
      status="suspended", pending=...)`` call site). The suspended
      snapshot carries the ``pending`` tool calls so ``resume()`` can
      apply per-call decisions. ``StrEnum`` equivalence means the
      string ``"suspended"`` at the emission site IS
      ``CheckpointStatus.SUSPENDED`` when the adapter round-trips it
      back (see ``tests/adapters/test_postgres_checkpoint.py`` for
      the wire-level round-trip). Coordinators + custom producers
      MUST use this status (not ``RUNNING``) when persisting a
      human-gate wait — auto-resume relies on it to distinguish
      "engine is chugging" from "waiting on the world".
    * ``DONE`` / ``FAILED`` — emitted by the producer at terminal
      transitions; ``Checkpointer.resume`` filters these by default
      so a "resume if any checkpoint exists" wiring cannot silently
      re-run a finished job.
    """

    RUNNING = "running"
    SUSPENDED = "suspended"
    DONE = "done"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """True if no further work is expected — auto-resume should
        NOT fire on a `done` or `failed` checkpoint."""
        return self in {CheckpointStatus.DONE, CheckpointStatus.FAILED}


@dataclass(frozen=True)
class Checkpoint:
    """A serialized snapshot of run state at a meaningful transition.

    `state` is opaque to the port — the application (a leaf `Agent`, a coordinator `Agent`,
    an application-defined state snapshot) interprets it. `version` is monotonic: v1 is the
    first save for a run, v2 the next, and so on. `status` is a `CheckpointStatus` and
    lets the port surface "is this run resumable?" without re-parsing `state`.
    `created_at` is a unix timestamp stamped by the producer's `ClockPort` (or
    `time.time()` when no clock is wired in) so replay and audit lines up across
    services."""

    run_id: str
    version: int
    state: dict[str, Any]
    created_at: float
    status: CheckpointStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CheckpointPort(Protocol):
    """The durable-run-state seam. Distinct from `StorePort` (a generic KV) because the *thing*
    being persisted — a versioned, status-tagged snapshot — has its own access pattern: latest,
    at-version (time-travel), list-versions, and delete-all-for-run. A single producer is the
    authority over a `run_id`; two producers sharing a `run_id` would collide on `version`."""

    async def save(self, cp: Checkpoint) -> None: ...
    async def latest(self, run_id: str) -> Checkpoint | None: ...
    async def at_version(self, run_id: str, version: int) -> Checkpoint | None: ...
    async def list_versions(self, run_id: str) -> list[int]: ...
    async def delete(self, run_id: str) -> None: ...
