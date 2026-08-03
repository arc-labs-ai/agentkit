# Changelog

All notable changes to `arc-agentkit` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes yet._

## [0.1.0] — 2026-08-04

Initial public release. Distributed on PyPI as `arc-agentkit`; imported as
`agentkit`.

### Added
- Complete framework: `kernel/` (opinion-free value types, ports, middleware
  contract, resilience, concurrency, observation), `runtime/` (`RunContext`,
  `Invoker`, `Budget`, `Quota`, `EventBus`, `NullCtx`), `middlewares/` (the
  chat + tool chain: `tracing`, `meter`, `retry`, `fallback`, `memoize`,
  `output_coerce`, `compaction`, `security`, `egress`, `audit`), `context/`
  (`WorkingContext` + `TokenCounter`), `memory/` (unified `MemorySource`
  Protocol + `Composite`/`Sequential`/`Vector`/`File`/`Journal`/`Scratchpad`/
  `Tool` sources + `Scoped`/`Compacted`/`Cached` decorators), `tools/`
  (Tool Protocol + `FunctionTool` + `ToolRegistry` + `@tool` +
  `as_tool`), `prompts/` (versioned `Prompt`), `skills/`
  (`Skill` Facade), `agents/` (`Agent` + `Workflow` + Cognition Protocol
  with `SingleCallCognition`/`ReActCognition`/`CoordinatorCognition` +
  signals + policies), `capabilities/` (`RequestBuilder`, `Grounder`,
  `Compactor` (4 strategies), `Guardrail`, `Evaluator`, `Checkpointer`,
  `SchemaAdapter`), `adapters/` (concrete Port impls behind opt-in extras),
  `testing/` (`FakeLLM`, `FakeFetch`, `FakeSearch`, `FakeMemory`, `FakeTool`,
  `FakeCompactor`, `FakeGrounder`, `make_test_ctx`).
- Batteries-included provider client: `Chat` + `claude` / `openai` /
  `deepseek` / `openrouter` presets, pre-wired with `tracing → meter →
  retry` on a `RunContext`.
- Typed error taxonomy raised at adapter/port boundaries: `AgentkitError`
  (base), `CheckpointerError`, `StoreUnavailable`, `ProviderAuthError`.
  Adapters wrap backend exceptions (`asyncpg`, `httpx`, `redis`, …) with
  `raise …Error(…) from exc` so callers pattern-match on the framework
  taxonomy without importing backend types.
- `RunPolicy` auto-invoked from `Agent.run` when set: `mode="deny"` raises
  `PermissionError` before the first cognition drive; `mode="flag"` stamps
  a `policy.flagged` observation and lets the run continue. One
  `policy.check` span per run carries mode, capabilities, and verdict.
- `Checkpointer` acquires a per-`run_id` `asyncio.Lock` around the
  read-compute-write cycle in `snapshot`, and deep-copies `state` +
  `metadata` on save so subsequent producer-side mutation cannot corrupt
  the stored version. Concurrent snapshots see monotonic distinct
  versions instead of racing on `next_version = max + 1`.
- Apache-2.0 `LICENSE` + `NOTICE` at the package root; `py.typed` marker
  ships in the wheel; `agentkit.__version__` populated from installed
  distribution metadata.
- mkdocs-material documentation site deployed to
  [`arc-labs-ai.github.io/agentkit`](https://arc-labs-ai.github.io/agentkit/).
- `docs/mental-models/` covering four canonical use cases (multi-tenant
  chat, autonomous DevOps investigator, long-running enrichment,
  coordinated research), each stressing a distinct set of framework
  invariants.
- Runnable examples under `examples/` — every example uses `FakeLLM` and
  needs no API key: `01_single_agent.py`, `02_streaming_and_tools.py`,
  `03_composed_middlewares.py`.
- Optional-extras: `http`, `postgres`, `redis`, `observability`, `fast`.

### Changed
- `Suspended.pending` typed as
  `tuple[ToolCall, ...] | tuple[str, ...]` (previously an unconstrained
  tuple). The tool-approval surface emits `ToolCall`s; `Workflow`'s
  `human_gate` node emits gate-name `str`s. The narrowed union catches
  drift from a third caller passing arbitrary objects, and pins the
  suspend/resume handshake at both ends of the wire.
- `ProviderAuthError` multi-inherits from
  `agentkit.kernel.errors.ProviderAuthError` (kernel taxonomy) and the
  transport-level `ProviderError`, so raised instances satisfy
  `except AgentkitError`, `except ProviderAuthError`, and legacy
  `except ProviderError` blocks simultaneously.

### Fixed
- Roughly 200 `mypy --strict` findings across `kernel/`, `runtime/`,
  `agents/`, `capabilities/`, and `adapters/` — missing return types,
  `Any`-typed public seams, `Optional` mis-annotations, and a handful of
  variance mistakes on Protocol generics. `uv run mypy --strict agentkit`
  is now clean on the full tree.
