# Changelog

All notable changes to `arc-agentkit` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Five gaps reported from production use. Everything here is **additive** —
no existing wiring changes behaviour, and the full pre-change test suite
passes untouched.

### Added

- **`StreamEvent.partial_output`** — the in-progress typed object, forwarded
  from `Delta.partial` by both chat cognitions. An application can now stream
  a typed object through `Agent.stream` alone; previously the framework could
  parse a partial structured output and had nowhere to deliver it, forcing
  callers to reach into `_resolve_request_builder()` / `_output_adapter` and
  drive `ctx.invoker.stream` by hand. Named `partial_output`, not `partial`,
  to stay distinguishable from `AgentResult.partial` (a `bool` meaning "this
  run terminated incompletely"). Consumers must gate on `model_fields_set` —
  required fields may be unset. `assemble_deltas` still drops `partial` by
  design; see its docstring.
- **`Agent` warns once** when an output schema is declared but `output_coerce()`
  is missing from the chat chain — previously `partial_output` was silently
  `None` forever while `AgentResult.parsed` kept working, so nothing looked
  broken. `Invoker` now exposes `chat_middleware` / `tool_middleware` so the
  chain is introspectable.
- **`agentkit.adapters.llm.model_registry`** — one table mapping a model name
  to the provider that serves it *and* the capabilities it declares.
  `resolve_llm(model)` / `client.from_env(model)` read credentials from the
  environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
  `OPENROUTER_API_KEY`), replacing the bespoke bootstrap every application was
  writing. Provider factories load lazily by dotted path, so the registry
  stays importable on a zero-dependency install. Fallback is **opt-in**
  (`fallback="fake"`, warns once); a missing optional extra raises
  `MissingProviderExtra` naming the pip extra and is never absorbed into the
  fallback. Applications extend it with `register_model` / `register_provider`
  / `register_rule`. Credentials never appear in an error, warning, or repr.
- **Model capability declaration + bind-time refusal.** `Capability` is
  tri-state (`YES` / `NO` / `UNKNOWN`); an unregistered model reports
  `UNKNOWN`, never a guess in either direction. `Agent(requires=(...),
  min_context_window=..., on_unknown_capability=...)` refuses a mismatch in
  `__post_init__` — before any spend — raising `CapabilityMismatch` naming the
  capability and the model. A tool-holding cognition implies `tools`
  automatically; derived requirements raise on a declared `NO` but stay silent
  on `UNKNOWN`, so existing wiring with custom model names is unaffected.
  `Agent.check_capabilities()` re-asserts after a post-construction mutation.
- **`Charge` verdict + recoverable budget exhaustion.** `Meter.guard` /
  `Meter.charge` now return a `Charge` (`ok`, `reason`, exact `spent` /
  `remaining` as `Decimal`, cumulative `usage`). With `Budget(on_exceeded="stop")` the
  meter records the spend and returns the verdict instead of raising, and the
  tool-loop cognition writes a **current `suspended` checkpoint** before ending
  the run with `stop_reason="budget_exhausted"`. Previously `MeterExceeded`
  unwound out of `invoker.stream` past every checkpoint write, leaving a stale
  `running` snapshot that recorded only part of the money actually spent.
- **`Budget.usage`** — the full `Usage` accumulates (input / output /
  cache-read / cache-write tokens), shared by reference across `ctx.child()`,
  so a multi-agent run's token totals survive to the end. Applications no
  longer re-aggregate what the framework already summed.
- **`AgentResult.stop_reason`** — a closed `Literal` (`complete`, `suspended`,
  `expired`, `budget_exhausted`, `max_iterations`, `invalid_output`,
  `terminated`) plus `is_suspended` / `is_resumable`. A suspended run is now
  distinguishable from a failed one without parsing the `evals` bag; a failed
  run still raises and produces no `AgentResult` at all. The free-form detail
  string stays in `evals["stop_reason"]`.
- **`agentkit.agents.control.elicit`** — human-in-the-loop as value
  elicitation. `Elicitation` names what the run needs and `Decision` carries
  what a person supplied, with `actor` and `at` for the audit trail. An
  injected `Asker` (on `Services.asker`) makes a gated decision **park in
  place** — the cognition awaits inside its own coroutine, so live
  unserialisable state survives — while the return-and-resume path stays
  available and unchanged for callers that can serialise. `elicit(ctx, ...)`
  takes a `Ctx`, so it works from any cognition, not just ReAct;
  `ask_human_tool()` exposes it to the model itself.
- **Deadlines on a suspended run.** `Elicitation.deadline_s` and
  `ReActCognition(approval_deadline_s=...)` bound the wait; expiry produces
  `Decision(kind="expired")` and the run **degrades and continues** rather
  than hanging. `Suspended.deadline_at` carries absolute wall-clock expiry for
  an operator UI. `elicit_or_raise` opts into `ElicitationExpired` instead.
- **`SecretValue`** — redacts itself in `repr`/`str`; `reveal()` is the one
  explicit way out. A working context that has handled a secret elicitation is
  **never checkpointed again** for the rest of that run, so a one-time code
  cannot outlive its validity inside a durable store.

### Changed

- **Money is `Decimal`.** `Budget` / `Quota` keep an exact ledger at six
  decimal places; a hundred charges of `0.01` now sum to exactly `1.00`.
  `budget.spent()` and `budget.spent_cents()` are the exact reads, and
  quantization happens at read time so sub-cent calls are not rounded away.
  `budget.spent_usd` remains a `float` **mirror**, re-derived after every
  charge — every existing reader and the documented
  `Budget(spent_usd=saved.state[...])` resume path keep working. An
  over-precise *ceiling* raises `MoneyPrecisionError` at construction; an
  over-precise *charge* is quantized rather than aborting a run mid-flight.
- **`_infer_response_format` reads the registry**, not `model.startswith("gpt-")`.
  Provider-native `json_schema` mode is wired from the declared
  `native_json_schema` capability; the `gpt-` prefix survives only as a
  last-resort guess for an unregistered name.
- **`Agent.resume` / `ReActCognition.resume` accept a typed `Decision`** as
  well as the legacy `str` form, which is coerced. Signatures widened to
  `Mapping[str, str | Decision]` (`dict` is invariant, so a caller holding a
  `dict[str, str]` would otherwise fail type-checking). A missing entry still
  defaults to a denial. Denial messages now name the actor.
- **`Quota` returns verdicts too**, so both `Meter` implementations behave
  uniformly under the middleware's `all_meters` iteration.

### Fixed

- **`output_coerce()` no longer defeats parse-and-repair.** The middleware
  strict-parses at end-of-stream and re-raises; that exception escaped past
  the cognitions' reflect-and-retry branch, so adding the middleware — the
  very wiring that enables streamed partials — aborted the run on the first
  malformed response. Both cognitions now catch `OutputCoercionError` and let
  `agent.parse` re-raise it inside the repair loop. (`output_coerce` itself is
  unchanged.)

### Fixed (found in review of the above)

- **`Agent.resume` bypassed the `RunPolicy` lethal-trifecta gate.** The gate
  lived inline in `stream()`, so an agent whose tool set combines
  private-data access, untrusted-content ingestion, and egress was denied on
  `run()` and then executed that exact tool on `resume()`. Now shared by both
  entry points via `Agent._run_policy_gate`. Pre-existing; security-relevant,
  because resume is the path a human has just approved something on — and
  approving one tool *call* is not approval of the capability *combination*.
- **Retrying an exhausted budget spent another call each time.** Under
  `on_exceeded="stop"`, `guard()`'s not-ok verdict is ignored by the meter
  middleware (a middleware cannot write a checkpoint), and the cognitions only
  checked *after* the call — so every retry of an already-stopped run bought
  one more turn before noticing. Both cognitions now pre-flight the budget.
- **`Budget.max_cost_usd` was ignored after construction.** The normalised
  ceiling was cached in `__post_init__`, so `budget.max_cost_usd = 10.0` — the
  documented way to raise a ceiling and resume — silently did nothing. Now
  re-derives on assignment (non-strictly, because that read path is reached
  from inside `charge()` and raising there would abort a run mid-flight). Same
  for `Quota.max_usd`.
- **A suspend deadline was decorative.** `Suspended.deadline_at` was stamped
  but never persisted and never checked, so an operator answering an hour late
  still got the tool executed. It is now written into the checkpoint and
  honoured on `resume()`: pending calls become `expired`, the run degrades and
  continues, and the tool does not run.
- **The secret-taint containment covered one of seven checkpoint writers.** It
  sat in `ReActCognition._save`; the coordinator policies persist a blackboard
  scratchpad through their own `snapshot` calls. Moved to `Checkpointer.snapshot`
  — the single seam every producer passes through.
- **The park path emitted no `tool_call` event**, so a consumer counting them
  to render "running X…" saw nothing on approved gates.
- **`ask_human_tool` elicitation ids were unstable across processes**
  (`hash(str)` is randomised by `PYTHONHASHSEED`), breaking audit-trail
  correlation. Now a SHA-256 prefix.
- **`Charge.spent_usd` collided with `Budget.spent_usd`** — same name, `Decimal`
  on one class and `float` on the other. Renamed to `Charge.spent` /
  `Charge.remaining`; `*_usd` now means float everywhere and `spent`/`remaining`
  mean Decimal everywhere.

### Notes

- An `Asker` whose `ask` blocks the event loop (`input()`, `requests.get()`,
  `time.sleep()`) **cannot be deadlined** — scheduling is cooperative, so the
  timeout coroutine never runs and `deadline_s` becomes an unbounded hang. The
  `Asker` Protocol carries an explicit warning and a test pins the behaviour.
  Wrap synchronous work in `asyncio.to_thread`.
- `Quota` never evicts per-tenant keys from its internal window dicts, so a
  long-running process with unbounded tenant churn grows slowly. Pre-existing,
  not addressed here.
- `Budget(on_exceeded=...)` defaults to `"raise"`. Flipping it would silently
  change control flow in every existing wiring — a run that used to abort
  would continue past its ceiling in any caller ignoring the return value.
  Callers opt into recoverability.
- `output_coerce` re-parses the whole accumulated buffer per text delta, which
  is O(n²) in response length. Left as-is deliberately; sample partials if it
  shows up in a profile.

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
