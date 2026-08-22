# Changelog

All notable changes to `arc-agentkit` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Two batches of work in this cycle: five gaps reported from production use, and
a follow-up sweep for other major issues. Everything is additive except the
concurrency-bound change called out below.

### Fixed — a provider failure that read as a complete answer

- **An in-band SSE error frame was swallowed by both providers.** Anthropic and
  OpenAI can deliver a failure INSIDE a 200 response, part-way through a
  stream, once the headers are long gone
  (`{"type":"error","error":{"type":"overloaded_error"}}` /
  `{"error":{"type":"server_error"}}`). Neither translator had a branch for it,
  so the frame fell through every `elif`, the loop ended normally, and the
  caller received a **truncated answer presented as a complete one** — partial
  text, `finish_reason=None`, no exception anywhere. An agent takes that
  half-sentence as the model's final word. And because nothing raised,
  `retry()` never fired: the most retryable provider failure there is, an
  overload, was the one the resilience layer never saw. Both paths now raise a
  classified `ProviderError` via a shared `raise_if_error_frame`.
- **The error classifier missed the underscore forms providers actually use.**
  `_TRANSIENT` had `rate limit` (with a space) while the wire carries
  `rate_limit_error`, and had no `server_error`, so the errors above landed in
  `UNKNOWN`. They were still retried — only `PERMANENT` fails fast — but
  classified on nothing. Added `rate_limit`, `server_error`, `529`. Bare `500`
  is deliberately still absent: it is a substring of `5000`, which appears in
  ordinary text like `max_tokens 5000`, and a false `TRANSIENT` there retries a
  request that can never succeed.

### Fixed — tenant isolation, and two fail-open controls

- **`memoize` leaked cached answers across tenants.** `Scope`'s own docstring
  calls itself the key "threaded through every memory recall / cache key /
  meter / callback", but `memoize` took an arbitrary `key` callable and added
  nothing to it — so isolation depended on every caller remembering. They did
  not, because the key the cheatsheet and the LangChain migration guide TAUGHT
  was `lambda c: c.request.messages[-1].content`, which ignores the model, the
  tools, the temperature and the tenant. Measured: two tenants asking the same
  question, one provider call, and tenant 999 receiving tenant 1's answer.
  Every key is now namespaced by `ctx.scope.key()` **inside** the middleware —
  a boundary that relies on a caller-supplied key is not a boundary.
- **`memoize()` now works with no arguments** and defaults to an exact-match
  key over the fields that change the answer (model, messages, tool names,
  response_format, temperature, max_tokens). `key` was required, which pushed
  the most dangerous decision in a cache — what counts as "the same call" —
  onto every caller. Two docs pages already showed a bare `memoize()`, which
  raised `TypeError`. Tool schemas reduce to their names, so editing a
  description does not invalidate the cache.
- **`egress(None)` built a security control that checked nothing.** It sat in
  the chain with every SSRF and allowlist check silently off, which is how
  `egress(config.guardrail)` behaves when the config is unset. Now raises at
  wiring time, along with a `TypeError` for an object that has no `check_url`.
- **A late success closed an OPEN circuit breaker**, adding an `OPEN → CLOSED`
  edge the class docstring says does not exist (`CLOSED → OPEN → cooldown →
  HALF_OPEN → CLOSED`). Reachable at any concurrency above one — the normal
  case for the documented pattern of one breaker shared per dependency: enough
  in-flight calls fail to trip it, then a straggler that started before the
  trip reports success and reopens the gate. Measured through the real
  `retry()` middleware: a 300-second cooldown skipped by one late success,
  sending the herd back at a failing provider. A call admitted under the old
  state is evidence about the past; only the post-cooldown probe speaks to
  recovery.

### Checked and found sound

- The SSRF host blocker was audited against the classic bypasses and blocks
  all of them: decimal (`2130706433`), hex, octal, short-form (`127.1`), IPv6
  loopback, IPv4-mapped IPv6 (`[::ffff:127.0.0.1]`), userinfo
  (`user@127.0.0.1`), cloud metadata (`169.254.169.254`), RFC1918, and
  unspecified. A hostname that RESOLVES to a private IP is allowed, which is
  documented behaviour — name resolution is the injected `url_check`'s job.
- `SlidingWindowCompactor` preserves the system prompt when trimming.

### Fixed — the fan-out reservation path

- **A starved fan-out silently produced no-op children on two of three axes.**
  Only `steps` failed fast when a slice would round to nothing; `tokens` and
  `cost` floored to zero, so a reservation of zero "succeeded" and each child
  was handed an already-exhausted envelope. A fan-out of 3 against 2 tokens ran
  three children that each stopped immediately and looked like a completed
  wave. All three axes now fail fast with `BudgetExhausted`, naming the axis,
  and a fan-out from an already-exhausted parent refuses outright rather than
  carving zero-sized slices.
- **A child was over-granted a step it was never reserved.** The step axis was
  handed `max(slice_steps, 1)`, so with a one-step slice a child could take a
  second turn the parent had not committed — and `settle_child` caps usage at
  the reservation, making that spend invisible on the parent's books. Children
  now get exactly their slice.
- **Slices are carved in `Decimal`**, off `remaining_cost()` rather than the
  float mirror. Equal shares, so reservation order cannot skew fairness.
- **`_tightest_axis` crashed on the path it exists to explain.** It mixed the
  float mirror with the request amount, so once `run_agents` began slicing in
  `Decimal` the diagnostic raised `TypeError: unsupported operand type(s) for
  -: 'float' and 'decimal.Decimal'` instead of naming the blocking axis. Every
  money-bearing `ActorBudget` parameter now accepts what `to_money` accepts.
- `kernel/concurrency.py` coverage **57% → 91%**, including the reservation /
  settlement path and `run_sync`'s nested-loop branch (a sync host calling in
  from inside an async caller — the branch that quietly regresses into a
  deadlock). Coverage ratchet raised 85 → 87.

### Fixed — an inert ActorBudget, and one durable seam

- **`ActorBudget` did nothing.** Nothing in the framework ever charged it: the
  `meter()` middleware charges `ctx.run.all_meters` (the run `Budget` plus any
  `Quota`), and `ActorBudget` is not a `Meter` — four axes, a sync `charge`, no
  guard/charge protocol — so it was never in that list. And no loop consulted
  `exhausted()`, even though `charge` is documented as soft-exceed-then-stop
  *because* "the loop checks `exhausted()` and stops cleanly". The only thing
  that ever touched the envelope was `run_agents` reserving slices and
  releasing them with zero usage. Measured: **$3.00 of real spend against a
  $1.00 cap left `used_cost` at zero and `exhausted()` False**, while the
  run-scoped `Budget` correctly recorded $3.00. A documented safety mechanism
  that never ran. Both ends are now wired, and the terminal reason names the
  exhausted axis.
- **`ActorBudget`'s cost axis is an exact `Decimal` ledger**, replacing the
  epsilon threshold added earlier this cycle. `max_cost_usd` / `used_cost_usd`
  / `reserved_cost_usd` remain float MIRRORS so `run_agents` and existing
  readers are untouched; `max_cost()` / `used_cost()` / `reserved_cost()` /
  `remaining_cost()` are the exact accessors.
- **A ceiling crossed by the CLOSING call no longer discards the answer.** The
  post-call check fired after every chat call, so a run whose final call
  happened to exhaust the budget reported `budget_exhausted` with
  `partial=True` — for work already paid for, with a good result in hand. The
  check now runs only where the loop is about to spend *more*: before a tool
  dispatch, or before a repair retry.
- **`Workflow` now persists through the same seam as everything else.** It
  wrote only to `ctx.store`, while the ReAct cognition prefers
  `ctx.checkpointer` — so wiring the documented durable seam left workflow
  human-gates silently unpersisted. Both producers now share one
  `resolve_checkpointer`, gates are marked `SUSPENDED` (a status a bare KV
  write could not express), and `resume` falls back to reading the legacy
  `workflow:<run_id>` key so in-flight suspends survive the upgrade.

### Changed — CI

- Action majors bumped together (`actions/checkout@v7`,
  `actions/upload-artifact@v7`, `actions/download-artifact@v8`,
  `astral-sh/setup-uv@v10`). GitHub had forced every Node-20 action onto Node
  24, annotating every run. Our usage is limited to long-stable inputs, so the
  majors carry no interface change for us. The PyPI publish action stays
  SHA-pinned — it holds signing authority.

### Fixed — concurrency, budgets and workflow suspend

- **Nested fan-out deadlocked.** `ctx.semaphore()` returned ONE semaphore for
  the whole agent tree, and a parent's fan-out holds its permits for the entire
  duration of each child run — so a nested fan-out drew from a pool its own
  ancestors had already drained. Reproduced through the public API: an Agent
  dispatching two `as_tool` sub-agents that each dispatch their own tools hangs
  forever at `max_concurrency=2`. The pool is now keyed on `ctx.depth`, which
  breaks the cycle structurally since every nesting boundary goes through
  `ctx.child()`. **Behaviour change:** the bound is now `max_concurrency` per
  LEVEL, so worst-case in-flight work is `max_concurrency * (max_depth + 1)`.
  A single tree-wide cap cannot be both deadlock-free and respected by nested
  acquisition.
- **`gather_best_effort` swallowed cooperative cancellation.** It correctly
  re-raised `asyncio.CancelledError` but caught agentkit's own `Cancelled`
  under `except Exception`, turning it into a `Failure` slot. The token is
  shared across the run tree, so a tripped token gave the caller N independent
  "failures" with no way to tell an aborted run from a batch where everything
  broke at once. `Cancelled` is an abort and now aborts.
- **`ActorBudget` under-reported exhaustion.** Its float axes kept an
  `== 0.0` check, which does not survive float arithmetic: a `$1.00` cap
  charged ten times at `$0.10` left `remaining == 1.11e-16` and
  `exhausted() == False`, so the agent loop ran past its cap. `remaining_*`
  clamps with `max(0.0, …)`, catching an overshoot but not an undershoot. It
  was also inconsistent — `0.1 + 0.2` against a `0.3` cap lands exactly on
  zero — so the bug depended on which numbers a caller picked. Now thresholded
  at one unit of `MONEY_SCALE`. (The run-scoped `Budget` already had an exact
  Decimal ledger; converting ActorBudget's float reservation API is a larger
  change worth doing on its own.)
- **`Quota` never evicted expired tenants.** `_prune` only touched the key
  being guarded, so a tenant that went quiet leaked its dict entry forever —
  5000 distinct scopes left 5000 retained keys long after every window had
  expired. That is a slow leak in exactly the multi-tenant deployment the class
  exists for, and worse than one-entry-per-customer when a scope carries a
  per-user id. Added a sweep, at most once per window.
- **A workflow gate suspended with no store persisted nothing, silently.**
  `run()` returned `stop_reason="suspended"` with a `Suspended` object while
  writing no checkpoint; the truth surfaced later, usually in another process,
  as "no suspended workflow <id> to resume" with nothing pointing at the cause.
  Now warns at suspend time. The warning also surfaces an asymmetry: Workflow
  persists via `ctx.store`, while the ReAct cognition prefers
  `ctx.checkpointer` — so wiring only a `Checkpointer` leaves a workflow gate
  unpersisted too.

#### Notes on the above

- Abandoning `Agent.stream` before the `final` event releases the provider's
  HTTP stream at generator finalization, not at your `break`. On CPython that
  is prompt (200 abandoned streams measured to zero un-released after one event
  loop turn) but not deterministic; use `contextlib.aclosing` for a hard
  guarantee. Making every framework layer cascade the close would mean
  re-indenting ~22 `async for` sites across the middleware chain, which is not
  worth the risk for a leak CPython already collects.

### The five production-feedback briefs

Five gaps reported from production use. Everything here is **additive** —
no existing wiring changes behaviour, and the full pre-change test suite
passes untouched.

#### Added

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

#### Changed

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

#### Fixed

- **`output_coerce()` no longer defeats parse-and-repair.** The middleware
  strict-parses at end-of-stream and re-raises; that exception escaped past
  the cognitions' reflect-and-retry branch, so adding the middleware — the
  very wiring that enables streamed partials — aborted the run on the first
  malformed response. Both cognitions now catch `OutputCoercionError` and let
  `agent.parse` re-raise it inside the repair loop. (`output_coerce` itself is
  unchanged.)

#### Fixed (found in review of the above)

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

#### Notes

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
