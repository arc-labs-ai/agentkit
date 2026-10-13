# One agent runtime for four model modes

**agentkit design for model-neutral calls, tools, MCP, skills and dynamic agents — and the thin layer Mirror keeps
on top**

Date: 2026-10-13. Status: proposal for the agentkit maintainer. Research for this document was done against the
vendors' current documentation and source on this date; every external fact is cited in §15 and marked where it
was not directly observed.

---

## Contents

1. Why this exists
2. Where agentkit stands today (baseline first)
3. The four modes at a glance
4. Architecture: layers and ownership
5. The call contract (C1–C12)
6. Model settings: temperature vs effort
7. Tools Mirror serves to agents
8. External MCP servers that users connect
9. Skills, agent definitions, instructions and dynamic prompts
10. Observability
11. Public API
12. Proof: conformance suite and the eval matrix
13. What stays in Mirror's `cognition`
14. Order of work and migration
15. Open questions to settle with live probes
16. Sources

---

## 1. Why this exists

Mirror calls models from about ten places — control (preflight, chat), assay (research, plan, design, drive),
forge (build lanes), the eval harness — and each built its own adapter. Every defect found in one path existed or
was fixed separately in another. Defects observed in live runs on 2026-10-11/12:

| Defect | Effect seen |
|---|---|
| Codex answers joined every assistant message into one string | a note written before a patch was glued onto `diff --git`; patches refused; JSON answers with a preface dropped |
| OpenAI strict mode returns `null` for optional fields; cleaning did not follow `anyOf`/`oneOf`/`$ref` | correct seed answers rejected on every run (17 of 23 eval cases failing); live repair turns spent |
| Disconnects ("Server disconnected") not retried | a 3-hour, $155 build failed with every stage finished |
| One call path deleted its credential home in one pass while Codex was still writing a plugin cache | the cleanup error replaced a correct answer; 61 leftover homes, two holding an API key |
| Tool limits only in JSON-schema keywords | Codex drops those keywords; agents asked past limits and were refused |
| Claude and Codex served tools through different wrappers | different overflow and image behaviour per runtime |
| Each plane priced usage itself; an unknown model cost 0 | spend ceilings that could not fire |

The goal: **one runtime every model call goes through**, with identical observable behaviour on all four modes,
so a defect is fixed once and a stage is evaluated the same way on every model.

---

## 2. Where agentkit stands today (baseline first)

`/Users/dododuck/Desktop/agentkit-main` is **two commits behind** what Mirror runs. Mirror resolves agentkit
`main` to commit `555bc57` (uv.lock); the local copy matches `c6eadc6`. The two missing commits
("harden the CLI cognitions against hangs, leaks and drift", "fixed cli") add about 6,800 lines in
`_cli_common.py`, `claude_cli.py`, `codex_cli.py` and three test files: `env_policy`, `native_tool_policy`,
process-group kill, 250 ms cancel polling, bounded stderr drain, 8 MiB line limit, `CliTimeouts`,
`StructuredOutputFailure`, `_validate_json_schema`. Mirror already passes some of these options.

**Start every change from `555bc57` (or current remote `main`), never from the local copy.**

What agentkit already does well and should be kept: `Checkpointer` and CLI sessions; `CancellationToken` /
`RunContext`; `Budget`; `ReActCognition` / `SingleCallCognition`; `serve_registry` with `cli_kwargs()` /
`codex_kwargs()`; `ProviderError.status_code`; the tracing middleware and OTel adapter; `Prompt.render` strictness.

---

## 3. The four modes at a glance

The engine already admits exactly these pairs (`kernel/authority/agent_routing.py`).

| | M1 Claude Code CLI / Agent SDK | M2 Codex CLI `exec --json` | M3 Anthropic Messages API | M4 OpenAI API (Responses) |
|---|---|---|---|---|
| Transport | subprocess, NDJSON `stream-json` | subprocess, JSONL events | HTTPS + SSE | HTTPS + SSE |
| Final answer | the `result` message (`result` / `structured_output`) — not the last `assistant` message | last `agent_message` of the turn; cleared on failure | the turn whose `stop_reason` is not `tool_use`/`pause_turn` | the output `message` item; check `status` |
| Structured output | synthetic tool + validator retry loop (draft-07); `success` can arrive without `structured_output` | `--output-schema` (OpenAI strict rules, inferred) | constrained decoding, `output_config.format` | constrained decoding, `text.format` strict |
| Usage | per-step `output_tokens` is a placeholder; result `usage` excludes sub-agents; `modelUsage` includes them | `turn.completed.usage`, possibly thread-cumulative; none on failed turns | `input_tokens` excludes cache; cache split by 5m/1h | `input_tokens` includes cache; `reasoning_tokens` explicit |
| Cost | client-side estimate (`total_cost_usd`, `costBasis`), cumulative on resume | not reported | not reported | not reported |
| Failure signal | exit code + `result.subtype`/`is_error`; SIGTERM leaves no result | `turn.failed` + exit code; details on stdout | HTTP status + mid-stream `error` events | HTTP status + `response.failed`/`error` events |
| Internal retries | yes (`system/api_retry`) | yes | SDK retries 2× | SDK retries |
| Sampling (temperature) | not exposed | not exposed | 400 on current models if non-default | removed unless effort `none` |
| Reasoning depth | `--effort` (clamped silently) | `model_reasoning_effort` | `output_config.effort` + `thinking` | `reasoning.effort` |
| MCP | config files; failed / needs-auth servers silently drop tools | config file; `required` servers fail hard | connector: remote servers, tools only | remote MCP tool with approvals, tools only |

The consequence: almost nothing about a call is the same across modes. The runtime's job is to make it the same.

---

## 4. Architecture: layers and ownership

```
 Mirror planes (control · assay · forge · evals)
        │  cognition.ask / cognition.run / cognition.run_agent      ← Mirror: routing, ledger, journals, prompts
 ───────┼─────────────────────────────────────────────────────────────
        ▼                                                               agentkit
   Entry points        ask (one-shot) · run (agent loop) · run_agent (definition + task)
   Composition         prompt parts → manifest → rendered prompt; skills activation; delegation
   Contract core       final answer · structured output · settings · events · usage/price · failures/retry ·
                       limits · cancellation · homes · sessions · caching
   Tools & MCP         one tool registry · MCP server (serve) · MCP client/gateway (external servers) ·
                       result hook (overflow, images) · limits-in-words · deferred loading
   Mode adapters       M1 Claude Code · M2 Codex · M3 Anthropic · M4 OpenAI
```

Rules:

- **Every behaviour in §5–§10 lives in the contract core or tools layer**, never in a mode adapter. An adapter
  only translates: request → mode, mode events → `CallEvent`s, mode usage → raw usage, mode failure → failure
  facts. A conformance test runs the same scenario against each adapter.
- **Planes never call a mode directly.** They call `cognition`, which calls agentkit. Old paths are deleted once
  nothing uses them (§14).
- **Sounding never dispatches.** Control names a harness only in its providers seam. These plane rules stay in
  Mirror and are enforced by Mirror's existing checks.

---

## 5. The call contract

Each item: why it matters, what the modes actually do, agentkit today, the design, the test.

### C1 — The answer is the last assistant message

**Why.** Models write commentary before answering. If commentary is concatenated into the answer, a patch or JSON
document becomes invalid.

**Modes.** Codex can emit several `agent_message` items per turn; its own CLI takes the *last* before
`turn.completed` and clears it on failure. Claude Code emits one `assistant` message per content block sharing one
`message.id`, and the authoritative answer is the `result` message. The APIs return one assistant message per
response; in a tool loop the final turn is the one that does not end in `tool_use`/`pause_turn`.

**agentkit today.** Both CLI cognitions concatenate every message (`codex_cli._TurnState.fold`,
`claude_cli` fold); a test asserts `"First. Second."`. Mirror works around it with `MessageTap`/`settle_answer`,
and two Mirror call sites still get joined text.

**Design.** Track message boundaries (Codex item id per `agent_message`; Claude `message.id`, deduplicated;
prefer `result.result`/`structured_output`). `Answer.text` = final message; `Answer.messages` = all, in order,
kept as transcript. Structured output is decoded from the final message only. On a failed turn there is no
answer.

**Test.** Two messages then a final answer → `text` is the final one on all four modes; the old concatenation
test is flipped.

### C2 — Structured output, one canonical schema compiled per mode

**Why.** "Structured output" is three different mechanisms with incompatible schema subsets; a schema valid on
one mode is rejected or silently degraded on another; success statuses do not guarantee a payload.

**Modes (observed).**
- *OpenAI strict:* every object `additionalProperties:false`; every property in `required`; optional = union with
  `null`; root must be an object, not `anyOf`; no `allOf`, `not`, `if/then/else`, `dependent*`; limits of 5,000
  properties, 10 levels, 120,000 characters of names/enums, 1,000 enum values. Numeric/length keywords: the
  current page lists several as supported on base models; older sources say unsupported — probe (§15).
- *Anthropic:* `output_config.format` (GA); `additionalProperties:false` required; no `minimum/maximum/
  multipleOf/minLength/maxLength/maxItems`, `minItems` only 0 or 1, no recursion; budgets of 20 strict tools,
  24 optional parameters, 16 `anyOf`/type-array parameters per request; 180 s compile timeout. Forced
  `tool_choice` (`any`/named) is a 400 on the newest models, so "force a tool to get JSON" no longer works there.
- *Claude Code:* `--json-schema` / `output_format`; draft-07 only; `format` is annotation-only; enforced by a
  synthetic tool plus re-prompting; failure is `error_max_structured_output_retries`, and `subtype: success` with
  no `structured_output` happens.
- *Codex:* `--output-schema` follows OpenAI strict rules (inferred); a looser schema is refused with the reason on
  stdout only.
- *All:* refusals (`stop_reason:"refusal"`, OpenAI `refusal` items) and truncation (`max_tokens`,
  `incomplete`) arrive as HTTP 200 with possibly invalid JSON.

**agentkit today.** Sends `strict:true` with the raw Pydantic schema (OpenAI rejects optionals); Codex adds only
`additionalProperties:false`; Anthropic API enforces by prompt only; no null cleaning; CLI cognitions ignore
`max_repairs`.

**Design.**
1. `OutputSpec(schema, name)` holds the caller's canonical schema (JSON Schema 2020-12).
2. `compile(schema, mode, model)` produces the mode's dialect: strict rewrite for OpenAI/Codex (required-all,
   nullable optionals, `additionalProperties:false`, unsupported keywords removed and moved into descriptions);
   Anthropic subset (bounds moved to descriptions, budgets checked before sending); draft-07 for Claude Code.
   A schema that cannot be expressed on the mode is refused before dispatch with the reason.
3. On the way back: detect refusal and truncation first; then `clean_nulls(value, canonical)` (follows `anyOf`,
   `oneOf`, `$ref`; never removes a required null; a value matching no branch is left for the validator); then
   validate against the **canonical** schema (constraints stripped for the mode are enforced here).
4. One bounded correction turn with the validation errors, on every mode, including the CLIs.
5. Absent and `null` normalise to one representation in `Answer.value`.

**Test.** Optional fields inside `anyOf` and `$ref`; a bound the mode cannot express (enforced client-side);
a refusal; a truncation; a `success` without payload (CLI) → failure; the correction turn fixes a near-miss.

### C3 — Model settings

Covered fully in §6 because the modes diverge most here.

### C4 — One event stream for every mode

**Why.** Planes need to observe a call identically regardless of mode — for logs, UIs and liveness.

**Modes.** Anthropic SSE (`message_start` … `message_stop`, `ping`, mid-stream `error`); OpenAI Responses events
(`response.created` … `response.completed`/`failed`/`incomplete`, text/function-argument/reasoning deltas, MCP
progress); Claude Code NDJSON (`system/init`, `system/api_retry`, `assistant`, `user`, `stream_event`,
`result`); Codex JSONL (`thread.started`, `turn.*`, `item.*`, `error`). Codex `error` events can be non-fatal
reconnect notices; only `turn.failed` ends a turn.

**Design.** A `CallEvent` union, emitted to an `EventSink` and as OTel spans (§10):

| Event | Fields |
|---|---|
| `call_started` | call_id, mode, model, effective settings, inputs digest, prompt manifest digest, attempt |
| `message_delta` | message index, text delta (opt-in) |
| `message_done` | message index, length, final? |
| `reasoning` | summary delta (only where the mode exposes it; never raw reasoning) |
| `tool_call` / `tool_result` | name, call id, argument shape (sizes/digests, not bodies by default), result size, overflowed, is_error |
| `usage` | raw usage so far (cumulative), normalised usage |
| `retry` | failure class, wait seconds, source (`retry-after` / backoff), attempt |
| `degraded` | requested vs applied setting (§6), reason |
| `call_finished` | outcome (`answered`, `refused`, `truncated`, `failed`, `cancelled`), usage, cost, cost source, duration |

Optional masked, bounded request/response capture for audit, off by default. Unknown mode event types are
recorded as such (version drift), never silently ignored.

**Test.** The same scripted scenario yields the same event sequence on all four modes.

### C5 — Usage and pricing

**Why.** Token semantics differ per mode, cost is reported by only one mode (as an estimate), and an unknown model
silently costing 0 makes every spend ceiling useless.

**Modes.** Anthropic `input_tokens` excludes cache reads and writes (total = read + creation + input; creation
split 5m/1h). OpenAI `input_tokens` includes cached tokens; `reasoning_tokens` explicit; `cache_write_tokens` on
newer models. Claude Code `total_cost_usd` is a client estimate, includes sub-agents, is cumulative on resume;
per-step output tokens are placeholders. Codex reports `input_tokens`, `cached_input_tokens`,
`cache_write_input_tokens`, `output_tokens`, `reasoning_output_tokens`, possibly thread-cumulative, absent on
failure. Price structure: cache write 1.25× (5m) / 2× (1h) on Anthropic, 1.25× on newer OpenAI models; cache read
0.1× (0.05× / 0.025× on some models); batch 50% off; regional 1.1×; long-context tiers on some models.

**agentkit today.** `Usage` has input/output/cache read/write and cost; no reasoning field; a stale 9-model price
table; unknown models cost 0 silently; pricing hook private.

**Design.**
- Keep **raw usage per mode** as received, plus a **normalised** `Usage(input_uncached, input_cache_read,
  input_cache_write_5m, input_cache_write_1h, output, reasoning | None)`. Reasoning is nullable, never 0 when
  unknown.
- Per-call vs cumulative is decided per adapter and tested (Claude Code resume, Codex thread totals).
- `PriceTable`: versioned, dated, per provider/model, rates for every token kind and modifiers (batch, region,
  long-context tier, fast/priority tier). Loaded from a file the host supplies; agentkit ships a dated default.
- `cost_source = reported | price_table | unpriced`. An unpriced model is a refusal by default for paid modes,
  never a free call. A mode-reported cost is kept as `reported` and cross-checked against the table.

**Test.** Each mode's raw usage maps to the same normalised usage for the same scripted call; resume does not
double-count; an unpriced model refuses.

### C6 — Failures and retries

**Why.** The same word means different things: a 429 may be a rate limit (retry) or a spend cap (never retry);
some failures arrive after a 200; CLIs report failure on stdout, not via status.

**Modes (observed).** Anthropic: 429 `rate_limit_error` with `retry-after` vs spend-cap 429 with
`details.error_code: "enforced_spend_limit_reached"` and no `retry-after`; 529 `overloaded_error`; 500, 504
retryable; 400/401/402/403/404/413 not; a 400 is also returned when a user-set spend limit is reached; errors can
arrive mid-stream after 200. OpenAI: 429 rate limit / `slow_down` vs `insufficient_quota` and spend codes — branch
on `error.code`, not `type`; 503 `server_is_overloaded` with `Retry-After`; `response.failed`. Claude Code:
`system/api_retry` with an `error` category; exit code + `result.subtype`. Codex: `turn.failed` + exit code;
details on stdout; non-fatal `error` notices.

**agentkit today.** Substring `classify`; headers dropped (no `Retry-After` anywhere); CLI failures not
classified.

**Design.**
- `FailureClass = rate_limited | overloaded | transient_network | timeout | quota_exhausted | auth | invalid_request
  | refused | context_exceeded | tool_error | cancelled | unknown`, decided from structured fields (status, type,
  code, CLI subtype/category) — never message wording.
- `Failure(class, retryable, retry_after_s, mode_detail)`.
- One retry loop: full-jitter backoff, honour `retry_after_s` as a minimum within a cap, bounded attempts and
  total time. **No stacking**: when the mode already retries internally (CLIs, SDKs), the loop counts those
  attempts and does not multiply them; SDK auto-retry is disabled where agentkit owns the loop.
- A terminal-state classifier per adapter, independent of transport status ("success" with no payload is a
  failure; a Codex non-fatal `error` is not).

**Test.** Dropped connection, 429 with `Retry-After`, spend-cap 429, overloaded, 400, auth, mid-stream error,
CLI `turn.failed`, Claude `success` without payload — each classified identically across modes.

### C7 — Limits per call

**Design.** `CallLimits(max_cost_usd, max_turns, max_output_tokens, wall_clock_s, idle_s, first_event_s)` honoured
by all four. Codex has no turn or cost flag: count turn/usage events and stop the process. Claude Code
`--max-turns`/`--max-budget-usd` are used where they exist and checked independently. A limit hit is an
outcome (`truncated` / `limit_reached`) with partial output kept and resumable (C10), never a silent stop.

### C8 — Cancellation and timeouts

Keep the pinned commit's process-group kill, cancel polling and bounded drains. Add: Claude Code prefers
SIGINT/`interrupt()` (clean turn end) over SIGTERM (exit 143, no result, no cost recorded); Codex signal semantics
are undocumented — probe (§15). Timeouts explicit per mode (startup, first event, idle, total), including MCP
startup and tool timeouts. A cancelled call still emits `call_finished(cancelled)` with whatever usage is known.

### C9 — Credential homes and isolation

**Why.** CLIs read configuration, credentials, MCP servers and hooks from their home and working directory.
Claude Code `-p` loads a project's `.mcp.json` and hooks even in untrusted folders unless `--bare`; Codex reads
project config for trusted projects.

**Design.**
- `IsolatedHome`: one root per process holding an exclusive owner lock for the process lifetime; homes created per
  call under it; removed on exit, cancel and SIGTERM; orphaned roots (owner lock free) swept at start, credential
  files deleted first. Port Mirror's `disposable_home` / `sweep_orphaned_homes` / owner lock.
- API keys installed through the CLI's own login flow on stdin (`codex login --with-api-key`); session bundles
  materialised into the home; nothing from the user's own config inherited (`env_policy="isolated"`).
- CLIs started so they load **only** what agentkit gives them: Claude Code `--bare` + `--strict-mcp-config`;
  Codex `--ignore-user-config` and `--ignore-rules`. The effective configuration is read back
  (`system/init`: model, tools, MCP server status) and recorded.

### C10 — Sessions, resume and reasoning state

`SessionHandle(mode, run_id, cli_session_id, home_bundle)`; `resume=` on both entry points. API modes must
replay reasoning state exactly: Anthropic thinking blocks unmodified (bound to the conversation prefix on newer
models), OpenAI reasoning items. Any history rewrite breaks replay, so the runtime owns history and never edits
earlier turns. Usage across resumed sessions is reconciled, not summed (C5).

### C11 — Prompt caching

**Why.** Cost and latency depend on a stable prefix; small changes silently invalidate it.

**Design.** Fixed order: tools → stable system parts → volatile parts → messages; cache breakpoint after the last
stable block (§9). The runtime records a **prefix digest** per layer and emits `cache_invalidated(reason)` when a
tool definition, `tool_choice`, effort, thinking config, schema or image toggles change. Cache read/write counts
are always reported separately. Controls are mapped per mode (Anthropic breakpoints and TTL; OpenAI
`prompt_cache_key`; Claude Code TTL env) but the reported counts are uniform.

### C12 — Public hooks instead of subclassing

`request_hook(payload) -> payload` on the HTTP adapters (Mirror currently overrides private `_post`), a public
`pricing=` everywhere, and a public `result_hook` on the MCP server (§7). Nothing in Mirror should need a private
attribute of agentkit.

---

## 6. Model settings: temperature vs effort

The modes do not share a vocabulary. The APIs take sampling settings on some models and reasoning controls on
others; the CLIs take model and effort and expose no temperature.

### 6.1 What each mode accepts (observed unless marked)

| Setting | Anthropic API | OpenAI API | Claude Code | Codex |
|---|---|---|---|---|
| model | `model` | `model` | `--model` (aliases; precedence `/model` > flag > env > settings) | `-m` / `model` |
| temperature, top_p | 400 when non-default on Opus 4.7+ and Sonnet 5.5; accepted on older models | remove unless effort `none` on models that allow it (400s reported otherwise) | not exposed (inferred) | not in config reference |
| top_k | same as above | none | not exposed | not exposed |
| max output | `max_tokens` (required; includes thinking) | `max_output_tokens` (includes reasoning; hit → `incomplete`) | none; `--max-turns`, `--max-budget-usd` | none |
| reasoning depth | `output_config.effort` (`low`…`max`, per model) + `thinking` mode | `reasoning.effort` (`none`, `minimal`, `low`…`max`, per model) | `--effort`, `CLAUDE_CODE_EFFORT_LEVEL` (**clamped silently** to the highest supported level) | `model_reasoning_effort` (examples include `ultra`) |
| reasoning visibility | `thinking.display` | `reasoning.summary` | summaries setting | `model_reasoning_summary` |
| verbosity | none | `text.verbosity` | none | `model_verbosity` (`low`/`medium`/`high`) |
| speed / tier | fast mode (some models) | `service_tier` (Fast = former Priority) | none found | `service_tier` |

Model-specific rejections exist on the Anthropic API: `thinking.enabled` and `between_tools`/`disabled`
combinations differ per model; prefill is a 400 on 4.6+; forced `tool_choice` is a 400 on the newest models.
Defaults differ (e.g. effort default `medium` on some models, `high` on others).

### 6.2 Design

- `ModelSettings(model, effort, temperature, top_p, top_k, max_output_tokens, reasoning_summary, verbosity,
  tier)` — every field optional.
- A **capability table as data** keyed by (mode, model): accepted settings and their valid values, defaults, and
  incompatible combinations. A new model is a data change. The table is versioned and dated; re-probed on model
  upgrades.
- **Checked before dispatch.** A 400 after dispatch is a fallback, not the check.
- **Policy for an unsupported setting:**
  1. *Refuse* when it changes the meaning of the result and has no equivalent (e.g. `temperature=0` for
     reproducibility on a CLI mode or a model that rejects it).
  2. *Map and record a degradation* when a deterministic nearest equivalent exists (effort `xhigh`→`high` where
     missing; `none`→`low` where `none` is unsupported).
  3. *Drop and record* only pure presentation hints (summary display, tier).
  4. *Read back the effective settings* the mode actually used (Claude Code `system/init`; Codex thread) and
     record any silent clamp.
- Settings that change answers (model, effort, temperature, top_p/top_k) are part of the call's identity
  (reuse keys, eval comparisons); presentation hints are not.

---

## 7. Tools Mirror serves to agents

### 7.1 One registry, one server, identical on every mode

- One `ToolRegistry`. For the CLIs it is served through one MCP server (Streamable HTTP, stateless JSON mode,
  readiness check); for the APIs the same registry runs in-process. The **same result handling** applies either
  way.
- `serve(tools, result_hook=…, images=True, stateless_json=True)`, public `check_ready()` and timeouts.
  Mirror's `EngineServer`/`EvidenceServer` subclasses and the private-field cloning disappear.

### 7.2 Results

- Text, image, and structured results. MCP results carry `content[]` (text, image, audio, `resource_link`) and
  `structuredContent` validated against the tool's `outputSchema`; structured results also serialised as text.
- Images are first-class in the in-process loop too (Anthropic image blocks; OpenAI image arrays), not
  base64 markers smuggled through text.
- **Overflow:** results over a size are written to a file and replaced by a short notice naming the path and how
  to read it (Claude Code itself spills MCP results over 25,000 tokens / 50,000 characters — the runtime must
  set its own bound so every mode behaves alike).
- Execution errors are `isError: true` results fed back to the model; protocol errors are not.

### 7.3 Self-description

- Every limit is stated **in words** in the tool description and in the parameter's own description (Codex drops
  numeric schema keywords; Anthropic strict tools reject some). `describe_limits` renders `minimum/maximum/
  maxItems/maxLength/pattern` into text automatically; keywords stay in the schema where harmless.
- Every parameter has a description; descriptions say when to use the tool versus its sibling, how to batch,
  what comes back and how to continue.
- System prompts never restate tools (§9.5).
- A tool that cannot do exactly what was asked does the nearest useful thing (clamp, apply) or returns what the
  next call needs — never a bare refusal.

### 7.4 Large catalogues

Tool-selection accuracy degrades past roughly 30–50 tools (vendor guidance). Support **deferred loading** plus a
**tool search** tool (regex/BM25/embeddings): a few hot tools always loaded, the rest found on demand; names
namespaced by server/service; deterministic ordering so the cached prefix stays stable.

---

## 8. External MCP servers that users connect

Later, users will connect any MCP server and agents must know about and use its tools. The MCP spec leaves most
safety to the client, so this is a runtime responsibility.

### 8.1 Protocol support (revision 2026-07-28)

- **Stateless base protocol.** No `initialize` handshake; each request carries version and client capabilities in
  `_meta`; unsupported version → retry with the server's `supported` list. **Also support legacy servers**
  (`initialize`), detect by probing, and cache which era each server speaks.
- **Transports:** stdio and Streamable HTTP (single POST endpoint, JSON or request-scoped SSE replies, required
  `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name` headers). HTTP+SSE is deprecated.
- **Discovery:** `server/discover` (versions, capabilities, `instructions`, TTL). `serverInfo` and annotations
  are self-reported and untrusted.
- **Lists:** pagination and TTL caching for every `*/list`; change notifications only via a
  `subscriptions/listen` stream (`toolsListChanged`, resource subscriptions).
- **Server-to-client asks:** elicitation arrives as `InputRequiredResult` and is answered on retry (form mode for
  plain inputs; URL mode for secrets and third-party OAuth — never collect secrets in form mode).
- **Deprecated, do not build on:** sampling, roots, logging. Never declare the `sampling` capability (it would
  let a server spend the user's model quota).
- **Progress and cancellation:** request-scoped progress; closing the HTTP stream cancels; stdio sends
  `notifications/cancelled`; client timeouts with a hard maximum.
- **Skills over MCP** (`io.modelcontextprotocol/skills`, `skill://` URIs, digest manifests) — see §9.

### 8.2 Authorization

OAuth 2.1 for HTTP servers: Protected Resource Metadata discovery from the 401 `WWW-Authenticate`; RFC 8414/OIDC
discovery; client registration by pre-registered id, Client ID Metadata Document, or DCR (deprecated) — in that
order; PKCE; RFC 8707 `resource` on authorization and token requests; validate the issuer (RFC 9207 `iss`);
`insufficient_scope` step-up with bounded retries; tokens only in the `Authorization` header, only to the server
they were issued for; refresh tokens stored encrypted (Mirror's vault). stdio servers take credentials from the
environment the runtime gives them, never the parent's.

### 8.3 Security policy (client-side; the spec leaves it to us)

| Threat | Control |
|---|---|
| Tool poisoning / prompt injection in descriptions, annotations, `instructions`, results | treat all as untrusted; place third-party text only in tool results, labelled by source; descriptions/instructions truncated to a bound; optional screening classifier on results |
| Rug pull (definitions change after approval) | **pin** each tool by a digest of its full raw definition (name, description, schemas, annotations, `_meta`); on `list_changed` or reconnect compare; changed or new tools need re-approval; never swap definitions mid-run silently |
| Over-broad access | per-server and per-tool allow/deny (deny wins) matched on URL or command, never on the user-chosen label; approval modes per tool (always / prompt / writes-only / never) keyed on annotations only for trusted servers |
| SSRF via OAuth discovery URLs | HTTPS only; block private/link-local ranges; validate every redirect; pin DNS between check and use; optional egress proxy |
| Local server compromise | sandbox stdio servers (filesystem, network, env); show the exact command for consent; never inherit the parent environment |
| Context flooding | per-tool output cap with spill-to-file; a global context budget for MCP content |
| Confused deputy / token passthrough | per-server tokens with audience binding; never forward a client token downstream |
| Silent capability loss | a server marked required that fails or needs auth fails the call; optional servers' status is recorded |

Audit: every external tool call recorded (server, tool, argument shape, result size, approval decision).

### 8.4 Delivering external tools to the four modes

The four modes consume MCP differently (Claude Code and Codex from config files; the APIs accept remote
servers and tools only, with their own approval semantics). To keep one behaviour and one security policy:

**agentkit acts as the MCP client and gateway.** It connects to the user's servers, applies §8.1–8.3, and
re-exposes the approved, pinned tools through the same tool registry and server it uses for its own tools
(§7). The CLIs are started with only agentkit's server (`--strict-mcp-config`/`--bare`, Codex ignoring user
config); the API modes run the registry in-process. The vendors' native connectors are not used for
user-supplied servers, because each would apply a different policy.

---

## 9. Skills, agent definitions, instructions and dynamic prompts

A person (or an agent) creates an agent at run time: picks skills that make up its **persona**, gives it
**instructions** that let it run itself, and the runtime composes its **prompt** dynamically. The agent behaves
the same on all four modes.

### 9.1 Skills follow the open standard

The industry has converged on the **Agent Skills** format (agentskills.io), adopted by Claude Code, Codex,
Google ADK, and an MCP extension:

- A skill is a directory with `SKILL.md`: YAML frontmatter (`name` 1–64 lowercase/digits/hyphens, equal to the
  directory name; `description` ≤1,024 characters saying what it does and when to use it; optional `license`,
  `compatibility`, `metadata`, experimental `allowed-tools`) and a body; optional `scripts/`, `references/`,
  `assets/`.
- **Three disclosure levels:** metadata (~100 tokens, always visible), body (loaded on activation; keep under
  ~5,000 tokens), resources (on demand, one level deep).
- Vendor extensions exist (Claude Code: `when_to_use`, `disable-model-invocation`, `user-invocable`, `model`,
  `effort`, `context: fork`…; Codex: `agents/openai.yaml`, `allow_implicit_invocation`); unknown fields are
  ignored but logged.

**Design (S1).** agentkit's `Skill` becomes a loader for this format plus a source abstraction (filesystem,
MCP `skill://`, in-memory) and a digest manifest (sha256 per file, size). Mirror's richer manifest fields
(`roles`, `required_tools`, `playbooks` keyed by event kind) live in `metadata` or a Mirror extension file, so
skills stay portable. Mirror's `agent_skills` remains the store, approval and permission layer.

### 9.2 Activation (S2)

- **Pinned:** body included from the start.
- **Discoverable:** only metadata listed; the agent calls `load_skill(name)`; resources via
  `read_skill_resource(path)`. The listing is budgeted (Codex caps it at ~2% of context; Claude Code truncates
  descriptions) — order and truncate deterministically.
- **Triggered:** the host loads a playbook when an event matches (e.g. a failure class).
- `required_tools` checked against the agent's grant before activation; a skill needing a tool the agent lacks
  is refused with the reason.
- Approval binds to the whole manifest digest; any file change revokes it. Same-name skills from different
  sources never silently replace each other (key by source + URI).
- Skill content is untrusted input; scripts run only with explicit approval and in the sandbox; dynamic shell
  injection into skill bodies is disabled by default.
- Active skill bodies are re-attached after context compaction.

### 9.3 Agent definitions (S3)

Industry shape (Claude Code sub-agents / Agent SDK `AgentDefinition`, OpenAI Agents SDK `Agent`): a named record
with a description, a prompt, tools, a model, and limits.

```
AgentDefinition(
  id, version, digest,
  description,                      # when another agent should delegate to it (distinct from its prompt)
  persona: [SkillRef],              # pinned skills that define how it thinks and acts
  discoverable_skills: [SkillRef],
  tools: ToolGrant(allow, deny),    # deny applied first; patterns for MCP servers
  model: Binding | "inherit", settings: ModelSettings,
  limits: CallLimits,
  memory: MemoryRef | None,
  delegation: DelegationPolicy(allowed: [AgentRef], max_depth, max_concurrency, budget_share),
)
```

Created by a person or an agent through one API, validated (§9.6), stored versioned, and its digest recorded on
every call and in reuse keys. Model resolution order is explicit (invocation > definition > environment >
parent).

### 9.4 Instructions that let an agent run itself (S4)

```
InstructionSet(
  goal,
  definition_of_done,               # a machine-checkable artifact where possible (a checklist only the verifier flips)
  rules,                            # what it must and must not do
  stop_conditions,                  # turns, spend, time, "no progress for N slices"
  escalation,                       # what to record and return when it cannot finish
  orientation,                      # what to read first when resuming (progress notes, prior decisions)
)
```

- Given as the **task**, separate from the persona.
- The loop (`run`) ends on the definition of done or a stop condition; partial results are kept and resumable;
  progress notes live outside the context window.
- Done is decided by a check, not the model's claim. Explicit "double-check" wording is model-dependent (some
  current models over-verify when told to) — keep that wording in the prompt part, not in code.
- Every autonomous decision is recorded.

### 9.5 Dynamic prompt composition (S5)

The prompt is assembled from **parts**, each a locked, versioned template:

1. **Rules** — engine-owned; highest precedence; untrusted-content policy; never overridable.
2. **Persona** — from pinned skills.
3. **Skill index** — metadata of discoverable skills.
4. **Task** — the instruction set and its inputs.
5. **Context references** — ids/paths the agent reads through tools; bodies are never pasted.

Rules for composition:

- Stable parts first, volatile last; the cache breakpoint after the last stable part (C11). Tool definitions
  precede the system prompt in the cache prefix, so tool changes are tracked separately.
- **Tools are never restated** in any part; their definitions describe them (§7.3).
- Third-party content (tool results, skill files from external sources, MCP text) is never placed in the system
  prompt.
- The composition produces a **manifest** (part id @ version @ digest, inputs digest); its digest is the
  prompt's identity for reuse and comparison.
- Strict rendering (undefined variables raise; leftover markers refused).
- Size is measured and fitted: split into parts or passed by reference — never truncated or dropped.

### 9.6 User-authored personas and instructions (S6)

Personas and instructions written by a person are **data**: bounded length, screened by guardrails (no secrets,
no verdict claims), recorded verbatim with their digest. Precedence is structural: user parts cannot redefine
tools, limits, fences or the rules part, because those are not expressed in the prompt at all — they are
enforced by the runtime.

### 9.7 Delegation (S7)

- Two patterns: **agent-as-tool** (manager keeps control; child returns one message) and **handoff** (a peer
  takes over the conversation). Support both.
- A child starts with a fresh context containing only its definition and an explicit briefing (objective, output
  format, tools, boundaries). It inherits nothing else.
- Depth, concurrency and spend caps are enforced by the runtime, not only stated in prompts (current models
  delegate eagerly).
- A child's final message is untrusted input to the parent (scan for imitated control tags and turn markers).
- Child events carry the parent's call id for attribution.

### 9.8 Four modes (S8)

Claude Code has native skills and sub-agents; use them only where the observable behaviour matches the
contract. Codex and the APIs receive the composed prompt plus `load_skill`/`read_skill_resource`/`delegate`
tools. A conformance scenario per concept proves identical behaviour: which skill loaded, which playbook,
what the agent saw, what it did.

---

## 10. Observability

- **Spans** follow the OpenTelemetry GenAI conventions (repo `semantic-conventions-genai`, status Development —
  pin the version): `invoke_agent {name}`, `execute_tool {tool}`, skill-load/resource spans, `create_agent`;
  attributes `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`,
  `gen_ai.request.{temperature,top_p,max_tokens}`, `gen_ai.response.{model,id,finish_reasons}`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, cache read/write subsets,
  `gen_ai.tool.{name,call.id,type}`, `gen_ai.skill.{name,resource.name,source.uri}`, `gen_ai.agent.{id,name,
  version}`, `gen_ai.conversation.id` (only when the caller supplies one), `error.type`.
- **Cost** has no standard attribute: use a clearly namespaced one (e.g. `agentkit.cost.usd`,
  `agentkit.cost.source`, `agentkit.price_table.version`).
- **Content capture** (messages, system instructions, tool arguments/results) is opt-in, separately for
  arguments and results, masked and bounded.
- **Events** (§5 C4) go to an `EventSink`; Mirror maps them into its journals (bounded, masked) and the UI Logs
  tab filters and totals by their fields.

---

## 11. Public API

```text
# Binding and settings
Binding(mode: Literal["claude_code", "codex", "anthropic_api", "openai_api"], model: str, credential: CredentialRef)
ModelSettings(effort=None, temperature=None, top_p=None, top_k=None, max_output_tokens=None,
              reasoning_summary=None, verbosity=None, tier=None)
CallLimits(max_cost_usd=None, max_turns=None, max_output_tokens=None, wall_clock_s=None, idle_s=None,
           first_event_s=None)

# Results
Answer[T](value: T | None, text: str, messages: tuple[str, ...], outcome: Outcome, usage: Usage,
          raw_usage: Mapping, cost_usd: Decimal | None, cost_source: CostSource, effective: EffectiveSettings,
          degradations: tuple[Degradation, ...], attempts: int, failure: Failure | None,
          session: SessionHandle | None, prompt_manifest: Manifest)

# Entry points
async def ask(binding, prompt: ComposedPrompt, output: OutputSpec | type[T], *, settings=ModelSettings(),
              limits=CallLimits(), cancel=None, events: EventSink | None = None, pricing: PriceTable | None = None,
              call_id: str) -> Answer[T]                                   # one-shot

async def run(binding, prompt: ComposedPrompt, tools: ToolSet, *, output=None, settings=ModelSettings(),
              limits=CallLimits(), turns_per_slice=None, resume: SessionHandle | None = None, cancel=None,
              events=None, pricing=None, call_id: str) -> Answer[T]        # agent loop

async def run_agent(definition: AgentDefinition, task: InstructionSet, inputs: Mapping, *, resume=None,
                    cancel=None, events=None, call_id: str) -> Answer[T]   # composes, activates skills, runs

# Composition
def compose(parts: Sequence[PromptPart], inputs: Mapping) -> ComposedPrompt   # manifest + digest + rendered text

# Tools and MCP
ToolSet(registry, overflow: OverflowPolicy, images=True, describe_limits=True, deferred: DeferPolicy | None)
def serve(tools: ToolSet, *, ctx) -> AsyncContextManager[McpServerSpec]
class McpGateway:                    # external servers (§8)
    async def connect(server: ServerConfig, policy: McpPolicy) -> ConnectedServer
    def tools(approved_only=True) -> ToolSet

# Skills and agents
class SkillSource(Protocol): list() / get(ref) / read(ref, path)    # filesystem, MCP skill://, memory
class AgentStore(Protocol): get(ref) / put(definition) -> validated, versioned

# Helpers (public, testable)
def compile_schema(schema, mode, model) -> dict;  def clean_nulls(value, schema) -> Any
def classify(exc_or_event) -> Failure | None;      def price(model, usage, table) -> (Decimal | None, CostSource)
```

---

## 12. Proof: conformance suite and the eval matrix

**In agentkit — `tests/conformance/`.** The same scripted scenarios run against a scripted fake of each mode
and assert identical observable results:

| Scenario | Proves |
|---|---|
| two messages then a final answer | C1 |
| optional fields inside `anyOf`/`$ref`; an inexpressible bound; refusal; truncation; CLI success without payload; correction turn | C2 |
| unsupported temperature on a CLI; effort clamp; incompatible thinking combination | §6 |
| identical event sequence | C4 |
| raw usage → normalised usage; resume without double counting; unpriced model | C5 |
| dropped connection; 429 + `Retry-After`; spend-cap 429; overloaded; 400; auth; mid-stream error; CLI `turn.failed` | C6 |
| turn, cost, output and wall-clock limits | C7 |
| cancel mid-call kills the process tree and reports usage | C8 |
| killed owner's home swept; project MCP config ignored | C9 |
| resume a sliced session; reasoning state replay | C10 |
| a tool change emits cache invalidation | C11 |
| image result; oversized result spilled; limits in words; tool error vs protocol error | §7 |
| external server: changed tool definition requires re-approval; deny list wins; SSRF URL refused; required server failing fails the call | §8 |
| skill activation (pinned, discoverable, triggered); missing required tool; digest change revokes approval; delegation caps | §9 |

Plus one opt-in live test per mode (skipped without credentials) running the same scenario set.

**In Mirror — the eval matrix.** `evals run --matrix` runs every prompt case on M1–M4 with real evidence,
per-answer caps, per-model thresholds only with a written reason, and a per-model report and baseline. A stage
is not ported until its cases pass on all four.

---

## 13. What stays in Mirror's `cognition`

- Routing an organisation's binding to a mode and model; vault credential references.
- The spend ledger (announce before the call, settle after) and Mirror's authoritative price file fed to
  agentkit's `PriceTable`.
- Per-stage budget and retry *policy*; tool-aware liveness bounds; repair policy (what to re-ask with).
- Mirror's event schema, journals and the Logs tab mapping.
- `prompting` (strict render, locks, guardrail cases) for every prompt part.
- Plane rules: sounding never dispatches; control's harness only in its providers seam.
- Mirror's skill store, approvals and the agent store.

---

## 14. Order of work and migration

**In agentkit** (each step merged with its conformance scenarios green):

1. Rebase on `555bc57`/remote `main`; tag current behaviour.
2. C1 last-message answer and C2 structured output — the two causing live failures today.
3. C6 failures and `Retry-After`; C5 usage and pricing; §6 settings and capability table; C7 limits.
4. §7 tool serving (result hook, images, overflow, limits in words, deferred loading).
5. C8–C11: cancellation details, homes, sessions, caching; C12 hooks; §10 observability.
6. §8 MCP gateway for external servers.
7. §9 skills (open standard + sources), agent definitions, instruction sets, composition, delegation.
8. Opt-in live conformance per mode.

**In Mirror** (one consumer at a time; each ported stage must pass the eval matrix):

1. Bump the agentkit pin; build `cognition.ask/run/run_agent` on it.
2. Port **preflight** (one-shot proposal prompts with existing evals) — first consumer.
3. Control chat; assay (research, plan, design, drive); forge build lanes; the eval harness.
4. Delete each old path once unused: `kernel/authority/agent_adapter.py`, both `adapter_contract.py`, the five
   `_build_env` overrides, `_widen_stream_lines`, `EngineServer`/`EvidenceServer`, the three `_table_cost`
   copies, control's `codex_agent.strict_json_schema`, the separate retry loops.
5. Expose user-connected MCP servers and user-defined agents once §8 and §9 are in.

---

## 15. Open questions to settle with live probes

These were not established from documentation and must be measured once, recorded in the capability table, and
re-checked on upgrades:

- OpenAI strict mode: which numeric/length keywords base models accept today.
- Anthropic: sampling acceptance on Opus 5, Fable, Mythos and Haiku 5.5; non-strict tool schema keyword handling.
- OpenAI: which models accept temperature at effort `none`; `text.verbosity` values; `service_tier` values; where
  usage appears in a Responses stream; cancel and `previous_response_id` semantics.
- Codex: exit codes; whether `turn.completed.usage` is per-turn or thread-cumulative; behaviour on an unsupported
  effort; signal/cancel behaviour; whether `--output-schema` works with `resume`; image handling in MCP results.
- Claude Code: whether it honours output and turn caps cleanly; the full flag/env list (confirm no temperature);
  how MCP images and `structuredContent` reach the model; cache TTL breakdown in result usage.
- Whether Claude Code's native skills/sub-agents match the contract closely enough to use natively.

---

## 16. Sources

Anthropic: structured outputs, errors, rate limits, streaming, pricing, prompt caching, tool use (handle tool
calls, strict tool use, tool search), stop reasons, effort, extended thinking, Opus 5.5 and Sonnet 5.5 migration
guides, mitigating jailbreaks and prompt injection, Agent Skills overview and best practices, MCP connector —
`https://platform.claude.com/docs/en/...`. Engineering: effective context engineering, effective harnesses for
long-running agents, multi-agent research system, advanced tool use — `https://www.anthropic.com/engineering/...`.

Claude Code and Agent SDK: headless, model configuration, MCP, managed MCP, skills, sub-agents, SDK cost tracking,
structured outputs, TypeScript reference, SDK MCP, SDK skills, SDK sub-agents — `https://code.claude.com/docs/en/...`.

OpenAI: structured outputs, function calling, streaming responses, error codes, rate limits, prompt caching,
reasoning, latest model, pricing, remote MCP and connectors — `https://developers.openai.com/api/docs/...`;
openai-python types (`response_stream_event.py`, `response_usage.py`, `completion_usage.py`); OpenAI Agents SDK
(agents, handoffs, guardrails, tracing) — `https://openai.github.io/openai-agents-python/`.

Codex: non-interactive mode, MCP, config reference, skills, AGENTS.md — `https://learn.chatgpt.com/docs/...`;
source `codex-rs/exec/src/exec_events.rs` and `event_processor_with_jsonl_output.rs` on `main`. Third-party,
treated as reported only: codex.danielvaughan.com JSONL, exit-code and output-schema notes;
pathscale/RustAgentAbstraction PR 10.

MCP: specification revision 2026-07-28 (transports, Streamable HTTP, authorization, versioning, discover, tools,
resources, prompts, cancellation, sampling, elicitation, deprecated features), schema.ts, security best practices,
Skills extension, registry — `https://modelcontextprotocol.io/...`. Security notes treated as reported: CSA
research note on tool poisoning; tool-poisoning threat model (trackr.live); Zuplo on rug pulls.

Skills standard: `https://agentskills.io/specification`. Google ADK skills guide; Pydantic AI harness skills;
LangChain supervisor migration (reported).

Observability: OpenTelemetry GenAI semantic conventions — `https://github.com/open-telemetry/semantic-conventions-genai`.

Mirror and agentkit code references in this document were read from the Mirror working tree and the pinned
agentkit commit `555bc57` on 2026-10-13.
