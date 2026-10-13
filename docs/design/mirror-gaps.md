# What agentkit `f8d20c1` lacks for Mirror

A verified gap list. It covers what Mirror's model-call stack does today, which agentkit at `f8d20c1` does not
yet do, and what to build so that every Mirror model call (Claude Code CLI, Codex CLI, Anthropic API, OpenAI
API) can go through `CallRuntime`. It was checked against both codebases on 2026-10-13.

The companion to `model-neutral-agents.md`, the design. This file is the distance between that design and
this commit, measured against a real consumer.

**Path prefixes:**
- `A:` is `agentkit/` in this repository.
- `M:` is `/Users/dododuck/Desktop/work/mirror/`.

**Original status:** **X** absent, **P** partial at the inspected revision `f8d20c1`.
The working-tree implementation status is recorded after the original gap descriptions below; those
labels do not change the source snapshot this gap analysis documents.

## The short version

### Working-tree implementation status (2026-10-13)

These statuses describe this working tree, not a release or a claim of parity with Mirror. “Implemented”
means the behavior has a source implementation and focused offline regression coverage. It does not mean
the real-world four-mode matrix has proven every behavior.

| Gap | Status | Current evidence or remaining work |
|---|---|---|
| G1 credential injection | Implemented | `Credential` supports environment and allowlisted home files; Claude OAuth/bearer credentials now select a non-bare CLI invocation while direct API keys retain bare mode. Both paths have focused conformance tests. |
| G2 credential refresh | Implemented | Changed files are returned once before cleanup; callback failure degrades without discarding the answer; cancellation cleanup is covered. |
| G3 CLI classification/retry | Partial | Both CLI parsers retain structured status/code/retry-after facts; fake-driver 429 recovery and 400 no-retry twins pass. A new real-provider rejection eval confirms 401/auth classification and one-attempt behavior on all four backends at $0 spend. A real 429/Retry-After response remains unobserved. |
| G4 executable/environment | Implemented | Per-mode executable and validated non-secret environment overrides are available. |
| G5 mid-call cost/turn caps | Partial | Claude budget and turn limits are passed through. Installed Codex CLI 0.162.1 help exposes neither a max-budget nor a max-turns control; its cost is observable only after a turn, so agentkit cannot promise a hard mid-call ceiling there. |
| G6 pricing policy | Implemented | TOML/JSON loading, derived rates, unpriced policy, and partial mixed-cost settlement are covered offline. |
| G7 CLI admission | Implemented | Per-runtime/per-mode concurrency limit and host admission hook are available. |
| G8 output decoding/repair | Implemented | Bounded tolerant JSON extraction, configurable corrections, and output validators have regression coverage. |
| G9 long schema off argv | Partial | Claude Code has no documented file-backed `--json-schema` form. Schemas above the safe inline-argv threshold use the documented system-prompt-file path plus strict agentkit local validation/correction; a live 40+ KB schema case passes. Schemas above the 256 KiB fallback bound fail closed. |
| G10 attempt cost events | Implemented | Attempt start/finish events include usage and cost provenance. |
| G11 shared login concurrency | Host-owned | Intentionally remains host responsibility; G1/G2 provide the per-call credential boundary. |
| G12 read-only credentials | Implemented | Omit `on_changed` to discard local refreshes. |
| G13 home placement and signal cleanup | Implemented | Configurable parent, normal/cancellation cleanup, and chained SIGTERM/SIGINT cleanup have focused tests. |
| G14 workspace/native tools | Implemented | Per-call Claude permissions/tools/hooks and Codex sandbox/paths are verified by fake CLI argv tests; incompatible mode fields are refused. |
| G15 MCP tool timeout | Implemented | CLI timeout derives from the tool timeout with a margin. |
| G16 tool arguments/events | Implemented | API and native CLI tool events carry masked arguments, result digests/sizes, and optional host-masked previews bounded to 512 bytes. |
| G17 images | Implemented | API, Codex and Claude Code image-result cases pass live checks; Claude Code MCP image serialization remains covered by the round-trip regression. The earlier failing report is retained as historical evidence. |
| G18 long CLI sessions | Partial | Claude supports turn-bounded slices, resume/fork and stable-prefix flags; Codex supports fork. Codex exposes no verified turn-slice control, and stable-prefix behavior is Claude-specific. |
| G19 first-event restart | Implemented | A first-event timeout can retry; idle timeout after output remains terminal. Identical-retry suppression stays host-owned. |
| G20 CLI request hook | Implemented | Hook can replace validated argv while preserving the configured executable. |
| G21 MCP skill source | Implemented | Pinned MCP skill resources are supported. |
| G22 persona screening | Implemented | Optional host screening hook runs before composition. |
| G23 native Claude subagents | Partial | Added fail-closed conversion from public `AgentDefinition` plus pinned persona resources to `--agents`; native skill installation and live delegated-agent conformance remain open. |
| G24 scripted runtime fake | Implemented | Public scripted adapter/runtime factory is available with focused tests. |
| G25 masked/bounded event sink | Implemented | `MaskingSink` applies host masking and bounded event values. |

The earlier offline suite passed **5,958 tests with 98 skips**. After the final regression additions,
the full offline suite passes **5,961 tests with 98 skips**. Ruff, strict typing and whitespace checks pass.
A bounded live run on
`image_result` and `model_switch` passed all seven `model_switch`
rows, and passed `image_result` on Anthropic API, OpenAI API and Codex. Claude Code failed `image_result`
before invoking its MCP tool; the retained report is
[`20261013T142237440683Z`](../../tests/evals/reports/20261013T142237440683Z/summary.json). A focused Claude-only
rerun after adding MCP image serialization failed the same way. Debugging showed
`spec.cli_kwargs()` already supplied `strict_mcp_config`, while `CliAdapter` passed that argument a
second time; the cognition constructor raised before spawning. The duplicate was removed, and a new
offline regression covers one strict config flag. Live `tool_evidence` and `image_result` both pass
on Claude Code alongside the other backends in
[`20261013T151551366051Z`](../../tests/evals/reports/20261013T151551366051Z/summary.json).

The all-mode `provider_auth_rejection` eval uses a fresh deliberately invalid credential. Each backend
returned an authentication failure in one attempt with zero observed usage and zero accounted spend;
the report is [`20261013T152207836480Z`](../../tests/evals/reports/20261013T152207836480Z/summary.json).
This validates real provider rejection shapes and confirms authentication failures do not retry.
Transient 429/Retry-After still has offline parser/retry coverage only; no controlled provider 429 was
induced. Claude Code's live long-schema fallback passed in
[`20261013T152051324301Z`](../../tests/evals/reports/20261013T152051324301Z/summary.json).

The public additions are documented here as well as in the API reference: `Credential`, `Workspace`,
`ImagePart`, `ClaudePermissionMode`, `CodexSandboxMode`, and `McpSkillSource`.

---

- **The two API modes are close.** Retries, strict schemas with null cleaning, pricing, events and OTel
  exist and work.
- **The two CLI modes cannot carry Mirror today.** Three gaps block them outright:
  1. **G1.** A signed-in CLI login cannot be supplied. Credentials arrive only as env vars, and `--bare`
     disables Claude subscription OAuth.
  2. **G2.** A login the CLI refreshes is deleted rather than handed back, so the stored copy dies.
  3. **G3.** CLI failures are never classified and never retried.
- **Gaps that hold back the build and drive lanes but not preflight:** native tools and per-call workspace
  (G14), tool arguments in events (G16), and long sessions (G18).

---

## Tier 0: blocks every CLI call

### G1. A CLI login cannot be supplied (X)

**Mirror today:**
- Logins are zipped session bundles holding Claude's credential files
  (`M:packages/kernel/kernel/authority/session.py:230-275`; the file set is in `session_group.py:32-36`).
- Codex runs from either its bundle or `codex login --with-api-key`
  (`M:assay/platform/providers/configured_agent.py:498-536`).

**agentkit:**
- `CredentialRef.resolve()` returns a `Mapping[str, str]` of env only (`A:models/contracts/types.py:19-23`).
- The CLI adapter maps it to env (`A:models/providers/cli/calls.py:163-182`).
- `CliAdapter` forces `bare=True` (`cli/calls.py:204`), and bare mode disables subscription OAuth
  (`A:models/providers/cli/claude/driver.py:372-373,386`).
- `IsolatedHome.restore()` accepts only `sessions/` and `projects/*.jsonl` (`cli/isolation.py:113-122`).
- Codex gets a fresh home with no `auth.json` (`isolation.py:77-80`). Only the lower-level
  `CodexCliCognition(config_home=…)` can use one (`codex/driver.py:401,1504-1505`).

**Build:**
- `CredentialRef` resolves to `Credential(env: Mapping[str,str] = {}, home: Mapping[str, bytes] = {})`.
- `home` files are laid into the isolated home before spawn with mode 0600, from an **allowlist per mode**:
  - Claude: `.credentials.json` and the minimal settings it needs;
  - Codex: `auth.json`.
- Anything outside the allowlist is refused before spawn, as an invalid request.
- When `home` carries a Claude login, the adapter runs **without `--bare`** but keeps every isolation `--bare`
  bought:
  - `--strict-mcp-config`;
  - no user settings;
  - no hooks;
  - no auto-memory.

  Make that an explicit option set rather than the single `bare` switch.
- `resolve` becomes `async`. It is called inside async code today (`cli/calls.py:163`,
  `frontiers/calls.py:48`), and a host that releases from a vault over HTTP cannot block there.

**Prove:** per CLI, a fake that refuses to start unless the expected file is in its home at 0600. The twin is
the env-only path, which still works. A file outside the allowlist is refused before spawn.

### G2. A refreshed login is never handed back (X)

**Mirror today:** seals the CLI home after the call and sends it back to the vault (`finish`,
`M:packages/kernel/kernel/authority/session_store.py:657-679`). A borrower's refreshed token travels back to
the group (`session_group.py:179`).

**agentkit:** deletes the credential files at close (`A:models/providers/cli/isolation.py:24-27`).
`snapshot()` exports transcripts only (`:83-101`).

**Build:**
- `Credential` carries `on_changed: Callable[[Mapping[str, bytes]], Awaitable[None]] | None`.
- After the CLI exits, and **before** deletion, compare the allowlisted files with what was laid in. If any
  changed, await `on_changed(changed)` exactly once.
- If the callback raises, the call's answer is kept, a `degraded` event records the failure, and the call
  still ends normally.
- Deletion happens after the callback, on every exit path, cancellation included.

**Prove:** a fake CLI that rewrites its credential file triggers one callback with the new bytes. The twin is
an unchanged file, which triggers none. A cancelled call still deletes the files after the callback.

### G3. CLI failures are unclassified and never retried (X)

**Mirror today:** retries `api_error`, `rate_limit`, `overloaded` and `cli_reported_error` on CLI calls
(`M:assay/platform/providers/model.py:216-225,1104-1119`). It detects auth failures from CLI output
(`model.py:202-213`), which is what makes the refresh-race retry possible
(`M:assay/platform/providers/execution.py:281-299,464-476`).

**agentkit:**
- CLI failures are always `UNKNOWN` or `CANCELLED` with `retryable=False` (`A:models/providers/cli/calls.py:284-307`).
- The retry loop stops for any non-API mode (`A:core/runtime/calls.py:481-483`).

**Build:**
- Classify the CLI's terminal event with the same `classify` used for APIs:
  - Claude: `api_error_status`, `api_retry`, result subtype;
  - Codex: `turn.failed` and `error.code`.
- Map auth failures to `FailureClass.AUTH`.
- Apply `RetryPolicy` to CLI modes. A retry is a new process in a fresh home with the credential resolved
  again, which is what makes the refresh-race case recover.
- Emit `retry` with `source="transport"` and the class.

**Prove:**
- a scripted 429 with retry-after, then success: one retry, after at least the advised wait;
- the twin is a 400, which is never retried;
- an auth failure is classified `AUTH` and retried only once, after `resolve` returns different credentials.

### G4. The executable and environment are fixed (X)

**Mirror today:**
- passes its own `claude` binary (`M:assay/platform/providers/model.py:1018`);
- sets `MCP_TOOL_TIMEOUT`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY` and `FORCE_PROMPT_CACHING_5M`
  (`M:forge/platform/providers/claude_cli.py:897-905`, `model.py:93`).

**agentkit:**
- `CliAdapter(spawn, working_dir, native_tool_policy)` only (`A:models/providers/cli/calls.py:28-34`);
- the isolated env policy admits only the credential and passthrough variables (`_common.py:501`).

**Build:**
- `CliAdapter(executable: dict[Mode, str] | None, env: Mapping[str,str] | None)`;
- `env` is merged after isolation and validated against a denylist, so credentials can only arrive through
  G1.

---

## Tier 1: blocks correct one-shot calls and correct money

### G5. Mid-call cost and turn caps (X)

- **Claude:** `--max-budget-usd` is never sent. `meter_spend=False` (`A:models/providers/cli/calls.py:185`)
  makes `_budget_cap` return `None` (`claude/driver.py:1281`). The ceiling built at `cli/calls.py:152-162` is
  unused, so `max_cost_usd` is only checked after the turn (`A:core/runtime/calls.py:521-525`). Mirror relies
  on the mid-call cap (`M:forge/platform/providers/claude_cli.py:539-542`). The installed Codex
  CLI 0.162.1 exposes no max-budget or max-turns option in `codex exec --help`; agentkit reports
  usage after the turn but cannot stop that process at an exact spend threshold.
- **Codex:** has no budget flag (`codex/driver.py:551-555`). `max_turns` is passed to Claude only
  (`cli/calls.py:206`). The core's `max_turns` counts adapter turns (`core/runtime/calls.py:349`), and a CLI
  run is one.

**Build:**
- Claude: send `--max-budget-usd` from `CallLimits.max_cost_usd`.
- Codex: meter `usage` events as they stream, price them with the call's `PriceTable`, and stop the process
  group when the running cost crosses the cap. Count turns from turn events and stop at `max_turns`.
- In both cases the outcome is `limit_reached`, with the partial usage settled.

**Prove:** scripted usage events that cross the cap mid-stream stop the process before the next event. The
twin is a stream under the cap, which runs to completion.

### G6. Pricing policy is fixed to refusal (P)

**Mirror today:**
- prices come from **TOML** (`M:packages/kernel/kernel/authority/model_prices.toml`);
- cache rates are derived from the input rate (`M:packages/kernel/kernel/authority/agent_adapter.py:544-546`);
- an unpriced model proceeds and is marked `unpriced` (`:562-590`).

**agentkit:**
- `PriceTable.load` reads JSON (`A:models/catalog/pricing.py:103-110`);
- every token kind needs an explicit rate (`:131-134`);
- an unpriced model is refused before dispatch (`A:core/runtime/calls.py:339-343`);
- an unpriced token kind fails **after** the money is spent (`:516-520`);
- the bundled rows do not include the models Mirror runs.

**Build:**
- `UnpricedPolicy = REFUSE | PROCEED_UNPRICED`, set per runtime.
- `PROCEED_UNPRICED` dispatches, reports `cost_source=unpriced`, and is only allowed where the caller sets a
  turn or token cap.
- An unpriced token kind **after** dispatch never fails the call. It reports `cost_source=mixed` with the
  priced part.
- `PriceTable.derive(input=…, output=…, cache_read_ratio=…, cache_write_ratio=…)` covers providers that
  publish ratios.
- `PriceTable.load` accepts TOML as well as JSON.

**Prove:** an unpriced model refused under `REFUSE`, dispatched and marked under `PROCEED_UNPRICED`. A
missing reasoning rate after dispatch yields `mixed`, not `failed`.

### G7. No real bound on concurrent CLI processes (X)

**Mirror today:** bounds spawns by a measured admission ceiling (`M:assay/platform/providers/model.py:1029`).

**agentkit:** the semaphore is keyed on `config_dir` (`A:models/providers/cli/_common.py:77-90`), and
`config_dir` is unique per call (`cli/calls.py:203`), so the semaphore never binds.

**Build:**
- one semaphore per `(runtime, mode)`, sized by a `CallRuntime(max_concurrent_cli=…)` the host passes;
- an `acquire` hook so a host can supply its own admission (Mirror's measured memory and CPU window).

### G8. Answer decoding and repair are too narrow (P)

**Mirror today:**
- accepts a fenced block, a stray trailing brace, and the last JSON object in the last message
  (`M:forge/platform/providers/claude_cli.py:2823-2876`,
  `M:control/tenancy/adapters/system/providers/chat_agent.py:851-855`);
- configures the number of repairs, and adds validators that check evidence references
  (`M:assay/platform/providers/model.py:910-918`).

**agentkit:** `json.loads(text)` as the fallback (`A:core/runtime/calls.py:605`), and exactly one
schema-only correction (`:612`).

**Build:**
- A tolerant extractor, applied only to the **last** message:
  - strip one fence;
  - take the last balanced top-level object;
  - drop one trailing unmatched brace.
- `OutputSpec(max_corrections=1, validators=[Callable[[value], list[str]]])`. Validator errors are fed back
  into the correction turn exactly like schema errors.

**Prove:** each tolerated form decodes. The twin is two objects in the last message with no clear last one,
which fails as invalid rather than guessing. A validator error triggers a correction turn.

### G9. A long schema goes on argv (X)

The installed Claude Code 2.1.296 help and current CLI reference expose `--json-schema <schema>`
as inline JSON but no file-backed schema option. The adapter therefore does not invent a
`--json-schema-file` flag. For schemas above 32 KiB and up to 256 KiB, it uses the documented
`--append-system-prompt-file` mechanism to provide schema guidance, then the core parser, validator
and correction turn enforce the actual contract. This is an explicit degradation from provider-native
constrained output, surfaced as a `structured_output` degradation event. The larger-schema live case
validates this path. See Anthropic's [CLI reference](https://code.claude.com/docs/en/cli-usage) and
[structured output guide](https://code.claude.com/docs/en/agent-sdk/structured-outputs).

**Mirror today:** keeps the schema off argv when it is too long (`M:packages/kernel/kernel/authority/schema_fit.py`).

**agentkit:** `--json-schema` goes straight onto argv (`A:models/providers/cli/claude/driver.py:1423`).

**Build:** pass the schema through a file in the isolated home when its encoded size crosses the platform's
argv room. Measure that from `os.sysconf('SC_ARG_MAX')`; never hardcode it.

### G10. No per-attempt cost event (P)

**Mirror today:** announces and settles spend per attempt
(`M:forge/platform/providers/claude_cli.py:404-462,722`; control's `claude_agent.py:631`).

**agentkit:** `call_started` and `call_finished` exist and are awaited (`A:core/runtime/calls.py:210-220,696-703`).
A retried call reports one total.

**Build:** `attempt_started(attempt, mode, model)` and `attempt_finished(attempt, outcome, usage, cost_usd,
cost_source)` events between them, so a ledger can settle every attempt that spent money.

---

## Tier 2: blocks the build and drive lanes (not preflight)

### G11. Shared login across concurrent calls (X)

**Mirror today:** many concurrent calls share one vault lease by borrowing snapshots of one CLI home
(`M:packages/kernel/kernel/authority/session_store.py:391-445`, `session_group.py:88-249`).

**Decision before building:** it may belong in the host rather than in agentkit. With G1 and G2, a host can
serialise per login, or hand out snapshots and merge refreshed tokens itself. Recommended: keep it in the
host, and have agentkit only guarantee that G2's callback fires once per call with the final files.

### G12. Read-only credential mode (X)

**Mirror today:** chat uses a snapshot that may refresh locally but never writes back
(`M:packages/kernel/kernel/authority/session_store.py:369-389`).

**Build:** this falls out of G2: `on_changed=None` means discard.

### G13. Home cleanup and placement (P)

- Cleanup is registered with `atexit` only (`A:models/providers/cli/isolation.py:75`). A SIGTERM from a
  container stop leaves credential files on disk until the next sweep.
  - **Build:** delete credential files from a SIGTERM/SIGINT handler that chains to any existing one, which is
    what Mirror's `session.py:107-124` does.
- `IsolatedHome(parent=)` exists (`isolation.py:62`), but `CliAdapter` constructs it without a parent
  (`cli/calls.py:117`).
  - **Build:** `CliAdapter(home_parent=Path)`.

### G14. Native tools, workspace and permissions are fixed (X)

**Mirror today:**
- build lanes run Claude Code with Read, Edit and Bash behind a `PreToolUse` fence, with read receipts
  (`M:assay/platform/providers/model.py:1007-1030`, `evidence_reads.py`,
  `M:forge/platform/providers/claude_cli.py:884-917`);
- the working directory, extra dirs, settings and permission mode are set per call (`claude_cli.py:1283,1394-1402`).

**agentkit:**
- Claude: no native tools, bare (`A:models/providers/cli/calls.py:198-209`);
- Codex: read-only sandbox, shell off (`:212-236`);
- the working directory is fixed per adapter (`:37,171`).

**Build:** a per-call `Workspace(working_dir, add_dirs, permission_mode, native_tools: set[str],
hooks: HookSpec | None, sandbox: CodexSandbox | None)` on `run()`. The hook spec is passed through as a
generated settings file, so a host fence (Mirror's `agent_hooks`) runs as the CLI's own `PreToolUse` hook.
Native tool use is reported in events (G16).

### G15. The CLI's MCP tool timeout is not derived (P)

`codex_kwargs()` is called without a timeout, and Claude's `MCP_TOOL_TIMEOUT` is never set
(`A:models/providers/cli/calls.py:226-230`). Both should come from `ToolSet.timeout_s`
(`A:tools/execution/policy.py:90`) plus a margin, so the CLI never times out a tool the server is still
allowed to run.

### G16. Tool-call arguments are missing from events (X)

**Mirror today:** records each tool call's arguments in build events, with file bodies recorded as shape only
(`M:forge/…/stream_events.py:295-351`). It reports what Codex read as evidence (`shell_read_paths`,
`M:assay/platform/providers/configured_agent.py:407-428`).

**agentkit:**
- API mode carries `argument_digest` only (`A:core/runtime/calls.py:538-543`);
- CLI mode carries the tool name only (`cli/calls.py:250-251`);
- full arguments end up only in `Answer.context` (`:100-114`).

**Build:**
- `tool_call` events carry `arguments` passed through a host `ArgumentMasker(tool, args) -> args` (the
  default replaces string values over a size with `{len, sha256}`);
- `tool_result` carries `size`, `overflowed` and an optional masked preview;
- the same events come from native CLI tools under G14.

### G17. Images in the prompt (X)

**Mirror today:** passes images to Codex (`M:assay/platform/providers/configured_agent.py:349-352`).

**agentkit:** `ComposedPrompt` has no image parts (`A:agents/prompts/composition.py:36-40`).

**Build:** image parts in `ComposedPrompt`, mapped per mode. A mode that cannot take them emits `degraded`
and refuses rather than silently dropping them.

### G18. Long sessions on the CLI (P)

**Mirror today:** continues, resumes and forks Claude sessions across slices
(`M:forge/platform/providers/claude_cli.py:1187-1188,1489-1491,1575`), and keeps a stable cached prefix
(`stable_prompt_prefix=True`, `M:assay/platform/providers/model.py:1023`).

**agentkit:**
- `turns_per_slice` works only in the API tool loop (`A:core/runtime/calls.py:599`);
- CLI resume uses `cli_session_id` and a transcript bundle capped at 16 MiB (`cli/calls.py:118-146,184`,
  `isolation.py:83`);
- there is no fork, and no stable-prefix option.

**Build:**
- `turns_per_slice` on CLI modes: stop the process at the slice's turn count and return a `SessionHandle`;
- `fork=True` on resume;
- `stable_prefix=True` passed to the CLI.

**Prove:** a three-slice run on each CLI fake resumes with the same session id and a growing transcript.

### G19. Stall restart is terminal (X)

**Mirror today:** restarts a starved call within the admission window
(`M:assay/platform/providers/execution.py:306-362,199-210`).

**agentkit:** an `idle_s` or `first_event_s` breach ends as `limit_reached` and is never retried
(`A:core/runtime/calls.py:310-313,472-474`).

**Build:** classify a `first_event_s` breach (nothing at all arrived) as `TIMEOUT`, which is retryable, and
an `idle_s` breach after output as terminal. Only the first is safe to repeat. Mirror's "refuse an identical
retry after a timeout" (`execution.py:479-490`) stays in the host.

---

## Tier 3: design items not yet met

| # | Gap | Evidence | Build |
|---|---|---|---|
| G20 | `request_hook` exists only for API modes | `A:models/providers/frontiers/calls.py:27` | A CLI equivalent, a hook over the built argv and settings. (`result_hook` on MCP serving **does** exist: `A:integrations/mcp/serve.py:294,532,753`.) |
| G21 | Skills: no MCP `skill://` source | `A:agents/skills/resources.py:87-160` | An MCP resource source |
| G22 | User-authored persona text is moved, not screened | `A:agents/prompts/composition.py:61` | A screening hook (no secrets, no verdict claims) before composition |
| G23 | Native Claude Code skills and subagents are unused | `--agents` exists (`claude/driver.py:375,1403`) but bare mode passes none | Map definitions onto `--agents` when G1's non-bare option set is used; a conformance scenario per mode |
| G24 | No public scripted fake for `CallRuntime` | only `tests/source/conformance/test_calls.py:54` | Ship `agentkit.testing.ScriptedAdapter` and `scripted_runtime(turns, pricing)` |
| G25 | Journals are not masked or bounded generically | events carry digests (`core/runtime/calls.py:206-207`); the CLI diagnostic is redacted and cut (`cli/calls.py:290-306`) | A `MaskingSink(inner, masker, max_field_bytes)` wrapper for any `EventSink` |

## Stays in Mirror, not agentkit

These are host policy. Mirror keeps them on top of the call contract:
- admission control by measured memory and CPU (`M:assay/platform/providers/admission.py`), plugged in
  through G7's hook;
- prompt fitting and splitting (`fitting.py`, `packages/prompting/fit`);
- vault leases, renewal and rotation (on top of G1, G2 and G12);
- the stage-tier to model mapping (agentkit's `ModelSettings.tier` is a presentation hint,
  `models/contracts/settings.py:16,43-46`);
- refusing an identical retry after a timeout;
- the ledger.

## Breakage when Mirror moves its pin

Mirror tracks `branch = "main"` (`M:pyproject.toml:158`, `M:packages/agent_skills/pyproject.toml:26`), so
**any lock refresh pulls `f8d20c1` in and breaks every plane at import.**

**Modules that no longer exist:** `agentkit.runtime`, `.testing`, `.capabilities`, `.adapters`, `.kernel`,
`.middlewares`, `.memory`, `.prompts`, `.integrations.claude_cli` and `.agents.cognition.claude_cli`.

**Other moves:**
- the package root exports only six namespaces (`A:__init__.py:3-6`);
- `ClaudeCliCognition` and `CodexCliCognition` moved to `A:models/providers/cli/__init__.py:12-34`.

**Non-test import sites:**

| Area | Files |
|---|---|
| assay | `platform/providers/{agentkit_sdk,configured_agent,evidence_fence,model}.py` |
| control | `tenancy/adapters/system/providers/{agentkit_api,chat_agent,chat_memory,chat_registry,claude_agent,codex_agent}.py` |
| evals | `adapters/{agentkit_model,codex_cli,local_model,runtimes}.py` |
| forge | `platform/providers/{agent_tools_adapter,agentkit_sdk,attestation,checkpoints,claude_cli,compaction,event_stream,mcp_server,policy,starter,tool_lifecycle,tool_registry}.py` |
| packages | `agent_hooks/…/claude_cli_fence.py`, `agent_skills/…/integrations/agentkit.py`, `cognition/…/vendor/agentkit.py` |
| sounding | `platform/guardrail.py` |

About 40 test files are affected as well.

Mirror's `EngineServer` and `EvidenceServer` subclass private seams of `McpServerSpec` (`_dispatch`,
`_write_config`, `_document`, `_asgi_app`; still present at `A:integrations/mcp/serve.py:301,445,647,709`).
They work, but any refactor there breaks them silently.

**Do now, before anything else:** pin Mirror to `rev = "555bc57…"` instead of `branch = "main"`. Move the pin
deliberately, plane by plane, once the gaps a plane needs are closed.

## Order

1. **G1, G2, G3, G4.** Without these no CLI mode carries a real login.
2. **G5, G6, G8, G10.** The money must be right and a correct answer must not be thrown away.
3. **G7, G9, G13, G24.** Small and contained.

That is everything preflight needs, which makes it the first consumer.

Next, for assay's drive and forge's build lanes:
4. **G14, G16, G15, G18, G19, G17.**

Then:
5. **G20–G23, G25**, the design items.

Each gap ships with its conformance scenario run across all four modes, or explicitly marked
not-applicable for a mode with the reason, in `tests/source/conformance/`.
