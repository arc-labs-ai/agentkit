# agentkit build plan

The ordered list of changes that makes agentkit the single, model-neutral agent runtime Mirror (and anyone
else) builds on. It is the plan to work from. `mirror-gaps.md` is the record of how each item was found and
measured; the item ids (G1–G53) are shared between the two.

Written 2026-10-13, against `5ebe4c0` plus the uncommitted working tree. **Re-check each item's status against
the code before starting it:** some are partly done in the working tree.

---

## 0. Rules for every change

1. **One runtime.** `CallRuntime` (`ask`/`run`) and `AgentRunner.run_agent` on top of it are the only way to call
   a model. Nothing new may be built on the `Invoker`, its middleware, `RunContext.services`, `Agent`,
   `Workflow` or the other older modules. They are being deleted (Phase 3).
2. **Same behaviour on all four modes.** Claude Code CLI, Codex CLI, Anthropic API and OpenAI API. Every
   behaviour lives in the runtime core, never in one adapter. An adapter only translates. Every behaviour gets
   a conformance scenario in `tests/source/conformance/`, run against each mode's fake, or explicitly marked not
   applicable to a mode with the reason.
3. **Test first, and prove the test can fail.** Write the test, watch it fail for the stated reason, then
   implement. Every test has a clean twin, and a test that would pass with the code removed is not a test.
4. **Fakes must be as strict as the real thing.** A fake CLI that accepts what the real CLI rejects hid G26.
   Every fake validates its inputs the way its real counterpart does.
5. **No backward-compatibility shims.** Delete old paths in the same change that replaces them. Consumers on an
   older revision stay pinned until they move.
6. **No timers that mean ownership.** No leases, heartbeats or renewals. Admission is the untimed `acquire`
   hook. A stall bound is the absence of events, never a wall clock on work that is still producing.
7. **Secrets never reach events, logs or snapshots.** Masking is on by default, and the host can supply its own
   masker.
8. **Report results with their scope.** "The conformance suite passes" and "the suite passes" are different
   claims.

---

## 1. Status at the start

| State | Items |
|---|---|
| Done and proven | G3, G10, G14, G17, G21, G22, G25, G26, G27, G28, G29, G31; G32 and G33 decided (an ambiguous reply is refused; `REFUSE` refuses) |
| Done, no proving test | G4, G12, G15, G19, G20 |
| Partial | G1, G2, G5, G6, G8, G9, G16, G18, G23, G34 |
| Open | G30, G35, G36, G37–G53 (some partly in the working tree: G37, G38, G40, G42/G50, the compaction hook, scope in the API loop) |
| Accepted limits | Codex has no mid-call cost cap or turn slices (G5, G18): the Codex CLI does not expose them. Documented, refused cleanly, not worked around |

### Rechecked working-tree status (2026-10-14)

These results describe the current working tree, not a commit. The full offline suite and targeted phase tests have
run; commit IDs will be added when this implementation is committed.

| Phase item | Current status | Proving evidence / remaining work |
|---|---|---|
| G35 | Implemented and tested | `test_claude_budgeted_retry_treats_failed_attempt_without_usage_as_zero_spent` and `test_cli_budget_does_not_retry_after_output_with_unknown_cost` in `tests/source/conformance/test_calls.py` |
| G34 | Implemented and tested | `test_reported_cost_with_partial_usage_is_mixed_and_remains_incomplete` in `tests/source/conformance/test_calls.py` |
| G36 | Implemented and tested | `test_cli_auth_failure_without_rotation_is_not_retried` asserts exactly one invocation |
| G1 | Implemented and tested | `test_cli_async_credential_resolver_and_unsafe_environment_refuse_before_spawn`, `test_cli_successful_async_credential_resolver_reaches_child`, and `test_strict_cli_fakes_require_private_subscription_login_files` prove pre-spawn refusal, async resolution, and missing/0644/0600 subscription-file behavior |
| G2 | Implemented and tested | `test_credential_refresh_bytes_are_delivered_once_through_cli` asserts exact bytes and one callback |
| G4, G12, G15, G19, G20, G31 | Implemented and tested | Corresponding executable, refresh-discard, timeout, first-event retry, request-hook and argv-boundary tests in `test_calls.py` |
| G28 | Implemented and tested | Scalar/key masking and per-runtime HMAC digest tests in `test_calls.py` |
| G37 | Implemented and tested | `test_agent_runner_forwards_workspace_price_table_and_turn_slice` captures runtime and adapter arguments |
| G38 | Implemented and tested | `test_host_can_override_the_brief_given_to_delegated_child` |
| G39 | Implemented and tested | Screening transformation and refusal-before-dispatch tests in `test_runner_reference_tools.py` |
| G40 | Implemented and tested | `test_persona_screening_and_skill_loaded_event_are_applied_once` plus `test_load_skill_and_triggered_activation_each_emit_once` |
| G41 + G53 | Implemented and tested for API tool loops and both CLI scopes | `test_tool_calls_execute_concurrently_with_the_invocation_scope` and `test_a_side_effecting_tool_turn_runs_sequentially` cover both APIs; `test_cli_hosted_tools_receive_the_invocation_scope` covers both CLIs. CLI tool-loop concurrency is owned by the CLI process, not agentkit's API tool dispatcher. |
| G42 + G50 | Implemented and tested | `test_isolated_event_subscriber_does_not_fail_publication_or_skip_others`, `test_raising_event_sink_is_reported_and_delivery_resumes`, and `test_raising_event_sink_isolated_and_answer_retained` |
| Phase 3 step 1 | Implemented and tested | `cli/calls.py` has no imports from retired runtime modules; CLI conformance tests pass |
| Phase 3 step 2 | Implemented and tested | Deterministic `SlidingWindowContextPolicy`, `test_cli_context_compaction_keeps_one_native_session_across_three_slices`, scoped memory composition, CLI approval config wiring, and `test_cli_permission_prompt_mcp_request_reaches_host_approval_callback` |
| Phase 3 deletion + claims | Implemented and tested in the working tree | Legacy runtime modules and stale tests are removed; `test_single_runtime.py` checks one public result/usage type and planted imports of removed modules |
| G44 | Implemented and conformance-tested | `test_tool_timeout_is_model_visible_and_cancels_body`, `test_cli_mcp_timeout_covers_largest_per_tool_deadline`, and `test_cli_idle_deadline_pauses_during_a_hosted_tool` cover per-tool deadlines, derived MCP timeout and a 3-second tool under a 1-second idle bound |
| G45 | Implemented and offline-verified | The native-hook fake CLI integration tests exercise the real hook subprocess through `CallRuntime` and `CliAdapter`; callback allow/deny/error/timeout, malformed input, authentication, cleanup and fail-closed fallback are covered |
| G47 | Implemented and conformance-tested | `test_native_cli_bundle_reference_restores_across_adapter_instances`, `test_portable_cli_replay_applies_policy_before_dispatch_and_obeys_byte_bound`, and `test_portable_cli_replay_over_limit_stops_before_spawn` prove bundle propagation, bounded host policy replay and no-spawn refusal. The bundle-store tests cover 17 MiB cross-process restore and incremental chunks. |
| G48 | Implemented and conformance-tested | `test_narration_capture_is_per_call_and_bounded`, `test_narration_capture_refuses_without_a_host_masker`, and `test_narration_masker_failure_keeps_answer_and_omits_text` |
| G49 | Implemented and conformance-tested | Anthropic/OpenAI payload mapping, explicit OpenAI model capability and pre-dispatch refusal tests in `test_calls.py`; the CLI modes refuse because their installed CLIs expose no verified TTL control |
| Cache stability | Implemented and conformance-tested | `test_cached_prefix_is_byte_stable_across_calls_with_different_tasks` asserts stable prompt bytes/key with varying task text and call IDs |
| G51 | Implemented and conformance-tested | `test_native_tool_policy_is_per_call_beneath_runtime_ceiling` and `test_claude_per_call_deny_refuses_explicit_native_tools_before_spawn` |
| G5/G18 | Unsupported hard guarantees refused | `test_codex_hard_cost_cap_refuses_before_spawn` and `test_codex_native_turn_slicing_fails_closed_before_dispatch`; Codex can still report measured post-turn cost when no hard cap is requested |
| G30 | Login probes partially complete; refresh proof open | Codex completed isolated real-`auth.json` login calls on macOS and Alpine Linux: `tests/evals/reports/g30-codex/20261013T195725488601Z/summary.json`, `tests/evals/reports/g30-linux-codex/20261013T201026545283Z/summary.json`. The Alpine run used Python 3.12, Claude Code 2.1.296 and Codex 0.162.1; auth was mounted read-only and the call answered in one attempt (Codex cost is unpriced). Claude Code 2.1.296 completed an `AUTH_OK` call through its default profile with `authMethod=claude.ai`, no `.credentials.json`, and no credential env var; the same CLI failed `Not logged in` under AgentKit's isolated config. This matches [Claude Code's macOS Keychain documentation](https://code.claude.com/docs/en/iam) and shows AgentKit isolation does not inherit that login. Evidence: `tests/evals/reports/g30-claude-keychain/default-profile/summary.json` and `tests/evals/reports/g30-claude-keychain/20261013T195720356820Z/summary.json`. The Linux auth eval's clean-home path bug is fixed; `test_codex_login_check_accepts_custom_auth_file_when_default_home_is_absent` failed before and passes after the fix. No refresh was forced; no Claude `.credentials.json` is available. |
| Phase 5 matrix | Scoped stable-revision run complete; four-mode execution is partial | `tests/evals/reports/20261013T194002512614Z/summary.json`: source unchanged at fingerprint `3226f3883ee9bdc9346d0e06956130b0b2cfd25e86fe9782862313f11c652209`; 40 selected rows, 30 passed, 10 failed, $0.1359731500000000036 spend and the same spend upper bound. Claude Code, Anthropic API and OpenAI API passed 10/10 rows each. Codex failed all 10 before spawn with the expected `INVALID_REQUEST` hard-cost-cap refusal (0 attempts, $0); this proves fail-closed behavior, not Codex task execution. No live retry/429 occurred. |

The final offline run passed **1,354 tests, 25 skipped**; the full conformance suite passed **438 tests**;
the architecture suite passed **9 tests**. Strict mypy over `agentkit`, Ruff and `git diff --check` passed;
the strict documentation build passed. These are results for the uncommitted working tree. Commit IDs will be
recorded after the implementation commit.

---

## Phase 1: Correctness debts on the current path

**Goal: nothing on the current path misreports money, drops a check, or ships untested.**

| Item | Change | Done when (tests) |
|---|---|---|
| **G35** A retried 429 makes the answer's cost unknown | an attempt that failed with no usage, no cost and no output contributes zero to the total; only an attempt that produced output or usage without a cost makes the total unknown | a usage-less 429 then success at $0.20 gives `cost_usd=0.20`, source `reported`; twin: output with no cost leaves the total unknown |
| **G34** Partial usage with a reported cost | keep `usage_complete`; settle as `mixed` when the cost is reported but the tokens are partial | reported $0.25 with partial usage gives `cost_source=mixed`, `usage_complete=False` |
| **G36** A dropped assertion | restore `assert len(cli.invocations) == 1` in `test_cli_auth_failure_without_rotation_is_not_retried` | removing the no-retry guard makes that test fail |
| **G1** Login proving tests | fake CLIs refuse to start unless the expected credential file is in their home at 0600; an adapter-level test that a disallowed home file is refused **before spawn**; an async resolver that succeeds | each fails with the guard removed |
| **G2** Refresh through a CLI | a fake CLI rewrites its credential file; `on_changed` receives the new bytes exactly once through the adapter | the callback count and bytes are asserted end to end |
| **G4, G12, G15, G19, G20** | add the missing tests: `executable=` per mode; `on_changed=None` discards a refresh; `MCP_TOOL_TIMEOUT` and Codex `tool_timeout_s` derived from `ToolSet.timeout_s`; the runtime retries a first-event timeout on CLI; the CLI `request_hook` replaces argv but not the executable | each fails with its code removed |
| **G31** Argv boundary | a payload just over the bound goes by file; just under goes inline | both asserted |
| **G28** Non-string leaves | mask keys and every scalar type; consider a keyed digest (HMAC with a per-runtime secret) so low-entropy values cannot be brute-forced from journals | a 6-digit secret's digest differs across runtimes; no raw value appears in events |

---

## Phase 2: Make `run_agent` complete

**Goal: everything `CallRuntime.run` can do is reachable through `run_agent`, safely.**

| Item | Change | Done when |
|---|---|---|
| **G37** Parameters not forwarded | forward every `CallRuntime.run` parameter `run_agent` accepts: `workspace`, `pricing`, `turns_per_slice`, `stable_prefix`, `fork`, `native_agents`, `scope`; refuse an unknown one rather than drop it | a scripted adapter records each parameter arriving from `run_agent` |
| **G38** Delegated brief | a `brief=` hook returning the child's `InstructionSet` and inputs | a host brief reaches the child's composed prompt; the default is unchanged |
| **G39** Persona screening | `AgentRunner(persona_screening=…)` forwarded to `compose` | an untrusted persona the screen refuses never reaches a model |
| **G40** `SKILL_LOADED` | emit it on `load_skill`, and on pinned and triggered activation, with ref and digest | each path emits exactly one event |
| **G42 + G50** Subscriber failures | `EventHub.subscribe(…, isolate=True)` logs a handler's error and keeps delivering; the runtime emits `degraded` immediately on a sink error and keeps delivering; it stops only after N consecutive failures | an isolated raising subscriber leaves the call answered and the others delivered; the `degraded` event is immediate |
| **G41 + G53** Tools and scope | run the model's concurrent tool calls in parallel when every one is side-effect-free; pass the host `scope` into each tool's `RunContext` in the API loop and in CLI `serve()` | two side-effect-free tools overlap, a side-effecting one runs alone; a tool reads its scope in both modes |

---

## Phase 3: One runtime (G43)

**Goal: one result type, one usage type, one path. Every feature that exists is reachable.**

Do it in this order, so nothing breaks midway.

1. **Move the CLI adapter off the older runtime.** The current CLI adapter still uses `Agent`, `RunContext` and
   `Budget` internally (`models/providers/cli/calls.py:107,126,242-246,488-491`). Replace them with what it
   actually needs.
   - **Done when:** `cli/calls.py` imports nothing from the older modules, and all CLI conformance passes.
2. **Rebuild the pieces that have users, as hooks on the current path.**
   - **Compaction:** `context_policy: Callable[[WorkingContext], WorkingContext]`, applied by
     `run`/`run_agent` between slices and before resume. It is **per mode** (G46): on the CLI it keeps
     `--resume` and sends nothing extra, rather than dropping the native session. Deterministic policies are
     built in; summarising policies are an explicit opt-in.
   - **Memory:** a provider `run_agent` turns into an untrusted `context` part at compose time.
   - **Approvals:** rebuild `ApprovalServer` on the current CLI adapter's `--permission-prompt-tool`, with the
     same callback shape as `McpPolicy.approval`.
   - **Done when:**
     - the same transcript compacts to the same result twice;
     - a Claude fake keeps one resume id across three compacted slices;
     - memory items reach the prompt as an untrusted part;
     - a CLI permission prompt reaches the host callback.
3. **Delete the older runtime in one change:**
   - the `Invoker`, `core/policies` middleware and `RunContext.services`;
   - `Observation`/`ObserverPort` and `integrations/observers`;
   - `state/checkpoints`, `replay`, `stores`, `vectors` and `leases`;
   - `agents/workflows`, `agents/cognition`, `agents/policies`, `agents/control` and `agents/evaluation`;
   - `Agent`, `RequestBuilder`, `skills.Skill.as_agent` and `tools/adapters.as_tool`.

   Keep `core/guardrails` as a library tools call, with an optional `ToolSet` hook. Update the docs and examples
   in the same change.
4. **Architecture claims test:**
   - one public `Answer` type and one `Usage` type;
   - a planted import of each deleted module is caught;
   - every public module is reachable from `CallRuntime`/`AgentRunner`, or declared a library.

---

## Phase 4: Long, parallel, tool-heavy builds on the CLI

**Goal: a build service can run hours-long, parallel, tool-heavy sessions on Claude Code and Codex without
losing work, cache or control.**

| Item | Change | Done when |
|---|---|---|
| **G44** Stall vs long tools | `Tool.timeout_s` per tool; the `serve()` wrapper signals tool start and end so `idle_s` pauses until that tool's deadline; `MCP_TOOL_TIMEOUT` derives from the largest per-tool deadline | a 3 s tool under `idle_s=1` completes; a silent model with no tool running times out at `idle_s`; a tool past its own deadline times out |
| **G45** Live native-tool fence — implemented, offline verified | `Workspace.pre_tool_use` uses typed `Allow` / `Deny`; Claude native calls cross an agentkit-owned authenticated Unix-socket hook bridge, without middleware. Denials emit `tool_call(refused=True)` without argument data/digests. MCP calls retain hosted authorization. Claude/POSIX only; Codex rejects this unsupported workspace option. | `AGENTKIT_SKIP_REAL_CLI=1 uv run pytest tests/source/conformance/test_native_tool_fence.py -q`: **9 passed**. Fake CLI invokes the actual hook subprocess through `CallRuntime`/`CliAdapter`; allow/deny, callback exceptions/timeout cleanup, malformed input, unauthorized connection, disconnected endpoint, missing script fallback, retained hooks and private configuration cleanup verified. Denial uses the official [Claude PreToolUse JSON contract](https://code.claude.com/docs/en/hooks#pretooluse-decision-control); no paid live probe claimed. |
| **G51** Native-tool policy per call | `native_tool_policy` per call, never wider than the runtime ceiling | under one runtime, a Claude call denies native tools while a Codex call runs under `warn` |
| **G47** Long resumes across processes | a bundle-store port (by reference, incremental); the fallback replay goes through `context_policy` with a byte bound, and `limit_reached` if it is still too large | a 17 MiB transcript resumed in a new process gives a bounded prompt and a `degraded` event; the second slice stores a reference |
| **G48** Narration | opt-in `capture_text` per call; `MESSAGE_DONE` carries bounded, host-masked text; missing or failing masker omits text while retaining the answer | on: the done text is masked and capped; off: length only; masker failure retains the answer |
| **G49** Cache TTL | `ModelSettings.cache_ttl`; Anthropic API maps 5m/1h, OpenAI API maps 30m only with a model-scoped verified capability, and CLI modes refuse until their native CLIs expose a verified control | API payload mappings are asserted; unprofiled or unsupported modes refuse before dispatch |
| **Cache stability** | a check that the cached prefix is byte-stable across calls with the same inputs: no timestamps, no per-call ids, stable tool order | two calls with the same inputs produce the same prefix digest |
| **G52** (optional) | `ToolResult(stop_call=True)` ends the call after the turn, with a resumable session | the tool ends a CLI call and the session id is kept |
| **G5, G18 limits** | document the Codex limits and refuse cleanly: a requested hard `max_cost_usd` or turn slice fails before dispatch because Codex cannot enforce it; measured post-turn cost remains available without a hard cap | the cost-cap and turn-slice refusal assertions pass before spawn |

---

## Phase 5: Live proof

**Goal: what the fakes prove, the real providers confirm.**

| Item | Change | Done when |
|---|---|---|
| **G30** Real logins | opt-in live checks: Claude Code with a real subscription `.credentials.json`, Codex with a ChatGPT `auth.json`; a refresh forced mid-call reaches `on_changed`; on macOS, confirm Claude reads the file rather than the keychain, and fail closed if it reads the keychain | **Partial:** isolated Codex `auth.json` login passes on macOS and Alpine. Claude default-profile login passes without a credential file or env var, while the isolated AgentKit config fails closed as not logged in; official Claude Code docs state macOS credentials are stored in Keychain. Keychain-vs-file behavior is evidenced; AgentKit cannot currently use that default Keychain-backed login through its isolated home. No refresh was forced, and no Claude subscription `.credentials.json` was available. See the G30 status row above. |
| **Four-mode live matrix** | run the contract and workload suites on all four modes at this revision | **Partial:** stable scoped report `tests/evals/reports/20261013T194002512614Z/summary.json`; 30/40 pass. Three executable modes pass 10/10 each. Codex cleanly refuses all 10 budgeted rows before dispatch because the CLI cannot enforce `max_cost_usd`; no Codex execution claim is made. |
| **A real 429** | observe a real rate-limit response and its `Retry-After` on each mode | **Open:** no real provider 429 was observed. Offline retry/error-shape tests remain the evidence for retry behavior. |

---

## Order, and what each phase unlocks for Mirror

| Phase | Unlocks |
|---|---|
| 1 | Mirror's one-shot calls (`cognition.ask`) on API keys, with honest money |
| 2 | Mirror's agent loops (`cognition.run_agent`) for conversation and analysis |
| 3 | one runtime; Mirror's compaction and memory plug in as hooks instead of living in Mirror |
| 4 | Mirror's build service (forge) on agentkit: long, parallel, fenced, cached CLI sessions |
| 5 | running on real subscription logins in production |

Phases 1 and 2 can run in parallel. Phase 3 step 1 must land before step 3. Phase 4 needs Phase 3's compaction
hook (for G46 and G47). Phase 5 runs at the end of each phase on whatever it unlocked.

---

## Definition of done for the plan

- Every item above is marked done in this file, with the commit that did it and the test that proves it.
- The offline suite, the conformance suite and the architecture claims test are green, each reported with its
  scope.
- The four-mode live matrix and the G30 login checks have recorded reports at the final revision.
- The older runtime does not exist in the tree.
