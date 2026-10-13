# Model-neutral runtime implementation status

This tracks the implementation of `design.md` in the generic agentkit library. Application routing, durable job scheduling, credential storage and distributed resource management remain host responsibilities. No dependency on Mirror is introduced.

## Implemented contracts

| Area | Implementation | Evidence |
| --- | --- | --- |
| C1 final answer | CLI message identity tracking; terminal answer selection in API translators | Four-mode conformance tests and live plain-answer cases |
| C2 structured output | Canonical validation, optional-null cleanup, conservative dialect compilation, bounded correction | Schema boundary tests and live nested output cases |
| C3 settings | Host-overridable capability profiles, refusal of unverified sampling, explicit degradations | Offline settings tests and live refusal cases |
| C4 events | Versioned call events, sequencing, parent IDs, bounded subscribers, journal/bus/trace ports | Event and architecture tests |
| C5 accounting | Disjoint cache buckets, nullable reasoning, cumulative CLI reconciliation, dated prices, unknown accounting stays unpriced | Accounting tests and live usage probes |
| C6 failure/retry | Structured classification and common bounded API retries; native CLI retry observations | Error, disconnect, retry and recovery tests |
| C7/C8 limits | Turn/output/spend checks, global wall-clock/cancellation including tool execution, stream liveness bounds | Deadline and budget regressions; live cancellation cases |
| C9 isolation | Owned POSIX CLI homes, parent/owner locks, bounded credential-first cleanup, bounded transcript bundles | Isolation tests; fresh-runtime live resume |
| C10 context | Immutable message blocks/metadata, canonical messages plus exact vendor history, restored prefix/grounding | Context/checkpoint tests; live resume and model switch |
| C11 caching | Stable prompt manifests and prefix digests; preserved vendor cache usage | Prompt/accounting tests and live API cache-hit, changed-prefix and revocation cases |
| C12 hooks | Public adapters, request/result/event hooks and host ports | Adapter, tool, event and composition tests |
| Tools | Existing registry, shared overflow/image/error policy, deferred discovery, one MCP server path | Tool/server conformance and live evidence/image/overflow cases |
| External MCP | Public-address pinning, explicit policy/definition approval, bounded HTTP JSON/SSE, version/session negotiation, sandboxed stdio seam, cancellation-safe embedded listener ownership | Protocol and security tests; loopback HTTP token rotation, revocation and sequential SSE listener lifecycle |
| Skills/agents | Approved skill manifests, persona/discovery/explicit event triggers, structural tool grants, versioned definitions, bounded verifier/delegation | Architecture tests and opt-in live cases |
| Minds | Separate single-call, ReAct and prompt-level chain-of-thought strategies | Mind conformance and opt-in live cases |

## Limits of this implementation

- Token/context/media counters are estimates unless a host supplies an exact counter. Dollar limits stop further dispatch using observed accounting; they cannot guarantee that a provider stops spending mid-request. Installed CLIs also retain provider-specific limit behavior.
- Capability and price tables deliberately cover a limited verified set. Unknown models/rates are not guessed. CLI accounting behavior was probed on the installed versions; other versions need acceptance probes.
- The external MCP client does not implement deprecated two-endpoint HTTP+SSE, persistent server-initiated streams, stream resumption, or an OAuth login/refresh service. A host can inject a client and resolve refreshed access tokens per request. Tests cover host-resolver token rotation over loopback HTTP, not OAuth login or automatic refresh after a 401.
- SQLite event journals and leases support local multi-process deployments on a local filesystem. Hosts own subscriber replay cursors and idempotent effects, lease renewal and downstream fencing, durable agent stores, global delegation ledgers and multi-host coordination. The in-memory hub provides bounded ordered delivery, not distributed exactly-once processing.
- The middleware Agent/Cognition interface and the model-neutral runner remain distinct execution contracts. They share authentication, dated pricing, canonical schema validation/compilation and accounting provenance. Both usage contracts represent unknown costs as `None`. Registry routing/capabilities and accepted parameter profiles serve different roles; unknown models are not assumed to support native output. Distinct HTTP protocols and CLI lifecycles retain explicit translators.
- Open-standard skill validation and approved resource loading do not constitute a general script execution sandbox. Hosts grant execution tools separately.

## Verification

Run `AGENTKIT_SKIP_REAL_CLI=1 uv run pytest` for the offline suite, `uv run mypy --strict agentkit`, `uv run ruff check agentkit tests`, and the existing coverage/mutation/doc gates. The coverage requirement remains 90 percent. The opt-in runner in `tests/evals/README.md` uses `.env.eval`, records incremental redacted reports, and accounts for every call in multi-call cases.

Failed live reports are retained. Focused reruns demonstrate a correction; only a consolidated final matrix demonstrates the integrated cases on the configured models and installed CLI versions. Passing these checks is not evidence for arbitrary providers, deployments or distributed workloads.

## Breaking ownership migration

The package root exports six owner namespaces and version metadata. Each owner root contains only `__init__.py`; implementation modules are grouped by responsibility. Direct API providers live under `models/providers/frontiers`, installed CLI providers under `models/providers/cli`, and all event implementations under `core/events`. Old namespace aliases are absent.

Source tests, live evaluations and examples live under `tests/source`, `tests/evals` and `tests/examples`. Pytest collects only the source suite by default. Architecture checks enforce package ownership and export-only initializers.

## Validation on 2026-10-13

The current offline source suite passed **5,917 tests with 98 skips on both Python 3.12 and 3.13**. Library branch coverage measures **93 percent**, above the unchanged 90 percent gate. Strict typing and Ruff pass across 265 source files. Mutation anchors resolve, and the strict documentation build passes. A focused live Claude Code test also confirms that partial-message interruption stops before the terminal sentinel and the same session can complete another turn. The broader optional real-CLI source group remains gated; the four-mode live matrix exercises configured providers and installed CLIs separately.

Configured models were `claude-haiku-4-5` and `gpt-5.4-mini`, with Claude Code 2.1.296 and Codex CLI 0.162.1. Each run below retains its source digest, per-case acceptance checks, usage, event traces and evidence. Reports live at `tests/evals/reports/<run>/summary.json`.

| Run | Scope | Passed | Failed | Not applicable | Not run |
| --- | --- | ---: | ---: | ---: | ---: |
| `20261013T020123194544Z` | All application workloads, four modes | 55 | 21 | 8 | 0 |
| `20261013T021807090448Z` | Affected workload reruns and API lifecycle pairs | 21 | 11 | 4 | 0 |
| `20261013T022646319151Z` | Claude research/incident pairs with workflow-sized limits | 4 | 0 | 0 | 0 |
| `20261013T022647365131Z` | Smoke, transport contracts and probes | 62 | 9 | 0 | 9 |
| `20261013T023546955249Z` | Remaining contracts and targeted corrections | 20 | 4 | 0 | 0 |
| `20261013T024303798399Z` | Hosted cancellation and context handoff | 5 | 0 | 0 | 1 |
| `20261013T024350198985Z` | Anthropic cancellation and context handoff | 2 | 0 | 0 | 0 |
| `20261013T024415694405Z` | OpenAI-to-Claude tool-evidence handoff | 1 | 0 | 0 | 0 |
| `20261013T024739431209Z` | Final cancellation on four modes and API cleanup/resume pairs | 6 | 0 | 2 | 0 |

That historical cancellation report records its own implementation digest. Both native CLIs cancel in-flight hosted diagnostic tools and finish cleanup before returning. Both APIs preserve interrupted evidence and safely resume. The final log contains no ASGI shutdown errors. Earlier budget-limited rows have subsequent scoped attempts; the original results remain unchanged.

Validation exposed and corrected CLI budget precision at verifier boundaries, invalid empty Anthropic history blocks, missing structured output forwarding, hosted tool cancellation/shutdown ownership and cooperative cancellation classification. Eval corrections distinguish expected pre-dispatch refusals, explicit citation namespaces, synthetic diagnostic markers and realistic workflow limits. Separate phase evidence preserves an interrupted first step even when its handoff later succeeds.

The first consolidated matrix on source digest `0593c4bdff13f4cb687a0f9fa882ba58485099a1371f96c23168ead80089920a` completed all 188 rows: **150 passed, 26 failed and 12 were not applicable**, with no budget skips and $3.9927593 accounted. Four framework cache rows were explicitly not applicable to CLI modes. The matrix found an embedded MCP SSE listener lifecycle bug; real sequential-listener regressions now cover and fix it. Tool descriptions now state lexical memory query behavior and incident identifiers have semantic schema descriptions. The retained first report is `tests/evals/reports/20261013T081123702859Z/summary.json`.

The first matrix also retained real model acceptance failures: skipped source inspection, wrong incident counts or actions, omitted authoritative lookups, incorrect escalation claims, ignored skill formats, missed delegated/deferred tool execution, memory retrieval and mutation errors, incident identifier swaps, and cache-policy decisions. These are not converted into passes by the source corrections.

The latest integrated matrix, `tests/evals/reports/20261013T115456391082Z/summary.json`, used source digest `5b28a2fcf71b7d131281bdb928c2b3c9fa3bb91b51f066399efc7e6a3389ab1a` and completed all **204 rows: 186 passed, 6 failed and 12 not applicable**, accounting for **$3.84214305** with `source_changed=false`. The 12 not-applicable rows are six API-only lifecycle/cache cases for each native CLI. All four modes passed the stale-lease refusal and scoped-vector positive/negative workflows. OpenAI API and Codex passed durable checkpoint/journal/replay; Anthropic API skipped the commit, and Claude Code committed but returned a status that did not match the stored receipt. Remaining model acceptance failures are Codex SQLite busy-error count and skill application, OpenAI memory handoff completeness, and Anthropic recommendation commit. The graders retain those as failures. This is evidence for the configured models and installed CLI versions, not a guarantee for arbitrary providers or tasks.

The state adapter regression group passed **514 tests with 6 external-service skips** on Python 3.12 and 3.13. It covers checkpoint isolation, SQLite journal rollback/corruption, lease ownership and fencing races, file replay, vector scope, context cancellation, and local stores. The new live state workflows exercise SQLite journal, FileReplay, FileStore-backed checkpoints, SQLite leases, and reference-vector retrieval. PostgreSQL, Redis and pgvector deployments remain environment-gated; local SQLite leases do not provide multi-host coordination or downstream effect fencing. Tool scenarios for evidence, deferred discovery, retry, images, overflow, context handoff, cancellation and scoped denial pass across all four modes, with the Codex discoverable-skill case retained as a model failure.

Native Codex `WARN` disables optional shell/unified-exec, apps, plugins, skill dependency installation and web search while retaining explicitly granted host MCP tools. Codex still has no host-enforced complete native-tool allow-list; `DENY` refuses dispatch when that guarantee is required. Positive interactive host approval is covered offline rather than by the live matrix. Hosts still own multi-host scheduling, replay cursors, idempotent downstream effects, secure credential refresh, and provider-specific mid-request spend controls.
