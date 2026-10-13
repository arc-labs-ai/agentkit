# Real-model workflow evaluations

The default `workload` suite contains fourteen success/recovery workflows plus thirteen paired refusal, insufficient-evidence, authentication, cancellation or limit variants across coding, research, support, analytics, operations and security with actual configured models and installed CLIs. The `smoke`, `contract` and `probes` suites provide smaller transport/conformance checks. A passing arithmetic check is separate evidence from a passing application workflow.

The `contract` suite includes `provider_auth_rejection`, which uses a deliberately invalid temporary credential to observe each provider's real authentication error shape. It must fail as `AUTH` in one attempt; the confirmed pre-generation rejection is recorded at zero spend. `claude_large_schema` runs only on Claude Code: it exercises the bounded system-prompt-file fallback and agentkit's local schema validation for a schema too large for inline argv. This is degraded structured output, not provider-native constrained decoding. Live probes on 2026-10-13 passed on all four backends for authentication rejection and on Claude Code for the large schema; report paths are recorded in `docs/design/mirror-gaps.md`.

Install dependencies with `uv sync --all-extras --group eval`. Place `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` in root `.env.eval`; credentials and generated reports are gitignored. Anthropic OAuth access tokens in the legacy API-key slot are recognized as bearer authentication. The host must refresh expired credentials. These opt-in evaluations incur provider charges and execute approved coding tests in disposable, restricted workspaces.

```sh
# Inspect the catalogue without making model calls.
uv run --group eval python -m tests.evals --list

# Run application workflows with explicit matrix and per-case ceilings.
uv run --group eval python -m tests.evals --suite workload --max-total-usd 6 --max-case-usd 0.85

# Select one domain or individual cases and transports.
uv run --group eval python -m tests.evals --suite workload --domain analytics --mode openai_api --max-total-usd 1 --max-case-usd 0.60
uv run --group eval python -m tests.evals --suite workload --case sqlite_migration_research --mode anthropic_api --max-total-usd 1 --max-case-usd 0.80

# One frozen matrix covers all declared application, transport and fault scenarios.
uv run --group eval python -m tests.evals --suite workload --suite smoke --suite contract --suite probes --max-total-usd 16 --max-case-usd 0.75 --jobs 4

# Smaller contract checks remain available explicitly.
uv run --group eval python -m tests.evals --suite smoke --suite contract --max-total-usd 1

# Offline tests require no provider credentials or installed CLI execution.
AGENTKIT_SKIP_REAL_CLI=1 uv run pytest tests/source/evaluation tests/source/conformance -q
```

`--mode` accepts `claude_code`, `codex`, `anthropic_api` and `openai_api`. `--domain` accepts `coding`, `research`, `support`, `analytics`, `operations` and `security`. `--case`, `--mode`, `--domain` and `--suite` select the matrix; lifecycle cases advertise `supported_modes` and unsupported CLI combinations are not dispatched; `--list` shows the actual catalogue. `--max-total-usd` caps the matrix, and `--max-case-usd` caps each case. Multi-call cases share their case envelope across calls. Use `--env-file` to select another credential file.

## Workloads and evidence

| Workload | Inputs and independent acceptance |
| --- | --- |
| Coding checkpoint repair | A regression introduced into a disposable copy of actual agentkit source; source inspection/edit receipts, reproduced failure, visible and hidden tests, writable-scope integrity and preserved message evidence. |
| Coding schema compatibility repair | Actual schema compiler regression; provider dialect checks plus canonical exclusivity tests and source/test receipts. |
| Coding structured-error repair | Actual failure-classifier regression; billing-error precedence and valid rate-limit retry tests, with controlled source changes. |
| SQLite migration research | Attributed primary documentation, version-scoped historical guidance and a contradictory adversarial memo; exact decisions/citations, actual approved skill loading, digest revocation and cross-model evidence handoff. |
| SQLite incident support | Official SQLite docs and ninety explicitly synthetic incident events; log counts, blocking transaction IDs, safe remedies, source/log citations and event-triggered peer review. |
| Bank campaign analytics | **All 45,211 UCI Bank Marketing records**, unmodified; read-only SQL receipts and independently calculated CSV/Decimal totals, conversion rates and segments. Correct memorized numbers without actual data evidence fail. |
| Inventory allocation | Stateful synthetic multi-warehouse inventory; approved partial shipment/backorder, other-customer reservation preservation, physical inventory conservation and once-only shipment. |
| Return recovery | Stateful synthetic order/payment/inventory services; a committed refund with lost response, idempotent recovery, approved restocking and ledger conservation. |
| Incident rollback | Stateful synthetic deployment/metrics services; scoped diagnostics, rate-limit recovery, approved rollback, transient failure recovery and verified final health. |
| Scoped diagnostic success | Actual telemetry tools and approved recommendation receipts; serves as the baseline for API lifecycle pairs. |
| Partner revocation | An injected-client MCP partner integration; raw-definition approval, changed-definition revocation, permission boundaries and rejection of malicious instructions in external evidence. |

Paired negative cases are explicit catalogue entries with `ScenarioExpectation` and `pair_id`. A negative case passes only when its expected safe behavior occurs and independent tool/state checks agree:

| Positive pair | Expected negative behavior and independent checks |
| --- | --- |
| Coding checkpoint repair | Missing write permission: diagnose and refuse the repair while preserving source/tests and avoiding unauthorized writes. |
| SQLite migration research | Missing authoritative current documentation: actually read contradictory archives, observe the failed authoritative lookup, return a null decision and honest insufficient evidence, with no invented citations. |
| SQLite incident support | Missing transaction-owner traces: read incomplete logs/docs, preserve the exact busy-error count, return null blocker IDs and request traces without attempting an unapproved rollback. |
| Bank campaign analytics | Malformed input: perform schema/row validation, report null metrics and cite the actual validation receipt; memorized totals, invented query receipts and fabricated conversions fail. |
| Return recovery | Missing refund approval: inspect the blocker and commit a scoped manual-review escalation while keeping payment/inventory state unchanged. |
| Incident rollback | Stale approval: escalate without changing the release or falsely resolving the incident. |
| Scoped diagnostic success | API-only variants require a real invalid-credential rejection before business work, an interrupted diagnostic with cleanup and safe resume, or a real turn-limit stop with preserved evidence and safe resume. Native CLI lifecycle equivalence is not claimed. |
| Partner revocation | No trusted fallback evidence: observe revoked access, escalate the missing invoice, and preserve reconciliation/retention state. |

Success after a deliberately injected transient fault is classified as `RECOVERY`; safe refusal and insufficient evidence are distinct from infrastructure/model failures. Normalized formatting never repairs factual errors: ambiguous IDs, wrong release dates, wrong counts, unsafe actions and fabricated citations remain failures.

Knowledge snapshots include primary URLs, attribution, licenses and hashes in `fixtures/knowledge/PROVENANCE.json`. The full bank dataset is [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). SQLite source paraphrases link to its official documentation. Operational service state and incident logs are explicitly synthetic; models still execute real multi-step workflows against those tools. No customer infrastructure or real payment service is mutated.

Knowledge workflows use explicit ReAct tool reasoning, chain-of-thought guidance for the research handoff, and a single-call incident peer reviewer. Private model reasoning is neither requested as output nor graded. Host assertions inspect artifacts, tool receipts and final state; a model judge alone cannot mark a workload passed.

## Reports and folder boundaries

```text
tests/evals/
  catalog.py               discoverable case catalogue and dispatch
  coverage.py              declared features versus actual passed evidence
  engine/                  workflow contracts, model configuration, runner, reports
  suites/
    conformance.py         small shared transport checks
    coding.py              source repair workflows
    knowledge.py           research, support and full-data analytics
    operations.py          stateful operation/security workflows
    framework.py           real API authentication, cancellation and limit checks
  graders/                 independent evidence and semantic oracles
  sandbox/                 restricted workspace and approved test execution
  fixtures/                source snapshots, provenance and workflow state
  reports/<run-id>/
    summary.json
    cases/<mode>/<case>/
      trial-1/
        result.json
        events.jsonl
        evidence.json
        artifacts/           approved coding patch artifacts, where applicable
```

The summary records case/mode outcomes, acceptance checks and usage/cost. Individual case folders preserve sanitized results, event traces and evidence receipts. Coding artifacts make the approved source repair reviewable. Reports are written incrementally; interrupted runs retain completed evidence. Secret values and environment-file contents are excluded or redacted. Treat conversations and artifacts as application data with an explicit host retention policy.

Preserve failed reports. A repaired case needs a rerun; a consolidated final matrix identifies the integrated source version, models and installed CLI versions. Multi-call accounting includes resume, model switches, child reviews and delegation. Missing usage/cost is unknown, not free; conservative reservations may prevent further dispatch. Providers can spend before reporting usage, so a ceiling cannot guarantee mid-request termination.

## What coverage proves

`coverage.py` defines public `FeatureAspect` values and maps every catalogue case to scoped witnesses. `CoverageSummary` separates planned contract/workload coverage from features demonstrated by **actual passed cases**; skipped, failed and unrun cases prove nothing. Filter coverage per mode to avoid treating one provider's pass as proof for all four. Pair coverage requires both the positive and expected-negative case to actually pass on the same backend; a planned pair or passes split across providers do not count.

Live witnesses cover final answers, structured outputs, settings refusal, events, observed accounting, error recovery, tool overflow, cancellation, fresh-runtime CLI continuation, context/model handoff, event hooks, hosted tools, selected MCP policies, skills, bounded delegation and minds. Request/result-hook variants, context/deadline/spend fault boundaries and external MCP transport/auth combinations also require their offline or deployment-specific checks. No live case currently proves cache-prefix reuse/invalidation or minimum cache hits; prompt caching remains explicitly missing from live acceptance coverage.

These are scoped workload demonstrations, not an all-applications guarantee. The partner workload uses an injected client and does not establish live interoperability with every external MCP server. Durable distributed execution, global concurrency, OAuth lifecycle management, production sandboxing and arbitrary provider/model behavior require separate host acceptance tests.


## Preserved baseline evidence

The first complete live application matrix at `reports/20261013T002947527956Z/summary.json` finished **21 passed, 19 failed, 0 skipped across 40 cases**, with no source changes during execution. That run covered the original ten positive/recovery workloads on four backends. Its failed reports are retained; they motivated formatter-grading corrections, stricter evidence checks and workflow fixes. The post-migration workload matrix at `reports/20261013T020123194544Z/summary.json` exercised all 21 catalogue workflows across four modes: **55 passed, 21 failed, 8 not applicable**, with unchanged source throughout. After CLI budget precision, Anthropic history serialization, structured agent output and citation-contract fixes, the focused matrix at `reports/20261013T021807090448Z/summary.json` recorded **21 passed, 11 failed, 4 not applicable**. It confirms corrected handoffs and CLI checkpoint repairs while retaining reasoning, evidence, timeout and workflow-limit failures. These reports identify different source digests; their passes must not be combined into an integrated green matrix. Later scoped runs remain separate evidence.

## Stateful and cache acceptance

`memory_customer_handoff` exercises actual StoreFiles/FileMemory CRUD, durable restart and cross-model recall for a customer logistics handoff. Its paired read-only case requires actual refusal/escalation, tenant boundaries and unchanged stored data. The compaction pair uses SlidingWindowCompactor with explicitly protected attributed tool evidence; decisions require preserved approval and business facts, and missing evidence must remain insufficient.

The cache pair reviews a 120-region refund handbook and records cold/warm provider counters. A pass requires at least 4,096 observed cache-read tokens. The changed-policy case changes the first stable prefix bytes and requires refusal of a revoked approval, with reduced reuse of the previous prefix. These cases are API-only; native CLI cache placement remains vendor-managed. Provider behavior and thresholds are described in [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) and [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). Cache routing is nondeterministic; a bounded persistent miss remains failed acceptance.

Workload verification uses the public AgentRunner with strict state/evidence checks and bounded corrective submissions. Rejected submissions remain evidence; feedback names failed checks without supplying oracle answers. Case envelopes remain shared across review phases. Coding workflows declare 120-second idle and 60-second first-event ceilings within a 300-second wall-clock bound; the larger inference window does not change held-out test requirements.

Real deployment boundaries are tested under `tests/source/integration/deployment`: SQLite restart/replay and conflicting-event rejection, four-process lease capacity/expiry/fencing, and controlled real HTTP MCP credential rotation/approval revocation. The loopback tests use an explicit host client override; the default public-HTTPS boundary remains enforced. These establish local process/service behavior; harnesses retain subscriber cursors, lease admission and downstream fencing and must validate their multi-host backends.
