# Implementation status

[`build-plan.md`](build-plan.md) is the implementation source of truth. This page summarizes the current execution
architecture and verification boundaries; it does not maintain a second item-by-item gap tracker. Historical evidence
for consumer-derived requirements remains in [`mirror-gaps.md`](mirror-gaps.md).

## Current architecture

- `CallRuntime.ask()` and `CallRuntime.run()` own model dispatch, continuation, retries, limits, pricing, structured
  output, tool execution, and call events.
- `AgentRunner.run_agent()` composes agent instructions, minds, context, scoped tools, skills, memory, delegation,
  and verification on that runtime.
- Anthropic API, OpenAI API, Claude Code CLI, and Codex CLI adapters translate the shared contracts. Provider
  capability limits remain explicit.
- `WorkingContext` carries message context across model changes. Context policies transform that context at runtime
  boundaries. Memory providers contribute untrusted prompt context.
- `EventHub` is the host-facing event subscription point. Isolated subscribers cannot fail publication; runtime event
  sinks report degradation while preserving the answer.
- Hosts own business workflows, durable domain state, scheduling, and domain-specific grading.

## Verification scope

The offline source suite verifies runtime contracts against scripted API adapters and fake CLIs. The conformance suite
must be reported separately from unit/source coverage. Architecture claims are checked by
`tests/source/architecture/test_single_runtime.py`; that test does not substitute for behavior tests.

The latest full offline run passed **1,326 tests with 1 skipped**. The conformance suite passed **407 tests**, and the
single-runtime architecture claims test passed **4 tests**. `uv run ruff check agentkit tests`, `git diff --check`,
and the strict MkDocs build pass after replacing the legacy-runtime documentation.

Historical provider-backed evaluations are retained under `tests/evals/reports/`. They use their recorded source
digests and configured model/CLI versions; they do not establish a live matrix for the current working tree. The
current revision has not yet run the four-mode live matrix or the real-login refresh checks. In particular, the
macOS Claude Code keychain-versus-file behavior and a forced credential refresh remain unproven here. The current
working-tree changes are not yet committed.

## Known provider limits

Codex CLI does not expose mid-call cost enforcement or turn slicing; the runtime refuses a requested hard cost cap
or turn slice before dispatch. Measured post-turn cost remains available when no hard cap is requested. Provider-native
transport for very large structured schemas and native Claude skills/delegation parity are tracked in the build plan.
Estimates for token/context/media size are not provider-exact unless a host supplies an exact counter.
