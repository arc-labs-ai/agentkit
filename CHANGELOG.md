# Changelog

All notable changes to `agentkit` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Typed error taxonomy raised at adapter/port boundaries: `AgentkitError`
  (base), `CheckpointerError`, `StoreUnavailable`, `ProviderAuthError`.
  Adapters wrap backend exceptions (`asyncpg`, `httpx`, `redis`, …) with
  `raise …Error(…) from exc` so callers pattern-match on the framework
  taxonomy without importing backend types.
- `RunPolicy` is now auto-invoked from `Agent.run` when set: `mode="deny"`
  raises `PermissionError` before the first cognition drive; `mode="flag"`
  stamps a `policy.flagged` observation and lets the run continue. One
  `policy.check` span per run carries mode, capabilities, and verdict.
- `Checkpointer` acquires a per-`run_id` `asyncio.Lock` around the
  read-compute-write cycle in `snapshot`, and deep-copies `state` +
  `metadata` on save so subsequent producer-side mutation cannot corrupt
  the stored version. Concurrent snapshots now see monotonic distinct
  versions instead of racing on `next_version = max + 1`.
- README + Apache-2.0 `LICENSE` + `NOTICE` at the package root; `docs/mental-models/`
  covering the four canonical use cases (multi-tenant chat, autonomous
  DevOps investigator, long-running enrichment, coordinated research)
  that each stress a distinct set of framework invariants.

### Changed
- `Suspended.pending` is now typed as
  `tuple[ToolCall, ...] | tuple[str, ...]` (previously an unconstrained
  tuple). The tool-approval surface emits `ToolCall`s; `Workflow`'s
  `human_gate` node emits gate-name `str`s. The narrowed union catches
  drift from a third caller passing arbitrary objects, and pins the
  suspend/resume handshake at both ends of the wire.
- `ProviderAuthError` now multi-inherits from
  `agentkit.kernel.errors.ProviderAuthError` (kernel taxonomy) and the
  transport-level `ProviderError`, so raised instances satisfy
  `except AgentkitError`, `except ProviderAuthError`, and legacy
  `except ProviderError` blocks simultaneously.

### Fixed
- Roughly 200 `mypy --strict` findings across `kernel/`, `runtime/`,
  `agents/`, `capabilities/`, and `adapters/` — missing return types,
  `Any`-typed public seams, `Optional` mis-annotations, and a handful of
  variance mistakes on Protocol generics. `uv run mypy agentkit` is now
  clean on the full tree.

## [0.1.0] — 2026-07-03

Initial public release.
