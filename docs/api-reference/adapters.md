# `agentkit.adapters`

Concrete `Port` implementations: LLM providers (Claude, OpenAI,
DeepSeek, OpenRouter), vector / store / checkpoint back-ends,
observer + replay tooling, and the OTel bridge.

Most adapters are behind an opt-in extra (`http`, `postgres`, `redis`,
`observability`) so the zero-dep core stays clean.

`agentkit.adapters` itself re-exports nothing — each adapter lives in
its own subpackage. Reference them directly:

::: agentkit.adapters.llm
    options:
      show_root_heading: false
      show_source: false
      members_order: source

## Model registry

The from-configuration layer above the explicit provider factories:
model name → provider → wired `LLMPort` (credential from the
environment), plus per-model capability declaration so a mismatch is
refused at bind time rather than surfacing as a plausible empty
answer. See the
[recipe](../recipes/provider-from-env.md) for the mental model.

::: agentkit.adapters.llm.model_registry
    options:
      show_root_heading: false
      show_source: false
      members_order: source

::: agentkit.adapters.store
    options:
      show_root_heading: false
      show_source: false
      members_order: source

::: agentkit.adapters.vector
    options:
      show_root_heading: false
      show_source: false
      members_order: source

::: agentkit.adapters.checkpoint
    options:
      show_root_heading: false
      show_source: false
      members_order: source

::: agentkit.adapters.observer
    options:
      show_root_heading: false
      show_source: false
      members_order: source

::: agentkit.adapters.observability
    options:
      show_root_heading: false
      show_source: false
      members_order: source

## Replay store

`FileReplayStore` keeps the full LLM request/response for a span on
disk, keyed by span id, so the trace stays the index and the payloads
live outside the span's attribute budget.

Configuration, in precedence order:

1. An explicit `FileReplayStore(root=...)`.
2. `AGENTKIT_REPLAY_DIR` — the env var `FileReplayStore.from_env()`
   reads. An empty value counts as unset.
3. `FileReplayStore.default()` — `$XDG_DATA_HOME/agentkit/replays`, or
   `~/.local/share/agentkit/replays` when `XDG_DATA_HOME` is unset.

That list is exhaustive. `AGENTKIT_REPLAY_DIR` is the only env var
read, and `default()` probes nothing on disk — it returns the same
path whatever directories happen to exist, so where the store writes
is always predictable from the environment alone.

!!! warning "`RIO_REPLAY_DIR` no longer does anything"

    `RIO_REPLAY_DIR` and the `~/.rio/replays` default date from before
    this package was `agentkit`, and neither is read any more. If you
    still set the old variable, replay is simply **off** —
    `from_env()` returns `None` and the caller falls back to
    `NoopReplayStore`, with nothing logged. Set `AGENTKIT_REPLAY_DIR`
    instead.

    Replays under an old directory are not migrated for you and are
    not found by `default()`. Move them yourself
    (`mv ~/.rio/replays ~/.local/share/agentkit/replays`) or point the
    store at them with `AGENTKIT_REPLAY_DIR`.

Writes are best-effort by contract — `ReplayStore.put` must never
raise into a run — but never silent: a dropped write logs (once per
failure class, to keep a full disk from flooding the log) and
increments `FileReplayStore.dropped_writes`. Read failures that are
not ordinary cache misses increment `failed_reads`. Alert on those
counters if replay data matters to you; a miss on its own is normal
and is neither logged above `DEBUG` nor counted.

::: agentkit.adapters.replay
    options:
      show_root_heading: false
      show_source: false
      members_order: source
