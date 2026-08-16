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

::: agentkit.adapters.replay
    options:
      show_root_heading: false
      show_source: false
      members_order: source
