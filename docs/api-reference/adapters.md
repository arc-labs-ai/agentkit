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
