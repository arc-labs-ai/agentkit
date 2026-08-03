# `agentkit.adapters`

Concrete `Port` implementations: LLM providers (Claude, OpenAI,
DeepSeek, OpenRouter), vector / store / checkpoint back-ends,
observer + replay tooling, and the OTel bridge.

Most adapters are behind an opt-in extra (`http`, `postgres`, `redis`,
`observability`) so the zero-dep core stays clean.

::: agentkit.adapters
    options:
      show_root_heading: false
      show_source: false
      members_order: source
