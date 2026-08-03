"""LLMPort adapters — `CallableLLM` (inject any provider).

Model fallover is no longer an adapter — it's the `fallback` middleware. These adapters just wrap a
single provider seam. `CallableLLM` adapts an injected `fn`/`chat_fn` (e.g. the OpenRouter
`ModelSDKTools`). The raw↔`LLMResult` translation helpers live in `_mapping`. The public surface
is `agentkit.adapters.llm`.

For the offline, deterministic `FakeLLM`/`Turn` test double, import from `agentkit.testing` —
real adapters live in `adapters/`, test doubles in `testing/`.
"""

from agentkit.adapters.llm.callable import CallableLLM

__all__ = ["CallableLLM"]
