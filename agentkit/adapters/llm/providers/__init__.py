"""Provider clients — thin async `LLMPort` terminals over **httpx** (one transport, no per-provider SDKs).

Two HTTP shapes cover the field: `OpenAICompatibleLLM` (OpenAI · DeepSeek · OpenRouter · Together · Groq ·
vLLM, via `base_url`) and `AnthropicLLM` (Messages API). Presets — `openai`/`deepseek`/`openrouter`/`claude`
— wire base_url/auth. Resilience, observability, and result-caching are the middleware chain's job; these
clients only do the call + parse + cache-aware cost. Requires the extra: `pip install 'arc-agentkit[http]'`.
"""

try:
    from agentkit.adapters.llm.providers.anthropic import AnthropicLLM, claude
    from agentkit.adapters.llm.providers.base import (
        HttpLLM,
        ProviderAuthError,
        ProviderError,
    )
    from agentkit.adapters.llm.providers.openai_compat import (
        OpenAICompatibleLLM,
        deepseek,
        openai,
        openrouter,
    )
except ImportError as exc:  # pragma: no cover — friendly nudge to install the transport
    raise ImportError(
        "agentkit provider clients need httpx — install the extra: pip install 'arc-agentkit[http]'"
    ) from exc

__all__ = [
    "OpenAICompatibleLLM",
    "AnthropicLLM",
    "HttpLLM",
    "ProviderError",
    "ProviderAuthError",
    "openai",
    "deepseek",
    "openrouter",
    "claude",
]
