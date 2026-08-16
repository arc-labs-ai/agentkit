"""LLMPort adapters — `CallableLLM` (inject any provider).

Model fallover is no longer an adapter — it's the `fallback` middleware. These adapters just wrap a
single provider seam. `CallableLLM` adapts an injected `fn`/`chat_fn` (e.g. the OpenRouter
`ModelSDKTools`). The raw↔`LLMResult` translation helpers live in `_mapping`. The public surface
is `agentkit.adapters.llm`.

For the offline, deterministic `FakeLLM`/`Turn` test double, import from `agentkit.testing` —
real adapters live in `adapters/`, test doubles in `testing/`.

`model_registry` is the from-configuration layer sitting ABOVE the explicit provider factories: it maps a
model name to a provider, reads the credential from the environment, and declares per-model
capabilities so a mismatch is refused at bind time. Importing it costs nothing — every provider
factory is loaded lazily by dotted path, so the registry stays usable on a zero-dependency install.
"""

from agentkit.adapters.llm.callable import CallableLLM
from agentkit.adapters.llm.model_registry import (
    CAPABILITY_NAMES,
    Capability,
    CapabilityMismatch,
    MissingProviderExtra,
    ModelCapabilities,
    ModelEntry,
    ModelRegistry,
    ProviderEntry,
    ProviderNotConfigured,
    RegistryError,
    UnknownModel,
    default_registry,
    model_capabilities,
    normalize_model_name,
    register_model,
    register_provider,
    register_rule,
    registry,
    require_capabilities,
    resolve_llm,
)

__all__ = [
    "CAPABILITY_NAMES",
    "CallableLLM",
    "Capability",
    "CapabilityMismatch",
    "MissingProviderExtra",
    "ModelCapabilities",
    "ModelEntry",
    "ModelRegistry",
    "ProviderEntry",
    "ProviderNotConfigured",
    "RegistryError",
    "UnknownModel",
    "default_registry",
    "model_capabilities",
    "normalize_model_name",
    "register_model",
    "register_provider",
    "register_rule",
    "registry",
    "require_capabilities",
    "resolve_llm",
]
