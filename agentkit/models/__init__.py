"""LLMPort adapters — `CallableLLM` (inject any provider).

Model fallover is no longer an adapter — it's the `fallback` middleware. These adapters just wrap a
single provider seam. `CallableLLM` adapts an injected `fn`/`chat_fn` (e.g. the OpenRouter
`ModelSDKTools`). The raw↔`LLMResult` translation helpers live in `_mapping`. The public surface
is `agentkit.models`.

For the offline, deterministic `FakeLLM`/`FakeTurn` test double, import from `agentkit.core.testing` —
real adapters live in `adapters/`, test doubles in `testing/`.

`model_registry` is the from-configuration layer sitting ABOVE the explicit provider factories: it maps a
model name to a provider, reads the credential from the environment, and declares per-model
capabilities so a mismatch is refused at bind time. Importing it costs nothing — every provider
factory is loaded lazily by dotted path, so the registry stays usable on a zero-dependency install.
"""

from agentkit.core.contracts.types import CostSource, Mode
from agentkit.models.adapters.callable import CallableLLM
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.catalog.registry import (
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
from agentkit.models.contracts.constants import (
    CacheTtl,
    ClaudePermissionMode,
    CodexSandboxMode,
    Effort,
    MindKind,
    ModelSelection,
    NativeToolPolicy,
    Outcome,
    ReasoningSummary,
    ServiceTier,
    UnpricedPolicy,
    Verbosity,
)
from agentkit.models.contracts.schema import OutputSpec, compile_schema, validate_output, validate_schema
from agentkit.models.contracts.types import (
    Allow,
    Answer,
    Binding,
    CallLimits,
    Credential,
    CredentialRef,
    Deny,
    ModelSettings,
    PreToolUse,
    SessionHandle,
    Workspace,
)
from agentkit.models.providers.endpoints import ApiEndpoint, ApiPath

__all__ = [
    "Answer",
    "PriceTable",
    "Mode",
    "ApiEndpoint",
    "ApiPath",
    "Binding",
    "CallLimits",
    "Credential",
    "CredentialRef",
    "Allow",
    "Deny",
    "PreToolUse",
    "UnpricedPolicy",
    "ClaudePermissionMode",
    "CodexSandboxMode",
    "Workspace",
    "ModelSettings",
    "SessionHandle",
    "CostSource",
    "compile_schema",
    "validate_output",
    "validate_schema",
    "Outcome",
    "MindKind",
    "ModelSelection",
    "NativeToolPolicy",
    "CacheTtl",
    "Effort",
    "ReasoningSummary",
    "ServiceTier",
    "Verbosity",
    "OutputSpec",
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
