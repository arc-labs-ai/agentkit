"""Only bindings vary by provider. Credentials resolve only after preflight."""

import os
from pathlib import Path

from agentkit.models import Binding, Credential, CredentialRef, ExecutionRequirement, ExecutionRequirements

MODES = ("openai_api", "anthropic_api", "claude_code", "codex")
MODELS = {
    "openai_api": "gpt-5.4",
    "anthropic_api": "claude-sonnet-4-6",
    "claude_code": "claude-sonnet-4-6",
    "codex": "gpt-5.4",
}
REQUIREMENTS = ExecutionRequirements(
    required=frozenset(
        {
            ExecutionRequirement.HOST_TOOLS,
            ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE,
            ExecutionRequirement.SCHEMA,
            ExecutionRequirement.USAGE,
        }
    )
)


def provider_binding(mode: str, *, model: str | None = None, auth_file: Path | None = None) -> Binding:
    """Use an explicitly selected key or native auth file, never an ambient home."""
    if mode not in MODES:
        raise ValueError("unknown provider mode")

    def resolve():
        if auth_file is not None:
            if mode not in ("claude_code", "codex"):
                raise ValueError("native authentication files are CLI-only")
            filename = ".credentials.json" if mode == "claude_code" else "auth.json"
            return Credential(env={}, home={filename: auth_file.read_bytes()})
        key = "ANTHROPIC_API_KEY" if mode in ("anthropic_api", "claude_code") else "OPENAI_API_KEY"
        return {key: os.environ[key]} if key in os.environ else {}

    return Binding(mode, model or MODELS[mode], CredentialRef(resolve))
