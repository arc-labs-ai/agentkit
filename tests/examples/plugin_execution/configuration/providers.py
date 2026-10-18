"""Only bindings vary by provider. Credentials resolve only after preflight."""

import os
from pathlib import Path

from agentkit.models import Binding, Credential, CredentialRef, ExecutionProfile, ExecutionRequirements

MODES = ("openai_api", "anthropic_api", "claude_code", "codex")
MODELS = {
    "openai_api": "gpt-5.4",
    "anthropic_api": "claude-sonnet-4-6",
    "claude_code": "claude-sonnet-4-6",
    "codex": "gpt-5.4",
}
REQUIREMENTS = ExecutionRequirements(profile=ExecutionProfile.HOST_TOOLS_ONLY)
# This distinct opt-in permits provider-native behavior under WARN. It does
# not promise native-tool absence, MCP discovery exclusion or containment.
HOST_TOOL_PATH_REQUIREMENTS = ExecutionRequirements(profile=ExecutionProfile.CLI_COMPATIBLE)


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
