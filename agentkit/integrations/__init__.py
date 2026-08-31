"""agentkit.integrations — adapters bridging external protocols into agentkit's Protocols.

Each subpackage is opt-in and gated behind a ``pyproject.toml`` extra so the
core stays dependency-free. An integration module MUST fail fast at import
time with a helpful message when its optional dependency isn't installed.

Available integrations:

- ``agentkit.integrations.mcp`` — consume Model Context Protocol servers
  and expose their tools / resources / prompts through the canonical
  ``Tool`` / ``MemorySource`` / ``Prompt`` shapes.
  Install with ``pip install "arc-agentkit[mcp]"``.
- ``agentkit.integrations.claude_cli`` — the other direction: project
  agentkit values (a ``Skill``) into the ``claude`` CLI's own configuration,
  so a thing expressed once in agentkit is not restated by hand as CLI JSON.
  No extra required; the CLI is installed separately.
"""
