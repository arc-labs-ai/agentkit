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
  agentkit values (a ``Skill``, a middleware chain) into the ``claude`` CLI's
  own configuration, so a thing expressed once in agentkit is not restated by
  hand as CLI JSON. No extra required; the CLI is installed separately.
- ``agentkit.integrations.codex_cli`` — the same direction for OpenAI's
  ``codex`` CLI, which reads its configuration as TOML keys rather than a JSON
  document. Deliberately smaller than its Claude sibling: Codex has no
  pre-tool hook and no sub-agent roster, so two of that package's three
  adapters have no counterpart, and its docstring says which and why rather
  than leaving a reader to search. No extra required.
"""
