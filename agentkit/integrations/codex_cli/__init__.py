"""Codex CLI integration — join agentkit's primitives to the ``codex`` CLI's own.

``CodexCliCognition`` (in ``agentkit.agents.cognition``) is the runner: it
subprocesses the CLI for one agent. This package is the other half — the
adapters that let agentkit values BE the CLI's configuration instead of being
restated as it by hand.

    from agentkit.integrations.codex_cli import as_codex_mcp

One ships, and the package docstring's more useful half is the list of things
that do NOT, because the sibling package next door has them and a reader
arriving here will look:

* :func:`as_codex_mcp` projects a served ``ToolRegistry`` into the
  ``mcp_servers=`` configuration Codex reads, so the CLI calls YOUR tools —
  through agentkit's own tool path, where the middleware chain applies. This is
  the counterpart of ``McpServerSpec.cli_kwargs()``.

WHAT HAS NO COUNTERPART, AND WHY
--------------------------------
``agentkit.integrations.claude_cli.hook_settings``
    No equivalent exists, and none can be built. It generates Claude Code
    ``PreToolUse`` hooks so the tool-middleware chain reaches the CLI's own
    ``Write`` / ``Bash`` / ``WebFetch``. Codex has no pre-tool hook: its
    extension points are ``notify`` (which fires AFTER the fact and cannot
    refuse) and execpolicy ``.rules`` files (which are static patterns, not a
    Python chain against a live ``ctx``). So for a Codex session there is no
    way to make ``egress()``, ``guard()``, ``audit()`` or ``memoize()`` apply to
    ``shell``. What contains ``shell`` instead is the OS sandbox
    (``sandbox="read-only"`` / ``"workspace-write"``), which is enforced below
    the CLI rather than by the model's cooperation — a different guarantee, and
    on that one axis a stronger one. ``CodexCliCognition``'s bypass warning
    says this at the wiring site rather than leaving it here to be found.

``agentkit.integrations.claude_cli.as_cli_agents``
    No equivalent exists. It projects a ``Skill`` into a Claude Code sub-agent
    definition; Codex has no sub-agent roster to project into. A skill that
    should run as its own Codex session is its own ``Agent`` with its own
    ``CodexCliCognition``, composed with ``Workflow`` or
    ``CoordinatorCognition`` on agentkit's side — which is where the roster and
    the policy live anyway.

``agentkit.integrations.mcp.approvals.ApprovalServer``
    Not projected, and the gap is narrower than it looks. It answers Claude
    Code's ``--permission-prompt-tool``, an MCP tool the CLI calls to ask
    permission. Codex's approval seam is ``--ask-for-approval``, which prompts
    the TERMINAL — there is no tool to point at a reviewer. A service therefore
    runs ``ask_for_approval="never"`` and relies on the sandbox; a
    human-in-the-loop gate belongs on agentkit's side of the boundary, in front
    of the run.

No extra required for this package itself. :func:`as_codex_mcp` takes a spec
that ``agentkit.integrations.mcp.serve_registry`` produced, and THAT needs the
``mcp`` extra (``pip install "arc-agentkit[mcp]"``).
"""

from __future__ import annotations

from agentkit.integrations.codex_cli.mcp_config import as_codex_mcp, token_env_var

__all__ = ["as_codex_mcp", "token_env_var"]
