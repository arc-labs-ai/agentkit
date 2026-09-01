"""``as_codex_mcp`` — point the ``codex`` CLI at a served agentkit ``ToolRegistry``.

:func:`~agentkit.integrations.mcp.serve_registry` already does the hard half:
it advertises agentkit ``Tool``s over MCP with their schemas, their
``requires_approval`` flags and their ``caps`` intact, and hands back an
:class:`~agentkit.integrations.mcp.serve.McpServerSpec`. What it writes for the
CLI is a ``--mcp-config`` **document**, which is Claude Code's format and not
Codex's: Codex reads MCP servers out of ``config.toml``, addressed as
``mcp_servers.<name>.<key>``, and overridden on the command line as
``-c mcp_servers.<name>.<key>=<value>``.

Same server, same tools, different spelling of "here is where it lives". This
module is that translation, so a caller writes::

    spec = serve_registry(registry, name="engine", ctx=ctx)
    async with spec:
        cognition = CodexCliCognition(**as_codex_mcp(spec))
        result = await Agent(name="dev", cognition=cognition).run(task, ctx)

and never assembles a TOML path by hand — which is the step where the quoting
goes wrong and the failure is a server that silently does not appear in the
session.

THE ONE THING THAT IS NOT A RENAME: THE BEARER TOKEN
----------------------------------------------------
``serve_registry`` defaults to an authenticated HTTP listener and puts the
token in the config document as an ``Authorization`` header. Codex has no
header field. What it has is ``bearer_token_env_var``: the config names an
ENVIRONMENT VARIABLE, and the CLI reads the token out of the child's
environment at connect time.

So the projection has to place the token somewhere the child will see it, which
is why it returns a ``env=`` entry alongside ``mcp_servers=`` and why
:class:`~agentkit.agents.cognition.CodexCliCognition` has an ``env`` field at
all. The variable is named after the server (``AGENTKIT_MCP_TOKEN_<SERVER>``)
rather than being one constant, because two served registries in one process
would otherwise share one variable and the second would overwrite the first's
credential — a server that fails to authenticate against a fence that looks
correctly wired.

The token is deliberately NOT put in ``config_overrides`` as a literal. Codex
persists nothing from ``-c``, so it would not leak to disk, but it WOULD land in
the child's argv — which is world-readable in ``ps`` output on most systems and
is the one placement a credential must never take. ``bearer_token_env_var``
exists precisely so it does not have to.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["as_codex_mcp", "token_env_var"]

# Codex reads an MCP server's key path as TOML, so the server name has to be a
# bare key. ``serve_registry`` already enforces ``[A-Za-z0-9_-]{1,64}`` on it,
# which is a superset of what a bare TOML key allows — a name with a hyphen is
# legal TOML but a name this pattern rejects is not addressable at all.
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Non-alphanumerics in a server name become underscores in the env-var name.
# ``AGENTKIT_MCP_TOKEN_MY-SERVER`` is not a portable shell identifier and some
# spawn paths silently drop it.
_NOT_IDENT = re.compile(r"[^A-Z0-9]")


def token_env_var(server: str) -> str:
    """The environment variable a server's bearer token travels in.

    Public because it is part of the contract: a caller who wires the server
    into ``config.toml`` themselves — rather than through ``as_codex_mcp`` —
    still needs the two halves to agree on the name, and a function is the only
    way for them to agree that does not involve copying a format string.
    """
    return f"AGENTKIT_MCP_TOKEN_{_NOT_IDENT.sub('_', server.upper())}"


def as_codex_mcp(spec: Any, *, tool_timeout_s: float | None = None) -> dict[str, Any]:
    """The ``CodexCliCognition`` kwargs that wire ``spec`` in.

    Returns ``{"mcp_servers": {...}}``, plus ``{"env": {...}}`` when the server
    is authenticated — splat it straight into the constructor::

        CodexCliCognition(model="gpt-5-codex", **as_codex_mcp(spec))

    Mirrors :meth:`~agentkit.integrations.mcp.serve.McpServerSpec.cli_kwargs`
    on purpose, including that it is a plain dict of fields rather than a
    mutated cognition: a caller composing two servers merges two dicts, and a
    function that took and returned a cognition could not be called twice.

    Unlike ``cli_kwargs`` there is no ``builtin_tools=`` switch. Codex has no
    tool allow-list, so "only OUR tools" is not something a flag can say — the
    session always has ``shell``. That is a real difference in what the two
    CLIs can be asked for, and inventing a parameter that quietly did nothing
    would be worse than its absence.

    ``tool_timeout_s`` sets Codex's own per-call ceiling
    (``mcp_servers.<name>.tool_timeout_sec``). Left ``None`` by default for the
    same reason ``serve_registry(timeout_s=None)`` is: a default deadline kills
    a legitimately slow tool — a long build, a human-in-the-loop approval — and
    the failure reads as a flake. Note it is a SECOND deadline, independent of
    the one ``serve_registry`` enforces on the agentkit side; set them together
    or the tighter one silently wins.

    Raises:
        ValueError: when the spec describes something Codex cannot be pointed
            at — see the checks below, each of which would otherwise be a
            server that never connects and a session that just seems worse at
            its job.
    """
    name = getattr(spec, "name", "")
    if not isinstance(name, str) or not _BARE_KEY.fullmatch(name):
        raise ValueError(
            f"MCP server name {name!r} cannot be addressed in Codex config: the key path is "
            "mcp_servers.<name>.<field>, so the name must match [A-Za-z0-9_-]{1,64}"
        )

    transport = getattr(spec, "transport", None)
    table: dict[str, Any]
    env: dict[str, str] = {}

    if transport == "http":
        url = getattr(spec, "url", None)
        if not url:
            raise ValueError(
                f"MCP server {name!r} reports transport='http' but no url. A spec is only "
                "complete once serve_registry has reserved its port — build the cognition "
                "from a spec that function returned, not from a hand-made one."
            )
        table = {"url": url}
        token = _bearer_token(spec)
        if token is not None:
            var = token_env_var(name)
            table["bearer_token_env_var"] = var
            env[var] = token
    elif transport == "stdio":
        command, args, server_env = _stdio_command(spec, name)
        table = {"command": command, "args": list(args)}
        if server_env:
            # The SERVER's own environment, which Codex passes to the process it
            # spawns — not the same thing as the ``env`` returned above, which is
            # the CODEX process's environment. Two different children; keeping
            # them distinct is what stops a server's secret from also being
            # visible to every shell command the model runs.
            table["env"] = dict(server_env)
    else:
        raise ValueError(
            f"MCP server {name!r} has transport={transport!r}, which this projection does not "
            "know how to express. Codex reads two shapes: a stdio server (command/args/env) "
            "and a streamable-HTTP one (url/bearer_token_env_var)."
        )

    if tool_timeout_s is not None:
        table["tool_timeout_sec"] = tool_timeout_s

    kwargs: dict[str, Any] = {"mcp_servers": {name: table}}
    if env:
        kwargs["env"] = env
    return kwargs


def _bearer_token(spec: Any) -> str | None:
    """The token from the spec's ``auth_headers``, or ``None`` when unauthenticated.

    Read off the headers rather than off a ``spec.token`` attribute because
    there deliberately is no such attribute — see ``McpServerSpec.auth_headers``,
    which exists in that shape so a bare token cannot be concatenated into a
    URL and then logged.
    """
    headers = getattr(spec, "auth_headers", None) or {}
    for key, value in headers.items():
        if key.lower() == "authorization" and isinstance(value, str):
            prefix = "bearer "
            return value[len(prefix) :] if value.lower().startswith(prefix) else value
    return None


def _stdio_command(spec: Any, name: str) -> tuple[str, list[str], dict[str, str]]:
    """``(command, args, env)`` for a stdio spec.

    Read out of the spec's own MCP document rather than re-derived, so the two
    CLIs are pointed at literally the same argv. A stdio server is spawned by
    the CLI, so a projection that reconstructed the command would be a second
    description of how to start this process — and the first one is already
    right.
    """
    document = getattr(spec, "_document", None) or {}
    entry = (document.get("mcpServers") or {}).get(name) or {}
    command = entry.get("command")
    if not command:
        raise ValueError(
            f"MCP server {name!r} reports transport='stdio' but its config names no command. "
            "serve_registry(transport='stdio') requires command=; a spec without one cannot "
            "be started by either CLI."
        )
    return str(command), [str(a) for a in entry.get("args") or ()], dict(entry.get("env") or {})
