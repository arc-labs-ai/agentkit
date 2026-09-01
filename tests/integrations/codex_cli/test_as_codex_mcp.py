"""``as_codex_mcp`` — the same served registry, spelled the way Codex reads it.

``serve_registry`` already does the hard half: it advertises agentkit ``Tool``s
over MCP with their schemas, ``requires_approval`` flags and ``caps`` intact.
What it writes for the CLI is a ``--mcp-config`` document, which is Claude
Code's format. Codex reads MCP servers out of ``config.toml``, addressed as
``mcp_servers.<name>.<key>``.

Most of this module is that rename, and a rename does not need many tests. Two
things are not renames, and they are what the file is really about:

**The bearer token.** ``serve_registry`` defaults to an authenticated HTTP
listener and puts the token in an ``Authorization`` header. Codex has no header
field — it has ``bearer_token_env_var``, which names an ENVIRONMENT VARIABLE the
CLI reads at connect time. So the projection has to place the token somewhere
the child will see it, and the one place it must NOT go is the argv, which is
world-readable in ``ps`` output on most systems.

**The refusals.** A projection that quietly produced a server the CLI cannot
reach would fail as "the model did not use my tools", which reads as a prompting
problem. Each shape that cannot work raises at wiring time instead.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentkit.agents.cognition import CodexCliCognition
from agentkit.integrations.codex_cli import as_codex_mcp, token_env_var
from tests.agents.cognition.test_codex_cli_flags import overrides


class _HttpSpec:
    """The public surface of an authenticated HTTP ``McpServerSpec``.

    A stub rather than a real ``serve_registry`` call for the pure-projection
    tests: those need no listener and no ``mcp`` extra, and a test that reserves
    a port to assert on a dict is a test that can fail for reasons unrelated to
    what it checks. The real object is exercised at the bottom of this file,
    behind an ``importorskip``.
    """

    name = "engine"
    transport = "http"
    url = "http://127.0.0.1:54321/mcp"

    def __init__(self, token: str | None = "s3cret") -> None:
        self.auth_headers = {"Authorization": f"Bearer {token}"} if token else {}


class _StdioSpec:
    name = "engine"
    transport = "stdio"
    url = None
    auth_headers: dict[str, str] = {}
    _document = {
        "mcpServers": {
            "engine": {
                "type": "stdio",
                "command": "/usr/bin/python3",
                "args": ["-m", "myapp.tools"],
                "env": {"MYAPP_MODE": "serve"},
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. the rename
# ─────────────────────────────────────────────────────────────────────────────


def test_an_http_server_projects_to_a_url_and_a_token_variable() -> None:
    kwargs = as_codex_mcp(_HttpSpec())
    assert kwargs["mcp_servers"] == {
        "engine": {
            "url": "http://127.0.0.1:54321/mcp",
            "bearer_token_env_var": "AGENTKIT_MCP_TOKEN_ENGINE",
        }
    }
    assert kwargs["env"] == {"AGENTKIT_MCP_TOKEN_ENGINE": "s3cret"}


def test_an_unauthenticated_http_server_needs_no_variable() -> None:
    """``auth="none"`` is still a supported choice on a host nobody else shares.
    Emitting an empty ``bearer_token_env_var`` would make the CLI look for a
    credential that does not exist."""
    kwargs = as_codex_mcp(_HttpSpec(token=None))
    assert kwargs["mcp_servers"]["engine"] == {"url": "http://127.0.0.1:54321/mcp"}
    assert "env" not in kwargs


def test_a_stdio_server_projects_its_command_verbatim() -> None:
    """Read out of the spec's own MCP document rather than re-derived, so both
    CLIs are pointed at literally the same argv. A projection that rebuilt the
    command would be a second description of how to start this process."""
    kwargs = as_codex_mcp(_StdioSpec())
    assert kwargs["mcp_servers"] == {
        "engine": {
            "command": "/usr/bin/python3",
            "args": ["-m", "myapp.tools"],
            "env": {"MYAPP_MODE": "serve"},
        }
    }
    assert "env" not in kwargs


def test_the_servers_own_env_is_not_the_codex_processs_env() -> None:
    """Two different children. The server's ``env`` is passed by Codex to the
    process IT spawns; the returned ``env`` is Codex's own. Merging them would
    make a server's secret visible to every shell command the model runs."""
    kwargs = as_codex_mcp(_StdioSpec())
    assert kwargs["mcp_servers"]["engine"]["env"] == {"MYAPP_MODE": "serve"}
    assert "env" not in kwargs


def test_a_tool_timeout_is_optional_and_renders_in_seconds() -> None:
    """Left ``None`` by default for the same reason ``serve_registry(timeout_s=None)``
    is: a default deadline kills a legitimately slow tool and the failure reads
    as a flake."""
    assert "tool_timeout_sec" not in as_codex_mcp(_StdioSpec())["mcp_servers"]["engine"]
    timed = as_codex_mcp(_StdioSpec(), tool_timeout_s=45)["mcp_servers"]["engine"]
    assert timed["tool_timeout_sec"] == 45


def test_the_token_variable_name_is_derived_from_the_server_name() -> None:
    """Not one constant. Two served registries in one process would otherwise
    share one variable and the second would overwrite the first's credential — a
    server that fails to authenticate against a fence that looks correctly
    wired."""
    assert token_env_var("engine") == "AGENTKIT_MCP_TOKEN_ENGINE"
    assert token_env_var("my-server") == "AGENTKIT_MCP_TOKEN_MY_SERVER"
    assert token_env_var("a_b") == "AGENTKIT_MCP_TOKEN_A_B"


def test_two_servers_get_two_variables() -> None:
    class _Second(_HttpSpec):
        name = "search"

    first = as_codex_mcp(_HttpSpec())
    second = as_codex_mcp(_Second())
    assert set(first["env"]) != set(second["env"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. it wires straight into the cognition
# ─────────────────────────────────────────────────────────────────────────────


def test_the_kwargs_splat_into_the_cognition_and_reach_the_argv() -> None:
    """The ergonomic claim: ``CodexCliCognition(**as_codex_mcp(spec))`` and a
    caller never assembles a TOML path by hand — which is the step where the
    quoting goes wrong and the failure is a server that silently does not
    appear."""
    cog = CodexCliCognition(model="gpt-5-codex", **as_codex_mcp(_HttpSpec()))
    argv = cog._build_argv("do it")
    got = overrides(argv)

    assert got["mcp_servers.engine.url"] == '"http://127.0.0.1:54321/mcp"'
    assert got["mcp_servers.engine.bearer_token_env_var"] == '"AGENTKIT_MCP_TOKEN_ENGINE"'


def test_the_token_never_reaches_the_argv() -> None:
    """The one placement a credential must not take: ``ps`` output is
    world-readable on most systems, and ``bearer_token_env_var`` exists
    precisely so the token does not have to travel there."""
    cog = CodexCliCognition(**as_codex_mcp(_HttpSpec(token="do-not-log-me")))
    assert not [a for a in cog._build_argv("do it") if "do-not-log-me" in a]


@pytest.mark.asyncio
async def test_the_token_does_reach_the_child_environment() -> None:
    """The other half. A variable name in the config with nothing behind it is a
    server that fails to authenticate."""
    from agentkit.testing.fakes import FakeCodexCli, codex_turn
    from tests.agents.cognition.test_codex_cli import drive

    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli, **as_codex_mcp(_HttpSpec(token="tok-123")))
    await drive(cog)
    assert cli.invocations[-1].env["AGENTKIT_MCP_TOKEN_ENGINE"] == "tok-123"


def test_two_servers_compose_by_merging_two_dicts() -> None:
    """Which is why this returns a dict of fields rather than mutating a
    cognition: a function that took and returned one could not be called
    twice."""

    class _Second(_HttpSpec):
        name = "search"

    first = as_codex_mcp(_HttpSpec())
    second = as_codex_mcp(_Second())
    cog = CodexCliCognition(
        mcp_servers={**first["mcp_servers"], **second["mcp_servers"]},
        env={**first["env"], **second["env"]},
    )
    got = overrides(cog._build_argv("do it"))
    assert "mcp_servers.engine.url" in got
    assert "mcp_servers.search.url" in got


# ─────────────────────────────────────────────────────────────────────────────
# 3. the refusals
# ─────────────────────────────────────────────────────────────────────────────


def test_a_name_that_cannot_be_a_toml_key_is_refused() -> None:
    """The key path is ``mcp_servers.<name>.<field>``, so a name with a dot or a
    space would address something else entirely — or nothing."""

    class _Bad(_HttpSpec):
        name = "my server.v2"

    with pytest.raises(ValueError, match="cannot be addressed"):
        as_codex_mcp(_Bad())


def test_an_unknown_transport_is_refused_by_name() -> None:
    """Codex reads two shapes. A third would project to a table the CLI ignores,
    and the failure would look like the model refusing to use the tools."""

    class _Weird(_HttpSpec):
        transport = "websocket"

    with pytest.raises(ValueError, match="which this projection does not"):
        as_codex_mcp(_Weird())


def test_an_http_spec_with_no_url_is_refused() -> None:
    """A spec is only complete once ``serve_registry`` has reserved its port."""

    class _Incomplete(_HttpSpec):
        url = None

    with pytest.raises(ValueError, match="no url"):
        as_codex_mcp(_Incomplete())


def test_a_stdio_spec_with_no_command_is_refused() -> None:
    class _Incomplete(_StdioSpec):
        _document: dict[str, Any] = {"mcpServers": {"engine": {"type": "stdio"}}}

    with pytest.raises(ValueError, match="names no command"):
        as_codex_mcp(_Incomplete())


# ─────────────────────────────────────────────────────────────────────────────
# 4. against a real McpServerSpec
# ─────────────────────────────────────────────────────────────────────────────


def test_a_real_spec_projects_and_the_method_agrees_with_the_function(tmp_path: Any) -> None:
    """The stubs above describe the surface; this asserts the real object has
    it. ``McpServerSpec.codex_kwargs()`` exists so a reader who found
    ``cli_kwargs`` finds the counterpart in the same place instead of concluding
    there is none, and it must not drift from the function it delegates to.
    """
    pytest.importorskip("mcp", reason="needs the `mcp` extra")
    from agentkit.integrations.mcp import serve_registry
    from agentkit.testing import make_test_ctx
    from agentkit.tools import tool

    @tool(side_effecting=False)
    async def add(a: int, b: int) -> int:
        """Add two integers together and return their sum. Read-only."""
        return a + b

    spec = serve_registry([add], name="engine", ctx=make_test_ctx(), config_path=tmp_path / "c.json")
    try:
        kwargs = as_codex_mcp(spec)
        assert kwargs == spec.codex_kwargs()
        assert kwargs["mcp_servers"]["engine"]["url"].startswith("http://127.0.0.1:")
        assert kwargs["mcp_servers"]["engine"]["bearer_token_env_var"] == "AGENTKIT_MCP_TOKEN_ENGINE"
        # The token came off the listener that will enforce it, not out of a
        # second generator — so the config and the fence cannot disagree.
        assert kwargs["env"]["AGENTKIT_MCP_TOKEN_ENGINE"]
        assert kwargs["env"]["AGENTKIT_MCP_TOKEN_ENGINE"] in spec.auth_headers["Authorization"]
    finally:
        (tmp_path / "c.json").unlink(missing_ok=True)


def test_the_codex_projection_has_no_builtin_tools_switch(tmp_path: Any) -> None:
    """``cli_kwargs(builtin_tools=False)`` says "only OUR tools" by disabling the
    CLI's own. Codex has no tool allow-list — every session has ``shell`` — so
    the parameter cannot exist here, and one that quietly did nothing would be
    worse than its absence. What contains the CLI's own tools in a Codex session
    is ``sandbox=``.
    """
    pytest.importorskip("mcp", reason="needs the `mcp` extra")
    import inspect

    from agentkit.integrations.mcp import McpServerSpec

    del tmp_path
    assert "builtin_tools" in inspect.signature(McpServerSpec.cli_kwargs).parameters
    assert "builtin_tools" not in inspect.signature(McpServerSpec.codex_kwargs).parameters
    assert "builtin_tools" not in inspect.signature(as_codex_mcp).parameters
