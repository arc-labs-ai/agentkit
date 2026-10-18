"""What ``as_codex_mcp`` hands a ``codex`` child, and what it refuses to hand it.

WHY THIS FILE EXISTS
--------------------
This projection is an AUTHORITY surface: everything it names, the child process
can use. It decides three things that nothing downstream re-checks.

* **Where the credential goes.** ``serve_registry`` fences its listener with a
  bearer token. Codex has no header field, so the config names an environment
  VARIABLE and the CLI reads the token out of the child's environment at connect
  time. The alternative — a literal in ``-c mcp_servers.<name>.bearer_token=…``
  — is argv, which every local account reads with ``ps``. ``codex exec`` has no
  config-file flag, so there is nowhere else to move it to afterwards: the
  decision is made here or not at all.
* **Which tools may run without a prompt.** ``approval_mode`` is read by a CLI
  running under a noninteractive approval policy, where nobody is going to be
  asked anything. A name that lands on ``approve`` runs unattended. So the one
  invariant that matters is that a tool the HOST marked ``requires_approval``
  can never end up there, however it was also marked.
* **Whether the server is addressable at all.** The key path is
  ``mcp_servers.<name>.<field>``, a TOML path. A name or a document that cannot
  be expressed in it produces a session with no server and no error — the model
  simply appears to have lost its tools.

Pure projection, no spawn: these are dicts, and the assertions are about the
dicts. ``tests/source/unit/integrations/mcp/test_serve_registry.py`` covers the
authenticated-HTTP shape via ``spec.codex_kwargs()`` and
``tests/source/unit/integrations/mcp/test_transport_auth.py`` covers the
credential's life on the ``serve_registry`` side (generation, the 401, the
``0600`` config file, the ``O_NOFOLLOW`` refusal); none of that is repeated.
What is here is the rest of the projection, and the refusals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agentkit.models.providers.cli.codex.driver import CodexCliCognition
from agentkit.models.providers.cli.codex.mcp_config import (
    CodexMcpApprovalMode,
    as_codex_mcp,
    token_env_var,
)


@dataclass
class _Spec:
    """A served-registry spec, reduced to the attributes the projection reads.

    A double rather than a real ``McpServerSpec`` because several of the
    decisions under test are only reachable through states ``serve_registry``
    will not produce on purpose — a name it already refused, an HTTP spec whose
    port was never reserved, a tool marked both auto-approve and
    requires-approval. Two tests below use a real spec instead, to keep this
    shape honest about the attribute names it is standing in for.
    """

    name: str = "engine"
    transport: str | None = "http"
    url: str | None = "http://127.0.0.1:8931/mcp"
    auth_headers: dict[str, str] = field(default_factory=dict)
    mcp_names: dict[str, str] | None = None
    auto_approve: tuple[str, ...] = ()
    requires_approval: tuple[str, ...] = ()
    _document: dict[str, Any] = field(default_factory=dict)


def _stdio_spec(**entry: Any) -> _Spec:
    """A stdio spec whose MCP document says what ``serve_registry`` would write."""
    return _Spec(transport="stdio", url=None, _document={"mcpServers": {"engine": entry}})


# ── addressability ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("my server", id="a-space"),
        pytest.param("engine.one", id="a-dot-that-would-split-the-key-path"),
        pytest.param("engine]", id="a-bracket"),
        pytest.param("", id="empty"),
        pytest.param("e" * 65, id="too-long"),
    ],
)
def test_a_server_name_the_toml_key_path_cannot_express_is_refused_at_wiring_time(name: str) -> None:
    """The name is interpolated into ``mcp_servers.<name>.<field>`` and handed
    to the CLI as TOML. A dot splits the path and silently configures a
    DIFFERENT table; a space or a bracket is a parse error the CLI reports
    against the whole ``-c`` argument. Both land as a session that has no server
    and says nothing about why, so the model just looks worse at its job. The
    refusal happens before a spawn and names the pattern."""
    with pytest.raises(ValueError, match="cannot be addressed in Codex config"):
        as_codex_mcp(_Spec(name=name))


def test_a_name_that_is_not_even_a_string_is_refused_rather_than_formatted() -> None:
    """``getattr(spec, "name", "")`` returns whatever the object has, and an
    object that is not a served spec at all — a cognition, a registry, a
    mapping — is the realistic way this function gets called wrong. Formatting
    it into a key path would produce ``mcp_servers.None.url``, which is legal
    TOML and configures a server nobody asked for."""
    with pytest.raises(ValueError, match="cannot be addressed in Codex config"):
        as_codex_mcp(_Spec(name=None))  # type: ignore[arg-type]


def test_an_http_spec_whose_port_was_never_reserved_is_refused_with_the_way_out() -> None:
    """A spec is only complete once ``serve_registry`` has reserved its port, and
    a hand-made one is the normal way to arrive here without a URL. Projecting
    it anyway writes a table with no ``url``, which the CLI reads as a server it
    cannot reach — "MCP server failed to connect", naming nothing actionable. So
    the message names the actual fix, which is not guessable from the symptom."""
    with pytest.raises(ValueError, match="no url") as exc:
        as_codex_mcp(_Spec(url=None))

    assert "serve_registry" in str(exc.value)


# ── the credential ──────────────────────────────────────────────────────────


def test_the_bearer_token_reaches_the_child_through_its_environment_and_never_through_argv() -> None:
    """The headline decision of the module, asserted all the way to the argument
    list the cognition builds. ``codex exec`` has no config-file flag, so every
    config value becomes ``-c key=value`` in a child's argv, which any local
    account reads with ``ps`` — and a stolen token is a tool server that answers
    to somebody else. The projection therefore names an ENVIRONMENT VARIABLE in
    the config and returns the token separately, for the one place it can live
    that is not an argument."""
    spec = _Spec(auth_headers={"Authorization": "Bearer t-0ken-value"})

    kwargs = as_codex_mcp(spec)
    table = kwargs["mcp_servers"]["engine"]

    assert table["bearer_token_env_var"] == "AGENTKIT_MCP_TOKEN_ENGINE"
    assert kwargs["env"] == {"AGENTKIT_MCP_TOKEN_ENGINE": "t-0ken-value"}
    assert "t-0ken-value" not in repr(table), "the credential itself is in the config table"

    # ``filterwarnings = error`` would fail this test if constructing the
    # cognition tripped its own argv-secret heuristic — which is the point: a
    # ``*_env_var`` key is the documented safe spelling, and the two halves have
    # to agree about that or every wiring emits a warning nobody can act on.
    cognition = CodexCliCognition(**kwargs)
    argv = cognition._build_argv("task", stream_input=True)
    overrides = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"]

    assert not [arg for arg in argv if "t-0ken-value" in arg], "the token reached the process argument list"
    assert 'mcp_servers.engine.bearer_token_env_var="AGENTKIT_MCP_TOKEN_ENGINE"' in overrides
    assert cognition.env == {"AGENTKIT_MCP_TOKEN_ENGINE": "t-0ken-value"}


def test_an_unauthenticated_server_is_given_no_token_variable_it_cannot_fill() -> None:
    """``serve_registry(auth="none")`` is a legitimate configuration and has no
    token to pass. Naming a ``bearer_token_env_var`` anyway would make the CLI
    read an unset variable and send either nothing or an empty credential, which
    the server answers with a 401 — a wiring that looks complete and connects to
    nothing. The absent ``env`` key matters for the same reason: an empty one
    would still be splatted into the cognition and override nothing."""
    kwargs = as_codex_mcp(_Spec())

    assert kwargs == {"mcp_servers": {"engine": {"url": "http://127.0.0.1:8931/mcp"}}}
    assert "env" not in kwargs


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        pytest.param({"Authorization": "Bearer abc"}, "abc", id="the-scheme-is-stripped"),
        pytest.param({"authorization": "bearer abc"}, "abc", id="case-is-not-part-of-the-scheme"),
        pytest.param({"Authorization": "abc"}, "abc", id="a-bare-token-survives"),
        pytest.param({"Authorization": "Token abc"}, "Token abc", id="another-scheme-is-kept-whole"),
        pytest.param({"X-Trace": "Bearer abc"}, None, id="only-the-authorization-header-is-a-credential"),
        pytest.param({"Authorization": 7}, None, id="a-non-string-is-not-a-credential"),
        pytest.param({}, None, id="no-headers-at-all"),
    ],
)
def test_the_token_is_read_off_the_authorization_header_and_unwrapped_exactly_once(
    headers: dict[str, Any], expected: str | None
) -> None:
    """Codex sends the variable's value as the bearer token itself, so the
    ``Bearer `` prefix has to come off here — leaving it on produces an
    ``Authorization: Bearer Bearer …`` on the wire and a 401 whose cause is
    invisible from both ends. Stripping it only for that scheme is the other
    half: a value that is not a bearer credential must be passed through whole
    rather than silently shortened by seven characters.

    A header that is not ``Authorization`` is not a credential and must not
    become one — ``auth_headers`` is a mapping and a tracing or version header
    sitting in it would otherwise be projected as the server's token.
    """
    kwargs = as_codex_mcp(_Spec(auth_headers=headers))
    table = kwargs["mcp_servers"]["engine"]

    if expected is None:
        assert "bearer_token_env_var" not in table and "env" not in kwargs
    else:
        assert kwargs["env"] == {table["bearer_token_env_var"]: expected}


@pytest.mark.parametrize(
    ("server", "variable"),
    [
        pytest.param("engine", "AGENTKIT_MCP_TOKEN_ENGINE", id="plain"),
        pytest.param("my-server", "AGENTKIT_MCP_TOKEN_MY_SERVER", id="a-hyphen-is-not-shell-portable"),
        pytest.param("tools_2", "AGENTKIT_MCP_TOKEN_TOOLS_2", id="digits-and-underscores-survive"),
    ],
)
def test_each_served_registry_gets_its_own_token_variable_named_after_it(server: str, variable: str) -> None:
    """One constant would mean two served registries in one process share a
    variable and the second overwrites the first's credential — a server that
    fails to authenticate against a fence that looks correctly wired, and the
    config it is reading looks right. The sanitisation is the other half:
    ``AGENTKIT_MCP_TOKEN_MY-SERVER`` is not a portable shell identifier and some
    spawn paths drop it silently, which is the same failure with no name in it."""
    assert token_env_var(server) == variable

    first = as_codex_mcp(_Spec(name=server, auth_headers={"Authorization": "Bearer one"}))
    second = as_codex_mcp(_Spec(name="other", auth_headers={"Authorization": "Bearer two"}))

    assert first["mcp_servers"][server]["bearer_token_env_var"] == variable
    assert {**first["env"], **second["env"]} == {variable: "one", "AGENTKIT_MCP_TOKEN_OTHER": "two"}


# ── the stdio shape ─────────────────────────────────────────────────────────


def test_a_stdio_server_is_projected_from_the_document_the_other_cli_reads(tmp_path: Path) -> None:
    """A stdio server is SPAWNED by the CLI, so the command is the only thing
    that reconstructs this process's registry on the other side. Reading it out
    of the spec's own MCP document rather than re-deriving it is what keeps the
    two CLIs pointed at literally the same argv: a second description of how to
    start the process is a second thing to keep in step, and the first one is
    already right. Driven from a real spec because a double cannot show that the
    document's key names are the ones this function reads."""
    import json

    pytest.importorskip("mcp", reason="needs the `mcp` extra")

    from agentkit.core.testing import make_test_ctx
    from agentkit.integrations.mcp.serve import serve_registry
    from agentkit.tools import tool

    @tool(side_effecting=False)
    def lookup_record(key: str) -> str:
        """Look up and return the stored record for a caller-provided key."""
        return key

    spec = serve_registry(
        [lookup_record],
        name="engine",
        ctx=make_test_ctx(),
        transport="stdio",
        command=("py", "-m", "mine"),
        config_path=tmp_path / "mcp.json",
    )
    document = json.loads((tmp_path / "mcp.json").read_text())["mcpServers"]["engine"]
    table = as_codex_mcp(spec)["mcp_servers"]["engine"]

    assert (table["command"], table["args"]) == (document["command"], document["args"])
    assert "url" not in table and "bearer_token_env_var" not in table


def test_a_stdio_servers_own_environment_stays_out_of_the_codex_processs_environment() -> None:
    """Two different children, and the projection must not conflate them: the
    table's ``env`` is passed by Codex to the server it spawns, while the
    returned ``env`` is the CODEX process's own — which every ``shell`` command
    the model runs inherits. Hoisting a server's secret into the second one
    hands it to the model's shell, and nothing downstream would flag it because
    it arrives as an ordinary environment variable."""
    spec = _stdio_spec(command="py", args=["-m", "mine"], env={"SERVER_ONLY": "s3cret"})

    kwargs = as_codex_mcp(spec)

    assert kwargs["mcp_servers"]["engine"]["env"] == {"SERVER_ONLY": "s3cret"}
    assert "env" not in kwargs, "the server's own secret was hoisted into the codex process's environment"


def test_a_stdio_entry_with_nothing_to_spawn_is_refused_instead_of_projected() -> None:
    """A stdio table with no ``command`` is a server the CLI cannot start, and
    the symptom is "MCP server failed to connect" naming nothing. Refusing here
    is also what stops an empty table from being written as a valid-looking
    ``mcp_servers.engine`` entry that shadows a working one in the user's own
    ``config.toml``."""
    for entry in ({}, {"command": ""}, {"args": ["-m", "mine"]}):
        with pytest.raises(ValueError, match="names no command"):
            as_codex_mcp(_stdio_spec(**entry))


def test_the_spawn_arguments_are_rendered_as_strings_because_the_value_is_read_as_toml() -> None:
    """``-c mcp_servers.engine.args=…`` is serialised as TOML, where a
    ``PosixPath`` is not a type that exists — a path object reaches the renderer
    as whatever its ``repr`` is, and ``PosixPath('/srv/x')`` is not a filename.
    Coercing here rather than at the renderer keeps the projection's output
    something a caller can also read and compare."""
    table = as_codex_mcp(_stdio_spec(command=Path("/usr/bin/py"), args=[Path("/srv/mine.py"), 3]))["mcp_servers"][
        "engine"
    ]

    assert table == {"command": "/usr/bin/py", "args": ["/srv/mine.py", "3"]}
    assert all(isinstance(arg, str) for arg in table["args"])


def test_a_transport_this_projection_cannot_express_is_refused_rather_than_guessed() -> None:
    """Codex reads exactly two shapes. An ``sse`` spec — or an object that is not
    a spec at all, which reports no transport — has no third spelling to fall
    back on, and emitting the HTTP table for it produces a server the CLI
    connects to with the wrong protocol."""
    for transport in ("sse", None, "ws"):
        with pytest.raises(ValueError, match="does not"):
            as_codex_mcp(_Spec(transport=transport))


# ── which tools may run unattended ──────────────────────────────────────────


def test_a_tool_the_host_marked_as_needing_approval_is_never_projected_as_auto_approved() -> None:
    """``approval_mode="approve"`` means the CLI runs that tool without asking,
    under a noninteractive policy where nobody would be asked anyway. So a name
    that appears in BOTH sets — a tool the host auto-approves in general and
    also declares ``requires_approval`` — must resolve to ``prompt``, and the
    unsafe resolution is silent: the tool runs, returns a result, and the
    transcript looks like an ordinary call the host allowed."""
    spec = _Spec(
        mcp_names={
            "read_record": "mcp__engine__read_record",
            "refund": "mcp__engine__refund",
            "deploy": "mcp__engine__deploy",
        },
        auto_approve=("mcp__engine__read_record", "mcp__engine__refund"),
        requires_approval=("mcp__engine__refund", "mcp__engine__deploy"),
    )

    tools = as_codex_mcp(spec)["mcp_servers"]["engine"]["tools"]

    assert tools["read_record"]["approval_mode"] == CodexMcpApprovalMode.APPROVE
    assert tools["refund"]["approval_mode"] == CodexMcpApprovalMode.PROMPT
    assert tools["deploy"]["approval_mode"] == CodexMcpApprovalMode.PROMPT


def test_a_tool_nobody_wrote_a_rule_for_still_has_to_be_asked_about() -> None:
    """The per-tool map only names the tools this spec published. Anything else
    the server advertises later — a tool added by a ``tools_provider``, a name
    the sanitiser changed — falls through to the table default, so that default
    is the whole fence for everything unenumerated and it has to be ``prompt``."""
    table = as_codex_mcp(_Spec(mcp_names={"read_record": "mcp__engine__read_record"}))["mcp_servers"]["engine"]

    assert table["default_tools_approval_mode"] == CodexMcpApprovalMode.PROMPT
    assert set(table["tools"]) == {"read_record"}


@pytest.mark.parametrize(
    "qualified",
    [
        pytest.param("mcp__other__refund", id="another-servers-tool"),
        pytest.param("refund", id="an-unqualified-name"),
        pytest.param("mcp__engine__", id="the-prefix-with-no-tool"),
        pytest.param("mcp__engineering__refund", id="a-server-whose-name-starts-the-same"),
        pytest.param(7, id="not-even-a-string"),
    ],
)
def test_approval_metadata_from_another_server_is_refused_instead_of_written_under_this_one(qualified: Any) -> None:
    """The per-tool keys are produced by stripping ``mcp__<name>__`` off the
    qualified names. A name from a different server does not carry that prefix,
    so stripping it would write a key that is some other server's tool — or, for
    the bare prefix, an empty key — into THIS server's approval table. Either
    way a rule the host wrote about one tool silently governs another, which is
    the one mistake an approval table must not be able to make quietly."""
    spec = _Spec(mcp_names={"refund": qualified}, auto_approve=(qualified,))

    with pytest.raises(ValueError, match="from another server"):
        as_codex_mcp(spec)


def test_a_spec_with_no_tool_metadata_gets_no_approval_table_at_all() -> None:
    """``mcp_names is None`` means this object does not describe the tool set —
    a hand-wired spec, or a future transport that reports its tools elsewhere.
    Synthesising an EMPTY ``tools`` map for it would be a statement that the
    server advertises nothing, and a reader of the rendered config cannot tell
    that apart from a server whose tools were dropped."""
    table = as_codex_mcp(_Spec())["mcp_servers"]["engine"]

    assert "tools" not in table and "default_tools_approval_mode" not in table


def test_the_per_call_ceiling_is_only_set_when_a_caller_asks_for_one() -> None:
    """``tool_timeout_sec`` is a SECOND deadline, independent of the one
    ``serve_registry`` enforces, and the tighter one silently wins. Defaulting
    it would kill a legitimately slow tool — a long build, a human-in-the-loop
    approval — and the failure reads as a flake rather than as a deadline
    somebody chose, so the key is absent until it is asked for."""
    assert "tool_timeout_sec" not in as_codex_mcp(_Spec())["mcp_servers"]["engine"]
    assert as_codex_mcp(_Spec(), tool_timeout_s=12.5)["mcp_servers"]["engine"]["tool_timeout_sec"] == 12.5


def test_the_projection_is_a_plain_dict_a_caller_can_merge_rather_than_a_mutated_cognition() -> None:
    """Two served registries in one session is the whole reason this returns
    fields instead of taking and returning a cognition — a function that mutated
    one could not be called twice. The shape is also what makes
    ``CodexCliCognition(**as_codex_mcp(a), ...)`` composable at all, so a
    returned object would be a quiet break of every call site."""
    one = as_codex_mcp(_Spec(name="engine", auth_headers={"Authorization": "Bearer a"}))
    two = as_codex_mcp(_Spec(name="audit", url="http://127.0.0.1:8932/mcp", auth_headers={"Authorization": "Bearer b"}))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cognition = CodexCliCognition(
            mcp_servers={**one["mcp_servers"], **two["mcp_servers"]},
            env={**one["env"], **two["env"]},
        )

    assert set(cognition.mcp_servers) == {"engine", "audit"}
    assert cognition.env == {"AGENTKIT_MCP_TOKEN_ENGINE": "a", "AGENTKIT_MCP_TOKEN_AUDIT": "b"}
