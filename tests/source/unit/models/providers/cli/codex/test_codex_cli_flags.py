"""The argv this cognition hands ``codex``, down to the ordering and the quoting.

WHY THIS FILE EXISTS
--------------------
Two of the three shipped bugs in this provider were argv bugs, and both failed
the same way: the CLI exited 2 on a clap usage message before starting a thread,
so a run that looked wired produced nothing at all.

* ``--search`` is a PARENT-level flag. It used to be accepted after the
  subcommand; codex 0.152.1 answers ``error: unexpected argument '--search'
  found``. Putting the subcommand before the options is the one layout that is
  wrong for both the global flags and the parent-only ones.
* ``--ask-for-approval`` was REMOVED from ``codex exec`` entirely and is a
  config key now, so a cognition still carrying the field has to render it as
  ``-c approval_policy=…``.

The quoting is the same class of failure one layer down. ``-c key=value`` is
parsed as TOML, where a bare word is not a string: an unquoted value is a parse
error, a quoted bool is the four-character string ``"True"``, and a Windows
path's backslashes end the value early. None of that is visible from the Python
side — it is a subprocess exiting with a complaint on stderr.

``_build_argv`` is called directly here rather than through a spawn, because the
argv IS the unit under test and a recorded invocation would only add a
subprocess's worth of indirection between the assertion and the thing it pins.
``tests/source/unit/models/providers/cli/test_cli_version_matrix.py`` is the
other half: it asks the installed binary whether these layouts parse.
"""

from __future__ import annotations

import warnings

import pytest

from agentkit.models.providers.cli.codex.driver import CodexCliCognition, _toml_scalar


def test_the_parent_only_search_flag_is_emitted_before_the_exec_subcommand() -> None:
    """``--search`` is documented on ``codex``, not on ``codex exec``. Emitting
    the subcommand first is the single layout clap cannot parse for both the
    global flags and the parent-only ones, and it killed every run with
    ``web_search=True`` — exit 2, no thread, no output, a usage message on
    stderr the caller had to go digging for."""
    argv = CodexCliCognition(web_search=True)._build_argv("task", stream_input=True)

    assert argv[:3] == ["codex", "--search", "exec"]
    assert argv.index("--search") < argv.index("exec")


def test_the_subcommand_comes_after_every_option_so_a_resume_still_parses() -> None:
    """``codex exec`` marks some flags global (they parse on either side of
    ``resume``) and flattens others onto the parent only (they do not). Ahead of
    the subcommand is the one position correct for both, and getting it wrong is
    a parse error several seconds into a run whose message names the flag rather
    than its position."""
    cognition = CodexCliCognition(model="gpt-5-codex", resume_session_id="thread-7")
    argv = cognition._build_argv("task", resume=cognition._resume_target(), stream_input=True)

    resume_at = argv.index("resume")
    assert argv[resume_at : resume_at + 2] == ["resume", "thread-7"]
    assert argv.index("--model") < resume_at
    assert argv.index("--json") < resume_at
    # The positional is last, after the sub-subcommand and its argument.
    assert argv[-1] == "-"


def test_a_config_override_value_is_serialised_as_toml_rather_than_interpolated() -> None:
    """``-c key=value`` is read as TOML, so the right-hand side has to be
    SERIALISED, not formatted in. Interpolating gets every type wrong in a
    different way: a bare string is a parse error, a bool becomes the string
    ``"True"`` (truthy, and not what the caller wrote), and a Windows path's
    backslashes are read as escapes that end the value early."""
    assert _toml_scalar("read-only") == '"read-only"'
    assert _toml_scalar(True) == "true"
    assert _toml_scalar(7) == "7"
    assert _toml_scalar(0.5) == "0.5"
    # A backslash survives as a backslash rather than starting an escape.
    assert _toml_scalar("C:\\Users\\svc") == '"C:\\\\Users\\\\svc"'
    # And a quote in the value cannot close the value.
    assert _toml_scalar('say "hi"') == '"say \\"hi\\""'


def test_the_approval_policy_reaches_the_cli_as_a_quoted_config_key_not_a_removed_flag() -> None:
    """``--ask-for-approval`` no longer exists on ``codex exec``; passing it is
    exit 2 before a thread starts. It is a config key now, and the value has to
    arrive quoted or the TOML parse fails — so this one field exercises both the
    moved-flag fix and the serialisation rule at once."""
    argv = CodexCliCognition(sandbox="read-only", ask_for_approval="never")._build_argv("task", stream_input=True)

    assert "--ask-for-approval" not in argv
    assert argv[argv.index("-c") + 1] == 'approval_policy="never"'
    assert ["--sandbox", "read-only"] == argv[argv.index("--sandbox") : argv.index("--sandbox") + 2]


def test_nested_config_tables_are_rendered_as_toml_inline_tables_not_as_json() -> None:
    """An MCP per-tool approval map is a nested table, and TOML inline tables use
    ``key = value`` where JSON uses ``key: value``. Handing the CLI the JSON form
    is a rejected string, which is the exact step a caller gets wrong by hand —
    and the reason this cognition flattens ``mcp_servers`` for them."""
    assert _toml_scalar({"approve": True, "name": "x"}) == '{"approve"=true,"name"="x"}'
    assert _toml_scalar(["a", "b"]) == '["a","b"]'
    assert _toml_scalar({"outer": {"inner": 1}}) == '{"outer"={"inner"=1}}'


def test_a_none_config_override_is_refused_because_toml_has_no_null() -> None:
    """``json.dumps(None)`` is ``null``, which the CLI reads either as the
    four-character string or as a parse error depending on the key. Both are a
    setting that is not what the caller wrote, and neither is distinguishable
    from "the key was ignored"."""
    with pytest.raises(ValueError, match="TOML has no null"):
        _toml_scalar(None)


def test_an_explicit_config_override_wins_over_one_the_cognition_derived() -> None:
    """Someone who writes ``config_overrides={"model_reasoning_effort": "high"}``
    alongside ``effort="low"`` has said two things, and the one typed as a config
    key is the more specific. Emitting both would leave the CLI applying
    whichever ``-c`` it saw last — a precedence nobody chose."""
    argv = CodexCliCognition(
        effort="low", config_overrides={"model_reasoning_effort": "high"}
    )._build_argv("task", stream_input=True)

    rendered = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    assert rendered == ['model_reasoning_effort="high"']


def test_a_prompt_that_begins_with_a_dash_is_separated_so_it_is_not_read_as_flags() -> None:
    """A task like ``"--force a rewrite"`` is a perfectly ordinary instruction
    and an unknown argument to clap. Without the separator the run dies on the
    user's own wording, which reads as the CLI rejecting the question."""
    argv = CodexCliCognition()._build_argv("--force a rewrite")

    assert argv[-2:] == ["--", "--force a rewrite"]


def test_stream_input_passes_a_bare_dash_without_the_separator_that_would_literalise_it() -> None:
    """``-`` is the CLI's documented spelling of "read the instructions from
    stdin". It starts with a dash, so the obvious reflex is to guard it with
    ``--`` — which turns it into a one-character prompt and sends the real task
    nowhere. This is the one place the separator must NOT appear."""
    argv = CodexCliCognition()._build_argv("the real task", stream_input=True)

    assert argv[-1] == "-"
    assert "--" not in argv
    assert "the real task" not in argv


def test_colour_is_disabled_unconditionally_because_stderr_is_stored_not_displayed() -> None:
    """A failed run's stderr is surfaced verbatim in ``tests.evals["stderr"]``,
    which is checkpointed, logged and fanned out to observers. ANSI escapes in a
    stored diagnostic are noise in every reader that is not a terminal."""
    argv = CodexCliCognition()._build_argv("task", stream_input=True)

    assert ["--color", "never"] == argv[argv.index("--color") : argv.index("--color") + 2]
    assert "--json" in argv


def test_a_secret_shaped_config_key_warns_because_argv_is_readable_with_ps() -> None:
    """``-c key=value`` is the only way to set a Codex override and it is an
    ARGUMENT — ``codex exec`` has no config-file flag, so unlike the Claude
    cognition there is nowhere to move the value to. Warned rather than refused,
    because the value may be a placeholder or a short-lived token the caller
    accepts on this host; silent is the one option that is wrong."""
    with pytest.warns(UserWarning, match="ps"):
        CodexCliCognition(config_overrides={"api_key": "sk-not-real"})


def test_the_default_sandbox_flag_is_omitted_so_the_clis_own_config_still_applies() -> None:
    """An unset ``sandbox`` means "pass no flag", which leaves the CLI's default
    and the user's ``config.toml`` in charge. Synthesising ``--sandbox
    read-only`` here would silently override a machine-level setting while
    ``effective_sandbox`` reports the same string either way."""
    argv = CodexCliCognition()._build_argv("task", stream_input=True)

    assert "--sandbox" not in argv
    assert CodexCliCognition().effective_sandbox == "read-only"


def test_a_credential_nested_in_a_stdio_servers_env_is_still_warned_about() -> None:
    """`codex exec` has no config-file flag, so a secret in `-c key=value` is
    readable by any local account with `ps` — which is why this cognition warns
    and points at the `*_env_var` pattern instead.

    The warning only ever inspected TOP-LEVEL table keys. A stdio MCP server
    carries its own process environment at `mcp_servers.<name>.env`, which
    reaches argv verbatim as
    `-c mcp_servers.engine.env={"SERVER_SECRET"="s3cr3t"}` — so the single
    placement the module docstring says a credential must never take was also
    the one placement the warning could not see: `env` does not look like a
    secret, and the name that does was one level down and never read.

    The clean twins matter as much: a top-level secret must still warn (no
    regression), and the documented-safe `bearer_token_env_var` spelling must
    still produce no warning at all, or the heuristic starts crying wolf about
    the very pattern it recommends.
    """
    nested = {"engine": {"command": "python", "args": ["-m", "srv"], "env": {"SERVER_SECRET": "s3cr3t"}}}
    with pytest.warns(UserWarning, match="rendered into the process argument list") as caught:
        CodexCliCognition(mcp_servers=nested)
    assert "env.SERVER_SECRET" in str(caught[0].message), (
        f"the warning must name the nested key by its dotted path: {caught[0].message}"
    )

    with pytest.warns(UserWarning, match="rendered into the process argument list"):
        CodexCliCognition(mcp_servers={"engine": {"command": "python", "api_key": "sk-top-level"}})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CodexCliCognition(mcp_servers={"engine": {"command": "python", "bearer_token_env_var": "MY_TOKEN"}})
