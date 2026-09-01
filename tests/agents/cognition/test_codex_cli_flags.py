"""The argv and the environment: what the CLI is actually asked for.

Every field on :class:`CodexCliCognition` is a promise that a flag reaches the
binary, and a field that renders nothing is a setting an operator believes is
in force. These tests hold each one to the argv, and hold the ORDER of the argv
to the one layout that parses.

The ordering is the part worth reading. ``codex exec`` marks some of its flags
``global = true`` — they parse on either side of the ``resume`` subcommand — and
flattens others onto the parent alone, where they do not. Putting every option
ahead of ``resume`` is the single layout that is correct for both, and getting
it wrong is a clap error several seconds into a run whose message names the
flag rather than its position. There is a real-binary test at the bottom that
checks the whole assembled argv is accepted, because that is the only thing
that can actually prove it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CodexCliCognition
from agentkit.prompts.prompt import Prompt
from agentkit.testing.fakes import FakeCodexCli, codex_turn
from agentkit.testing.fakes.ctx import FakeCtx
from tests.agents.cognition.test_codex_cli import drive, final_of

_REAL_CODEX = shutil.which("codex")
_SKIP_REAL = os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1"

real_codex = pytest.mark.skipif(
    _REAL_CODEX is None or _SKIP_REAL,
    reason="codex CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


def argv_of(cog: CodexCliCognition, **kw: object) -> list[str]:
    """The argv this cognition would build, without spawning anything.

    Calling ``_build_argv`` rather than driving a fake for the pure-rendering
    assertions: those need no process, and a test that spawns to read an argv
    also has to supply a plausible stream, which is noise around the one line
    it is checking. The tests that need the ARGV AS SPAWNED (the env, the cwd,
    the temp-file paths) drive a fake and read ``invocations[-1]``.
    """
    return cog._build_argv("do the thing", **kw)  # type: ignore[arg-type]


def _exists(path: Path) -> bool:
    """``Path.exists`` behind a sync helper.

    ``ASYNC240`` refuses blocking pathlib calls inside an async def, and it is
    right in general — but the assertion is that the run's scratch directory is
    GONE after the run, so it has to happen in the test body, and one ``stat``
    on a path that does not exist has nothing to block on.
    """
    return path.exists()


def value_after(argv: list[str], flag: str) -> str:
    assert flag in argv, f"{flag} not in {argv}"
    return argv[argv.index(flag) + 1]


def overrides(argv: list[str]) -> dict[str, str]:
    """Every ``-c key=value`` pair, as a dict."""
    out: dict[str, str] = {}
    for i, token in enumerate(argv):
        if token == "-c":
            key, _, value = argv[i + 1].partition("=")
            out[key] = value
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. the shape of the command
# ─────────────────────────────────────────────────────────────────────────────


def test_the_default_argv_is_the_documented_non_interactive_invocation() -> None:
    """``codex exec --json`` and nothing else. Every other flag is opt-in, so a
    bare cognition inherits the operator's own configuration — which is the
    right default for a CLI they installed and configured themselves."""
    assert argv_of(CodexCliCognition()) == [
        "codex",
        "exec",
        "--json",
        "--color",
        "never",
        "do the thing",
    ]


def test_color_is_always_disabled() -> None:
    """Not a knob. The CLI writes human progress to stderr and this cognition
    surfaces that stderr verbatim in ``evals["stderr"]`` on a failure; ANSI
    escapes in a stored diagnostic are noise in every reader that is not a
    terminal."""
    assert value_after(argv_of(CodexCliCognition()), "--color") == "never"


def test_the_prompt_is_the_last_argument() -> None:
    """Positional, and last, so the subcommand's own positionals (a session id)
    cannot be confused with it."""
    assert argv_of(CodexCliCognition(model="gpt-5-codex"))[-1] == "do the thing"


def test_a_prompt_that_starts_with_a_dash_is_protected_by_a_separator() -> None:
    """``"--force a rewrite"`` is a legitimate task and would otherwise be
    parsed as flags, dying on an unknown argument with a message that names
    neither the prompt nor the cause."""
    argv = CodexCliCognition()._build_argv("--force a rewrite")
    assert argv[-2:] == ["--", "--force a rewrite"]


def test_an_ordinary_prompt_gets_no_separator() -> None:
    """The separator is only for the hazard. Emitting it always would be
    harmless today and is exactly the kind of unexplained token that gets
    copied into a bug report."""
    assert "--" not in argv_of(CodexCliCognition())


# ─────────────────────────────────────────────────────────────────────────────
# 2. model, workspace, containment
# ─────────────────────────────────────────────────────────────────────────────


def test_the_model_and_workspace_render(tmp_path: Path) -> None:
    argv = argv_of(CodexCliCognition(model="gpt-5-codex", working_dir=tmp_path))
    assert value_after(argv, "--model") == "gpt-5-codex"
    assert value_after(argv, "--cd") == str(tmp_path)


@pytest.mark.asyncio
async def test_the_working_dir_is_both_cd_and_the_subprocess_cwd(tmp_path: Path) -> None:
    """Both, deliberately. ``--cd`` is what Codex treats as the workspace root
    (it is what the sandbox is scoped to); the process ``cwd`` is what makes a
    relative path in the task text mean the same thing. Setting only one leaves
    the other pointing at wherever the service happens to run."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    await drive(CodexCliCognition(working_dir=tmp_path, spawn=cli))
    invocation = cli.invocations[-1]
    assert invocation.cwd == str(tmp_path)
    assert value_after(list(invocation.argv), "--cd") == str(tmp_path)


def test_the_sandbox_and_approval_flags_render() -> None:
    argv = argv_of(CodexCliCognition(sandbox="workspace-write", ask_for_approval="never"))
    assert value_after(argv, "--sandbox") == "workspace-write"
    assert value_after(argv, "--ask-for-approval") == "never"


def test_bypassing_the_sandbox_renders_one_flag_and_not_the_others() -> None:
    """The bypass turns off BOTH the sandbox and the prompt, so emitting a
    ``--sandbox`` alongside it would state a policy that is not in force."""
    argv = argv_of(CodexCliCognition(bypass_sandbox=True))
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert "--sandbox" not in argv and "--ask-for-approval" not in argv


def test_added_directories_each_get_their_own_flag(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    argv = argv_of(CodexCliCognition(add_dirs=(a, b)))
    assert argv.count("--add-dir") == 2
    assert str(a) in argv and str(b) in argv


def test_web_search_is_a_flag_not_a_config_key() -> None:
    assert "--search" in argv_of(CodexCliCognition(web_search=True))


def test_network_access_is_a_config_key_because_it_has_no_flag() -> None:
    """``sandbox_workspace_write.network_access`` is the only way to turn the
    workspace sandbox's network back on, and rendering it as TOML ``true`` (not
    ``"true"``) is what makes the CLI read it as a boolean."""
    argv = argv_of(CodexCliCognition(sandbox="workspace-write", network_access=True))
    assert overrides(argv)["sandbox_workspace_write.network_access"] == "true"


# ─────────────────────────────────────────────────────────────────────────────
# 3. reproducibility and the rest
# ─────────────────────────────────────────────────────────────────────────────


def test_every_reproducibility_flag_renders() -> None:
    argv = argv_of(
        CodexCliCognition(
            ignore_user_config=True,
            ignore_rules=True,
            strict_config=True,
            skip_git_repo_check=True,
            ephemeral=True,
            oss=True,
            profile="ci",
        )
    )
    for flag in (
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--ephemeral",
        "--oss",
    ):
        assert flag in argv, flag
    assert value_after(argv, "--profile") == "ci"


def test_none_of_them_render_by_default() -> None:
    """A default that changes what a run can see is not a default this
    cognition gets to flip under anyone."""
    argv = argv_of(CodexCliCognition())
    for flag in (
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--ephemeral",
        "--oss",
        "--profile",
        "--search",
        "--sandbox",
        "--ask-for-approval",
        "--model",
        "--cd",
        "-c",
    ):
        assert flag not in argv, flag


def test_images_each_get_their_own_flag(tmp_path: Path) -> None:
    one, two = tmp_path / "1.png", tmp_path / "2.png"
    one.write_bytes(b"x")
    two.write_bytes(b"y")
    argv = argv_of(CodexCliCognition(images=(one, two)))
    assert argv.count("--image") == 2


def test_the_reasoning_effort_is_a_config_key() -> None:
    """Not a top-level flag on ``codex`` the way ``--effort`` is on ``claude``.
    Passing it as the config key it is keeps this working on CLI versions that
    have not promoted it."""
    assert overrides(argv_of(CodexCliCognition(effort="xhigh")))["model_reasoning_effort"] == '"xhigh"'


def test_extra_args_are_appended_before_the_prompt() -> None:
    """The escape hatch for a flag this class does not know yet. It has to land
    among the OPTIONS — after it, and it would be read as another positional."""
    argv = argv_of(CodexCliCognition(extra_args=("--future-flag", "7")))
    assert argv[-3:] == ["--future-flag", "7", "do the thing"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. config overrides and MCP servers
# ─────────────────────────────────────────────────────────────────────────────


def test_a_string_override_is_quoted_and_a_bool_is_not() -> None:
    """The value is parsed as TOML, so the quoting IS the type. An unquoted
    string is a parse error and a quoted boolean is the string ``"true"``."""
    got = overrides(argv_of(CodexCliCognition(config_overrides={"a": "text", "b": True, "c": 7})))
    assert got == {"a": '"text"', "b": "true", "c": "7"}


def test_a_list_override_renders_as_a_toml_array() -> None:
    got = overrides(argv_of(CodexCliCognition(config_overrides={"k": ["x", "y"]})))
    assert got["k"] == '["x","y"]'


def test_a_windows_style_path_in_an_override_is_escaped() -> None:
    """``json.dumps`` rather than manual quoting, so a backslash cannot end the
    string early and turn the rest of the value into garbage TOML."""
    got = overrides(argv_of(CodexCliCognition(config_overrides={"p": r"C:\tmp\x"})))
    assert json.loads(got["p"]) == r"C:\tmp\x"


def test_a_none_override_is_refused_rather_than_rendered() -> None:
    """TOML has no null, so ``null`` would reach the CLI as the four-character
    string or as a parse error depending on the key — both a setting that is not
    what the caller wrote."""
    with pytest.raises(ValueError, match="cannot be None"):
        argv_of(CodexCliCognition(config_overrides={"k": None}))


def test_an_mcp_server_is_flattened_into_dotted_config_keys() -> None:
    argv = argv_of(
        CodexCliCognition(
            mcp_servers={
                "engine": {"command": "/usr/bin/python", "args": ["-m", "svc"], "env": {"A": "1"}}
            }
        )
    )
    got = overrides(argv)
    assert got["mcp_servers.engine.command"] == '"/usr/bin/python"'
    assert got["mcp_servers.engine.args"] == '["-m","svc"]'
    assert got["mcp_servers.engine.env"] == '{"A":"1"}'


def test_an_explicit_config_override_wins_over_a_derived_one() -> None:
    """Someone who writes both ``effort="low"`` and the config key has said two
    things; the one they typed as a config key is the more specific, and
    silently preferring the other would make the explicit line look broken."""
    got = overrides(
        argv_of(CodexCliCognition(effort="low", config_overrides={"model_reasoning_effort": "high"}))
    )
    assert got["model_reasoning_effort"] == '"high"'


# ─────────────────────────────────────────────────────────────────────────────
# 5. resume
# ─────────────────────────────────────────────────────────────────────────────


def test_resuming_a_thread_puts_the_subcommand_after_every_option() -> None:
    """THE ordering assertion. ``resume`` and its positionals come last so the
    parent's non-global options are all on the side of the subcommand that
    parses them."""
    cog = CodexCliCognition(model="gpt-5-codex", resume_session_id="abc-123")
    argv = argv_of(cog, resume=cog._resume_target())

    assert argv[-3:] == ["resume", "abc-123", "do the thing"]
    assert argv.index("--model") < argv.index("resume")
    assert argv.index("--json") < argv.index("resume")


def test_continuing_the_latest_thread_uses_the_last_flag() -> None:
    cog = CodexCliCognition(continue_session=True)
    argv = argv_of(cog, resume=cog._resume_target())
    assert argv[-3:] == ["resume", "--last", "do the thing"]


def test_a_fresh_run_names_no_subcommand() -> None:
    assert "resume" not in argv_of(CodexCliCognition())


# ─────────────────────────────────────────────────────────────────────────────
# 6. the environment
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_config_home_is_layered_onto_the_process_environment(tmp_path: Path) -> None:
    """``CODEX_HOME`` holds auth, config AND session history, so it is the whole
    of per-tenant isolation for this CLI — one variable rather than three."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    await drive(CodexCliCognition(config_home=tmp_path, spawn=cli))
    env = cli.invocations[-1].env
    assert env["CODEX_HOME"] == str(tmp_path)
    # Layered, not replaced: the CLI needs PATH to find node, and a fresh
    # environment is how a working install becomes "command not found".
    assert "PATH" in env


@pytest.mark.asyncio
async def test_extra_env_is_layered_and_wins_over_the_parents() -> None:
    """The reason the field exists: Codex reads an HTTP MCP server's bearer
    token from an env var it names, so the token has to reach the child somehow
    and it must not be the argv."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    await drive(CodexCliCognition(env={"AGENTKIT_MCP_TOKEN_ENGINE": "tok", "PATH": "/only/here"}, spawn=cli))
    env = cli.invocations[-1].env
    assert env["AGENTKIT_MCP_TOKEN_ENGINE"] == "tok"
    assert env["PATH"] == "/only/here"


@pytest.mark.asyncio
async def test_no_invented_trace_variable_is_set(tmp_path: Path) -> None:
    """The Claude cognition bridges ``correlation_id`` into
    ``CLAUDE_TRACE_EXTERNAL_ID`` because that variable exists. Codex has no
    counterpart, and a made-up one would read from the outside exactly like a
    working trace bridge while nothing consumed it. The id is on the result
    instead."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert not [k for k in cli.invocations[-1].env if k.startswith("CODEX_TRACE")]
    assert result.evals["external_run_id"] == "fake-run"


# ─────────────────────────────────────────────────────────────────────────────
# 7. the system prompt, which has no flag at all
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prepend_mode_puts_the_agent_prompt_above_the_task() -> None:
    """Codex has no ``--append-system-prompt``. Prepending keeps the CLI's own
    base instructions — the tool guidance, the sandbox explanation, the
    apply_patch format — which is what makes it a capable coding agent."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="local", prompt="You are terse.", cognition=cog)
    await drive(cog, agent=agent, task="count the files")

    assert cli.invocations[-1].argv[-1] == "You are terse.\n\n---\n\ncount the files"


@pytest.mark.asyncio
async def test_the_separator_is_there_so_a_one_line_prompt_is_not_run_on() -> None:
    """Without a rule between them a one-line prompt and a one-line task read
    to the model as a single instruction, which is how "be terse" ends up being
    treated as part of the question."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    await drive(cog, agent=Agent(name="a", prompt="Be terse.", cognition=cog), task="Why?")
    assert "\n\n---\n\n" in cli.invocations[-1].argv[-1]


@pytest.mark.asyncio
async def test_a_versioned_prompt_is_rendered_not_stringified() -> None:
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    prompt = Prompt(id="reviewer", version="1", template="Act as {role}.", inputs=("role",)).bind(
        role="a reviewer"
    )
    agent = Agent(name="a", prompt=prompt, cognition=cog)
    await drive(cog, agent=agent, task="go")
    assert cli.invocations[-1].argv[-1].startswith("Act as a reviewer.")


@pytest.mark.asyncio
async def test_an_agent_with_no_prompt_sends_the_task_alone() -> None:
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    await drive(cog, task="just this")
    assert cli.invocations[-1].argv[-1] == "just this"


@pytest.mark.asyncio
async def test_replace_mode_writes_an_instructions_file_and_sends_the_task_alone() -> None:
    """``experimental_instructions_file`` REPLACES the base instructions, the
    way ``claude --system-prompt`` does. The prompt therefore leaves the user
    message entirely — if it stayed there too the model would see it twice."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(system_prompt_mode="replace", spawn=cli)
    agent = Agent(name="a", prompt="You are a linter and nothing else.", cognition=cog)
    await drive(cog, agent=agent, task="check main.py")

    argv = list(cli.invocations[-1].argv)
    assert argv[-1] == "check main.py"
    assert "experimental_instructions_file" in overrides(argv)


@pytest.mark.asyncio
async def test_replace_mode_with_no_prompt_writes_no_file() -> None:
    """An empty instructions file would replace Codex's base instructions with
    nothing — a strictly worse agent than the caller started with, and it would
    look like the mode had no effect."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(system_prompt_mode="replace", spawn=cli)
    await drive(cog)
    assert "experimental_instructions_file" not in overrides(list(cli.invocations[-1].argv))


@pytest.mark.asyncio
async def test_the_instructions_file_is_removed_when_the_run_ends() -> None:
    """It lives in a 0700 directory for the life of the spawn. A caller never
    sees it and must never be asked to clean one up."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(system_prompt_mode="replace", spawn=cli)
    agent = Agent(name="a", prompt="be terse", cognition=cog)
    await drive(cog, agent=agent)

    path = Path(overrides(list(cli.invocations[-1].argv))["experimental_instructions_file"].strip('"'))
    assert not _exists(path)
    assert not _exists(path.parent)


# ─────────────────────────────────────────────────────────────────────────────
# 8. construction-time refusals
# ─────────────────────────────────────────────────────────────────────────────


def test_two_resume_targets_are_refused() -> None:
    with pytest.raises(ValueError, match="pass one, not both"):
        CodexCliCognition(continue_session=True, resume_session_id="abc")


def test_bypassing_the_sandbox_while_also_naming_one_is_refused() -> None:
    """A contradiction the CLI would resolve silently, leaving a caller with a
    ``sandbox="read-only"`` in their source that is not in force."""
    with pytest.raises(ValueError, match="will not be in force"):
        CodexCliCognition(bypass_sandbox=True, sandbox="read-only")
    with pytest.raises(ValueError, match="will not be in force"):
        CodexCliCognition(bypass_sandbox=True, ask_for_approval="never")


def test_network_access_under_the_wrong_sandbox_is_refused() -> None:
    """Under ``read-only`` it does nothing; under ``danger-full-access`` the
    network is already open and the flag reads as a restriction that is not
    there. Both are a caller believing they configured something."""
    with pytest.raises(ValueError, match="only applies under"):
        CodexCliCognition(sandbox="read-only", network_access=True)
    with pytest.raises(ValueError, match="only applies under"):
        CodexCliCognition(sandbox="danger-full-access", network_access=True)


def test_network_access_with_no_explicit_sandbox_is_allowed() -> None:
    """``None`` means "whatever the CLI and config.toml decide", which may well
    be workspace-write. Refusing here would refuse a legitimate wiring."""
    assert CodexCliCognition(network_access=True).network_access is True


def test_ephemeral_plus_resume_is_refused() -> None:
    """Nothing was written, so there is nothing to resume — and the CLI's own
    error for it names neither flag."""
    with pytest.raises(ValueError, match="nothing for resume"):
        CodexCliCognition(ephemeral=True, continue_session=True)


def test_a_missing_add_dir_is_refused_at_construction() -> None:
    """A typo'd path is otherwise a subprocess that dies three seconds in."""
    with pytest.raises(ValueError, match="not directories"):
        CodexCliCognition(add_dirs=("/tmp/definitely-not-a-dir-XYZ",))


def test_a_missing_image_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="not files"):
        CodexCliCognition(images=("/tmp/definitely-not-a-file-XYZ.png",))


def test_an_mcp_server_with_neither_command_nor_url_is_refused() -> None:
    """The CLI has no way to start or reach it, and the failure is a server
    that silently never appears in the session."""
    with pytest.raises(ValueError, match="neither 'command'"):
        CodexCliCognition(mcp_servers={"engine": {"args": ["-m", "svc"]}})


def test_an_empty_mcp_server_table_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty mapping"):
        CodexCliCognition(mcp_servers={"engine": {}})


# ─────────────────────────────────────────────────────────────────────────────
# 9. against the real binary
# ─────────────────────────────────────────────────────────────────────────────


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_real_cli_accepts_the_argv_we_build(tmp_path: Path) -> None:
    """The only test that can prove the ordering. Everything above asserts what
    we MEANT to send; this one asserts the binary parses it — the flags before
    the subcommand, the ``-c`` overrides, the schema path, the lot.

    A real call, so it is gated on the binary and kept to one cheap task in an
    empty sandbox. What it must NOT produce is ``spawn_failed`` or an exit code
    from clap, which is what a misordered argv looks like.
    """
    (tmp_path / "hello.txt").write_text("The magic number is 137.\n")
    cog = CodexCliCognition(
        working_dir=tmp_path,
        sandbox="read-only",
        ask_for_approval="never",
        skip_git_repo_check=True,
        ignore_user_config=True,
        ignore_rules=True,
        strict_config=True,
        effort="low",
        config_overrides={"model_reasoning_summary": "none"},
    )
    result = final_of(await drive(cog, task="Reply with exactly: OK", ctx=FakeCtx()))

    assert result.evals.get("stop_reason") != "spawn_failed", result.evals
    assert result.evals["cli_return_code"] == 0, result.evals
    assert result.evals["session_id"], "no thread id came back"


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_real_cli_reads_a_file_in_its_sandbox(tmp_path: Path) -> None:
    """One end-to-end run that actually uses a tool: the answer has to contain
    something only the sandbox could have told it."""
    (tmp_path / "hello.txt").write_text("The magic number is 137.\n")
    cog = CodexCliCognition(
        working_dir=tmp_path,
        sandbox="read-only",
        ask_for_approval="never",
        skip_git_repo_check=True,
    )
    result = final_of(await drive(cog, task="What is the magic number in hello.txt? Answer with digits only."))

    assert "137" in result.output, result.output
    assert result.partial is False
    assert result.usage.total_tokens > 0
