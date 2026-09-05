"""``ClaudeCliCognition`` argv: the flags mean what the CLI says they mean.

Three mappings were wrong, each in a way that produced a working-looking run:

* ``agent.prompt`` went to ``--system-prompt``, which **replaces the entire
  Claude Code system prompt** — tool guidance, environment info, all of it. An
  agent given a one-line persona silently became a bare chat model that still
  had tools it no longer knew how to drive. The CLI docs recommend
  ``--append-system-prompt`` for "add instructions while keeping Claude Code's
  default behavior", and that is now the default.
* ``session_id`` was documented as "resume an existing session" and passed
  ``--session-id``, which per the CLI reference "Use[s] a specific session ID
  for the conversation (must be a valid UUID)" — it NAMES a new session.
  Resuming is ``--resume``. Following the old docstring produced a fresh
  session with no history and no error.
* ``allowed_tools`` reads like a sandbox and is not one: ``--allowed-tools`` is
  an auto-approve list, and every unnamed tool stays available (merely
  prompting). Restricting the session is ``--tools``.

Argv assertions are made against the real spawn call, so they break if a flag
name drifts. One real-CLI test confirms the binary actually accepts the argv
we build — the thing a mock can never tell us.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.agents.cognition.claude_cli import _BARE_CREDENTIAL_ENV
from agentkit.context import WorkingContext
from agentkit.testing.fakes.ctx import FakeCtx
from tests.agents.cognition.test_claude_cli import _FakeProcess, _happy_path_lines

_UUID = "2b1b8f1e-0f2a-4c1e-9a3e-2f4b6c8d0e1f"

real_cli = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


def _argv(cog: ClaudeCliCognition, *, prompt: str | None = None) -> tuple[str, ...]:
    """Drive the cognition against a fake subprocess and return the real argv."""
    proc = _FakeProcess(stdout_lines=_happy_path_lines())
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        agent = Agent(name="local", prompt=prompt, cognition=cog)

        async def _go() -> None:
            async for _ in cog.drive(agent, "do it", FakeCtx(), WorkingContext()):
                pass

        asyncio.run(_go())
    return tuple(spawn.await_args.args)


def _value_after(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _argv_with(
    cog: ClaudeCliCognition, *, prompt: str | None, inspect: Any
) -> tuple[str, ...]:
    """``_argv``, but ``inspect(argv)`` runs while the run is still in flight.

    The scratch directory is deleted when ``drive`` returns, so anything that
    needs to READ a generated file has to look before then."""
    proc = _FakeProcess(stdout_lines=_happy_path_lines())
    seen: dict[str, tuple[str, ...]] = {}
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        agent = Agent(name="local", prompt=prompt, cognition=cog)

        async def _go() -> None:
            async for _ in cog.drive(agent, "do it", FakeCtx(), WorkingContext()):
                if "argv" not in seen and spawn.await_args is not None:
                    seen["argv"] = tuple(spawn.await_args.args)
                    inspect(seen["argv"])

        asyncio.run(_go())
    return seen.get("argv", tuple(spawn.await_args.args))


# ── 1. the system prompt is APPENDED, not substituted ───────────────────────


def test_an_agent_prompt_is_appended_by_default() -> None:
    """THE regression. ``--system-prompt`` would have discarded the CLI's own
    prompt, and with it every instruction that makes its tools usable."""
    argv = _argv(ClaudeCliCognition(), prompt="You are terse.")
    assert _value_after(argv, "--append-system-prompt") == "You are terse."
    assert "--system-prompt" not in argv


def test_replacing_the_system_prompt_is_opt_in() -> None:
    """Still reachable — it is the right flag when you genuinely want the CLI's
    default prompt gone — but it has to be asked for."""
    argv = _argv(ClaudeCliCognition(system_prompt_mode="replace"), prompt="You are terse.")
    assert _value_after(argv, "--system-prompt") == "You are terse."
    assert "--append-system-prompt" not in argv


def test_no_prompt_means_no_flag() -> None:
    """An agent with no prompt must not pass an empty one: an empty
    ``--system-prompt`` would blank the CLI's prompt with nothing in its place."""
    argv = _argv(ClaudeCliCognition(), prompt=None)
    assert "--append-system-prompt" not in argv and "--system-prompt" not in argv


# ── 1b. a system prompt too big for argv goes to a file ─────────────────────
#
# The kernel copies argv AND the environment into the new process image, under
# two caps: darwin allows 1 MiB for the pair COMBINED (measured — one
# 1,000,000-byte argument is fine, twenty 100,000-byte ones are not, so it is a
# shared pot rather than a per-argument limit), and Linux adds a 128 KiB cap on
# any SINGLE argument on top of that.
#
# The system prompt was the last payload still travelling that way, and the one
# most likely to grow: the task is usually a sentence, while the system prompt
# is where a retrieved context or a compiled instruction set ends up. Measured
# through the real cognition: 2,000,000 bytes came back ``spawn_failed`` /
# ``OSError: [Errno 7] Argument list too long`` — thrown before the binary ran,
# so with no stderr and an error blaming "the argument list" rather than naming
# which of the four things in it was too big.


_BIG = "You are terse. " * 4000  # 60,000 bytes — over the 32 KiB limit


def _scratch_path(argv: tuple[str, ...], flag: str) -> Path:
    assert flag in argv, f"{flag} not in argv: {argv}"
    return Path(_value_after(argv, flag))


def test_an_ordinary_system_prompt_still_travels_inline() -> None:
    """The common case must be untouched.

    Always writing a file would make every run depend on a writable temp dir to
    do something that today needs no filesystem at all — trading a bug that
    bites large prompts for a failure mode that could bite every locked-down
    sandbox."""
    argv = _argv(ClaudeCliCognition(), prompt="You are terse.")
    assert _value_after(argv, "--append-system-prompt") == "You are terse."
    assert "--append-system-prompt-file" not in argv


def test_a_system_prompt_too_big_for_argv_is_passed_as_a_file() -> None:
    holder: dict[str, str] = {}

    def _capture(argv: tuple[str, ...]) -> None:
        holder["path"] = _value_after(argv, "--append-system-prompt-file")
        holder["content"] = Path(holder["path"]).read_text()
        holder["mode"] = oct(os.stat(holder["path"]).st_mode)[-3:]

    argv = _argv_with(ClaudeCliCognition(), prompt=_BIG, inspect=_capture)

    assert "--append-system-prompt" not in argv, "the inline flag must not also be sent"
    assert not any(_BIG[:200] in a for a in argv), "the prompt itself is still in argv"
    assert holder["content"] == _BIG, "the file must carry the prompt VERBATIM"
    # 0600: a system prompt is not a credential, but it is the caller's
    # proprietary instructions and there is no reason to widen it.
    assert holder["mode"] == "600", holder["mode"]


def test_the_file_transport_tracks_system_prompt_mode() -> None:
    """The trap in this change.

    ``--system-prompt-file`` REPLACES the CLI's own prompt; the ``--append-``
    form adds to it. Swapping the pair turns one into the other silently — the
    run still succeeds, the argv still looks plausible, and every instruction
    that makes the CLI's tools usable is gone."""
    appended = _argv(ClaudeCliCognition(), prompt=_BIG)
    assert "--append-system-prompt-file" in appended
    assert "--system-prompt-file" not in appended

    replaced = _argv(ClaudeCliCognition(system_prompt_mode="replace"), prompt=_BIG)
    assert "--system-prompt-file" in replaced
    assert "--append-system-prompt-file" not in replaced


def test_the_threshold_is_measured_in_bytes_not_characters() -> None:
    """``len(text)`` on a str understates what ``execve`` copies by up to 4x.

    This prompt is comfortably under the limit counted in characters and over
    it counted in UTF-8 bytes, which is the only count the kernel applies."""
    prompt = "\u65e5" * 20_000  # 20,000 chars, 60,000 UTF-8 bytes
    assert len(prompt) < 32 * 1024 < len(prompt.encode())

    argv = _argv(ClaudeCliCognition(), prompt=prompt)
    assert "--append-system-prompt-file" in argv, (
        "counted in characters this looked small enough to inline"
    )


def test_the_prompt_file_is_cleaned_up_with_the_run() -> None:
    seen: dict[str, str] = {}
    _argv_with(
        ClaudeCliCognition(),
        prompt=_BIG,
        inspect=lambda argv: seen.update(path=_value_after(argv, "--append-system-prompt-file")),
    )
    assert not os.path.exists(seen["path"])


@real_cli
def test_the_real_cli_honours_a_system_prompt_it_reads_from_a_file() -> None:
    """Accepting a flag is not obeying it.

    Neither ``--system-prompt-file`` nor ``--append-system-prompt-file``
    appears in ``claude --help`` on 2.1.236, so "exit 0" alone would prove only
    that the CLI tolerated an argument it ignored. This asserts the CONTENT
    reached the model: a 2,000,000-byte prompt — an order of magnitude past
    what argv can carry — whose persona comes back in the answer."""
    persona = "You are a pirate. Always answer in pirate speak, starting with ARRR. "
    prompt = persona + ("Ignore this filler sentence. " * 70_000)
    assert len(prompt.encode()) > 2_000_000

    cog = ClaudeCliCognition(max_turns=1)
    agent = Agent(name="local", prompt=prompt, cognition=cog)

    async def _go() -> str:
        out = ""
        async for ev in cog.drive(agent, "What is 2+2?", FakeCtx(), WorkingContext()):
            if ev.type == "final":
                assert ev.result.stop_reason == "complete", ev.result.evals
                out = ev.result.output or ""
        return out

    answer = asyncio.run(_go())
    assert "ARRR" in answer.upper(), f"the file's persona never reached the model: {answer!r}"


# ── 2. session identity: three distinct things ──────────────────────────────


def test_resuming_uses_the_resume_flag() -> None:
    argv = _argv(ClaudeCliCognition(resume_session_id="abc-123"))
    assert _value_after(argv, "--resume") == "abc-123"
    assert "--session-id" not in argv


def test_naming_a_new_session_uses_session_id() -> None:
    argv = _argv(ClaudeCliCognition(session_id=_UUID))
    assert _value_after(argv, "--session-id") == _UUID
    assert "--resume" not in argv


def test_continue_and_fork() -> None:
    argv = _argv(ClaudeCliCognition(continue_session=True, fork_session=True))
    assert "--continue" in argv and "--fork-session" in argv


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"session_id": "not-a-uuid"}, "must be a valid UUID"),
        ({"session_id": _UUID, "resume_session_id": "x"}, "pass one, not both"),
        ({"continue_session": True, "resume_session_id": "x"}, "pass one, not both"),
        ({"fork_session": True}, "only applies when resuming"),
    ],
)
def test_impossible_session_combinations_are_refused_at_construction(
    kwargs: dict, match: str
) -> None:
    """Each of these is rejected by the CLI too — three seconds later, from
    stderr, after a subprocess spawn. A malformed ``session_id`` in particular
    used to burn a spawn to learn it was malformed."""
    with pytest.raises(ValueError, match=match):
        ClaudeCliCognition(**kwargs)


def test_the_uuid_error_points_at_the_resume_flag() -> None:
    """The overwhelmingly likely cause of a non-UUID ``session_id`` is someone
    following the old docstring and passing a previous run's id back to
    resume it. The error says so."""
    with pytest.raises(ValueError, match="resume_session_id"):
        ClaudeCliCognition(session_id="my-app-run-42")


# ── 3. restricting tools vs auto-approving them ─────────────────────────────


def test_tools_restricts_and_allowed_tools_auto_approves() -> None:
    """Different flags, different meanings. ``--tools`` is variadic on the CLI,
    so each name is its own argv entry."""
    argv = _argv(ClaudeCliCognition(tools=("Read", "Grep"), allowed_tools=("Read",)))
    i = argv.index("--tools")
    assert argv[i + 1 : i + 3] == ("Read", "Grep")
    assert _value_after(argv, "--allowed-tools") == "Read"


def test_an_empty_tools_entry_disables_every_tool() -> None:
    """``--tools ""`` is the CLI's documented spelling of "no tools". It has to
    survive as a single empty argv entry rather than being dropped as falsy."""
    argv = _argv(ClaudeCliCognition(tools=("",)))
    i = argv.index("--tools")
    assert argv[i + 1] == ""


def test_an_empty_tools_tuple_is_refused() -> None:
    """``--tools`` is variadic and needs a value, so ``()`` would emit a bare
    flag the CLI rejects — and it is ambiguous anyway between "leave the
    default set alone" and "no tools at all", which are opposite intents."""
    with pytest.raises(ValueError, match="ambiguous"):
        ClaudeCliCognition(tools=())


def test_omitting_tools_passes_no_flag() -> None:
    """``None`` means "leave the CLI's default tool set alone" — distinct from
    ``()`` -style emptiness, which is why the field is ``None``-defaulted."""
    assert "--tools" not in _argv(ClaudeCliCognition())


def test_permission_prompt_tool_is_passed_through() -> None:
    argv = _argv(ClaudeCliCognition(permission_prompt_tool="mcp__approvals__ask"))
    assert _value_after(argv, "--permission-prompt-tool") == "mcp__approvals__ask"


# ── 4. the binary accepts what we build ─────────────────────────────────────


@real_cli
def test_the_real_cli_accepts_the_argv_we_build() -> None:
    """A mock cannot tell us a flag name is wrong — only the binary can. This
    spawns the real CLI with the full flag set and asserts it does not reject
    the arguments.

    Tools are disabled and the task is one word, so the call is trivially cheap.
    """
    cog = ClaudeCliCognition(
        model="claude-haiku-4-5-20251001",
        tools=("",),
        permission_mode="dontAsk",
        max_turns=1,
    )
    argv = cog._build_argv("Reply with the single word: OK", system_prompt="Be terse.")

    # ``stdin=DEVNULL`` matters: ``claude -p`` reads stdin and waits ~3s for
    # data before giving up. The cognition passes DEVNULL for the same reason,
    # so this test spawns it the way production does.
    proc = subprocess.run(  # noqa: S603
        argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL
    )

    assert "unknown option" not in proc.stderr.lower(), proc.stderr
    assert "error: --" not in proc.stderr.lower(), proc.stderr
    assert proc.returncode == 0, f"exit={proc.returncode} stderr={proc.stderr[:400]}"


# ── 5. the flags a service needs ────────────────────────────────────────────
#
# Everything below was reachable only through ``extra_args`` — i.e. by
# hand-writing CLI syntax inside application code, with no validation and no
# way for the cognition to know what was set.


def test_bare_mode_is_passed_and_is_opt_in(monkeypatch) -> None:
    """``--bare`` skips auto-discovery of hooks, plugins, MCP servers, auto
    memory and CLAUDE.md. It is what makes a run reproducible across machines —
    without it, a hook in a teammate's ``~/.claude`` or an MCP server in the
    checked-out repo executes inside your service. Opt-in, because turning it
    on changes what a run can see.

    The key is set so this test is about the FLAG; the credential interaction
    below is its own test."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert "--bare" in _argv(ClaudeCliCognition(bare=True))
    assert "--bare" not in _argv(ClaudeCliCognition())


def test_bare_mode_without_a_credential_warns(monkeypatch) -> None:
    """Bare mode never reads OAuth credentials or the system keychain, so a
    ``claude`` that works perfectly in a terminal fails here with an auth error
    ("Not logged in · Please run /login") pointing at exactly the wrong fix.
    Warned, not refused: the credential may arrive through an ``apiKeyHelper``
    in ``settings`` or a provider mechanism this list does not know.

    The names are cleared from ``_BARE_CREDENTIAL_ENV`` itself rather than from
    a copy of it pasted here. A copy is a silent order-dependence: add a
    credential variable to the tuple, leave one exported in the ambient
    environment (or let another test leak one), and the warning is correctly
    suppressed while this test reports ``DID NOT WARN`` — a failure about the
    wrong thing, and one that only appears in a full run.
    """
    assert _BARE_CREDENTIAL_ENV, "nothing to clear — the check would pass vacuously"
    for name in _BARE_CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)

    with pytest.warns(UserWarning, match="never reads OAuth"):
        ClaudeCliCognition(bare=True)._build_env()

    # A settings blob may carry an apiKeyHelper — no warning in that case.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ClaudeCliCognition(bare=True, settings='{"apiKeyHelper":"..."}')._build_env()

    # And no warning at all when bare mode is off: OAuth works there.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ClaudeCliCognition()._build_env()


def test_the_prompt_prefix_can_be_made_machine_independent() -> None:
    """``--exclude-dynamic-system-prompt-sections`` moves cwd / environment /
    memory paths into the first user message so the cache-stable prefix is
    identical across users and machines. The CLI docs name this workload
    exactly: "Use with -p for scripted, multi-user workloads"."""
    argv = _argv(ClaudeCliCognition(stable_prompt_prefix=True))
    assert "--exclude-dynamic-system-prompt-sections" in argv


def test_it_is_refused_when_the_prompt_is_replaced_wholesale() -> None:
    """The CLI only moves the dynamic sections out of ITS OWN default prompt,
    which ``--system-prompt`` discards anyway — so the combination is a silent
    no-op, and a silent no-op in a caching optimisation is worse than an
    error."""
    with pytest.raises(ValueError, match="no effect"):
        ClaudeCliCognition(stable_prompt_prefix=True, system_prompt_mode="replace")


def test_a_fallback_chain_is_comma_joined() -> None:
    """``--fallback-model`` accepts a comma-separated list tried in order."""
    argv = _argv(ClaudeCliCognition(fallback_model=("sonnet", "haiku")))
    assert _value_after(argv, "--fallback-model") == "sonnet,haiku"
    assert _value_after(_argv(ClaudeCliCognition(fallback_model="sonnet")), "--fallback-model") == (
        "sonnet"
    )


def test_mcp_servers_are_variadic_and_strictness_needs_them() -> None:
    """``--mcp-config`` takes several entries. A PATH is passed through; an
    inline JSON blob is written to a 0600 file and replaced by its path,
    because an argument list is world-readable and these blobs routinely carry
    bearer tokens — see ``_materialise_secrets``.

    ``--strict-mcp-config`` on its own would leave the session with no MCP
    servers at all, which is not what anyone means by "strict"."""
    argv = _argv(
        ClaudeCliCognition(mcp_config=("/tmp/a.json", '{"b":1}'), strict_mcp_config=True)
    )
    i = argv.index("--mcp-config")
    passed_through, materialised = argv[i + 1 : i + 3]
    assert passed_through == "/tmp/a.json", "a path is already a reference, not a value"
    assert materialised != '{"b":1}', "the inline blob must not reach argv"
    assert materialised.endswith(".json")
    assert "--strict-mcp-config" in argv

    with pytest.raises(ValueError, match="only means something alongside"):
        ClaudeCliCognition(strict_mcp_config=True)


def test_additional_directories_are_checked_at_construction(tmp_path) -> None:
    """The CLI validates each path exists as a directory. Doing it here turns a
    subprocess that dies three seconds in into an error at wiring time."""
    argv = _argv(ClaudeCliCognition(add_dirs=(tmp_path,)))
    assert _value_after(argv, "--add-dir") == str(tmp_path)

    with pytest.raises(ValueError, match="not directories"):
        ClaudeCliCognition(add_dirs=(tmp_path / "nope",))


def test_settings_agents_effort_and_session_persistence() -> None:
    """The remaining service-shaped knobs. ``agents`` is serialised for the
    caller — hand-writing JSON into an argv string is how quoting bugs get
    into production."""
    cog = ClaudeCliCognition(
        settings='{"x":1}',
        agents={"reviewer": {"description": "d", "prompt": "p"}},
        effort="low",
        no_session_persistence=True,
    )
    argv = _argv(cog)
    # Inline settings JSON is materialised to a 0600 file, not passed as an
    # argument: ``settings`` can carry an ``apiKeyHelper`` — a credential — and
    # an argument list is readable by every local account.
    settings_arg = _value_after(argv, "--settings")
    assert settings_arg != '{"x":1}', "the inline blob must not reach argv"
    assert settings_arg.endswith(".json")
    # ...and the file is gone once the run is over. A secret written to disk to
    # keep it out of argv would be a poor trade if it outlived the process.
    # Its CONTENT is asserted in the contract suite, which can look while the
    # spawn is still in flight.
    assert not Path(settings_arg).exists()
    assert json.loads(_value_after(argv, "--agents")) == {
        "reviewer": {"description": "d", "prompt": "p"}
    }
    assert _value_after(argv, "--effort") == "low"
    assert "--no-session-persistence" in argv


@real_cli
def test_the_real_cli_accepts_the_service_flag_set() -> None:
    """Same reasoning as the argv test above, for the flags a service actually
    ships with: bare mode, a stable prompt prefix, a fallback chain and no
    session persistence.

    Bare mode has a documented catch — it never reads OAuth credentials or the
    keychain — so on a developer machine authenticated with ``claude login``
    this run fails with an AUTH error rather than a flag error. That
    distinction is the assertion: the flags parsed, the credential did not
    exist. With ``ANTHROPIC_API_KEY`` set (CI), it must exit 0.
    """
    cog = ClaudeCliCognition(
        model="claude-haiku-4-5-20251001",
        tools=("",),
        permission_mode="dontAsk",
        max_turns=1,
        bare=True,
        stable_prompt_prefix=True,
        fallback_model=("claude-haiku-4-5-20251001",),
        no_session_persistence=True,
    )
    argv = cog._build_argv("Reply with the single word: OK", system_prompt="Be terse.")

    proc = subprocess.run(  # noqa: S603
        argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL
    )

    assert "unknown option" not in proc.stderr.lower(), proc.stderr
    if os.environ.get("ANTHROPIC_API_KEY"):
        assert proc.returncode == 0, f"exit={proc.returncode} stderr={proc.stderr[:400]}"
    else:
        # No API key: bare mode cannot authenticate. The failure must be about
        # the CREDENTIAL, which is what proves the flags themselves parsed.
        assert "/login" in (proc.stdout + proc.stderr).lower(), proc.stdout[:400]
