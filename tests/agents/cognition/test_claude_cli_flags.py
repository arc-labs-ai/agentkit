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
import os
import shutil
import subprocess
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
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

    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)  # noqa: S603

    assert "unknown option" not in proc.stderr.lower(), proc.stderr
    assert "error: --" not in proc.stderr.lower(), proc.stderr
    assert proc.returncode == 0, f"exit={proc.returncode} stderr={proc.stderr[:400]}"
