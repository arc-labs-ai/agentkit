"""Does the installed binary still accept the argv we build?

WHY THIS FILE EXISTS
--------------------
Three shipped bugs in this codebase were the same bug: a CLI moved a flag, and
nothing noticed until a run died in production with a usage message.

    codex exec --ask-for-approval   removed      → exit 2, no thread started
    codex exec --search             moved to the parent command → exit 2
    codex task_complete             legacy end-of-turn vocabulary the current
                                    parser stopped recognising

Each was found by accident — one because an unrelated real-CLI test happened to
set the field, one while auditing argv for secrets, one because a parity test
existed. None of them was found by the thing that should find them.

WHY NOT A HAND-MAINTAINED MATRIX
--------------------------------
The obvious design — a table of flag → minimum version — rots faster than the
CLIs move, and it is wrong in both directions. Measured against
claude 2.1.236: ``--max-turns`` and ``--permission-prompt-tool`` are ABSENT
from ``--help`` and work perfectly. A preflight that rejected flags missing
from the help text would have broken two working configurations while still
missing ``--search``, which IS documented — on a different command.

So the only ground truth is the binary itself. These tests build argv for a
matrix of representative configurations and ask the real CLI to parse it. A
usage error is the failure; anything else is not this file's business.

WHAT COUNTS AS A FAILURE
------------------------
Only argument REJECTION. A run that starts and then fails for its own reasons —
no credentials, a sandbox denial, an unreachable MCP server — has already
proved the point: the flags parsed. Asserting more would make this file fail
for reasons that have nothing to do with version drift, which is how a canary
gets deleted.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from agentkit.agents.cognition import ClaudeCliCognition, CodexCliCognition

real_claude = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)
real_codex = pytest.mark.skipif(
    shutil.which("codex") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="codex CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)

# Substrings every argument parser in use here prints when it refuses a flag.
# clap (codex) and commander (claude) word it differently; both are unambiguous.
_USAGE_ERRORS = (
    "unexpected argument",
    "unknown option",
    "unrecognized option",
    "invalid value for",
    "error: the following required arguments",
    "usage:",
)


async def _parses(argv: list[str], stdin: bytes) -> tuple[bool, str]:
    """Spawn ``argv`` and report whether the binary ACCEPTED its arguments.

    The prompt is empty and stdin is closed immediately, so a CLI that gets
    past parsing has nothing to do and exits — no model call, no spend. That
    is the whole trick: argument parsing happens before any work, so it can be
    tested without paying for the work.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    assert proc.stdin is not None
    proc.stdin.write(stdin)
    await proc.stdin.drain()
    proc.stdin.close()
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=60)
    except TimeoutError:
        # Still running after a minute means it got well past parsing.
        proc.kill()
        await proc.wait()
        return True, ""
    text = err.decode("utf-8", errors="replace").lower()
    bad = [marker for marker in _USAGE_ERRORS if marker in text]
    return (not bad), text[:400]


def _claude_matrix(tmp: Path) -> dict[str, ClaudeCliCognition]:
    """Configurations a service actually ships, each exercising a different
    corner of the argv builder."""
    (tmp / "mcp.json").write_text('{"mcpServers":{}}')
    (tmp / "settings.json").write_text("{}")
    (tmp / "extra").mkdir(exist_ok=True)
    return {
        "minimal": ClaudeCliCognition(),
        "restricted_tools": ClaudeCliCognition(
            tools=("Read", "Grep"), allowed_tools=("Read",), disallowed_tools=("Bash",)
        ),
        "no_tools": ClaudeCliCognition(tools=("",)),
        "permissions": ClaudeCliCognition(
            permission_mode="acceptEdits", permission_prompt_tool="mcp__x__approve", max_turns=3
        ),
        "service_flags": ClaudeCliCognition(
            bare=True,
            stable_prompt_prefix=True,
            no_session_persistence=True,
            settings=str(tmp / "settings.json"),
        ),
        "mcp": ClaudeCliCognition(
            mcp_config=(str(tmp / "mcp.json"),), strict_mcp_config=True, tools=("",)
        ),
        "model_selection": ClaudeCliCognition(
            model="claude-opus-4-5", fallback_model=("claude-sonnet-4-5",), effort="low"
        ),
        "streaming": ClaudeCliCognition(partial_messages=True, add_dirs=(tmp / "extra",)),
        "session_identity": ClaudeCliCognition(session_id=str(uuid.uuid4())),
        "agents": ClaudeCliCognition(agents={"reviewer": {"description": "d", "prompt": "p"}}),
    }


def _codex_matrix(tmp: Path) -> dict[str, CodexCliCognition]:
    (tmp / "extra").mkdir(exist_ok=True)
    return {
        "minimal": CodexCliCognition(skip_git_repo_check=True),
        "sandboxed": CodexCliCognition(
            sandbox="read-only", ask_for_approval="never", skip_git_repo_check=True
        ),
        "workspace_write": CodexCliCognition(
            sandbox="workspace-write", network_access=True, skip_git_repo_check=True
        ),
        # The flag that moved to the parent command and killed every run.
        "web_search": CodexCliCognition(web_search=True, skip_git_repo_check=True),
        "reproducible": CodexCliCognition(
            ignore_user_config=True,
            ignore_rules=True,
            strict_config=True,
            skip_git_repo_check=True,
        ),
        "model_selection": CodexCliCognition(
            model="gpt-5-codex", effort="low", skip_git_repo_check=True
        ),
        "add_dirs": CodexCliCognition(
            add_dirs=(tmp / "extra",), sandbox="workspace-write", skip_git_repo_check=True
        ),
        "mcp": CodexCliCognition(
            mcp_servers={"svc": {"command": "true", "args": []}}, skip_git_repo_check=True
        ),
        "ephemeral": CodexCliCognition(ephemeral=True, skip_git_repo_check=True),
    }


@real_claude
@pytest.mark.parametrize("name", sorted(_claude_matrix(Path("/tmp"))))
def test_the_installed_claude_accepts_every_configuration(name: str, tmp_path: Path) -> None:
    cog = _claude_matrix(tmp_path)[name]
    argv = cog._build_argv("", system_prompt="You are terse.", stream_input=True)
    ok, err = asyncio.run(_parses(argv, b""))
    assert ok, f"claude rejected the argv for {name!r}:\n  {' '.join(argv)}\n\n{err}"


@real_codex
@pytest.mark.parametrize("name", sorted(_codex_matrix(Path("/tmp"))))
def test_the_installed_codex_accepts_every_configuration(name: str, tmp_path: Path) -> None:
    cog = _codex_matrix(tmp_path)[name]
    argv = cog._build_argv("", stream_input=True)
    ok, err = asyncio.run(_parses(argv, b""))
    assert ok, f"codex rejected the argv for {name!r}:\n  {' '.join(argv)}\n\n{err}"


@real_claude
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"json_schema": {"type": "object", "properties": {}}}, id="json-schema"),
        pytest.param({"max_budget_usd": "1.500000"}, id="max-budget-usd"),
    ],
)
def test_claude_accepts_the_per_run_flags(kwargs: dict[str, Any], tmp_path: Path) -> None:
    """Flags that come from a RUN rather than from construction, so no matrix
    entry carries them."""
    argv = ClaudeCliCognition()._build_argv("", system_prompt="", stream_input=True, **kwargs)
    ok, err = asyncio.run(_parses(argv, b""))
    assert ok, f"claude rejected {kwargs}:\n  {' '.join(argv)}\n\n{err}"


@real_codex
def test_codex_accepts_the_output_schema_path(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object","properties":{},"additionalProperties":false}')
    argv = CodexCliCognition(skip_git_repo_check=True)._build_argv(
        "", output_schema_path=schema, stream_input=True
    )
    ok, err = asyncio.run(_parses(argv, b""))
    assert ok, f"codex rejected --output-schema:\n  {' '.join(argv)}\n\n{err}"


@real_codex
def test_codex_accepts_a_resume_invocation(tmp_path: Path) -> None:
    """``resume`` is a sub-subcommand and the flags have to land on the right
    side of it — the ordering ``_build_argv`` documents at length."""
    cog = CodexCliCognition(resume_session_id=str(uuid.uuid4()), skip_git_repo_check=True)
    argv = cog._build_argv("", resume=cog._resume_target(), stream_input=True)
    ok, err = asyncio.run(_parses(argv, b""))
    assert ok, f"codex rejected a resume argv:\n  {' '.join(argv)}\n\n{err}"
