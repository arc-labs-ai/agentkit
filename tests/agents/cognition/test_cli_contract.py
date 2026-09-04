"""Contracts every CLI cognition must satisfy, run against BOTH of them.

The two adapters are ~2,100 lines each and diverge for real reasons — a held
process versus a resumed thread, a tool allow-list versus a sandbox, inline
JSON versus a schema file. What they must NOT diverge on is the handful of
properties a service depends on to be safe, and every one of these was, at some
point, true of one adapter and not the other.

Parametrised over both rather than written twice on purpose. A copy is how the
Claude path grew a fix the Codex path never got — the module docstring of
``_cli_common`` lists the ones that already happened — and a contract that only
one implementation is held to is a contract in name.

Where a property genuinely cannot be symmetric it gets its own test with the
asymmetry stated: ``native_tool_policy="deny"`` is satisfiable for Claude and
unsatisfiable for Codex, and pretending otherwise would be inventing a
capability Codex does not have.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pydantic
import pytest

from agentkit import Agent, Scope, Usage
from agentkit.agents.cognition import ClaudeCliCognition, CodexCliCognition
from agentkit.agents.cognition._cli_common import (
    _CANCEL_POLL_S,
    START_NEW_SESSION,
    CliLineTooLong,
    CliTimeouts,
    InvalidSchemaError,
    StructuredOutputFailure,
    _build_child_env,
    _CliDeadline,
    _drain_stderr,
    _iter_stdout,
    _terminate,
    _validate_json_schema,
)
from agentkit.agents.control.budget import ActorBudget
from agentkit.context import WorkingContext
from agentkit.runtime import Budget, Quota, RunContext, Services
from agentkit.testing.fakes import FakeClaudeCli, FakeCodexCli, FakeCtx, codex_turn
from tests.agents.cognition.test_codex_cli import CancellingCtx

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ─────────────────────────────────────────────────────────────────────────────
# The two cognitions behind one interface, so a contract can be stated once.
# ─────────────────────────────────────────────────────────────────────────────


def _claude_case() -> tuple[Any, Any, str]:
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": "s",
                "duration_ms": 1,
                "total_cost_usd": 0.0,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ]
    )
    return ClaudeCliCognition(spawn=cli), cli, "claude"


def _codex_case() -> tuple[Any, Any, str]:
    cli = FakeCodexCli.script(codex_turn(text="hi", usage=(1, 0, 1)))
    return CodexCliCognition(spawn=cli), cli, "codex"


CASES = [pytest.param(_claude_case, id="claude"), pytest.param(_codex_case, id="codex")]


async def _pid_alive(pid: int) -> bool:
    """``ps -p`` without blocking the loop — ASYNC221 refuses ``subprocess.run``
    inside an ``async def``, and it is right to: this test's whole subject is a
    process tree, so a blocking call here would be the one place a stall is
    least welcome."""
    proc = await asyncio.create_subprocess_exec(
        "ps", "-p", str(pid), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    return await proc.wait() == 0


async def _drive(cog: Any, task: str = "the task") -> Any:
    agent = Agent(name="local", cognition=cog)
    events = [
        ev
        async for ev in cog.drive(agent, task, FakeCtx(), WorkingContext())
    ]
    return events[-1].result


# ─────────────────────────────────────────────────────────────────────────────
# 1. The prompt travels on stdin, not in argv.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CASES)
async def test_the_task_is_not_in_argv(make: Any) -> None:
    """An argv-carried prompt has three problems and this closes all of them:
    the OS caps it (measured: ``OSError`` E2BIG past ~1 MB on darwin; Linux caps
    a SINGLE argument at 128 KiB, so an evidence package that works on a laptop
    fails on a server), anyone who can run ``ps`` can read the whole task, and
    it was a second transport that had to be kept in step with the session
    path's.
    """
    cog, cli, _ = make()
    task = "SENTINEL-TASK-TEXT"
    await _drive(cog, task)
    argv = cli.invocations[-1].argv
    assert task not in argv, f"the task must not be an argument: {argv}"
    assert not any(task in a for a in argv), f"the task must not be inside an argument: {argv}"


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CASES)
async def test_the_task_arrives_on_stdin(make: Any) -> None:
    """...and it must actually arrive, or the run is a prompt-less CLI waiting
    on a pipe that never closes."""
    cog, cli, kind = make()
    await _drive(cog, "SENTINEL-TASK-TEXT")
    written = cli.invocations[-1].stdin.decode()
    if kind == "claude":
        # One stream-json user turn — the same encoding a session writes.
        assert json.loads(written)["message"]["content"] == "SENTINEL-TASK-TEXT"
    else:
        # Codex reads the instructions raw; ``-`` in argv is the marker.
        assert written == "SENTINEL-TASK-TEXT"
        assert cli.invocations[-1].argv[-1] == "-"


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CASES)
async def test_a_multi_megabyte_task_survives(make: Any) -> None:
    """The case the argv transport could not serve at all.

    4 MB is past ``ARG_MAX`` on every platform this package supports, so before
    the stdin transport this run died in ``create_subprocess_exec`` with
    ``OSError: [Errno 7] Argument list too long`` and reported ``spawn_failed``
    — for a prompt the model would have had no trouble with.
    """
    cog, cli, _ = make()
    task = "x" * (4 * 1024 * 1024)
    result = await _drive(cog, task)
    assert result.partial is False
    assert len(cli.invocations[-1].stdin) >= len(task)


# ─────────────────────────────────────────────────────────────────────────────
# 2. The child is its own process group, and the whole group is reaped.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CASES)
async def test_the_spawn_asks_for_its_own_process_group(make: Any) -> None:
    cog, cli, _ = make()
    await _drive(cog)
    assert cli.invocations[-1].start_new_session is True, (
        "without start_new_session the child shares the SERVICE's process group, "
        "and the group kill in _terminate would signal the service"
    )


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups only")
@pytest.mark.asyncio
async def test_terminate_reaps_the_grandchildren() -> None:
    """A real subprocess, because a double cannot have grandchildren.

    This is the bug the group kill exists for, and it is not theoretical: the
    CLI runs its own tools in its own process, so a cancelled run that had
    started ``npm install`` left that install running — holding a pipe, burning
    the machine — until it finished on its own. ``proc.terminate()`` signals
    exactly one pid.

    Measured before the fix, and the reason BOTH halves are required:

        start_new_session=False -> grandchild SURVIVED
        start_new_session=True  -> grandchild SURVIVED   (group exists, unused)
        killpg(SIGTERM)         -> grandchild reaped
    """
    script = (
        "import subprocess, sys, time\n"
        "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "print(gc.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=START_NEW_SESSION,
    )
    assert proc.stdout is not None
    grandchild = int((await asyncio.wait_for(proc.stdout.readline(), 10)).decode())
    try:
        await _terminate(proc, grace_s=2.0)
        await asyncio.sleep(0.5)
        alive = await _pid_alive(grandchild)
        assert not alive, f"grandchild {grandchild} outlived the CLI it belonged to"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(grandchild, signal.SIGKILL)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Environment policy.
# ─────────────────────────────────────────────────────────────────────────────


def test_inherit_keeps_the_ambient_credential(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient")
    env, removed = _build_child_env(policy="inherit", credential_vars=("ANTHROPIC_API_KEY",))
    assert env["ANTHROPIC_API_KEY"] == "ambient"
    assert removed == ()


def test_profile_removes_the_ambient_credential(monkeypatch: Any) -> None:
    """The measured failure. claude 2.1.236 says it on stderr itself:

        ⚠ claude.ai connectors are disabled because ANTHROPIC_API_KEY or
          another auth source is set and takes precedence over your claude.ai
          login

    So a per-tenant ``config_dir=`` was silently overridden by whatever key the
    machine happened to export — the isolation the caller asked for was exactly
    what did not happen.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient")
    monkeypatch.setenv("PATH", "/usr/bin")
    env, removed = _build_child_env(policy="profile", credential_vars=("ANTHROPIC_API_KEY",))
    assert "ANTHROPIC_API_KEY" not in env
    assert removed == ("ANTHROPIC_API_KEY",)
    assert env["PATH"] == "/usr/bin", "profile must not disturb anything but the credentials"


def test_an_explicit_credential_outranks_the_strip(monkeypatch: Any) -> None:
    """``profile`` + an explicit key is how per-tenant credentials work. A strip
    that outranked an explicit assignment would make the safe policy unusable
    for the exact deployment it was written for."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient")
    env, removed = _build_child_env(
        policy="profile",
        credential_vars=("ANTHROPIC_API_KEY",),
        overrides={"ANTHROPIC_API_KEY": "tenant"},
    )
    assert env["ANTHROPIC_API_KEY"] == "tenant"
    assert removed == (), "an overridden variable was not removed, it was replaced"


def test_isolated_keeps_only_what_a_run_needs(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient")
    monkeypatch.setenv("SOME_UNRELATED_THING", "leaked")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HTTPS_PROXY", "http://corp:3128")
    env, _ = _build_child_env(policy="isolated", credential_vars=("ANTHROPIC_API_KEY",))
    assert "SOME_UNRELATED_THING" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["PATH"] == "/usr/bin"
    # Dropping the proxy would turn "isolated" into "does not work behind a
    # corporate network", which is how a policy gets switched off wholesale
    # instead of being fixed.
    assert env["HTTPS_PROXY"] == "http://corp:3128"


@pytest.mark.parametrize(
    ("cog", "expected"),
    [
        (ClaudeCliCognition(), "inherit"),
        (ClaudeCliCognition(config_dir=Path("/tmp")), "profile"),
        (ClaudeCliCognition(config_dir=Path("/tmp"), env_policy="inherit"), "inherit"),
        (CodexCliCognition(), "inherit"),
        (CodexCliCognition(config_home=Path("/tmp")), "profile"),
        (CodexCliCognition(config_home=Path("/tmp"), env_policy="inherit"), "inherit"),
    ],
)
def test_a_config_directory_implies_the_profile_policy(cog: Any, expected: str) -> None:
    """Pointing the CLI at a configuration directory IS the statement of intent
    to isolate. Inheriting a credential that overrides it is never what that
    caller meant — but an explicit ``env_policy=`` still wins, because a
    default that cannot be turned off is not a default."""
    assert cog.effective_env_policy == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CASES)
async def test_the_policy_reaches_the_spawn(make: Any, monkeypatch: Any) -> None:
    """The unit tests above check the helper; this checks it is actually wired
    into the environment the subprocess gets."""
    cog, cli, kind = make()
    var = "ANTHROPIC_API_KEY" if kind == "claude" else "OPENAI_API_KEY"
    monkeypatch.setenv(var, "ambient")
    object.__setattr__(cog, "env_policy", "profile")
    await _drive(cog)
    assert var not in cli.invocations[-1].env


# ─────────────────────────────────────────────────────────────────────────────
# 4. Native tool policy — the one contract that CANNOT be symmetric.
# ─────────────────────────────────────────────────────────────────────────────


def test_claude_deny_refuses_a_session_that_holds_native_tools() -> None:
    with pytest.raises(ValueError, match="native_tool_policy='deny'"):
        ClaudeCliCognition(native_tool_policy="deny")


def test_claude_deny_accepts_a_session_served_over_mcp() -> None:
    """Satisfiable, and this is the configuration the warning has always been
    steering toward: no native tools, every call back through the Invoker where
    the middleware chain applies."""
    cog = ClaudeCliCognition(native_tool_policy="deny", tools=("",))
    assert cog.native_tools() == ()


def test_codex_deny_cannot_be_satisfied_and_says_so() -> None:
    """Not an oversight — a property of the program.

    Every Codex session has ``shell``; there is no tool allow-list and no
    PreToolUse hook, so there is no configuration in which agentkit's
    middleware governs it. A deny policy means an ungovernable tool must not
    start, and here that is the whole cognition. Failing at construction is the
    honest answer; silently downgrading to a warning would be a policy that
    reports itself as enforced while enforcing nothing.
    """
    with pytest.raises(ValueError, match="cannot be satisfied"):
        CodexCliCognition(native_tool_policy="deny")


@pytest.mark.parametrize(
    "cog_factory",
    [lambda: ClaudeCliCognition(native_tool_policy="allow"), lambda: CodexCliCognition(native_tool_policy="allow")],
)
def test_allow_constructs_for_both(cog_factory: Any) -> None:
    assert cog_factory() is not None


# ─────────────────────────────────────────────────────────────────────────────
# 5. A schema is refused before it costs a subprocess.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param({"type": "object", "required": ["total"], "properties": {"amount": {}}}, id="renamed-field"),
        pytest.param({}, id="empty"),
        pytest.param({"type": 5}, id="type-not-a-string"),
        pytest.param({"properties": []}, id="properties-not-an-object"),
    ],
)
@pytest.mark.parametrize("cls", [ClaudeCliCognition, CodexCliCognition])
def test_a_bad_schema_is_refused_at_construction(cls: Any, schema: Any) -> None:
    """Both binaries reject a bad schema — two to five seconds into a spawn,
    from inside a subprocess, with the complaint on a stderr these cognitions
    surface only on a run that already failed.

    The renamed-field case is the one that actually happens and the one that
    hides best: a field renamed in ``properties`` and not in ``required``
    produces a schema no output can satisfy, so the CLI burns its whole
    structured-output retry budget and returns
    ``error_max_structured_output_retries`` — a failure that reads as the model
    refusing to comply with a schema that was never satisfiable.
    """
    with pytest.raises(InvalidSchemaError):
        cls(json_schema=schema)


def test_a_valid_schema_is_returned_unchanged() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert _validate_json_schema(schema, who="T") == schema


def test_codex_closes_every_object_for_openai_strict_mode() -> None:
    """OpenAI's structured-output mode refuses a schema whose objects are open:

        'additionalProperties' is required to be supplied and to be false.
        (param: text.format.schema, status 400)

    agentkit's two schema adapters disagreed about this — the dataclass one
    emits the key, Pydantic's ``model_json_schema()`` does not — so
    ``output=SomeDataclass`` worked on Codex and ``output=SomePydanticModel``
    failed 100% of the time, with the 400 visible only in ``evals["stderr"]`` of
    a failed run. Verified against codex 0.152.1: same schema, exit 1 without
    the key, exit 0 with it.

    Normalised here rather than in the adapters because it is a PROVIDER
    constraint — ``claude --json-schema`` takes the open form happily.
    """
    from agentkit.agents.cognition.codex_cli import _strict_object_schema

    pydantic_shaped = {
        "type": "object",
        "title": "Invoice",
        "properties": {
            "vendor": {"type": "string"},
            "line": {"type": "object", "properties": {"qty": {"type": "number"}}},
        },
        "required": ["vendor"],
    }
    out = _strict_object_schema(pydantic_shaped)
    assert out["additionalProperties"] is False
    assert out["properties"]["line"]["additionalProperties"] is False, "nested objects too"
    assert pydantic_shaped == {
        "type": "object",
        "title": "Invoice",
        "properties": {
            "vendor": {"type": "string"},
            "line": {"type": "object", "properties": {"qty": {"type": "number"}}},
        },
        "required": ["vendor"],
    }, "the caller's schema must not be mutated"


def test_codex_leaves_an_explicit_additional_properties_alone() -> None:
    """Including a ``True`` the provider will reject. A caller who wrote it
    meant it, and silently inverting a stated instruction is worse than the
    error that names it."""
    from agentkit.agents.cognition.codex_cli import _strict_object_schema

    out = _strict_object_schema({"type": "object", "properties": {}, "additionalProperties": True})
    assert out["additionalProperties"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CASES)
async def test_a_refused_schema_never_spawns(make: Any) -> None:
    """The whole point of moving the check earlier."""
    _, cli, kind = make()
    cls = ClaudeCliCognition if kind == "claude" else CodexCliCognition
    with pytest.raises(InvalidSchemaError):
        cls(spawn=cli, json_schema={"required": ["x"], "properties": {}})
    assert cli.invocations == []


# ─────────────────────────────────────────────────────────────────────────────
# 6. Liveness — CliTimeouts.
#
# The symptom these exist for, measured against claude 2.1.236: an invalid
# ``ANTHROPIC_API_KEY`` produced no stdout, no stderr and no exit for 45+
# seconds. The process was alive and the run looked active. Nothing in the
# stream said otherwise, and an outer ``asyncio.wait_for`` — which does work —
# would have reported the same bare ``TimeoutError`` it reports for a model
# that is merely slow.
# ─────────────────────────────────────────────────────────────────────────────


class _StallingStdout:
    """Emits ``(delay, line)`` pairs, then stalls forever."""

    def __init__(self, script: list[tuple[float, bytes]]) -> None:
        self._script = list(script)

    def __aiter__(self) -> _StallingStdout:
        return self

    async def __anext__(self) -> bytes:
        if not self._script:
            await asyncio.sleep(3600)  # the hang under test
            raise AssertionError("unreachable — the deadline must fire first")
        delay, line = self._script.pop(0)
        await asyncio.sleep(delay)
        return line


class _StallingProcess:
    def __init__(self, script: list[tuple[float, bytes]], stderr: bytes) -> None:
        self.stdout = _StallingStdout(script)
        self.stderr = _Drainable(stderr)
        self.stdin = _NullStdin()
        self.terminated = False
        self.killed = False
        self._returncode: int | None = None

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def wait(self) -> int:
        if self._returncode is None:
            self._returncode = -15
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        if self._returncode is None:
            self._returncode = -15

    def kill(self) -> None:
        self.killed = True
        if self._returncode is None:
            self._returncode = -9


class _Drainable:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self, n: int = -1) -> bytes:
        out, self._data = self._data, b""
        return out


class _NullStdin:
    def write(self, data: bytes) -> None:
        return None

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return False

    def close(self) -> None:
        return None


def _stalling_spawn(script: list[tuple[float, bytes]], stderr: bytes = b"") -> Any:
    proc = _StallingProcess(script, stderr)

    async def spawn(*_a: Any, **_kw: Any) -> Any:
        return proc

    spawn.proc = proc  # type: ignore[attr-defined]
    return spawn


_CLAUDE_INIT = b'{"type":"system","subtype":"init","session_id":"s"}\n'
_CLAUDE_TEXT = b'{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
_CODEX_START = b'{"type":"thread.started","thread_id":"t"}\n'
_CODEX_TEXT = b'{"type":"item.completed","item":{"id":"i","type":"agent_message","text":"hi"}}\n'


def _cog_for(kind: str, spawn: Any, timeouts: CliTimeouts) -> Any:
    cls = ClaudeCliCognition if kind == "claude" else CodexCliCognition
    return cls(spawn=spawn, timeouts=timeouts)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_cli_that_never_speaks_is_a_startup_timeout(kind: str) -> None:
    """The measured hang. Before this, the run sat there — and the terminal
    event, if the caller ever got one, said nothing about why."""
    spawn = _stalling_spawn([])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    result = await _drive(cog)

    assert result.evals["stop_reason"] == "startup_timeout"
    assert result.evals["timeout_s"] == 0.05
    assert result.partial is True
    assert result.stop_reason == "failed", (
        "not 'expired' — that means a human-gate deadline passed and the run "
        "CONTINUED; and not 'terminated' — nobody chose this"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_the_hung_process_is_actually_killed(kind: str) -> None:
    """A bound that reports a hang without ending it has moved the leak rather
    than fixed it — the CLI would keep running, and keep spending."""
    spawn = _stalling_spawn([])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    await _drive(cog)
    assert spawn.proc.terminated, "the timed-out process must be terminated"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_silence_after_the_first_line_is_an_idle_timeout(kind: str) -> None:
    """A different bound and a different fix from ``startup_timeout``: the
    binary came up fine, so the credential is not the problem."""
    first = _CLAUDE_INIT if kind == "claude" else _CODEX_START
    spawn = _stalling_spawn([(0.0, first)])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=5.0, idle=0.05))
    result = await _drive(cog)
    assert result.evals["stop_reason"] == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_cli_that_boots_but_never_answers_is_a_first_event_timeout(kind: str) -> None:
    """The CLI announcing itself is not the CLI answering. ``system/init`` and
    ``thread.started`` are the process talking about itself, so they must not
    satisfy a bound whose whole subject is time-to-first-visible-output."""
    first = _CLAUDE_INIT if kind == "claude" else _CODEX_START
    spawn = _stalling_spawn([(0.0, first)])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=5.0, first_event=0.05))
    result = await _drive(cog)
    assert result.evals["stop_reason"] == "first_event_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_run_that_is_merely_too_long_is_a_total_timeout(kind: str) -> None:
    first, text = (
        (_CLAUDE_INIT, _CLAUDE_TEXT) if kind == "claude" else (_CODEX_START, _CODEX_TEXT)
    )
    spawn = _stalling_spawn([(0.0, first), (0.0, text), (0.0, text)])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=5.0, total=0.05))
    result = await _drive(cog)
    assert result.evals["stop_reason"] == "total_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "stderr"),
    [
        ("claude", b"Invalid API key \xc2\xb7 Please run /login"),
        ("codex", b"Not logged in. Run `codex login` first."),
    ],
)
async def test_a_startup_timeout_that_smells_of_auth_says_so(kind: str, stderr: bytes) -> None:
    """The refinement that points at the actual fix.

    ``startup_timeout`` alone sends an operator looking at the network or the
    model. The credential is the most common cause of this exact symptom, and
    the CLI does say so on stderr — a stream this cognition otherwise surfaces
    only after the fact.
    """
    spawn = _stalling_spawn([], stderr=stderr)
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    result = await _drive(cog)
    assert result.evals["stop_reason"] == "authentication_failed"
    assert result.evals["timeout_kind"] == "startup_timeout", (
        "the underlying bound stays visible — the refinement renames, it does not hide"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_an_ordinary_warning_on_stderr_is_not_an_auth_failure(kind: str) -> None:
    """The trap this heuristic has to avoid.

    ``claude`` prints "ANTHROPIC_API_KEY ... takes precedence over your
    claude.ai login" on runs that WORK. Treating it as an auth marker would
    relabel every unrelated startup hang as a credential problem — the exact
    false diagnosis the refinement exists to prevent.
    """
    spawn = _stalling_spawn(
        [], stderr=b"warning: ANTHROPIC_API_KEY or another auth source takes precedence"
    )
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    result = await _drive(cog)
    assert result.evals["stop_reason"] == "startup_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_exactly_one_terminal_event_survives_a_timeout(kind: str) -> None:
    """The guarantee both cognitions make, on the newest exit path."""
    spawn = _stalling_spawn([])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    agent = Agent(name="local", cognition=cog)
    events = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())]
    assert [e.type for e in events].count("final") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CASES)
async def test_a_normal_run_is_untouched_by_the_default_bounds(make: Any) -> None:
    """``startup`` is on by default, so the common path has to be proved
    unaffected — a default that fires on healthy runs would be worse than the
    hang it prevents."""
    cog, _, _ = make()
    assert cog.timeouts.startup == 120.0
    assert (cog.timeouts.first_event, cog.timeouts.idle, cog.timeouts.total) == (None, None, None)
    result = await _drive(cog)
    assert result.partial is False
    assert "timeout_kind" not in result.evals


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_timeouts_can_be_turned_off_entirely(kind: str) -> None:
    """``CliTimeouts.off()`` is the pre-existing behaviour, by name. A run with
    no bounds and a stalling CLI hangs — which is what the caller asked for, so
    the assertion is that WE do not intervene."""
    spawn = _stalling_spawn([])
    cog = _cog_for(kind, spawn, CliTimeouts.off())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_drive(cog), timeout=0.2)


async def _session_turn(cog: Any, task: str = "t") -> Any:
    """One turn through the SESSION entry point, whatever that means here.

    The two cognitions reach it very differently — Claude holds one process and
    feeds it turns over stdin, Codex re-spawns and resumes a thread by id — and
    that is exactly why the bound has to be asserted through this door as well
    as through ``drive``. It was not, once: every liveness bound lived in the
    shared driver, Claude's session had its own read loop, and ``timeouts=`` on
    a session was configuration that was accepted and silently discarded.
    """
    async with cog.session() as chat:
        events = [ev async for ev in chat.turn(task)]
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1, f"expected exactly one terminal event, got {len(finals)}"
    return finals[0].result


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_session_turn_is_bounded_like_a_drive(kind: str) -> None:
    """Same hang, same typed reason, through the other entry point.

    A session is the LONGER-lived process, so an unbounded hang there does not
    end at the next spawn — it never ends at all."""
    spawn = _stalling_spawn([])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    result = await _session_turn(cog)

    assert result.evals["stop_reason"] == "startup_timeout"
    assert result.evals["timeout_s"] == 0.05
    assert result.stop_reason == "failed" and result.partial is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_hung_session_process_is_actually_killed(kind: str) -> None:
    """Reporting a hang without ending it moves the leak rather than fixing it,
    and a session's process is the one most able to outlive the report."""
    spawn = _stalling_spawn([])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    await _session_turn(cog)
    assert spawn.proc.terminated, "the timed-out session process must be terminated"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_silence_mid_session_turn_is_an_idle_timeout(kind: str) -> None:
    """The bound that catches a stuck tool call, asserted on the path where a
    stuck tool call is most likely — a long interactive conversation."""
    first = _CLAUDE_INIT if kind == "claude" else _CODEX_START
    spawn = _stalling_spawn([(0.0, first)])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=5.0, idle=0.05))
    result = await _session_turn(cog)
    assert result.evals["stop_reason"] == "idle_timeout"


# ── cancellation while the CLI is silent ────────────────────────────────────
#
# ``ctx.check_cancelled()`` is polled inside the read loop, so before the poll
# tick existed a caller only got to look BETWEEN LINES. A CLI that had gone
# quiet — mid ``Bash(npm install)``, mid long test suite, or simply hung —
# could not be stopped until it spoke again. Measured before the fix: against a
# silent process the tripped token was never noticed at all, and against a real
# one it took 83 seconds.
#
# This is not the same mechanism as ``asyncio.CancelledError``, which has
# always worked: that arrives mid-await and unwinds immediately. It is the
# COOPERATIVE token, the one a stop button in a service actually holds.


async def _cancel_after_first_line(cog: Any, *, session: bool, ctx: Any) -> tuple[Any, float]:
    """Trip the token while the CLI is mid-turn and silent; time the response."""
    agent = Agent(name="local", cognition=cog)

    async def run() -> list[Any]:
        if session:
            async with cog.session() as chat:
                return [ev async for ev in chat.turn("t", ctx=ctx)]
        return [ev async for ev in cog.drive(agent, "t", ctx, WorkingContext())]

    task = asyncio.create_task(run())
    await asyncio.sleep(0.2)  # let the first line land; now nothing is coming
    started = time.monotonic()
    ctx.trip()
    try:
        events = await asyncio.wait_for(task, timeout=5.0)
    except TimeoutError:
        task.cancel()
        raise AssertionError("the cancellation token was never noticed") from None
    elapsed = time.monotonic() - started
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1, f"expected one terminal event, got {len(finals)}"
    return finals[0].result, elapsed


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
@pytest.mark.parametrize("session", [False, True], ids=["drive", "session"])
async def test_a_silent_cli_is_still_cancellable(kind: str, session: bool) -> None:
    """The token is honoured on a stream that has stopped producing lines."""
    first = _CLAUDE_INIT if kind == "claude" else _CODEX_START
    spawn = _stalling_spawn([(0.0, first)])
    cog = _cog_for(kind, spawn, CliTimeouts.off())
    result, elapsed = await _cancel_after_first_line(
        cog, session=session, ctx=CancellingCtx(after=10_000)
    )

    assert result.evals["stop_reason"] == "cancelled"
    assert result.stop_reason == "terminated", "somebody chose this; nothing failed"
    assert elapsed < 1.0, f"took {elapsed:.2f}s to notice a tripped token"
    assert spawn.proc.terminated, (
        "noticing the cancel without ending the process moves the leak rather "
        "than fixing it — the CLI would keep running, and keep spending"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_poll_tick_is_not_progress(kind: str) -> None:
    """A tick must not hold the ``idle`` bound open.

    The two mechanisms share a wait, so the tempting implementation — treat
    every wakeup as a line — would make polling silently disable the one bound
    whose entire subject is silence."""
    first = _CLAUDE_INIT if kind == "claude" else _CODEX_START
    spawn = _stalling_spawn([(0.0, first)])
    # Longer than the 0.25s poll cadence, so ticks certainly happen first.
    cog = _cog_for(kind, spawn, CliTimeouts(startup=5.0, idle=0.4))
    result = await _drive(cog)
    assert result.evals["stop_reason"] == "idle_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_the_poll_never_delays_a_bound(kind: str) -> None:
    """A bound shorter than the poll cadence still fires on its own schedule.

    The poll is a floor on how long the loop will sit still, never a floor on
    how long a bound waits."""
    spawn = _stalling_spawn([])
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    started = time.monotonic()
    result = await _drive(cog)
    elapsed = time.monotonic() - started

    assert result.evals["stop_reason"] == "startup_timeout"
    assert elapsed < _CANCEL_POLL_S, (
        f"a 0.05s bound took {elapsed:.2f}s — the poll cadence delayed it"
    )


@pytest.mark.asyncio
async def test_no_ctx_means_no_ticks_at_all() -> None:
    """The fast path is load-bearing and has to stay reachable.

    A token-streamed run is thousands of lines, and bounding each one costs
    roughly twenty times an unbounded read. With no ctx there is no token to
    check, so the loop must not pay that — nor wake up on a cadence nobody
    reads."""
    stdout = _StallingStdout([(0.0, _CLAUDE_INIT), (0.0, _CLAUDE_TEXT)])
    deadline = _CliDeadline(CliTimeouts.off())
    seen = []
    with pytest.raises(TimeoutError):
        # No ``poll_s``: after the scripted lines the loop parks forever rather
        # than surfacing, which is precisely what "no ticks" means.
        async def _read() -> None:
            async for line in _iter_stdout(stdout, deadline):
                seen.append(line)

        await asyncio.wait_for(_read(), timeout=0.6)
    assert seen == [_CLAUDE_INIT, _CLAUDE_TEXT], seen
    assert None not in seen


@pytest.mark.asyncio
async def test_ticks_arrive_while_the_stream_is_silent() -> None:
    """The positive half of the above, at the level the behaviour lives."""
    stdout = _StallingStdout([(0.0, _CLAUDE_INIT)])
    deadline = _CliDeadline(CliTimeouts.off())
    seen: list[Any] = []

    async def _read() -> None:
        async for line in _iter_stdout(stdout, deadline, poll_s=0.05):
            seen.append(line)

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(_read(), timeout=0.4)

    assert seen[0] == _CLAUDE_INIT
    assert seen[1:] and all(x is None for x in seen[1:]), seen
    assert len(seen) >= 4, f"expected repeated ticks at 0.05s, got {len(seen)-1}"


@pytest.mark.asyncio
async def test_a_cancelled_read_does_not_lose_buffered_lines() -> None:
    """The property the whole design rests on, against a REAL pipe.

    A tick cancels the pending ``readline``. That is safe because
    ``StreamReader`` holds its buffer on the reader and only drains it once a
    complete line is in hand — but it is a property of the LINE protocol, not
    of the reader in general (``readexactly`` accumulates into a local and does
    lose data). This test is the guard on that distinction: rewrite the loop to
    read by size rather than by line and it goes red.
    """
    producer = (
        "import sys, time\n"
        "for i in range(2000):\n"
        "    sys.stdout.write(f'line-{i}\\n')\n"
        "    if i % 200 == 0:\n"
        "        sys.stdout.flush(); time.sleep(0.004)\n"
        "sys.stdout.flush()\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", producer, stdout=asyncio.subprocess.PIPE
    )
    assert proc.stdout is not None
    deadline = _CliDeadline(CliTimeouts.off())
    got, ticks = [], 0
    # A cadence far tighter than production, to force cancellations mid-read.
    async for line in _iter_stdout(proc.stdout, deadline, poll_s=0.001):
        if line is None:
            ticks += 1
            continue
        got.append(line.decode().strip())
    await proc.wait()

    assert ticks > 0, "no read was actually cancelled — the test proved nothing"
    assert got == [f"line-{i}" for i in range(2000)], (
        f"{2000 - len(got)} lines lost across {ticks} cancelled reads"
    )


# ── metering: one implementation, and a test that says so ───────────────────
#
# Both cognitions bypass the ``Invoker``, so the ``meter()`` middleware never
# sees their usage and every meter on the context would sit at zero. Each
# adapter therefore books its own spend — and for a while each did it with its
# own copy of the code, byte-identical apart from the docstrings.
#
# That is the worst place in this module to keep a duplicate. A fix applied to
# one copy and not the other fails no test and no type check; it just silently
# stops charging for one of the two CLIs, and the symptom is a ledger reading
# $0.00 — which looks like "the run was cheap", not like a bug.
#
# There is now one implementation. These tests are what keeps it that way: they
# assert the guarantees through BOTH front doors, so a future adapter-local
# reimplementation has to keep every one of them.


def _spend(kind: str, cost: float) -> Any:
    """A cognition whose one turn costs exactly ``cost``, whichever CLI it is.

    The two learn their cost differently — Claude is TOLD it by the CLI's
    ``total_cost_usd``, Codex computes it from a price table — so the fixture
    absorbs that difference and the tests below can be about the charging."""
    if kind == "claude":
        return ClaudeCliCognition(
            spawn=FakeClaudeCli.script(
                [
                    {"type": "system", "subtype": "init", "session_id": "s"},
                    {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}},
                    {"type": "result", "subtype": "success", "is_error": False,
                     "session_id": "s", "duration_ms": 5, "total_cost_usd": cost,
                     "usage": {"input_tokens": 1000, "output_tokens": 500}},
                ]
            )
        )

    def price(model: str | None, usage: Usage) -> float:
        del model, usage
        return cost

    return CodexCliCognition(
        pricing=price,
        spawn=FakeCodexCli.script(codex_turn(text="hi", usage=(1000, 0, 500))),
    )


async def _drive_with(cog: Any, ctx: Any) -> Any:
    agent = Agent(name="local", cognition=cog)
    events = [ev async for ev in cog.drive(agent, "t", ctx, WorkingContext())]
    return events[-1].result


def _run_ctx(budget: Budget, **kw: Any) -> RunContext:
    return RunContext("run-1", Scope(), services=Services(), budget=budget, **kw)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_the_spend_reaches_every_book(kind: str) -> None:
    """Run budget, actor budget and tenant quota, all from the one charge.

    The ``Quota`` is not decoration: it reads ``call.ctx.scope.key()`` to
    partition per tenant, which is the entire reason the charge goes through a
    ``_CliCall`` shim rather than a bare ``None``. Drop the shim and the run
    budget still looks fine while the quota raises."""
    budget = Budget(max_cost_usd=10.0)
    actor = ActorBudget(max_cost_usd=5.0, max_tokens=100_000, max_steps=10, max_wall_seconds=600.0)
    quota = Quota(max_usd=10.0)
    ctx = _run_ctx(budget, actor_budget=actor, meters=[quota])

    result = await _drive_with(_spend(kind, 0.75), ctx)

    assert result.usage.cost_usd == pytest.approx(0.75)
    # Everything is checked against what the RUN reported, not against a
    # literal, so the assertion is "the books agree with the answer" rather
    # than "this adapter happens to price things this way".
    assert budget.spent() == pytest.approx(result.usage.cost_usd)
    assert budget.usage.total_tokens == result.usage.total_tokens
    assert actor.used_cost() == pytest.approx(result.usage.cost_usd)
    assert actor.used_tokens == result.usage.total_tokens
    assert actor.used_steps == 1, "one CLI run is one step, on both adapters"
    assert quota.spent_in_window(Scope().key()) == pytest.approx(result.usage.cost_usd)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_meter_spend_false_charges_nothing_anywhere(kind: str) -> None:
    """The opt-out has to reach BOTH ends — meters and actor budget."""
    budget = Budget(max_cost_usd=10.0)
    actor = ActorBudget(max_cost_usd=5.0, max_tokens=100_000, max_steps=10, max_wall_seconds=600.0)
    quota = Quota(max_usd=10.0)
    ctx = _run_ctx(budget, actor_budget=actor, meters=[quota])

    cog = _spend(kind, 0.75)
    result = await _drive_with(dataclasses.replace(cog, meter_spend=False), ctx)

    assert result.usage.cost_usd == pytest.approx(0.75), "the run still COSTS it"
    assert budget.spent() == 0
    assert actor.used_cost() == 0 and actor.used_steps == 0
    assert quota.spent_in_window(Scope().key()) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_ceiling_crossed_by_this_run_is_recorded_not_raised(kind: str) -> None:
    """The money is spent and the answer exists.

    Turning the charge into an exception would lose a result the caller has
    already paid for, and break the exactly-one-terminal-event guarantee on top
    of it. The books still tell the truth."""
    budget = Budget(max_cost_usd=1.0, on_exceeded="raise")
    result = await _drive_with(_spend(kind, 3.00), _run_ctx(budget))

    assert result.stop_reason == "complete", "the RUN succeeded; only the charge complained"
    assert result.output == "hi"
    assert "MeterExceeded" in result.evals["meter_error"]
    assert budget.spent() == pytest.approx(3.00)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_meter_that_explodes_is_contained(kind: str) -> None:
    """A custom meter is third-party code running in a ``finally``-shaped spot.

    It gets the same containment a ceiling does — reported as data, never
    allowed to cost the answer — and the note names the exception so the
    failure is diagnosable rather than merely survived."""

    class _Exploding:
        async def charge(self, call: Any, usage: Usage) -> None:
            del call, usage
            raise ZeroDivisionError("boom")

    budget = Budget(max_cost_usd=10.0)
    ctx = _run_ctx(budget, meters=[_Exploding()])
    result = await _drive_with(_spend(kind, 0.50), ctx)

    assert result.stop_reason == "complete" and result.output == "hi"
    assert "ZeroDivisionError: boom" in result.evals["meter_error"]
    # The run budget is a separate meter and must still have been charged.
    assert budget.spent() == pytest.approx(0.50)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_no_ctx_and_no_meters_are_both_fine(kind: str) -> None:
    """The charge is best-effort by construction: nothing to charge is not an
    error, on either adapter."""
    result = await _drive_with(_spend(kind, 0.25), _run_ctx(Budget()))
    assert result.stop_reason == "complete"


# ── one very long line must not kill the run ────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_line_larger_than_the_default_buffer_survives(kind: str) -> None:
    """Both CLIs speak NDJSON where ONE line can carry a whole tool result.

    ``create_subprocess_exec`` defaults its reader to 64 KiB, so a file the
    agent read used to blow up the stream: measured against real codex on
    "read every file under agentkit/ and summarise each", the run came back
    ``parse_failed`` after three events with ``ValueError: Separator is not
    found, and chunk exceed the limit`` — while the CLI itself exited 0 and had
    the answer.

    Spawns a REAL process, because the thing under test is the buffer size the
    spawn asks for; a double would happily hand over a line of any length and
    prove nothing."""
    big = "x" * 200_000
    payload = (
        {"type": "assistant", "message": {"content": [{"type": "text", "text": big}]}}
        if kind == "claude"
        else {"type": "item.completed",
              "item": {"id": "i", "type": "agent_message", "text": big}}
    )
    tail = (
        {"type": "result", "subtype": "success", "is_error": False,
         "session_id": "s", "duration_ms": 1, "usage": {}}
        if kind == "claude"
        else {"type": "turn.completed", "usage": {}}
    )
    prog = (
        "import sys\n"
        f"sys.stdout.write({json.dumps(json.dumps(payload))} + chr(10))\n"
        f"sys.stdout.write({json.dumps(json.dumps(tail))} + chr(10))\n"
    )

    async def spawn(*_a: Any, **kw: Any) -> Any:
        return await asyncio.create_subprocess_exec(
            sys.executable, "-c", prog,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=kw.get("limit", 64 * 1024),
        )

    cog = _cog_for(kind, spawn, CliTimeouts(startup=10.0))
    result = await _drive(cog)

    assert result.evals.get("stop_reason") != "parse_failed", (
        f"a {len(big):,}-byte line broke the reader: {result.evals.get('error')}"
    )
    assert result.output == big


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_line_past_even_the_raised_limit_says_what_went_wrong(kind: str) -> None:
    """Raising the ceiling moved the failure; it did not make it legible.

    ``StreamReader.readline`` reports an over-limit line as a bare
    ``ValueError: Separator is not found, and chunk exceed the limit``, which
    both adapters classified as ``parse_failed`` — a reason that sends an
    operator hunting a corrupt payload when nothing was corrupt, and a message
    naming no limit, no payload and no fix. It reads like a bug in agentkit's
    parser rather than a buffer sized for a different protocol."""
    payload = json.dumps(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "y" * 200_000}]}}
        if kind == "claude"
        else {"type": "item.completed",
              "item": {"id": "i", "type": "agent_message", "text": "y" * 200_000}}
    )
    prog = (
        "import sys\n"
        f"sys.stdout.write({json.dumps(payload)} + chr(10))\n"
    )

    async def spawn(*_a: Any, **_kw: Any) -> Any:
        # A 64 KiB reader against a 200 KB line: the same overrun the default
        # buffer used to hit, without needing an 8 MB fixture to provoke it.
        return await asyncio.create_subprocess_exec(
            sys.executable, "-c", prog,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=64 * 1024,
        )

    cog = _cog_for(kind, spawn, CliTimeouts(startup=10.0))
    result = await _drive(cog)

    assert result.evals["stop_reason"] == "output_line_too_long", (
        "not 'parse_failed' — the line was valid JSON, just too big to assemble, "
        "so the fix is a size rather than a corruption hunt"
    )
    assert result.stop_reason == "failed" and result.partial is True

    error = result.evals["error"]
    assert "CliLineTooLong" in error
    assert "65,536-byte" in error, f"the limit has to be IN the message: {error}"
    assert "Separator is not found" not in error, "the asyncio wording must not leak out"
    # The advice has to be something the reader can act on without patching
    # library internals.
    assert "read the file in parts" in error


@pytest.mark.asyncio
async def test_the_exception_names_the_limit_it_was_given() -> None:
    exc = CliLineTooLong(8 * 1024 * 1024)
    assert exc.limit_bytes == 8 * 1024 * 1024
    assert "8,388,608" in str(exc)


# ── a pipe nobody closes must not hang the run ──────────────────────────────


class _NeverEofStderr:
    """stderr with a writer that never closes it.

    Not hypothetical. ``codex exec`` spawns a ``codex-code-mode-host`` helper
    that inherits stderr and outlives the run, so the pipe stays open after the
    CLI itself is finished and ``read()`` — which waits for EOF, meaning for
    EVERY writer to close — never returns.
    """

    def __init__(self, data: bytes = b"") -> None:
        self._data = data

    async def read(self, n: int = -1) -> bytes:
        """``StreamReader.read`` semantics, which is the whole point.

        ``read()`` with no size means "until EOF" — so on a pipe nobody closes
        it never returns, no matter how much data has already arrived. That
        distinction IS the bug: the chunked ``read(n)`` below returns what it
        has, and only the sizeless form hangs. A double that returned data for
        both would let the broken version pass."""
        if n < 0:
            await asyncio.Event().wait()  # never set: EOF is never reached
            raise AssertionError("unreachable")
        if self._data:
            out, self._data = self._data[:n], self._data[n:]
            return out
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_draining_stderr_keeps_what_it_read_and_gives_up_on_the_rest() -> None:
    """The bound must not cost the diagnostics.

    A single bounded ``read()`` would: with no size argument it accumulates
    into a local, so cancelling it discards everything. Chunked reads keep
    what arrived, which matters because this is the text explaining a
    failure."""
    started = time.monotonic()
    out = await _drain_stderr(_NeverEofStderr(b"Invalid API key"), 0.2)
    assert out == b"Invalid API key"
    assert time.monotonic() - started < 1.0


@pytest.mark.asyncio
async def test_an_empty_pipe_nobody_closes_still_gives_up() -> None:
    out = await _drain_stderr(_NeverEofStderr(b""), 0.2)
    assert out == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_run_ends_even_when_stderr_is_never_closed(kind: str) -> None:
    """The bug this guards is the worst shape a bug can have.

    The unbounded read lived in the driver's ``finally``, so it hung AFTER the
    stream was fully consumed — past every liveness bound, past cancellation,
    past anything the caller could do, and with the answer already in hand but
    never delivered. Measured against the real codex binary at roughly one run
    in three, which is exactly the profile of a defect that gets blamed on the
    model rather than reported."""
    first = _CLAUDE_INIT if kind == "claude" else _CODEX_START
    text = _CLAUDE_TEXT if kind == "claude" else _CODEX_TEXT
    spawn = _stalling_spawn([(0.0, first), (0.0, text)])
    spawn.proc.stderr = _NeverEofStderr(b"some diagnostics")
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.5, idle=0.2))

    result = await asyncio.wait_for(_drive(cog), timeout=10.0)
    assert result is not None


def test_the_production_preset_bounds_everything() -> None:
    t = CliTimeouts.production()
    assert all(v is not None for v in (t.startup, t.first_event, t.idle, t.total))
    assert t.idle >= 600, (
        "idle has to clear the longest TOOL call the session can make, not the "
        "longest model turn — a Bash(npm install) is minutes of legitimate silence"
    )


@pytest.mark.parametrize("field", ["startup", "first_event", "idle", "total"])
def test_a_zero_bound_is_refused(field: str) -> None:
    """Zero would fire before the process could be scheduled; ``None`` is how a
    bound is disabled, and conflating the two is a silently dead cognition."""
    with pytest.raises(ValueError, match="positive number of seconds or None"):
        CliTimeouts(**{field: 0})


# ─────────────────────────────────────────────────────────────────────────────
# 7. Structured-output failures carry structure.
#
# The diagnostics were always produced and always destroyed on the way out:
# every schema adapter yields per-field ``"path: message"`` entries, and both
# cognitions joined them into one sentence. An application that wanted to mark
# the offending field in a form, or group failures by path across a fleet, had
# to parse English that a library upgrade can reword.
# ─────────────────────────────────────────────────────────────────────────────


class Line(pydantic.BaseModel):
    qty: int
    sku: str


class Invoice(pydantic.BaseModel):
    vendor: str
    total: float
    lines: list[Line]


def _claude_structured(payload: Any) -> Any:
    return FakeClaudeCli.script(
        [
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": "s",
                "duration_ms": 1,
                "total_cost_usd": 0.0,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "structured_output": payload,
            }
        ]
    )


async def _typed_run(kind: str, payload: Any) -> Any:
    """Drive one cognition with an ``output=Invoice`` agent over ``payload``."""
    if kind == "claude":
        cli = _claude_structured(payload)
        cog = ClaudeCliCognition(spawn=cli)
    else:
        cli = FakeCodexCli.script(codex_turn(text=json.dumps(payload), usage=(1, 0, 1)))
        cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="local", output=Invoice, cognition=cog)
    events = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())]
    return events[-1].result


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_mismatch_names_the_failing_paths(kind: str) -> None:
    result = await _typed_run(kind, {"vendor": 5, "lines": [{"qty": "x"}]})
    assert result.evals["stop_reason"] == "structured_output_mismatch"

    failure = StructuredOutputFailure.of(result.evals)
    assert failure is not None
    assert failure.kind == "schema_mismatch"
    paths = {v.path for v in failure.violations}
    # JSONPath, not the adapters' dotted form: ``lines.0.qty`` is ambiguous the
    # moment a mapping has a numeric-looking key, and ``$``-rooted paths are
    # what a caller can hand to jq, a UI, or an error-grouping key.
    assert "$.vendor" in paths
    assert "$.total" in paths, "a missing required field is a violation of its own path"
    assert "$.lines[0].qty" in paths, "nested list indices must be bracketed"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_the_failure_survives_serialisation(kind: str) -> None:
    """``evals`` is deep-frozen, checkpointed and logged. A dataclass in there
    would make a result that cannot round-trip through JSON, which is why the
    stored form is a dict and ``.of()`` is the typed view of it."""
    result = await _typed_run(kind, {"vendor": 5})
    round_tripped = json.loads(json.dumps(dict(result.evals)))
    assert StructuredOutputFailure.of(round_tripped) == StructuredOutputFailure.of(result.evals)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_the_legacy_string_is_still_there(kind: str) -> None:
    """Callers and tests have read ``structured_output_error`` since this
    cognition shipped. Adding structure must not take the sentence away."""
    result = await _typed_run(kind, {"vendor": 5})
    assert isinstance(result.evals["structured_output_error"], str)
    assert result.evals["structured_output_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_no_raw_traceback_reaches_the_caller(kind: str) -> None:
    """"Never show raw Pydantic tracebacks to application users." The message
    is the adapter's own per-field diagnostics, not a stack."""
    result = await _typed_run(kind, {"vendor": 5})
    text = result.evals["structured_output_error"]
    assert "Traceback" not in text
    assert "pydantic_core" not in text
    assert "File \"" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_the_repair_prompt_is_compact_and_names_the_fields(kind: str) -> None:
    """Rendered, not sent. Neither cognition retries on it — see
    ``repair_prompt``'s own note on why an agentkit-side loop on top of the
    Claude CLI's internal one would be a second retry layer over an opaque
    one."""
    result = await _typed_run(kind, {"vendor": 5, "lines": [{"qty": "x"}]})
    prompt = StructuredOutputFailure.of(result.evals).repair_prompt()
    assert "$.vendor" in prompt and "$.lines[0].qty" in prompt
    assert len(prompt.splitlines()) <= 12, "a repair prompt the model has to read must stay short"


def test_the_four_kinds_are_distinguished() -> None:
    """Each has a different fix, so collapsing them would lose the diagnosis."""
    assert {
        StructuredOutputFailure(kind=k, detail="d").kind
        for k in ("missing", "undecodable", "schema_mismatch", "retries_exhausted")
    } == {"missing", "undecodable", "schema_mismatch", "retries_exhausted"}


@pytest.mark.asyncio
async def test_codex_prose_instead_of_json_is_undecodable_and_keeps_the_text() -> None:
    """For Codex the structured answer IS the final message, so "there was text
    and it was not JSON" is a different fault from "there was nothing" — and the
    text is the whole diagnosis."""
    cli = FakeCodexCli.script(codex_turn(text="Sure! Here is the invoice:", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="local", output=Invoice, cognition=cog)
    events = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())]

    failure = StructuredOutputFailure.of(events[-1].result.evals)
    assert failure is not None
    assert failure.kind == "undecodable"
    assert "Sure!" in (failure.raw_excerpt or "")


@pytest.mark.asyncio
async def test_claude_exhausted_retries_is_its_own_kind() -> None:
    """The CLI validated and re-prompted ITSELF and gave up. agentkit never saw
    the attempts, so there are no per-field violations to report — and claiming
    otherwise would be inventing diagnostics we do not have."""
    cli = FakeClaudeCli.script(
        [
            {
                "type": "result",
                "subtype": "error_max_structured_output_retries",
                "is_error": True,
                "session_id": "s",
                "duration_ms": 1,
                "total_cost_usd": 0.0,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ]
    )
    cog = ClaudeCliCognition(spawn=cli)
    agent = Agent(name="local", output=Invoice, cognition=cog)
    events = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())]
    result = events[-1].result

    assert result.stop_reason == "invalid_output"
    failure = StructuredOutputFailure.of(result.evals)
    assert failure is not None and failure.kind == "retries_exhausted"
    assert failure.violations == ()


def test_a_diagnostic_with_no_field_is_scoped_to_the_root() -> None:
    """A JSON Schema violation like "'z' is a required property" belongs to the
    OBJECT, not to ``z``. Inventing ``$.z`` would point a caller at a field the
    payload does not have."""
    from agentkit.agents.cognition._cli_common import _parse_violation

    assert _parse_violation("<root>: 'z' is a required property").path == "$"
    assert _parse_violation("no separator here").path == "$"


def test_a_message_containing_a_colon_keeps_its_whole_message() -> None:
    """Split on the FIRST separator only — splitting on the last moves half the
    message into the path."""
    from agentkit.agents.cognition._cli_common import _parse_violation

    v = _parse_violation("a.b: got {'k': 1}, expected int")
    assert v.path == "$.a.b"
    assert v.message == "got {'k': 1}, expected int"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Secrets must not ride in argv, and must not be persisted verbatim.
#
# An argument list is world-readable — any local account can ``ps`` it. Moving
# the PROMPT off argv was about size and privacy; this is about credentials,
# and it is the same fix applied to a payload that IS the secret.
# ─────────────────────────────────────────────────────────────────────────────

_SECRET_MCP = '{"mcpServers":{"s":{"type":"http","url":"https://x/mcp",' \
              '"headers":{"Authorization":"Bearer sk-SUPERSECRETVALUE123"}}}}'


@pytest.mark.asyncio
async def test_claude_inline_mcp_and_settings_never_reach_argv() -> None:
    """Measured before the fix: two argv entries carrying a bearer token from
    an ordinary ``mcp_config=`` / ``settings=`` pair, readable by every local
    account on the machine."""
    cog, cli, _ = _claude_case()
    object.__setattr__(cog, "mcp_config", (_SECRET_MCP,))
    object.__setattr__(cog, "settings", '{"apiKeyHelper":"echo sk-ANOTHERSECRET123"}')
    await _drive(cog)

    argv = cli.invocations[-1].argv
    assert not any("SUPERSECRET" in a for a in argv), f"secret in argv: {argv}"
    assert not any("ANOTHERSECRET" in a for a in argv), f"secret in argv: {argv}"
    # The flags are still there — the fix is a change of transport, not of
    # configuration.
    assert "--mcp-config" in argv and "--settings" in argv


@pytest.mark.asyncio
async def test_the_materialised_file_has_the_content_and_is_private() -> None:
    """Looked at DURING the spawn, because the scratch directory is removed as
    soon as the run ends — a secret written to disk to keep it out of argv
    would be a poor trade if it outlived the process."""
    seen: dict[str, Any] = {}

    inner = _claude_case()[1]

    def _peek(path: Path) -> None:
        """The stat/read pair, out of the async frame. ASYNC240 refuses
        blocking pathlib calls inside an ``async def`` and is right in general;
        this is one small local file and the alternative is an async-fs
        dependency for a two-line assertion."""
        seen["text"] = path.read_text()
        seen["mode"] = oct(path.stat().st_mode)[-3:]
        seen["path"] = path

    async def spying_spawn(*argv: str, **kw: Any) -> Any:
        _peek(Path(argv[argv.index("--mcp-config") + 1]))
        return await inner(*argv, **kw)

    cog = ClaudeCliCognition(spawn=spying_spawn, mcp_config=(_SECRET_MCP,))
    await _drive(cog)

    assert seen["text"] == _SECRET_MCP, "the CLI must still get the real configuration"
    assert seen["mode"] == "600", "world- or group-readable would defeat the point"
    assert not seen["path"].exists(), "the scratch file must not outlive the run"


@pytest.mark.asyncio
async def test_a_path_is_passed_through_untouched() -> None:
    """A path is already a reference rather than a value, so there is nothing
    to move — and rewriting it would break a caller pointing at a file they
    manage themselves."""
    cog, cli, _ = _claude_case()
    object.__setattr__(cog, "mcp_config", ("/etc/agentkit/mcp.json",))
    await _drive(cog)
    argv = cli.invocations[-1].argv
    assert argv[argv.index("--mcp-config") + 1] == "/etc/agentkit/mcp.json"


def test_codex_warns_because_it_cannot_move_the_secret() -> None:
    """``codex exec`` has no config-file flag, so ``-c key=value`` is the only
    way to set an override and agentkit cannot move it out of argv. Warned, not
    refused — the value may be a placeholder or a short-lived scoped token, and
    refusing would break a working configuration over a name heuristic."""
    with pytest.warns(UserWarning, match="process argument list"):
        CodexCliCognition(config_overrides={"model_providers.x.api_key": "sk-SECRET"})


def test_codex_does_not_warn_about_the_documented_safe_pattern() -> None:
    """``bearer_token_env_var`` NAMES an environment variable to read the token
    from — it is the fix, not the leak. Warning about it would teach people to
    silence the warning that matters."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CodexCliCognition(mcp_servers={"s": {"url": "u", "bearer_token_env_var": "T"}})


@pytest.mark.parametrize(
    ("raw", "must_not_contain"),
    [
        ("Authorization: Bearer sk-ant-abc123DEF456ghi789xyz", "sk-ant-abc123"),
        ('api_key="sk-SUPERSECRETVALUE12345" rejected', "SUPERSECRET"),
        ("token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", "ghp_ABCDEF"),
        ("AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ],
)
def test_credential_shapes_are_redacted(raw: str, must_not_contain: str) -> None:
    from agentkit.agents.cognition._cli_common import _redact_secrets

    assert must_not_contain not in _redact_secrets(raw)


@pytest.mark.parametrize(
    "diagnostic",
    [
        "model claude-opus-4-5 not found, retrying in 2s",
        "working_dir does not exist: /tmp/nope",
        "MCP server 'database' failed to start after 30000ms",
    ],
)
def test_ordinary_diagnostics_survive_redaction(diagnostic: str) -> None:
    """The trap. This text is what an operator reads to diagnose a failed run,
    and a redactor that eats ordinary words makes it useless — which is how
    redaction gets switched off wholesale. ``claude-opus-4-5`` is exactly the
    kind of long hyphenated token a naive rule would swallow."""
    from agentkit.agents.cognition._cli_common import _redact_secrets

    assert _redact_secrets(diagnostic) == diagnostic


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_stderr_is_redacted_before_it_is_persisted(kind: str) -> None:
    """An ``AgentResult`` is checkpointed, logged and fanned out to observers,
    so verbatim stderr is a credential written to durable storage."""
    leak = b"request failed: Authorization: Bearer sk-ant-LEAKEDSECRET12345"
    spawn = _stalling_spawn([], stderr=leak)
    cog = _cog_for(kind, spawn, CliTimeouts(startup=0.05))
    result = await _drive(cog)
    assert "LEAKEDSECRET" not in result.evals.get("stderr", "")
    assert "[redacted" in result.evals["stderr"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. A stream that stopped is not a run that finished.
# ─────────────────────────────────────────────────────────────────────────────


def _truncated(kind: str) -> Any:
    """A stream cut off mid-JSON-object — a killed process, a full disk, a
    broken pipe. The complete lines before the fragment are real."""
    if kind == "claude":
        return FakeClaudeCli.script(
            [
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "half"}]}},
                '{"type":"result","subtype":"suc',
            ]
        )
    return FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "half"}},
            '{"type":"turn.comp',
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_truncated_stream_is_not_a_complete_run(kind: str) -> None:
    """The worst shape a failure can take, and what both adapters used to do.

    Measured before ``saw_terminal`` existed: ``stop_reason="complete"``,
    ``partial=False``, and half an answer handed back as a finished one. The
    reason it hid so well is that a SUCCESSFUL terminal payload also leaves
    ``stop_reason`` at ``None``, so "the CLI finished" and "the stream stopped"
    were the same value.
    """
    cls = ClaudeCliCognition if kind == "claude" else CodexCliCognition
    cog = cls(spawn=_truncated(kind))
    agent = Agent(name="local", cognition=cog)
    result = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())][-1].result

    assert result.evals["stop_reason"] == "malformed_output"
    assert result.partial is True
    assert result.stop_reason == "failed"
    assert result.output == "half", "the lines that did arrive are still real"


@pytest.mark.asyncio
@pytest.mark.parametrize("make", CASES)
async def test_a_complete_run_is_still_complete(make: Any) -> None:
    """The other half. A stream-integrity check that fires on healthy runs is
    worse than no check at all."""
    cog, _, _ = make()
    result = await _drive(cog)
    assert result.partial is False
    assert result.evals.get("stop_reason") != "malformed_output"


@pytest.mark.asyncio
async def test_the_legacy_codex_vocabulary_still_completes() -> None:
    """The regression this nearly shipped as.

    Older ``codex`` ends a turn with ``task_complete``, not ``turn.completed``.
    The first version of the stream-integrity check only knew the current
    vocabulary, so every complete run on an older binary would have been
    reported ``malformed_output`` — a version-compatibility feature turned into
    a version-compatibility bug. Caught by the existing legacy-parity test.
    """
    cli = FakeCodexCli.script(
        [
            {"id": "0", "msg": {"type": "session_configured", "session_id": "s", "model": "m"}},
            {"id": "1", "msg": {"type": "agent_message", "message": "Nine files."}},
            {"id": "2", "msg": {"type": "task_complete", "last_agent_message": "Nine files."}},
        ]
    )
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="local", cognition=cog)
    result = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())][-1].result
    assert result.partial is False
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_a_signal_death_is_a_crash_not_an_exit_code(kind: str) -> None:
    """``cli_exit_-9`` reads as an exit status that does not exist. A negative
    return code is a death by signal — the OOM killer, a segfault, an
    operator's ``kill`` — and only reaches this branch for a signal WE did not
    send, because cancellation and timeouts are decided first and win."""
    cls = ClaudeCliCognition if kind == "claude" else CodexCliCognition
    fake = FakeClaudeCli if kind == "claude" else FakeCodexCli
    cog = cls(spawn=fake.script([], returncode=-9))
    agent = Agent(name="local", cognition=cog)
    result = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())][-1].result
    assert result.evals["stop_reason"] == "process_crashed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "stderr"),
    [
        ("claude", b"Error: --json-schema is not a valid JSON Schema"),
        ("codex", b"error: failed to parse output schema at --output-schema path"),
    ],
)
async def test_a_schema_the_cli_refuses_says_so(kind: str, stderr: bytes) -> None:
    """Pre-spawn validation catches the structural mistakes, but each binary
    has its own extra rules. ``cli_exit_2`` makes an operator dig the reason out
    of stderr; naming it does not."""
    cls = ClaudeCliCognition if kind == "claude" else CodexCliCognition
    fake = FakeClaudeCli if kind == "claude" else FakeCodexCli
    cog = cls(spawn=fake.script([], returncode=2, stderr=stderr))
    agent = Agent(name="local", cognition=cog)
    result = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())][-1].result
    assert result.evals["stop_reason"] == "schema_rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude", "codex"])
async def test_an_ordinary_non_zero_exit_keeps_its_code(kind: str) -> None:
    """The fallback has to stay. Refinements that swallow the exit code would
    lose the only thing an operator can look up."""
    cls = ClaudeCliCognition if kind == "claude" else CodexCliCognition
    fake = FakeClaudeCli if kind == "claude" else FakeCodexCli
    cog = cls(spawn=fake.script([], returncode=3, stderr=b"something else went wrong"))
    agent = Agent(name="local", cognition=cog)
    result = [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())][-1].result
    assert result.evals["stop_reason"] == "cli_exit_3"
