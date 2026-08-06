"""Tests for :class:`ClaudeCliCognition`.

The CLI process is stubbed with an ``AsyncMock`` on
``asyncio.create_subprocess_exec`` returning a fake ``Process`` whose
``stdout`` yields canned JSONL lines. That's the seam these tests own —
everything downstream (event translation, cost extraction, cancellation)
is behavior tested through the resulting ``StreamEvent`` stream.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.agents.cognition.claude_cli import _get_semaphore
from agentkit.context import WorkingContext
from agentkit.prompts.prompt import Prompt
from agentkit.testing.fakes.ctx import FakeCtx

# ─────────────────────────────────────────────────────────────────────────────
# Fake subprocess plumbing
# ─────────────────────────────────────────────────────────────────────────────


class _FakeStdout:
    """Async iterator yielding one line per iteration — mirrors
    ``asyncio.subprocess.Process.stdout``'s ``async for line in stdout``.

    Each line is either bytes (a JSON payload) or a callable executed
    inline (used to inject side effects like flipping a cancel token
    mid-stream). The callable is called and its return value (bytes | None)
    is yielded — ``None`` means "skip this position", which lets tests
    interleave state changes between JSON payloads without emitting a
    junk line.
    """

    def __init__(self, lines: list[bytes | Any]) -> None:
        self._lines = list(lines)

    def __aiter__(self) -> _FakeStdout:
        return self

    async def __anext__(self) -> bytes:
        while self._lines:
            item = self._lines.pop(0)
            if callable(item):
                out = item()
                if out is None:
                    continue
                return out
            return item
        raise StopAsyncIteration


class _FakeStderr:
    def __init__(self, data: bytes = b"") -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[bytes | Any],
        stderr: bytes = b"",
        returncode: int = 0,
        terminate_hook: Any = None,
    ) -> None:
        self.stdout = _FakeStdout(stdout_lines)
        self.stderr = _FakeStderr(stderr)
        self._returncode: int | None = None
        self._final_returncode = returncode
        self._terminate_hook = terminate_hook
        self.terminated = False
        self.killed = False

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def wait(self) -> int:
        # If terminate wasn't called, close cleanly with the configured exit code.
        if self._returncode is None:
            self._returncode = self._final_returncode
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        if self._terminate_hook is not None:
            self._terminate_hook()
        # Simulate a well-behaved process: exit on SIGTERM.
        self._returncode = -15

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9


def _line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload) + "\n").encode()


# ─────────────────────────────────────────────────────────────────────────────
# Canned CLI event streams
# ─────────────────────────────────────────────────────────────────────────────


def _happy_path_lines() -> list[bytes]:
    return [
        _line(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "sess-happy",
                "cwd": "/tmp",
            }
        ),
        _line(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world."},
                        {
                            "type": "tool_use",
                            "id": "tu-1",
                            "name": "Read",
                            "input": {"path": "/tmp/x"},
                        },
                    ]
                },
            }
        ),
        _line(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu-1",
                            "content": "file contents",
                        }
                    ]
                },
            }
        ),
        _line(
            {
                "type": "result",
                "session_id": "sess-happy",
                "duration_ms": 1234,
                "total_cost_usd": 0.0123,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 3,
                },
            }
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _drain(cog: ClaudeCliCognition, agent: Agent, ctx: FakeCtx, task: str = "do it") -> list[Any]:
    async def _go() -> list[Any]:
        events = []
        async for ev in cog.drive(agent, task, ctx, WorkingContext()):
            events.append(ev)
        return events

    return _run(_go())


def test_happy_path_emits_all_events_in_order() -> None:
    """init → assistant text × 2 → tool_use → tool_result → result → final.

    Verify: two message_delta events, one tool_call, one tool_result, one final.
    """
    proc = _FakeProcess(stdout_lines=_happy_path_lines())
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        cog = ClaudeCliCognition(model="claude-opus-4-5")
        agent = Agent(name="local", prompt="You are helpful.", cognition=cog)
        events = _drain(cog, agent, FakeCtx())

    kinds = [e.type for e in events]
    assert kinds == [
        "message_delta",
        "message_delta",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert events[0].text == "Hello "
    assert events[1].text == "world."
    assert events[2].tool_call.name == "Read"
    assert events[2].tool_call.arguments == {"path": "/tmp/x"}
    assert events[3].tool_result == "file contents"

    final = events[-1]
    assert final.result.output == "Hello world."
    assert final.result.partial is False

    # The argv must include the required flags.
    argv = spawn.await_args.args
    assert argv[0] == "claude"
    assert "-p" in argv and "do it" in argv
    assert "--output-format" in argv and "stream-json" in argv
    assert "--verbose" in argv
    assert "--model" in argv and "claude-opus-4-5" in argv
    assert "--system-prompt" in argv and "You are helpful." in argv


def test_cost_extraction_from_result_message() -> None:
    proc = _FakeProcess(stdout_lines=_happy_path_lines())
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        events = _drain(cog, agent, FakeCtx())

    final = events[-1]
    assert final.result.usage.cost_usd == pytest.approx(0.0123)
    assert final.result.usage.input_tokens == 100
    assert final.result.usage.output_tokens == 25
    assert final.result.usage.cache_read_tokens == 5
    assert final.result.usage.cache_write_tokens == 3


def test_session_id_lands_in_evals() -> None:
    proc = _FakeProcess(stdout_lines=_happy_path_lines())
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        events = _drain(cog, agent, FakeCtx())

    final = events[-1]
    assert final.result.evals["session_id"] == "sess-happy"
    assert final.result.evals["cli_duration_ms"] == 1234


def test_non_zero_exit_yields_final_with_stop_reason_and_stderr() -> None:
    """Non-zero CLI exit does NOT raise. Per the terminal-event guarantee,
    drive() must yield exactly one final event with partial=True,
    stop_reason=cli_exit_<code>, and the captured stderr on evals."""
    proc = _FakeProcess(
        stdout_lines=[],
        stderr=b"auth: no OAuth token\n",
        returncode=2,
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        events = _drain(cog, agent, FakeCtx())

    final = events[-1]
    assert final.type == "final"
    assert final.result.partial is True
    assert final.result.evals["stop_reason"] == "cli_exit_2"
    assert "auth: no OAuth token" in final.result.evals["stderr"]
    assert final.result.evals["cli_return_code"] == 2


def test_thinking_block_folds_into_evals_and_emits_delta() -> None:
    """Extended-thinking blocks surface as message_delta events for live
    consumers AND fold into AgentResult.evals['thinking'] for run() callers.
    They MUST NOT contaminate AgentResult.output — that's the response text."""
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "sess-t"}),
            _line(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "Let me consider.", "signature": "s"},
                            {"type": "text", "text": "Answer: 42"},
                        ]
                    },
                }
            ),
            _line({"type": "result", "session_id": "sess-t", "duration_ms": 5, "usage": {}}),
        ],
        returncode=0,
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        events = _drain(cog, Agent(name="local", cognition=cog), FakeCtx())

    deltas = [e.text for e in events if e.type == "message_delta"]
    assert "Let me consider." in deltas
    assert "Answer: 42" in deltas
    final = events[-1]
    assert final.result.output == "Answer: 42"  # thinking NOT in output
    assert final.result.evals["thinking"] == "Let me consider."


def test_result_with_is_error_marks_partial_with_stop_reason() -> None:
    """The CLI can exit 0 with is_error=true and a subtype like
    error_max_turns. drive() must honour the semantic outcome, not just
    the exit code."""
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "sess-e"}),
            _line(
                {
                    "type": "result",
                    "is_error": True,
                    "subtype": "error_max_turns",
                    "session_id": "sess-e",
                    "duration_ms": 100,
                    "usage": {},
                }
            ),
        ],
        returncode=0,
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        events = _drain(cog, Agent(name="local", cognition=cog), FakeCtx())

    final = events[-1]
    assert final.result.partial is True
    assert final.result.evals["stop_reason"] == "error_max_turns"


def test_spawn_failure_yields_final_with_stop_reason_and_error() -> None:
    """If the claude binary isn't on PATH, create_subprocess_exec raises
    FileNotFoundError. drive() must still yield exactly one final event
    (partial=True, stop_reason=spawn_failed, error message on evals)."""

    async def _raise(*_a: object, **_kw: object) -> object:
        raise FileNotFoundError("[Errno 2] No such file or directory: 'claude'")

    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=_raise,
    ):
        cog = ClaudeCliCognition(claude_bin="claude")
        events = _drain(cog, Agent(name="local", cognition=cog), FakeCtx())

    assert len(events) == 1
    final = events[-1]
    assert final.type == "final"
    assert final.result.partial is True
    assert final.result.evals["stop_reason"] == "spawn_failed"
    assert "FileNotFoundError" in final.result.evals["error"]


def test_cancellation_terminates_process_and_still_emits_final() -> None:
    """A cancel token flip mid-stream must (a) call terminate on the
    subprocess and (b) still emit exactly one terminal final event marked
    partial=True."""

    class _CancellingCtx(FakeCtx):
        def __init__(self) -> None:
            super().__init__()
            self._cancel = False

        def trip(self) -> None:
            self._cancel = True

        def check_cancelled(self) -> None:
            if self._cancel:
                raise RuntimeError("run cancelled")

    ctx = _CancellingCtx()

    # Two payloads then a callable that trips cancel, then more payloads
    # the drive loop should NEVER reach.
    def _trip_cancel() -> None:
        ctx.trip()
        return None

    lines: list[Any] = [
        _line({"type": "system", "session_id": "sess-cancel"}),
        _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "start"}]}}),
        _trip_cancel,
        # Should not reach these — the loop breaks after cancel is detected.
        _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "MORE"}]}}),
        _line({"type": "result", "session_id": "sess-cancel", "total_cost_usd": 0.0}),
    ]
    proc = _FakeProcess(stdout_lines=lines)
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        events = _drain(cog, agent, ctx)

    assert proc.terminated is True
    # Final event still emitted with partial=True and the accumulated text so far.
    final = events[-1]
    assert final.type == "final"
    assert final.result.partial is True
    assert final.result.evals["stop_reason"] == "cancelled"
    assert final.result.output == "start"
    # Did not emit the post-cancel token.
    texts = [e.text for e in events if e.type == "message_delta"]
    assert texts == ["start"]


def test_prompt_instance_is_rendered() -> None:
    """When ``agent.prompt`` is a ``Prompt`` (versioned), the cognition MUST
    call ``.render()`` and pass the rendered text via ``--system-prompt``."""
    proc = _FakeProcess(stdout_lines=_happy_path_lines())
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        p = Prompt(id="sys.v1", version="1", template="   Concise assistant.   ")
        cog = ClaudeCliCognition()
        agent = Agent(name="local", prompt=p, cognition=cog)
        _drain(cog, agent, FakeCtx())

    argv = spawn.await_args.args
    idx = argv.index("--system-prompt")
    # ``Prompt.render()`` strips surrounding whitespace.
    assert argv[idx + 1] == "Concise assistant."


def test_config_dir_landing_in_env() -> None:
    proc = _FakeProcess(stdout_lines=_happy_path_lines())
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        cog = ClaudeCliCognition(config_dir=Path("/tmp/isolated-claude"))
        agent = Agent(name="local", cognition=cog)
        _drain(cog, agent, FakeCtx())

    env = spawn.await_args.kwargs["env"]
    assert env["CLAUDE_CONFIG_DIR"] == "/tmp/isolated-claude"
    # Default watchdog toggle is present unless the caller pre-set it.
    assert env["CLAUDE_ENABLE_STREAM_WATCHDOG"] == "1"


def test_concurrency_semaphore_serializes_parallel_drives() -> None:
    """Three concurrent drives with max_concurrent=1 must NOT overlap —
    each drive holds the semaphore for the whole CLI lifetime."""
    spawn_timestamps: list[float] = []
    finish_timestamps: list[float] = []

    async def _fake_spawn(*args: Any, **kwargs: Any) -> _FakeProcess:
        spawn_timestamps.append(time.perf_counter())

        # Build stdout lines with a small delay embedded via awaits inside
        # the fake stdout. Simplest: return a Process whose stdout yields
        # one payload after a sleep, then the result.
        class _SlowStdout:
            def __init__(self) -> None:
                self._first = True

            def __aiter__(self) -> Any:
                return self

            async def __anext__(self) -> bytes:
                if self._first:
                    self._first = False
                    await asyncio.sleep(0.05)
                    return _line({"type": "system", "session_id": "s"})
                if getattr(self, "_second", True):
                    self._second = False
                    await asyncio.sleep(0.05)
                    return _line({"type": "result", "session_id": "s", "total_cost_usd": 0.0})
                raise StopAsyncIteration

        p = _FakeProcess(stdout_lines=[])
        p.stdout = _SlowStdout()  # type: ignore[assignment]
        return p

    async def _go() -> None:
        # Fresh key so any prior test's semaphore doesn't shift the count.
        cog = ClaudeCliCognition(claude_bin="claude-serialize-test", max_concurrent=1)
        agent = Agent(name="local", cognition=cog)
        with patch(
            "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
            side_effect=_fake_spawn,
        ):

            async def _one() -> None:
                async for _ in cog.drive(agent, "task", FakeCtx(), WorkingContext()):
                    pass
                finish_timestamps.append(time.perf_counter())

            await asyncio.gather(_one(), _one(), _one())

    _run(_go())

    # Serialized: each spawn happens AFTER the prior drive finished.
    assert len(spawn_timestamps) == 3
    assert len(finish_timestamps) == 3
    for i in range(1, 3):
        # Some clock jitter is fine — just ensure ordering: the i-th spawn
        # happens after the (i-1)-th finish (up to a tiny tolerance).
        assert spawn_timestamps[i] >= finish_timestamps[i - 1] - 0.005


def test_semaphore_is_shared_per_bin_config_key() -> None:
    """Two cognitions with the same (bin, config_dir, max_concurrent) tuple
    share a semaphore; a different key gets a fresh one."""
    s1 = _get_semaphore("claude", None, 4)
    s2 = _get_semaphore("claude", None, 4)
    s3 = _get_semaphore("claude", "/tmp/isolated", 4)
    assert s1 is s2
    assert s1 is not s3
