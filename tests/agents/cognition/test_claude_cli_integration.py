"""Integration + non-happy-path coverage for :class:`ClaudeCliCognition`.

Extends the base contract in ``test_claude_cli.py`` with two categories:

- **Goal A** — non-happy-path exit shapes (mock-heavy, a small handful of real
  CLI invocations gated on the ``claude`` binary being on PATH). Every path
  must land the terminal-event guarantee: exactly one ``final`` StreamEvent
  regardless of failure mode.

- **Goal B** — how the cognition plays with other agentkit components:
  Budget / trace / observer / autonomy / streaming / ReAct wrapping /
  Workflow / Checkpointer / MemorySource / semaphore / correlation_id.

Real-CLI tests are marked with ``@pytest.mark.real_cli`` and skipped when
the binary isn't on PATH or the env var ``AGENTKIT_SKIP_REAL_CLI=1`` is
set. Cost budget for the whole file: ~5-6 real CLI calls (Sonnet, small
tasks — ~$0.05-0.10 total).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.agents.workflow import Workflow
from agentkit.context import WorkingContext
from agentkit.kernel.types import StreamEvent
from agentkit.testing.fakes.ctx import FakeCtx
from agentkit.tools.from_agent import as_tool
from tests.agents.cognition.test_claude_cli import (
    _FakeProcess,
    _FakeStdin,
    _line,
)

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────


_REAL_CLI = shutil.which("claude")
_SKIP_REAL = os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1"

real_cli = pytest.mark.skipif(
    _REAL_CLI is None or _SKIP_REAL,
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


class _CancellingCtx(FakeCtx):
    """FakeCtx whose ``check_cancelled`` raises after ``trip()``."""

    def __init__(self) -> None:
        super().__init__()
        self._cancel = False

    def trip(self) -> None:
        self._cancel = True

    def check_cancelled(self) -> None:
        if self._cancel:
            raise RuntimeError("run cancelled")


class _RecordingObserverCtx(FakeCtx):
    """FakeCtx that records every ctx.emit(...) call for later assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.emissions: list[dict[str, Any]] = []

    async def emit(
        self,
        kind: str,
        render: str = "",
        *,
        payload: Any = None,
        agent: str = "",
        parent_id: Any = None,
    ) -> None:
        self.emissions.append({"kind": kind, "render": render, "payload": payload, "agent": agent})


async def _drain_async(gen: Any) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for ev in gen:
        events.append(ev)
    return events


# ============================================================================
# Goal A — non-happy paths
# ============================================================================


# A.1 — missing binary path via both agent.run and agent.stream ---------------


def test_A1_missing_binary_run_returns_final_no_exception() -> None:
    cog = ClaudeCliCognition(claude_bin="/tmp/definitely-not-here-XYZ")
    agent = Agent(name="local", cognition=cog)
    ctx = FakeCtx()

    async def _go() -> Any:
        return await agent.run("hi", ctx)

    result = asyncio.run(_go())
    assert result.partial is True
    assert result.evals["stop_reason"] == "spawn_failed"
    assert "FileNotFoundError" in result.evals["error"]


def test_A1_missing_binary_stream_returns_single_final() -> None:
    cog = ClaudeCliCognition(claude_bin="/tmp/definitely-not-here-XYZ")
    agent = Agent(name="local", cognition=cog)
    ctx = FakeCtx()

    async def _go() -> list[StreamEvent]:
        return await _drain_async(agent.stream("hi", ctx))

    events = asyncio.run(_go())
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1
    assert finals[0].result.partial is True
    assert finals[0].result.evals["stop_reason"] == "spawn_failed"


# A.2 — real-CLI cancellation via ctx.check_cancelled ------------------------


@real_cli
def test_A2_real_cli_cancellation_via_ctx_terminates_process() -> None:
    """Start a real CLI run and trip ctx.check_cancelled after ~1s. The
    subprocess must be terminated, one terminal ``final`` emitted with
    partial=True + stop_reason=cancelled + negative cli_return_code."""
    cog = ClaudeCliCognition()
    agent = Agent(name="local", cognition=cog)
    ctx = _CancellingCtx()

    async def _go() -> list[StreamEvent]:
        events: list[StreamEvent] = []

        async def _consume() -> None:
            async for ev in agent.stream(
                "Write me a haiku, then a limerick, then a sonnet about octopuses. Take your time.",
                ctx,
            ):
                events.append(ev)

        task = asyncio.create_task(_consume())
        await asyncio.sleep(1.2)
        ctx.trip()
        await task
        return events

    events = asyncio.run(_go())
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1, "terminal-event guarantee: exactly one final"
    final = finals[0].result
    assert final.partial is True
    assert final.evals["stop_reason"] == "cancelled"
    # Native SIGTERM = -15, SIGKILL = -9 — but the ``claude`` binary is a
    # node wrapper that translates a signal into 128+signum (143 for SIGTERM,
    # 137 for SIGKILL). Accept both shapes; the point is "abnormal exit".
    rc = final.evals["cli_return_code"]
    assert rc < 0 or rc in (137, 143), f"unexpected cli_return_code={rc}"


# A.3 — external asyncio.wait_for timeout ------------------------------------


def test_A3_wait_for_timeout_raises_TimeoutError_and_terminates_child() -> None:
    """Regression lock-in for the HIGH bug caught by the integration audit.

    When a caller wraps ``agent.run(...)`` in ``asyncio.wait_for(..., timeout)``,
    the timeout MUST propagate to the caller as ``TimeoutError``. Prior to
    the fix, ``drive()``'s outer ``except BaseException`` swallowed
    ``asyncio.CancelledError`` — the caller lost the timeout signal and got
    a silent partial result instead.

    Fix in ``drive()``: ``asyncio.CancelledError`` is handled in a dedicated
    ``except`` arm that terminates the subprocess, yields the terminal
    ``final(stop_reason='cancelled', partial=True)`` event, and then
    re-raises so ``wait_for`` sees the cancel.
    """
    # A "long-running" but cooperative fake: stdout awaits a slow sleep
    # (giving wait_for a chance to fire mid-await), and wait()/stderr resolve
    # quickly so the drive's finally block can proceed once the inner
    # cancel is caught. This mirrors "the child eventually exits after the
    # parent's read pipes close on cancel."

    class _SlowStdout:
        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> bytes:
            # Sleep past the wait_for timeout; wait_for's cancel raises
            # asyncio.CancelledError here.
            await asyncio.sleep(2.0)
            return b'{"type":"result","session_id":"s","duration_ms":1,"usage":{}}\n'

    class _SlowStderr:
        async def read(self, n: int = -1) -> bytes:
            return b""

    class _CooperativeProcess:
        def __init__(self) -> None:
            self.stdout = _SlowStdout()
            self.stderr = _SlowStderr()
            self.stdin = _FakeStdin()
            self._returncode: int | None = None

        @property
        def returncode(self) -> int | None:
            return self._returncode

        async def wait(self) -> int:
            # Simulate child exiting quickly once the parent closes pipes.
            if self._returncode is None:
                self._returncode = 0
            return self._returncode

        def terminate(self) -> None:
            self._returncode = -15

        def kill(self) -> None:
            self._returncode = -9

    async def _fake_spawn(*_a: Any, **_kw: Any) -> Any:
        return _CooperativeProcess()

    async def _go() -> Any:
        with patch(
            "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
            side_effect=_fake_spawn,
        ):
            cog = ClaudeCliCognition()
            agent = Agent(name="local", cognition=cog)
            return await asyncio.wait_for(agent.run("hi", FakeCtx()), timeout=0.15)

    # Post-fix: TimeoutError MUST propagate. Drive still yields the
    # terminal final event (so consumers of `agent.stream(...)` see one)
    # but then re-raises CancelledError, which `wait_for` translates into
    # `TimeoutError` for the caller.
    with pytest.raises(TimeoutError):
        asyncio.run(_go())


# A.4 — non-existent working_dir → FileNotFoundError → spawn_failed ----------


def test_A4_nonexistent_working_dir_is_spawn_failed() -> None:
    """A non-existent working_dir is a distinct failure mode from a
    missing binary. drive() pre-checks working_dir.exists() before spawn
    and surfaces stop_reason='working_dir_missing' so operators can
    diagnose without confusing it with a PATH problem."""
    nonexistent = Path("/tmp/agentkit-nowhere-XYZ")
    assert not nonexistent.exists()
    cog = ClaudeCliCognition(working_dir=nonexistent)
    agent = Agent(name="local", cognition=cog)
    events = asyncio.run(_drain_async(cog.drive(agent, "hi", FakeCtx(), WorkingContext())))

    assert len(events) == 1
    assert events[0].type == "final"
    assert events[0].result.partial is True
    assert events[0].result.evals["stop_reason"] == "working_dir_missing"
    assert "working_dir does not exist" in events[0].result.evals["error"]


# A.5 — empty task string (real CLI rejects with exit 1) --------------------


def test_A5_empty_task_surfaces_stderr_partial() -> None:
    """The real CLI writes 'Input must be provided...' to stderr and exits 1
    when -p is empty. Mock that shape here to keep the test offline; a
    separate real_cli test would just confirm the same wire behaviour."""
    proc = _FakeProcess(
        stdout_lines=[],
        stderr=b"Error: Input must be provided either through stdin or as a prompt argument when using --print\n",
        returncode=1,
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "", FakeCtx(), WorkingContext())))

    assert len(events) == 1
    final = events[0].result
    assert final.partial is True
    assert final.evals["stop_reason"] == "cli_exit_1"
    assert "Input must be provided" in final.evals["stderr"]


# A.6 — invalid session_id — CLI validates as UUID --------------------------


def test_A6_invalid_session_id_partial_with_stderr() -> None:
    """Real CLI observed: exit=1, stderr='Error: Invalid session ID...'.
    The drive should map that to cli_exit_1 + stderr surfaced.

    The trigger is now an unknown ``resume_session_id`` rather than a malformed
    ``session_id``: a malformed one is refused at CONSTRUCTION (see
    ``test_a_non_uuid_session_id_is_refused_at_construction``), which is
    strictly better than spending a subprocess spawn to learn it. The contract
    under test here is the runtime one — a non-zero CLI exit still lands the
    terminal event with the stderr attached."""
    proc = _FakeProcess(
        stdout_lines=[],
        stderr=b"Error: Invalid session ID. Must be a valid UUID.\n",
        returncode=1,
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition(resume_session_id="no-such-session")
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "hi", FakeCtx(), WorkingContext())))

    final = events[-1].result
    assert final.partial is True
    assert final.evals["stop_reason"] == "cli_exit_1"
    assert "Invalid session ID" in final.evals["stderr"]


# A.7 — invalid model name — CLI observed to return is_error=true with
# subtype="success" and an assistant text message ---------------------------


def test_A7_invalid_model_yields_cli_reported_error() -> None:
    """Real-CLI observation (v2.1.218): passing --model claude-does-not-exist
    yields a `result` payload with is_error=true AND subtype="success", plus
    an assistant text message describing the failure. The is_error+
    subtype='success' combination is a UX cliff — we upgrade it to a
    generic 'cli_reported_error' so `partial=True` isn't paired with a
    misleading `stop_reason='success'`."""
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "sess-1"}),
            _line(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "There's an issue with the selected model.",
                            }
                        ]
                    },
                }
            ),
            _line(
                {
                    "type": "result",
                    "is_error": True,
                    "subtype": "success",  # observed real-CLI shape
                    "session_id": "sess-1",
                    "duration_ms": 1154,
                    "total_cost_usd": 0.0,
                    "usage": {},
                    "terminal_reason": "api_error",
                    "api_error_status": 404,
                    "result": "There's an issue with the selected model.",
                }
            ),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition(model="claude-does-not-exist")
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "hi", FakeCtx(), WorkingContext())))

    final = events[-1].result
    assert final.partial is True
    # Post-fix: subtype=='success' + is_error=true is upgraded to a
    # generic marker so partial=True has a coherent stop_reason.
    assert final.evals["stop_reason"] == "cli_reported_error"


# A.8 — disallowed_tools; real CLI honours restriction ----------------------


@real_cli
def test_A8_real_cli_respects_disallowed_tools() -> None:
    """allowed=(Read,), disallowed=(Bash,Grep). Ask a question that would
    normally want Bash. Verify no tool_call with a disallowed name lands on
    the stream, and we still get a clean final."""
    cog = ClaudeCliCognition(
        allowed_tools=("Read",),
        disallowed_tools=("Bash", "Grep"),
        permission_mode="bypassPermissions",
    )
    agent = Agent(name="local", cognition=cog)
    events = asyncio.run(
        _drain_async(
            agent.stream(
                "In one short sentence, name any file at /tmp. Do not run shell commands.",
                FakeCtx(),
            )
        )
    )
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1
    tool_calls = [e for e in events if e.type == "tool_call"]
    for tc in tool_calls:
        assert tc.tool_call.name not in ("Bash", "Grep")


# A.9 — multiple assistant messages across separate JSONL lines -------------


def test_A9_multiple_assistant_payloads_accumulate_in_order() -> None:
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "one "}]},
                }
            ),
            _line(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "two "}]},
                }
            ),
            _line(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "three"}]},
                }
            ),
            _line(
                {
                    "type": "result",
                    "session_id": "s",
                    "duration_ms": 5,
                    "total_cost_usd": 0.0,
                    "usage": {},
                }
            ),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "hi", FakeCtx(), WorkingContext())))

    deltas = [e.text for e in events if e.type == "message_delta"]
    assert deltas == ["one ", "two ", "three"]
    assert events[-1].result.output == "one two three"
    assert events[-1].result.partial is False


# A.10 — stderr written but exit 0 — should NOT be surfaced -----------------


def test_A10_stderr_warning_without_error_is_not_surfaced() -> None:
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ],
        stderr=b"warning: some cosmetic notice on stderr\n",
        returncode=0,
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "hi", FakeCtx(), WorkingContext())))

    final = events[-1].result
    assert final.partial is False
    assert "stderr" not in final.evals


# A.11 — subprocess ignores SIGTERM → kill path fires -----------------------


def test_A11_terminate_ignored_falls_back_to_sigkill() -> None:
    """_FakeProcess with a terminate_hook that no-ops. The drive's
    _terminate helper must fall back to kill() after ``terminate_grace_s``.
    """

    class _StubbornProcess(_FakeProcess):
        def terminate(self) -> None:  # no-op — ignore SIGTERM
            self.terminated = True
            # do NOT set _returncode → wait() will hang until we kill

        async def wait(self) -> int:
            # First call (post-terminate): sleep forever until kill flips it.
            while self._returncode is None:  # noqa: ASYNC110 — polling fake process state, not real IO
                await asyncio.sleep(0.02)
            return self._returncode

    ctx = _CancellingCtx()

    def _trip() -> None:
        ctx.trip()
        return None

    lines: list[Any] = [
        _line({"type": "system", "session_id": "s"}),
        _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}),
        _trip,
        _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "NEVER"}]}}),
    ]
    proc = _StubbornProcess(stdout_lines=lines)

    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        # Tiny grace so the test stays fast.
        cog = ClaudeCliCognition(terminate_grace_s=0.1)
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "hi", ctx, WorkingContext())))

    assert proc.terminated is True
    assert proc.killed is True, "SIGKILL fallback did not fire"
    final = events[-1].result
    assert final.partial is True
    assert final.evals["stop_reason"] == "cancelled"


# A.12 — --max-turns limit + is_error=true + subtype=error_max_turns --------


def test_A12_max_turns_error_maps_to_error_max_turns_stop_reason() -> None:
    """max_turns=1 with a task the CLI would normally split across turns:
    real CLI returns is_error=true, subtype='error_max_turns' → our
    stop_reason must reflect that."""
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "starting search..."},
                            {
                                "type": "tool_use",
                                "id": "tu-1",
                                "name": "Grep",
                                "input": {"pattern": "x"},
                            },
                        ]
                    },
                }
            ),
            _line(
                {
                    "type": "result",
                    "session_id": "s",
                    "duration_ms": 50,
                    "usage": {},
                    "is_error": True,
                    "subtype": "error_max_turns",
                }
            ),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition(max_turns=1)
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "search then summarise", FakeCtx(), WorkingContext())))

    final = events[-1].result
    assert final.partial is True
    assert final.evals["stop_reason"] == "error_max_turns"


# ============================================================================
# Goal B — integration with other agentkit components
# ============================================================================


# B.1 — Budget / MeterExceeded — NOT honoured ------------------------------


def test_B1_budget_is_ignored_by_cli_cognition() -> None:
    """The cognition subprocesses the CLI — it never touches ctx.invoker,
    so the ``meter()`` middleware never fires and Budget.charge is never
    called. Confirmed via a FakeCtx that would blow up if guard/charge
    were invoked."""
    calls: list[str] = []

    class _WatchingBudget:
        max_cost_usd = 0.001
        spent_usd = 0.0
        calls = 0

        async def guard(self, call: Any = None) -> None:  # noqa: ARG002
            calls.append("guard")

        async def charge(self, call: Any, usage: Any) -> None:  # noqa: ARG002
            calls.append("charge")

    ctx = FakeCtx()
    ctx.budget = _WatchingBudget()  # type: ignore[attr-defined]

    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line(
                {
                    "type": "result",
                    "session_id": "s",
                    "duration_ms": 1,
                    "total_cost_usd": 5.0,  # WAY over budget — cognition should ignore
                    "usage": {"input_tokens": 100, "output_tokens": 100},
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
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "hi", ctx, WorkingContext())))

    # Charged nothing to the budget — this is the GAP.
    assert calls == []
    assert events[-1].result.usage.cost_usd == 5.0
    # There's a $5 CLI cost but the run-level Budget is unaware.


# B.2 / B.3 — trace + observer are NOT touched by the cognition ------------


def test_B2_B3_no_span_no_observation_emitted_by_cognition() -> None:
    """The cognition doesn't open an ``invoke_agent`` (that's the Agent's job)
    but ALSO doesn't open a cognition-level span or emit any observations.
    Confirmed here so the gap is documented as a locked-in behaviour."""
    ctx = _RecordingObserverCtx()

    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        # Call drive() DIRECTLY so we skip Agent.stream's invoke_agent span.
        asyncio.run(_drain_async(cog.drive(agent, "hi", ctx, WorkingContext())))

    # Cognition itself opened zero spans and emitted zero observations.
    assert ctx.trace.spans == []
    assert ctx.emissions == []


def test_B2_agent_stream_still_opens_invoke_agent_span() -> None:
    """Sanity: the Agent wrapper still opens an ``invoke_agent`` span even
    though the cognition doesn't. This keeps the outer boundary observable
    for callers that go through agent.stream / agent.run."""
    ctx = FakeCtx()
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        asyncio.run(_drain_async(agent.stream("hi", ctx)))

    names = [name for (name, _kind, _attrs) in ctx.spans]
    assert "invoke_agent" in names


# B.4 — ctx.autonomy is ignored — permission_mode owns it ------------------


def test_B4_ctx_autonomy_does_not_influence_argv() -> None:
    """A ctx with autonomy='manual' must NOT rewrite --permission-mode.
    The CLI owns its permission model; agentkit's Autonomy is orthogonal."""
    ctx = FakeCtx()
    ctx.autonomy = "manual"  # would gate a ReAct tool loop

    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        cog = ClaudeCliCognition(permission_mode="acceptEdits")
        agent = Agent(name="local", cognition=cog)
        asyncio.run(_drain_async(cog.drive(agent, "hi", ctx, WorkingContext())))

    argv = spawn.await_args.args
    idx = argv.index("--permission-mode")
    assert argv[idx + 1] == "acceptEdits"
    # `--permission-mode` appears exactly once — autonomy did not rewrite it.
    assert argv.count("--permission-mode") == 1


# B.5 — end-to-end streaming: consumer receives events in order -----------


def test_B5_stream_consumer_sees_events_in_arrival_order() -> None:
    """Mimic an SSE consumer: consume agent.stream in an async for loop,
    push each event onto a queue as it arrives. Confirm order and that no
    event is buffered until final."""
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}),
            _line(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
                }
            ),
            _line(
                {
                    "type": "user",
                    "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
                }
            ),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)

        async def _consume() -> list[str]:
            received: list[str] = []
            async for ev in agent.stream("hi", FakeCtx()):
                received.append(ev.type)
                # Ensure we're getting events *incrementally* — a caller
                # awaiting `agent.stream` should be able to yield to the loop
                # between events without stalling. asyncio.sleep(0) checks that.
                await asyncio.sleep(0)
            return received

        received = asyncio.run(_consume())

    assert received == ["message_delta", "tool_call", "tool_result", "final"]


# B.6 — wrap ClaudeCliCognition agent as a tool for a parent ReAct agent ---


def test_B6_claude_cli_agent_can_be_wrapped_as_tool() -> None:
    """as_tool works because it only requires `async run(task, ctx)` — which
    ClaudeCliCognition-driven Agents have. This confirms composition works
    at the surface. (Semantic worth is another question — the parent's
    Budget/observer still won't see the child CLI's costs; see B.1.)"""
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "child said hi"}]}}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )

    async def _go() -> str:
        with patch(
            "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            child = Agent(name="child", cognition=ClaudeCliCognition())
            tool = as_tool(child, name="ask_child", description="delegate")
            # Fire the tool directly rather than driving a real ReAct loop.
            return await tool.fn({"task": "hello"}, FakeCtx())

    out = asyncio.run(_go())
    assert out == "child said hi"


# B.7 — put a ClaudeCliCognition agent inside a Workflow node --------------


def test_B7_workflow_node_with_cli_cognition_completes() -> None:
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "node output"}]}}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )

    async def _go() -> Any:
        with patch(
            "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            wf = Workflow(name="wf")
            agent = Agent(name="cli_node", cognition=ClaudeCliCognition())
            wf.agent("n1", agent)
            return await wf.run("do it", FakeCtx())

    result = asyncio.run(_go())
    # WorkflowResult carries per-node outputs on ``outputs``.
    assert result.outputs["n1"] == "node output"


# B.8 — Checkpointer / Suspended.resume() is NOT supported -----------------


def test_B8_cli_cognition_does_not_support_resume() -> None:
    """Agent.resume() explicitly restricts to ReActCognition. Prove the
    CLI-driven agent raises the expected RuntimeError."""
    agent = Agent(name="local", cognition=ClaudeCliCognition())
    with pytest.raises(RuntimeError, match="does not support resume"):
        asyncio.run(agent.resume("run-1", {}, FakeCtx()))


# B.9 — agent.memory is NEVER queried by the CLI cognition ------------------


def test_B9_memory_source_is_never_queried_by_cli_cognition() -> None:
    """The CLI runs its own context management. agentkit's memory hook is
    intentionally bypassed. Prove it: attach a memory whose query() raises
    if called. The run must complete cleanly."""

    class _ExplodingMemory:
        name = "explode"

        async def query(self, *_a: Any, **_kw: Any) -> Any:
            raise AssertionError("query() called; but the CLI cognition should never call it")

        async def write(self, *_a: Any, **_kw: Any) -> None:
            raise AssertionError("write() called; but the CLI cognition should never call it")

    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog, memory=_ExplodingMemory())
        events = asyncio.run(_drain_async(cog.drive(agent, "hi", FakeCtx(), WorkingContext())))

    assert events[-1].result.output == "ok"


# B.10 — concurrency guard under many drives -------------------------------


def test_B10_concurrent_drives_are_capped_by_max_concurrent() -> None:
    """Twelve concurrent drives with max_concurrent=2 → at most two
    subprocesses live at once. Track live count via a shared counter that
    each fake spawn increments on entry and decrements when its stdout
    iterator exhausts."""
    live = 0
    max_live = 0
    lock = asyncio.Lock()

    async def _fake_spawn(*_a: Any, **_kw: Any) -> Any:
        nonlocal live, max_live
        async with lock:
            live += 1
            max_live = max(max_live, live)

        class _Stdout:
            def __init__(self) -> None:
                self._items = [
                    _line({"type": "system", "session_id": "s"}),
                    _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
                ]
                self._i = 0

            def __aiter__(self) -> Any:
                return self

            async def __anext__(self) -> bytes:
                if self._i >= len(self._items):
                    # Decrement on exhaustion — one drive is finishing.
                    nonlocal live
                    async with lock:
                        live -= 1
                    raise StopAsyncIteration
                await asyncio.sleep(0.03)
                item = self._items[self._i]
                self._i += 1
                return item

        p = _FakeProcess(stdout_lines=[])
        p.stdout = _Stdout()  # type: ignore[assignment]
        return p

    async def _go() -> None:
        with patch(
            "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
            side_effect=_fake_spawn,
        ):
            cog = ClaudeCliCognition(
                claude_bin="claude-b10-concurrency-test",
                max_concurrent=2,
            )
            agent = Agent(name="local", cognition=cog)

            async def _one() -> None:
                async for _ in cog.drive(agent, "task", FakeCtx(), WorkingContext()):
                    pass

            await asyncio.gather(*(_one() for _ in range(12)))

    asyncio.run(_go())

    assert max_live <= 2, f"cap violated — {max_live} concurrent spawns"


# B.11 — correlation_id bridging: env var + evals field --------------------


def test_B11_correlation_id_bridges_to_env_and_evals() -> None:
    """The cognition bridges agentkit's ``ctx.correlation_id`` into the child
    subprocess as ``CLAUDE_TRACE_EXTERNAL_ID`` AND surfaces it on the
    terminal ``AgentResult.evals['external_run_id']`` so downstream
    observability can join agentkit and CLI traces on a single id."""
    ctx = FakeCtx()
    ctx.correlation_id = "run-abc-123"
    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        events = asyncio.run(_drain_async(cog.drive(agent, "hi", ctx, WorkingContext())))

    env = spawn.await_args.kwargs["env"]
    assert env.get("CLAUDE_TRACE_EXTERNAL_ID") == "run-abc-123"
    final = events[-1].result
    assert final.evals.get("external_run_id") == "run-abc-123"


# B.12 — no MetricsPort / ReplayPort interaction — document and lock in ---


def test_B12_no_metrics_or_replay_interaction() -> None:
    """The cognition doesn't drive ctx.services.replay or ctx.services.metrics
    (there's no invoker path to intercept). This test locks in that lack of
    interaction — if a future refactor adds one, this test forces a
    conscious choice."""
    ctx = FakeCtx()
    # Attach probes that would fail loudly if touched.

    class _Boom:
        def __getattr__(self, _n: str) -> Any:
            raise AssertionError("replay/metrics probed by CLI cognition")

    ctx.replay = _Boom()  # type: ignore[attr-defined]
    ctx.metrics = _Boom()  # type: ignore[attr-defined]

    proc = _FakeProcess(
        stdout_lines=[
            _line({"type": "system", "session_id": "s"}),
            _line({"type": "result", "session_id": "s", "duration_ms": 1, "usage": {}}),
        ]
    )
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="local", cognition=cog)
        # Must not raise.
        asyncio.run(_drain_async(cog.drive(agent, "hi", ctx, WorkingContext())))
