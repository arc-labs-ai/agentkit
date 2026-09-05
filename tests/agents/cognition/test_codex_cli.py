"""The base contract of :class:`CodexCliCognition`.

The subprocess is replaced at the ``spawn=`` seam with
:class:`~agentkit.testing.FakeCodexCli`, so everything below it — line parsing,
payload→event mapping, ``_TurnState.fold``, the stop-reason priority, the cost
computation, the meter charge — runs exactly as a real run runs it. A double
that returned finished ``AgentResult``s would cover none of those, and all six
are where a cognition of this shape puts its bugs.

What is asserted here is the contract every other file in this set relies on:
one and only one ``final`` event on every path, the event vocabulary a consumer
sees, and the stop reason each failure shape produces. The real binary is
exercised in ``test_codex_cli_integration.py`` behind a PATH check.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CodexCliCognition
from agentkit.context import WorkingContext
from agentkit.kernel.types import StreamEvent
from agentkit.testing.fakes import CliRun, CliStderr, FakeCodexCli, codex_turn
from agentkit.testing.fakes.ctx import FakeCtx

# ─────────────────────────────────────────────────────────────────────────────
# helpers shared with the rest of the codex test files
# ─────────────────────────────────────────────────────────────────────────────


async def drive(
    cog: CodexCliCognition,
    *,
    ctx: Any = None,
    task: str = "summarise the repo",
    agent: Agent | None = None,
) -> list[StreamEvent]:
    """Drain one ``drive()`` into a list. Every assertion in these files starts
    here, so the terminal-event guarantee is exercised by every single test
    rather than by the one that names it."""
    agent = agent or Agent(name="local", cognition=cog)
    return [ev async for ev in cog.drive(agent, task, ctx or FakeCtx(), WorkingContext())]


def final_of(events: list[StreamEvent]) -> Any:
    """The one terminal result, asserting there is exactly one."""
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1, f"expected exactly one final event, got {len(finals)}"
    return finals[0].result


def shape(events: list[StreamEvent]) -> list[tuple[str, str]]:
    """The event stream reduced to what a consumer actually sees.

    ``(type, payload)`` pairs, so two runs can be compared without comparing
    object identities and a test that cares about ORDER can say so in one line.
    """
    out: list[tuple[str, str]] = []
    for ev in events:
        if ev.type == "final":
            out.append(("final", str(ev.result.stop_reason)))
        elif ev.tool_call is not None and ev.type == "tool_call":
            out.append(("tool_call", f"{ev.tool_call.name}{sorted(ev.tool_call.arguments)}"))
        elif ev.tool_call is not None:
            out.append(("tool_result", ev.tool_result or ""))
        else:
            out.append((ev.type, ev.text))
    return out


class CancellingCtx(FakeCtx):
    """FakeCtx whose ``check_cancelled`` raises after ``trip()``.

    The only cancellation route that works through the double: reading a line
    from a replayed stdout never awaits, so nothing else gets a turn and
    ``asyncio.wait_for`` cannot fire mid-stream. The cognition polls
    ``check_cancelled`` once per line, which does.
    """

    def __init__(self, *, after: int = 0) -> None:
        super().__init__()
        self._checks = 0
        self._after = after

    def trip(self) -> None:
        self._after = 0

    def check_cancelled(self) -> None:
        self._checks += 1
        if self._checks > self._after:
            raise RuntimeError("run cancelled")


# ─────────────────────────────────────────────────────────────────────────────
# 1. the happy path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_complete_turn_produces_text_usage_and_a_thread_id() -> None:
    """The whole contract in one assertion set: the answer is the accumulated
    assistant text, the tokens come off ``turn.completed``, and the thread id
    off ``thread.started`` — which is the value a caller passes back as
    ``resume_session_id`` and therefore the one field a run is useless without.
    """
    cli = FakeCodexCli.script(
        codex_turn(text="Nine files, three of them tests.", usage=(1200, 1000, 40), duration_ms=812)
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))

    assert result.output == "Nine files, three of them tests."
    assert result.partial is False
    assert result.stop_reason == "complete"
    assert result.evals["session_id"] == "0199a213-81c0-7800-8aa1-bbab2a035a53"
    assert result.evals["cli_duration_ms"] == 812
    assert result.evals["cli_return_code"] == 0
    assert "stop_reason" not in result.evals


@pytest.mark.asyncio
async def test_the_consumer_sees_deltas_calls_and_results_in_stream_order() -> None:
    """Order matters to a UI: the reasoning arrives before the command that it
    justified, the command's result after the command, and the answer last."""
    cli = FakeCodexCli.script(
        codex_turn(
            reasoning="Scanning for tests",
            items=[
                {
                    "id": "c1",
                    "type": "command_execution",
                    "command": "bash -lc 'ls tests'",
                    "aggregated_output": "a.py\nb.py\n",
                    "exit_code": 0,
                }
            ],
            text="Two test files.",
            usage=(10, 0, 5),
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))

    assert shape(events) == [
        ("message_delta", "Scanning for tests"),
        ("tool_call", "shell['command']"),
        ("tool_result", "a.py\nb.py\n"),
        ("message_delta", "Two test files."),
        ("final", "complete"),
    ]


@pytest.mark.asyncio
async def test_reasoning_is_folded_into_thinking_and_not_into_the_answer() -> None:
    """``output`` is the response; the reasoning is how it got there. A caller
    that concatenated both would return the model's scratchpad to its user."""
    cli = FakeCodexCli.script(
        codex_turn(reasoning="**Weighing options**", text="Use a dict.", usage=(10, 0, 5))
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))

    assert result.output == "Use a dict."
    assert result.evals["thinking"] == "**Weighing options**"


@pytest.mark.asyncio
async def test_several_assistant_messages_concatenate_in_order() -> None:
    """A turn can produce more than one ``agent_message``. The answer is all of
    them, joined, and the ``_ItemLog`` suffix bookkeeping must not make the
    second one inherit the first's offset."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "First. "}},
            {"type": "item.completed", "item": {"id": "m2", "type": "agent_message", "text": "Second."}},
            {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}},
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.output == "First. Second."


# ─────────────────────────────────────────────────────────────────────────────
# 2. tool items
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_started_then_completed_command_is_one_call_and_one_result() -> None:
    """The pairing case. A consumer counting ``tool_call`` events must see ONE
    for an item that arrives twice — the started/completed pair is one command,
    and double-counting it is what an observer's step count would show."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {
                "type": "item.started",
                "item": {
                    "id": "c1",
                    "type": "command_execution",
                    "command": "bash -lc ls",
                    "aggregated_output": "",
                    "exit_code": None,
                    "status": "in_progress",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "c1",
                    "type": "command_execution",
                    "command": "bash -lc ls",
                    "aggregated_output": "docs\nsrc\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    events = await drive(CodexCliCognition(spawn=cli))

    assert [e.type for e in events] == ["tool_call", "tool_result", "final"]
    assert events[0].tool_call.name == "shell"
    assert events[0].tool_call.arguments == {"command": "bash -lc ls"}
    assert events[1].tool_result == "docs\nsrc\n"


@pytest.mark.asyncio
async def test_a_failed_command_carries_its_exit_code_into_the_result() -> None:
    """A command that fails silently has an empty output and a non-zero code.
    Without the code appended the ``tool_result`` is the empty string, and an
    observer reading it cannot tell a no-op from a failure."""
    cli = FakeCodexCli.script(
        codex_turn(
            items=[{"type": "command_execution", "command": "false", "aggregated_output": "", "exit_code": 1}],
            text="that command failed",
            usage=(1, 0, 1),
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    results = [e.tool_result for e in events if e.type == "tool_result"]
    assert results == ["[exit 1]"]


@pytest.mark.asyncio
async def test_a_successful_command_does_not_have_its_exit_code_appended() -> None:
    """Zero is noise on every single command, which is how a useful marker
    becomes one nobody reads."""
    cli = FakeCodexCli.script(
        codex_turn(
            items=[{"type": "command_execution", "command": "true", "aggregated_output": "ok\n", "exit_code": 0}],
            usage=(1, 0, 1),
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert [e.tool_result for e in events if e.type == "tool_result"] == ["ok\n"]


@pytest.mark.asyncio
async def test_a_file_change_reports_the_paths_it_touched() -> None:
    """``file_change`` only ever completes, so its call event has to be emitted
    by the completion — an item type that never 'starts' must not silently
    produce a result with no call."""
    cli = FakeCodexCli.script(
        codex_turn(
            items=[
                {
                    "type": "file_change",
                    "changes": [
                        {"path": "docs/a.md", "kind": "add"},
                        {"path": "src/b.py", "kind": "update"},
                    ],
                    "status": "completed",
                }
            ],
            usage=(1, 0, 1),
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    calls = [e for e in events if e.type == "tool_call"]
    results = [e for e in events if e.type == "tool_result"]
    assert len(calls) == 1 and calls[0].tool_call.name == "apply_patch"
    assert results[0].tool_result == "add docs/a.md\nupdate src/b.py"


@pytest.mark.asyncio
async def test_an_mcp_tool_call_uses_the_clis_own_qualified_name() -> None:
    """``mcp__<server>__<tool>``, so a name in an agentkit audit record matches
    the name in a Codex transcript and a grep finds both."""
    cli = FakeCodexCli.script(
        codex_turn(
            items=[
                {
                    "type": "mcp_tool_call",
                    "server": "engine",
                    "tool": "search",
                    "arguments": {"q": "orders"},
                    "result": {"content": [{"type": "text", "text": "3 matches"}]},
                }
            ],
            usage=(1, 0, 1),
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    call = next(e for e in events if e.type == "tool_call")
    result = next(e for e in events if e.type == "tool_result")
    assert call.tool_call.name == "mcp__engine__search"
    assert call.tool_call.arguments == {"q": "orders"}
    assert result.tool_result == "3 matches"


@pytest.mark.asyncio
async def test_a_plan_update_is_a_step_and_a_web_search_is_a_tool_call() -> None:
    """Two item types a consumer renders differently: a plan is progress, a
    search is an action with a result."""
    cli = FakeCodexCli.script(
        codex_turn(
            items=[
                {"type": "todo_list", "items": [{"text": "scan", "completed": True}, {"text": "write"}]},
                {"type": "web_search", "query": "codex exec json"},
            ],
            usage=(1, 0, 1),
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert ("step", "plan:1/2") in shape(events)
    call = next(e for e in events if e.type == "tool_call")
    assert call.tool_call.name == "web_search"
    assert call.tool_call.arguments == {"query": "codex exec json"}


@pytest.mark.asyncio
async def test_an_item_level_error_is_recorded_but_does_not_fail_the_run() -> None:
    """"command output truncated" is a diagnostic, not a failed run. Promoting
    it would make every large build a partial result."""
    cli = FakeCodexCli.script(
        codex_turn(
            items=[{"type": "error", "message": "command output truncated"}],
            text="done anyway",
            usage=(1, 0, 1),
        )
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.partial is False
    assert result.stop_reason == "complete"
    assert result.evals["cli_errors"] == ["command output truncated"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. failure shapes — every one lands exactly one final event
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_missing_binary_is_reported_as_data_not_raised() -> None:
    cog = CodexCliCognition(codex_bin="/tmp/definitely-not-here-XYZ")
    result = final_of(await drive(cog))
    assert result.partial is True
    assert result.stop_reason == "failed"
    assert result.evals["stop_reason"] == "spawn_failed"
    assert "FileNotFoundError" in result.evals["error"]


@pytest.mark.asyncio
async def test_a_missing_working_dir_is_distinguished_from_a_missing_binary() -> None:
    """Both would be ``FileNotFoundError`` from ``create_subprocess_exec``, and
    the fix differs: install the CLI versus create the directory."""
    from pathlib import Path

    cog = CodexCliCognition(working_dir=Path("/tmp/definitely-not-a-dir-XYZ"), spawn=FakeCodexCli())
    result = final_of(await drive(cog))
    assert result.evals["stop_reason"] == "working_dir_missing"
    assert result.stop_reason == "failed"


@pytest.mark.asyncio
async def test_a_turn_failed_payload_is_partial_and_names_itself() -> None:
    """The CLI can report a failed turn while exiting 0, so the exit code alone
    would call this a success."""
    cli = FakeCodexCli.script(codex_turn(failed="model response stream ended unexpectedly"))
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.partial is True
    assert result.stop_reason == "failed"
    assert result.evals["stop_reason"] == "turn_failed"
    assert result.evals["cli_errors"] == ["model response stream ended unexpectedly"]


@pytest.mark.asyncio
async def test_a_stream_level_error_is_a_different_stop_reason_than_a_failed_turn() -> None:
    """A broken pipe and a failed model run are different problems with
    different fixes; collapsing them loses the only signal that says which."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "error", "message": "stream error: broken pipe"},
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.evals["stop_reason"] == "cli_reported_error"


@pytest.mark.asyncio
async def test_a_non_zero_exit_keeps_the_text_that_arrived_and_names_the_code() -> None:
    cli = FakeCodexCli(
        [
            CliRun.of(
                [
                    {"type": "thread.started", "thread_id": "t"},
                    {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "half an "}},
                ],
                stderr=b"codex: killed",
                returncode=137,
            )
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.output == "half an "
    assert result.partial is True
    assert result.evals["stop_reason"] == "cli_exit_137"
    assert result.evals["cli_return_code"] == 137
    assert result.evals["stderr"] == "codex: killed"


@pytest.mark.asyncio
async def test_stderr_is_only_attached_to_a_failed_run() -> None:
    """A successful run's progress chatter is not a diagnostic, and putting it
    in ``evals`` would make every result carry a paragraph nobody reads."""
    cli = FakeCodexCli([CliRun.of(codex_turn(text="fine", usage=(1, 0, 1)), stderr=b"working...")])
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert "stderr" not in result.evals


@pytest.mark.asyncio
async def test_interleaved_stderr_written_mid_run_survives_to_the_end() -> None:
    """The cognition reads stderr once, after stdout hits EOF. A diagnostic
    emitted in the middle of a long run has to still be there — it is the only
    channel that explains a bare ``cli_exit_1`` to an operator."""
    cli = FakeCodexCli(
        [
            CliRun.of(
                [
                    {"type": "thread.started", "thread_id": "t"},
                    CliStderr(b"warn: sandbox denied a write\n"),
                    {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "x"}},
                ],
                returncode=1,
            )
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.evals["stderr"] == "warn: sandbox denied a write"


@pytest.mark.asyncio
async def test_a_truncated_final_line_is_skipped_rather_than_ending_the_run() -> None:
    """A process killed mid-JSON-object. The lines before it are real and must
    survive; the fragment is not JSON and must not raise — AND the run must not
    claim to have finished.

    The last part is the reversal. This used to assert ``partial is False``
    ("exit 0, nothing said otherwise"), so a killed process handed back a
    half-written answer labelled complete. Nothing said otherwise because
    nothing was LOOKING: a successful ``turn.completed`` leaves ``stop_reason``
    at ``None``, exactly like a stream that stopped early, so the two were
    indistinguishable until ``saw_terminal`` made the difference explicit."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "kept"}},
            '{"type":"turn.comp',
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.output == "kept", "the complete lines before the fragment are real"
    assert result.partial is True
    assert result.evals["stop_reason"] == "malformed_output"


@pytest.mark.asyncio
async def test_blank_and_non_json_lines_are_skipped() -> None:
    """Both binaries write human diagnostics on the same streams they write
    JSON on. One of them must not end a run."""
    cli = FakeCodexCli.script(
        [
            "\n",
            "warn: could not read ~/.codex/config.toml\n",
            {"type": "thread.started", "thread_id": "t"},
            b"\xff\xfe not utf-8 at all\n",
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "still here"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.output == "still here"
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
async def test_an_unknown_payload_type_is_ignored() -> None:
    """Forward-compat is "don't crash". The CLI adds event types over time and
    a service pinned to an older agentkit must keep working."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "some.future.event", "payload": {"nested": [1, 2, 3]}},
            {"type": "item.completed", "item": {"id": "m", "type": "future_item", "text": "?"}},
            {"type": "item.completed", "item": {"id": "n", "type": "agent_message", "text": "ok"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_a_run_with_no_turn_completed_is_reported_as_malformed_output() -> None:
    """The moment the old test anticipated, arrived.

    It pinned "clean but empty success" and said: *the moment this cognition
    starts calling that a failure, THIS is the test that says so.* It is now a
    failure, because reporting it as a success was the worst shape available —
    ``partial=False`` on a fragment, no usage, no duration, and nothing in the
    result to suggest the CLI had not finished writing. The matching Claude
    test makes the identical change, which is why they still agree."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "answered"}},
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.output == "answered", "whatever did arrive is still real"
    assert result.partial is True
    assert result.usage.total_tokens == 0
    assert result.evals["stop_reason"] == "malformed_output"


@pytest.mark.asyncio
async def test_an_empty_recording_still_produces_a_terminal_event() -> None:
    cli = FakeCodexCli.script([])
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.output == ""
    assert result.evals["session_id"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. cancellation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_tripped_cancel_token_terminates_the_process_and_reports_it() -> None:
    cli = FakeCodexCli.script(codex_turn(text="never read", usage=(1, 0, 1)))
    events = await drive(CodexCliCognition(spawn=cli), ctx=CancellingCtx(after=1))
    result = final_of(events)

    assert result.partial is True
    assert result.stop_reason == "terminated"
    assert result.evals["stop_reason"] == "cancelled"
    assert cli.invocations[0].terminated is True
    # SIGTERM's native encoding, reported verbatim rather than normalised —
    # an operator reading -15 knows the process was signalled, not that it
    # chose to exit.
    assert result.evals["cli_return_code"] == -15


@pytest.mark.asyncio
async def test_an_injected_cancel_is_re_raised_after_the_terminal_event() -> None:
    """Suppressing ``CancelledError`` would make a caller's ``wait_for``
    timeout silently return a partial result instead of raising."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="local", cognition=cog)
    seen: list[StreamEvent] = []

    async def _go() -> None:
        async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext()):
            seen.append(ev)
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _go()
    assert len(seen) == 1


class SuspendingCodexCli(FakeCodexCli):
    """A double whose spawn actually awaits.

    Reading a line from the offline double never suspends, so nothing else on
    the loop gets a turn and ``asyncio.wait_for`` cannot fire mid-stream — the
    limitation the double documents. A spawn that sleeps restores the one
    property the cancellation tests below need: a suspension point the
    scheduler can cancel at.
    """

    def __init__(self, *, delay_s: float = 5.0) -> None:
        super().__init__([CliRun.of(codex_turn(text="never arrives", usage=(1, 0, 1)))])
        self._delay_s = delay_s

    async def __call__(self, *argv: str, **kw: Any):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self._delay_s)
        return await super().__call__(*argv, **kw)


@pytest.mark.asyncio
async def test_a_wait_for_timeout_around_a_drive_raises_rather_than_returning_partial() -> None:
    """The reason ``CancelledError`` is re-raised after the terminal event.
    Suppressing it would make a caller's timeout silently return a partial
    result — a run that looks like it answered badly rather than one that ran
    out of time."""
    cog = CodexCliCognition(spawn=SuspendingCodexCli(delay_s=5.0), terminate_grace_s=0.1)
    agent = Agent(name="local", cognition=cog)

    async def _consume() -> list[StreamEvent]:
        return [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())]

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_consume(), timeout=0.05)


@pytest.mark.asyncio
async def test_a_cancelled_task_still_delivers_one_terminal_event_first() -> None:
    """Both halves of the contract at once: the consumer sees the ``final``, and
    THEN the cancel propagates. Asserted through a consumer that keeps its own
    list, because a ``wait_for`` discards whatever the coroutine had built."""
    cog = CodexCliCognition(spawn=SuspendingCodexCli(delay_s=5.0), terminate_grace_s=0.1)
    agent = Agent(name="local", cognition=cog)
    seen: list[StreamEvent] = []

    async def _consume() -> None:
        async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext()):
            seen.append(ev)

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [e.type for e in seen] == ["final"]
    assert seen[0].result.evals["stop_reason"] == "cancelled"
    assert seen[0].result.stop_reason == "terminated"


@pytest.mark.asyncio
async def test_a_keyboard_interrupt_reaches_the_caller_after_the_final_event() -> None:
    """The outer ``except BaseException`` widens past ``Exception`` so a Ctrl-C
    still produces the terminal event. It must then PROPAGATE — the identical
    bug in the Claude cognition swallowed it, returned a tidy failed result,
    and the caller's own handler never fired."""

    async def boom(*_a: Any, **_k: Any) -> Any:
        raise KeyboardInterrupt()

    cog = CodexCliCognition(spawn=boom)
    agent = Agent(name="local", cognition=cog)
    seen: list[StreamEvent] = []

    with pytest.raises(KeyboardInterrupt):
        async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext()):
            seen.append(ev)

    assert len(seen) == 1
    assert seen[0].result.evals["error"].startswith("KeyboardInterrupt")


@pytest.mark.asyncio
async def test_an_ordinary_exception_is_reported_as_data() -> None:
    """The other side of the same line: a run's failure modes are all
    ``Exception``s and every one arrives as a ``final`` event and nothing
    more."""

    async def boom(*_a: Any, **_k: Any) -> Any:
        raise PermissionError("no exec bit")

    result = final_of(await drive(CodexCliCognition(spawn=boom)))
    assert result.evals["stop_reason"] == "spawn_failed"
    assert "PermissionError: no exec bit" == result.evals["error"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. the spawn seam itself
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_default_spawn_is_the_real_create_subprocess_exec() -> None:
    """``spawn=None`` must not quietly resolve to anything else. The seam
    exists for tests; production takes the real path, and this is the assertion
    that keeps that true."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    with patch("agentkit.agents.cognition.codex_cli.asyncio.create_subprocess_exec", new=cli):
        result = final_of(await drive(CodexCliCognition()))
    assert cli.spawns == 1
    assert result.output == "x"


@pytest.mark.asyncio
async def test_the_seam_is_per_instance_so_two_cognitions_do_not_share_it() -> None:
    """A ``patch`` of ``create_subprocess_exec`` replaces it process-wide; an
    injected callable is scoped to one cognition, which is what lets a faked
    run and a real one coexist."""
    a = FakeCodexCli.script(codex_turn(text="from a", usage=(1, 0, 1)))
    b = FakeCodexCli.script(codex_turn(text="from b", usage=(1, 0, 1)))
    ra = final_of(await drive(CodexCliCognition(spawn=a)))
    rb = final_of(await drive(CodexCliCognition(spawn=b)))
    assert (ra.output, rb.output) == ("from a", "from b")
    assert a.spawns == b.spawns == 1
