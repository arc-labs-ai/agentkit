"""Is `FakeClaudeCli` still a faithful stand-in for the `claude` binary?

WHY THIS FILE EXISTS
--------------------
Almost every CLI test in this repo runs against this double, which inverts the
usual risk. A fake that is too STRICT fails loudly and gets fixed in an hour. A
fake that is too PERMISSIVE — one that replays a stream the binary would never
emit, or reports a process state the kernel would never report — makes the real
driver's tests pass for the wrong reason, and there is nothing to notice:
``tests/source/unit/models/providers/cli/test_cli_version_matrix.py`` records three
shipped bugs that were all the same bug (a CLI moved a flag and the suite stayed
green), and it exists because the suite's ground truth had quietly stopped being
ground truth. The double is the other half of that exposure: argv fidelity is
checked against the real binaries there, and STREAM and PROCESS fidelity are
checked here, because no binary is available to check them against.

So every test below is an assertion about fidelity, not about convenience:

* exhaustion must escape as a ``BaseException`` all the way to the test body.
  A ``ScriptExhausted`` that gets caught by one of agentkit's ~20 deliberate
  ``except Exception`` sites comes back as ``AgentResult(partial=True)``, which
  is the stable, plausible, wrong answer the class was created to prevent — and
  the double built to catch a loop that does not terminate would be the thing
  hiding it;
* a ``KeyboardInterrupt`` raised while a SESSION turn polls for cancellation
  must still reach the caller. The session's re-raise is a separate line in a
  separate method from the drive path's, so the two halves fail independently:
  Ctrl-C came back once as a tidy ``AgentResult`` with
  ``evals["error"] == "KeyboardInterrupt: "`` while the interpreter kept running;
* a non-UTF-8 diagnostic line is an ordinary thing for a real binary to write
  (filenames are arbitrary bytes) and must be skipped, not fatal — it ended a
  whole run once and billed the completed work at $0.00;
* ``returncode`` must stay ``None`` while the replayed process is alive, because
  a session checks exactly that before every turn to decide whether the
  conversation still exists;
* a recording's last line may have no newline, and that is where the ``result``
  payload — cost, usage, session id — lives in a truncated capture;
* two stderr chunks written at one point in the stdout stream must come back in
  WRITE order. A bare ``sorted()`` compares whole tuples, so a tie on the
  position falls through to comparing bytes and a two-write traceback is
  reassembled alphabetically;
* ``terminate()`` must record that it stopped the child, because cancellation
  outranks the exit code in the stop-reason priority — so a double that leaked
  its process would still report ``stop_reason="cancelled"`` and the leak would
  be invisible from the result alone.

Nothing here spawns a binary or touches the network; the ``@real_cli`` tests own
that, and this file owns the part they cannot reach.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.core.testing.fakes import CliRun, FakeClaudeCli, ScriptExhausted
from agentkit.core.testing.fakes.claude_cli import CliStderr
from agentkit.models.providers.cli.claude.driver import ClaudeCliCognition

# ── stream-json payload vocabulary ───────────────────────────────────────────
# Written out rather than recorded: the interesting part of every recording
# below is the BYTES or the ORDER, never the content.


def _init(**extra: Any) -> dict[str, Any]:
    """The CLI's ``system/init`` greeting — the first line of every real run."""
    return {"type": "system", "subtype": "init", "session_id": "sess-1", **extra}


def _assistant(text: str, message_id: str = "msg-1") -> dict[str, Any]:
    """A completed assistant message, the authoritative copy of its text."""
    return {
        "type": "assistant",
        "message": {"id": message_id, "role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _result(text: str, **extra: Any) -> dict[str, Any]:
    """A turn's end-of-turn payload — the ONLY carrier of cost, usage and the
    session id. Losing it loses the bill for work that was done."""
    return {
        "type": "result",
        "subtype": "success",
        "result": text,
        "session_id": "sess-1",
        "usage": {"input_tokens": 7, "output_tokens": 11},
        "total_cost_usd": 0.0001,
        **extra,
    }


# ── stubs ────────────────────────────────────────────────────────────────────


@dataclass
class _Agent:
    """An agent-like with the attributes this cognition reads off one."""

    prompt: Any = None
    model: str | None = None
    _output_adapter: Any = None


class _StopButton:
    """A ``Ctx``-shaped stub carrying nothing but cooperative cancellation.

    The driver polls ``check_cancelled()`` once per stdout line, which is the
    only cancellation route through a replayed process — this double's reads
    never await, so ``wait_for`` cannot fire mid-stream.
    """

    def __init__(self, *, after_lines: int = 0) -> None:
        self.checks = 0
        self._after = after_lines

    def check_cancelled(self) -> None:
        self.checks += 1
        if self.checks > self._after:
            raise RuntimeError("the caller pressed stop")


class _CtrlC:
    """The same poll, but raising what Ctrl-C actually raises.

    ``KeyboardInterrupt`` is a ``BaseException``, so it escapes the driver's
    ``except Exception`` around the cancellation poll and lands on the
    terminal-event path as a FATAL exception rather than as a cancel — which is
    the distinction the re-raise exists to preserve.
    """

    def __init__(self, *, after_lines: int = 0) -> None:
        self.checks = 0
        self._after = after_lines

    def check_cancelled(self) -> None:
        self.checks += 1
        if self.checks > self._after:
            raise KeyboardInterrupt()


async def _drive(
    cognition: ClaudeCliCognition,
    *,
    agent: Any = None,
    ctx: Any = None,
    task: str = "summarise the diff",
) -> list[Any]:
    """Every event one ``drive`` produced, terminal event last."""
    return [ev async for ev in cognition.drive(agent or _Agent(), task, ctx, None)]


def _final(events: list[Any]) -> Any:
    """The one terminal result, asserting the exactly-one-``final`` contract."""
    types = [e.type for e in events]
    assert types.count("final") == 1, types
    assert types[-1] == "final", types
    return events[-1].result


# ── exhaustion is louder than the terminal-event guarantee ───────────────────


@pytest.mark.asyncio
async def test_asking_the_double_for_a_spawn_it_has_no_recording_for_reaches_the_test_body() -> None:
    """``ScriptExhausted`` is a ``BaseException`` so that it survives the
    deliberate ``except BaseException`` on the driver's terminal-event path,
    and surviving it is the whole point.

    ``drive`` promises exactly one ``final`` event on EVERY exit path, spawn
    failures included — a promise kept by catching everything. If the catch is
    allowed to be the end of the story, "the code under test spawned the CLI
    more times than the recording claims it would" arrives as a tidy
    ``AgentResult(partial=True, stop_reason="failed")``: a retry nobody asked
    for, or a loop that should have stopped, reported as a plausible answer.
    The double whose job is to catch non-termination would be the thing hiding
    it. So both halves have to hold at once — the terminal event is yielded
    AND the exception continues on afterwards.
    """
    events: list[Any] = []
    with pytest.raises(ScriptExhausted) as raised:
        async for ev in ClaudeCliCognition(spawn=FakeClaudeCli([])).drive(_Agent(), "go", None, None):
            events.append(ev)

    assert [e.type for e in events] == ["final"], "the terminal-event guarantee is not waived by a loud double"
    # The message is the entire value of raising here: "recording exhausted"
    # alone sends the reader into the double, and the defect is in the caller.
    assert "FakeClaudeCli exhausted" in str(raised.value)
    assert "repeat_last=True" in str(raised.value), "the message must carry the way out, not just the complaint"


@pytest.mark.asyncio
async def test_a_ctrl_c_during_a_session_turn_reaches_the_caller_after_the_terminal_event() -> None:
    """A session turn re-raises its fatal non-``Exception`` on its OWN line, in
    its own method, separately from the one-shot drive path.

    Which means the two halves break independently, and this is the half that
    is only reachable through a live conversation. Without it, Ctrl-C pressed
    while a session turn is waiting on the CLI comes back as a terminal event
    carrying ``evals["error"] == "KeyboardInterrupt: "``, the caller's own
    ``except KeyboardInterrupt`` never fires, and the interpreter the operator
    just tried to stop keeps running — reported, measured and fixed once
    already on the drive path. The terminal event still has to arrive first:
    the guarantee is "one ``final`` and then the truth", not one or the other.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("half an answ"), _result("half an answ")])])
    events: list[Any] = []

    with pytest.raises(KeyboardInterrupt):
        async with ClaudeCliCognition(spawn=cli).session() as chat:
            async for ev in chat.turn("one", ctx=_CtrlC(after_lines=1)):
                events.append(ev)

    assert events[-1].type == "final", [e.type for e in events]
    assert events[-1].result.evals["error"].startswith("KeyboardInterrupt")


# ── bytes a real binary writes and this one must survive ─────────────────────


@pytest.mark.asyncio
async def test_a_non_utf8_diagnostic_line_is_skipped_so_the_finished_work_is_still_billed() -> None:
    """``json.loads`` on **bytes** sniffs the encoding before parsing, so a line
    opening with a UTF-8 BOM whose tail is latin-1 raises ``UnicodeDecodeError``
    — a ``ValueError``, but NOT a ``JSONDecodeError``.

    A parser that catches only the narrow one lets that escape the line-skipping
    contract entirely and hit the driver's ``except BaseException``, which ends
    the RUN: no output, ``stop_reason="parse_failed"``, and — because the reader
    never got as far as the terminal payload — a completed run reporting $0.00
    to the budget. Reachable without anything exotic: both binaries print human
    diagnostics on the same streams they speak JSON on, and a filename is
    arbitrary bytes, so a warning naming a latin-1 path is this line. The double
    exists to be able to produce it at all.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    # A UTF-8 BOM, then a byte that is not valid UTF-8 — what a
                    # warning naming a latin-1 path looks like on the wire.
                    b"\xef\xbb\xbfwarning: skipping caf\xe9.txt\n",
                    _assistant("done"),
                    _result("done"),
                ]
            )
        ]
    )
    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.output == "done"
    assert result.partial is False
    assert result.stop_reason == "complete"
    assert result.usage.cost_usd == pytest.approx(0.0001), "the run finished, so the bill for it must survive"


@pytest.mark.asyncio
async def test_a_recordings_last_line_without_a_newline_still_delivers_its_result() -> None:
    """A ``StreamReader`` yields a trailing fragment with no separator at EOF,
    and the double splits lines the same way — deliberately, because that
    fragment is where a truncated capture keeps its ``result`` payload.

    Dropping it is silent in the worst way: the stream still ends cleanly, the
    process still exits 0, so the run reports ``partial=False`` and
    ``stop_reason="complete"`` while cost, usage and the session id have all
    vanished. A recording that loses the only payload carrying money is a
    recording that makes a metering test pass by having nothing to meter.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    _assistant("done"),
                    # Verbatim, no trailing newline: the shape of a capture cut
                    # at the last byte the binary wrote.
                    json.dumps(_result("done"), separators=(",", ":")),
                ]
            )
        ]
    )
    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.usage.cost_usd == pytest.approx(0.0001)
    assert result.usage.output_tokens == 11
    assert result.evals["session_id"] == "sess-1"
    assert result.partial is False


@pytest.mark.asyncio
async def test_two_stderr_chunks_written_at_one_point_in_the_stream_keep_their_write_order() -> None:
    """A multi-line traceback reaches stderr as SEVERAL writes at one position
    in the stdout stream, and stderr is the only channel that explains a bare
    ``cli_exit_1`` to an operator.

    Queuing those chunks with a bare ``sorted()`` compares the whole
    ``(position, bytes)`` tuple, so a tie on the position falls through to
    comparing the BYTES — and a traceback, whose continuation lines are
    indented and therefore sort before the header, is reassembled in
    alphabetical order. The diagnostic is still all there, still looks like a
    traceback, and says the wrong thing about where the failure came from.
    Sorting on the position alone leans on Python's stable sort, which is
    exactly "ties keep their write order".
    """
    header = b"Traceback (most recent call last):\n"
    frame = b'  File "cli.js", line 1\n'
    assert frame < header, "the fixture only means something if byte order disagrees with write order"

    cli = FakeClaudeCli([CliRun.of([CliStderr(header), CliStderr(frame), _result("done")], returncode=1)])
    proc = await cli("claude", "-p")
    assert proc.stdout is not None and proc.stderr is not None
    async for _line in proc.stdout:
        pass

    assert await proc.stderr.read() == header + frame


# ── the process surface, which a session reads as "are you still there?" ─────


@pytest.mark.asyncio
async def test_a_replayed_process_reports_no_exit_code_until_it_is_waited_on() -> None:
    """``returncode is None`` means "still running" to everything that reads a
    ``Process``, and a persistent session reads it before every single turn to
    decide whether the conversation still exists.

    A double that published its recorded exit code from the moment it was
    spawned would therefore be dead on arrival: turn one would work (nobody has
    asked yet), and turn two onwards would refuse with ``session_closed``
    against a perfectly healthy recording. The code has to come from the
    invocation record, which ``wait()``/``terminate()``/``kill()`` fill in, and
    not from the immutable recording.
    """
    cli = FakeClaudeCli([CliRun(stdout=b"", returncode=7)])
    proc = await cli("claude", "-p")

    assert proc.returncode is None, "a spawned-but-unwaited process is alive"
    assert proc.stdout is not None
    async for _line in proc.stdout:
        pass
    assert proc.returncode is None, "reaching EOF on stdout is not the same event as the process exiting"

    assert await proc.wait() == 7
    assert proc.returncode == 7, "and once reaped, the code is the recording's"


@pytest.mark.asyncio
async def test_a_session_reaches_its_second_turn_because_the_double_stays_alive_between_them() -> None:
    """The consequence of the rule above, through the code that depends on it.

    One recording serves many turns of a session, and the only thing standing
    between turn two and ``session_closed`` is the double declining to claim it
    has exited. Asserted end to end rather than only on the process surface
    because this is the failure a reader would actually hit: every multi-turn
    test in the suite would collapse to "turn one works", and each would still
    be green about turn one.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    _assistant("first", "msg-1"),
                    _result("first"),
                    _init(),
                    _assistant("second", "msg-2"),
                    _result("second"),
                ]
            )
        ]
    )
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        one = [ev async for ev in chat.turn("one")]
        two = [ev async for ev in chat.turn("two")]

    assert _final(one).output == "first"
    assert _final(two).output == "second", "turn two got its own answer, not a refusal from a double playing dead"
    assert cli.spawns == 1, "a session is ONE spawn; a second would mean the conversation context was thrown away"


@pytest.mark.asyncio
async def test_a_cancelled_run_leaves_the_double_recorded_as_having_been_stopped() -> None:
    """Cancellation outranks the exit code in the stop-reason priority, and
    deliberately so: a cancel that races a crash is still a cancel.

    The cost of that ordering is that ``stop_reason="cancelled"`` is reported
    whether or not the child was actually stopped — so the result alone cannot
    tell a clean cancel from a leaked ``claude`` holding whatever it forked.
    The invocation record is the only witness, which makes a double whose
    ``terminate()`` does not record the signal worse than no double: every
    cancellation test in the suite would pass while the production path leaked
    a process tree. ``killed`` too, because the SIGKILL sweep is the half that
    reaps what the CLI forked.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("a"), _assistant("b"), _result("done")])])
    result = _final(await _drive(ClaudeCliCognition(spawn=cli), ctx=_StopButton(after_lines=1)))

    assert result.evals["stop_reason"] == "cancelled"
    assert cli.invocations[0].terminated is True, "the result says cancelled; only this says the child was stopped"
    assert cli.invocations[0].killed is True, "the sweep that reaps the grandchildren has to have run too"
