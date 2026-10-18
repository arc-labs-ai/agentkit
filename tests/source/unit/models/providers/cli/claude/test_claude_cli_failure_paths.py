"""What `ClaudeCliCognition` reports when the run did NOT work.

WHY THIS FILE EXISTS
--------------------
``drive`` promises exactly one terminal ``final`` event on every exit path and a
``evals["stop_reason"]`` that names the failure mode. The happy path proves
neither: a cognition that reported ``complete`` for everything would pass every
test about an answer. So each test below drives one way a run can be lost and
pins the DECISION the driver makes about it — because the decisions are ranked,
and the ranking is the part that is wrong silently:

* **a truncated stream must never read as a finished answer.** Measured before
  ``malformed_output`` existed: stdout that stopped mid-object reported
  ``stop_reason="complete"``, ``partial=False``, and handed back half an answer
  as a finished one. That is the worst shape a failure can take, because there
  is nothing for a caller to branch on;
* a missing ``working_dir`` and a missing binary both raise ``FileNotFoundError``
  from ``create_subprocess_exec`` — one distinction, two completely different
  fixes;
* a death by signal must outrank a terminal payload, and a terminal payload must
  outrank a positive exit code, because claude 2.1.251 publishes
  ``error_max_turns`` and THEN exits 1 — replacing that with ``cli_exit_1``
  discards the only actionable diagnosis;
* a crossed liveness bound must outrank the parse error that cutting the stream
  mid-line produces, or a hang reports ``parse_failed``;
* a cancel must terminate the process and its descendants, and must still
  propagate so the caller's ``wait_for`` raises ``TimeoutError``;
* a ``KeyboardInterrupt`` must arrive as a terminal event AND keep propagating.

``FakeClaudeCli`` supplies the bytes and nothing else, so ``_parse_line``,
``_events_from_payload``, ``_TurnState.fold``, the stop-reason priority and the
meter charge all run exactly as a real run runs them. Where the shape is not
expressible as a recording — a stdout that never speaks, a reader whose buffer
overran, a cleanup that throws — there is a hand-written process double, because
those are the branches a recording cannot reach.
"""

from __future__ import annotations

import asyncio
import contextlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agentkit.agents.prompts.prompt import Prompt
from agentkit.core.contracts.types import CostSource, StreamEvent
from agentkit.core.testing.fakes import CliRun, FakeClaudeCli
from agentkit.models.providers.cli._common import CliTimeouts
from agentkit.models.providers.cli.claude import driver as cc
from agentkit.models.providers.cli.claude.driver import ClaudeCliCognition

# ── the shapes the binary emits ──────────────────────────────────────────────


def _init(**extra: Any) -> dict[str, Any]:
    """``system/init`` — the CLI greeting itself. NOT turn content, which is why
    it cannot satisfy the ``first_event`` bound."""
    return {"type": "system", "subtype": "init", "session_id": "sess-1", **extra}


def _assistant(text: str, message_id: str = "msg-1") -> dict[str, Any]:
    """A completed assistant message — real answer text, folded into output."""
    return {
        "type": "assistant",
        "message": {"id": message_id, "role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _result(**extra: Any) -> dict[str, Any]:
    """The CLI's own end-of-turn payload. Reaching it is what separates "the run
    finished" from "the stream stopped"."""
    return {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "usage": {"input_tokens": 5, "output_tokens": 9},
        "total_cost_usd": 0.25,
        **extra,
    }


# ── stubs ────────────────────────────────────────────────────────────────────


class _Adapter:
    """The output-schema surface ``_resolve_json_schema`` reads off an agent."""

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

    def validate(self, value: Any) -> Any:
        return ("coerced", value)


@dataclass
class _Agent:
    """An agent-like with the two attributes this cognition reads."""

    prompt: Any = None
    model: str | None = None
    _output_adapter: Any = None


class _StopButton:
    """A ``Ctx``-shaped stub carrying nothing but cooperative cancellation.

    The driver polls ``check_cancelled()`` once per stdout line, which is the
    only cancellation route through a replayed process — ``FakeClaudeCli``'s
    reads never await, so ``wait_for`` cannot fire mid-stream.
    """

    def __init__(self, *, after_lines: int = 0) -> None:
        self.checks = 0
        self._after = after_lines

    def check_cancelled(self) -> None:
        self.checks += 1
        if self.checks > self._after:
            raise RuntimeError("the caller pressed stop")


def retry() -> Any:
    """A raw ``(call, next)`` middleware, in the shape the real ones have.

    ``memoize()``, ``retry()`` and ``tracing()`` all return a closure whose
    ``__name__`` is literally ``mw``; the factory a caller typed survives only on
    ``__qualname__``. Defined at module level so that qualname is
    ``retry.<locals>.mw`` and not nested inside a test's.
    """

    def mw(call: Any, nxt: Any) -> Any:  # pragma: no cover - never invoked
        return nxt(call)

    return mw


def egress() -> Any:
    """A second factory, so the warning can be checked for chain ORDER."""

    def mw(call: Any, nxt: Any) -> Any:  # pragma: no cover - never invoked
        return nxt(call)

    return mw


class _SpentBudget:
    def remaining(self) -> float:
        return 0.0


@dataclass
class _ExhaustedCtx:
    """A context whose run budget has no headroom left."""

    budget: _SpentBudget = field(default_factory=_SpentBudget)

    def check_cancelled(self) -> None:
        return None


class _RecordingMeter:
    def __init__(self) -> None:
        self.charges: list[tuple[float | None, Any, int]] = []

    async def charge(self, _call: Any, usage: Any) -> None:
        self.charges.append((usage.cost_usd, usage.cost_source, usage.input_tokens))


class _MeteredCtx:
    def __init__(self, meter: _RecordingMeter) -> None:
        self.all_meters = [meter]
        self.actor_budget = None

    def check_cancelled(self) -> None:
        return None


# ── a process double, for the shapes no recording can express ────────────────


class _Stderr:
    def __init__(self, data: bytes = b"") -> None:
        self._buf = bytearray(data)

    async def read(self, n: int = -1) -> bytes:
        del n
        out, self._buf = bytes(self._buf), bytearray()
        return out


class _Stdin:
    def write(self, data: bytes) -> None:
        self.written = getattr(self, "written", b"") + data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _NeverSpeaks:
    """A stdout that is open and silent — the measured shape of ``claude -p``
    with an invalid credential: no stdout, no stderr, no exit for 45+ seconds."""

    def __aiter__(self) -> _NeverSpeaks:
        return self

    async def __anext__(self) -> bytes:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _OverrunsTheReader:
    """A stdout whose first line is bigger than the reader will assemble.

    ``StreamReader.readline`` reports that as a bare ``ValueError``, which is
    why there is a named exception for it.
    """

    def __aiter__(self) -> _OverrunsTheReader:
        return self

    async def __anext__(self) -> bytes:
        raise ValueError("Separator is not found, and chunk exceed the limit")


class _Proc:
    def __init__(self, stdout: Any, *, stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.stderr = _Stderr(stderr)
        self.stdin = _Stdin()
        self.returncode: int | None = None
        self.terminated = 0
        self.killed = 0

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.killed += 1
        if self.returncode is None:
            self.returncode = -9


def _spawning(proc: Any) -> Any:
    async def spawn(*_argv: str, **_kwargs: Any) -> Any:
        return proc

    return spawn


# ── drivers ──────────────────────────────────────────────────────────────────


async def _drive(
    cognition: ClaudeCliCognition,
    *,
    agent: Any = None,
    ctx: Any = None,
    task: str = "summarise the diff",
) -> list[StreamEvent]:
    """Every event one ``drive`` produced, terminal event last."""
    return [ev async for ev in cognition.drive(agent or _Agent(), task, ctx, None)]


def _final(events: list[StreamEvent]) -> Any:
    """The one terminal result, asserting the exactly-one-``final`` contract."""
    types = [e.type for e in events]
    assert types.count("final") == 1, types
    assert types[-1] == "final", types
    return events[-1].result


# ── the terminal event exists at all ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_binary_that_is_not_on_path_still_produces_one_terminal_event() -> None:
    """``drive`` is an async generator, so a spawn failure has nowhere to raise
    that a caller's ``async for`` would survive.

    The guarantee callers are told to rely on — one and only one ``final``,
    whatever happened — is most load-bearing here, because a missing ``claude``
    is the single most common way this cognition fails and it fails before the
    binary has a chance to say anything. ``execution_started=False`` is the fact
    a retry policy needs: nothing ran, so nothing was half-done.
    """

    async def missing(*_argv: str, **_kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", "claude")

    result = _final(await _drive(ClaudeCliCognition(spawn=missing)))

    assert result.stop_reason == "failed"
    assert result.evals["stop_reason"] == "spawn_failed"
    assert result.partial is True
    assert result.output == ""
    assert result.evals["execution_started"] is False
    assert result.evals["execution_uncertain"] is False, "nothing ran, so nothing is uncertain"
    assert "FileNotFoundError" in result.evals["error"]


@pytest.mark.asyncio
async def test_an_exec_the_os_refuses_is_reported_as_a_spawn_failure_not_a_parse_bug() -> None:
    """A binary that exists and cannot be executed (no +x, a noexec mount, a
    seccomp refusal) is still "the CLI never started".

    Classifying it below the spawn — as ``parse_failed`` — would send the reader
    hunting for a corrupt payload from a process that produced no bytes at all.
    """

    async def refused(*_argv: str, **_kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", "claude")

    result = _final(await _drive(ClaudeCliCognition(spawn=refused)))

    assert result.evals["stop_reason"] == "spawn_failed"
    assert result.evals["execution_started"] is False


@pytest.mark.asyncio
async def test_a_mistyped_working_dir_is_not_reported_as_a_missing_binary(tmp_path: Any) -> None:
    """One exception type, two problems, two different fixes.

    ``create_subprocess_exec`` raises ``FileNotFoundError`` for a missing ``cwd``
    AND for a missing binary, so without the pre-flight check an operator who
    typo'd a path is told to install the CLI. The spawn must not even be
    attempted, because the error it would produce is the ambiguous one.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("hi"), _result()])])
    cognition = ClaudeCliCognition(spawn=cli, working_dir=tmp_path / "not-here")

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "working_dir_missing"
    assert result.stop_reason == "failed"
    assert cli.spawns == 0, "the check is a pre-flight; the subprocess must not be paid for"


# ── the worst shape: a truncated stream reported as a finished answer ────────


@pytest.mark.asyncio
async def test_a_stream_that_ends_without_the_clis_own_terminal_payload_is_never_complete() -> None:
    """The run produced text and the CLI never said it was done.

    Measured before this reason existed: a stdout that stopped mid-object
    reported ``stop_reason="complete"``, ``partial=False``, and handed back half
    an answer as a finished one — the single worst shape a failure can take,
    because the result carries no flag a caller could branch on and the text is
    perfectly well-formed. ``saw_terminal`` is tracked explicitly rather than
    inferred because a SUCCESSFUL terminal payload leaves ``stop_reason`` at
    ``None`` — the same value a stream that stopped early has.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("The three risks are")])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "malformed_output"
    assert result.stop_reason == "failed"
    assert result.partial is True, "half an answer must never be handed back as a finished one"
    assert result.output == "The three risks are", "the fragment is kept — it is real, just incomplete"


@pytest.mark.asyncio
async def test_a_final_line_cut_mid_object_is_a_truncated_stream_and_not_a_parse_failure() -> None:
    """A process killed partway through writing its ``result`` payload.

    The half-line is skipped by ``_parse_line`` (it is not a JSON object), so
    nothing raises and the run looks like one that simply ended — which is
    exactly why the terminal-payload check has to be what catches it. Reporting
    ``parse_failed`` would also be wrong: nothing in the stream was corrupt, the
    writer was interrupted.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("partial"), '{"type":"result","subtype":'])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "malformed_output"
    assert result.partial is True


@pytest.mark.asyncio
async def test_a_run_that_reached_its_terminal_payload_is_reported_complete() -> None:
    """The other half of the contract, and the reason the check above cannot
    just be "always partial".

    A ``malformed_output`` that fired on healthy runs would make ``partial``
    meaningless in the other direction — every successful run flagged as
    suspect, and callers taught to ignore the flag.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("all three risks"), _result()])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.stop_reason == "complete"
    assert result.partial is False
    assert result.output == "all three risks"


# ── the stop-reason ranking ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_death_by_signal_outranks_the_terminal_payload_the_cli_managed_to_write() -> None:
    """No terminal payload can make a crashed process successful.

    An OOM kill mid-run can still leave a ``result`` on the pipe, and taking
    that at face value reports a complete run for a process the kernel shot.
    ``process_crashed`` is a refinement of ``cli_exit_<n>`` because a negative
    code is not an exit the program chose.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("answer"), _result()], returncode=-9)])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "process_crashed"
    assert result.evals["cli_return_code"] == -9
    assert result.partial is True


@pytest.mark.asyncio
async def test_a_cli_that_reported_its_own_failure_keeps_saying_so_over_a_positive_exit_code() -> None:
    """claude 2.1.251 publishes ``error_max_turns`` and THEN exits 1.

    Letting the exit code win replaces the only actionable diagnosis — the
    session ran out of turns, raise ``max_turns`` — with ``cli_exit_1``, which
    names nothing. Versions disagree about whether a semantic error exits zero,
    so the result payload has to be the authority wherever it exists.
    """
    cli = FakeClaudeCli(
        [CliRun.of([_init(), _assistant("as far as I got"), _result(subtype="error_max_turns", is_error=True)],
                   returncode=1)]
    )

    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "error_max_turns"
    assert result.partial is True
    assert result.output == "as far as I got"


@pytest.mark.asyncio
async def test_an_error_whose_subtype_still_reads_success_falls_back_to_a_generic_marker() -> None:
    """``is_error: true`` with ``subtype: "success"`` is a shape the CLI really
    emits, and ``stop_reason="success"`` for a failed run is a contradiction.

    Reporting the subtype verbatim would produce a result that says it succeeded
    and is flagged partial — so the two halves disagree and whichever one a
    caller reads, it is wrong half the time.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("x"), _result(is_error=True)])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "cli_reported_error"
    assert result.stop_reason == "failed"
    assert result.partial is True


@pytest.mark.asyncio
async def test_a_non_zero_exit_with_no_terminal_payload_keeps_its_own_number() -> None:
    """``cli_exit_2`` is a clap usage error — a flag this cognition built wrong
    — and ``cli_exit_1`` is not.

    Collapsing both into ``failed`` throws away the number an operator starts
    from, and the CLI's stderr is the only other clue; both have to survive onto
    the result.
    """
    cli = FakeClaudeCli([CliRun.of([], returncode=2, stderr=b"error: unexpected argument '--search' found")])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli)))

    assert result.evals["stop_reason"] == "cli_exit_2"
    assert result.evals["cli_return_code"] == 2
    assert "unexpected argument" in result.evals["stderr"]


@pytest.mark.asyncio
async def test_a_schema_the_cli_refused_at_startup_is_named_rather_than_left_as_an_exit_code() -> None:
    """The CLI validates ``--json-schema`` at startup and complains on stderr.

    ``cli_exit_1`` plus a stderr the result only carries on failure sends an
    operator to read logs; ``schema_rejected`` names the thing to go fix. It is
    read off stderr because the exit code is identical to every other startup
    refusal.
    """
    cli = FakeClaudeCli(
        [CliRun.of([], returncode=1, stderr=b"Error: --json-schema is not a valid JSON Schema")]
    )
    cognition = ClaudeCliCognition(spawn=cli, json_schema={"type": "object", "properties": {"a": {"type": "string"}}})

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "schema_rejected"
    assert result.stop_reason == "failed"


@pytest.mark.asyncio
async def test_a_single_payload_too_big_for_the_reader_gets_a_size_reason_not_a_corruption_reason() -> None:
    """Measured against real codex on "read every file under agentkit/": the CLI
    exited 0 holding the answer and the run reported ``parse_failed``.

    Nothing was malformed — the payload was well-formed JSON bigger than the
    reader will assemble, in practice a tool result carrying a large file — so
    the fix is a size and ``parse_failed`` sends the reader hunting for a corrupt
    payload instead. ``execution_uncertain`` is True because the CLI had already
    started doing work.
    """
    proc = _Proc(_OverrunsTheReader())

    result = _final(await _drive(ClaudeCliCognition(spawn=_spawning(proc))))

    assert result.evals["stop_reason"] == "output_line_too_long"
    assert result.stop_reason == "failed"
    assert result.evals["execution_uncertain"] is True
    assert "CliLineTooLong" in result.evals["error"]
    assert "read the file in parts" in result.evals["error"]


# ── liveness ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_cli_that_never_speaks_is_bounded_killed_and_told_apart_from_a_hanging_provider() -> None:
    """``startup`` covers LOCAL work only — boot, read config, resolve auth —
    which is why it is the one bound that can default to on.

    The run looks active the whole time, so without a bound the caller waits
    indefinitely; and the process must be killed, because a bound that gives up
    on reading while leaving the child alive is a leak dressed as a timeout. The
    limit goes on the result so "raise it" or "this was a real hang" does not
    need the construction site to hand.
    """
    proc = _Proc(_NeverSpeaks(), stderr=b"some unrelated warning")
    cognition = ClaudeCliCognition(spawn=_spawning(proc), timeouts=CliTimeouts(startup=0.05))

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "startup_timeout"
    assert result.evals["timeout_kind"] == "startup_timeout"
    assert result.evals["timeout_s"] == 0.05
    assert result.partial is True
    assert proc.terminated and proc.killed, "a bound that leaves the child running is a leak"


@pytest.mark.asyncio
async def test_a_startup_hang_whose_stderr_names_a_credential_says_so_instead() -> None:
    """The measurement this whole mechanism exists for: ``claude -p`` with an
    invalid ``ANTHROPIC_API_KEY`` emitted no stdout and no exit for 45+ seconds.

    ``startup_timeout`` alone would send an operator to look at the network or
    the model when the answer was a credential. The refinement may only ever
    make an already-failed run's label more specific — the fallback is the
    bound's own name, which is always correct.
    """
    proc = _Proc(_NeverSpeaks(), stderr=b"Invalid API key \xc2\xb7 Please run /login")
    cognition = ClaudeCliCognition(spawn=_spawning(proc), timeouts=CliTimeouts(startup=0.05))

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "authentication_failed"
    assert result.evals["timeout_kind"] == "startup_timeout", "the bound that fired is still recorded"
    assert result.stop_reason == "failed"


@pytest.mark.asyncio
async def test_a_run_that_goes_quiet_after_speaking_crosses_the_idle_bound_not_the_startup_one() -> None:
    """``idle`` is the bound that bites: the CLI runs its own tools in its own
    process, so a ``Bash(npm install)`` is minutes of legitimate silence.

    Which bound fired is the whole diagnosis — ``startup_timeout`` says the
    binary never came up and ``idle_timeout`` says a tool call is stuck, and they
    send an operator to two different places. The phase transition is what
    separates them, so it has to be observable through the driver and not only
    on the deadline object.
    """

    class _OneLineThenSilence:
        def __init__(self) -> None:
            self._spoken = False

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> bytes:
            if not self._spoken:
                self._spoken = True
                return b'{"type":"system","subtype":"init","session_id":"sess-1"}\n'
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    proc = _Proc(_OneLineThenSilence())
    cognition = ClaudeCliCognition(
        spawn=_spawning(proc), timeouts=CliTimeouts(startup=None, idle=0.05)
    )

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "idle_timeout"
    assert result.evals["timeout_kind"] == "idle_timeout"
    assert proc.terminated and proc.killed


@pytest.mark.asyncio
async def test_a_crossed_bound_outranks_the_parse_error_that_cutting_the_stream_produces() -> None:
    """Killing a process mid-line leaves a half-written payload behind.

    If the parse error that produces outranked the bound, every timeout would be
    reported as ``parse_failed`` — a correct description of the last 40 bytes and
    a useless one of the run. The bound is WHY the run ended.
    """
    proc = _Proc(_NeverSpeaks())
    cognition = ClaudeCliCognition(spawn=_spawning(proc), timeouts=CliTimeouts(startup=0.05))

    result = _final(await _drive(cognition))

    assert result.evals["stop_reason"] == "startup_timeout"
    assert "parse_failed" not in str(result.evals.get("stop_reason"))


# ── cancellation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_tripped_stop_token_terminates_the_process_and_reports_a_partial_answer() -> None:
    """A stop button must leave no ``claude`` holding a pipe.

    The group — not the pid — because a cancelled run is exactly the one whose
    CLI may have forked something expensive and stopped reporting on it. The
    result is ``cancelled`` rather than ``failed``: somebody chose to stop this,
    nothing went wrong, and ``partial=True`` because whatever text arrived is a
    fragment.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("the first"), _assistant("the second"), _result()])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli), ctx=_StopButton(after_lines=1)))

    assert result.evals["stop_reason"] == "cancelled"
    assert result.stop_reason == "cancelled"
    assert result.partial is True
    invocation = cli.invocations[0]
    assert invocation.terminated is True
    assert invocation.killed is True, "the SIGKILL sweep is what reaps what the CLI forked"


@pytest.mark.asyncio
async def test_an_asyncio_cancel_delivers_the_terminal_event_and_then_keeps_propagating() -> None:
    """Both halves, in that order.

    Swallowing ``CancelledError`` — which a bare ``except BaseException`` does —
    breaks ``asyncio.wait_for``'s timeout and a ``TaskGroup``'s sibling
    cancellation, so the caller's own cancel mechanism silently stops working.
    Not delivering the event first breaks the exactly-one-``final`` contract on
    the path callers most need it.
    """

    async def cancelled_mid_spawn(*_argv: str, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError()

    seen: list[StreamEvent] = []
    with pytest.raises(asyncio.CancelledError):
        async for event in ClaudeCliCognition(spawn=cancelled_mid_spawn).drive(_Agent(), "go", None, None):
            seen.append(event)

    assert [e.type for e in seen] == ["final"]
    assert seen[-1].result.evals["stop_reason"] == "cancelled"
    assert seen[-1].result.partial is True


@pytest.mark.asyncio
async def test_a_keyboard_interrupt_is_reported_as_a_result_and_then_re_raised() -> None:
    """Ctrl-C during ``agent.run()`` used to come back as a tidy
    ``AgentResult(partial=True)`` while the interpreter kept running.

    The caller's own ``except KeyboardInterrupt`` never fired, so a script that
    meant to shut down carried on. The terminal-event guarantee is about the
    EVENT, not about swallowing; ``Exception`` is the dividing line because that
    is the line the language draws.
    """

    async def interrupted(*_argv: str, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt()

    seen: list[StreamEvent] = []
    with pytest.raises(KeyboardInterrupt):
        async for event in ClaudeCliCognition(spawn=interrupted).drive(_Agent(), "go", None, None):
            seen.append(event)

    assert [e.type for e in seen] == ["final"]
    assert seen[-1].result.partial is True


# ── spend on a run that failed ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_budget_with_no_headroom_refuses_before_paying_for_a_subprocess() -> None:
    """Spawning a CLI to be told what we already know costs two to five seconds
    of warm-up.

    ``budget_exhausted`` is deliberately NOT in the failure set: it is the
    RESUMABLE stop reason — raise the ceiling and run again — so mapping it to
    ``failed`` would tell a caller to go debug instead of to top up.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _result()])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli), ctx=_ExhaustedCtx()))

    assert result.evals["stop_reason"] == "budget_exhausted"
    assert result.stop_reason == "budget_exhausted", "resumable, not failed"
    assert cli.spawns == 0


@pytest.mark.asyncio
async def test_the_remaining_headroom_is_handed_to_the_cli_so_it_can_stop_itself() -> None:
    """A budget that is only checked before and after cannot stop a run
    mid-flight.

    Before the spend integration a $50 CLI run against a $1 budget completed
    happily and the ledger read $0.00. ``--max-budget-usd`` is the half that
    makes the ceiling real while the run is in progress, so its absence is a
    documented safety mechanism doing nothing.
    """

    class _Headroom:
        def remaining(self) -> float:
            return 1.5

    class _Ctx:
        budget = _Headroom()

        def check_cancelled(self) -> None:
            return None

    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("ok"), _result()])])
    await _drive(ClaudeCliCognition(spawn=cli), ctx=_Ctx())

    argv = cli.invocations[0].argv
    assert "--max-budget-usd" in argv
    assert float(argv[argv.index("--max-budget-usd") + 1]) == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_a_run_that_reached_its_terminal_payload_charges_what_the_cli_reported() -> None:
    """The baseline the next test is measured against.

    The CLI's ``result.total_cost_usd`` is the only cost figure on this path, and
    it has to reach the meters — the cognition bypasses the ``Invoker``, so the
    ``meter()`` middleware never sees this usage and every meter would otherwise
    stay at zero.
    """
    meter = _RecordingMeter()
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("ok"), _result()])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli), ctx=_MeteredCtx(meter)))

    assert result.usage.cost_usd == pytest.approx(0.25)
    assert [c[0] for c in meter.charges] == [pytest.approx(0.25)]
    assert result.usage.input_tokens == 5


@pytest.mark.asyncio
async def test_a_truncated_run_reports_its_cost_as_unknown_rather_than_as_zero() -> None:
    """A failed run's cost is UNKNOWN, and unknown is not ``$0.00``.

    ``total_cost_usd`` arrives only on the ``result`` payload, so a run cut off
    before it — a timeout, a cancel, a crash, a truncated stream — has no cost
    figure at all. Reporting ``0.0`` with ``cost_source=REPORTED`` is not a
    missing number, it is the wrong number stated as fact: the meters are charged
    nothing, the budget keeps its headroom, and a dashboard summing
    ``usage.cost_usd`` shows a run that burned ten minutes of Opus as free. The
    honest shape is ``cost_usd=None`` / ``CostSource.UNPRICED``, which is what the
    Codex cognition produces for the identical case and what the framework's
    ``UNPRICED`` source exists to say.
    """
    meter = _RecordingMeter()
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("ten minutes of expensive work")])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli), ctx=_MeteredCtx(meter)))

    assert result.evals["stop_reason"] == "malformed_output", "precondition: the terminal payload never arrived"
    assert result.usage.cost_usd is None
    assert result.usage.cost_source is CostSource.UNPRICED
    assert meter.charges and meter.charges[0][0] is None, "a meter must not be told the run was free"


@pytest.mark.asyncio
async def test_a_meter_that_refuses_the_charge_records_the_refusal_without_losing_the_answer() -> None:
    """The spend already happened and the run already produced an answer.

    A ceiling crossed on the LAST call has to be recorded and reported, not
    converted into a lost result — and a custom meter that misbehaves is
    contained for the same reason. The note comes back as data so it lands in
    ``evals`` rather than being dropped.
    """

    class _Refusing:
        async def charge(self, _call: Any, _usage: Any) -> None:
            raise RuntimeError("the monthly ceiling is behind us")

    class _Ctx:
        all_meters = [_Refusing()]
        actor_budget = None

        def check_cancelled(self) -> None:
            return None

    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("the answer"), _result()])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli), ctx=_Ctx()))

    assert result.output == "the answer", "a meter must not cost the caller the run's answer"
    assert result.stop_reason == "complete", "a charge cannot change the stop reason"
    assert result.evals["meter_error"] == "RuntimeError: the monthly ceiling is behind us"


# ── diagnostics on the way out ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stderr_reaches_the_result_only_on_a_failure_and_with_credentials_removed() -> None:
    """``evals["stderr"]`` is the CLI's own diagnostics, and an ``AgentResult``
    is checkpointed, logged and fanned out to observers.

    So it is attached only when it explains something — a successful run's
    stderr is progress chatter — and credential-shaped runs are replaced,
    because a CLI printing a failing request's ``Authorization`` header is
    ordinary diagnostic behaviour and would otherwise have that header
    persisted.
    """
    failing = FakeClaudeCli(
        [CliRun.of([], returncode=1, stderr=b"POST /v1/messages\nauthorization: Bearer abcdefghijklmnopqrstuvwx\n")]
    )
    failed = _final(await _drive(ClaudeCliCognition(spawn=failing)))

    assert "POST /v1/messages" in failed.evals["stderr"]
    assert "abcdefghijklmnopqrstuvwx" not in failed.evals["stderr"]
    assert "[redacted]" in failed.evals["stderr"]
    assert "authorization" in failed.evals["stderr"], "which credential leaked is not itself the secret"

    healthy = FakeClaudeCli([CliRun.of([_init(), _assistant("ok"), _result()], stderr=b"reading 3 files...")])
    assert "stderr" not in _final(await _drive(ClaudeCliCognition(spawn=healthy))).evals


@pytest.mark.asyncio
async def test_an_exception_message_is_redacted_like_stderr_because_it_is_stored_the_same_way() -> None:
    """``evals["error"]`` is checkpointed and fanned out exactly like the rest of
    ``evals``.

    An exception message can carry a URL with a token in it — a connection error
    naming the endpoint it failed against is the common shape — so leaving this
    one unredacted would make the credential durable.
    """

    async def leaky(*_argv: str, **_kwargs: Any) -> Any:
        raise RuntimeError("POST https://api.example.test/v1?api_key=0123456789abcdefghij failed")

    result = _final(await _drive(ClaudeCliCognition(spawn=leaky)))

    assert "0123456789abcdefghij" not in result.evals["error"]
    assert "api_key=" in result.evals["error"]


@pytest.mark.asyncio
async def test_a_reasoning_chain_is_kept_beside_the_answer_and_not_inside_it() -> None:
    """``thinking`` blocks are the model's scratch work, not its response.

    Folding them into ``output`` puts reasoning in the answer a caller shows a
    user; dropping them loses the only record of how the run got there. They are
    streamed live AND folded onto the result, because a consumer that joined
    late saw none of the deltas.
    """
    thinking_block = {
        "type": "assistant",
        "message": {
            "id": "msg-1",
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "weighing the two options"}],
        },
    }
    cli = FakeClaudeCli([CliRun.of([_init(), thinking_block, _assistant("Option B."), _result()])])

    events = await _drive(ClaudeCliCognition(spawn=cli))
    result = _final(events)

    assert result.evals["thinking"] == "weighing the two options"
    assert result.output == "Option B.", "the reasoning must not become the answer"
    assert "weighing the two options" in [e.text for e in events if e.type == "message_delta"]


# ── configuration refused before it costs a subprocess ───────────────────────


@pytest.mark.parametrize(
    ("kwargs", "fragment", "why"),
    [
        (
            {"session_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "resume_session_id": "x"},
            "pass one, not both",
            "session_id NAMES a new session; it has never resumed one, and passing a "
            "finished run's id through it silently starts a fresh conversation",
        ),
        (
            {"continue_session": True, "resume_session_id": "x"},
            "pass one, not both",
            "two different resume mechanisms, so one of them is being ignored",
        ),
        (
            {"fork_session": True},
            "fork_session only applies when resuming",
            "a fork of nothing is a flag with no effect",
        ),
        (
            {"strict_mcp_config": True},
            "only means something alongside",
            "on its own it leaves the session with NO MCP servers, which reads as the opposite",
        ),
        (
            {"stable_prompt_prefix": True, "system_prompt_mode": "replace"},
            "has no effect with",
            "the CLI moves sections out of ITS OWN prompt, which a replacement discards",
        ),
        (
            {"tools": ()},
            "tools=\\(\\) is ambiguous",
            "an empty tuple is genuinely ambiguous between 'leave the defaults' and 'disable all'",
        ),
        (
            {"env_policy": "strict"},
            "env_policy must be",
            "a typo'd policy would silently fall through to inherit, which is the unsafe one",
        ),
        (
            {"session_id": "not-a-uuid"},
            "must be a valid UUID",
            "an invalid UUID is otherwise only rejected once the binary starts",
        ),
    ],
)
def test_a_configuration_the_cli_itself_refuses_is_refused_at_construction(
    kwargs: dict[str, Any], fragment: str, why: str
) -> None:
    """Every one of these is cheaper to catch here than as a subprocess that
    exits non-zero three seconds later with the message on a stderr this
    cognition surfaces only on an already-failed run.

    Two of them are worse than merely late: ``strict_mcp_config`` alone and
    ``session_id`` as a resume both run SUCCESSFULLY and do the opposite of what
    the caller meant, so there is no failed run to read a stderr from.
    """
    with pytest.raises(ValueError, match=fragment):
        ClaudeCliCognition(**kwargs)
    assert why


def test_a_deny_policy_refuses_a_session_holding_tools_the_middleware_cannot_reach() -> None:
    """The CLI runs its own tools inside its own process, so no egress/SSRF
    check, input guard, audit record or idempotency key applies to any of them.

    ``deny`` says an ungovernable tool must not start — and it is SATISFIABLE
    here, which is what makes it a policy rather than a ban: ``tools=("",)``
    plus ``mcp_config=`` serves every tool back through agentkit. The refusal
    has to name that way out, or it reads as "this cognition is unusable".
    """
    with pytest.raises(ValueError, match="native CLI tools"):
        ClaudeCliCognition(native_tool_policy="deny")

    satisfied = ClaudeCliCognition(
        native_tool_policy="deny", tools=("",), mcp_config=('{"mcpServers": {}}',)
    )
    assert satisfied.native_tools() == ()
    assert satisfied.caps == (), "nothing native is held, so nothing native is claimed"


def test_the_trifecta_tags_follow_the_tools_the_session_actually_holds() -> None:
    """``RunPolicy`` reads ``caps`` off everything it is handed, and this
    cognition had none at all — so a session holding ``Read`` + ``WebFetch`` +
    ``Bash`` sailed past the check that refuses the equivalent agentkit tool set.

    The gate was not bypassed; it had nothing to look at. ``allowed_tools`` is
    deliberately not consulted: it is an auto-approve list and cannot hand back a
    tool ``--tools ''`` removed, so reading it here would re-enable, on paper,
    tools the session does not have.
    """
    assert set(ClaudeCliCognition().caps) == {"private_data", "untrusted_content", "egress"}
    assert ClaudeCliCognition(tools=("Read", "Grep")).caps == ("private_data",)
    assert ClaudeCliCognition(tools=("Read", "WebFetch")).caps == ("egress", "private_data", "untrusted_content")
    # An auto-approve list cannot grant a tool the session does not have.
    assert ClaudeCliCognition(tools=("",), allowed_tools=("Bash", "WebFetch")).caps == ()
    # A scoped disallow under-subtracts on purpose: SOME Bash is still Bash.
    assert "egress" in ClaudeCliCognition(tools=("Bash",), disallowed_tools=("Bash(rm:*)",)).caps
    assert ClaudeCliCognition(tools=("Bash",), disallowed_tools=("Bash",)).caps == ()


def test_an_off_contract_disallowed_tools_does_not_turn_a_safety_warning_into_a_crash() -> None:
    """``native_tools()`` runs as the first statement of ``drive``.

    Every earlier version of this class simply ignored
    ``disallowed_tools=None``, because the argv builder only tests it for
    truthiness — so an off-contract value that used to be harmless must not
    start raising ``TypeError`` out of the caller's first run, from inside the
    code that exists to WARN them about something.
    """
    cognition = ClaudeCliCognition()
    cognition.disallowed_tools = None  # type: ignore[assignment]
    assert "Bash" in cognition.native_tools()


# ── the middleware bypass is invisible by construction ───────────────────────


@pytest.mark.asyncio
async def test_the_first_drive_names_the_middlewares_that_will_not_run_and_then_stays_quiet() -> None:
    """A caller who wires the documented tool chain gets a session where none of
    it applies, with nothing saying so.

    A generic "middleware may not apply" would leave the reader no better off,
    so the warning names the chain entries AND the tools. It fires once per
    instance: repeated every iteration it becomes noise, and noise is what
    teaches people to add a ``filterwarnings`` line — which is how the NEXT
    silent misconfiguration gets through.
    """

    class _Ctx:
        invoker = type("_Inv", (), {"tool_middleware": (retry(), egress(), retry())})()

        def check_cancelled(self) -> None:
            return None

    cli = FakeClaudeCli([CliRun.of([_init(), _result()]), CliRun.of([_init(), _result()])])
    cognition = ClaudeCliCognition(spawn=cli)

    with pytest.warns(UserWarning, match="bypasses the Invoker") as caught:
        await _drive(cognition, ctx=_Ctx())
    message = str(caught[0].message)
    # The FACTORY the caller typed, not the closure's own ``__name__``: every
    # raw middleware a factory returns is literally called ``mw``, so three
    # entries would read as three identical names in a warning that is supposed
    # to be matchable against the chain the reader wrote.
    assert "retry, egress" in message, (
        "the names must be the factories the caller typed, in chain order and de-duplicated — "
        "every closure a factory returns is literally called 'mw', so three entries would "
        "otherwise read as three identical names"
    )
    assert "Bash" in message or "every built-in CLI tool" in message
    assert "mcp_config" in message, "the warning has to point at the configuration that works"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await _drive(cognition, ctx=_Ctx())  # the latch holds: a second warning would raise here


@pytest.mark.asyncio
async def test_a_session_with_nothing_native_is_not_warned_about_a_bypass_that_is_not_happening() -> None:
    """Everything then arrives over MCP, through agentkit's own tool path, where
    the chain genuinely does apply.

    Warning anyway is the cry-wolf this feature exists to avoid — and it would
    be warning about the exact configuration the warning tells people to adopt.
    """

    class _Ctx:
        invoker = type("_Inv", (), {"tool_middleware": (retry(),)})()

        def check_cancelled(self) -> None:
            return None

    cli = FakeClaudeCli([CliRun.of([_init(), _result()])])
    cognition = ClaudeCliCognition(spawn=cli, tools=("",), mcp_config=('{"mcpServers": {}}',))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await _drive(cognition, ctx=_Ctx())


# ── which auth the child resolves ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_profile_policy_strips_the_ambient_key_that_would_override_a_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``config_dir=`` exists to run the CLI as a specific signed-in profile, and
    the binary prefers an ambient ``ANTHROPIC_API_KEY`` over that profile —
    announcing it on its own stderr.

    So a service setting ``config_dir=`` per tenant silently ran on whatever
    machine-wide key happened to be exported: the isolation the caller asked for
    is precisely the thing that did not happen, and the only signal was a stderr
    line surfaced on failure. ``config_dir`` therefore implies the policy, and
    the strip is announced, because a silent strip is as invisible as the silent
    inherit it replaces.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-planted-not-a-real-key-000000")
    monkeypatch.setenv("CLAUDE_CODE_MESSAGING_TOKEN", "parent-session-ipc-token")
    cli = FakeClaudeCli([CliRun.of([_init(), _result()])])
    cognition = ClaudeCliCognition(spawn=cli, config_dir=tmp_path)

    with pytest.warns(UserWarning, match="removed ANTHROPIC_API_KEY") as caught:
        await _drive(cognition)

    env = cli.invocations[0].env
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_CODE_MESSAGING_TOKEN" not in env, "a parent session's IPC credential must not be inherited"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path)
    assert "implied by config_dir=" in str(caught[0].message), "the caller never typed this policy"
    assert "env_policy='inherit'" in str(caught[0].message), "the way back has to be in the message"


@pytest.mark.asyncio
async def test_a_session_credential_is_stripped_without_a_warning_nobody_could_act_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every process started from inside a running Claude Code session has the
    session variables set.

    Warning about those would mean warning on essentially every developer
    machine, every run, about something that was never going to be wanted — and
    they decide nothing about WHICH credentials are used, so the name helps
    nobody debugging an auth failure. The auth half is where silence costs
    something; this half is where noise does.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "the-parent-conversation")
    cli = FakeClaudeCli([CliRun.of([_init(), _result()])])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await _drive(ClaudeCliCognition(spawn=cli, env_policy="profile"))

    assert "CLAUDE_CODE_SESSION_ID" not in cli.invocations[0].env


@pytest.mark.asyncio
async def test_an_explicit_env_outranks_the_strip_so_a_per_tenant_credential_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strip removes the AMBIENT key; ``env=`` puts back a CHOSEN one.

    A policy that outranked an explicit assignment would be unusable with
    per-tenant credentials, which is how a safe default gets switched off
    wholesale rather than fixed.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient-not-a-real-key-00000")
    cli = FakeClaudeCli([CliRun.of([_init(), _result()])])
    cognition = ClaudeCliCognition(spawn=cli, env_policy="profile", env={"ANTHROPIC_API_KEY": "tenant-key"})

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing was taken from the caller, so nothing is announced
        await _drive(cognition)

    assert cli.invocations[0].env["ANTHROPIC_API_KEY"] == "tenant-key"


@pytest.mark.asyncio
async def test_bare_mode_without_a_credential_is_warned_about_before_the_cli_misdirects_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In bare mode the CLI never reads OAuth credentials or the keychain.

    So a developer whose ``claude`` works perfectly in a terminal gets "Not
    logged in · Please run /login" the first time their service turns bare mode
    on — a message that points at exactly the wrong fix. The check runs on the
    environment the CHILD will get, not on the raw ``os.environ``, because under
    a stripping policy those differ and the raw copy would clear a run that has
    no credential left.
    """
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                 "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
        monkeypatch.delenv(name, raising=False)
    cli = FakeClaudeCli([CliRun.of([_init(), _result()])])

    with pytest.warns(UserWarning, match="never reads OAuth credentials"):
        await _drive(ClaudeCliCognition(spawn=cli, bare=True))


# ── the prompt and the schema, which are both process-level ──────────────────


@pytest.mark.asyncio
async def test_replace_mode_overrides_the_clis_own_prompt_and_append_mode_adds_to_it() -> None:
    """``--system-prompt`` REPLACES the entire Claude Code system prompt — tool
    guidance, environment info, the lot.

    Wiring an agentkit prompt straight into it turned a capable coding agent into
    a bare chat model that still had tools it no longer knew how to use, with no
    error and nothing in the result to notice. ``append`` is the documented way
    to add instructions while keeping the CLI's behaviour, so it is the default
    and ``replace`` has to be chosen deliberately — which means the two must
    compile to different flags.
    """
    appended = FakeClaudeCli([CliRun.of([_init(), _result()])])
    await _drive(ClaudeCliCognition(spawn=appended), agent=_Agent(prompt="Be terse."))
    assert "--append-system-prompt" in appended.invocations[0].argv
    assert "--system-prompt" not in appended.invocations[0].argv

    replaced = FakeClaudeCli([CliRun.of([_init(), _result()])])
    await _drive(
        ClaudeCliCognition(spawn=replaced, system_prompt_mode="replace"), agent=_Agent(prompt="Be terse.")
    )
    argv = replaced.invocations[0].argv
    assert "--system-prompt" in argv
    assert "--append-system-prompt" not in argv
    assert argv[argv.index("--system-prompt") + 1] == "Be terse."


@pytest.mark.asyncio
async def test_a_versioned_prompt_object_is_rendered_rather_than_stringified() -> None:
    """``agent.prompt`` accepts ``None``, ``str`` or ``Prompt``.

    A ``Prompt`` that reached argv via ``str()`` would send the CLI a repr of a
    dataclass as its system prompt — which the model would dutifully treat as
    instructions, and which no assertion about the answer would catch.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _result()])])
    prompt = Prompt(id="auditor", version="3", template="You are an auditor.")
    await _drive(ClaudeCliCognition(spawn=cli), agent=_Agent(prompt=prompt))

    argv = cli.invocations[0].argv
    assert argv[argv.index("--append-system-prompt") + 1] == "You are an auditor."


@pytest.mark.asyncio
async def test_an_agents_declared_output_type_reaches_the_cli_and_comes_back_coerced() -> None:
    """``output=Invoice`` used to be silently ignored on this path.

    The schema is taken from the agent's own adapter when no explicit
    ``json_schema=`` is given, so the same declaration that types a normal
    agentkit run types a CLI-delegated one — and the validated dict goes BACK
    through the same adapter, so the caller gets the type they declared rather
    than a dict that looks like it.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _result(structured_output={"answer": "42"})])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli), agent=_Agent(_output_adapter=_Adapter())))

    argv = cli.invocations[0].argv
    assert "--json-schema" in argv
    assert '"answer"' in argv[argv.index("--json-schema") + 1]
    assert result.parsed == ("coerced", {"answer": "42"}), "the declared type, not the raw dict"
    assert result.evals["structured_output"] == {"answer": "42"}, "the raw dict is kept alongside it"
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
async def test_a_schema_that_produced_no_structured_output_is_a_failure_and_not_quiet_prose() -> None:
    """A ``success`` result with no ``structured_output`` is a failure the CLI's
    own docs are explicit about.

    Without this branch the run returns ``partial=False`` and ``parsed=None``,
    and a caller who declared ``output=Invoice`` reads the prose as if the object
    had simply not been wired. ``invalid_output`` and not ``failed``: the run
    worked, the shape did not.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("Here is a summary instead."), _result()])])

    result = _final(await _drive(ClaudeCliCognition(spawn=cli), agent=_Agent(_output_adapter=_Adapter())))

    assert result.evals["stop_reason"] == "structured_output_missing"
    assert result.stop_reason == "invalid_output"
    assert result.partial is True
    assert result.evals["structured_output_failure"]["kind"] == "missing"


@pytest.mark.asyncio
async def test_the_cli_exhausting_its_own_structured_retries_is_reported_as_the_cli_giving_up() -> None:
    """The CLI validates against ``--json-schema`` itself, re-prompts itself, and
    reports ``error_max_structured_output_retries`` when it gives up.

    agentkit never saw the intermediate attempts, so there are no per-field
    violations to report — naming the KIND is the honest limit of what we know,
    and inventing violations would send a caller to fix fields nobody complained
    about.
    """
    cli = FakeClaudeCli(
        [CliRun.of([_init(), _assistant("tried"), _result(subtype="error_max_structured_output_retries")])]
    )

    result = _final(await _drive(ClaudeCliCognition(spawn=cli), agent=_Agent(_output_adapter=_Adapter())))

    assert result.evals["stop_reason"] == "error_max_structured_output_retries"
    assert result.stop_reason == "invalid_output"
    assert result.evals["structured_output_failure"]["kind"] == "retries_exhausted"
    assert result.evals["structured_output_failure"]["violations"] == []


# ── a session's failures, which are a different question ─────────────────────


@pytest.mark.asyncio
async def test_a_cli_that_dies_mid_turn_ends_the_session_and_every_later_turn_refuses() -> None:
    """A dead process stays dead: there is no conversation left to continue.

    The alternative is worse than an error — a later ``turn()`` that silently
    spawned a fresh process would answer from an empty context while the caller
    believes the history is there, so the model appears to have forgotten
    everything it was just told. ``session_closed`` is ``failed`` rather than
    ``terminated`` because nobody chose this.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("half an ans")])])
    cognition = ClaudeCliCognition(spawn=cli)

    async with cognition.session() as chat:
        first = [ev async for ev in chat.turn("one")]
        second = [ev async for ev in chat.turn("two")]

    assert _final(first).evals["stop_reason"] == "session_closed"
    assert _final(second).evals["stop_reason"] == "session_closed"
    assert _final(second).stop_reason == "failed"
    assert cli.spawns == 1, "a dead session must not quietly start a new conversation"


@pytest.mark.asyncio
async def test_a_turn_refused_for_a_process_level_flag_costs_one_turn_and_not_the_conversation() -> None:
    """``--json-schema`` is fixed at spawn, so a turn cannot introduce one.

    Reporting that as ``session_closed`` is the conflation that used to kill a
    healthy process: measured, turn 1 complete, turn 2 refused for carrying
    ``output=``, turn 3 — a plain valid turn — came back ``failed
    session_closed`` from a CLI that was alive and idle. The fixes differ, which
    is why the reasons must: re-send this turn without the schema, versus open a
    new session.
    """
    cli = FakeClaudeCli(
        [CliRun.of([_init(), _assistant("first"), _result(result="first"),
                    _init(), _assistant("third"), _result(result="third")])]
    )
    cognition = ClaudeCliCognition(spawn=cli)

    async with cognition.session() as chat:
        first = [ev async for ev in chat.turn("one")]
        refused = [ev async for ev in chat.turn("two", agent=_Agent(_output_adapter=_Adapter()))]
        third = [ev async for ev in chat.turn("three")]

    assert _final(first).stop_reason == "complete"
    assert _final(refused).evals["stop_reason"] == "turn_refused"
    assert _final(refused).stop_reason == "failed"
    assert _final(third).stop_reason == "complete", "the conversation survived a per-turn refusal"
    assert _final(third).output == "third"


@pytest.mark.asyncio
async def test_a_turn_whose_agent_carries_a_different_system_prompt_is_refused_rather_than_ignored() -> None:
    """``--system-prompt`` is fixed at spawn too, so a per-turn prompt cannot be
    honoured.

    It used to be discarded silently: the turn answered as whatever the process
    was spawned as, which reads like the model ignoring its instructions rather
    than like a wiring mistake. The comparison is against what the PROCESS was
    spawned with, not "has a prompt at all", so the ordinary
    ``session(agent=a)`` / ``turn(t, agent=a)`` pattern does not refuse itself.
    """
    agent = _Agent(prompt="You are an auditor.")
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("ok"), _result(result="ok"),
                                    _init(), _assistant("ok2"), _result(result="ok2")])])
    cognition = ClaudeCliCognition(spawn=cli)

    async with cognition.session(agent=agent) as chat:
        same = [ev async for ev in chat.turn("one", agent=agent)]
        different = [ev async for ev in chat.turn("two", agent=_Agent(prompt="You are a poet."))]
        after = [ev async for ev in chat.turn("three", agent=agent)]

    assert _final(same).stop_reason == "complete", "the same agent through both doors must not refuse itself"
    assert _final(different).evals["stop_reason"] == "turn_refused"
    assert "cannot be changed per turn" in _final(different).evals["error"]
    assert _final(after).stop_reason == "complete", "the session is intact"


@pytest.mark.asyncio
async def test_a_turn_level_failure_reads_the_rest_of_that_turn_before_the_next_one_starts() -> None:
    """The CLI does not know anything went wrong on OUR side and keeps writing.

    Leaving those lines in the pipe is how a healthy-looking session returns
    somebody else's answer: the next turn reads this one's tail and ends at its
    stale ``result``. Resynchronising to the turn boundary is what keeps the
    conversation usable; failing to get there means the stream cannot carry
    another turn and saying so beats guessing.
    """
    real_parser = cc._events_from_payload
    failures = {"left": 1}

    async def explode_once(payload: dict[str, Any], *, partial: bool = False) -> Any:
        if payload.get("type") == "assistant" and failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("a planted fold bug")
        async for item in real_parser(payload, partial=partial):
            yield item

    cli = FakeClaudeCli(
        [CliRun.of([_init(), _assistant("turn one's answer"), _result(result="turn one's answer"),
                    _init(), _assistant("turn two's answer"), _result(result="turn two's answer")])]
    )
    cognition = ClaudeCliCognition(spawn=cli)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cc, "_events_from_payload", explode_once)
        async with cognition.session() as chat:
            broken = [ev async for ev in chat.turn("one")]
            recovered = [ev async for ev in chat.turn("two")]

    assert _final(broken).evals["stop_reason"] == "parse_failed"
    assert _final(broken).partial is True
    assert _final(recovered).stop_reason == "complete"
    assert _final(recovered).output == "turn two's answer", "the next turn got ITS answer, not the tail of the last"


@pytest.mark.asyncio
async def test_cancelling_a_turn_ends_the_session_because_nothing_retracts_a_half_finished_one() -> None:
    """There is no protocol message for "forget the turn you were mid-way
    through".

    So the process is terminated, which is the honest outcome: the alternative
    is a session whose CLI-side context contains half an answer nobody saw, and
    every later turn silently reasoning from it.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("a"), _assistant("b"), _result(result="done")])])
    cognition = ClaudeCliCognition(spawn=cli)
    stop = _StopButton(after_lines=1)

    async with cognition.session() as chat:
        cancelled = [ev async for ev in chat.turn("one", ctx=stop)]
        after = [ev async for ev in chat.turn("two")]

    assert _final(cancelled).evals["stop_reason"] == "cancelled"
    assert cli.invocations[0].terminated is True
    assert _final(after).evals["stop_reason"] == "session_closed"


def test_a_session_cleans_up_its_generated_credential_files_however_it_ends() -> None:
    """An inline ``mcp_config`` blob routinely carries a bearer token, so it is
    written to a 0600 file and referenced by path — an argument list is readable
    by anyone who can run ``ps``.

    A session needs this MORE than a one-shot run, not less: it carries the
    value for the whole conversation. The file has to outlast every turn (the
    CLI may re-read it at any point while it is alive) and must not outlast the
    session — a service that leaked one 0600 directory per conversation presents
    as an unrelated disk outage.

    Synchronous, with the filesystem assertions outside the coroutine: blocking
    ``pathlib`` calls on the event loop are what ruff's ASYNC240 exists to
    catch, and this suite is held to it too.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("ok"), _result(result="ok")])])
    inline = (
        '{"mcpServers": {"ledger": {"type": "http", "url": "https://x.test", '
        '"headers": {"Authorization": "Bearer 0123456789abcdefghij"}}}}'
    )
    cognition = ClaudeCliCognition(spawn=cli, mcp_config=(inline,))

    async def converse() -> tuple[str, bool]:
        async with cognition.session() as chat:
            async for _ in chat.turn("one"):
                pass
            argv = cli.invocations[0].argv
            path = argv[argv.index("--mcp-config") + 1]
            assert "Bearer" not in " ".join(argv), "the token must never be in the argument list"
            alive = await asyncio.to_thread(Path(path).exists)
        return path, alive

    path, alive_during = asyncio.run(converse())

    assert alive_during is True, "the CLI may re-read it at any point while it is alive"
    assert not Path(path).exists(), "and it must not survive the session"
    assert not Path(path).parent.exists()


@pytest.mark.asyncio
async def test_interrupting_when_no_turn_is_running_returns_a_receipt_rather_than_raising() -> None:
    """"Stop" with nothing running is a no-op, and a no-op must not be an
    exception.

    Raising would make every UI wrap its stop button in a try block — and this
    once raised the module-private ``_SessionClosed`` straight out of that
    button, which is precisely the try block the receipt exists to spare them.
    ``delivered=False`` is the honest answer: nothing was stopped.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("ok"), _result(result="ok")])])
    cognition = ClaudeCliCognition(spawn=cli)

    async with cognition.session() as chat:
        idle = await chat.interrupt()
        assert idle.delivered is False
        assert idle.still_queued == ()
        async for _ in chat.turn("one"):
            pass
        after = await chat.interrupt()
        assert after.delivered is False, "the turn had already finished"


@pytest.mark.asyncio
async def test_asking_to_drop_queued_messages_on_a_cli_that_cannot_is_refused_before_it_is_ignored() -> None:
    """``cancel_queued`` needs the ``interrupt_cancel_queued_v1`` capability.

    Sent to a CLI that does not advertise it, the extra intent is silently
    dropped: the turn stops and the three queued messages run anyway, which is
    the difference between "the agent stopped" and "the agent stopped and has
    three more things to do". Feature-detect against the advertised set rather
    than a version string, and refuse BEFORE the request rather than after it is
    ignored.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("ok"), _result(result="ok")])])
    cognition = ClaudeCliCognition(spawn=cli)
    refused: list[str] = []

    async with cognition.session() as chat:
        async for event in chat.turn("one"):
            if event.type != "message_delta":
                continue
            # Mid-turn: the capability set has been folded from ``system/init``
            # live, precisely so a stop button can feature-detect against it
            # rather than against a version string.
            assert chat.capabilities == frozenset(), "this recording advertised none"
            assert chat.supports("interrupt_cancel_queued_v1") is False
            with pytest.raises(ValueError, match="interrupt_cancel_queued_v1"):
                await chat.interrupt(cancel_queued=True)
            refused.append("refused")

    assert refused == ["refused"], "the refusal has to happen while a turn is actually interruptible"


@pytest.mark.asyncio
async def test_a_turn_that_crosses_a_liveness_bound_ends_the_session_like_a_cancel_does() -> None:
    """A crossed bound is a cancel arriving by itself.

    The CLI is mid-turn and no protocol message retracts a half-finished one, so
    the process — and with it the conversation — ends here. ``timeouts`` are per
    TURN and measured from the write to stdin: reading ``total`` as a ceiling on
    the whole conversation would kill the thing this class exists for, and
    ``startup`` bounds each turn's first line rather than only the first turn's,
    which is the right shape anyway since the turn that pays the warm-up is the
    slowest one.
    """
    proc = _Proc(_NeverSpeaks())
    cognition = ClaudeCliCognition(spawn=_spawning(proc), timeouts=CliTimeouts(startup=0.05))

    async with cognition.session() as chat:
        timed_out = _final([ev async for ev in chat.turn("one")])
        after = _final([ev async for ev in chat.turn("two")])

    assert timed_out.evals["stop_reason"] == "startup_timeout"
    assert timed_out.evals["timeout_s"] == 0.05
    assert proc.terminated and proc.killed, "a bound that leaves the child running is a leak"
    assert after.evals["stop_reason"] == "session_closed"


@pytest.mark.asyncio
async def test_an_asyncio_cancel_mid_turn_delivers_the_terminal_event_and_then_propagates() -> None:
    """A session turn keeps the same contract as a one-shot drive.

    Swallowing ``CancelledError`` here would break the caller's ``wait_for``
    exactly as it does in ``drive``, and the process has to go too — the
    conversation cannot continue past a turn nobody saw the end of.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("a"), _result(result="a")])])
    real_parser = cc._events_from_payload

    async def cancelled_mid_fold(payload: dict[str, Any], *, partial: bool = False) -> Any:
        if payload.get("type") == "assistant":
            raise asyncio.CancelledError()
        async for item in real_parser(payload, partial=partial):
            yield item

    seen: list[StreamEvent] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cc, "_events_from_payload", cancelled_mid_fold)
        async with ClaudeCliCognition(spawn=cli).session() as chat:
            with pytest.raises(asyncio.CancelledError):
                async for event in chat.turn("one"):
                    seen.append(event)

    assert [e.type for e in seen] == ["final"]
    assert seen[-1].result.evals["stop_reason"] == "cancelled"
    assert cli.invocations[0].terminated is True


@pytest.mark.asyncio
async def test_a_stream_that_cannot_be_resynchronised_is_declared_unusable_rather_than_guessed_at() -> None:
    """The other half of the resync decision, and the one that has to fail
    closed.

    A turn-level failure whose remaining stream never reaches a ``result`` leaves
    stdout at an unknown offset. A session that carried on would hand the NEXT
    turn whatever bytes happened to be left; saying the session is over is the
    only answer that cannot silently return somebody else's answer.
    """
    real_parser = cc._events_from_payload
    failures = {"left": 1}

    async def explode_once(payload: dict[str, Any], *, partial: bool = False) -> Any:
        if payload.get("type") == "assistant" and failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("a planted fold bug")
        async for item in real_parser(payload, partial=partial):
            yield item

    # No ``result`` anywhere: the resync drains to EOF and never finds a
    # boundary, which is "the CLI is gone", not "we are merely mid-turn".
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("an answer nobody will read")])])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cc, "_events_from_payload", explode_once)
        async with ClaudeCliCognition(spawn=cli).session() as chat:
            broken = _final([ev async for ev in chat.turn("one")])
            after = _final([ev async for ev in chat.turn("two")])

    assert broken.evals["stop_reason"] == "parse_failed"
    assert after.evals["stop_reason"] == "session_closed", "an unsynchronised stdout cannot carry another turn"


@pytest.mark.asyncio
async def test_a_session_can_be_an_agents_cognition_so_consecutive_runs_share_one_process() -> None:
    """``drive`` is the ``Cognition``-shaped door onto a session.

    Without it a caller who wants consecutive ``agent.run(...)`` calls to share
    one CLI-side conversation has to drive ``turn()`` by hand and reimplement
    everything ``Agent`` does around a cognition — and the shape that makes the
    sharing possible is exactly the shape ``Agent`` already calls.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("hi"), _result(result="hi"),
                                    _init(), _assistant("again"), _result(result="again")])])

    async with ClaudeCliCognition(spawn=cli).session() as chat:
        first = _final([ev async for ev in chat.drive(_Agent(), "one", None, None)])
        second = _final([ev async for ev in chat.drive(_Agent(), "two", None, None)])

    assert (first.output, second.output) == ("hi", "again")
    assert cli.spawns == 1, "one process, two runs — which is the whole point of the class"


@pytest.mark.asyncio
async def test_a_session_reports_its_own_name_so_a_trace_can_tell_the_two_regimes_apart() -> None:
    """``Agent._span_attrs`` reads ``cognition.name`` on EVERY run.

    A session without one raised ``AttributeError: 'ClaudeCliSession' object has
    no attribute 'name'`` straight out of ``agent.run(...)`` — the exact usage
    the class advertises. Suffixed rather than delegated verbatim because the two
    regimes have genuinely different behaviour to debug, and that attribute is
    what is supposed to say which one ran.
    """
    cognition = ClaudeCliCognition(spawn=FakeClaudeCli([CliRun.of([_init(), _result()])]))
    session = cognition.session()
    assert session.name == "claude_cli_session"
    assert session.caps == cognition.caps, "a session can BE the cognition handed to RunPolicy"
    with contextlib.suppress(Exception):
        await session.close()
