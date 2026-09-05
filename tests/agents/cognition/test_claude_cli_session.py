"""One `claude` process, many turns — `ClaudeCliCognition.session()`.

`drive()` spawns a subprocess per turn. That costs two to five seconds of CLI
warm-up EVERY time, and — the part that actually matters — the turns share no
context, so a follow-up question has no idea what was just discussed. Measured
against the real binary on the same two-turn conversation:

    session:   spawn + 7.6s + 1.6s = 9.7s   turn 2 answers "8137"
    drive x2:         9.9s + 6.2s  = 16.1s  turn 2: "I don't have a record of
                                            you asking me to remember a number"

A session feeds turns over stdin as newline-delimited JSON
(`--input-format stream-json`) and the CLI keeps its own conversation context
in memory.

Every per-turn contract of `drive` holds unchanged here, because both paths run
the same `_TurnState` fold and the same `_finalise`: exactly one terminal
`final` event, the same stop-reason taxonomy, the same metering. What differs
is what a *shared process* implies, and each of those is a deliberate trade
tested below — serialised turns, a session that ends when its process does, and
a cancel that ends the session because no protocol message retracts a
half-finished turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent, Scope
from agentkit.agents.cognition import ClaudeCliCognition, CliTimeouts, claude_cli
from agentkit.agents.cognition.claude_cli import _user_turn
from agentkit.kernel.types import StreamEvent
from agentkit.prompts.prompt import Prompt
from agentkit.runtime import Budget, RunContext, Services
from agentkit.testing.fakes.ctx import FakeCtx
from tests._assertions import assert_money
from tests.agents.cognition.test_claude_cli import _FakeStderr, _line

real_cli = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


def _turn_lines(text: str, *, cost: float = 0.01, session: str = "sess-1") -> list[bytes]:
    """One complete turn as the CLI emits it: an assistant message, a result."""
    return [
        _line({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}),
        _line(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": session,
                "duration_ms": 5,
                "total_cost_usd": cost,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        ),
    ]


class _FakeStdin:
    """Records what the session writes, so a test can assert on the protocol."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    @property
    def messages(self) -> list[dict[str, Any]]:
        return [json.loads(b.decode()) for b in self.written]


class _SessionStdout:
    """Serves canned lines and then BLOCKS, exactly as a live process does.

    This is the shape a session parser has to get right: after a turn's
    ``result`` payload the stream does not end, it goes quiet until the next
    turn is written. A stdout that raised ``StopAsyncIteration`` instead would
    let a broken "read until EOF" implementation pass.
    """

    def __init__(self, script: list[list[bytes]]) -> None:
        self._script = [list(t) for t in script]
        self._current: list[bytes] = []
        self.exhausted = asyncio.Event()

    def feed_next_turn(self) -> None:
        if self._script:
            self._current.extend(self._script.pop(0))
        else:
            self.exhausted.set()

    def __aiter__(self) -> _SessionStdout:
        return self

    async def __anext__(self) -> bytes:
        while True:
            # Yield to the loop on EVERY read, including one that has a line
            # ready. A real pipe read is a suspension point, and without it a
            # turn runs start to finish without the scheduler ever getting a
            # look in — which would make any concurrency test pass by accident.
            await asyncio.sleep(0)
            if self._current:
                return self._current.pop(0)
            if self.exhausted.is_set():
                raise StopAsyncIteration


class _FakeSessionProcess:
    """A long-lived process: stdin recorded, stdout scripted per turn."""

    def __init__(self, turns: list[list[bytes]], *, returncode: int | None = None) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _SessionStdout(turns)
        self.stderr = _FakeStderr(b"")
        self._returncode = returncode
        self.terminated = False

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def wait(self) -> int:
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = -15

    def kill(self) -> None:
        self._returncode = -9


async def _collect(session: Any, task: str, **kw: Any) -> list[StreamEvent]:
    """Drive one turn, releasing the scripted output as the turn is read."""
    session._proc.stdout.feed_next_turn()
    return [ev async for ev in session.turn(task, **kw)]


def _patched(proc: Any) -> Any:
    return patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    )


# ── 1. the protocol ─────────────────────────────────────────────────────────


def test_the_spawn_takes_turns_over_stdin_not_as_an_argument() -> None:
    """``--input-format stream-json`` means the prompt arrives on stdin. Passing
    one as an ARGUMENT too would make the CLI run a first turn nobody asked
    for."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    with _patched(proc) as spawn:

        async def _go() -> None:
            async with ClaudeCliCognition().session() as chat:
                await _collect(chat, "hello")

        asyncio.run(_go())

    argv = list(spawn.await_args.args)
    assert argv[:2] == ["claude", "-p"]
    assert argv[2:4] == ["--input-format", "stream-json"]
    assert "hello" not in argv


def test_each_turn_is_one_ndjson_user_message() -> None:
    """The shape is the SDK's own and was confirmed against the binary."""
    proc = _FakeSessionProcess([_turn_lines("a"), _turn_lines("b")])
    with _patched(proc):

        async def _go() -> None:
            async with ClaudeCliCognition().session() as chat:
                await _collect(chat, "first")
                await _collect(chat, "second")

        asyncio.run(_go())

    assert proc.stdin.messages == [
        {"type": "user", "message": {"role": "user", "content": "first"}, "parent_tool_use_id": None},
        {"type": "user", "message": {"role": "user", "content": "second"}, "parent_tool_use_id": None},
    ]
    assert all(b.endswith(b"\n") for b in proc.stdin.written), "NDJSON needs the newline"


def test_the_turn_encoder_is_newline_delimited_json() -> None:
    msg = _user_turn("hi")
    assert msg.endswith("\n") and json.loads(msg)["message"]["content"] == "hi"


# ── 2. a turn ends at its result, not at EOF ────────────────────────────────


def test_a_turn_ends_at_its_result_and_the_process_lives_on() -> None:
    """THE structural difference from ``drive``, which reads to EOF. The
    scripted stdout blocks rather than ending after each turn, so an
    implementation that waited for EOF would hang here instead of passing."""
    proc = _FakeSessionProcess([_turn_lines("first"), _turn_lines("second")])
    with _patched(proc):

        async def _go() -> list[str]:
            outs = []
            async with ClaudeCliCognition().session() as chat:
                for task in ("a", "b"):
                    events = await asyncio.wait_for(_collect(chat, task), timeout=2.0)
                    assert [e.type for e in events][-1] == "final"
                    assert sum(e.type == "final" for e in events) == 1
                    outs.append(events[-1].result.output)
                assert proc.returncode is None, "the process must still be alive"
            return outs

        assert asyncio.run(_go()) == ["first", "second"]


def test_the_session_id_is_captured_from_the_stream() -> None:
    """A caller wants it for `--resume` after the session ends."""
    proc = _FakeSessionProcess(
        [
            [
                _line({"type": "system", "subtype": "init", "session_id": "sess-9", "model": "m"}),
                *_turn_lines("hi", session="sess-9"),
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> str | None:
            async with ClaudeCliCognition().session() as chat:
                await _collect(chat, "hello")
                return chat.session_id

        assert asyncio.run(_go()) == "sess-9"


def test_closing_the_session_closes_stdin() -> None:
    """Closing stdin is the protocol's own end-of-conversation signal; the CLI
    exits 0 on it, so this is a clean shutdown rather than a kill."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    with _patched(proc):

        async def _go() -> None:
            async with ClaudeCliCognition().session() as chat:
                await _collect(chat, "hello")

        asyncio.run(_go())

    assert proc.stdin.closed and not proc.terminated


# ── 3. what a shared process implies ────────────────────────────────────────


def test_a_turn_after_close_is_reported_not_raised() -> None:
    """The conversation context died with the process, so a later turn cannot
    silently start a fresh one. Reported through the normal terminal event, so
    a session turn keeps the same exactly-one-``final`` contract."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    with _patched(proc):

        async def _go() -> Any:
            chat = ClaudeCliCognition().session()
            await chat.start()
            await chat.close()
            return [ev async for ev in chat.turn("anyone there?")]

        events = asyncio.run(_go())

    assert [e.type for e in events] == ["final"]
    result = events[0].result
    assert result.evals["stop_reason"] == "session_closed"
    assert result.stop_reason == "failed"
    assert result.partial
    assert "context is gone" in str(result.evals["error"])


def test_a_dead_process_is_noticed_before_the_turn_is_sent() -> None:
    """A CLI that exited (crash, OOM, ``--max-turns``) must not receive a write
    that vanishes into a closed pipe."""
    proc = _FakeSessionProcess([_turn_lines("hi")], returncode=1)
    with _patched(proc):

        async def _go() -> Any:
            chat = ClaudeCliCognition().session()
            await chat.start()
            return [ev async for ev in chat.turn("hello")]

        events = asyncio.run(_go())

    assert events[0].result.evals["stop_reason"] == "session_closed"
    assert proc.stdin.written == [], "nothing should have been written to a dead process"


def test_the_cli_closing_mid_turn_ends_the_session() -> None:
    """stdout ending without a ``result`` means the CLI died mid-answer."""
    proc = _FakeSessionProcess([[_line({"type": "assistant", "message": {"content": []}})]])
    with _patched(proc):

        async def _go() -> Any:
            chat = ClaudeCliCognition().session()
            await chat.start()
            chat._proc.stdout.feed_next_turn()
            chat._proc.stdout.exhausted.set()  # EOF right after
            first = [ev async for ev in chat.turn("hello")]
            second = [ev async for ev in chat.turn("still there?")]
            return first, second

        first, second = asyncio.run(_go())

    assert first[-1].result.evals["stop_reason"] == "session_closed"
    # And the session stays closed rather than looking healthy again. Checking
    # the stop reason alone is not enough — a second turn that WROTE and then
    # hit the dead stream reports the same thing. The distinction that matters
    # is that nothing was sent: the first turn's message is the only one.
    assert second[-1].result.evals["stop_reason"] == "session_closed"
    assert len(proc.stdin.messages) == 1, "a closed session must not send a turn"


def test_turns_are_serialised() -> None:
    """One stdin and one transcript, so two concurrent turns would interleave
    two conversations into one context. The second caller waits.

    Ordering the OUTPUTS is not enough to prove it — two racing readers can
    still emerge in order by luck. The discriminator is the interleaving of
    WRITES against COMPLETIONS: a serialised session writes turn 2 only after
    turn 1 has finished, so "send" and "done" strictly alternate.
    """
    proc = _FakeSessionProcess([_turn_lines("first"), _turn_lines("second")])
    trace: list[str] = []

    original_write = proc.stdin.write

    def _traced(data: bytes) -> None:
        trace.append(f"send:{json.loads(data.decode())['message']['content']}")
        original_write(data)

    proc.stdin.write = _traced  # type: ignore[method-assign]

    with _patched(proc):

        async def _one(chat: Any, task: str) -> None:
            async for ev in chat.turn(task):
                if ev.type == "final":
                    trace.append(f"done:{ev.result.output}")

        async def _go() -> None:
            async with ClaudeCliCognition().session() as chat:
                # Both turns' output is available up front: if the lock were
                # missing, the two readers would race over one stream.
                chat._proc.stdout.feed_next_turn()
                chat._proc.stdout.feed_next_turn()
                await asyncio.wait_for(
                    asyncio.gather(_one(chat, "a"), _one(chat, "b")), timeout=3.0
                )

        asyncio.run(_go())

    assert trace == ["send:a", "done:first", "send:b", "done:second"], trace


def test_structured_output_per_turn_is_refused_with_a_reason() -> None:
    """``--json-schema`` is fixed at spawn, so it cannot be turned on per turn.
    Saying so beats silently returning prose for an ``output=`` the caller
    believes is wired — which is exactly the failure the schema work fixed for
    ``drive``."""
    pydantic = pytest.importorskip("pydantic")

    class Out(pydantic.BaseModel):
        v: str

    proc = _FakeSessionProcess([_turn_lines("hi")])
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                agent = Agent(name="x", output=Out)
                return [ev async for ev in chat.turn("hello", agent=agent)]

        events = asyncio.run(_go())

    err = str(events[-1].result.evals["error"])
    assert "process-level flag" in err and "json_schema=" in err
    assert proc.stdin.written == []


def test_a_refused_turn_does_not_end_the_session() -> None:
    """A turn-level rejection is not a session-level death.

    THE headline. The refusal above is correct — structured output is a
    process-level flag — but it used to leave the session marked closed on a
    process that was alive and idle, so every LATER turn died too. Measured
    before the fix::

        t1: complete 'reply1'  closed=False
        t2 (schema): failed session_closed  _closed=True  proc alive=True
        t3 (plain, after): failed ''  evals=session_closed

    and after::

        t2 (schema): failed turn_refused  _closed=False  proc alive=True
        t3 (plain, after): complete 'reply3'

    Turn 3 is the assertion that matters: a healthy conversation must survive
    one turn being turned away.
    """
    pydantic = pytest.importorskip("pydantic")

    class Out(pydantic.BaseModel):
        v: str

    proc = _FakeSessionProcess([_turn_lines("reply1"), _turn_lines("reply3")])
    with _patched(proc):

        async def _go() -> tuple[Any, Any, Any, bool]:
            async with ClaudeCliCognition().session() as chat:
                first = (await _collect(chat, "one"))[-1].result
                refused_events = [
                    ev async for ev in chat.turn("two", agent=Agent(name="x", output=Out))
                ]
                # Sampled INSIDE the session: leaving the context manager closes
                # the process, so a check afterwards proves nothing.
                alive = proc.returncode is None and not chat._closed
                third = (await _collect(chat, "three"))[-1].result
                return first, refused_events, third, alive

        first, refused_events, third, alive_after_refusal = asyncio.run(_go())

    # The refusal still refuses, with the explanation intact.
    refused = refused_events[-1].result
    assert [e.type for e in refused_events] == ["final"]
    assert "process-level flag" in str(refused.evals["error"])
    assert refused.partial and refused.stop_reason == "failed"
    # ...but it is named for what it is: THIS turn, not the conversation.
    assert refused.evals["stop_reason"] == "turn_refused"
    assert alive_after_refusal, "a refused turn must not close a live session"
    # The turns either side of it are untouched.
    assert first.output == "reply1" and first.stop_reason == "complete"
    assert third.output == "reply3" and third.stop_reason == "complete"


def _parse_bomb(marker: str) -> Any:
    """Patch the payload parser to blow up on the line containing ``marker``.

    A parse bug is the realistic non-process failure: the CLI is fine, our
    reader is not. It used to take the session down with it.
    """
    real = claude_cli._events_from_payload

    async def _boom(payload: dict[str, Any], **kw: Any) -> Any:
        if marker in json.dumps(payload):
            raise ValueError("synthetic parse bug")
        async for item in real(payload, **kw):
            yield item

    return patch.object(claude_cli, "_events_from_payload", _boom)


def test_a_parse_failure_costs_one_turn_not_the_session() -> None:
    """Same bug as the refusal, reached from the other side: ANY exception
    while reading a turn used to set ``_closed``. Measured before the fix,
    ``parse t1: failed parse_failed closed=True`` and then
    ``parse t2: failed ''`` — after it, turn 2 answers ``'reply2'``.

    And it answers with its OWN words. The failed turn was abandoned mid-stream
    with its ``result`` still in the pipe; a session that just carries on would
    hand those leftovers to the next turn, which would end instantly on a stale
    result with empty output. Asserting the TEXT is what separates "the session
    survived" from "the session survived and is still in sync".
    """
    proc = _FakeSessionProcess(
        [
            [
                _line(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "boom"}]},
                    }
                ),
                *_turn_lines("tail-of-the-failed-turn"),
            ],
            _turn_lines("reply2"),
        ]
    )
    with _patched(proc), _parse_bomb("boom"):

        async def _go() -> tuple[Any, Any, bool]:
            async with ClaudeCliCognition().session() as chat:
                first_events = await _collect(chat, "one")
                alive = proc.returncode is None and not chat._closed
                second = (await _collect(chat, "two"))[-1].result
                return first_events, second, alive

        first_events, second, alive_after_parse_failure = asyncio.run(_go())

    first = first_events[-1].result
    assert [e.type for e in first_events][-1] == "final"
    assert sum(e.type == "final" for e in first_events) == 1
    assert first.evals["stop_reason"] == "parse_failed" and first.partial
    assert alive_after_parse_failure, "our own parse bug is not the CLI dying"
    assert second.output == "reply2", "the next turn read the failed turn's leftovers"
    assert second.stop_reason == "complete"


def test_a_stream_that_cannot_be_resynced_does_close_the_session() -> None:
    """The other half of that trade, and the reason it is not just "never
    close". If the failed turn's ``result`` never arrives, the stream is stuck
    mid-answer and the next turn would read one conversation as another. Being
    unable to line the stream back up IS the session ending, so it closes —
    bounded by ``_RESYNC_TIMEOUT_S`` rather than waiting out a turn nobody will
    read."""
    proc = _FakeSessionProcess(
        [
            # A turn that blows up on its first line and then goes quiet: no
            # result, no EOF — exactly a CLI still grinding away.
            [
                _line(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "boom"}]},
                    }
                )
            ],
            _turn_lines("never reached"),
        ]
    )
    with _patched(proc), _parse_bomb("boom"), patch.object(claude_cli, "_RESYNC_TIMEOUT_S", 0.05):

        async def _go() -> tuple[Any, Any, bool]:
            chat = ClaudeCliCognition().session()
            await chat.start()
            first = (await _collect(chat, "one"))[-1].result
            closed = chat._closed
            second = [ev async for ev in chat.turn("two")][-1].result
            return first, second, closed

        first, second, closed = asyncio.run(_go())

    assert first.evals["stop_reason"] == "parse_failed"
    assert closed, "an unsynchronisable stream cannot carry another turn"
    assert second.evals["stop_reason"] == "session_closed"
    assert len(proc.stdin.messages) == 1, "a closed session must not send a turn"


def test_a_process_that_dies_between_turns_still_closes_the_session() -> None:
    """POSITIVE CONTROL for the fix above: keeping a session alive through a
    turn-level failure must not keep it alive through a DEAD PROCESS. A CLI
    that exited (crash, OOM kill, ``--max-turns``) has taken the conversation
    with it, and a session that shrugged that off would silently start a fresh
    one with no history — the failure the whole ``session_closed`` path
    exists to prevent."""
    proc = _FakeSessionProcess([_turn_lines("alive"), _turn_lines("never reached")])
    with _patched(proc):

        async def _go() -> tuple[Any, Any, bool]:
            chat = ClaudeCliCognition().session()
            await chat.start()
            first = (await _collect(chat, "one"))[-1].result
            proc._returncode = 1  # the CLI died between turns
            second = [ev async for ev in chat.turn("two")][-1].result
            closed = chat._closed
            third = [ev async for ev in chat.turn("three")][-1].result
            return first, (second, third), closed

        first, (second, third), closed = asyncio.run(_go())

    assert first.output == "alive"
    assert second.evals["stop_reason"] == "session_closed"
    assert closed, "a dead process must close the session"
    assert third.evals["stop_reason"] == "session_closed"
    assert len(proc.stdin.messages) == 1, "nothing may be written to a dead process"


# ── 4. the shared per-turn contracts still hold ─────────────────────────────


def test_a_turn_charges_the_meters_like_a_drive_does() -> None:
    """Both paths go through ``_finalise``, which is the point of extracting
    it: the spend integration cannot apply to one and not the other."""
    budget = Budget(max_cost_usd=10.0)
    ctx = RunContext("run-1", Scope(), services=Services(), budget=budget)
    proc = _FakeSessionProcess([_turn_lines("a", cost=0.25), _turn_lines("b", cost=0.75)])

    with _patched(proc):

        async def _go() -> None:
            async with ClaudeCliCognition().session() as chat:
                await _collect(chat, "one", ctx=ctx)
                await _collect(chat, "two", ctx=ctx)

        asyncio.run(_go())

    assert_money(budget.spent(), "1.000000")
    assert budget.usage.total_tokens == 300


def test_a_session_turn_reports_a_cli_semantic_error() -> None:
    """``is_error: true`` with a subtype is the CLI signalling a failure while
    exiting cleanly — same handling as a one-shot drive."""
    proc = _FakeSessionProcess(
        [
            [
                _line(
                    {
                        "type": "result",
                        "subtype": "error_max_turns",
                        "is_error": True,
                        "session_id": "s",
                        "total_cost_usd": 0.0,
                        "usage": {},
                    }
                )
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                return await _collect(chat, "hello")

        events = asyncio.run(_go())

    assert events[-1].result.evals["stop_reason"] == "error_max_turns"
    assert events[-1].result.partial


def test_a_cancelled_turn_terminates_the_session() -> None:
    """No protocol message retracts a half-finished turn, so the process ends
    with it. The alternative is a session whose context holds half an answer
    nobody saw."""

    class _Cancelling(FakeCtx):
        def check_cancelled(self) -> None:
            raise RuntimeError("cancelled")

    proc = _FakeSessionProcess([_turn_lines("partial")])
    with _patched(proc):

        async def _go() -> Any:
            chat = ClaudeCliCognition().session()
            await chat.start()
            chat._proc.stdout.feed_next_turn()
            return [ev async for ev in chat.turn("hello", ctx=_Cancelling())]

        events = asyncio.run(_go())

    assert events[-1].result.evals["stop_reason"] == "cancelled"
    assert proc.terminated


# ── 5. against the real binary ──────────────────────────────────────────────


@real_cli
def test_the_real_cli_keeps_context_across_turns() -> None:
    """The whole point, and the one thing a mock cannot demonstrate: turn 2 can
    see turn 1. A per-turn subprocess answers "I don't have a record of you
    asking me to remember a number"."""
    cog = ClaudeCliCognition(
        model="claude-haiku-4-5-20251001", tools=("",), permission_mode="dontAsk"
    )

    async def _go() -> tuple[str, str, float, str | None]:
        outs = []
        async with cog.session() as chat:
            t0 = time.monotonic()
            async for ev in chat.turn("Remember the number 8137. Reply with only: ok"):
                if ev.type == "final":
                    outs.append(ev.result.output)
            first_done = time.monotonic() - t0

            t1 = time.monotonic()
            async for ev in chat.turn(
                "What number did I ask you to remember? Reply with only the number."
            ):
                if ev.type == "final":
                    outs.append(ev.result.output)
            second = time.monotonic() - t1
            return outs[0], outs[1], second / max(first_done, 1e-9), chat.session_id

    first, second, ratio, session_id = asyncio.run(_go())

    assert "8137" in second, f"the session lost its context: {second!r}"
    assert session_id, "no session id was captured"
    # The second turn skips CLI warm-up entirely. Compared as a RATIO rather
    # than an absolute, so the assertion does not encode one machine's speed.
    assert ratio < 0.9, f"the second turn was not faster (ratio {ratio:.2f})"


@real_cli
def test_the_real_cli_session_exits_cleanly_on_close() -> None:
    """Closing stdin ends the conversation; the CLI exits 0 rather than being
    killed."""
    cog = ClaudeCliCognition(
        model="claude-haiku-4-5-20251001", tools=("",), permission_mode="dontAsk"
    )

    async def _go() -> int | None:
        chat = cog.session()
        await chat.start()
        proc = chat._proc
        async for _ in chat.turn("Reply with only: ok"):
            pass
        await chat.close()
        return proc.returncode

    assert asyncio.run(_go()) == 0


# ── 6. interrupt: stop the turn, keep the conversation ─────────────────────
#
# The piece a chat UI needs and cancellation cannot give it. Cancelling a turn
# terminates the process — no protocol message retracts a half-finished turn,
# so the conversation ends with it. An interrupt is the CLI's own verb for the
# same intent: the in-flight turn stops, the process stays up, and the next
# turn continues the SAME conversation.
#
# The wire format was read off the binary:
#
#   ->  {"type":"control_request","request_id":"agentkit-1",
#        "request":{"subtype":"interrupt"}}
#   <-  {"type":"control_response","response":{"subtype":"success",
#        "request_id":"agentkit-1","response":{"still_queued":[]}}}


def _control_response(request_id: str, **inner: Any) -> bytes:
    return _line(
        {
            "type": "control_response",
            "response": {"subtype": "success", "request_id": request_id, "response": inner},
        }
    )


def _init(*capabilities: str) -> bytes:
    return _line(
        {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-1",
            "capabilities": list(capabilities),
        }
    )


def test_an_interrupt_stops_the_turn_and_keeps_the_session() -> None:
    """THE contract. The turn ends ``interrupted``; the next turn runs in the
    same process and completes normally."""
    proc = _FakeSessionProcess(
        [
            [
                _init("interrupt_receipt_v1"),
                _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "1 2"}]}}),
                _control_response("agentkit-1", still_queued=[]),
                _line(
                    {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "session_id": "sess-1",
                        "total_cost_usd": 0.01,
                        "usage": {},
                    }
                ),
            ],
            _turn_lines("still here"),
        ]
    )
    with _patched(proc):

        async def _go() -> tuple[Any, Any, Any, bool]:
            async with ClaudeCliCognition().session() as chat:
                chat._proc.stdout.feed_next_turn()
                stopping = None
                first = None
                async for ev in chat.turn("count to 300"):
                    if ev.type == "message_delta" and stopping is None:
                        # A separate task, which is the shape a UI's stop button
                        # has: the loop below keeps draining, so the CLI's
                        # acknowledgement can actually arrive.
                        stopping = asyncio.create_task(chat.interrupt())
                    if ev.type == "final":
                        first = ev.result
                receipt = await stopping
                second = (await _collect(chat, "are you there?"))[-1].result
                # Sampled INSIDE the session: leaving the context manager
                # closes the process, so a check afterwards would prove
                # nothing about whether the interrupt kept it alive.
                alive = proc.returncode is None and not proc.terminated
                return first, second, receipt, alive

        first, second, receipt, alive_between_turns = asyncio.run(_go())

    assert first.evals["stop_reason"] == "interrupted"
    assert first.stop_reason == "terminated"  # somebody stopped it ON PURPOSE
    assert first.partial, "the turn stopped mid-answer; its text is a fragment"
    assert receipt.delivered
    # ...and the conversation continues in the same process.
    assert second.output == "still here" and second.stop_reason == "complete"
    assert alive_between_turns, "an interrupt must not kill the process — that is cancel()"


def test_the_request_is_the_shape_the_cli_answers() -> None:
    """Read off the binary. A wrong envelope is not rejected — it is ignored,
    so the turn runs to completion and the stop button silently does nothing."""
    proc = _FakeSessionProcess(
        [
            [
                _init("interrupt_receipt_v1"),
                _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}),
                _control_response("agentkit-1"),
                *_turn_lines("done"),
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> None:
            async with ClaudeCliCognition().session() as chat:
                chat._proc.stdout.feed_next_turn()
                stopping = None
                async for ev in chat.turn("go"):
                    if ev.type == "message_delta" and stopping is None:
                        stopping = asyncio.create_task(chat.interrupt())
                if stopping is not None:
                    await stopping

        asyncio.run(_go())

    control = [m for m in proc.stdin.messages if m["type"] == "control_request"]
    assert control == [
        {
            "type": "control_request",
            "request_id": "agentkit-1",
            "request": {"subtype": "interrupt"},
        }
    ]


def test_the_receipt_reports_what_survived() -> None:
    """"The agent stopped" and "the agent stopped and has three more things to
    do" are different states, and only the receipt distinguishes them."""
    proc = _FakeSessionProcess(
        [
            [
                _init("interrupt_receipt_v1"),
                _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}),
                _control_response("agentkit-1", still_queued=["uuid-a", "uuid-b"]),
                *_turn_lines("done"),
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                chat._proc.stdout.feed_next_turn()
                stopping = None
                async for ev in chat.turn("go"):
                    if ev.type == "message_delta" and stopping is None:
                        stopping = asyncio.create_task(chat.interrupt())
                return await stopping

        receipt = asyncio.run(_go())

    assert receipt.delivered
    assert receipt.still_queued == ("uuid-a", "uuid-b")


def test_interrupting_an_idle_session_is_a_no_op() -> None:
    """A control request nobody is reading never gets an answer, so sending one
    would hang. It is also not an error — "stop" with nothing running is a
    no-op, and raising would make every UI wrap its stop button in a try."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                return await asyncio.wait_for(chat.interrupt(), timeout=2.0)

        receipt = asyncio.run(_go())

    assert receipt.delivered is False
    assert not [m for m in proc.stdin.messages if m["type"] == "control_request"]


def test_a_turn_that_ends_first_does_not_strand_the_waiter() -> None:
    """The reader is what resolves a control response, so a turn that finishes
    before the CLI answers leaves nobody to deliver it. Failing the waiter beats
    hanging it forever — but it must fail as a RECEIPT.

    A stop button pressed as the last token lands is a normal race, not an
    error. It used to hand the caller a module-private exception straight out
    of ``interrupt()``::

        interrupt mid/after turn: RAISED _SessionClosed: the turn ended before
        the CLI answered

    which is exactly the try block this method's docstring promises a UI it
    will not need ("a field on a receipt, not a hung application").
    """
    proc = _FakeSessionProcess(
        [
            [
                _init("interrupt_receipt_v1"),
                _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}),
                *_turn_lines("done"),  # a result, and NO control_response
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                chat._proc.stdout.feed_next_turn()
                pending = None
                async for ev in chat.turn("go"):
                    if ev.type == "message_delta" and pending is None:
                        pending = asyncio.create_task(chat.interrupt())
                        await asyncio.sleep(0)
                return pending

        pending = asyncio.run(_go())

    # Resolved by the turn ending, not by ``ack_timeout_s`` expiring: a waiter
    # nobody will ever answer must not be left to time out.
    assert pending.done()
    receipt = pending.result()  # a receipt, not a raise
    assert receipt.delivered is False, "the turn had already finished; nothing was stopped"
    assert receipt.error is not None and "before the CLI answered" in receipt.error


def test_interrupting_after_the_session_is_closed_is_a_no_op() -> None:
    """Same reasoning as an idle session, one step further along: the process
    is gone, so there is nothing to stop and nothing to raise about. A stop
    button on a finished conversation is a no-op, and it must stay one whether
    it is pressed a second before the end or a second after."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    with _patched(proc):

        async def _go() -> Any:
            chat = ClaudeCliCognition().session()
            await chat.start()
            await _collect(chat, "hello")
            await chat.close()
            return await asyncio.wait_for(chat.interrupt(), timeout=2.0)

        receipt = asyncio.run(_go())

    assert receipt.delivered is False and receipt.error is None
    assert not [m for m in proc.stdin.messages if m["type"] == "control_request"]


def test_a_control_response_is_not_folded_into_the_turn() -> None:
    """It arrives on the SAME stdout the turn reader consumes. Treating it as
    turn content would put protocol JSON into the answer."""
    proc = _FakeSessionProcess(
        [
            [
                _init("interrupt_receipt_v1"),
                _control_response("nobody-is-waiting"),
                *_turn_lines("clean answer"),
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                return (await _collect(chat, "go"))[-1].result

        result = asyncio.run(_go())

    assert result.output == "clean answer"
    assert result.stop_reason == "complete"


def test_capabilities_are_feature_detected_not_version_sniffed() -> None:
    """The CLI advertises an OPEN set on ``system/init``; that is what it is
    for, and an unrecognised value is not an error."""
    proc = _FakeSessionProcess(
        [[_init("interrupt_receipt_v1", "something_new_v9"), *_turn_lines("hi")]]
    )
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                assert chat.capabilities == frozenset()  # nothing seen yet
                await _collect(chat, "hello")
                return chat

        chat = asyncio.run(_go())

    assert chat.supports("interrupt_receipt_v1")
    assert chat.supports("something_new_v9")
    assert not chat.supports("interrupt_cancel_queued_v1")


def test_cancel_queued_is_refused_when_unsupported() -> None:
    """Checked BEFORE the request is sent, rather than after the CLI ignores
    it: a stop button that silently does half of what it says is worse than one
    that reports it cannot."""
    proc = _FakeSessionProcess(
        [
            [
                _init("interrupt_receipt_v1"),  # no cancel_queued
                _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}),
                *_turn_lines("done"),
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> None:
            async with ClaudeCliCognition().session() as chat:
                chat._proc.stdout.feed_next_turn()
                async for ev in chat.turn("go"):
                    if ev.type == "message_delta":
                        with pytest.raises(ValueError, match="interrupt_cancel_queued_v1"):
                            await chat.interrupt(cancel_queued=True)

        asyncio.run(_go())

    assert not [m for m in proc.stdin.messages if m["type"] == "control_request"]


def test_cancel_queued_uses_its_own_subtype() -> None:
    proc = _FakeSessionProcess(
        [
            [
                _init("interrupt_receipt_v1", "interrupt_cancel_queued_v1"),
                _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}),
                _control_response("agentkit-1", cancelled=["uuid-a"]),
                *_turn_lines("done"),
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                chat._proc.stdout.feed_next_turn()
                stopping = None
                async for ev in chat.turn("go"):
                    if ev.type == "message_delta" and stopping is None:
                        stopping = asyncio.create_task(chat.interrupt(cancel_queued=True))
                return await stopping

        receipt = asyncio.run(_go())

    assert receipt.cancelled == ("uuid-a",)
    control = [m for m in proc.stdin.messages if m["type"] == "control_request"]
    assert control[0]["request"]["subtype"] == "interrupt_cancel_queued"


def test_an_inline_interrupt_degrades_to_a_bounded_wait() -> None:
    """Calling ``interrupt()`` from inside the loop consuming ``turn()``
    suspends the only reader, so the acknowledgement cannot arrive. That has to
    cost a field on a receipt, not a hung application — the WRITE already
    happened, so the interrupt still takes effect."""
    proc = _FakeSessionProcess(
        [
            [
                _init("interrupt_receipt_v1"),
                _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}),
                _control_response("agentkit-1"),
                *_turn_lines("done"),
            ]
        ]
    )
    with _patched(proc):

        async def _go() -> Any:
            async with ClaudeCliCognition().session() as chat:
                chat._proc.stdout.feed_next_turn()
                receipt = None
                async for ev in chat.turn("go"):
                    if ev.type == "message_delta" and receipt is None:
                        receipt = await chat.interrupt(ack_timeout_s=0.05)
                return receipt

        receipt = asyncio.run(_go())

    assert receipt.delivered, "the request was written even though nothing read the reply"
    assert receipt.error is not None and "separate task" in receipt.error
    # And the request really did go out.
    assert [m["request"]["subtype"] for m in proc.stdin.messages if m["type"] == "control_request"] == [
        "interrupt"
    ]


@real_cli
def test_the_real_cli_honours_an_interrupt_and_stays_alive() -> None:
    """The whole contract against the binary: a long turn is stopped mid-answer
    and the NEXT turn runs in the same process.

    A mock cannot tell us the envelope is right — a wrong one is ignored rather
    than rejected, so the stop button would silently do nothing and the test
    would still pass.
    """
    cog = ClaudeCliCognition(
        model="claude-haiku-4-5-20251001", tools=("",), permission_mode="dontAsk"
    )

    async def _go() -> tuple[Any, Any, Any]:
        async with cog.session() as chat:
            stopping = None
            first = None
            async for ev in chat.turn(
                "Count slowly from 1 to 300, one number per line, with a short "
                "comment on each. When you have finished all 300, write the word "
                "PIPPIN on the last line."
            ):
                if ev.type == "message_delta" and stopping is None:
                    stopping = asyncio.create_task(chat.interrupt())
                if ev.type == "final":
                    first = ev.result
            receipt = await stopping
            second = None
            async for ev in chat.turn("Reply with only: still here"):
                if ev.type == "final":
                    second = ev.result
            return first, second, receipt

    first, second, receipt = asyncio.run(asyncio.wait_for(_go(), timeout=180))

    assert receipt.delivered, "the CLI did not acknowledge the interrupt"
    assert first.evals["stop_reason"] == "interrupted"
    assert first.stop_reason == "terminated"
    # A SENTINEL, not a proxy. The first version asserted `"300" not in output`
    # and flaked: the model echoes the target in its preamble ("I'll count from
    # 1 to 300..."), so the assertion tested its phrasing rather than whether
    # the turn was stopped. `PIPPIN` exists only if the turn ran to completion.
    assert "PIPPIN" not in first.output.upper(), (
        f"the turn was not actually stopped early: {first.output[-200:]!r}"
    )
    # The session survived: the same process answered a second turn.
    assert second.stop_reason == "complete"
    assert "still here" in second.output.lower()


# ── the session as an agent's cognition ─────────────────────────────────────


@pytest.mark.asyncio
async def test_a_session_can_actually_be_an_agents_cognition() -> None:
    """``ClaudeCliSession.drive`` exists so consecutive ``agent.run(...)`` calls
    share one process and one CLI-side conversation, and the class docstring
    says so. It did not work: ``Agent._span_attrs`` reads ``cognition.name`` for
    the ``agentkit.agent.cognition`` trace attribute on every run, the session
    had no such attribute, and ``agent.run`` raised ``AttributeError:
    'ClaudeCliSession' object has no attribute 'name'`` before reaching the CLI
    at all.

    Found while building the same surface for the Codex cognition, which is
    exactly the value of writing the second one: the documented usage of the
    first had no test.
    """
    from agentkit.testing.fakes import CliRun, FakeClaudeCli

    def _turn(text: str) -> list[dict[str, Any]]:
        return [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": "sess-1",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ]

    cli = FakeClaudeCli(
        [CliRun.of([{"type": "system", "subtype": "init", "session_id": "sess-1"}, *_turn("one"), *_turn("two")])]
    )
    cog = ClaudeCliCognition(spawn=cli)
    chat = cog.session()
    await chat.start()
    try:
        agent = Agent(name="x", cognition=chat)
        first = await agent.run("a", FakeCtx())
        second = await agent.run("b", FakeCtx())
    finally:
        await chat.close()

    assert (first.output, second.output) == ("one", "two")
    # One spawn for two turns — the whole reason to hold a process.
    assert cli.spawns == 1
    # And the name a trace will carry marks it as the session regime rather
    # than the one-shot one, which are different things to debug.
    assert chat.name == "claude_cli_session"


# ── 7. the policies a session used to skip ──────────────────────────────────
#
# `drive()` and a session are two entry points into the same binary with
# genuinely different PROCESS lifecycles: one spawn per turn versus one spawn
# for the conversation. That difference is why the session has its own read
# loop instead of going through `_run_cli_process`, and it is a real one — a
# session turn ends at a `result` payload with the process still alive, and it
# multiplexes `control_response` messages off the same stdout so `interrupt()`
# can be awaited.
#
# What does NOT follow is that the two paths should differ in POLICY. They did.
# Secrets were kept out of `drive`'s argv and left in the session's, and the
# liveness bounds applied to `drive` and were silently ignored by the session —
# the worse half of the two, because `timeouts=` is a field on the cognition a
# session holds, so it was configuration that was accepted and discarded.
#
# Both are asymmetries in the wrong direction: a session is the LONGER-lived
# process. A leaked credential sits in `ps` for the whole conversation rather
# than for one turn, and a hang with no bound hangs forever rather than until
# the next spawn.


_TOKEN = "sk-ant-secret-value-do-not-leak"
_MCP_BLOB = json.dumps({"mcpServers": {"x": {"url": "https://e", "headers": {"Authorization": f"Bearer {_TOKEN}"}}}})
_SETTINGS_BLOB = json.dumps({"env": {"MY_TOKEN": _TOKEN}})


def _argv_of(spawn: Any) -> list[str]:
    return list(spawn.await_args.args)


def _one_turn(cog: ClaudeCliCognition, proc: Any) -> Any:
    """Open a session, run one turn, close it. Returns the patched spawn mock."""

    async def _go() -> None:
        async with cog.session() as chat:
            await _collect(chat, "hello")

    with _patched(proc) as spawn:
        asyncio.run(_go())
    return spawn


def test_an_inline_mcp_config_never_reaches_the_session_argv() -> None:
    """The documented way to wire an HTTP MCP server puts a bearer token in an
    inline blob. An argument list is world-readable — any local account can
    read it with ``ps`` — so the blob is written to a 0600 file and passed by
    path, exactly as ``drive`` does.

    This is the check that used to fail. ``start()`` called ``_build_argv``
    without ``mcp_config=``, which made it fall back to the DECLARED value; the
    fallback exists for argv-shape tests and reaching it here is how the
    session shipped the token it was supposed to hide."""
    cog = ClaudeCliCognition(mcp_config=(_MCP_BLOB,))
    spawn = _one_turn(cog, _FakeSessionProcess([_turn_lines("hi")]))

    argv = _argv_of(spawn)
    assert not any(_TOKEN in a for a in argv), f"credential in argv: {argv}"
    path = argv[argv.index("--mcp-config") + 1]
    assert path.endswith("mcp-0.json"), path


def test_an_inline_settings_blob_never_reaches_the_session_argv() -> None:
    """``settings`` is the same hazard by another door — it may carry an
    ``env`` block, or an ``apiKeyHelper``."""
    cog = ClaudeCliCognition(settings=_SETTINGS_BLOB)
    spawn = _one_turn(cog, _FakeSessionProcess([_turn_lines("hi")]))

    argv = _argv_of(spawn)
    assert not any(_TOKEN in a for a in argv), f"credential in argv: {argv}"
    assert argv[argv.index("--settings") + 1].endswith("settings.json")


def _probe(path: str) -> tuple[bool, str, str]:
    """Existence, permission bits and content of a materialised file.

    Called through ``asyncio.to_thread`` by the tests below, because they look
    at the file from inside a LIVE session's coroutine and blocking file IO
    there is what ASYNC240 exists to catch. A test that models an open session
    should not model it with a stalled event loop.
    """
    if not os.path.exists(path):
        return False, "", ""
    return True, oct(os.stat(path).st_mode)[-3:], Path(path).read_text()


def test_the_materialised_file_is_readable_only_by_this_user() -> None:
    """0600, and set BEFORE the content is written. A file that exists for one
    umask-wide instant with the secret already in it would close one disclosure
    by opening a smaller one."""
    holder: dict[str, str] = {}
    cog = ClaudeCliCognition(mcp_config=(_MCP_BLOB,))
    proc = _FakeSessionProcess([_turn_lines("hi")])

    async def _go() -> None:
        async with cog.session() as chat:
            argv = _argv_of(spawn)
            holder["path"] = argv[argv.index("--mcp-config") + 1]
            # Read it while the session is OPEN: the file has to be there for
            # the whole conversation, not merely at spawn.
            exists, mode, text = await asyncio.to_thread(_probe, holder["path"])
            assert exists and mode == "600", f"mode {mode!r}"
            assert json.loads(text)["mcpServers"]["x"]["headers"]
            await _collect(chat, "hello")

    with _patched(proc) as spawn:
        asyncio.run(_go())

    # ...and gone once the session is closed.
    assert not os.path.exists(holder["path"])


def test_a_path_valued_mcp_config_is_passed_through_untouched() -> None:
    """A path is already a reference rather than a value, so there is nothing
    to hide and no scratch directory to make."""
    cog = ClaudeCliCognition(mcp_config=("/etc/mcp.json",))
    spawn = _one_turn(cog, _FakeSessionProcess([_turn_lines("hi")]))

    argv = _argv_of(spawn)
    assert argv[argv.index("--mcp-config") + 1] == "/etc/mcp.json"


def test_the_scratch_survives_every_turn_and_dies_with_the_session() -> None:
    """Ownership is the SESSION's, not a turn's. The CLI may re-read an
    ``--mcp-config`` file at any point while it is alive, so deleting it after
    the turn that spawned it — which is what ``drive`` correctly does with its
    own copy — would break a later turn in a way that looks like a config
    error."""
    seen: list[bool] = []
    holder: dict[str, str] = {}
    cog = ClaudeCliCognition(mcp_config=(_MCP_BLOB,))
    proc = _FakeSessionProcess([_turn_lines("a"), _turn_lines("b")])

    async def _go() -> None:
        async with cog.session() as chat:
            argv = _argv_of(spawn)
            holder["path"] = argv[argv.index("--mcp-config") + 1]
            await _collect(chat, "first")
            seen.append((await asyncio.to_thread(_probe, holder["path"]))[0])
            await _collect(chat, "second")
            seen.append((await asyncio.to_thread(_probe, holder["path"]))[0])

    with _patched(proc) as spawn:
        asyncio.run(_go())

    assert seen == [True, True], "the file was deleted out from under a live CLI"
    assert not os.path.exists(holder["path"])


def test_a_failed_spawn_does_not_leave_the_secret_on_disk() -> None:
    """``start()`` writes the file before it spawns, so the failure path has to
    clean up. Otherwise every failed session start leaks a credential into
    /tmp, which is strictly worse than the argv it replaced — argv dies with
    the process."""
    made: list[str] = []
    real_make = claude_cli._make_scratch

    def _spy() -> str:
        path = real_make()
        made.append(path)
        return path

    cog = ClaudeCliCognition(mcp_config=(_MCP_BLOB,))
    boom = AsyncMock(side_effect=OSError("no such binary"))
    with (
        patch.object(claude_cli, "_make_scratch", _spy),
        patch("agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec", new=boom),
        pytest.raises(OSError, match="no such binary"),
    ):
        asyncio.run(cog.session().start())

    assert made, "the spy never ran — this test is asserting nothing"
    assert not os.path.exists(made[0])


# ── the agent's system prompt ───────────────────────────────────────────────
#
# ``agent.prompt`` IS the system prompt. ``drive()`` renders it into
# ``--append-system-prompt``; the session used to call ``_build_argv`` with
# ``system_prompt=""`` and never supply one, so an agent handed to a session
# kept its schema, its meters and its name and silently lost the instructions
# that say what it IS.
#
# The worst part was which door it broke. ``ClaudeCliSession.drive`` exists so
# a session can BE an agent's cognition — swap ``cognition=cog`` for
# ``cognition=chat`` and keep the process warm — and that advertised
# substitution changed the model's behaviour with no error and nothing missing
# from the result to notice. Measured, same agent both ways:
#
#     cognition=cog   ->  "ARRR, 2 + 2 be 4, matey!..."
#     cognition=chat  ->  "4"
#
# which debugs like a model ignoring its instructions rather than like wiring.


_PERSONA = "You are a pirate. Always answer in pirate speak."


def _system_prompt_of(spawn: Any) -> str | None:
    argv = list(spawn.await_args.args)
    for flag in ("--append-system-prompt", "--system-prompt"):
        if flag in argv:
            return str(argv[argv.index(flag) + 1])
    return None


def test_an_agent_given_to_the_session_supplies_the_system_prompt() -> None:
    """THE regression. The prompt has to reach argv at SPAWN, because
    ``--system-prompt`` is process-level and there is no later chance."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    agent = Agent(name="p", prompt=_PERSONA)

    async def _go() -> None:
        async with ClaudeCliCognition().session(agent=agent) as chat:
            await _collect(chat, "hello")

    with _patched(proc) as spawn:
        asyncio.run(_go())

    assert _system_prompt_of(spawn) == _PERSONA


def test_a_session_with_no_agent_sends_no_system_prompt() -> None:
    """An empty ``--append-system-prompt`` is not the same as omitting it, and
    a bare conversational session must stay bare."""
    proc = _FakeSessionProcess([_turn_lines("hi")])

    async def _go() -> None:
        async with ClaudeCliCognition().session() as chat:
            await _collect(chat, "hello")

    with _patched(proc) as spawn:
        asyncio.run(_go())

    assert _system_prompt_of(spawn) is None


def test_the_session_honours_system_prompt_mode() -> None:
    """``replace`` discards the CLI's own prompt — and with it every
    instruction that makes its tools usable — so it must stay opt-in here
    exactly as it is on ``drive``."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    agent = Agent(name="p", prompt=_PERSONA)

    async def _go() -> None:
        cog = ClaudeCliCognition(system_prompt_mode="replace")
        async with cog.session(agent=agent) as chat:
            await _collect(chat, "hello")

    with _patched(proc) as spawn:
        asyncio.run(_go())

    argv = list(spawn.await_args.args)
    assert "--system-prompt" in argv and "--append-system-prompt" not in argv


def test_a_huge_session_system_prompt_goes_to_a_file() -> None:
    """The argv ceiling applies at ``start()`` too — the session builds its own
    argv, which is how it missed the credential fix and the liveness bounds."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    agent = Agent(name="p", prompt="You are terse. " * 4000)  # 60,000 bytes

    async def _go() -> None:
        async with ClaudeCliCognition().session(agent=agent) as chat:
            await _collect(chat, "hello")

    with _patched(proc) as spawn:
        asyncio.run(_go())

    argv = list(spawn.await_args.args)
    assert "--append-system-prompt-file" in argv
    assert "--append-system-prompt" not in argv


def test_the_same_agent_through_both_doors_is_not_refused() -> None:
    """``session(agent=a)`` then ``turn(t, agent=a)`` is the ordinary pattern.

    The refusal below compares against what the PROCESS was spawned with, not
    against "does this turn carry a prompt" — the naive check would make a
    session refuse its own agent."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    agent = Agent(name="p", prompt=_PERSONA)

    async def _go() -> Any:
        async with ClaudeCliCognition().session(agent=agent) as chat:
            proc.stdout.feed_next_turn()
            return [ev async for ev in chat.turn("hello", agent=agent)][-1].result

    with _patched(proc):
        result = asyncio.run(_go())

    assert result.stop_reason == "complete", result.evals


def test_an_equivalent_prompt_object_is_not_refused() -> None:
    """A ``Prompt`` is a versioned object, so two equivalent ones are not the
    same instance. The comparison renders both sides for that reason."""
    proc = _FakeSessionProcess([_turn_lines("hi")])
    # Two DIFFERENT versions of the same text: the rendered prompt is what the
    # CLI is given, so that is what has to match. Comparing the objects — or
    # their versions — would refuse a caller who merely rebuilt an equivalent
    # one, which is the common case when a Prompt is loaded per turn.
    opened = Agent(name="p", prompt=Prompt(id="persona", version="1", template=_PERSONA))
    turned = Agent(name="p", prompt=Prompt(id="persona", version="2", template=_PERSONA))

    async def _go() -> Any:
        async with ClaudeCliCognition().session(agent=opened) as chat:
            proc.stdout.feed_next_turn()
            return [ev async for ev in chat.turn("hello", agent=turned)][-1].result

    with _patched(proc):
        result = asyncio.run(_go())

    assert result.stop_reason == "complete", result.evals


def test_a_turn_that_brings_a_different_prompt_is_refused_not_dropped() -> None:
    """Same treatment ``output=`` already got, for the same reason.

    Silently answering as whatever the process was spawned as is the behaviour
    this replaces — and it is the one that reads like a bad model instead of a
    wiring mistake. A ``_TurnRefused`` costs the TURN and leaves the
    conversation alive."""
    proc = _FakeSessionProcess([_turn_lines("hi")])

    async def _go() -> tuple[Any, Any]:
        async with ClaudeCliCognition().session() as chat:  # spawned bare
            refused = [
                ev async for ev in chat.turn("hello", agent=Agent(name="p", prompt=_PERSONA))
            ][-1].result
            after = await _collect(chat, "and again")
            return refused, after[-1].result

    with _patched(proc):
        refused, after = asyncio.run(_go())

    assert refused.evals["stop_reason"] == "turn_refused"
    assert "cannot be changed per turn" in (refused.evals.get("error") or "")
    assert "session(agent=" in (refused.evals.get("error") or "")
    sent = [m["message"]["content"] for m in proc.stdin.messages]
    assert sent == ["and again"], (
        f"only the follow-up turn should have reached the CLI, got {sent}"
    )
    # The session survives it, and the refusal cost NOTHING on the wire: the
    # next turn gets the first scripted answer, because the refused one was
    # never sent and never consumed it.
    assert after.stop_reason == "complete" and after.output == "hi"


# ── liveness ────────────────────────────────────────────────────────────────


class _TimedStdout:
    """Emits lines on a schedule, then goes quiet — and STAYS quiet.

    ``_SessionStdout`` above busy-waits between lines, which is right for the
    protocol tests and wrong for these. A liveness bound is wall-clock, so the
    reader has to actually park; a spin would burn the CPU the timeout is
    supposed to be waiting through, and "quiet" has to be a suspension rather
    than a hot loop for the same reason it is in a real pipe.
    """

    def __init__(self, steps: list[tuple[float, bytes]]) -> None:
        self._steps = list(steps)
        self._never = asyncio.Event()

    def __aiter__(self) -> _TimedStdout:
        return self

    async def __anext__(self) -> bytes:
        if not self._steps:
            await self._never.wait()  # a live CLI with nothing to say
            raise AssertionError("unreachable")
        delay, line = self._steps.pop(0)
        await asyncio.sleep(delay)
        return line


def _timed(steps: list[tuple[float, bytes]], *, stderr: bytes = b"") -> Any:
    proc = _FakeSessionProcess([])
    proc.stdout = _TimedStdout(steps)
    proc.stderr = _FakeStderr(stderr)
    return proc


async def _timed_turn(chat: Any, task: str = "hello", **kw: Any) -> list[StreamEvent]:
    """Like ``_collect``, minus the ``feed_next_turn`` — a ``_TimedStdout``
    releases itself on a clock."""
    return [ev async for ev in chat.turn(task, **kw)]


def _final(events: list[StreamEvent]) -> Any:
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1, f"expected exactly one terminal event, got {len(finals)}"
    return finals[0].result


def test_a_turn_that_goes_silent_trips_the_idle_bound() -> None:
    """THE bound that was ignored. Before this, a session turn whose CLI hung
    hung with it — forever, with no stop reason and no process teardown."""
    cog = ClaudeCliCognition(timeouts=CliTimeouts(startup=None, idle=0.05))
    proc = _timed([(0.0, _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "par"}]}}))])

    async def _go() -> Any:
        async with cog.session() as chat:
            return _final(await _timed_turn(chat))

    with _patched(proc):
        result = asyncio.run(_go())

    assert result.evals["stop_reason"] == "idle_timeout"
    assert result.stop_reason == "failed" and result.partial is True
    # And the process is gone: no protocol message retracts a half-finished
    # turn, so the conversation ends with it.
    assert proc.terminated is True


def test_a_turn_whose_cli_never_speaks_trips_the_startup_bound() -> None:
    """``startup`` is per TURN on a session, measured from the write to stdin —
    see the comment on the deadline. The first turn pays the CLI's warm-up, so
    a bound that clears that clears every later turn with room to spare."""
    cog = ClaudeCliCognition(timeouts=CliTimeouts(startup=0.05))
    proc = _timed([])

    async def _go() -> Any:
        async with cog.session() as chat:
            return _final(await _timed_turn(chat))

    with _patched(proc):
        result = asyncio.run(_go())

    assert result.evals["stop_reason"] == "startup_timeout"
    assert result.stop_reason == "failed" and result.partial is True


def test_a_startup_bound_crossed_on_a_credential_error_says_so() -> None:
    """The refinement ``drive`` already had, now reachable from a session:
    ``startup_timeout`` alone sends an operator looking at the network when the
    answer is a credential."""
    cog = ClaudeCliCognition(timeouts=CliTimeouts(startup=0.05))
    proc = _timed([], stderr=b"Invalid API key \xc2\xb7 Please run /login")

    async def _go() -> Any:
        async with cog.session() as chat:
            return _final(await _timed_turn(chat))

    with _patched(proc):
        result = asyncio.run(_go())

    assert result.evals["stop_reason"] == "authentication_failed"
    assert result.evals["timeout_kind"] == "startup_timeout", (
        "the refinement must not lose WHICH bound was crossed"
    )


def test_first_event_measures_turn_content_not_the_clis_own_chatter() -> None:
    """A ``system/init`` is the CLI talking to itself. Counting it would let a
    session that emits an init and then hangs satisfy a bound that exists to
    catch exactly that."""
    cog = ClaudeCliCognition(timeouts=CliTimeouts(startup=None, first_event=0.1))
    proc = _timed([(0.0, _line({"type": "system", "subtype": "init", "session_id": "s"}))])

    async def _go() -> Any:
        async with cog.session() as chat:
            return _final(await _timed_turn(chat))

    with _patched(proc):
        result = asyncio.run(_go())

    assert result.evals["stop_reason"] == "first_event_timeout"


def test_total_bounds_the_turn_not_the_conversation() -> None:
    """The one bound whose meaning had to be CHOSEN rather than copied.

    A session exists in order to outlive its turns, so reading ``total`` as a
    ceiling on the whole conversation would kill the thing the class is for.
    Per-turn is the only reading that works: here two turns each finish well
    inside ``total`` while the conversation as a whole runs past it."""
    # Each turn is two lines 0.1s apart, so a turn costs ~0.2s against a 0.45s
    # bound; three of them cost ~0.6s, which the WHOLE conversation would fail.
    cog = ClaudeCliCognition(timeouts=CliTimeouts(startup=None, total=0.45))
    proc = _timed([(0.1, b) for turn in "abc" for b in _turn_lines(turn)])

    async def _go() -> tuple[list[Any], float]:
        started = time.monotonic()
        async with cog.session() as chat:
            results = [_final(await _timed_turn(chat, t)) for t in ("first", "second", "third")]
        return results, time.monotonic() - started

    with _patched(proc):
        results, elapsed = asyncio.run(_go())

    assert [r.stop_reason for r in results] == ["complete"] * 3
    assert [r.output for r in results] == ["a", "b", "c"]
    assert elapsed > 0.45, "the conversation did not actually outrun `total`"


def test_a_control_response_counts_as_liveness_not_as_silence() -> None:
    """An in-flight ``interrupt()`` is the CLI talking. Counting its
    acknowledgement as silence would let a stop button trip an idle bound on a
    turn that is working perfectly well — the read loop consumes both off the
    same stdout, so the two are indistinguishable to anything but this rule."""
    ack = _line({"type": "control_response", "response": {"request_id": "nobody", "subtype": "success"}})
    cog = ClaudeCliCognition(timeouts=CliTimeouts(startup=None, idle=0.12))
    # Four gaps of 0.08s: each is inside the bound, their sum is not.
    proc = _timed([(0.08, ack), (0.08, ack), (0.08, ack), *[(0.08, b) for b in _turn_lines("done")]])

    async def _go() -> Any:
        async with cog.session() as chat:
            return _final(await _timed_turn(chat))

    with _patched(proc):
        result = asyncio.run(_go())

    assert result.stop_reason == "complete", "a control response was treated as silence"


def test_a_timed_out_turn_ends_the_session() -> None:
    """Same standing as a cancel, for the same reason: the process is gone, so
    there is no conversation left to continue. A later turn must refuse rather
    than silently start a fresh one with no history."""
    cog = ClaudeCliCognition(timeouts=CliTimeouts(startup=0.05))
    proc = _timed([])

    async def _go() -> tuple[Any, Any]:
        async with cog.session() as chat:
            first = _final(await _timed_turn(chat, "first"))
            second = _final(await _timed_turn(chat, "second"))
            return first, second

    with _patched(proc):
        first, second = asyncio.run(_go())

    assert first.evals["stop_reason"] == "startup_timeout"
    assert second.evals["stop_reason"] == "session_closed"


def test_timeouts_off_leaves_a_session_unbounded() -> None:
    """The other direction, so the bounds cannot be turned on by accident. A
    default that kills a working conversation is worse than the hang it
    prevents."""
    cog = ClaudeCliCognition(timeouts=CliTimeouts.off())
    proc = _timed([(0.15, b) for b in _turn_lines("slow")])

    async def _go() -> Any:
        async with cog.session() as chat:
            return _final(await _timed_turn(chat))

    with _patched(proc):
        result = asyncio.run(_go())

    assert result.stop_reason == "complete"
