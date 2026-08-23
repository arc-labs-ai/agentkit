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
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent, Scope
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.agents.cognition.claude_cli import _user_turn
from agentkit.kernel.types import StreamEvent
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
