"""One `claude` process, many turns — the lifecycle half of `ClaudeCliSession`.

WHY THIS FILE EXISTS
--------------------
A session is the one CLI path where the process OUTLIVES the call. Everything
that follows from that is invisible to an assertion about an answer, and every
one of the properties below has a failure mode that still returns plausible
text:

* a turn that reads to EOF instead of to its own ``result`` payload hangs the
  conversation on turn two, or hands turn one turn two's answer;
* two unserialised turns interleave into a single transcript and each gets the
  other's reply, with no error anywhere;
* a turn written into a process that has already exited reports success for an
  answer nobody produced;
* a cancelled turn that forgets to terminate leaves a ``claude`` — and whatever
  it forked — alive in ``ps`` for the rest of the service's life, holding half
  an answer in a context nobody will read;
* an interrupt whose acknowledgement is never routed, or whose waiter is never
  failed, hangs a UI's stop button rather than returning a receipt.

The double is :class:`FakeClaudeCli`, injected as ``spawn=``, so no binary runs
and no network is touched. Its one surprise is documented on the class and
relied on below: reading a replayed line never awaits, so a task that must run
mid-turn (an interrupt from a stop button) only gets scheduled when the loop
consuming ``turn()`` yields — hence the ``await asyncio.sleep(0)`` in those
tests. It is the double's shape, not a timing hack.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agentkit.core.testing.fakes import CliRun, FakeClaudeCli
from agentkit.models.providers.cli.claude.driver import ClaudeCliCognition

# ── stream-json payload vocabulary ───────────────────────────────────────────
# Written out rather than recorded: every shape here is one the binary emits
# and none of them is worth a fixture file, because the interesting part is
# always the ORDER they arrive in.


def _init(**extra: Any) -> dict[str, Any]:
    """The CLI's ``system/init`` greeting, the first line of every real turn."""
    return {"type": "system", "subtype": "init", "session_id": "sess-1", **extra}


def _assistant(text: str, message_id: str = "msg-1") -> dict[str, Any]:
    """A completed assistant message — the authoritative copy of its text."""
    return {
        "type": "assistant",
        "message": {"id": message_id, "role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _result(text: str, **extra: Any) -> dict[str, Any]:
    """A turn's own end-of-turn payload. A session STOPS here; it does not read
    on to EOF, because the process has the next turn to serve."""
    return {
        "type": "result",
        "subtype": "success",
        "result": text,
        "session_id": "sess-1",
        "usage": {"input_tokens": 7, "output_tokens": 11},
        "total_cost_usd": 0.0001,
        **extra,
    }


def _control_response(request_id: str = "agentkit-1", **response: Any) -> dict[str, Any]:
    """The CLI's answer to a control request, which arrives on the SAME stdout
    the turn reader is draining — the reason routing it is not optional."""
    return {
        "type": "control_response",
        "response": {"subtype": "success", "request_id": request_id, "response": dict(response)},
    }


def _stdin_messages(cli: FakeClaudeCli) -> list[dict[str, Any]]:
    """Everything the session wrote to the CLI, parsed.

    The input side of a session reaches neither the filesystem nor the event
    stream, so this is the only way to assert that something was NOT sent —
    which is the whole content of several guarantees below.
    """
    return [json.loads(line) for line in cli.invocations[0].stdin.splitlines() if line.strip()]


class _StubAdapter:
    """The output-schema surface ``_resolve_json_schema`` reads off an agent."""

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

    def validate(self, value: Any) -> Any:
        return value


class _StubAgent:
    """An agent-like with the two attributes this cognition reads: the system
    prompt (fixed at spawn) and the output adapter (``--json-schema``, also
    fixed at spawn). ``Agent`` is ``Any`` here, so a structural stub is the
    honest spelling rather than a shortcut."""

    def __init__(self, *, prompt: str = "", adapter: _StubAdapter | None = None) -> None:
        self.prompt = prompt
        self._output_adapter = adapter


class _StopButton:
    """A ``Ctx``-shaped stub carrying nothing but cooperative cancellation.

    The session polls ``check_cancelled()`` once per stdout line (and on a
    timer), which is the ONLY cancellation route through a replayed process —
    see ``FakeClaudeCli``'s docstring on why ``wait_for`` cannot fire
    mid-stream.
    """

    def __init__(self) -> None:
        self.pressed = False

    def check_cancelled(self) -> None:
        if self.pressed:
            raise RuntimeError("the caller pressed stop")


# ── spawn / argv ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_session_carries_its_turns_over_stdin_and_never_in_argv() -> None:
    """A session's process is spawned before anyone has typed a turn, so argv
    must describe a stream and nothing else.

    Without ``--input-format stream-json`` the CLI has no way to receive a
    second turn at all; and with the prompt ALSO passed positionally the binary
    runs a first turn nobody asked for — spending money and poisoning the
    conversation context before the caller's own turn arrives. The empty
    positional is what that mistake looks like from here.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("hi"), _result("hi")])])
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        assert [ev.type async for ev in chat.turn("hello")][-1] == "final"

    argv = cli.invocations[0].argv
    assert "--input-format" in argv, argv
    assert argv[argv.index("--input-format") + 1] == "stream-json", argv
    assert "" not in argv, f"the task was passed as an argv entry as well: {argv}"
    assert _stdin_messages(cli)[0]["message"]["content"] == "hello"


# ── turn boundaries ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_turn_stops_at_its_own_result_so_the_next_turn_reads_its_own_answer() -> None:
    """A turn ends at the CLI's ``result`` payload, NOT at EOF.

    EOF is the one-shot drive's terminator; a session's process stays alive to
    serve the next turn, so reading to EOF consumes every later turn's output
    too. The conversation then either hangs waiting for bytes that will never
    come or — exactly as here — hands turn one turn two's answer and refuses
    turn two outright. Both are indistinguishable from a healthy session until
    somebody reads the text.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    _assistant("first", "msg-1"),
                    _result("first"),
                    _assistant("second", "msg-2"),
                    _result("second"),
                ]
            )
        ]
    )
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        first = [ev async for ev in chat.turn("one")][-1].result
        second = [ev async for ev in chat.turn("two")][-1].result

    assert first.output == "first", first.evals
    assert second.output == "second", second.evals
    assert (first.partial, second.partial) == (False, False)
    assert cli.spawns == 1, "both turns must share one process, or the session buys nothing"


@pytest.mark.asyncio
async def test_two_concurrent_turns_are_serialised_rather_than_interleaved() -> None:
    """One stdin and one transcript mean a second concurrent turn cannot share
    them.

    Unserialised, both turns write immediately and then race the same stdout:
    measured on this exact stream, each one returns the OTHER's answer with
    ``partial=False`` and no error anywhere. A lock makes the second caller
    wait, which is the only outcome that keeps a conversation a conversation.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    _assistant("one-answer", "msg-1"),
                    _result("one-answer"),
                    _assistant("two-answer", "msg-2"),
                    _result("two-answer"),
                ]
            )
        ]
    )
    answers: dict[str, Any] = {}
    async with ClaudeCliCognition(spawn=cli).session() as chat:

        async def ask(task: str) -> None:
            async for ev in chat.turn(task):
                if ev.type == "final":
                    answers[task] = ev.result
                # The double never suspends on a read, so this is what gives
                # the other task a chance to interleave if nothing stops it.
                await asyncio.sleep(0)

        await asyncio.gather(ask("one"), ask("two"))

    assert answers["one"].output == "one-answer", "a turn was handed the other turn's answer"
    assert answers["two"].output == "two-answer", "a turn was handed the other turn's answer"
    assert [m["message"]["content"] for m in _stdin_messages(cli)] == ["one", "two"]


# ── a dead process stays dead ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_turn_is_never_written_into_a_process_that_has_already_exited() -> None:
    """The CLI can exit between turns — a crash, an OOM kill, ``--max-turns``
    reached — and its conversation context goes with it.

    Writing the turn anyway puts a user message into a pipe nobody is reading
    and then waits for an answer that cannot arrive. The pre-flight check is
    what turns that into an immediate ``session_closed`` naming the exit code,
    so the caller knows to open a new session instead of retrying into a void.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("first"), _result("first")])])
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        assert [ev async for ev in chat.turn("one")][-1].result.output == "first"
        # The recorded process exits of its own accord, as a real CLI does.
        cli.invocations[0].returncode = 0

        refused = [ev async for ev in chat.turn("second")][-1].result

    assert refused.evals["stop_reason"] == "session_closed"
    assert "exited with code 0" in refused.evals["error"]
    assert [m["message"]["content"] for m in _stdin_messages(cli)] == ["one"], (
        "the turn was written into a process that had already exited"
    )


@pytest.mark.asyncio
async def test_a_cli_that_closes_its_output_mid_turn_ends_the_conversation() -> None:
    """stdout reaching EOF before the turn's ``result`` is the CLI dying
    mid-answer.

    Treated as anything softer, the turn reports whatever text had arrived as a
    finished answer — the worst shape a failure can take, because nothing
    downstream can tell it from a complete one. It is reported as
    ``session_closed`` rather than ``spawn_failed`` or ``parse_failed``: nothing
    failed to start and nothing was malformed, a conversation ended, and the
    fix is a new session rather than a retry.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("half an ans")])])
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        result = [ev async for ev in chat.turn("go")][-1].result

    assert result.evals["stop_reason"] == "session_closed", result.evals
    assert result.partial is True
    assert "mid-turn" in result.evals["error"]


@pytest.mark.asyncio
async def test_a_closed_session_refuses_a_later_turn_instead_of_starting_over() -> None:
    """Once the conversation is over, every later turn must say so.

    A session that accepts one silently begins a FRESH conversation with no
    history — the caller's next question is answered by a model that never saw
    the first ten, and the only evidence is an answer that reads oddly. Refusing
    costs a turn; accepting costs the transcript.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("half an ans")])])
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        dead = [ev async for ev in chat.turn("go")][-1].result
        assert dead.evals["stop_reason"] == "session_closed"

        refused = [ev async for ev in chat.turn("still there?")][-1].result

    assert refused.evals["stop_reason"] == "session_closed"
    assert "the CLI session is closed" in refused.evals["error"]
    assert [m["message"]["content"] for m in _stdin_messages(cli)] == ["go"], (
        "a closed session wrote a turn and would have answered it from a new conversation"
    )


@pytest.mark.asyncio
async def test_per_turn_structured_output_is_refused_and_the_session_survives_it() -> None:
    """``--json-schema`` is fixed at spawn, so a turn cannot turn it on.

    Accepting the turn returns prose for an ``output=`` the caller believes is
    wired, and ``parsed`` comes back ``None`` with ``partial=False`` — a typed
    run that silently is not one. The refusal is reported as ``turn_refused``
    rather than ``session_closed`` because nothing was written and the process
    never heard about it: the conversation is intact and the very next turn
    answers normally, which is the half that used to be wrong.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("prose"), _result("prose")])])
    typed = _StubAgent(adapter=_StubAdapter())
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        refused = [ev async for ev in chat.turn("invoice me", agent=typed)][-1].result
        assert [m["message"]["content"] for m in _stdin_messages(cli)] == [], (
            "the refused turn reached the CLI"
        )

        after = [ev async for ev in chat.turn("plain question")][-1].result

    assert refused.evals["stop_reason"] == "turn_refused", refused.evals
    assert refused.parsed is None
    assert "process-level flag" in refused.evals["error"]
    assert after.output == "prose", "a per-turn refusal killed a healthy conversation"
    assert after.partial is False


# ── process lifecycle ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_cancelled_turn_terminates_the_cli_rather_than_leaving_it_alive() -> None:
    """No protocol message retracts a half-finished turn, so a cancel ends the
    process with it.

    Leaving it running is the expensive failure, twice over: the session's
    context now contains half an answer nobody saw, and the ``claude`` — plus
    whatever it forked, a ``Bash(npm install)`` being the usual case — stays in
    ``ps`` holding its pipe until it finishes work the caller has already
    abandoned. A session's process is the long-lived one, so the leak lasts.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("half"), _result("half")])])
    stop = _StopButton()
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        events = []
        async for ev in chat.turn("long one", ctx=stop):
            events.append(ev)
            if ev.type == "message_delta":
                stop.pressed = True

    result = events[-1].result
    assert result.evals["stop_reason"] == "cancelled"
    assert result.partial is True
    assert cli.invocations[0].terminated is True, "the cancelled turn left its CLI running"
    assert cli.invocations[0].returncode == -15


@pytest.mark.asyncio
async def test_closing_a_session_ends_the_conversation_over_stdin_not_with_a_signal() -> None:
    """Closing stdin IS the protocol's end-of-conversation signal; the CLI
    exits 0 on it.

    Skipping it means every orderly ``close()`` degrades into the kill path
    after the terminate grace — so a service that opens and closes sessions
    normally pays the grace period every time and reports signal deaths for
    conversations that ended perfectly well. The exit code is the whole
    diagnostic value of the field.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("hi"), _result("hi")])])
    spawned: list[Any] = []

    async def spawn(*argv: str, **kwargs: Any) -> Any:
        proc = await cli(*argv, **kwargs)
        spawned.append(proc)
        return proc

    chat = ClaudeCliCognition(spawn=spawn).session()
    await chat.start()
    assert [ev async for ev in chat.turn("hi")][-1].result.output == "hi"
    await chat.close()

    assert spawned[0].stdin.is_closing() is True, "close() never signalled end-of-conversation"
    assert cli.invocations[0].terminated is False
    assert cli.invocations[0].killed is False
    assert cli.invocations[0].returncode == 0


# ── interrupt ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_interrupt_acknowledgement_reaches_its_waiter_and_not_the_answer() -> None:
    """A ``control_response`` arrives on the same stdout the turn is draining,
    so the turn reader is the only thing that can route it.

    Unrouted, it is protocol bookkeeping sitting in the middle of turn content
    — and the waiter it belongs to is never resolved, so a UI's stop button
    reports that nothing was stopped on a turn it did in fact stop.
    ``still_queued`` is the field that makes the receipt worth having: it is
    the difference between "the agent stopped" and "the agent stopped and has
    three more things to do".
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(capabilities=["interrupt_cancel_queued_v1"]),
                    _assistant("half an ans"),
                    _control_response(still_queued=["deploy", "notify"]),
                    _result("half an ans", is_error=True, subtype="error_during_execution"),
                ]
            )
        ]
    )
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        stop: asyncio.Task[Any] | None = None
        events = []
        async for ev in chat.turn("do the thing"):
            events.append(ev)
            if ev.type == "message_delta" and stop is None:
                stop = asyncio.create_task(chat.interrupt())
            await asyncio.sleep(0)  # let the stop button's task run
        assert stop is not None
        receipt = await stop

    assert receipt.delivered is True, receipt
    assert receipt.still_queued == ("deploy", "notify"), receipt
    assert receipt.error is None
    assert "control_response" not in events[-1].result.output


@pytest.mark.asyncio
async def test_an_interrupted_turn_reports_that_somebody_stopped_it() -> None:
    """The CLI ends an interrupted turn as ``error_during_execution`` — the same
    subtype it uses for a genuine execution failure.

    The payload alone cannot tell the two apart; WE can, because we sent the
    interrupt. Without that stamp a stop button produces a run that looks like
    the CLI broke, which is the difference between a deliberate stop and an
    incident. ``partial`` stays true either way: the answer is a fragment.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    _assistant("half an ans"),
                    _control_response(),
                    _result("half an ans", is_error=True, subtype="error_during_execution"),
                ]
            )
        ]
    )
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        stop: asyncio.Task[Any] | None = None
        events = []
        async for ev in chat.turn("do the thing"):
            events.append(ev)
            if ev.type == "message_delta" and stop is None:
                stop = asyncio.create_task(chat.interrupt())
            await asyncio.sleep(0)
        assert stop is not None
        await stop

    result = events[-1].result
    assert result.evals["stop_reason"] == "interrupted", result.evals
    assert result.stop_reason == "interrupted"
    assert result.partial is True
    assert result.output == "half an ans"


@pytest.mark.asyncio
async def test_interrupting_an_idle_session_sends_nothing_and_stops_nothing() -> None:
    """"Stop" with nothing running is a no-op, and must stay one.

    The acknowledgement only ever arrives while a turn is draining stdout, so a
    control request sent between turns is a question nobody will answer: the
    caller waits out the ack timeout for a receipt that reports a failure that
    never happened, and the CLI is left holding a stray request that may be
    answered into the NEXT turn's stream. An empty receipt is the honest
    outcome, and it is not an exception — otherwise every UI wraps its stop
    button in a try block.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("done"), _result("done")])])
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        assert [ev async for ev in chat.turn("question")][-1].result.output == "done"

        receipt = await chat.interrupt(ack_timeout_s=0.05)

    assert receipt.delivered is False, receipt
    assert receipt.still_queued == ()
    assert [m["type"] for m in _stdin_messages(cli)] == ["user"], (
        "an idle session sent a control request nobody would ever answer"
    )


@pytest.mark.asyncio
async def test_an_interrupt_the_turn_outlives_is_failed_rather_than_stranded() -> None:
    """A stop button raced by the last token is the normal case, not an error.

    The reader that resolves a control request exists only while a turn is in
    flight, so a turn that ends first leaves the waiter with nobody to answer
    it — forever, unless the end of the turn fails it. The receipt says
    ``delivered=False`` and explains why; it used to raise the module-private
    ``_SessionClosed`` straight out of the caller's stop button, which is
    precisely the try block this method exists to spare a UI. The interrupt
    flag is still set, so the turn itself reports who stopped it.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    _assistant("all done"),
                    # No control_response: the turn finishes first.
                    _result("all done", is_error=True, subtype="error_during_execution"),
                ]
            )
        ]
    )
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        stop: asyncio.Task[Any] | None = None
        events = []
        async for ev in chat.turn("do the thing"):
            events.append(ev)
            if ev.type == "message_delta" and stop is None:
                stop = asyncio.create_task(chat.interrupt(ack_timeout_s=30.0))
            await asyncio.sleep(0)
        assert stop is not None
        receipt = await asyncio.wait_for(stop, timeout=5.0)

    assert receipt.delivered is False, receipt
    assert "already finished" in (receipt.error or ""), receipt
    assert events[-1].result.evals["stop_reason"] == "interrupted"


@pytest.mark.asyncio
async def test_an_inline_interrupt_degrades_to_a_receipt_rather_than_hanging() -> None:
    """Called from inside the loop consuming ``turn()``, nothing is draining
    stdout — so the acknowledgement cannot arrive.

    That is a wiring mistake a caller makes easily (the documented shape is
    ``asyncio.create_task(chat.interrupt())``), and unbounded it deadlocks the
    application: the consumer is suspended at a yield waiting for the
    interrupt, and the interrupt is waiting for the consumer to read its
    answer. Bounding it costs a field on a receipt instead. The interrupt still
    takes effect, because the WRITE already happened — hence
    ``delivered=True``.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    _assistant("half an ans"),
                    _result("half an ans", is_error=True, subtype="error_during_execution"),
                ]
            )
        ]
    )
    receipts: list[Any] = []
    hung: list[BaseException] = []
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        async for ev in chat.turn("do the thing"):
            if ev.type == "message_delta" and not receipts and not hung:
                try:
                    receipts.append(await asyncio.wait_for(chat.interrupt(ack_timeout_s=0.05), timeout=5.0))
                except TimeoutError as exc:  # pragma: no cover — the mutant's path
                    hung.append(exc)

    assert not hung, "an inline interrupt hung the caller instead of degrading"
    assert receipts[0].delivered is True, receipts[0]
    assert "no acknowledgement" in (receipts[0].error or ""), receipts[0]
    assert "create_task" in (receipts[0].error or ""), "the receipt must name the way out"


@pytest.mark.asyncio
async def test_the_clis_advertised_capabilities_are_visible_while_a_turn_runs() -> None:
    """Capabilities are folded the moment ``system/init`` arrives, not at the
    terminal event.

    ``interrupt()`` is called MID-turn and feature-detects against them, so a
    capability that only lands when the turn ends is one nobody can act on:
    every mid-turn ``cancel_queued`` is refused as unsupported by a CLI that
    supports it. Feature-detect against this rather than a version string —
    the set is open, so an unrecognised entry is not an error.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(capabilities=["interrupt_cancel_queued_v1", "something_new_v2"]),
                    _assistant("hi"),
                    _result("hi"),
                ]
            )
        ]
    )
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        assert chat.capabilities == frozenset(), "nothing is known before the CLI has spoken"
        mid_turn: list[frozenset[str]] = []
        async for ev in chat.turn("hi"):
            if ev.type == "message_delta":
                mid_turn.append(chat.capabilities)
                assert chat.supports("interrupt_cancel_queued_v1") is True

    assert mid_turn == [frozenset({"interrupt_cancel_queued_v1", "something_new_v2"})]


@pytest.mark.asyncio
async def test_cancel_queued_is_refused_before_it_is_sent_to_a_cli_that_ignores_it() -> None:
    """``cancel_queued`` is checked against the advertised capability BEFORE the
    request goes out.

    A CLI that does not understand the subtype does not complain about it — it
    ignores the field, stops the current turn, and runs every queued message
    anyway. So the caller who asked for "stop, and drop the rest" gets "stop",
    with a receipt that looks successful and three more things about to happen.
    Raising names the capability and the two ways out.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("half"), _result("half")])])
    refusals: list[BaseException] = []
    receipts: list[Any] = []
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        async for ev in chat.turn("do the thing"):
            if ev.type == "message_delta":
                try:
                    receipts.append(await chat.interrupt(cancel_queued=True, ack_timeout_s=0.05))
                except ValueError as exc:
                    refusals.append(exc)

    assert receipts == [], "cancel_queued was sent to a CLI that never advertised it"
    assert len(refusals) == 1
    assert "interrupt_cancel_queued_v1" in str(refusals[0])
    assert b"control_request" not in cli.invocations[0].stdin


@pytest.mark.asyncio
async def test_cancel_queued_asks_for_its_own_subtype_so_queued_work_is_dropped() -> None:
    """The plain ``interrupt`` subtype stops the current turn and nothing else.

    Sending it for a ``cancel_queued=True`` call is the quietest possible
    failure: the receipt comes back ``delivered=True``, the turn does stop, and
    the queued messages the caller asked to drop all run. Only the wire subtype
    distinguishes the two intents, which is why it is asserted on the bytes.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(capabilities=["interrupt_cancel_queued_v1"]),
                    _assistant("half an ans"),
                    _control_response(cancelled=["queued-1"], still_queued=[]),
                    _result("half an ans", is_error=True, subtype="error_during_execution"),
                ]
            )
        ]
    )
    async with ClaudeCliCognition(spawn=cli).session() as chat:
        stop: asyncio.Task[Any] | None = None
        async for ev in chat.turn("do the thing"):
            if ev.type == "message_delta" and stop is None:
                stop = asyncio.create_task(chat.interrupt(cancel_queued=True))
            await asyncio.sleep(0)
        assert stop is not None
        receipt = await stop

    requests = [m for m in _stdin_messages(cli) if m["type"] == "control_request"]
    assert [m["request"]["subtype"] for m in requests] == ["interrupt_cancel_queued"], requests
    assert receipt.delivered is True
    assert receipt.cancelled == ("queued-1",), receipt


@pytest.mark.asyncio
async def test_a_session_opened_with_an_agents_schema_spawns_with_it_and_runs_typed() -> None:
    """The other half of the refusal above, and the half that was broken.

    ``--json-schema`` is fixed at spawn, so the ONLY way a session can be typed
    is for the schema to be on the spawn argv — and ``start`` never built it.
    A cognition carrying ``json_schema=``, or a ``session(agent=...)`` whose agent
    declares ``output=``, therefore spawned a process that could not honour it and
    then refused EVERY turn, advising the caller to "pass json_schema= on the
    cognition before opening the session" — the exact thing they had just done.
    The class docstring promises a session has "the same structured-output
    handling" as ``drive``; this is what makes that true.
    """
    typed = _StubAgent(adapter=_StubAdapter())
    cli = FakeClaudeCli(
        [CliRun.of([_init(), _assistant("prose"), _result("prose", structured_output={"answer": "42"})])]
    )
    async with ClaudeCliCognition(spawn=cli).session(agent=typed) as chat:
        result = [ev async for ev in chat.turn("answer me", agent=typed)][-1].result

    argv = cli.invocations[0].argv
    assert "--json-schema" in argv, "the session spawned without the schema it was opened with"
    sent = json.loads(argv[argv.index("--json-schema") + 1])
    assert sent["properties"] == {"answer": {"type": "string"}}, sent
    assert sent["required"] == ["answer"]
    assert "stop_reason" not in result.evals, (
        f"the turn was refused rather than run: {result.evals}"
    )
    assert result.parsed == {"answer": "42"}
    assert result.partial is False


@pytest.mark.asyncio
async def test_a_turn_may_not_swap_the_schema_the_session_was_spawned_with() -> None:
    """Matching is what is permitted, not merely "asked for something". The flag
    is one process-level value, so a turn carrying a DIFFERENT schema cannot be
    honoured and must be refused rather than validated against the wrong one —
    which would reject every correct answer the model gave."""

    class _OtherAdapter(_StubAdapter):
        def json_schema(self) -> dict[str, Any]:
            return {"type": "object", "properties": {"other": {"type": "integer"}}}

    opened = _StubAgent(adapter=_StubAdapter())
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("prose"), _result("prose")])])
    async with ClaudeCliCognition(spawn=cli).session(agent=opened) as chat:
        refused = [
            ev async for ev in chat.turn("mismatch", agent=_StubAgent(adapter=_OtherAdapter()))
        ][-1].result

    assert refused.evals["stop_reason"] == "turn_refused", refused.evals
    assert "different schema" in refused.evals["error"]
