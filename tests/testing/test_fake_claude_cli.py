"""`FakeClaudeCli` — the offline double for the `claude` CLI path.

Every other fake in `agentkit.testing` fakes a PORT. The CLI is not behind a
port: `ClaudeCliCognition` spawns a binary and parses its stream-json stdout,
so a test of anything CLI-shaped either spent real money or stood up a real
`claude` binary. Measured on the suite before this file existed: 6 of the 48
skips in a clean `pytest -q` run were CLI tests that need the binary, and the
three most interesting failure modes on that path — a truncated line, a
non-zero exit mid-answer, a `result` payload that never arrives — had no test
at all, because nothing could produce them.

WHAT THIS FILE IS DEFENDING, in one sentence: the double must sit at the
subprocess seam, not above it. A double that hands back a finished
`AgentResult` tests nothing that has ever gone wrong on this path — every bug
this cognition has shipped lived in the line parser, the payload→event
mapping, the stop-reason priority, or the meter charge, and all four are
downstream of the seam.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agentkit import Agent, Scope, Usage
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.context import WorkingContext
from agentkit.runtime import Budget, RunContext, Services
from agentkit.testing.fakes import CliRun, CliStderr, FakeClaudeCli, ScriptExhausted
from agentkit.testing.fakes.ctx import FakeCtx
from tests._assertions import assert_money

RECORDING = Path(__file__).parent.parent / "agents" / "cognition" / "sessions" / "adds_endpoint.jsonl"

_ANSWER_TAIL = "`tests/test_routes.py` passes (4 tests)."


async def _drive(cog: ClaudeCliCognition, ctx: Any = None, task: str = "add a GET /orders/{id}") -> list[Any]:
    agent = Agent(name="local", cognition=cog)
    return [ev async for ev in cog.drive(agent, task, ctx or FakeCtx(), WorkingContext())]


def _shape(events: list[Any]) -> list[tuple[str, str]]:
    """The event stream reduced to what a consumer actually sees, so two
    replays can be compared for byte-equality without comparing object ids."""
    out: list[tuple[str, str]] = []
    for ev in events:
        if ev.type == "final":
            r = ev.result
            out.append(("final", json.dumps([r.output, r.partial, r.stop_reason, r.evals], sort_keys=True, default=str)))
        elif ev.tool_call is not None:
            out.append((ev.type, f"{ev.tool_call.id}:{ev.tool_call.name}:{json.dumps(ev.tool_call.arguments, sort_keys=True)}:{ev.tool_result}"))
        else:
            out.append((ev.type, ev.text))
    return out


# ── 1. the double sits at the seam the real one does ─────────────────────────


@pytest.mark.asyncio
async def test_a_recorded_session_is_replayed_through_the_real_parser() -> None:
    """The recording is a real dispatch's stdout. Everything in the assertions
    below was PRODUCED by `ClaudeCliCognition` from those bytes — the double
    supplies bytes and nothing else, so this test covers `_parse_line`,
    `_events_from_payload`, `_TurnState.fold` and `_finalise` exactly as a run
    against the binary would."""
    cli = FakeClaudeCli.replay(RECORDING)
    events = await _drive(ClaudeCliCognition(spawn=cli))

    kinds = [ev.type for ev in events]
    assert kinds.count("tool_call") == 3  # Read, Edit, Bash
    assert kinds.count("tool_result") == 3
    assert kinds.count("step") == 1  # the system/api_retry payload
    assert kinds.count("final") == 1

    calls = [ev.tool_call.name for ev in events if ev.type == "tool_call"]
    assert calls == ["Read", "Edit", "Bash"]

    result = events[-1].result
    assert result.output.endswith(_ANSWER_TAIL)
    assert result.partial is False
    assert result.stop_reason == "complete"
    # Straight off the recording's `result` payload — nothing here is invented
    # by the double.
    assert result.usage == Usage(
        input_tokens=27, output_tokens=635, cost_usd=0.28414, cache_read_tokens=43828, cache_write_tokens=16276
    )
    assert result.evals["session_id"] == "a3f1c0de-7b42-4f19-9c88-2e6d5a1b0f37"
    assert result.evals["cli_duration_ms"] == 18432
    assert result.evals["cli_init"]["model"] == "claude-opus-4-5-20251101"
    assert result.evals["api_retries"][0]["error"] == "overloaded_error"


@pytest.mark.asyncio
async def test_the_replayed_cost_is_charged_to_the_frameworks_meters() -> None:
    """The budget charge is downstream of the seam, so a double that returned
    a finished `AgentResult` would skip it entirely — and this is the exact
    mechanism that once read $0.00 for a $50 CLI run."""
    budget = Budget(max_cost_usd=10.0)
    ctx = RunContext("run-1", Scope(), services=Services(), budget=budget)
    cli = FakeClaudeCli.replay(RECORDING)

    await _drive(ClaudeCliCognition(spawn=cli), ctx)

    assert_money(budget.spent(), "0.284140")
    assert budget.usage.total_tokens == 662


@pytest.mark.asyncio
async def test_the_argv_the_cognition_built_is_visible_on_the_double() -> None:
    """The seam records what was spawned, so a flags test no longer has to
    patch `asyncio.create_subprocess_exec` — which resolves through the
    module's `asyncio` reference to the REAL `asyncio` module and therefore
    disables subprocess spawning process-wide for the duration of the patch."""
    cli = FakeClaudeCli.replay(RECORDING)
    await _drive(ClaudeCliCognition(spawn=cli, model="claude-opus-4-5", max_turns=4))

    inv = cli.invocations[0]
    assert inv.argv[0] == "claude"
    assert "--output-format" in inv.argv and inv.argv[inv.argv.index("--output-format") + 1] == "stream-json"
    assert inv.argv[inv.argv.index("--model") + 1] == "claude-opus-4-5"
    assert inv.argv[inv.argv.index("--max-turns") + 1] == "4"
    assert inv.env["CLAUDE_ENABLE_STREAM_WATCHDOG"] == "1"
    assert inv.env["CLAUDE_TRACE_EXTERNAL_ID"] == "fake-run"  # ctx.correlation_id, bridged


@pytest.mark.asyncio
async def test_the_seam_is_per_instance_so_one_process_can_hold_two() -> None:
    """Instance-scoped, unlike a module patch: two cognitions in the same
    process, only one of them faked. A `patch(...)` on the spawn call would
    have caught both."""
    cli = FakeClaudeCli.replay(RECORDING)
    faked = ClaudeCliCognition(spawn=cli)
    untouched = ClaudeCliCognition(claude_bin="definitely-not-on-path-zzz")

    await _drive(faked)
    events = await _drive(untouched)

    assert cli.invocations and len(cli.invocations) == 1
    # The un-faked one really tried to spawn and really failed.
    assert events[-1].result.evals["stop_reason"] == "spawn_failed"


# ── 2. a recording replays byte-identically, twice ───────────────────────────


@pytest.mark.asyncio
async def test_two_doubles_over_one_recording_produce_identical_streams() -> None:
    """No clock, no randomness, no leakage between instances."""
    a = await _drive(ClaudeCliCognition(spawn=FakeClaudeCli.replay(RECORDING)))
    b = await _drive(ClaudeCliCognition(spawn=FakeClaudeCli.replay(RECORDING)))
    assert _shape(a) == _shape(b)


@pytest.mark.asyncio
async def test_one_double_replayed_twice_does_not_consume_its_recording() -> None:
    """The sharper version: the SAME double, spawned twice. A double that
    handed its buffer to the first reader would answer the second with an
    empty stream and a plausible-looking `final` event carrying no output."""
    cli = FakeClaudeCli.replay(RECORDING, repeat_last=True)
    cog = ClaudeCliCognition(spawn=cli)
    assert _shape(await _drive(cog)) == _shape(await _drive(cog))
    assert len(cli.invocations) == 2


@pytest.mark.asyncio
async def test_two_doubles_can_be_read_interleaved() -> None:
    """No shared mutable state, checked with the two reads ACTUALLY overlapping.

    `asyncio.gather` alone does not overlap them. Reading a line from the
    double never awaits, so a gathered task runs to EOF before the other one
    starts — measured, 100 deltas from two runs with a single changeover, i.e.
    exactly the sequential test this was meant to be stronger than. The
    consumer has to be the thing that yields; then it alternates on every
    event, and a cursor shared between the two doubles would tear.
    """
    one, two = FakeClaudeCli.replay(RECORDING), FakeClaudeCli.replay(RECORDING)
    order: list[str] = []

    async def _read(tag: str, cli: FakeClaudeCli) -> list[Any]:
        cog = ClaudeCliCognition(spawn=cli)
        out = []
        async for ev in cog.drive(Agent(name=tag, cognition=cog), "t", FakeCtx(), WorkingContext()):
            out.append(ev)
            order.append(tag)
            await asyncio.sleep(0)  # the suspension point the double does not have
        return out

    a, b = await asyncio.gather(_read("a", one), _read("b", two))
    assert _shape(a) == _shape(b)
    # The reads really overlapped: without this the assertion above passes on
    # two runs that never coexisted.
    changeovers = sum(1 for i in range(1, len(order)) if order[i] != order[i - 1])
    assert changeovers > 1, f"the two reads did not overlap: {''.join(order)}"


# ── 3. exhaustion is loud ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_asking_for_a_run_the_recording_does_not_have_raises() -> None:
    """A recording is a claim about how many times the CLI is spawned, exactly
    as a `FakeLLM.script` is a claim about how many turns a run takes. Code
    that spawns twice against a one-spawn recording has a defect — a retry that
    should not have fired, a loop that should have stopped — and replaying the
    recording again would answer it with a stable, plausible, wrong result."""
    cli = FakeClaudeCli.replay(RECORDING)
    cog = ClaudeCliCognition(spawn=cli)
    await _drive(cog)

    with pytest.raises(ScriptExhausted):
        await _drive(cog)


@pytest.mark.asyncio
async def test_exhaustion_escapes_the_cognitions_own_except_baseexception() -> None:
    """THE regression this file exists for.

    `drive()` catches `BaseException` to keep its terminal-event guarantee, and
    that arm sat between `FakeClaudeCli` and the test body. `ScriptExhausted`
    is a `BaseException` precisely so it survives the ~20 deliberate `except`
    sites on the path to a test — and this was the twenty-first. Before the
    fix, an over-spawn came back as a tidy `final` event with
    `stop_reason='spawn_failed'` and the test went green."""
    cli = FakeClaudeCli.replay(RECORDING)
    cog = ClaudeCliCognition(spawn=cli)
    await _drive(cog)

    with pytest.raises(ScriptExhausted) as exc:
        await _drive(cog)
    # And the message carries the two numbers plus the way out.
    text = str(exc.value)
    assert "1" in text and "2" in text and "repeat_last=True" in text


@pytest.mark.asyncio
async def test_the_terminal_event_still_arrives_before_exhaustion_propagates() -> None:
    """Raising must not cost the caller the terminal-event guarantee. The
    consumer sees exactly one `final`, THEN the exception."""
    cli = FakeClaudeCli.replay(RECORDING)
    cog = ClaudeCliCognition(spawn=cli)
    await _drive(cog)

    seen: list[str] = []
    with pytest.raises(ScriptExhausted):
        async for ev in cog.drive(Agent(name="local", cognition=cog), "again", FakeCtx(), WorkingContext()):
            seen.append(ev.type)
    assert seen == ["final"]


@pytest.mark.asyncio
async def test_repeat_last_is_the_documented_way_out() -> None:
    """The escape hatch, spelled as a real keyword so it appears in the
    signature a reader reaches for at the moment the exception fires."""
    cli = FakeClaudeCli.replay(RECORDING, repeat_last=True)
    cog = ClaudeCliCognition(spawn=cli)
    for _ in range(3):
        assert (await _drive(cog))[-1].result.output.endswith(_ANSWER_TAIL)
    assert len(cli.invocations) == 3


@pytest.mark.asyncio
async def test_a_keyboard_interrupt_during_a_drive_reaches_the_caller() -> None:
    """Same fix, wider blast radius. `drive()`'s comment says it widens to
    `BaseException` so `KeyboardInterrupt` and `SystemExit` "also produce a
    terminal event BEFORE PROPAGATING" — and the code never propagated them.
    Ctrl-C during a CLI run was swallowed into `stop_reason='spawn_failed'`
    and the process kept going."""

    async def _boom(*_a: Any, **_kw: Any) -> Any:
        raise KeyboardInterrupt("ctrl-c")

    cog = ClaudeCliCognition(spawn=_boom)
    seen: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        async for ev in cog.drive(Agent(name="local", cognition=cog), "t", FakeCtx(), WorkingContext()):
            seen.append(ev.type)
    assert seen == ["final"]


@pytest.mark.asyncio
async def test_a_keyboard_interrupt_during_a_session_turn_reaches_the_caller() -> None:
    """The SESSION half of the same fix, which is a separate line of code in a
    separate method — `ClaudeCliSession._turn` has its own `except
    BaseException` and its own re-raise past the final yield, and branch
    coverage showed that line never executing. A fix applied twice and tested
    once is a fix that survives in one place.

    Ctrl-C lands on `ctx.check_cancelled()`, which `_turn` guards with `except
    Exception` — deliberately narrow, so a KeyboardInterrupt goes straight to
    the terminal-event arm rather than being filed as an ordinary cancel."""

    class _CtrlC(FakeCtx):  # type: ignore[misc]
        def check_cancelled(self) -> None:
            raise KeyboardInterrupt("ctrl-c")

    cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "partway"}]}},
            {"type": "result", "subtype": "success", "session_id": "s", "duration_ms": 1,
             "total_cost_usd": 0.01, "usage": {}},
        ]
    )
    cog = ClaudeCliCognition(spawn=cli)
    seen: list[str] = []
    async with cog.session() as chat:
        with pytest.raises(KeyboardInterrupt):
            async for ev in chat.turn("go", ctx=_CtrlC()):
                seen.append(ev.type)
    # The guarantee holds on the way out: terminal event first, then the raise.
    assert seen == ["final"]


@pytest.mark.asyncio
async def test_an_ordinary_exception_is_still_reported_as_data() -> None:
    """The counterweight: a real spawn fault (a missing binary, a bad cwd) is
    an `Exception` and must stay data, or every caller of a cognition whose
    whole contract is 'failures arrive as a terminal event' has to grow a try
    block."""

    async def _missing(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("no such file: claude")

    events = await _drive(ClaudeCliCognition(spawn=_missing))
    assert events[-1].result.evals["stop_reason"] == "spawn_failed"
    assert events[-1].result.partial is True


# ── 4. malformed sessions are constructible ──────────────────────────────────


@pytest.mark.asyncio
async def test_a_session_that_ends_mid_json_line() -> None:
    """Truncated output — the CLI was killed while writing its `result`. The
    half-line must be skipped, not crash the loop, and the run must not claim
    the answer it never finished sending."""
    cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "half an "}]}},
            '{"type":"result","subtype":"success","session_id":"s","total_cost',
        ]
    )
    events = await _drive(ClaudeCliCognition(spawn=cli))
    result = events[-1].result
    assert result.output == "half an "
    assert result.usage.cost_usd == 0.0  # the truncated line carried the cost
    assert result.evals["cli_duration_ms"] == 0


@pytest.mark.asyncio
async def test_a_final_line_with_no_trailing_newline_is_still_delivered() -> None:
    """Recordings routinely end without one — a `>` capture of a CLI that
    exits before flushing, a file an editor trimmed. `StreamReader.readline`
    hands the fragment over at EOF and so must the double.

    The failure it guards is quiet: drop the fragment and the `result` payload
    disappears, taking the cost, the usage and the session id with it. The run
    still reports `partial=False` and `stop_reason='complete'` — a green test
    and a ledger reading $0.00."""
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
            '{"type":"result","subtype":"success","session_id":"s","duration_ms":7,'
            '"total_cost_usd":0.05,"usage":{"input_tokens":11,"output_tokens":2}}',
        ]
    )
    result = (await _drive(ClaudeCliCognition(spawn=cli)))[-1].result
    assert result.usage.cost_usd == 0.05
    assert result.evals["session_id"] == "s"
    assert result.evals["cli_duration_ms"] == 7


@pytest.mark.asyncio
async def test_a_line_that_is_valid_json_but_an_unknown_event_type() -> None:
    """Forward-compat is 'do not crash': the CLI adds event types over time and
    a pinned agentkit must survive a newer binary."""
    cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "wombat", "payload": {"anything": [1, 2, 3]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "fine"}]}},
            {"type": "result", "subtype": "success", "session_id": "s", "duration_ms": 3, "total_cost_usd": 0.01, "usage": {}},
        ]
    )
    events = await _drive(ClaudeCliCognition(spawn=cli))
    assert [ev.type for ev in events] == ["message_delta", "final"]
    assert events[-1].result.output == "fine"


@pytest.mark.asyncio
async def test_a_session_with_no_result_event_at_all() -> None:
    """The CLI answered and then exited 0 without its `result` payload. Today
    that reads as a clean success with zero usage and no session id — worth
    pinning as-is, because the moment the cognition starts calling it a
    failure this test is the thing that says so."""
    cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "answered"}]}},
        ]
    )
    result = (await _drive(ClaudeCliCognition(spawn=cli)))[-1].result
    assert result.output == "answered"
    assert result.usage.cost_usd == 0.0
    assert result.partial is False
    assert "stop_reason" not in result.evals


@pytest.mark.asyncio
async def test_a_non_zero_exit_partway_through() -> None:
    """The binary died mid-answer. The text that did arrive is kept, the run is
    `partial`, and the stop reason names the code so an operator can look it
    up rather than guess."""
    cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I was saying"}]}},
        ],
        returncode=2,
    )
    result = (await _drive(ClaudeCliCognition(spawn=cli)))[-1].result
    assert result.output == "I was saying"
    assert result.partial is True
    assert result.evals["stop_reason"] == "cli_exit_2"
    assert result.evals["cli_return_code"] == 2
    assert result.stop_reason == "failed"


@pytest.mark.asyncio
async def test_stderr_interleaved_with_stdout_is_all_there_at_the_end() -> None:
    """stderr is read ONCE, after stdout hits EOF — so a diagnostic emitted in
    the middle of a run has to survive until the terminal event is built, and
    must not disturb the stdout stream on its way. It is the only channel that
    explains a bare `cli_exit_1` to an operator."""
    cli = FakeClaudeCli.script(
        [
            CliStderr(b"node:internal/fs: ENOSPC: no space left on device\n"),
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "kept going"}]}},
            CliStderr(b"Aborting.\n"),
        ],
        returncode=1,
    )
    result = (await _drive(ClaudeCliCognition(spawn=cli)))[-1].result
    assert "ENOSPC" in result.evals["stderr"] and "Aborting." in result.evals["stderr"]
    # stdout was unaffected by the writes around it.
    assert result.output == "kept going"
    assert result.evals["stop_reason"] == "cli_exit_1"


@pytest.mark.asyncio
async def test_an_empty_recording() -> None:
    """Zero bytes on stdout, exit 0. Degenerate, and the one a `--version`
    style misconfiguration actually produces."""
    cli = FakeClaudeCli.script([])
    events = await _drive(ClaudeCliCognition(spawn=cli))
    assert [ev.type for ev in events] == ["final"]
    assert events[-1].result.output == ""
    assert events[-1].result.evals["session_id"] == ""


@pytest.mark.asyncio
async def test_blank_and_non_json_diagnostic_lines_are_skipped() -> None:
    """The real binary emits a blank warm-up line and the occasional plain-text
    warning on stdout. Neither is an event and neither may end the run."""
    cli = FakeClaudeCli.script(
        [
            "\n",
            "warning: config at ~/.claude/settings.json is not valid JSON, ignoring\n",
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
            {"type": "result", "subtype": "success", "session_id": "s", "duration_ms": 1, "total_cost_usd": 0.0, "usage": {}},
        ]
    )
    result = (await _drive(ClaudeCliCognition(spawn=cli)))[-1].result
    assert result.output == "ok"
    assert result.partial is False


@pytest.mark.asyncio
async def test_a_refusal_nobody_recorded() -> None:
    """`script` covers the cases a recording cannot: the CLI exiting 0 with
    `is_error: true`, which is how a max-turns stop and a refusal both arrive."""
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I can't help with that."}]}},
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "session_id": "s",
                "duration_ms": 900,
                "total_cost_usd": 0.02,
                "usage": {"input_tokens": 40, "output_tokens": 9},
            },
        ]
    )
    result = (await _drive(ClaudeCliCognition(spawn=cli)))[-1].result
    assert result.evals["stop_reason"] == "error_max_turns"
    assert result.partial is True
    # `error_max_turns` is a deliberate stop, not a fault, so it lands on
    # `terminated` in the closed taxonomy rather than `failed` — the `partial`
    # flag and the free-form reason carry the rest.
    assert result.stop_reason == "terminated"


@pytest.mark.asyncio
async def test_a_malformed_final_answer_against_a_declared_schema() -> None:
    """`--json-schema` was passed and the CLI came back `success` with no
    `structured_output`. The docs call that a failure and so must we, or a
    caller who declared `output=` reads prose as if the object was never
    wired."""
    cli = FakeClaudeCli.script(
        [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "{sort of json}"}]}},
            {"type": "result", "subtype": "success", "session_id": "s", "duration_ms": 5, "total_cost_usd": 0.01, "usage": {}},
        ]
    )
    cog = ClaudeCliCognition(spawn=cli, json_schema={"type": "object", "properties": {"n": {"type": "integer"}}})
    result = (await _drive(cog))[-1].result
    assert result.evals["stop_reason"] == "structured_output_missing"
    assert result.partial is True
    assert result.stop_reason == "invalid_output"


# ── 5. large recordings stream ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_large_recording_is_streamed_not_slurped() -> None:
    """20k payloads. If either side buffered the run to completion before
    handing over the first event, `lines_read` at the first `message_delta`
    would be 20_000 rather than 2 — which is the difference between a double
    you can use to test incremental UI rendering and one you cannot."""
    payloads: list[Any] = [{"type": "system", "subtype": "init", "session_id": "s"}]
    payloads += [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": f"chunk{i} "}]}}
        for i in range(20_000)
    ]
    payloads.append(
        {"type": "result", "subtype": "success", "session_id": "s", "duration_ms": 1, "total_cost_usd": 0.0, "usage": {}}
    )
    cli = FakeClaudeCli.script(payloads)
    cog = ClaudeCliCognition(spawn=cli)

    at_first_delta: int | None = None
    deltas = 0
    async for ev in cog.drive(Agent(name="local", cognition=cog), "t", FakeCtx(), WorkingContext()):
        if ev.type == "message_delta":
            deltas += 1
            if at_first_delta is None:
                at_first_delta = cli.invocations[0].lines_read

    assert deltas == 20_000
    assert at_first_delta == 2


# ── 6. sessions: one spawn, many turns ───────────────────────────────────────


@pytest.mark.asyncio
async def test_a_persistent_session_runs_many_turns_from_one_spawn() -> None:
    """A session reads to each turn's `result` payload and parks the stream
    there. Two turns off ONE recording is the thing a `-p` double cannot
    express, and getting the boundary wrong makes turn 2 read turn 1's tail."""

    def _turn(text: str) -> list[Any]:
        return [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
            {"type": "result", "subtype": "success", "session_id": "s", "duration_ms": 2, "total_cost_usd": 0.01, "usage": {"input_tokens": 3, "output_tokens": 4}},
        ]

    cli = FakeClaudeCli.script(
        [{"type": "system", "subtype": "init", "session_id": "s"}, *_turn("first"), *_turn("second")]
    )
    cog = ClaudeCliCognition(spawn=cli)
    said: list[str] = []
    async with cog.session() as chat:
        for prompt in ("hello", "and again"):
            async for ev in chat.turn(prompt):
                if ev.type == "final":
                    said.append(ev.result.output)

    assert said == ["first", "second"]
    assert len(cli.invocations) == 1
    # The turns the cognition WROTE are on the double, so a session test can
    # assert the stream-json input format without a second spawn.
    written = [json.loads(line) for line in cli.invocations[0].stdin.decode().splitlines()]
    assert [w["message"]["content"] for w in written] == ["hello", "and again"]


@pytest.mark.asyncio
async def test_a_session_whose_output_ends_mid_turn_is_over() -> None:
    """stdout EOF before this turn's `result`: the process died mid-answer, and
    every later turn must refuse rather than silently start a fresh
    conversation."""
    cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I was "}]}},
        ]
    )
    cog = ClaudeCliCognition(spawn=cli)
    async with cog.session() as chat:
        first = [ev async for ev in chat.turn("hello")][-1].result
        second = [ev async for ev in chat.turn("still there?")][-1].result

    assert first.evals["stop_reason"] == "session_closed"
    assert second.evals["stop_reason"] == "session_closed"


@pytest.mark.asyncio
async def test_stderr_drains_once_like_the_real_pipe() -> None:
    """A session reads stderr on EVERY failing turn. A double that replayed
    its buffer each time would stamp turn 1's diagnostic onto turn 2's
    terminal event, and an operator would go chasing a fault that had already
    been reported once."""
    cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            CliStderr(b"stream disconnected before message completed\n"),
        ]
    )
    cog = ClaudeCliCognition(spawn=cli)
    async with cog.session() as chat:
        first = [ev async for ev in chat.turn("hello")][-1].result
        second = [ev async for ev in chat.turn("still there?")][-1].result

    assert "stream disconnected" in first.evals["stderr"]
    assert "stderr" not in second.evals


# ── 7. construction ──────────────────────────────────────────────────────────


def test_runs_can_be_assembled_by_hand() -> None:
    """`CliRun` is the shared unit: `replay` builds them from files, `script`
    from payloads, and a test that needs three spawns with different exit codes
    builds them directly."""
    cli = FakeClaudeCli([CliRun(stdout=b"{}\n"), CliRun(returncode=3, stderr=b"boom")])
    assert cli.remaining == 2
    assert cli.spawns == 0


@pytest.mark.asyncio
async def test_a_stdout_line_that_is_not_valid_utf8_is_skipped_like_any_other() -> None:
    """A PRODUCTION bug, found by pointing the new double at the one input
    nobody had been able to produce before it existed.

    `_parse_line` promises that a non-JSON diagnostic on stdout "must NOT crash
    the loop", and caught `json.JSONDecodeError` to keep that promise. But
    `json.loads` on BYTES sniffs the encoding first, so `b'\\xff\\xfe...'` is
    taken for a UTF-16-LE BOM and the failure comes back as a
    `UnicodeDecodeError` — a `ValueError`, but not a `JSONDecodeError`. It
    escaped into `drive`'s `except BaseException`.

    The damage is the quiet kind. One undecodable byte ended the entire run,
    and because the reader never got past that line the `result` payload was
    never seen — so a run that finished and cost real money reported
    `parse_failed`, no output, and **$0.00** to the budget. `claude` is Node
    and filenames are arbitrary bytes; a warning naming a latin-1 path is all
    it takes.

    This also covers the `bytes` half of `CliRun.of`'s documented verbatim
    escape hatch, which had no test — and which is the only way to express a
    line that is not valid UTF-8 at all."""
    budget = Budget(max_cost_usd=10.0)
    ctx = RunContext("run-utf8", Scope(), services=Services(), budget=budget)
    cli = FakeClaudeCli.script(
        [
            b"\xff\xfe warning: cannot read /tmp/caf\xe9/settings.json\n",
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "survived"}]}},
            b'{"type":"result","subtype":"success","session_id":"s","duration_ms":2,'
            b'"total_cost_usd":0.03,"usage":{"input_tokens":5,"output_tokens":7}}\n',
        ]
    )
    result = (await _drive(ClaudeCliCognition(spawn=cli), ctx))[-1].result

    assert result.output == "survived"
    assert result.partial is False
    assert "stop_reason" not in result.evals  # NOT parse_failed
    # The run was read to its end, so the money is on the ledger.
    assert result.usage.cost_usd == 0.03
    assert_money(budget.spent(), "0.030000")
    assert cli.spawns == 1


def test_a_recording_that_is_not_there_says_so_at_construction() -> None:
    """Not at spawn time, three layers down inside a cognition's
    `except BaseException`, where the message becomes `evals['error']`."""
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        FakeClaudeCli.replay(Path("/tmp/definitely/nope.jsonl"))


# ── 8. the parts of the PROCESS the double also has to be ────────────────────
#
# `terminate()`/`returncode` and the control protocol are not parsing — they
# are process lifecycle, and the cognition leans on them just as hard: a cancel
# that does not terminate leaks a subprocess, and `interrupt()` is the one
# thing `ClaudeCliSession` offers that cancellation cannot. Neither had a test
# against the double, and neither is reachable at all without a binary.


@pytest.mark.asyncio
async def test_a_cancelled_run_terminates_the_replayed_process() -> None:
    """`ctx.check_cancelled()` is polled once per line, so it is the cancel a
    replayed stream can actually deliver (`asyncio.wait_for` cannot — the
    double has no suspension point; see the module docstring).

    The assertion that matters is on the INVOCATION, not the result. A double
    whose `terminate()` did nothing would still report `stop_reason='cancelled'`,
    because cancellation outranks the exit code in `_finalise` — so the result
    alone cannot tell "we stopped the child" from "we left it running"."""

    class _CancelsOnThirdLine(FakeCtx):  # type: ignore[misc]
        seen = 0

        def check_cancelled(self) -> None:
            _CancelsOnThirdLine.seen += 1
            if _CancelsOnThirdLine.seen > 3:
                raise RuntimeError("cancelled by the operator")

    payloads: list[Any] = [{"type": "system", "subtype": "init", "session_id": "s"}]
    payloads += [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": f"x{i}"}]}}
        for i in range(10)
    ]
    payloads.append(
        {"type": "result", "subtype": "success", "session_id": "s", "duration_ms": 1,
         "total_cost_usd": 9.99, "usage": {"input_tokens": 1, "output_tokens": 1}}
    )
    cli = FakeClaudeCli.script(payloads)
    result = (await _drive(ClaudeCliCognition(spawn=cli), _CancelsOnThirdLine()))[-1].result

    assert result.evals["stop_reason"] == "cancelled"
    assert result.partial is True
    inv = cli.invocations[0]
    assert inv.terminated is True
    assert inv.returncode == -15  # SIGTERM, the way a real child reports it
    # And it stopped where it was told to rather than draining to the end: the
    # recording's `result` payload carried $9.99 and never arrived.
    assert inv.lines_read < len(payloads)
    assert result.usage.cost_usd == 0.0


@pytest.mark.asyncio
async def test_an_interrupt_is_delivered_and_acknowledged_over_the_double() -> None:
    """The whole control protocol: the request written to stdin, the
    `control_response` routed off the SAME stdout the turn reader is draining,
    the receipt, and `stop_reason='interrupted'` stamped over the CLI's
    ambiguous `error_during_execution`.

    Note the `await asyncio.sleep(0)`. Reading a line from the double never
    awaits, so the stop-button task cannot run until the CONSUMER yields —
    the double's one divergence from a real pipe, and this is where a reader
    meets it."""
    cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s",
             "capabilities": ["interrupt_cancel_queued_v1"]},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "thinking "}]}},
            {"type": "control_response",
             "response": {"request_id": "agentkit-1", "subtype": "success"}},
            {"type": "result", "subtype": "error_during_execution", "is_error": True,
             "session_id": "s", "duration_ms": 5, "total_cost_usd": 0.01, "usage": {}},
        ]
    )
    cog = ClaudeCliCognition(spawn=cli)
    stop: Any = None
    result: Any = None
    async with cog.session() as chat:
        async for ev in chat.turn("go"):
            if ev.type == "message_delta" and stop is None:
                stop = asyncio.create_task(chat.interrupt())
                await asyncio.sleep(0)  # hand the loop to the stop button
            if ev.type == "final":
                result = ev.result
        receipt = await stop

    assert receipt.delivered is True
    assert receipt.error is None  # the acknowledgement really came back
    # Somebody stopped this on purpose — not a CLI fault, despite the subtype.
    assert result.evals["stop_reason"] == "interrupted"
    assert result.partial is True
    # The request is on the double's stdin, which is the only place it exists.
    written = [json.loads(line) for line in cli.invocations[0].stdin.decode().splitlines()]
    assert written[-1] == {
        "type": "control_request",
        "request_id": "agentkit-1",
        "request": {"subtype": "interrupt"},
    }


@pytest.mark.asyncio
async def test_two_stderr_chunks_at_one_point_keep_their_write_order() -> None:
    """A process writing a multi-line diagnostic as several writes.

    The chunks are queued as `(stdout position, bytes)` and were sorted as
    whole TUPLES, so a tie on the position fell through to comparing the bytes
    and the traceback came back alphabetised — measured:
    `'alpha...\\nzebra...'` for chunks written zebra-first. Silent, too: the run
    still fails the way it should and `stderr` is still non-empty, so the only
    symptom is an operator reading the cause after the effect."""
    cli = FakeClaudeCli.script(
        [
            CliStderr(b"Traceback (most recent call last):\n"),
            CliStderr(b"  File 'cli.js', line 1\n"),
            CliStderr(b"Error: ENOSPC, no space left on device\n"),
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}},
        ],
        returncode=1,
    )
    result = (await _drive(ClaudeCliCognition(spawn=cli)))[-1].result
    assert result.evals["stderr"] == (
        "Traceback (most recent call last):\n"
        "  File 'cli.js', line 1\n"
        "Error: ENOSPC, no space left on device"
    )
