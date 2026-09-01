"""`FakeCodexCli` — the offline double for the `codex` CLI path.

The sibling of `test_fake_claude_cli.py`, and it exists for the same reason: the
CLI is a subprocess, not a port, so the double has to sit at the spawn seam and
supply BYTES. A double that handed back finished `AgentResult`s would test none
of the places a cognition of this shape puts its bugs — the line parser, the
payload→event mapping, the stop-reason priority, the token split, the cost
computation.

What is tested here is the DOUBLE, not the cognition: that a recording replays
deterministically, that exhaustion is loud, that the process surface behaves
like a real one (returncode `None` while alive, stderr draining once, a
trailing line with no newline still delivered), and that `codex_turn` builds
the event order a real turn has.

The two doubles share their replay machinery (`agentkit/testing/fakes/_cli_replay.py`)
and their value types, so several properties asserted in the Claude file hold
here by construction. The ones re-asserted below are the ones a Codex-specific
change could break on its own.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agentkit import Agent, Scope
from agentkit.agents.cognition import CodexCliCognition
from agentkit.context import WorkingContext
from agentkit.runtime import Budget, RunContext, Services
from agentkit.testing.fakes import CliRun, CliStderr, FakeCodexCli, ScriptExhausted, codex_turn
from agentkit.testing.fakes.ctx import FakeCtx
from tests._assertions import assert_money


async def _drive(cog: CodexCliCognition, ctx: Any = None, task: str = "count the files") -> list[Any]:
    agent = Agent(name="local", cognition=cog)
    return [ev async for ev in cog.drive(agent, task, ctx or FakeCtx(), WorkingContext())]


def _shape(events: list[Any]) -> list[tuple[str, str]]:
    """The stream reduced to what a consumer sees, so two replays can be
    compared for equality without comparing object ids."""
    out: list[tuple[str, str]] = []
    for ev in events:
        if ev.type == "final":
            r = ev.result
            out.append(("final", json.dumps([r.output, r.partial, r.stop_reason, r.evals], sort_keys=True, default=str)))
        elif ev.tool_call is not None:
            out.append(
                (
                    ev.type,
                    f"{ev.tool_call.id}:{ev.tool_call.name}:"
                    f"{json.dumps(ev.tool_call.arguments, sort_keys=True, default=str)}:{ev.tool_result}",
                )
            )
        else:
            out.append((ev.type, ev.text))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. codex_turn builds a real turn
# ─────────────────────────────────────────────────────────────────────────────


def test_a_built_turn_has_the_event_order_a_real_turn_has() -> None:
    """The order carries meaning the cognition depends on: the thread id arrives
    before anything else (it is what a session resumes) and the usage arrives
    last (it is what the budget is charged from). A builder that got it wrong
    would let every test assert against a stream the binary never emits."""
    payloads = codex_turn(reasoning="think", text="answer", usage=(10, 4, 2))
    assert [p["type"] for p in payloads] == [
        "thread.started",
        "turn.started",
        "item.completed",
        "item.completed",
        "turn.completed",
    ]
    assert payloads[0]["thread_id"]
    assert payloads[-1]["usage"] == {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2}


def test_usage_is_passed_in_the_clis_own_convention() -> None:
    """Input INCLUSIVE of cache, because that is what Codex reports. Pre-splitting
    it in the builder would make the one test that exists to pin the
    cognition's subtraction assert against the builder's arithmetic instead."""
    payload = codex_turn(text="x", usage=(1200, 1000, 40))[-1]
    assert payload["usage"]["input_tokens"] == 1200
    assert payload["usage"]["cached_input_tokens"] == 1000


def test_a_failed_turn_replaces_the_terminal_event_rather_than_adding_one() -> None:
    """Exactly one terminal payload, or the cognition would see a completed turn
    after a failed one and fold the wrong stop reason."""
    payloads = codex_turn(text="half", failed="boom")
    terminals = [p["type"] for p in payloads if p["type"].startswith("turn.")]
    assert terminals == ["turn.started", "turn.failed"]


def test_a_thread_id_of_none_omits_the_thread_started_payload() -> None:
    """How a test builds the resumed turn of a CLI version that does not
    re-announce the thread."""
    assert [p["type"] for p in codex_turn(text="x", thread_id=None)][0] == "turn.started"


def test_items_are_numbered_and_a_supplied_id_wins() -> None:
    payloads = codex_turn(
        reasoning="r",
        items=[{"type": "command_execution", "command": "ls"}, {"id": "mine", "type": "web_search", "query": "q"}],
        text="t",
    )
    ids = [p["item"]["id"] for p in payloads if p["type"] == "item.completed"]
    assert ids == ["item_0", "item_1", "mine", "item_3"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. determinism
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_doubles_over_one_recording_produce_identical_streams() -> None:
    run = CliRun.of(
        codex_turn(
            reasoning="looking",
            items=[{"type": "command_execution", "command": "ls", "aggregated_output": "a\n", "exit_code": 0}],
            text="one file",
            usage=(10, 4, 2),
        )
    )
    first = await _drive(CodexCliCognition(spawn=FakeCodexCli([run])))
    second = await _drive(CodexCliCognition(spawn=FakeCodexCli([run])))
    assert _shape(first) == _shape(second)


@pytest.mark.asyncio
async def test_one_double_replayed_twice_does_not_consume_its_recording() -> None:
    """Each SPAWN consumes a run; the runs themselves are immutable, so two
    doubles over the same ``CliRun`` cannot affect each other."""
    run = CliRun.of(codex_turn(text="same", usage=(1, 0, 1)))
    cli = FakeCodexCli([run, run])
    cog = CodexCliCognition(spawn=cli)
    assert _shape(await _drive(cog)) == _shape(await _drive(cog))
    assert cli.spawns == 2 and cli.remaining == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. exhaustion is loud
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_asking_for_a_run_the_recording_does_not_have_raises() -> None:
    cli = FakeCodexCli.script(codex_turn(text="one", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    await _drive(cog)
    with pytest.raises(ScriptExhausted) as caught:
        await _drive(cog)

    message = str(caught.value)
    assert "FakeCodexCli exhausted" in message
    assert "1 run(s)" in message and "2 time(s)" in message
    # Both ways out are named, because the reader's next question is "so what do
    # I do", and the defect is almost never in the double.
    assert "FakeCodexCli([CliRun.of(...)" in message
    assert "repeat_last=True" in message


@pytest.mark.asyncio
async def test_exhaustion_escapes_the_cognitions_own_except_baseexception() -> None:
    """``ScriptExhausted`` is a ``BaseException`` precisely so it survives sites
    like this one. The cognition converts every ``Exception`` into a tidy
    terminal event, which for THIS exception would be the stable, plausible,
    wrong answer the class exists to prevent."""
    cli = FakeCodexCli()
    with pytest.raises(ScriptExhausted):
        await _drive(CodexCliCognition(spawn=cli))


@pytest.mark.asyncio
async def test_the_terminal_event_still_arrives_before_exhaustion_propagates() -> None:
    """Both contracts hold at once: the caller sees exactly one ``final`` and
    THEN the exception. Swallowing one for the other is the bug in either
    direction."""
    cli = FakeCodexCli()
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="local", cognition=cog)
    seen: list[Any] = []

    with pytest.raises(ScriptExhausted):
        async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext()):
            seen.append(ev)

    assert [e.type for e in seen] == ["final"]
    assert seen[0].result.partial is True


@pytest.mark.asyncio
async def test_repeat_last_is_the_documented_way_out() -> None:
    cli = FakeCodexCli.script(codex_turn(text="again", usage=(1, 0, 1)), repeat_last=True)
    cog = CodexCliCognition(spawn=cli)
    for _ in range(4):
        assert (await _drive(cog))[-1].result.output == "again"
    # Counted past the end, so a test can assert HOW MANY spawns a loop made
    # rather than merely that it made extra ones.
    assert cli.spawns == 4


# ─────────────────────────────────────────────────────────────────────────────
# 4. the process surface
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_argv_cwd_and_env_the_cognition_built_are_visible(tmp_path: Path) -> None:
    """The whole point of recording an invocation: a flags test can assert on
    the spawn without the process-wide ``create_subprocess_exec`` patch."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(model="gpt-5-codex", working_dir=tmp_path, config_home=tmp_path, spawn=cli)
    await _drive(cog)

    invocation = cli.invocations[-1]
    assert invocation.argv[:3] == ("codex", "exec", "--json")
    assert invocation.cwd == str(tmp_path)
    assert invocation.env["CODEX_HOME"] == str(tmp_path)
    assert cli.argv() == invocation.argv


@pytest.mark.asyncio
async def test_a_final_line_with_no_trailing_newline_is_still_delivered() -> None:
    """What a real ``StreamReader`` does at EOF, and the reason ``_count_lines``
    counts a trailing fragment. A double that dropped it would make every
    "process killed mid-answer" test silently pass."""
    payloads = codex_turn(text="last word", usage=(1, 0, 1))
    blob = b"".join(json.dumps(p).encode() + b"\n" for p in payloads[:-1])
    blob += json.dumps(payloads[-1]).encode()  # no newline
    result = (await _drive(CodexCliCognition(spawn=FakeCodexCli([CliRun(stdout=blob)]))))[-1].result
    assert result.output == "last word"
    assert result.usage.input_tokens == 1


@pytest.mark.asyncio
async def test_stderr_drains_once_like_the_real_pipe() -> None:
    """A second read returns nothing, so a caller reading stderr on two
    consecutive failed runs does not attribute the first's diagnostic to the
    second."""
    cli = FakeCodexCli(
        [
            CliRun.of(codex_turn(text="a"), stderr=b"first problem", returncode=1),
            CliRun.of(codex_turn(text="b"), stderr=b"", returncode=1),
        ]
    )
    cog = CodexCliCognition(spawn=cli)
    first = (await _drive(cog))[-1].result
    second = (await _drive(cog))[-1].result
    assert first.evals["stderr"] == "first problem"
    assert "stderr" not in second.evals


@pytest.mark.asyncio
async def test_two_stderr_chunks_at_one_point_keep_their_write_order() -> None:
    """A two-line traceback written as two calls. Sorting on the whole tuple
    fell through to comparing BYTES and reassembled it alphabetically —
    backwards — and the run still looked fine."""
    cli = FakeCodexCli(
        [
            CliRun.of(
                [
                    {"type": "thread.started", "thread_id": "t"},
                    CliStderr(b"zeta: the cause\n"),
                    CliStderr(b"alpha: the detail\n"),
                    {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "x"}},
                ],
                returncode=1,
            )
        ]
    )
    result = (await _drive(CodexCliCognition(spawn=cli)))[-1].result
    assert result.evals["stderr"] == "zeta: the cause\nalpha: the detail"


@pytest.mark.asyncio
async def test_a_cancelled_run_terminates_the_replayed_process() -> None:
    """``terminate()`` sets a negative return code, the way a signalled process
    does, and ``_finalise`` reports it verbatim."""
    from tests.agents.cognition.test_codex_cli import CancellingCtx

    cli = FakeCodexCli.script(codex_turn(text="never read", usage=(1, 0, 1)))
    result = (await _drive(CodexCliCognition(spawn=cli), ctx=CancellingCtx()))[-1].result
    assert cli.invocations[0].terminated is True
    assert cli.invocations[0].returncode == -15
    assert result.evals["cli_return_code"] == -15


@pytest.mark.asyncio
async def test_a_recording_is_streamed_not_slurped() -> None:
    """``lines_read`` at the time the FIRST delta arrives. A double that
    buffered the run to completion first would show it at the recording's full
    length, and every streaming assertion in the suite would be vacuous."""
    payloads = [{"type": "thread.started", "thread_id": "t"}]
    for i in range(500):
        payloads.append(
            {"type": "item.completed", "item": {"id": f"m{i}", "type": "agent_message", "text": "x"}}
        )
    payloads.append({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})

    cli = FakeCodexCli.script(payloads)
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="local", cognition=cog)

    at_first_delta = None
    async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext()):
        if ev.type == "message_delta" and at_first_delta is None:
            at_first_delta = cli.invocations[0].lines_read

    assert at_first_delta is not None
    assert at_first_delta < 10, f"{at_first_delta} lines were read before the first delta"


# ─────────────────────────────────────────────────────────────────────────────
# 5. replay from disk
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_recorded_session_file_replays_through_the_real_parser(tmp_path: Path) -> None:
    recording = tmp_path / "session.jsonl"
    recording.write_text(
        "\n".join(
            json.dumps(p)
            for p in codex_turn(
                reasoning="scanning",
                items=[
                    {
                        "type": "command_execution",
                        "command": "bash -lc 'ls src'",
                        "aggregated_output": "app.py\n",
                        "exit_code": 0,
                    },
                    {"type": "file_change", "changes": [{"path": "src/app.py", "kind": "update"}]},
                ],
                text="Added the route.",
                usage=(24763, 24448, 122),
            )
        )
        + "\n"
    )

    events = await _drive(CodexCliCognition(spawn=FakeCodexCli.replay(recording)))
    result = events[-1].result

    assert result.output == "Added the route."
    assert result.evals["thinking"] == "scanning"
    assert [e.tool_call.name for e in events if e.type == "tool_call"] == ["shell", "apply_patch"]
    assert result.usage.input_tokens == 315
    assert result.usage.cache_read_tokens == 24448


def test_a_recording_that_is_not_there_says_so_at_construction() -> None:
    """Read at construction, once. A recording that vanished between then and
    the spawn three layers down would surface as ``evals["error"]`` inside the
    cognition's ``except BaseException``, where a missing fixture is nearly
    unreadable."""
    with pytest.raises(OSError):
        FakeCodexCli.replay(Path("/tmp/definitely-not-a-recording-XYZ.jsonl"))


@pytest.mark.asyncio
async def test_one_file_per_spawn(tmp_path: Path) -> None:
    """Which is what a multi-turn session needs: ``codex exec`` spawns per turn,
    so a two-turn conversation is two recordings."""
    for i, text in enumerate(("first", "second")):
        (tmp_path / f"{i}.jsonl").write_text(
            "\n".join(json.dumps(p) for p in codex_turn(text=text, thread_id="T", usage=(1, 0, 1))) + "\n"
        )

    cli = FakeCodexCli.replay(tmp_path / "0.jsonl", tmp_path / "1.jsonl")
    async with CodexCliCognition(spawn=cli).session() as chat:
        one = [ev async for ev in chat.turn("a")][-1].result
        two = [ev async for ev in chat.turn("b")][-1].result

    assert (one.output, two.output) == ("first", "second")
    assert "resume" in cli.invocations[1].argv


# ─────────────────────────────────────────────────────────────────────────────
# 6. the meters still run below the seam
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_replayed_usage_is_charged_to_the_frameworks_meters() -> None:
    """The point of the seam being where it is: the double supplies bytes, so
    the real parse, the real token split, the real cost computation and the real
    meter charge all run. A double above this level would test none of them."""
    budget = Budget(max_cost_usd=10.0)
    ctx = RunContext("run-1", Scope(), services=Services(), budget=budget)
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1_000_000, 0, 0)))

    await _drive(CodexCliCognition(model="gpt-4.1", spawn=cli), ctx=ctx)

    # 1M input tokens of gpt-4.1 at the table's $2.00/1M.
    assert_money(budget.spent(), "2.000000")
    assert budget.usage.input_tokens == 1_000_000


@pytest.mark.asyncio
async def test_a_keyboard_interrupt_from_the_double_reaches_the_caller() -> None:
    """The double can raise into the test's stack — the property that ruled out
    generating a fake binary. A subprocess cannot do this, and the cognition
    converting a non-zero exit into a tidy failed result is exactly what would
    have hidden it."""

    class _Interrupting(FakeCodexCli):
        async def __call__(self, *argv: str, **kw: Any):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt()

    cog = CodexCliCognition(spawn=_Interrupting())
    agent = Agent(name="local", cognition=cog)
    seen: list[Any] = []

    with pytest.raises(KeyboardInterrupt):
        async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext()):
            seen.append(ev)

    assert [e.type for e in seen] == ["final"]


@pytest.mark.asyncio
async def test_two_drives_can_be_read_interleaved_by_the_consumer() -> None:
    """Reading a line from this double never awaits, so ``asyncio.gather`` runs
    one drive to completion and then the other. A test that needs them genuinely
    interleaved awaits in its OWN consuming loop — documented on the double, and
    pinned here so the documentation stays true."""
    a = FakeCodexCli.script(codex_turn(text="AAA", usage=(1, 0, 1)))
    b = FakeCodexCli.script(codex_turn(text="BBB", usage=(1, 0, 1)))
    order: list[str] = []

    async def consume(cli: FakeCodexCli, tag: str) -> None:
        cog = CodexCliCognition(spawn=cli)
        agent = Agent(name=tag, cognition=cog)
        async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext()):
            if ev.type == "message_delta":
                order.append(tag)
            await asyncio.sleep(0)

    await asyncio.gather(consume(a, "a"), consume(b, "b"))
    assert set(order) == {"a", "b"}
