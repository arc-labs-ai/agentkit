"""A Codex session is a resumed thread, not a held process — and that changes things.

``ClaudeCliSession`` holds one subprocess open and feeds it turns over stdin.
``codex exec`` cannot do that: it is one-shot, and its continuation seam is
``codex exec resume <thread-id>``. So :class:`CodexCliSession` spawns per turn
and threads the conversation through the id it learned from turn one.

That is a worse deal in one way and a better one in three, and each is a test
here:

* worse:  a CLI warm-up per turn, where the Claude session pays it once.
* better: a cancelled turn does NOT end the conversation — the thread is on
          disk and the next turn resumes it. The Claude session has to declare
          itself over, because its process WAS the conversation.
* better: per-turn structured output works (pinned in
          ``test_codex_cli_structured.py``).
* better: a turn that fails for a reason that is not the process dying costs
          one turn, not the session.

The invariant that matters most: turn one starts a thread, and EVERY later turn
resumes THAT id. A session that quietly started a fresh thread on turn two
would answer every question competently with no memory of the last one — the
failure looks like a bad model, not a bug.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CodexCliCognition
from agentkit.kernel.types import StreamEvent
from agentkit.testing.fakes import CliRun, FakeCodexCli, codex_turn
from agentkit.testing.fakes.ctx import FakeCtx
from agentkit.testing.fakes.llm import ScriptExhausted
from tests.agents.cognition.test_codex_cli import CancellingCtx

real_codex = pytest.mark.skipif(
    shutil.which("codex") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="codex CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


async def take(gen: Any) -> list[StreamEvent]:
    return [ev async for ev in gen]


def resume_arg(argv: tuple[str, ...]) -> tuple[str, ...]:
    """The ``resume`` fragment of one spawn's argv, or ``()`` for a fresh run."""
    return argv[argv.index("resume") :][:2] if "resume" in argv else ()


# ─────────────────────────────────────────────────────────────────────────────
# 1. threading the conversation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_one_starts_a_thread_and_later_turns_resume_it() -> None:
    """THE invariant. A session that started a fresh thread on turn two would
    answer every question competently with no memory of the last one."""
    cli = FakeCodexCli.answering("first", "second", "third", thread_id="thread-A")
    cog = CodexCliCognition(spawn=cli)

    async with cog.session() as chat:
        one = (await take(chat.turn("a")))[-1].result
        two = (await take(chat.turn("b")))[-1].result
        three = (await take(chat.turn("c")))[-1].result

    assert [one.output, two.output, three.output] == ["first", "second", "third"]
    assert resume_arg(cli.invocations[0].argv) == ()
    assert resume_arg(cli.invocations[1].argv) == ("resume", "thread-A")
    assert resume_arg(cli.invocations[2].argv) == ("resume", "thread-A")
    assert chat.session_id == "thread-A"
    assert chat.turns_taken == 3


@pytest.mark.asyncio
async def test_a_session_spawns_once_per_turn() -> None:
    """Not a criticism, a contract. The recording's length is a claim about how
    many times the binary runs, and a session that spawned twice for one turn —
    a retry nobody asked for — would exhaust the double rather than replay a
    plausible wrong answer."""
    cli = FakeCodexCli.answering("one", "two")
    async with CodexCliCognition(spawn=cli).session() as chat:
        await take(chat.turn("a"))
        await take(chat.turn("b"))
    assert cli.spawns == 2
    assert cli.remaining == 0


@pytest.mark.asyncio
async def test_a_third_turn_past_the_recording_raises_rather_than_replaying() -> None:
    cli = FakeCodexCli.answering("one")
    async with CodexCliCognition(spawn=cli).session() as chat:
        await take(chat.turn("a"))
        with pytest.raises(ScriptExhausted, match="FakeCodexCli exhausted"):
            await take(chat.turn("b"))


@pytest.mark.asyncio
async def test_a_cognition_that_already_names_a_thread_starts_there() -> None:
    """A session opened to CONTINUE yesterday's conversation. Turn one already
    has an id, so it resumes rather than starting a thread and orphaning the
    one the caller named."""
    cli = FakeCodexCli.script(codex_turn(text="picked up", thread_id="yesterday", usage=(1, 0, 1)))
    cog = CodexCliCognition(resume_session_id="yesterday", spawn=cli)

    async with cog.session() as chat:
        assert chat.session_id == "yesterday"
        await take(chat.turn("carry on"))

    assert resume_arg(cli.invocations[0].argv) == ("resume", "yesterday")


@pytest.mark.asyncio
async def test_a_first_turn_that_never_learned_an_id_starts_fresh_next_time() -> None:
    """The honest fallback. A first turn that died before ``thread.started``
    left no conversation, so resuming nothing would be a resume of nothing —
    the CLI's error for that names a thread id the caller never saw."""
    cli = FakeCodexCli(
        [
            CliRun.of([{"type": "turn.failed", "error": {"message": "auth failed"}}], returncode=1),
            CliRun.of(codex_turn(text="working now", thread_id="thread-B", usage=(1, 0, 1))),
        ]
    )
    cog = CodexCliCognition(spawn=cli)

    async with cog.session() as chat:
        first = (await take(chat.turn("a")))[-1].result
        second = (await take(chat.turn("b")))[-1].result

    assert first.partial is True
    assert resume_arg(cli.invocations[1].argv) == ()
    assert second.output == "working now"
    assert chat.session_id == "thread-B"


# ─────────────────────────────────────────────────────────────────────────────
# 2. the system prompt belongs to the conversation, not to each turn
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_agent_prompt_is_prepended_only_on_the_first_turn() -> None:
    """Prepending it again on turn three would put three copies in the
    transcript and read to the model as escalating emphasis."""
    cli = FakeCodexCli.answering("a", "b")
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="x", prompt="You are terse.", cognition=cog)

    async with cog.session(agent=agent) as chat:
        await take(chat.turn("first question"))
        await take(chat.turn("second question"))

    assert cli.invocations[0].argv[-1] == "You are terse.\n\n---\n\nfirst question"
    assert cli.invocations[1].argv[-1] == "second question"


@pytest.mark.asyncio
async def test_replace_mode_rewrites_the_instructions_file_on_every_turn() -> None:
    """Under ``replace`` the prompt is not in the transcript at all — it is the
    base instructions, and those are a per-spawn flag. Sending it once would
    leave turn two running on Codex's own instructions instead of the
    caller's."""
    cli = FakeCodexCli.answering("a", "b")
    cog = CodexCliCognition(system_prompt_mode="replace", spawn=cli)
    agent = Agent(name="x", prompt="You are a linter.", cognition=cog)

    async with cog.session(agent=agent) as chat:
        await take(chat.turn("one"))
        await take(chat.turn("two"))

    for invocation in cli.invocations:
        argv = list(invocation.argv)
        assert any(a.startswith("experimental_instructions_file=") for a in argv), argv
        assert argv[-1] in ("one", "two")


# ─────────────────────────────────────────────────────────────────────────────
# 3. what a failure costs
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failed_turn_does_not_end_the_session() -> None:
    """A turn is a process. One dying costs that turn; the thread it was
    resuming is still on disk. This is the case that used to kill the Claude
    session, where the process WAS the conversation."""
    cli = FakeCodexCli(
        [
            CliRun.of(codex_turn(text="ok", thread_id="T", usage=(1, 0, 1))),
            CliRun.of(codex_turn(failed="rate limited", thread_id="T"), returncode=1),
            CliRun.of(codex_turn(text="back to normal", thread_id="T", usage=(1, 0, 1))),
        ]
    )
    cog = CodexCliCognition(spawn=cli)

    async with cog.session() as chat:
        first = (await take(chat.turn("a")))[-1].result
        second = (await take(chat.turn("b")))[-1].result
        third = (await take(chat.turn("c")))[-1].result

    assert first.output == "ok"
    assert second.partial is True and second.evals["stop_reason"] == "cli_exit_1"
    assert third.output == "back to normal"
    # And the third turn still resumed the SAME thread, so the failure did not
    # cost the conversation either.
    assert resume_arg(cli.invocations[2].argv) == ("resume", "T")


@pytest.mark.asyncio
async def test_a_cancelled_turn_does_not_end_the_session() -> None:
    """The clearest difference from the Claude session, which must declare
    itself over: no protocol message retracts a half-finished turn there, so
    the process is killed and the conversation goes with it. Here the process
    was never the conversation."""
    cli = FakeCodexCli(
        [
            CliRun.of(codex_turn(text="first", thread_id="T", usage=(1, 0, 1))),
            CliRun.of(codex_turn(text="never read", thread_id="T", usage=(1, 0, 1))),
            CliRun.of(codex_turn(text="still talking", thread_id="T", usage=(1, 0, 1))),
        ]
    )
    cog = CodexCliCognition(spawn=cli)

    async with cog.session() as chat:
        await take(chat.turn("a"))
        cancelled = (await take(chat.turn("b", ctx=CancellingCtx())))[-1].result
        after = (await take(chat.turn("c")))[-1].result

    assert cancelled.evals["stop_reason"] == "cancelled"
    assert cli.invocations[1].terminated is True
    assert after.output == "still talking"
    assert resume_arg(cli.invocations[2].argv) == ("resume", "T")


@pytest.mark.asyncio
async def test_a_turn_on_a_closed_session_is_refused_as_a_terminal_event() -> None:
    """Reported, not raised, so a session turn keeps the same
    exactly-one-``final`` contract as a one-shot drive — and the message names
    the thread id, because the conversation is still there and resumable."""
    cli = FakeCodexCli.answering("one", thread_id="thread-C")
    cog = CodexCliCognition(spawn=cli)

    chat = cog.session()
    await chat.start()
    await take(chat.turn("a"))
    await chat.close()

    events = await take(chat.turn("b"))
    result = events[-1].result

    assert len(events) == 1
    assert result.partial is True
    assert result.stop_reason == "failed"
    assert result.evals["stop_reason"] == "session_closed"
    assert "thread-C" in result.evals["error"]
    # No spawn was attempted, so the recording is untouched.
    assert cli.spawns == 1


@pytest.mark.asyncio
async def test_closing_twice_is_fine_and_nothing_is_killed() -> None:
    """There is never a process running between turns, so closing is
    bookkeeping. It deliberately does NOT delete the CLI-side thread: the id is
    on every terminal event and a caller may want to resume it tomorrow."""
    cog = CodexCliCognition(spawn=FakeCodexCli.answering("x"))
    chat = cog.session()
    await chat.close()
    await chat.close()
    assert chat.turns_taken == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. serialisation, and the shapes a session refuses
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_concurrent_turns_are_serialised() -> None:
    """One thread and one transcript: two turns resuming the same thread at once
    would fork it. The lock makes the second caller wait, so the spawns come out
    in order and the second one carries the id the first learned."""
    cli = FakeCodexCli.answering("one", "two", thread_id="thread-D")
    cog = CodexCliCognition(spawn=cli)

    async with cog.session() as chat:

        async def run(task: str) -> Any:
            return (await take(chat.turn(task)))[-1].result

        first, second = await asyncio.gather(run("a"), run("b"))

    assert {first.output, second.output} == {"one", "two"}
    assert resume_arg(cli.invocations[0].argv) == ()
    assert resume_arg(cli.invocations[1].argv) == ("resume", "thread-D")


@pytest.mark.asyncio
async def test_an_ephemeral_cognition_cannot_open_a_session() -> None:
    """Nothing is written, so there is nothing to resume — and the failure would
    be a conversation with no memory rather than an error."""
    with pytest.raises(ValueError, match="ephemeral"):
        CodexCliCognition(ephemeral=True).session()


@pytest.mark.asyncio
async def test_a_session_can_be_an_agents_cognition() -> None:
    """``drive`` on the session, so consecutive ``agent.run(...)`` calls
    continue one CLI-side thread rather than starting a conversation each
    time."""
    cli = FakeCodexCli.answering("one", "two", thread_id="thread-E")
    cog = CodexCliCognition(spawn=cli)
    chat = cog.session()
    agent = Agent(name="x", cognition=chat)

    first = await agent.run("a", FakeCtx())
    second = await agent.run("b", FakeCtx())

    assert (first.output, second.output) == ("one", "two")
    assert resume_arg(cli.invocations[1].argv) == ("resume", "thread-E")


@pytest.mark.asyncio
async def test_the_session_agent_is_the_default_and_a_turn_can_override_it() -> None:
    cli = FakeCodexCli.answering("a", "b")
    cog = CodexCliCognition(spawn=cli)
    default = Agent(name="default", prompt="DEFAULT", cognition=cog)
    other = Agent(name="other", prompt="OTHER", cognition=cog)

    async with cog.session(agent=default) as chat:
        await take(chat.turn("one"))
        await take(chat.turn("two", agent=other))

    # Turn one carries the session's agent prompt; turn two is a resume, so no
    # prompt is prepended at all — which is the point of the previous test and
    # is what makes the override observable only through the schema.
    assert cli.invocations[0].argv[-1].startswith("DEFAULT")
    assert cli.invocations[1].argv[-1] == "two"


@pytest.mark.asyncio
async def test_a_turn_with_no_agent_and_no_ctx_works() -> None:
    """The bare conversational session. Both are optional, and a session that
    required an ``Agent`` would make the simplest use of this the fiddliest."""
    cli = FakeCodexCli.answering("hello")
    async with CodexCliCognition(spawn=FakeCodexCli.answering("hello")).session() as chat:
        result = (await take(chat.turn("hi")))[-1].result
    assert result.output == "hello"
    del cli


@pytest.mark.asyncio
async def test_every_turn_yields_exactly_one_final_event() -> None:
    """The contract the whole cognition rests on, asserted over a session's
    mixed run of success, failure and refusal."""
    cli = FakeCodexCli(
        [
            CliRun.of(codex_turn(text="ok", thread_id="T", usage=(1, 0, 1))),
            CliRun.of(codex_turn(failed="boom", thread_id="T"), returncode=1),
        ]
    )
    cog = CodexCliCognition(spawn=cli)
    chat = cog.session()
    await chat.start()

    for task in ("a", "b"):
        events = await take(chat.turn(task))
        assert len([e for e in events if e.type == "final"]) == 1

    await chat.close()
    events = await take(chat.turn("c"))
    assert len([e for e in events if e.type == "final"]) == 1


@pytest.mark.asyncio
async def test_a_session_turn_charges_the_meters_like_a_drive_does() -> None:
    """Both paths go through ``_finalise``, and this is the assertion that keeps
    them from drifting: a session whose turns did not reach the budget would be
    the same $0.00 ledger the Claude cognition shipped once."""
    from agentkit import Scope
    from agentkit.runtime import Budget, RunContext, Services

    budget = Budget(max_cost_usd=10.0)
    ctx = RunContext("run-1", Scope(), services=Services(), budget=budget)
    cli = FakeCodexCli.answering("one", "two")

    async with CodexCliCognition(spawn=cli).session() as chat:
        await take(chat.turn("a", ctx=ctx))
        await take(chat.turn("b", ctx=ctx))

    # Two turns, two charges. The token counts come from ``codex_turn``'s
    # default usage block, which is absent — so this asserts the STEPS, which is
    # the part that would silently be zero.
    assert budget.usage.total_tokens == 0
    assert budget.spent() == 0  # no priced model, so no cost — see the budget file


# ─────────────────────────────────────────────────────────────────────────────
# 5. against the real binary
# ─────────────────────────────────────────────────────────────────────────────


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_the_real_cli_keeps_context_across_turns(tmp_path: Path) -> None:
    """The claim the whole class makes, against the binary: turn two remembers a
    number given in turn one, and one thread id spans both."""
    cog = CodexCliCognition(
        working_dir=tmp_path,
        sandbox="read-only",
        ask_for_approval="never",
        skip_git_repo_check=True,
    )
    async with cog.session() as chat:
        await take(chat.turn("Remember the number 4271. Reply with just 'ok'."))
        second = (await take(chat.turn("What number did I ask you to remember? Digits only.")))[-1].result
        thread = chat.session_id

    assert "4271" in second.output, second.output
    assert thread, "no thread id came back"
    assert second.evals["session_id"] == thread


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_the_real_cli_session_survives_a_cancelled_turn(tmp_path: Path) -> None:
    """The advantage over a held process, proved rather than asserted: cancel a
    turn mid-flight and the next one still answers on the same thread."""
    cog = CodexCliCognition(
        working_dir=tmp_path,
        sandbox="read-only",
        ask_for_approval="never",
        skip_git_repo_check=True,
    )
    async with cog.session() as chat:
        await take(chat.turn("Remember the number 8813. Reply with just 'ok'."))
        thread = chat.session_id

        cancelling = _TrippedAfter(0.75)
        cancelled = (await take(chat.turn("Write a very long essay about compilers.", ctx=cancelling)))[-1]
        assert cancelled.result.evals["stop_reason"] == "cancelled"

        third = (await take(chat.turn("What number did I ask you to remember? Digits only.")))[-1].result

    assert chat.session_id == thread
    assert "8813" in third.output, third.output


class _TrippedAfter(FakeCtx):
    """A ctx whose cancel token trips after a wall-clock delay.

    Needed only for the real-binary tests: a real ``StreamReader`` DOES suspend,
    so time passes between lines and a deadline can be reached. Through the
    offline double nothing awaits, which is why every other test here trips on
    the first check instead.
    """

    def __init__(self, after_s: float) -> None:
        super().__init__()
        self._deadline = asyncio.get_event_loop().time() + after_s

    def check_cancelled(self) -> None:
        if asyncio.get_event_loop().time() >= self._deadline:
            raise RuntimeError("run cancelled")
