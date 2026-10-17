"""One Codex thread across many turns — a spawn each, resumed by thread id.

WHY THIS FILE EXISTS
--------------------
``codex exec`` is one-shot, so a session here is not a held process: it is a
thread id and the discipline to pass it back. Both halves of that fail quietly.

* **Forgetting the id** does not error. Every turn opens a fresh conversation,
  the model answers each question competently with no memory of the last, and
  the symptom reads as a bad model rather than as a bug. Nothing in the result
  says "this turn resumed nothing" — the only evidence is a missing ``resume``
  in the argv, which is why that is what these tests assert on.
* **Not being nameable** errors in the wrong place. ``Agent._span_attrs`` reads
  ``cognition.name`` on EVERY run, so a session without one raised
  ``AttributeError`` out of ``agent.run(...)`` — before the CLI was reached at
  all, on precisely the usage ``CodexCliSession.drive`` exists for.

A spawn is a TURN for this double, so ``FakeCodexCli`` exhaustion is itself an
assertion: a session that spawned twice for one ``turn()`` would raise rather
than quietly double-charging a conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.core.contracts.types import StreamEvent
from agentkit.core.testing.fakes import CliRun, FakeCodexCli, codex_turn
from agentkit.models.providers.cli.codex.driver import CodexCliCognition, CodexCliSession


@dataclass
class _Agent:
    prompt: str | None = None


async def _collect(stream: Any) -> list[StreamEvent]:
    return [ev async for ev in stream]


def _final(events: list[StreamEvent]) -> Any:
    assert [e.type for e in events].count("final") == 1, [e.type for e in events]
    return events[-1].result


@pytest.mark.asyncio
async def test_a_session_resumes_the_thread_its_first_turn_opened() -> None:
    """The whole of a Codex session is this: learn the thread id from turn one's
    ``thread.started`` and pass it back on turn two. Drop it and every turn
    starts a new conversation — a failure with no error, no stop reason and no
    trace in the result, which surfaces as a model that cannot follow up."""
    cli = FakeCodexCli.answering("first", "second")
    cognition = CodexCliCognition(spawn=cli)

    async with cognition.session() as chat:
        first = await _collect(chat.turn("who are you"))
        thread_id = _final(first).evals["session_id"]
        second = await _collect(chat.turn("and before that"))

    assert thread_id
    assert _final(second).output == "second"
    assert cli.spawns == 2
    # Turn one opens a thread; turn two must name the one turn one opened.
    assert "resume" not in cli.argv(0)
    resume_at = cli.argv(1).index("resume")
    assert cli.argv(1)[resume_at : resume_at + 2] == ("resume", thread_id)


@pytest.mark.asyncio
async def test_a_session_names_itself_so_it_can_be_an_agents_cognition() -> None:
    """``Agent._span_attrs`` reads ``cognition.name`` for the
    ``agentkit.agent.cognition`` trace attribute on every run, so a session
    without the attribute raises ``AttributeError`` straight out of
    ``agent.run(...)`` — before the CLI is reached, on the exact usage
    :meth:`CodexCliSession.drive` advertises. Suffixed rather than delegated
    verbatim because a trace has to distinguish one conversation across many
    turns from a fresh one each time."""
    cli = FakeCodexCli.answering("ok")
    session = CodexCliCognition(spawn=cli, name="codex_cli").session()

    assert session.name == "codex_cli_session"
    # Still derived from the cognition's own name, not hardcoded.
    assert CodexCliCognition(name="tenant_a").session().name == "tenant_a_session"
    # The attribute every observability path reads is present on the object a
    # caller actually hands to an agent.
    assert getattr(session, "name", None) == "codex_cli_session"

    async with session as chat:
        assert _final(await _collect(chat.turn("hi"))).output == "ok"


@pytest.mark.asyncio
async def test_the_system_prompt_is_sent_once_rather_than_re_prepended_per_turn() -> None:
    """The prompt belongs to the CONVERSATION, not to each turn. Prepending it
    again on turn three would put three copies in the transcript, which reads to
    the model as escalating emphasis and costs the prompt's tokens on every
    resumed turn."""
    cli = FakeCodexCli.answering("one", "two")
    cognition = CodexCliCognition(spawn=cli)
    agent = _Agent(prompt="Be terse.")

    async with cognition.session(agent=agent) as chat:
        await _collect(chat.turn("first question"))
        await _collect(chat.turn("second question"))

    assert cli.invocations[0].stdin == b"Be terse.\n\n---\n\nfirst question"
    assert cli.invocations[1].stdin == b"second question"


@pytest.mark.asyncio
async def test_a_closed_session_reports_a_terminal_event_instead_of_raising() -> None:
    """A session turn keeps the same exactly-one-``final`` contract a one-shot
    drive has, so "this session is over" has to arrive as a result rather than as
    an exception from inside a generator — otherwise a caller's ``async for``
    loop has two completely different failure shapes to handle depending on
    which side of ``close()`` it ran."""
    cli = FakeCodexCli.answering("ok")
    cognition = CodexCliCognition(spawn=cli)

    chat = cognition.session()
    await chat.start()
    await chat.close()
    events = await _collect(chat.turn("too late"))
    result = _final(events)

    assert result.evals["stop_reason"] == "session_closed"
    assert result.stop_reason == "failed"
    assert result.partial is True
    # Nothing was spawned, so the recording is untouched and the thread the
    # caller may still want to resume is named in the error.
    assert cli.spawns == 0
    assert "resume_session_id" in result.evals["error"]


@pytest.mark.asyncio
async def test_close_is_idempotent_so_an_async_with_inside_a_finally_is_safe() -> None:
    """Nothing is killed on close — a turn owns its subprocess for the whole of
    its life and there is never one running between turns — so a second close
    has nothing to do and must not object to being asked."""
    chat = CodexCliCognition(spawn=FakeCodexCli.answering("ok")).session()
    await chat.close()
    await chat.close()

    assert _final(await _collect(chat.turn("x"))).evals["stop_reason"] == "session_closed"


@pytest.mark.asyncio
async def test_a_session_counts_the_turns_it_actually_completed() -> None:
    """A session reporting one turn where the caller expected three is the
    difference between "the model was unhelpful" and "turns two and three never
    ran" — and those have nothing in common to debug."""
    cli = FakeCodexCli.answering("a", "b", "c")
    async with CodexCliCognition(spawn=cli).session() as chat:
        assert chat.turns_taken == 0
        await _collect(chat.turn("one"))
        assert chat.turns_taken == 1
        await _collect(chat.turn("two"))
        await _collect(chat.turn("three"))
        assert chat.turns_taken == 3

    assert cli.remaining == 0


@pytest.mark.asyncio
async def test_a_first_turn_that_learned_no_thread_id_starts_fresh_rather_than_resuming_nothing() -> None:
    """A first turn that died before ``thread.started`` leaves no thread, and a
    conversation with no history is exactly what it left behind. Resuming the
    empty string would be a CLI error; refusing the next turn would strand a
    session the caller can still use."""
    cli = FakeCodexCli([CliRun(stdout=b""), CliRun.of(codex_turn(text="recovered", thread_id=None))])
    async with CodexCliCognition(spawn=cli).session() as chat:
        first = _final(await _collect(chat.turn("one")))
        second = _final(await _collect(chat.turn("two")))

    assert first.evals["session_id"] == ""
    assert "resume" not in cli.argv(1)
    assert second.output == "recovered"


@pytest.mark.asyncio
async def test_a_session_adopts_a_thread_the_cognition_was_already_pointed_at() -> None:
    """``resume_session_id`` on the cognition names a thread from yesterday, and
    a session opened over it must continue THAT conversation from its first turn
    — otherwise the id is silently ignored and the caller gets a new thread
    while believing they reopened an old one."""
    cli = FakeCodexCli.answering("continued")
    cognition = CodexCliCognition(spawn=cli, resume_session_id="thread-from-yesterday")

    async with cognition.session() as chat:
        assert chat.session_id == "thread-from-yesterday"
        await _collect(chat.turn("still there"))

    resume_at = cli.argv(0).index("resume")
    assert cli.argv(0)[resume_at : resume_at + 2] == ("resume", "thread-from-yesterday")


def test_a_session_is_refused_when_the_cli_is_told_to_write_no_session_file() -> None:
    """``ephemeral=True`` means the CLI persists nothing, and a thread that was
    never written cannot be resumed. Accepting the combination would produce a
    session whose every turn silently started over — the failure this class is
    hardest to notice."""
    with pytest.raises(ValueError, match="ephemeral"):
        CodexCliCognition(ephemeral=True).session()


def test_a_session_is_a_distinct_type_from_the_cognition_that_opened_it() -> None:
    """The two have deliberately different turn-taking behaviour to debug, and
    ``session()`` returning the cognition itself would make a resumed thread
    indistinguishable from a fresh run in every trace and every ``isinstance``
    check a caller writes."""
    cognition = CodexCliCognition()
    chat = cognition.session()

    assert isinstance(chat, CodexCliSession)
    assert chat.name != cognition.name
