"""A `FakeLLM.script` that runs out must SAY SO, not repeat its last turn.

The behaviour this file pins down was a one-character bug with an outsized
blast radius. `FakeLLM.stream` picked its turn with
``self._turns[min(self._turn_idx, len(self._turns) - 1)]``, so the index was
clamped and an exhausted script replayed its final turn forever. Measured on
the shipped code:

    FakeLLM.script([Turn(content="one"), Turn(content="two")])
    4 calls on a 2-turn script -> ['one', 'two', 'two', 'two']

`FakeLLM` is this framework's primary test double — it backs most of the suite
and nearly every doc example — and a script exists to drive an agent through an
EXPECTED number of turns. So "the agent asked for a turn the script does not
have" is never a fact about the script; it is a fact about the loop under test,
which failed to stop. The clamp took that finding and answered it with a
stable, plausible, wrong reply, and the assertion at the end of the test still
passed. The harness written to catch non-termination was the thing hiding it.

That is the same failure shape as a broad ``except`` that reports success, and
this repo has shipped that one before.

BLAST RADIUS, measured before changing anything: the clamp was instrumented to
log every overrun and the full suite was run — 3621 tests, 4 overrun events
across exactly 3 tests, all three in ``tests/agents/test_agent_loop.py`` and all
three with a 1-turn script deliberately used as a never-terminating loop
(``max_iterations``/budget is what ends the run). So raising by default costs 3
deliberate call sites, all of which now say ``repeat_last=True`` out loud. An
opt-in ``strict=True`` was rejected for the usual reason: a safety net nobody
opts into catches nothing, and the 3618 tests that consume their script exactly
would keep no protection at all.
"""

from __future__ import annotations

import pytest

from agentkit.kernel.types import Message, ToolCall
from agentkit.testing import FakeLLM, ScriptExhausted, Turn

_MSGS = [Message(role="user", content="hi")]


async def _say(llm: FakeLLM, n: int) -> list[str]:
    """`n` chat calls, contents in order."""
    return [(await llm.chat(messages=_MSGS, model="m")).content for _ in range(n)]


# ── the regression itself ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_script_asked_for_more_turns_than_it_has_raises() -> None:
    """THE regression. Pre-fix this returned ``['one', 'two', 'two', 'two']``."""
    llm = FakeLLM.script([Turn(content="one"), Turn(content="two")])

    with pytest.raises(ScriptExhausted):
        await _say(llm, 4)


@pytest.mark.asyncio
async def test_the_turns_before_exhaustion_are_still_delivered_in_order() -> None:
    """Raising must not corrupt what came before — a test that consumed two
    turns and then looped once too often should still be able to see that its
    first two turns were right, which is half the diagnosis."""
    llm = FakeLLM.script([Turn(content="one"), Turn(content="two")])
    said = []

    with pytest.raises(ScriptExhausted):
        for _ in range(4):
            said.append((await llm.chat(messages=_MSGS, model="m")).content)

    assert said == ["one", "two"]


@pytest.mark.asyncio
async def test_the_message_names_both_numbers_and_the_way_out() -> None:
    """The only value of raising is in the string. Someone reading it at 2am
    needs to learn: how long the script was, which turn was asked for, and that
    the loop — not the script — is the likely defect. Asserting on the numbers
    keeps a future reword from quietly dropping them."""
    llm = FakeLLM.script([Turn(content="one"), Turn(content="two")])

    with pytest.raises(ScriptExhausted) as caught:
        await _say(llm, 3)

    text = str(caught.value)
    assert "2 turn(s)" in text  # how many the script had
    assert "turn 3" in text  # how many were requested
    assert "repeat_last=True" in text  # the escape hatch, by its real name


@pytest.mark.asyncio
async def test_exhaustion_is_not_catchable_as_exception() -> None:
    """``ScriptExhausted`` is a `BaseException` on purpose. Between
    `FakeLLM.stream` and the test body sit ~20 deliberate ``except Exception``
    handlers — react reflecting bad output back to the model,
    ``_invoke_tool_safe`` turning every tool failure into a tool message,
    ``resilience`` classifying pre-stream faults into retry-vs-fail. Every one
    of them is right for a real provider fault and every one of them would
    swallow "your loop does not terminate" and carry on. If someone re-parents
    this to `Exception` for tidiness, the bug comes back wearing a new hat."""
    llm = FakeLLM.script([Turn(content="only")])
    await _say(llm, 1)

    with pytest.raises(ScriptExhausted):
        try:
            await _say(llm, 1)
        except Exception:  # noqa: BLE001 — this is the point of the test
            pytest.fail("ScriptExhausted was catchable as Exception; production handlers eat it")


# ── the deliberate never-terminating script ──────────────────────────────────


@pytest.mark.asyncio
async def test_repeat_last_replays_the_final_turn_forever() -> None:
    """The escape hatch, which is what the 3 pre-existing call sites needed:
    a single tool-call turn that keeps coming until a ceiling stops the run."""
    llm = FakeLLM.script(
        [Turn(tool_calls=(ToolCall("c1", "fetch", {}),))],
        repeat_last=True,
    )

    res = [await llm.chat(messages=_MSGS, model="m") for _ in range(5)]

    assert all(r.tool_calls == (ToolCall("c1", "fetch", {}),) for r in res)
    assert all(r.finish_reason == "tool_calls" for r in res)


@pytest.mark.asyncio
async def test_repeat_last_repeats_only_the_last_turn_not_the_whole_script() -> None:
    """It is a clamp, not a cycle. A test asserting on turn 4 of a 3-turn
    script should get the terminal turn again, never wrap around to turn 1."""
    llm = FakeLLM.script(
        [Turn(content="a"), Turn(content="b"), Turn(content="c")],
        repeat_last=True,
    )

    assert await _say(llm, 5) == ["a", "b", "c", "c", "c"]


@pytest.mark.asyncio
async def test_an_empty_script_raises_even_with_repeat_last() -> None:
    """There is no last turn to repeat. Guarding this explicitly because
    ``self._turns[-1]`` on an empty list raises `IndexError`, and an
    `IndexError` from inside a fake is the least informative failure available."""
    llm = FakeLLM.script([], repeat_last=True)

    with pytest.raises(ScriptExhausted) as caught:
        await _say(llm, 1)

    assert "0 turn(s)" in str(caught.value)


# ── positive controls: everything that was already right stays right ─────────


@pytest.mark.asyncio
async def test_the_single_response_form_still_repeats() -> None:
    """`FakeLLM("x")` is a RULE for answering, not a finite script — it has
    nothing to exhaust, so call 1 and call 100 both answer. Making this raise
    too would have been the over-correction."""
    assert await _say(FakeLLM("x"), 3) == ["x", "x", "x"]


@pytest.mark.asyncio
async def test_the_dict_and_callable_forms_are_unaffected() -> None:
    """The other two single-answer forms take the same non-script branch."""
    assert await _say(FakeLLM({"hi": "matched"}), 2) == ["matched", "matched"]
    assert await _say(FakeLLM(lambda **kw: "called"), 2) == ["called", "called"]


@pytest.mark.asyncio
async def test_a_script_consumed_exactly_to_its_length_is_silent() -> None:
    """The overwhelmingly common case — 3618 of the 3621 tests in this suite.
    If this ever fails, the fix has become the bug."""
    llm = FakeLLM.script([Turn(content="one"), Turn(content="two"), Turn(content="three")])

    assert await _say(llm, 3) == ["one", "two", "three"]


@pytest.mark.asyncio
async def test_call_counting_is_unchanged_including_the_exhausted_call() -> None:
    """`FakeLLM.calls` counts ATTEMPTS, and always has — `_enter()` increments
    before anything can fail. The call that hits exhaustion is a real call the
    agent made, so it counts; tests asserting ``llm.calls == n`` must keep
    reading the same number they read before this change."""
    llm = FakeLLM.script([Turn(content="one")])
    await _say(llm, 1)
    assert llm.calls == 1

    with pytest.raises(ScriptExhausted):
        await _say(llm, 1)
    assert llm.calls == 2


# ── fail_times, which must not interact with exhaustion ──────────────────────


@pytest.mark.asyncio
async def test_a_failed_call_does_not_burn_a_scripted_turn() -> None:
    """`_enter()` raises BEFORE the turn is picked, so a `fail_times` failure
    produces no reply and must not consume script. Otherwise ``fail_times=2``
    plus a 3-turn script would silently mean "two failures that eat two thirds
    of the script", and the retry test it was written for would assert against
    the wrong turn."""
    llm = FakeLLM.script([Turn(content="one"), Turn(content="two")], fail_times=2)

    for _ in range(2):
        with pytest.raises(TimeoutError):
            await _say(llm, 1)

    assert await _say(llm, 2) == ["one", "two"]
    assert llm.calls == 4  # 2 failures + 2 delivered turns


@pytest.mark.asyncio
async def test_fail_times_still_wins_over_an_already_exhausted_script() -> None:
    """Ordering check. A configured failure is about the transport and fires
    first; exhaustion is about the script and is only reached by a call that
    got past the transport. Pinned so a future reorder cannot turn a retry
    test's expected `TimeoutError` into a `ScriptExhausted`."""
    llm = FakeLLM.script([], fail_times=1)

    with pytest.raises(TimeoutError):
        await _say(llm, 1)
