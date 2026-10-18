"""Is `FakeCodexCli` still a faithful stand-in for `codex exec --json`?

WHY THIS FILE EXISTS
--------------------
A test double is only worth what its fidelity is worth, and this one carries
more of the Codex path's coverage than any single production module does. The
asymmetry is what makes it dangerous: a double that is too strict fails loudly,
while a double that is too permissive — one that answers a spawn the recording
never promised, or emits payloads in an order the binary never emits — turns
every test downstream of it green for the wrong reason.
``tests/source/unit/models/providers/cli/test_cli_version_matrix.py`` is the
record of what that costs: three shipped bugs, all of them the suite agreeing
with itself while the real binary had moved. It can check argv against the real
``codex``; nothing can check a replayed STREAM against it, so the stream's shape
is pinned here instead.

Two properties, both of which fail silently:

* **Exhaustion raises; it does not improvise.** ``codex exec`` is one-shot, so a
  spawn is a TURN, and the recording is therefore a claim about how many turns
  the code under test takes. A session that spawns twice for one ``turn()`` has
  a defect — a retry nobody asked for, a loop that re-entered — and replaying
  the last run answers that defect with a stable, plausible, wrong result. The
  ``repeat_last`` opt-in is what a test that deliberately drives an unbounded
  number of spawns uses instead, and the two cases have to stay distinct: an
  exhausted recording raises, an exhausted ``repeat_last`` recording does not,
  and an EMPTY recording raises either way rather than indexing off the end of
  nothing.
* **A built turn is in the binary's own order.** ``thread.started`` first,
  because the thread id is what a session resumes; the terminal
  ``turn.completed`` last, because the usage on it is what the budget is charged
  from. ``codex_turn`` exists so that no test hand-writes five payloads and gets
  that wrong — which means if the builder itself emits them out of order, every
  test that asserts on a stream is asserting against a stream that cannot
  happen, and the parser is free to depend on an order the binary never gives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.core.testing.fakes import FakeCodexCli, ScriptExhausted, codex_turn
from agentkit.models.providers.cli.codex.driver import CodexCliCognition


@dataclass
class _Agent:
    """The only attributes this cognition reads off an agent."""

    prompt: str | None = None
    model: str | None = None


async def _drive(cognition: CodexCliCognition, *, task: str = "go") -> list[Any]:
    """Every event one ``drive`` produced, terminal event last."""
    return [ev async for ev in cognition.drive(_Agent(), task, None, None)]


def _final(events: list[Any]) -> Any:
    """The one terminal result, asserting the exactly-one-``final`` contract."""
    types = [e.type for e in events]
    assert types.count("final") == 1, types
    assert types[-1] == "final", types
    return events[-1].result


# ── exhaustion ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_double_with_nothing_recorded_raises_rather_than_reaching_for_a_run() -> None:
    """An empty recording is the degenerate end of the exhaustion rule, and the
    one the condition is easiest to get wrong at.

    "Replay the last run" has no last run to replay here, so a double that
    decided on ``repeat_last`` before checking that it has anything at all
    indexes off the end of an empty tuple — and an ``IndexError`` is an
    ordinary ``Exception``, which means it is caught by the cognition's
    terminal-event path and reported as ``AgentResult(partial=True)``. The
    fixture problem then presents as a model failure, inside the result, in
    exactly the place a reader does not look for it. ``ScriptExhausted`` is a
    ``BaseException`` precisely so it comes out here instead, and it holds
    both the count and both ways forward because the message IS the fix.
    """
    for cli in (FakeCodexCli(), FakeCodexCli([], repeat_last=True)):
        with pytest.raises(ScriptExhausted) as raised:
            await _drive(CodexCliCognition(spawn=cli))

        assert "FakeCodexCli exhausted" in str(raised.value)
        assert "has 0 run(s)" in str(raised.value)
        assert "repeat_last=True" in str(raised.value), "the complaint has to carry the way out"


@pytest.mark.asyncio
async def test_repeat_last_replays_the_final_run_instead_of_raising_at_the_end_of_it() -> None:
    """The other side of the same condition: a test that deliberately drives an
    unbounded number of spawns asks for ``repeat_last`` and must get it.

    Collapsing the two cases in either direction breaks a whole class of test
    and the breakage reads as something else. Lose ``repeat_last`` and a loop
    test fails with a fixture error on iteration two rather than measuring the
    loop. Lose the raise and a retry nobody asked for is answered with a
    plausible result. The cursor keeps advancing past the end as well, so such
    a test can assert HOW MANY spawns the loop made and not merely that it made
    extra ones.
    """
    cli = FakeCodexCli.script(codex_turn(text="same answer", usage=(10, 0, 5)), repeat_last=True)
    cognition = CodexCliCognition(spawn=cli)

    answers = [_final(await _drive(cognition)).output for _ in range(3)]

    assert answers == ["same answer"] * 3
    assert cli.spawns == 3, "three turns are three spawns: this cognition resumes rather than holding a process"
    assert cli.remaining == 0


# ── the order the binary actually emits ──────────────────────────────────────


def test_a_built_turn_announces_its_thread_first_and_ends_on_its_terminal_payload() -> None:
    """The order in a Codex turn carries meaning the parser is entitled to rely
    on: the thread id arrives before anything else because it is what a session
    resumes, and the terminal event arrives last because its usage is what the
    turn is billed by.

    A builder that emitted the terminal payload FIRST would still produce a
    stream every test parses happily — the same payloads are all present, the
    answer still comes out right — while every one of those tests silently
    asserts against a stream ``codex`` cannot produce. That is the failure mode
    worth paying for a builder at all: a stream in the wrong order lets the
    parser depend on the wrong order, and nothing notices until a real binary
    is on the other end. Asserted as the type sequence rather than by spot
    checks, because "last" and "first" are the whole property.
    """
    turn = codex_turn(reasoning="thinking", text="answer", usage=(1200, 1000, 40))

    assert [p["type"] for p in turn] == [
        "thread.started",
        "turn.started",
        "item.completed",  # the reasoning
        "item.completed",  # the agent message
        "turn.completed",
    ]
    assert turn[-1]["usage"] == {"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 40}, (
        "the usage rides the LAST payload; a turn that is billed before it has answered is billed for nothing"
    )
    # The failure flavour is the same shape, terminal and last — so a test can
    # swap one for the other without the rest of the stream moving.
    assert [p["type"] for p in codex_turn(text="partial", failed="sandbox denied")][-1] == "turn.failed"


@pytest.mark.asyncio
async def test_a_replayed_turn_is_billed_from_the_payload_that_closed_it() -> None:
    """The order above, through the parser that consumes it.

    The point of checking it twice is that the builder's order and the
    cognition's reading of it are the same assumption held in two places, and a
    terminal payload arriving early is survivable enough to hide: the fold still
    sees every item and the answer still reads correctly. What moves is the
    metering — a ``turn.completed`` read before the turn has said anything bills
    a turn that produced nothing, and the budget is wrong in the direction that
    looks like the model being cheap.
    """
    cli = FakeCodexCli.script(codex_turn(text="done", thread_id="thread-9", usage=(1200, 1000, 40)))
    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.output == "done"
    assert result.stop_reason == "complete"
    assert result.evals["session_id"] == "thread-9", "the thread id is read off the FIRST payload"
    assert result.usage.input_tokens == 200
    assert result.usage.cache_read_tokens == 1000
    assert result.usage.output_tokens == 40
