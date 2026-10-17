"""Does the Codex stream fold put every fact in the answer exactly once?

WHY THIS FILE EXISTS
--------------------
``codex exec --json`` says the same thing more than once on purpose. An item
arrives as ``item.updated`` and then again as ``item.completed``, each carrying
the WHOLE text so far, and the current vocabulary is not the only one in the
wild — an older ``codex`` speaks ``{"id": …, "msg": {…}}`` instead. Three
different mistakes are available at that seam and none of them raises:

* billing the cached prefix at the full input rate, because Codex counts input
  INCLUSIVE of cache and agentkit's convention is the other one — and a Codex
  session is mostly cache, so the error is roughly the whole prompt per turn;
* re-emitting text the updates already streamed, so a UI with a cursor in it
  shows every sentence twice while the final answer still looks right;
* folding an item's text on every phase, so the answer holds two copies of one
  message;
* dropping the legacy vocabulary, which produces a run that exits 0, emits no
  events and returns an empty answer with no stop reason. Forward-compat here
  is "don't crash", so a parser that stops recognising a shape does not fail —
  it goes quiet.

Each test below pins one of those, against payloads written in the order the
binary emits them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.core.contracts.types import StreamEvent
from agentkit.core.testing.fakes import FakeCodexCli, codex_turn
from agentkit.models.providers.cli.codex.driver import CodexCliCognition


@dataclass
class _Agent:
    """The only attributes this cognition reads off an agent."""

    prompt: str | None = None
    model: str | None = None


async def _drive(cognition: CodexCliCognition, *, task: str = "go") -> list[StreamEvent]:
    """Every event one ``drive`` produced, terminal event last."""
    return [ev async for ev in cognition.drive(_Agent(), task, None, None)]


def _final(events: list[StreamEvent]) -> Any:
    """The one terminal result, asserting the exactly-one-``final`` contract."""
    assert [e.type for e in events].count("final") == 1, [e.type for e in events]
    assert events[-1].type == "final"
    return events[-1].result


def _deltas(events: list[StreamEvent]) -> list[str]:
    return [e.text for e in events if e.type == "message_delta"]


def _two_phase_message() -> tuple[dict[str, Any], ...]:
    """One assistant message announced twice — the shape every real turn has.

    ``item.updated`` carries the text so far and ``item.completed`` carries all
    of it again, which is the sequence both the suffix rule and the fold-once
    rule exist for.
    """
    return (
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {"type": "item.updated", "item": {"id": "m1", "type": "agent_message", "text": "Hello"}},
        {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "Hello world"}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}},
    )


@pytest.mark.asyncio
async def test_the_cached_prefix_is_subtracted_from_the_input_the_run_is_billed_for() -> None:
    """Codex reports input INCLUSIVE of the cached prefix and reports the cache
    separately; ``ProviderUsage.input_tokens`` means the fresh, full-rate half.
    Leaving the two unsubtracted bills every cached token at the uncached rate,
    and a Codex session is mostly cache — so the estimate, the meters and the
    budget are all wrong by about a whole prompt on every single turn, in the
    direction that looks like the model simply costs more than quoted."""
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(1200, 1000, 40)))
    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.usage.input_tokens == 200
    assert result.usage.cache_read_tokens == 1000
    assert result.usage.output_tokens == 40
    # The CLI's own numbers survive untouched alongside the split, so the
    # subtraction is auditable rather than something a reader has to infer.
    assert result.evals["raw_usage"]["input_tokens"] == 1200
    assert result.evals["raw_usage"]["cached_input_tokens"] == 1000


@pytest.mark.asyncio
async def test_a_cached_count_larger_than_the_total_clamps_instead_of_crediting_the_budget() -> None:
    """The two counts come from different places in the CLI, so ``cached >
    total`` is reachable without anything exotic. Unclamped it produces a
    NEGATIVE ``input_tokens``, which flows straight into a meter and makes a
    budget go UP — a run that pays for itself and then some."""
    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(10, 99, 5)))
    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.usage.input_tokens == 0
    assert result.usage.cache_read_tokens == 99
    assert result.usage.total_tokens >= 0


@pytest.mark.asyncio
async def test_a_completed_item_streams_only_the_text_its_updates_had_not_sent() -> None:
    """An item's text is CUMULATIVE: the completion repeats everything the
    updates already carried. A consumer rendering ``message_delta`` appends what
    it is handed, so emitting the whole value again shows the reader every
    sentence twice — and it is invisible from the result, because the folded
    answer stays correct while the live stream is doubled."""
    cli = FakeCodexCli.script(_two_phase_message())
    events = await _drive(CodexCliCognition(spawn=cli))

    assert _deltas(events) == ["Hello", " world"]
    assert "".join(_deltas(events)) == _final(events).output


@pytest.mark.asyncio
async def test_an_items_text_is_folded_into_the_answer_once_however_many_phases_carried_it() -> None:
    """The flip side of the suffix rule: only the ``completed`` phase folds text
    into the turn. Folding on every phase records the update AND the completion
    as two separate assistant messages, so a transcript — and anything built
    from ``tests.evals["messages"]``, a repair prompt or a replay — holds one
    message twice while ``output`` looks fine."""
    cli = FakeCodexCli.script(_two_phase_message())
    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.output == "Hello world"
    assert result.evals["messages"] == ("Hello world",)




@pytest.mark.asyncio
async def test_a_reasoning_summary_is_streamed_live_but_kept_out_of_the_answer() -> None:
    """Reasoning is how the answer was reached, not the answer. Folding it into
    ``output`` would hand a caller — and any structured-output decode — the
    model's scratchpad as the response, while dropping it from the stream would
    lose the only explanation a 40-second run has."""
    cli = FakeCodexCli.script(codex_turn(reasoning="weighing options", text="42", usage=(1, 0, 1)))
    events = await _drive(CodexCliCognition(spawn=cli))
    result = _final(events)

    assert result.output == "42"
    assert result.evals["thinking"] == "weighing options"
    assert _deltas(events) == ["weighing options", "42"]


@pytest.mark.asyncio
async def test_a_failed_turn_keeps_the_diagnosis_a_retry_decision_needs() -> None:
    """``turn.failed`` is the turn going wrong, as opposed to the pipe breaking,
    and the fields on it are what a caller retries against. A failure that
    arrives as a bare ``failed`` with no status and no ``retry_after`` turns a
    rate limit into an unclassifiable error, which is retried immediately or
    not at all."""
    cli = FakeCodexCli.script(
        (
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.started"},
            {
                "type": "turn.failed",
                "error": {"message": "rate limited", "status": 429, "code": "rate_limit", "retry_after": 0.25},
            },
        )
    )
    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.stop_reason == "failed"
    assert result.evals["stop_reason"] == "turn_failed"
    assert result.evals["api_error_status"] == 429
    assert result.evals["cli_error_code"] == "rate_limit"
    assert result.evals["retry_after_s"] == 0.25
    assert result.partial is True


@pytest.mark.asyncio
async def test_an_item_level_error_is_recorded_without_failing_a_run_that_worked() -> None:
    """"Command output truncated" is a diagnostic, not an outcome. Promoting it
    to a failed run would throw away an answer the CLI produced; dropping it
    leaves a run that took forty seconds and looks fine with nothing in the
    result that explains why."""
    cli = FakeCodexCli.script(
        codex_turn(
            text="done",
            usage=(1, 0, 1),
            items=[{"id": "e1", "type": "error", "message": "output truncated"}],
        )
    )
    events = await _drive(CodexCliCognition(spawn=cli))
    result = _final(events)

    assert result.partial is False
    assert result.stop_reason == "complete"
    assert result.evals["cli_errors"] == ["output truncated"]
    assert any(e.type == "step" and e.text == "item_error:output truncated" for e in events)


@pytest.mark.asyncio
async def test_a_stream_that_stops_before_the_clis_end_of_turn_payload_is_not_a_success() -> None:
    """A successful terminal payload leaves ``stop_reason`` at ``None`` — the
    same value a stream killed mid-flight has — so "did the CLI say how this
    ended" has to be tracked rather than inferred. Measured before it was:
    stdout truncated mid-object reported ``complete``, ``partial=False``, and
    handed back half an answer as a finished one."""
    cli = FakeCodexCli.script(
        (
            {"type": "thread.started", "thread_id": "thread-1"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "half an ans"}},
        )
    )
    result = _final(await _drive(CodexCliCognition(spawn=cli)))

    assert result.partial is True
    assert result.evals["stop_reason"] == "malformed_output"
    assert result.stop_reason == "failed"
    assert result.output == "half an ans"




@pytest.mark.asyncio
async def test_an_old_codex_is_refused_loudly_rather_than_answering_with_nothing() -> None:
    """`codex` is a binary the user installs, so a service does not get to pick
    their version — and agentkit reads one vocabulary, the current `thread.*` one.
    Declining the superseded `{"id", "msg"}` stream is a decision; declining it
    SILENTLY would be a bug.

    An unrecognised payload is ignored by design, because forward-compat here is
    "don't crash". So without an explicit refusal an old binary exits 0, emits no
    events, and hands back an empty answer with `stop_reason=None` — a run that
    looks like it worked. The failure has to name the cause and the fix, because
    "upgrade codex" is not something the caller can infer from an empty string.
    """
    legacy = (
        {"id": "0", "msg": {"type": "session_configured", "session_id": "old-1", "model": "gpt-5-codex"}},
        {"id": "1", "msg": {"type": "agent_message", "message": "an answer nobody should receive"}},
        {"id": "2", "msg": {"type": "task_complete"}},
    )
    result = _final(await _drive(CodexCliCognition(spawn=FakeCodexCli.script(legacy))))

    # `stop_reason` is the CLOSED taxonomy; the specific reason is in `evals`,
    # which is where a caller looks to tell "upgrade your binary" apart from
    # every other way a codex run can fail.
    assert result.stop_reason == "failed", result.stop_reason
    assert result.evals["stop_reason"] == "cli_version_unsupported", result.evals
    assert result.output == "", "the superseded stream was translated after all"
    assert list(result.evals["cli_errors"]) == [
        'this `codex` emits the superseded {"id", "msg"} event stream; agentkit reads the '
        "`thread.*` / `turn.*` vocabulary only. Upgrade the `codex` binary on PATH."
    ], (
        "the message must name both the shape it saw and the fix, exactly ONCE — the "
        "refusal is a property of the process, and three old-shaped lines used to "
        "produce three identical errors"
    )
