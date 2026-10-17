"""Token streaming and startup facts — what `--include-partial-messages` costs.

WHY THIS FILE EXISTS
--------------------
Partial streaming makes the same text arrive TWICE on one stdout: once as
provider SSE deltas (``stream_event`` payloads, a few characters each) and once
as the completed ``assistant`` message, which is the authoritative copy. Every
bug on this path is a double-count or a drop, and neither raises:

* re-emitting the completed block renders every sentence twice in a UI that was
  already shown the tokens;
* accumulating the tokens as well as the block doubles ``AgentResult.output``,
  and the doubled text is perfectly well-formed;
* omitting ``--include-partial-messages`` produces no tokens at all, so a
  streaming UI shows nothing until the answer lands — and looks merely slow;
* dropping the ``system/init`` payload loses the one record of which MCP
  servers were SKIPPED, which the CLI's own docs recommend gating CI on, and
  manufacturing empty error lists destroys the signal entirely, since the CLI
  omits those keys when there is nothing wrong;
* swallowing ``system/api_retry`` leaves a forty-second pause with no
  explanation anywhere in the result.

The parser is exercised through a real session turn wherever the behaviour is
observable there, and directly where the invariant is about the state DELTA a
payload carries rather than about the event it produces. ``FakeClaudeCli``
supplies the bytes and nothing else, so everything below the subprocess seam —
``_parse_line``, ``_events_from_payload``, ``_TurnState.fold``, ``_finalise`` —
runs exactly as a real run runs it. No binary, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentkit.core.testing.fakes import CliRun, FakeClaudeCli
from agentkit.models.providers.cli.claude import driver as cc
from agentkit.models.providers.cli.claude.driver import ClaudeCliCognition

# ── stream-json payload vocabulary ───────────────────────────────────────────


def _init(**extra: Any) -> dict[str, Any]:
    """``system/init`` — the startup metadata an operator needs when a run
    behaves oddly: which model ran, which MCP servers connected, which did
    not."""
    return {"type": "system", "subtype": "init", "session_id": "sess-1", **extra}


def _text_token(text: str) -> dict[str, Any]:
    """One provider SSE ``text_delta``, as ``--include-partial-messages``
    forwards it."""
    return {
        "type": "stream_event",
        "session_id": "sess-1",
        "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    }


def _thinking_token(text: str) -> dict[str, Any]:
    """One extended-thinking SSE delta. Renderable text, but NOT the answer."""
    return {
        "type": "stream_event",
        "session_id": "sess-1",
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": text},
        },
    }


def _assistant(text: str, message_id: str = "msg-1") -> dict[str, Any]:
    """The completed assistant message — the authoritative copy of the text the
    tokens above spelled out one fragment at a time."""
    return {
        "type": "assistant",
        "message": {"id": message_id, "role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _result(**extra: Any) -> dict[str, Any]:
    """A turn's end-of-turn payload. Deliberately carries NO ``result`` string
    in most tests here, so the folded text is the only source of
    ``AgentResult.output`` and a double-count has nowhere to hide."""
    return {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "usage": {"input_tokens": 5, "output_tokens": 9},
        **extra,
    }


async def _one_turn(cognition: ClaudeCliCognition) -> list[Any]:
    """Drive one session turn over the recording and hand back its events."""
    async with cognition.session() as chat:
        return [ev async for ev in chat.turn("summarise it")]


# ── token streaming ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_token_deltas_stream_once_and_the_completed_block_is_not_repeated() -> None:
    """Under partial streaming the consumer has already seen the text token by
    token.

    The completed ``assistant`` message carries the same characters again,
    because it is the authoritative copy the state folds from — but emitting it
    as an event too shows every sentence twice in a live UI, with no error and
    nothing in the result to notice. The state still flows from that message,
    which is why ``output`` is the text exactly once.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    _text_token("Hel"),
                    _text_token("lo "),
                    _text_token("there"),
                    _assistant("Hello there"),
                    _result(),
                ]
            )
        ]
    )
    events = await _one_turn(ClaudeCliCognition(spawn=cli, partial_messages=True))

    deltas = [ev.text for ev in events if ev.type == "message_delta"]
    assert deltas == ["Hel", "lo ", "there"], deltas
    assert events[-1].type == "final"
    assert events[-1].result.output == "Hello there"
    assert events[-1].result.partial is False


@pytest.mark.asyncio
async def test_without_partial_streaming_the_completed_block_is_the_only_delta() -> None:
    """The other half of the same contract: with no token stream to duplicate,
    the completed message IS the event.

    Suppressing it unconditionally would leave a non-streaming run with a
    terminal event and nothing before it — a consumer that renders
    ``message_delta`` shows an empty answer and then a finished result.
    """
    cli = FakeClaudeCli([CliRun.of([_init(), _assistant("Hello there"), _result()])])
    events = await _one_turn(ClaudeCliCognition(spawn=cli))

    assert [ev.text for ev in events if ev.type == "message_delta"] == ["Hello there"]
    assert events[-1].result.output == "Hello there"


@pytest.mark.asyncio
async def test_a_token_delta_carries_no_state_so_the_answer_is_not_counted_twice() -> None:
    """A token event is for RENDERING; the completed message is for state.

    Both describe the same characters, so a token delta that also carries text
    hands the fold a second copy of every fragment — the same answer twice in
    ``AgentResult.output``, well-formed, with the duplication invisible until
    somebody reads it. The invariant is on the delta rather than on the output
    because that is where it is enforceable: the fold applies text only
    alongside a completed message today, so a token delta carrying text is
    latent damage that the next change to ``fold`` would release.
    """
    emitted: list[tuple[Any, Any]] = []
    async for event, delta in cc._events_from_payload(_text_token("Hel"), partial=True):
        emitted.append((event, delta))

    assert [ev.text for ev, _ in emitted] == ["Hel"], "the token must still be renderable"
    assert [d for _, d in emitted] == [cc._EventDelta()], (
        "a token delta must contribute nothing to the turn's state"
    )

    state = cc._TurnState()
    for _, delta in emitted:
        state.fold(delta)
    assert state.text == ""


@pytest.mark.asyncio
async def test_thinking_tokens_stream_as_text_without_becoming_the_answer() -> None:
    """A reasoning chain is renderable and is not the response.

    Folding it into ``output`` would put the model's scratch work in the answer
    a caller returns to a user; dropping the event would make extended thinking
    invisible while it happens, which is the whole reason it is streamed.
    """
    emitted: list[tuple[Any, Any]] = []
    async for event, delta in cc._events_from_payload(_thinking_token("weighing it"), partial=True):
        emitted.append((event, delta))

    assert [(ev.type, ev.text) for ev, _ in emitted] == [("message_delta", "weighing it")]
    state = cc._TurnState()
    for _, delta in emitted:
        state.fold(delta)
    assert state.text == ""


@pytest.mark.asyncio
async def test_asking_for_token_streaming_passes_the_flag_that_produces_tokens() -> None:
    """``partial_messages=True`` is a request the CLI only honours when it is
    told.

    Without ``--include-partial-messages`` the binary emits no ``stream_event``
    payload at all, so the configuration reads as enabled, every parsing path
    for it stays correct, and a streaming UI simply never receives a token —
    which looks like provider latency rather than a missing flag.
    """
    streaming = FakeClaudeCli([CliRun.of([_init(), _assistant("hi"), _result()])])
    await _one_turn(ClaudeCliCognition(spawn=streaming, partial_messages=True))

    plain = FakeClaudeCli([CliRun.of([_init(), _assistant("hi"), _result()])])
    await _one_turn(ClaudeCliCognition(spawn=plain))

    assert "--include-partial-messages" in streaming.invocations[0].argv, streaming.invocations[0].argv
    assert "--include-partial-messages" not in plain.invocations[0].argv, plain.invocations[0].argv


# ── startup facts ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_skipped_mcp_server_survives_into_the_result() -> None:
    """``mcp_server_errors`` is the one record that a tool the run was supposed
    to have never connected.

    Discarding the init payload throws it away with everything else, and the
    run then looks like a model that chose not to use a tool rather than a
    session that never had it. The CLI's own docs recommend gating CI on the
    presence of these keys, which is only possible if they reach the result.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(
                        model="claude-opus-4-5",
                        claude_code_version="2.1.236",
                        permissionMode="acceptEdits",
                        mcp_servers=[{"name": "ledger", "status": "connected"}],
                        mcp_server_errors={"invoices": "spawn failed: ENOENT"},
                    ),
                    _assistant("I had no invoice tool."),
                    _result(),
                ]
            )
        ]
    )
    events = await _one_turn(ClaudeCliCognition(spawn=cli))

    init = events[-1].result.evals["cli_init"]
    assert init["mcp_server_errors"] == {"invoices": "spawn failed: ENOENT"}, init
    assert init["model"] == "claude-opus-4-5"
    assert init["permissionMode"] == "acceptEdits"


@pytest.mark.asyncio
async def test_the_init_record_holds_only_the_fields_the_cli_actually_sent() -> None:
    """The CLI omits ``mcp_server_errors`` / ``plugin_errors`` entirely when
    nothing went wrong, so PRESENCE is the signal.

    Manufacturing the keys — empty lists for a healthy run — makes every run
    look like it has an error record, and the CI gate that tests for one can
    never fire again. Absence has to stay absence.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(model="claude-opus-4-5", capabilities=["interrupt_cancel_queued_v1"]),
                    _assistant("all tools connected"),
                    _result(),
                ]
            )
        ]
    )
    events = await _one_turn(ClaudeCliCognition(spawn=cli))

    init = events[-1].result.evals.get("cli_init")
    assert init == {"model": "claude-opus-4-5", "capabilities": ["interrupt_cancel_queued_v1"]}, init
    assert "mcp_server_errors" not in init
    assert "plugin_errors" not in init


@pytest.mark.asyncio
async def test_a_provider_retry_is_announced_live_and_recorded_in_the_result() -> None:
    """A CLI retrying the provider is why a run went quiet for thirty seconds.

    Nothing else in the stream or the result explains the pause: the usage is
    one call's worth, the answer is correct, and the duration is four times what
    it should be. Surfaced as a ``step`` so an operator watching live sees it,
    and kept on the result so the one who reads the trace afterwards can too.
    """
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    _init(),
                    {
                        "type": "system",
                        "subtype": "api_retry",
                        "session_id": "sess-1",
                        "attempt": 2,
                        "max_retries": 5,
                        "error": "overloaded_error",
                    },
                    _assistant("eventually"),
                    _result(),
                ]
            )
        ]
    )
    events = await _one_turn(ClaudeCliCognition(spawn=cli))

    steps = [ev.text for ev in events if ev.type == "step"]
    assert steps == ["api_retry:overloaded_error (2/5)"], steps
    retries = events[-1].result.evals["api_retries"]
    assert [(r["attempt"], r["error"]) for r in retries] == [(2, "overloaded_error")], retries
    assert events[-1].result.output == "eventually", "a retry must not disturb the answer"
