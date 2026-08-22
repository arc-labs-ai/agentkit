"""Token-level streaming, and the diagnostics a running service needs.

Without `--include-partial-messages` the CLI emits one `assistant` message per
COMPLETED content block, so `message_delta` arrives per paragraph — fine for a
backend, wrong for a UI with a cursor in it. The class docstring called these
"token chunks", which they were not.

Turning partial streaming on creates a duplication hazard the parser has to
handle: the same text arrives twice, once as `stream_event` deltas and again in
the completed `assistant` message. The rule is that deltas are for EMITTING and
the completed message is for ACCUMULATING, so a consumer sees each token once
and `AgentResult.output` is written once.

The stream also carries operational facts that were being dropped on the floor:
which MCP servers failed to load (the CLI docs recommend gating CI on exactly
this), and whether the provider was retried — the only thing in the payload
that explains a run which took forty seconds to make one API call.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.context import WorkingContext
from agentkit.kernel.types import StreamEvent
from agentkit.testing.fakes.ctx import FakeCtx
from tests.agents.cognition.test_claude_cli import _FakeProcess, _line

real_cli = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


def _delta(kind: str, **fields: Any) -> bytes:
    return _line({"type": "stream_event", "event": {"type": "content_block_delta",
                                                    "delta": {"type": kind, **fields}}})


def _result() -> bytes:
    return _line(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "s",
            "duration_ms": 1,
            "total_cost_usd": 0.0,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def _run(cog: ClaudeCliCognition, lines: list[bytes]) -> tuple[list[StreamEvent], tuple[str, ...]]:
    proc = _FakeProcess(stdout_lines=lines)
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ) as spawn:
        agent = Agent(name="x", cognition=cog)

        async def _go() -> list[StreamEvent]:
            return [ev async for ev in cog.drive(agent, "t", FakeCtx(), WorkingContext())]

        events = asyncio.run(_go())
    return events, tuple(spawn.await_args.args)


_TOKENS = ["Hel", "lo ", "world."]
_PARTIAL_STREAM = [
    _line({"type": "system", "subtype": "init", "session_id": "s", "model": "claude-x"}),
    *[_delta("text_delta", text=t) for t in _TOKENS],
    _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello world."}]}}),
    _result(),
]


# ── 1. the flag ─────────────────────────────────────────────────────────────


def test_partial_streaming_is_opt_in() -> None:
    """It costs one payload per provider SSE event, so it is not the default —
    but a UI that renders tokens needs it."""
    _, argv = _run(ClaudeCliCognition(partial_messages=True), _PARTIAL_STREAM)
    assert "--include-partial-messages" in argv
    _, argv = _run(ClaudeCliCognition(), [_result()])
    assert "--include-partial-messages" not in argv


# ── 2. tokens out, text accumulated once ────────────────────────────────────


def test_tokens_are_emitted_and_the_text_is_accumulated_once() -> None:
    """THE hazard. The same text arrives twice — as deltas and as the completed
    message — so a naive parser either shows every sentence twice or writes the
    output twice."""
    events, _ = _run(ClaudeCliCognition(partial_messages=True), _PARTIAL_STREAM)

    chunks = [e.text for e in events if e.type == "message_delta"]
    assert chunks == _TOKENS, "the consumer must see tokens, and each exactly once"
    assert events[-1].result.output == "Hello world."


def test_without_partial_streaming_the_block_is_the_event() -> None:
    """The unchanged path: one ``message_delta`` per completed block."""
    events, _ = _run(
        ClaudeCliCognition(),
        [
            _line(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Hello world."}]},
                }
            ),
            _result(),
        ],
    )
    chunks = [e.text for e in events if e.type == "message_delta"]
    assert chunks == ["Hello world."]
    assert events[-1].result.output == "Hello world."


def test_thinking_streams_but_never_enters_the_output() -> None:
    """Reasoning is surfaced live and folded into ``evals["thinking"]`` — it is
    not the answer, and appending it to ``output`` would corrupt every caller
    that renders the result."""
    events, _ = _run(
        ClaudeCliCognition(partial_messages=True),
        [
            _delta("thinking_delta", thinking="hmm..."),
            _line(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "thinking", "thinking": "hmm..."}]},
                }
            ),
            _delta("text_delta", text="42"),
            _line({"type": "assistant", "message": {"content": [{"type": "text", "text": "42"}]}}),
            _result(),
        ],
    )
    assert [e.text for e in events if e.type == "message_delta"] == ["hmm...", "42"]
    result = events[-1].result
    assert result.output == "42"
    assert result.evals["thinking"] == "hmm..."


def test_non_text_deltas_are_ignored() -> None:
    """``signature_delta`` carries a cryptographic blob and ``input_json_delta``
    carries tool-argument fragments. Rendering either as assistant text would
    put base64 in a chat window."""
    events, _ = _run(
        ClaudeCliCognition(partial_messages=True),
        [
            _delta("signature_delta", signature="EtIDCrIB..."),
            _delta("input_json_delta", partial_json='{"path":'),
            _line({"type": "stream_event", "event": {"type": "message_stop"}}),
            _result(),
        ],
    )
    assert [e.type for e in events] == ["final"]


# ── 3. diagnostics ──────────────────────────────────────────────────────────


def test_startup_metadata_is_surfaced() -> None:
    """Which model actually ran, and which MCP servers connected."""
    events, _ = _run(
        ClaudeCliCognition(),
        [
            _line(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "s",
                    "model": "claude-x",
                    "mcp_servers": [{"name": "db", "status": "connected"}],
                    "claude_code_version": "2.1.238",
                }
            ),
            _result(),
        ],
    )
    init = events[-1].result.evals["cli_init"]
    assert init["model"] == "claude-x"
    assert init["mcp_servers"] == [{"name": "db", "status": "connected"}]


def test_a_skipped_mcp_server_is_visible() -> None:
    """The CLI validates each ``--mcp-config`` entry, skips the invalid ones and
    RUNS ANYWAY, exiting cleanly. The error key is omitted when empty, so its
    presence is the signal — which is what the docs recommend gating CI on, and
    it was previously discarded with the rest of the init payload."""
    events, _ = _run(
        ClaudeCliCognition(),
        [
            _line(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "s",
                    "mcp_servers": [],
                    "mcp_server_errors": [
                        {"name": "db", "type": "url_missing_type", "message": "no type"}
                    ],
                }
            ),
            _result(),
        ],
    )
    assert events[-1].result.evals["cli_init"]["mcp_server_errors"][0]["name"] == "db"


def test_a_clean_startup_carries_no_error_keys() -> None:
    """Presence is the signal, so an empty list must not be manufactured."""
    events, _ = _run(
        ClaudeCliCognition(),
        [_line({"type": "system", "subtype": "init", "session_id": "s", "model": "m"}), _result()],
    )
    assert "mcp_server_errors" not in events[-1].result.evals["cli_init"]
    assert "plugin_errors" not in events[-1].result.evals["cli_init"]


def test_api_retries_are_visible_live_and_in_the_result() -> None:
    """A run that goes quiet for forty seconds is explained by this and nothing
    else in the payload."""
    events, _ = _run(
        ClaudeCliCognition(),
        [
            _line(
                {
                    "type": "system",
                    "subtype": "api_retry",
                    "session_id": "s",
                    "attempt": 2,
                    "max_retries": 5,
                    "retry_delay_ms": 1000,
                    "error": "overloaded",
                }
            ),
            _result(),
        ],
    )
    steps = [e for e in events if e.type == "step"]
    assert steps and "overloaded" in steps[0].text and "2/5" in steps[0].text
    retries = events[-1].result.evals["api_retries"]
    assert len(retries) == 1 and retries[0]["error"] == "overloaded"


def test_an_unknown_system_subtype_is_ignored() -> None:
    """The CLI adds event kinds over time (``status``, ``thinking_tokens``,
    ``plugin_install``). Forward-compat is "don't crash and don't invent"."""
    events, _ = _run(
        ClaudeCliCognition(),
        [
            _line({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 12}),
            _line({"type": "system", "subtype": "brand_new_thing"}),
            _result(),
        ],
    )
    assert [e.type for e in events] == ["final"]
    assert "cli_init" not in events[-1].result.evals


# ── 4. against the real binary ──────────────────────────────────────────────


@real_cli
def test_the_real_cli_streams_tokens_without_duplicating_them() -> None:
    """Only the binary can tell us the delta shape is still what we parse — and
    duplication is the failure mode a mock is least likely to catch, because a
    hand-written stream has whatever consistency the author assumed."""
    cog = ClaudeCliCognition(
        model="claude-haiku-4-5-20251001",
        tools=("",),
        permission_mode="dontAsk",
        max_turns=1,
        partial_messages=True,
    )
    agent = Agent(name="x", prompt="Be terse. No preamble.", cognition=cog)

    async def _go() -> list[StreamEvent]:
        return [
            ev
            async for ev in cog.drive(
                agent, "Count from 1 to 8, space separated.", FakeCtx(), WorkingContext()
            )
        ]

    events = asyncio.run(_go())
    result = events[-1].result

    assert [e.text for e in events if e.type == "message_delta"], "no token deltas arrived"
    assert result.output.count("1 2 3") <= 1, f"text was duplicated: {result.output!r}"
    assert "1 2 3 4 5 6 7 8" in result.output
