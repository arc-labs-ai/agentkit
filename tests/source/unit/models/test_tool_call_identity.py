"""One rule for a tool call's id, across every provider that builds one.

WHY THIS FILE EXISTS
--------------------
A tool call's `id` is what pairs a RESULT back to its call, and every transport
keys on it: `tool_call_id` on an OpenAI `tool` message, `tool_use_id` on an
Anthropic `tool_result` block, the per-item dedupe in the codex driver. So two
calls sharing an id is not cosmetic — one tool's answer silently overwrites the
other's and the model is handed a result it never asked for, attributed to a
call it did not make. Nothing raises, and the model's next turn reasons from it.

Six places build a `ToolCall`, and they had four different rules:

    openai_compat, both paths   position     correct, and documented
    _mapping._extract_tool_calls  the NAME   collides for parallel calls
    anthropic, both paths       ""           collides for every id-less call
    codex driver                item id, ""  collided AND then deduped the
                                             collision away, dropping calls

`tool_call_id` is now the single rule. These tests are per-provider on purpose:
a shared helper only helps if every caller actually routes through it, and the
way that regresses is a new provider spelling its own fallback — which is
exactly how this started.

The hostile case throughout is TWO PARALLEL CALLS TO THE SAME TOOL with no
provider id. It is the one a name-based fallback cannot survive, and it is not
exotic: it is what a model does when asked to look up two things at once.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from agentkit.models.adapters._mapping import _extract_tool_calls, tool_call_id

_TWO_PARALLEL = "two parallel calls to one tool must not share an id"


# ── the rule itself ──────────────────────────────────────────────────────────


def test_an_absent_id_falls_back_to_the_position() -> None:
    """Position is the only fallback that is unique per call without inventing
    information. The index is already in hand at every call site."""
    assert tool_call_id(None, 0) == "call_0"
    assert tool_call_id("", 3) == "call_3"
    assert tool_call_id("   ", 7) == "call_7", "whitespace is not an id"


def test_a_provider_supplied_id_is_kept_verbatim() -> None:
    """The fallback must never override a real id — results are keyed on the
    provider's spelling and the provider is the one echoing it back."""
    assert tool_call_id("toolu_01ABC", 4) == "toolu_01ABC"


def test_the_fallback_is_never_the_tool_name() -> None:
    """The regression this whole file guards. A name-based fallback reads as
    sensible and is the one choice that is guaranteed to collide exactly when
    it matters."""
    first, second = tool_call_id(None, 0), tool_call_id(None, 1)
    assert first != second, _TWO_PARALLEL
    assert "search" not in (first, second)


# ── the duck-typed OpenAI-shape adapter ──────────────────────────────────────


def test_the_duck_typed_adapter_gives_parallel_calls_distinct_ids() -> None:
    """`_extract_tool_calls` fell back to the tool name. Measured, two id-less
    calls to `search` both came back as `"search"`, so the second result
    overwrote the first in any map keyed by id."""
    raw = {
        "tool_calls": [
            {"function": {"name": "search", "arguments": '{"q": "a"}'}},
            {"function": {"name": "search", "arguments": '{"q": "b"}'}},
        ]
    }
    calls = _extract_tool_calls(raw)

    assert [c.name for c in calls] == ["search", "search"]
    assert len({c.id for c in calls}) == 2, _TWO_PARALLEL
    assert all(c.id for c in calls), "an empty id collides with every other empty id"
    assert [c.arguments["q"] for c in calls] == ["a", "b"], "the pairing must survive"


# ── the Anthropic adapter, both paths ────────────────────────────────────────

httpx = pytest.importorskip("httpx")

from agentkit.models.providers.frontiers import AnthropicLLM  # noqa: E402 — needs the extra


def _anthropic(handler: Any) -> AnthropicLLM:
    return AnthropicLLM(
        api_key="k",
        base_url="http://x",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_anthropic_gives_parallel_tool_use_blocks_distinct_ids() -> None:
    """Anthropic does send ids, so this is the defensive path — and it was the
    worst kind of defensive: `b.get("id", "")` produced `""` for every block,
    which is one id shared by all of them rather than a missing one."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-4-6",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "name": "search", "input": {"q": "a"}},
                    {"type": "tool_use", "name": "search", "input": {"q": "b"}},
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    result = asyncio.run(_anthropic(handler).chat(messages=[], model="claude-sonnet-4-6"))

    assert len(result.tool_calls) == 2
    assert len({c.id for c in result.tool_calls}) == 2, _TWO_PARALLEL
    assert all(c.id for c in result.tool_calls)
    assert [c.arguments["q"] for c in result.tool_calls] == ["a", "b"]


def test_anthropic_streaming_gives_parallel_tool_use_blocks_distinct_ids() -> None:
    """The streamed path assembles blocks in a dict keyed by the stream's own
    content index, so the position was already in hand — it just was not used."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": "search"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"q":"a"}'}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "search"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"q":"b"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(e) + "\n\n" for e in events),
        )

    async def collect() -> list[Any]:
        return [d async for d in _anthropic(handler).stream(messages=[], model="claude-sonnet-4-6")]

    calls = [c for d in asyncio.run(collect()) for c in (d.tool_calls or ())]

    assert len(calls) == 2
    assert len({c.id for c in calls}) == 2, _TWO_PARALLEL
    assert all(c.id for c in calls)
    assert [c.arguments["q"] for c in calls] == ["a", "b"]


# ── the two CLI drivers ──────────────────────────────────────────────────────

from agentkit.core.testing.fakes import CliRun, FakeClaudeCli, FakeCodexCli  # noqa: E402
from agentkit.models.providers.cli.claude.driver import ClaudeCliCognition  # noqa: E402
from agentkit.models.providers.cli.codex.driver import CodexCliCognition  # noqa: E402

_AGENT = SimpleNamespace(name="a", prompt="", output=None, _output_adapter=None)


def _tool_calls(events: list[Any]) -> list[Any]:
    return [e.tool_call for e in events if e.type == "tool_call"]


@pytest.mark.asyncio
async def test_the_claude_driver_gives_parallel_tool_use_blocks_distinct_ids() -> None:
    """Claude sends ids, so this is the defensive branch — and `str(block.get("id")
    or "")` made every id-less block share the one empty id rather than have none.
    A consumer pairing results by id would attribute both answers to one call."""
    cli = FakeClaudeCli(
        [
            CliRun.of(
                [
                    {"type": "system", "subtype": "init", "session_id": "s1"},
                    {
                        "type": "assistant",
                        "message": {
                            "id": "m1",
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "name": "Read", "input": {"path": "a"}},
                                {"type": "tool_use", "name": "Read", "input": {"path": "b"}},
                            ],
                        },
                    },
                    {"type": "result", "subtype": "success", "result": "done", "session_id": "s1"},
                ]
            )
        ]
    )
    events = [ev async for ev in ClaudeCliCognition(spawn=cli).drive(_AGENT, "go", None, None)]

    calls = _tool_calls(events)
    assert len(calls) == 2, [c.name for c in calls]
    assert len({c.id for c in calls}) == 2, _TWO_PARALLEL
    assert all(c.id for c in calls)
    assert [c.arguments["path"] for c in calls] == ["a", "b"], "the pairing must survive"


@pytest.mark.asyncio
async def test_the_codex_driver_does_not_drop_a_second_id_less_tool_item() -> None:
    """Codex's was the worst of the four, because the collision then went through
    a dedupe. Every id-less item hashed to the same key `"call:"`, so the FIRST
    emitted a `tool_call` and every later one was silently discarded — two
    parallel shell commands reported to the caller as one, with one result.

    Losing a call outright is worse than mislabelling it: an observer counting
    tool calls undercounts, and a consumer pairing results has a result with no
    call to attach it to.
    """
    cli = FakeCodexCli.script(
        (
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "ls a", "exit_code": 0}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "ls b", "exit_code": 0}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}},
        )
    )
    events = [ev async for ev in CodexCliCognition(spawn=cli).drive(_AGENT, "go", None, None)]

    calls = _tool_calls(events)
    assert len(calls) == 2, f"a tool call was dropped, not just mislabelled: {[c.name for c in calls]}"
    assert len({c.id for c in calls}) == 2, _TWO_PARALLEL
    assert all(c.id for c in calls)
    assert [c.arguments["command"] for c in calls] == ["ls a", "ls b"]


# ── the result side, across the CLI/MCP bridge ───────────────────────────────


# ── still owed: the result side, offline ─────────────────────────────────────
#
# `tool_call_id` is now attached to TOOL_RESULT as well as TOOL_CALL
# (`cli/calls.py`), which fixes a Claude tool result arriving identified by
# nothing — observed live as `[('', None), ('', None), ('', None)]` against
# claude 2.1.302, because the payload carries only `tool_use_id` and the driver
# correctly leaves `name` blank.
#
# That fix is asserted by `tests/source/live/test_live_execution_contract.py`
# and NOT yet offline. The reason is recorded so the next attempt starts ahead
# of where this one stopped: CLI tool events reach a caller only on the
# tools-SERVING path (a granted `ToolSet` bridged over MCP), so a bare
# `runtime.ask()` against a replayed stream emits no TOOL_CALL/TOOL_RESULT at
# all — the fake replays, the answer is right, and the tool events simply are
# not part of that path. An offline regression needs the MCP-bridge harness, so
# it belongs beside the other bridge tests rather than here.
