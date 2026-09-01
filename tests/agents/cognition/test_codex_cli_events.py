"""The two event vocabularies, the delta arithmetic, and the token split.

Three things in this cognition are pure translation, and all three are the kind
that pass every casual assertion while being wrong:

**Two vocabularies.** ``codex exec --json`` has shipped both ``{"type":
"item.completed", ...}`` thread events and the older ``{"id": …, "msg":
{"type": "agent_message", …}}`` shape, and the CLI is something users install
themselves — a service does not get to pick their version. Supporting one and
ignoring the other is the worst available outcome, because the unknown-payload
rule is "don't crash": an older CLI would produce a run that exits 0, emits no
events, returns an empty answer and reports no stop reason. Silent, plausible,
wrong.

**Delta arithmetic.** Both vocabularies send an item's text more than once —
``item.updated`` then ``item.completed``, or a run of ``agent_message_delta``
then the whole ``agent_message``. A consumer must see every character exactly
once and the fold must count it exactly once. Emitting the full text twice
shows a UI every sentence twice; folding it twice doubles the answer.

**The token split.** Codex counts input tokens INCLUSIVE of the cached prefix
and reports the cached part separately; agentkit's ``Usage.input_tokens`` is the
fresh input and ``cache_read_tokens`` is the discount. A Codex session is mostly
cache, so getting this backwards is not a rounding difference — it is a cost
estimate that double-counts nearly the whole prompt on every turn.
"""

from __future__ import annotations

import pytest

from agentkit.agents.cognition import CodexCliCognition
from agentkit.agents.cognition.codex_cli import _split_input_tokens
from agentkit.testing.fakes import FakeCodexCli, codex_turn
from tests.agents.cognition.test_codex_cli import drive, final_of, shape

# ─────────────────────────────────────────────────────────────────────────────
# 1. incremental text: emit each character once, fold it once
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_item_updated_streams_suffixes_and_the_completion_does_not_repeat_them() -> None:
    """The case a UI with a cursor in it depends on. Three updates and a
    completion carrying the cumulative text must produce four disjoint deltas
    whose concatenation is the answer — not four cumulative ones, which is what
    re-emitting ``item.completed`` in full would give."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.started", "item": {"id": "m", "type": "agent_message", "text": ""}},
            {"type": "item.updated", "item": {"id": "m", "type": "agent_message", "text": "The "}},
            {"type": "item.updated", "item": {"id": "m", "type": "agent_message", "text": "The magic "}},
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "The magic number is 137."}},
            {"type": "turn.completed", "usage": {"input_tokens": 9, "output_tokens": 6}},
        ]
    )
    events = await drive(CodexCliCognition(spawn=cli))
    deltas = [e.text for e in events if e.type == "message_delta"]

    assert deltas == ["The ", "magic ", "number is 137."]
    assert "".join(deltas) == "The magic number is 137."
    assert final_of(events).output == "The magic number is 137."


@pytest.mark.asyncio
async def test_a_completion_with_no_prior_updates_emits_the_whole_text_once() -> None:
    """The ordinary case, and the one the suffix logic must not break: with
    nothing emitted yet the suffix IS the whole string."""
    cli = FakeCodexCli.script(codex_turn(text="all at once", usage=(1, 0, 1)))
    events = await drive(CodexCliCognition(spawn=cli))
    assert [e.text for e in events if e.type == "message_delta"] == ["all at once"]


@pytest.mark.asyncio
async def test_a_shorter_re_send_for_the_same_item_does_not_wedge_the_stream() -> None:
    """Defensive, and cheap. A CLI that re-sent a SHORTER text for one item id
    would leave the offset past the end, and a naive slice would then return
    the empty string forever — the item, and every later update to it, silently
    gone. Resetting on a shrink means the worst case is a repeated delta rather
    than a lost answer."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.updated", "item": {"id": "m", "type": "agent_message", "text": "a long first draft"}},
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "short"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert [e.text for e in events if e.type == "message_delta"] == ["a long first draft", "short"]
    assert final_of(events).output == "short"


@pytest.mark.asyncio
async def test_reasoning_updates_stream_separately_from_the_answer() -> None:
    """Two items, two independent offsets. Sharing one counter would make the
    answer's first delta start at the reasoning's length."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.updated", "item": {"id": "r", "type": "reasoning", "text": "Think"}},
            {"type": "item.completed", "item": {"id": "r", "type": "reasoning", "text": "Thinking hard"}},
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "Done"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    events = await drive(CodexCliCognition(spawn=cli))
    result = final_of(events)
    assert [e.text for e in events if e.type == "message_delta"] == ["Think", "ing hard", "Done"]
    assert result.output == "Done"
    assert result.evals["thinking"] == "Thinking hard"


@pytest.mark.asyncio
async def test_an_item_that_only_ever_updates_never_reaches_the_answer() -> None:
    """Folding happens on ``completed`` alone. An item the CLI abandoned
    mid-stream contributed live deltas — a consumer saw them — but is not part
    of the authoritative answer, because nothing ever said it was finished."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.updated", "item": {"id": "m", "type": "agent_message", "text": "half-writ"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert [e.text for e in events if e.type == "message_delta"] == ["half-writ"]
    assert final_of(events).output == ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. the token split
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("total", "cached", "expected"),
    [
        (1200, 1000, (200, 1000)),
        (1200, 0, (1200, 0)),
        (1000, 1000, (0, 1000)),
        # Nonsense the two counters can disagree into. Clamped rather than
        # trusted: a negative ``input_tokens`` flows straight into a meter and
        # makes a budget go UP.
        (100, 500, (0, 500)),
        (100, -5, (100, 0)),
    ],
)
def test_the_cached_prefix_is_subtracted_from_the_input_count(
    total: int, cached: int, expected: tuple[int, int]
) -> None:
    assert _split_input_tokens(total, cached) == expected


@pytest.mark.asyncio
async def test_a_turns_usage_reaches_the_result_in_agentkits_convention() -> None:
    """End to end, because the unit test above proves the arithmetic and this
    one proves it is actually wired to the payload field."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(24763, 24448, 122)))
    usage = final_of(await drive(CodexCliCognition(spawn=cli))).usage

    assert usage.input_tokens == 315
    assert usage.cache_read_tokens == 24448
    assert usage.output_tokens == 122
    # ``total_tokens`` is fresh input + output by definition, so a cached-heavy
    # turn reads as small — which is what it costs.
    assert usage.total_tokens == 437
    # No cache-creation figure is reported by the CLI, and deriving one from the
    # difference would be a number with no source.
    assert usage.cache_write_tokens == 0


@pytest.mark.asyncio
async def test_a_turn_completed_with_no_usage_block_is_zero_not_a_crash() -> None:
    cli = FakeCodexCli.script(codex_turn(text="x"))
    usage = final_of(await drive(CodexCliCognition(spawn=cli))).usage
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. the legacy vocabulary
# ─────────────────────────────────────────────────────────────────────────────


def _legacy(*msgs: dict) -> list[dict]:
    """Wrap each ``msg`` body in the legacy envelope, numbering the events."""
    return [{"id": str(i), "msg": m} for i, m in enumerate(msgs)]


@pytest.mark.asyncio
async def test_a_legacy_stream_produces_the_same_result_as_a_thread_stream() -> None:
    """The point of parsing both: a caller's code, assertions and dashboards do
    not change when the operator upgrades or downgrades their ``codex``."""
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "legacy-thread", "model": "gpt-5-codex"},
            {"type": "task_started"},
            {"type": "agent_reasoning", "text": "Counting files"},
            {"type": "agent_message", "message": "Nine files."},
            {
                "type": "token_count",
                "info": {"total_token_usage": {"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 40}},
            },
            {"type": "task_complete", "last_agent_message": "Nine files."},
        )
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))

    assert result.output == "Nine files."
    assert result.evals["thinking"] == "Counting files"
    assert result.evals["session_id"] == "legacy-thread"
    # ``session_configured`` names the model, which ``thread.started`` does not
    # — so on this vocabulary the cost is priced against what actually ran.
    assert result.evals["cli_model"] == "gpt-5-codex"
    assert result.usage.input_tokens == 200
    assert result.usage.cache_read_tokens == 1000
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
async def test_legacy_deltas_are_not_repeated_by_the_message_that_follows() -> None:
    """The legacy stream reaches the same problem from the other direction:
    deltas first, then the whole message. The message must emit only the part
    the deltas did not."""
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "agent_message_delta", "delta": "Nine "},
            {"type": "agent_message_delta", "delta": "files"},
            {"type": "agent_message", "message": "Nine files."},
            {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 5, "output_tokens": 2}}},
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert [e.text for e in events if e.type == "message_delta"] == ["Nine ", "files", "."]
    assert final_of(events).output == "Nine files."


@pytest.mark.asyncio
async def test_two_legacy_messages_in_one_turn_each_start_their_own_delta_run() -> None:
    """The reset. Without it the second message's offset is the first's length
    and its text is silently truncated."""
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "agent_message_delta", "delta": "one"},
            {"type": "agent_message", "message": "one"},
            {"type": "agent_message_delta", "delta": "tw"},
            {"type": "agent_message", "message": "two"},
            {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 5, "output_tokens": 2}}},
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert [e.text for e in events if e.type == "message_delta"] == ["one", "tw", "o"]
    assert final_of(events).output == "onetwo"


@pytest.mark.asyncio
async def test_a_legacy_shell_command_pairs_begin_with_end() -> None:
    """Keyed on ``call_id``, not the envelope's ``id``. The envelope id is a
    per-event sequence number, so keying on it would make every begin/end pair
    two unrelated tool calls and no results at all."""
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "exec_command_begin", "call_id": "c1", "command": ["bash", "-lc", "ls"], "cwd": "/tmp"},
            {"type": "exec_command_end", "call_id": "c1", "stdout": "docs\n", "stderr": "", "exit_code": 0},
            {"type": "agent_message", "message": "one dir"},
            {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 5, "output_tokens": 2}}},
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    calls = [e for e in events if e.type == "tool_call"]
    results = [e for e in events if e.type == "tool_result"]

    assert len(calls) == 1
    assert calls[0].tool_call.id == "c1"
    assert calls[0].tool_call.name == "shell"
    assert calls[0].tool_call.arguments == {"command": ["bash", "-lc", "ls"], "cwd": "/tmp"}
    assert [r.tool_result for r in results] == ["docs\n"]


@pytest.mark.asyncio
async def test_a_legacy_failed_command_carries_stderr_and_its_exit_code() -> None:
    """The legacy stream splits stdout and stderr, and both are the command's
    output as far as the model is concerned."""
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "exec_command_begin", "call_id": "c1", "command": ["false"]},
            {"type": "exec_command_end", "call_id": "c1", "stdout": "", "stderr": "boom\n", "exit_code": 2},
            {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1, "output_tokens": 1}}},
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert [e.tool_result for e in events if e.type == "tool_result"] == ["boom\n\n[exit 2]"]


@pytest.mark.asyncio
async def test_a_legacy_mcp_call_reads_the_invocation_table() -> None:
    """The legacy shape nests server/tool/arguments under ``invocation``, and
    the flat spelling also exists in some builds. Both are read, because a
    missing name would make the tool call anonymous in an audit trail."""
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {
                "type": "mcp_tool_call_begin",
                "call_id": "t1",
                "invocation": {"server": "engine", "tool": "search", "arguments": {"q": "x"}},
            },
            {
                "type": "mcp_tool_call_end",
                "call_id": "t1",
                "result": {"content": [{"type": "text", "text": "1 match"}]},
            },
            {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1, "output_tokens": 1}}},
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    call = next(e for e in events if e.type == "tool_call")
    assert call.tool_call.name == "mcp__engine__search"
    assert call.tool_call.arguments == {"q": "x"}
    assert [e.tool_result for e in events if e.type == "tool_result"] == ["1 match"]


@pytest.mark.asyncio
async def test_a_legacy_patch_apply_pairs_and_names_its_files() -> None:
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "patch_apply_begin", "call_id": "p1", "changes": {"src/b.py": {}, "docs/a.md": {}}},
            {"type": "patch_apply_end", "call_id": "p1", "stdout": "2 files changed", "success": True},
            {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1, "output_tokens": 1}}},
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    call = next(e for e in events if e.type == "tool_call")
    assert call.tool_call.name == "apply_patch"
    # Sorted, so the argv a test asserts on does not depend on dict ordering
    # from a JSON object.
    assert call.tool_call.arguments == {"changes": ["docs/a.md", "src/b.py"]}
    assert [e.tool_result for e in events if e.type == "tool_result"] == ["2 files changed"]


@pytest.mark.asyncio
async def test_a_legacy_error_is_a_reported_failure() -> None:
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "error", "message": "stream disconnected before completion"},
        )
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.partial is True
    assert result.evals["stop_reason"] == "cli_reported_error"
    assert result.evals["cli_errors"] == ["stream disconnected before completion"]


@pytest.mark.asyncio
async def test_a_stream_that_mixes_both_vocabularies_is_read_correctly() -> None:
    """The shape is detected per PAYLOAD rather than sniffed once from the first
    line, so a CLI that changed mid-stream — or a recording stitched from two
    versions — still reads. Sniffing once is the cheaper implementation and its
    failure is half a run silently dropped."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "mixed"},
            {"id": "0", "msg": {"type": "agent_message", "message": "from legacy. "}},
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "from thread."}},
            {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}},
        ]
    )
    result = final_of(await drive(CodexCliCognition(spawn=cli)))
    assert result.output == "from legacy. from thread."
    assert result.evals["session_id"] == "mixed"


@pytest.mark.asyncio
async def test_a_legacy_payload_whose_msg_is_not_a_dict_is_ignored() -> None:
    """``"msg"`` is how the two shapes are told apart, so a payload with a
    string there must fall through to the thread parser rather than crash the
    detection."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"id": "0", "msg": "not a dict"},
            {"type": "item.completed", "item": {"id": "m", "type": "agent_message", "text": "fine"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    assert final_of(await drive(CodexCliCognition(spawn=cli))).output == "fine"


@pytest.mark.asyncio
async def test_an_item_using_the_older_item_type_key_is_still_read() -> None:
    """Some builds spell the field ``item_type`` rather than ``type``. Reading
    only one drops every item silently, which is the same empty-answer failure
    as not parsing the vocabulary at all."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.completed", "item": {"id": "m", "item_type": "agent_message", "text": "read me"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    assert final_of(await drive(CodexCliCognition(spawn=cli))).output == "read me"


@pytest.mark.asyncio
async def test_an_item_payload_that_is_not_a_dict_is_ignored() -> None:
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.completed", "item": "nonsense"},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert shape(events) == [("final", "complete")]


# ─────────────────────────────────────────────────────────────────────────────
# 4. the defensive edges
#
# Small branches, each of which turns a malformed-but-plausible payload into a
# crash if it is missing. They are cheap to test and the alternative to testing
# them is finding out from a run that died on somebody else's data.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_todo_list_that_only_starts_emits_nothing() -> None:
    """A plan is progress, and "the model is about to have a plan" is not."""
    cli = FakeCodexCli.script(
        [
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.started", "item": {"id": "p", "type": "todo_list", "items": []}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
    )
    assert shape(await drive(CodexCliCognition(spawn=cli))) == [("final", "complete")]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ([{"path": "a.py", "kind": "add"}], "add a.py"),
        # A change with neither key still renders a line, so a consumer counting
        # them is not silently short.
        ([{}], "? ?"),
        ([{"path": "a.py", "kind": "add"}, "not a dict"], "add a.py"),
        ("not a list", ""),
        (None, ""),
    ],
)
def test_patch_changes_render_defensively(changes: object, expected: str) -> None:
    from agentkit.agents.cognition.codex_cli import _render_changes

    assert _render_changes(changes) == expected


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (None, ""),
        ("plain text", "plain text"),
        ({"content": [{"type": "text", "text": "a"}, {"type": "image"}]}, "a"),
        ([{"type": "text", "text": "a"}, "b"], "ab"),
        # A dict with no ``content`` is dumped rather than dropped: an MCP server
        # that returns a bare object has still told the model something, and an
        # empty tool_result would read as a tool that did nothing.
        ({"rows": 3}, '{"rows": 3}'),
        (42, "42"),
    ],
)
def test_mcp_results_flatten_defensively(result: object, expected: str) -> None:
    from agentkit.agents.cognition.codex_cli import _flatten_content

    assert _flatten_content(result) == expected


@pytest.mark.asyncio
async def test_empty_legacy_deltas_are_dropped_rather_than_emitted() -> None:
    """An empty ``message_delta`` is an event a UI renders as a no-op flicker,
    and it would also advance the suffix offset by zero — harmless, but the
    guard is what keeps the offset arithmetic honest."""
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "agent_message_delta", "delta": ""},
            {"type": "agent_reasoning_delta", "delta": ""},
            {"type": "agent_message", "message": "only this"},
            {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1}}},
        )
    )
    events = await drive(CodexCliCognition(spawn=cli))
    assert [e.text for e in events if e.type == "message_delta"] == ["only this"]


@pytest.mark.asyncio
async def test_an_unknown_legacy_message_type_is_ignored() -> None:
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "some_future_msg", "payload": 1},
            {"type": "turn_aborted"},
            {"type": "agent_message", "message": "still fine"},
            {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1}}},
        )
    )
    assert final_of(await drive(CodexCliCognition(spawn=cli))).output == "still fine"


@pytest.mark.asyncio
async def test_a_legacy_token_count_with_no_info_is_zero() -> None:
    cli = FakeCodexCli.script(
        _legacy(
            {"type": "session_configured", "session_id": "s"},
            {"type": "agent_message", "message": "x"},
            {"type": "token_count"},
        )
    )
    assert final_of(await drive(CodexCliCognition(spawn=cli))).usage.total_tokens == 0


@pytest.mark.asyncio
async def test_an_output_adapter_that_cannot_render_a_schema_is_not_a_run_ender() -> None:
    """A schema we cannot render means the run goes out untyped, which is a
    degradation. Raising here would make it an outage instead — and the agent
    was constructible, so the caller has no signal that anything is wrong with
    their type until this moment."""

    class _Broken:
        def json_schema(self) -> dict:
            raise RuntimeError("cannot introspect this type")

        def validate(self, value: object) -> object:
            return value

    class _Agent:
        prompt = None
        _output_adapter = _Broken()

    cli = FakeCodexCli.script(codex_turn(text="prose", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    events = [
        ev
        async for ev in cog.drive(_Agent(), "t", None, None)  # type: ignore[arg-type]
    ]
    assert events[-1].result.output == "prose"
    assert "--output-schema" not in cli.invocations[-1].argv
