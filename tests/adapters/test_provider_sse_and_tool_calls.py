"""SSE frame reassembly (`providers/base._stream_events`) and tool-call identity (`openai_compat`).

Both are places where the failure is SILENT: a dropped frame ends the stream as a complete answer,
and two tool calls sharing one id make one tool's result overwrite the other's. Nothing raises in
either case, so only a test that asserts the CONTENT catches them.
"""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from agentkit.adapters.llm.providers import OpenAICompatibleLLM
from agentkit.kernel.types import Message


def _run(coro):
    return asyncio.run(coro)


def _sse(body: str) -> OpenAICompatibleLLM:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    return OpenAICompatibleLLM(
        api_key="k",
        base_url="http://x",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _json_client(payload: dict) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        api_key="k",
        base_url="http://x",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )


async def _collect(stream):
    return [d async for d in stream]


def _stream(llm, **kw):
    return _run(_collect(llm.stream(messages=[Message("user", "q")], model="m", **kw)))


def _texts(deltas):
    return [d.text for d in deltas if d.text]


# ── SSE: a `data` field may span several lines ──────────────────────────────


def test_a_data_field_split_over_several_lines_is_reassembled() -> None:
    """Regression: each `data:` line was JSON-parsed on its own, so a payload the SSE spec
    permits to be split across lines failed to parse and hit `except JSONDecodeError: continue`
    — dropped whole, silently. Measured: `multi-line data field -> text deltas: []`, expected
    `['hello']`. The stream then ends as if the answer were complete."""
    deltas = _stream(
        _sse(
            "data: {\n"
            'data: "choices": [{"delta": {"content": "hello"}}]\n'
            "data: }\n"
            "\n"
            "data: [DONE]\n\n"
        )
    )
    assert _texts(deltas) == ["hello"]


def test_a_data_field_split_over_four_lines_with_following_events() -> None:
    """Edge: 3+ continuation lines, then further single-line events after the blank-line
    boundary. Reassembly must not swallow what comes next."""
    deltas = _stream(
        _sse(
            "data: {\n"
            'data:  "choices": [{"delta":\n'
            'data: {"content": "abc"}}]\n'
            "data: }\n"
            "\n"
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}\n\n'
            "data: [DONE]\n\n"
        )
    )
    assert _texts(deltas) == ["abc"]
    assert deltas[-1].finish_reason == "stop"
    assert deltas[-1].usage.input_tokens == 10


def test_a_blank_line_between_two_split_payloads_separates_them() -> None:
    """Edge: two multi-line events back to back. The blank line is the only boundary there is,
    so both must decode — not one merged blob."""
    deltas = _stream(
        _sse(
            "data: {\n"
            'data: "choices": [{"delta": {"content": "one"}}]}\n'
            "\n"
            "data: {\n"
            'data: "choices": [{"delta": {"content": "two"}}]}\n'
            "\n"
            "data: [DONE]\n\n"
        )
    )
    assert _texts(deltas) == ["one", "two"]


def test_a_server_that_omits_the_blank_line_between_events_still_streams() -> None:
    """POSITIVE CONTROL. Plenty of OpenAI-compatible servers send `data:` lines with no blank
    line between them; that worked before this change and must keep working. A "fix" that only
    dispatched on the spec's blank-line boundary drops every one of these frames."""
    deltas = _stream(
        _sse(
            'data: {"choices": [{"delta": {"content": "a"}}]}\n'
            'data: {"choices": [{"delta": {"content": "b"}}]}\n'
            "data: [DONE]\n"
        )
    )
    assert _texts(deltas) == ["a", "b"]


def test_a_final_event_without_a_trailing_blank_line_is_not_lost() -> None:
    """Edge: the connection ends right after the last payload — no blank line, no `[DONE]`."""
    deltas = _stream(_sse('data: {"choices": [{"delta": {"content": "last"}}]}'))
    assert _texts(deltas) == ["last"]


def test_comments_and_non_data_fields_are_ignored() -> None:
    """Positive control: `: keep-alive` comments and `event:`/`id:`/`retry:` fields are part of
    every real SSE stream. They must be skipped, not accumulated into the payload."""
    deltas = _stream(
        _sse(
            ": keep-alive\n"
            "event: message\n"
            "id: 7\n"
            'data: {"choices": [{"delta": {"content": "c"}}]}\n'
            "\n"
            "retry: 1000\n"
            "data: [DONE]\n\n"
        )
    )
    assert _texts(deltas) == ["c"]


def test_an_undecodable_frame_is_dropped_without_killing_the_stream() -> None:
    """Positive control: garbage that never becomes valid JSON must not poison what follows once
    a blank line closes it — the pre-existing tolerance."""
    deltas = _stream(
        _sse(
            "data: not json at all\n"
            "\n"
            'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
            "data: [DONE]\n\n"
        )
    )
    assert _texts(deltas) == ["ok"]


def test_done_still_terminates_the_stream_early() -> None:
    """Positive control: `[DONE]` stops iteration. Anything after it is not the model's answer."""
    deltas = _stream(
        _sse(
            'data: {"choices": [{"delta": {"content": "in"}}]}\n\n'
            "data: [DONE]\n\n"
            'data: {"choices": [{"delta": {"content": "after"}}]}\n\n'
        )
    )
    assert _texts(deltas) == ["in"]


# ── tool-call identity: non-streamed ────────────────────────────────────────


def _tool_payload(*names, ids=None):
    calls = []
    for i, name in enumerate(names):
        call = {"type": "function", "function": {"name": name, "arguments": json.dumps({"i": i})}}
        if ids is not None:
            call["id"] = ids[i]
        calls.append(call)
    return {
        "choices": [
            {"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": calls}}
        ]
    }


def test_parallel_calls_to_the_same_tool_get_distinct_ids() -> None:
    """Regression: with no provider id, `_parse_tool_calls` fell back to the tool NAME — exactly
    what `_frag_to_toolcall`'s own comment forbids for the streamed path. Measured:
    `non-streaming ids: ['search', 'search'] unique? False`. Two calls sharing one id means the
    `tool` result messages keyed by `tool_call_id` collide and one answer is silently lost."""
    res = _run(
        _json_client(_tool_payload("search", "search")).chat(
            messages=[Message("user", "q")], model="m"
        )
    )
    ids = [t.id for t in res.tool_calls]
    assert len(res.tool_calls) == 2
    assert len(set(ids)) == 2
    assert all(t.name == "search" for t in res.tool_calls)
    assert [dict(t.arguments) for t in res.tool_calls] == [{"i": 0}, {"i": 1}]


def test_parallel_calls_to_different_tools_also_get_distinct_ids() -> None:
    """Edge: distinct names USED to hide the bug (the name fallback happened to be unique).
    The id must not depend on that coincidence."""
    res = _run(
        _json_client(_tool_payload("search", "fetch")).chat(
            messages=[Message("user", "q")], model="m"
        )
    )
    assert len({t.id for t in res.tool_calls}) == 2
    assert [t.name for t in res.tool_calls] == ["search", "fetch"]


def test_a_provider_supplied_id_is_always_preserved() -> None:
    """POSITIVE CONTROL. The synthetic id is a FALLBACK. A "fix" that always numbered the calls
    would throw away the id the provider expects back on the tool result — the id that makes the
    round trip work at all."""
    res = _run(
        _json_client(_tool_payload("search", "search", ids=["call_abc", "call_xyz"])).chat(
            messages=[Message("user", "q")], model="m"
        )
    )
    assert [t.id for t in res.tool_calls] == ["call_abc", "call_xyz"]


# ── tool-call identity: streamed fragments ──────────────────────────────────


def _frag(**fields) -> str:
    return json.dumps({"choices": [{"delta": {"tool_calls": [fields]}}]})


def test_streamed_fragments_without_an_index_do_not_merge() -> None:
    """Regression: `tc.get("index", 0)` filed every index-less fragment under slot 0, so two
    distinct parallel calls collapsed into one. Measured: `[('a2', 'search')]` — a single call
    whose id came from the second fragment and whose arguments were the two concatenated."""
    deltas = _stream(
        _sse(
            f"data: {_frag(id='a1', function={'name': 'search', 'arguments': '{\"q\": 1}'})}\n\n"
            f"data: {_frag(id='a2', function={'name': 'search', 'arguments': '{\"q\": 2}'})}\n\n"
            "data: [DONE]\n\n"
        )
    )
    calls = deltas[-1].tool_calls
    assert [(t.id, t.name) for t in calls] == [("a1", "search"), ("a2", "search")]
    assert [dict(t.arguments) for t in calls] == [{"q": 1}, {"q": 2}]


def test_an_index_less_argument_fragment_continues_the_open_call() -> None:
    """Edge: a continuation fragment carries neither id nor name — only argument text. With no
    `index` to key on, it belongs to the call most recently opened, not to a new one."""
    deltas = _stream(
        _sse(
            f"data: {_frag(id='a1', function={'name': 'search', 'arguments': '{\"q\":'})}\n\n"
            f"data: {_frag(function={'arguments': ' \"x\"}'})}\n\n"
            "data: [DONE]\n\n"
        )
    )
    calls = deltas[-1].tool_calls
    assert len(calls) == 1
    assert calls[0].id == "a1" and dict(calls[0].arguments) == {"q": "x"}


def test_indexed_fragments_are_still_assembled_by_index() -> None:
    """POSITIVE CONTROL. `index` is the spec's mechanism and stays authoritative — a "fix" that
    started a new call per fragment would shred every ordinary streamed tool call into pieces."""
    deltas = _stream(
        _sse(
            f"data: {_frag(index=0, id='c1', function={'name': 'fetch', 'arguments': '{\"url\"'})}\n\n"
            f"data: {_frag(index=1, id='c2', function={'name': 'fetch', 'arguments': '{\"url\"'})}\n\n"
            f"data: {_frag(index=0, function={'arguments': ': \"a\"}'})}\n\n"
            f"data: {_frag(index=1, function={'arguments': ': \"b\"}'})}\n\n"
            "data: [DONE]\n\n"
        )
    )
    calls = deltas[-1].tool_calls
    assert [(t.id, dict(t.arguments)) for t in calls] == [
        ("c1", {"url": "a"}),
        ("c2", {"url": "b"}),
    ]


def test_streamed_calls_with_no_id_at_all_fall_back_to_the_index() -> None:
    """Positive control for the pre-existing rule this fix aligns the non-streamed path with."""
    deltas = _stream(
        _sse(
            f"data: {_frag(index=0, function={'name': 'search', 'arguments': '{}'})}\n\n"
            f"data: {_frag(index=1, function={'name': 'search', 'arguments': '{}'})}\n\n"
            "data: [DONE]\n\n"
        )
    )
    ids = [t.id for t in deltas[-1].tool_calls]
    assert len(ids) == 2 and len(set(ids)) == 2
