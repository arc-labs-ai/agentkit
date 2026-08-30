"""``ToolMemory`` — adapt a ``FunctionTool`` (or any Tool-Protocol object)
into a ``MemorySource``.

These tests pin:

1. Wrapping a tool that returns a list of strings yields ``MemoryItem``s
   whose ``content`` is the string and whose ``source`` is the
   ToolMemory's name.
2. A custom ``result_to_items`` parser is honoured (overrides default).
3. The default parser handles the four documented shapes: dict-with-
   ``"results"``, ``list[dict]``, ``list[str]``, and a single string.
4. Writes are no-ops (tools are read-only as memory).
5. ``where`` keys merge into the tool's args so a filter-aware tool can
   consume them; ``query_arg`` is configurable.
6. The result is capped at ``k``.
"""

from __future__ import annotations

import asyncio

from agentkit.memory import MemoryItem, ToolMemory, default_parse
from agentkit.memory.tool import default_parse as _direct_default_parse
from agentkit.testing import make_test_ctx
from agentkit.tools import FunctionTool


def _run(coro):
    return asyncio.run(coro)


# ── helpers ───────────────────────────────────────────────────────────────


def _make_tool(fn):
    """Wrap a Python function as a FunctionTool with a sufficient docstring."""
    return FunctionTool.from_callable(fn, side_effecting=False)


# ── tests ─────────────────────────────────────────────────────────────────


def test_wraps_list_of_strings_into_memory_items():
    """A tool returning ``list[str]`` becomes a list of MemoryItem.content."""

    def lookup(query: str, limit: int = 5) -> list[str]:
        """Toy lookup that returns one snippet per limit, prefixed with the query."""
        del query
        return [f"hit-{i}" for i in range(limit)]

    mem = ToolMemory(tool=_make_tool(lookup), name="lookup")
    ctx = make_test_ctx()
    items = _run(mem.query("dogs", k=3, ctx=ctx))

    assert [i.content for i in items] == ["hit-0", "hit-1", "hit-2"]
    assert all(i.source == "lookup" for i in items)


def test_custom_result_to_items_parser_is_called():
    """A custom parser overrides the default — verifies the indirection."""

    def lookup(query: str, limit: int = 5) -> dict:
        """Returns a weird custom shape only the custom parser understands."""
        return {"q": query, "n": limit}

    def custom_parse(result, query):
        # the parser sees the raw tool return and the original query string
        return [
            MemoryItem(
                content=f"custom:{result['q']}:{result['n']}",
                source="ignored",  # ToolMemory will stamp over this
            )
        ]

    mem = ToolMemory(
        tool=_make_tool(lookup),
        name="lookup",
        result_to_items=custom_parse,
    )
    ctx = make_test_ctx()
    items = _run(mem.query("birds", k=2, ctx=ctx))

    assert len(items) == 1
    assert items[0].content == "custom:birds:2"
    assert items[0].source == "lookup"  # stamped over


def test_default_parse_handles_dict_with_results_key():
    """A dict ``{"results": [...]}`` unwraps to the inner list."""
    raw = {"results": [{"content": "a", "score": 0.5}, {"text": "b"}]}
    items = default_parse(raw, query="")
    assert [i.content for i in items] == ["a", "b"]
    assert items[0].score == 0.5


def test_default_parse_handles_list_of_dicts_with_aliases():
    """``content`` / ``text`` / ``snippet`` are accepted as content aliases; the
    rest of the dict becomes metadata."""
    raw = [
        {"content": "first", "url": "u1"},
        {"snippet": "second", "rank": 2},
    ]
    items = default_parse(raw, query="q")
    assert [i.content for i in items] == ["first", "second"]
    # extra keys land in metadata
    assert items[0].metadata == {"url": "u1"}
    assert items[1].metadata == {"rank": 2}


def test_default_parse_handles_list_of_strings():
    items = default_parse(["one", "two", "three"], query="q")
    assert [i.content for i in items] == ["one", "two", "three"]
    assert all(i.source == "tool" for i in items)


def test_default_parse_splits_single_string_on_blank_lines():
    """A single string is split on blank lines (paragraph splits)."""
    raw = "para one\n\npara two\n\npara three"
    items = default_parse(raw, query="")
    assert [i.content for i in items] == ["para one", "para two", "para three"]


def test_default_parse_single_paragraph_single_item():
    """A string with no blank lines yields exactly one item."""
    items = default_parse("just one paragraph here", query="")
    assert len(items) == 1
    assert items[0].content == "just one paragraph here"


def test_default_parse_helper_is_re_exported_at_top_level():
    """Both ``agentkit.memory.default_parse`` and the module-level helper
    refer to the same callable so callers can compose freely."""
    assert default_parse is _direct_default_parse


def test_write_is_no_op():
    """Tools are query-only as memory; write must not raise and must not
    invoke anything on the wrapped tool."""
    called: list[int] = []

    def lookup(query: str, limit: int = 5) -> list[str]:
        """Toy lookup that records each call so we can assert write didn't trigger one."""
        called.append(1)
        return ["x"]

    mem = ToolMemory(tool=_make_tool(lookup), name="lookup")
    ctx = make_test_ctx()
    _run(mem.write([MemoryItem(content="ignored", source="x")], ctx=ctx))
    assert called == []  # tool was never invoked


def test_where_merges_into_tool_args_and_query_arg_is_configurable():
    """``where`` keys flow to the tool's args; ``query_arg`` picks which
    parameter receives the query string."""
    seen_args: dict = {}

    def search(q: str, limit: int = 5, site: str = "") -> list[str]:
        """Search tool that records its incoming args for assertion in the test."""
        seen_args["q"] = q
        seen_args["limit"] = limit
        seen_args["site"] = site
        return ["r1", "r2"]

    mem = ToolMemory(tool=_make_tool(search), name="web", query_arg="q")
    ctx = make_test_ctx()
    items = _run(mem.query("cats", k=4, ctx=ctx, where={"site": "wikipedia.org"}))

    assert seen_args == {"q": "cats", "limit": 4, "site": "wikipedia.org"}
    assert [i.content for i in items] == ["r1", "r2"]


def test_result_cap_at_k():
    """Even when the tool returns more than k items, the adapter caps."""

    def lookup(query: str, limit: int = 5) -> list[str]:
        """Returns 10 hits regardless of the requested limit so the cap is exercised."""
        del query, limit
        return [f"hit-{i}" for i in range(10)]

    mem = ToolMemory(tool=_make_tool(lookup), name="lookup")
    ctx = make_test_ctx()
    items = _run(mem.query("q", k=3, ctx=ctx))
    assert len(items) == 3


def test_tool_memory_keeps_the_record_id_a_custom_parser_set():
    """``ToolMemory.query`` re-stamps ``source`` by rebuilding each item, which
    means every field the rebuild omits is dropped. A parser that lifts the
    upstream API's record id — the whole reason a search tool is worth deduping
    against a vector store holding the same corpus — must not have it thrown
    away by the attribution pass."""

    def search(query: str, limit: int = 5) -> list[dict]:
        """Toy search returning documents that carry their own stable ids."""
        del query, limit
        return [{"content": "doc one", "id": "r7"}]

    def parse(result, query):
        del query
        return [MemoryItem(content=r["content"], source="unstamped", id=r["id"]) for r in result]

    mem = ToolMemory(tool=_make_tool(search), name="web", result_to_items=parse)
    items = _run(mem.query("q", k=3, ctx=make_test_ctx()))
    assert [(i.source, i.id) for i in items] == [("web", "r7")]


def test_default_parse_lifts_an_id_out_of_a_result_dict():
    """A search API returning ``{"content": ..., "id": ...}`` is the common
    shape, and an ``id`` that lands in metadata instead of on the field is an
    ``id`` the default fan-out dedupe never sees. It stays in metadata TOO,
    because callers already ``where``-filter and read it from there."""
    items = default_parse([{"content": "doc one", "id": "r7", "url": "u"}], "q")
    assert [i.id for i in items] == ["r7"]
    assert items[0].metadata["id"] == "r7"
    assert items[0].metadata["url"] == "u"


def test_default_parse_coerces_a_numeric_id_but_refuses_a_structured_one():
    """Ids arrive from JSON, so an integer id is routine — dropping it would
    disable dedupe for every store that numbers its rows. A structured id (a
    dict, a list) is NOT stringified: ``str({"a": 1})`` is a key that depends
    on dict ordering, so it would dedupe by accident and stop deduping after an
    unrelated serialisation change."""
    assert default_parse([{"content": "c", "id": 42}], "q")[0].id == "42"
    assert default_parse([{"content": "c", "id": {"a": 1}}], "q")[0].id is None
