"""``JournalMemory`` — recency-ordered ``MemorySource`` over a ``MutationJournal``.

Pins recency semantics (most-recent ``k``), the lexical filter on
rendered text, the optional ``role`` filter, and the read-only
contract (``write`` is a no-op).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agentkit.context.context import MutationJournal
from agentkit.memory import JournalMemory, MemoryItem
from agentkit.testing import make_test_ctx


def _run(coro):
    return asyncio.run(coro)


@dataclass(frozen=True)
class _Entry:
    text: str
    role: str | None = None


def test_query_returns_most_recent_k_entries():
    journal: MutationJournal[_Entry] = MutationJournal()
    for i in range(5):
        journal.record(_Entry(text=f"step-{i}"))

    mem = JournalMemory(journal=journal, render=lambda e: e.text)
    ctx = make_test_ctx()
    items = _run(mem.query("", k=3, ctx=ctx))

    # Newest-first order; score is None (no relevance ranking).
    assert [i.content for i in items] == ["step-4", "step-3", "step-2"]
    assert all(i.score is None for i in items)
    assert all(i.source == "journal" for i in items)


def test_query_filters_by_lexical_substring_on_rendered_text():
    journal: MutationJournal[_Entry] = MutationJournal()
    journal.record(_Entry(text="planner: decompose"))
    journal.record(_Entry(text="researcher: fetch"))
    journal.record(_Entry(text="planner: refine"))

    mem = JournalMemory(journal=journal, render=lambda e: e.text)
    ctx = make_test_ctx()
    items = _run(mem.query("planner", k=10, ctx=ctx))

    # Newest matches first, only "planner" entries.
    assert [i.content for i in items] == [
        "planner: refine",
        "planner: decompose",
    ]


def test_query_role_where_filter():
    journal: MutationJournal[_Entry] = MutationJournal()
    journal.record(_Entry(text="a", role="planner"))
    journal.record(_Entry(text="b", role="researcher"))
    journal.record(_Entry(text="c", role="planner"))
    journal.record(_Entry(text="d", role="critic"))

    mem = JournalMemory(journal=journal, render=lambda e: e.text)
    ctx = make_test_ctx()
    items = _run(mem.query("", k=10, ctx=ctx, where={"role": "planner"}))

    assert [i.content for i in items] == ["c", "a"]


def test_query_role_filter_skips_entries_without_role_attr():
    # Entries lacking a `role` attribute should not satisfy a role filter.
    @dataclass(frozen=True)
    class _Plain:
        text: str

    journal: MutationJournal[_Plain] = MutationJournal()
    journal.record(_Plain(text="x"))
    journal.record(_Plain(text="y"))

    mem = JournalMemory(journal=journal, render=lambda e: e.text)
    ctx = make_test_ctx()
    items = _run(mem.query("", k=10, ctx=ctx, where={"role": "planner"}))
    assert items == []


def test_write_is_a_noop():
    journal: MutationJournal[_Entry] = MutationJournal()
    journal.record(_Entry(text="seed"))

    mem = JournalMemory(journal=journal, render=lambda e: e.text)
    ctx = make_test_ctx()
    _run(mem.write([MemoryItem(content="should not append", source="x")], ctx=ctx))

    # Journal still has its single seeded entry.
    assert [e.text for e in journal.view()] == ["seed"]
