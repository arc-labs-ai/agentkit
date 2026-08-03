"""``FileMemory`` — lexical ``MemorySource`` over a file-tree backend.

These tests pin the contract: lexical substring scoring, zero-match
exclusion, ``path_prefix`` scoping, and a ``write`` path that round-
trips through the backend's ``create``/``view`` surface.
"""

from __future__ import annotations

import asyncio

from agentkit.memory import FileMemory, MemoryItem
from agentkit.testing import make_test_ctx
from agentkit.tools.file_tool import InMemoryFiles


def _run(coro):
    return asyncio.run(coro)


def test_query_returns_matching_files_ordered_by_match_count():
    files = InMemoryFiles()

    async def setup() -> FileMemory:
        await files.create("/memories/a.md", "alpha alpha alpha")
        await files.create("/memories/b.md", "alpha")
        await files.create("/memories/c.md", "no match here")
        return FileMemory(files=files)

    mem = _run(setup())
    ctx = make_test_ctx()
    items = _run(mem.query("alpha", k=5, ctx=ctx))

    paths = [i.metadata["path"] for i in items]
    assert paths == ["/memories/a.md", "/memories/b.md"]
    assert items[0].score == 3.0
    assert items[1].score == 1.0
    # Files with zero matches are excluded.
    assert "/memories/c.md" not in paths


def test_query_excludes_files_with_zero_matches():
    files = InMemoryFiles()

    async def setup() -> FileMemory:
        await files.create("/memories/a.md", "lorem")
        await files.create("/memories/b.md", "ipsum")
        return FileMemory(files=files)

    mem = _run(setup())
    ctx = make_test_ctx()
    items = _run(mem.query("dolor", k=5, ctx=ctx))
    assert items == []


def test_query_respects_path_prefix_where_filter():
    files = InMemoryFiles()

    async def setup() -> FileMemory:
        await files.create("/memories/keep/x.md", "needle here")
        await files.create("/memories/keep/y.md", "needle there")
        await files.create("/memories/skip/z.md", "needle elsewhere")
        return FileMemory(files=files)

    mem = _run(setup())
    ctx = make_test_ctx()
    items = _run(mem.query("needle", k=10, ctx=ctx, where={"path_prefix": "/memories/keep"}))
    paths = sorted(i.metadata["path"] for i in items)
    assert paths == ["/memories/keep/x.md", "/memories/keep/y.md"]


def test_query_caps_results_at_k():
    files = InMemoryFiles()

    async def setup() -> FileMemory:
        for i in range(5):
            await files.create(f"/memories/{i}.md", "needle")
        return FileMemory(files=files)

    mem = _run(setup())
    ctx = make_test_ctx()
    items = _run(mem.query("needle", k=2, ctx=ctx))
    assert len(items) == 2


def test_write_uses_metadata_path():
    files = InMemoryFiles()
    mem = FileMemory(files=files)
    ctx = make_test_ctx()

    _run(
        mem.write(
            [
                MemoryItem(
                    content="note body",
                    source="anywhere",
                    metadata={"path": "/memories/note.md"},
                )
            ],
            ctx=ctx,
        )
    )

    assert _run(files.view("/memories/note.md")) == "note body"


def test_write_hashes_when_metadata_path_missing():
    files = InMemoryFiles()
    mem = FileMemory(files=files)
    ctx = make_test_ctx()

    _run(
        mem.write(
            [MemoryItem(content="anonymous content", source="x")],
            ctx=ctx,
        )
    )

    listing = _run(files.view("/memories"))
    assert listing  # something was created under /memories
    # The single created file holds the original content.
    path = [line for line in listing.split("\n") if line][0]
    assert _run(files.view(path)) == "anonymous content"
