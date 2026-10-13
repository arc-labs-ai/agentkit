import asyncio

import pytest

from agentkit.core.contracts.types import Scope
from agentkit.core.runtime.context import RunContext
from agentkit.state.memory import FileMemory, MemoryItem
from agentkit.state.stores.file import FileStore
from agentkit.tools import FileTool, StoreFiles


@pytest.mark.asyncio
async def test_durable_tree_edits_are_recalled_after_fresh_memory_and_store(tmp_path):
    files = StoreFiles(FileStore(str(tmp_path)), namespace="tenant-a")
    ctx = RunContext(correlation_id="memory", scope=Scope())
    memory = FileMemory(files)
    await memory.write(
        [MemoryItem("clinical incident routing old", source="source", metadata={"path": "/memories/case.md"})], ctx=ctx
    )
    await files.str_replace("/memories/case.md", "old", "approved")
    await files.insert("/memories/case.md", 1, "signed evidence owner=sre")
    await files.rename("/memories/case.md", "/memories/incident.md")
    fresh = FileMemory(StoreFiles(FileStore(str(tmp_path)), namespace="tenant-a"))
    recalled = await fresh.query("incident", k=2, ctx=ctx)
    assert recalled[0].content == "clinical incident routing approved\nsigned evidence owner=sre"
    assert recalled[0].metadata["path"] == "/memories/incident.md"
    await fresh.files.delete("/memories/incident.md")
    assert await fresh.query("incident", k=2, ctx=ctx) == []


@pytest.mark.asyncio
async def test_namespace_and_virtual_paths_cannot_escape_or_collide(tmp_path):
    store = FileStore(str(tmp_path))
    a, b = StoreFiles(store, namespace="a/b"), StoreFiles(store, namespace="a:b")
    await a.create("/memories/private.md", "private")
    assert await b.view("/") == ""
    for path in (
        "/memories/../outside",
        "/memories//alias",
        "relative",
        "/memories/./alias",
        "/memories\\outside",
        "/memories/null\x00.md",
    ):
        with pytest.raises(ValueError):
            await a.create(path, "bad")
    with pytest.raises(PermissionError):
        await a.create("/outside.md", "bad")
    with pytest.raises(ValueError):
        await a.create("/memories/private.md/child", "bad")
    with pytest.raises(FileExistsError):
        await a.create("/memories/private.md", "clobber")
    assert await a.view("/memories/private.md") == "private"


@pytest.mark.asyncio
async def test_atomic_rename_and_concurrent_creates_preserve_tree(tmp_path):
    store = FileStore(str(tmp_path))
    files = StoreFiles(store, namespace="a")
    await asyncio.gather(*(files.create(f"/memories/group/note{i}.md", f"evidence{i}") for i in range(20)))
    assert len((await files.view("/memories/group")).splitlines()) == 20
    with pytest.raises(ValueError):
        await files.rename("/memories/group/note0.md", "/memories/group/note1.md")
    assert await files.view("/memories/group/note0.md") == "evidence0"
    await files.delete("/memories/group")
    assert await files.view("/") == ""


@pytest.mark.asyncio
async def test_tree_bounds_refuse_without_committing_and_file_tool_uses_backend(tmp_path):
    files = StoreFiles(FileStore(str(tmp_path)), namespace="a", max_bytes=100, max_files=1)
    tool = FileTool(backend=files)
    ctx = RunContext(correlation_id="files", scope=Scope())
    await tool.run({"command": "create", "path": "/memories/a", "file_text": "kept"}, ctx)
    with pytest.raises(ValueError, match="bounds"):
        await files.create("/memories/b", "extra")
    with pytest.raises(ValueError, match="bounds"):
        await files.create("/memories/a", "x" * 200, overwrite=True)
    assert await files.view("/memories/a") == "kept"


@pytest.mark.asyncio
async def test_corrupted_store_tree_cannot_publish_outside_root_paths(tmp_path):
    store = FileStore(str(tmp_path))
    files = StoreFiles(store, namespace="a")
    await store.set(files._key, {"/outside/private": "corrupted"})
    with pytest.raises(PermissionError):
        await files.view("/")
