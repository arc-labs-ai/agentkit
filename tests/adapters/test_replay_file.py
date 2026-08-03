"""FileReplayStore — disk-backed ReplayStore round-trip + concurrency."""

import asyncio

import pytest

from agentkit.adapters.replay import FileReplayStore
from agentkit.kernel.replay import ReplayRecord


@pytest.mark.asyncio
async def test_put_then_get_round_trips(tmp_path):
    """A record written with put() is readable with get()."""
    store = FileReplayStore(tmp_path)
    rec = ReplayRecord(
        span_id="a" * 16,
        operation="chat",
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"content": "hello"},
    )
    await store.put(rec)
    got = await store.get(rec.span_id)
    assert got is not None
    assert got.span_id == rec.span_id
    assert got.operation == "chat"
    assert got.request == {"messages": [{"role": "user", "content": "hi"}]}
    assert got.response == {"content": "hello"}


@pytest.mark.asyncio
async def test_get_returns_none_for_missing_span_id(tmp_path):
    store = FileReplayStore(tmp_path)
    assert await store.get("b" * 16) is None


@pytest.mark.asyncio
async def test_put_rejects_invalid_span_id_silently(tmp_path):
    """A malformed span_id is dropped silently (defense against
    path traversal / arbitrary file names)."""
    store = FileReplayStore(tmp_path)
    bad = ReplayRecord(span_id="../../etc/passwd", operation="chat", request={}, response=None)
    await store.put(bad)  # must not raise
    # Nothing should be written to the root
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_directory_auto_created(tmp_path):
    """The store creates its root directory on first write."""
    nested = tmp_path / "does" / "not" / "exist"
    store = FileReplayStore(nested)
    rec = ReplayRecord(span_id="c" * 16, operation="chat", request={}, response=None)
    await store.put(rec)
    assert nested.exists()
    assert (nested / f"{rec.span_id}.json").exists()


@pytest.mark.asyncio
async def test_concurrent_writes_atomic(tmp_path):
    """Two concurrent writers to the same span_id don't corrupt
    the file — the last write wins, no torn reads."""
    store = FileReplayStore(tmp_path)
    span = "d" * 16

    async def write(content: str):
        await store.put(
            ReplayRecord(span_id=span, operation="chat", request={"c": content}, response=None)
        )

    await asyncio.gather(*(write(f"v{i}") for i in range(50)))
    got = await store.get(span)
    assert got is not None
    # The content is ONE of the writes — no merge, no corruption
    assert got.request["c"].startswith("v")


@pytest.mark.asyncio
async def test_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("RIO_REPLAY_DIR", raising=False)
    assert FileReplayStore.from_env() is None


@pytest.mark.asyncio
async def test_from_env_constructs_with_path(monkeypatch, tmp_path):
    monkeypatch.setenv("RIO_REPLAY_DIR", str(tmp_path))
    store = FileReplayStore.from_env()
    assert store is not None
    rec = ReplayRecord(span_id="e" * 16, operation="chat", request={"x": 1}, response=None)
    await store.put(rec)
    assert (tmp_path / f"{rec.span_id}.json").exists()


@pytest.mark.asyncio
async def test_dataclass_request_serializes_via_to_jsonable(tmp_path):
    """Dataclasses in request/response serialize via dataclasses.asdict."""
    from dataclasses import dataclass

    @dataclass
    class Req:
        prompt: str
        model: str

    store = FileReplayStore(tmp_path)
    rec = ReplayRecord(
        span_id="f" * 16,
        operation="chat",
        request=Req(prompt="hi", model="claude"),
        response=None,
    )
    await store.put(rec)
    got = await store.get(rec.span_id)
    assert got is not None
    assert got.request == {"prompt": "hi", "model": "claude"}
