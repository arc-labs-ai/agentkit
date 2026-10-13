"""Returned checkpoint handles must not rewrite persisted state across backends."""

from dataclasses import replace

import pytest

from agentkit.core.contracts.ports import Checkpoint, CheckpointStatus
from agentkit.core.serialization._immutable import thaw
from agentkit.state.checkpoints import Checkpointer, StoreBackedCheckpointStore
from agentkit.state.checkpoints.adapters import InMemoryCheckpointStore
from agentkit.state.stores.file import FileStore
from agentkit.state.stores.memory import InMemoryStore


def backend(name, path):
    if name == "versioned_memory":
        return InMemoryCheckpointStore()
    return StoreBackedCheckpointStore(InMemoryStore() if name == "kv_memory" else FileStore(path))


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ("versioned_memory", "kv_memory", "kv_file"))
async def test_snapshot_returned_state_and_metadata_cannot_rewrite_persisted_record(name, tmp_path):
    checkpointer = Checkpointer(backend(name, tmp_path))
    snapshot = await checkpointer.snapshot("run", {"nested": {"items": ["approved"]}},
                                          metadata={"audit": {"owners": ["host"]}})
    with pytest.raises(TypeError):
        snapshot.state["nested"]["items"].append("caller-modification")
    with pytest.raises(TypeError):
        snapshot.metadata["audit"]["owners"].append("caller")
    editable = thaw(snapshot.state)
    editable["nested"]["items"].append("copy-only")
    loaded = await checkpointer.resume("run")
    assert loaded.state == {"nested": {"items": ["approved"]}}
    assert loaded.metadata == {"audit": {"owners": ["host"]}}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ("versioned_memory", "kv_memory", "kv_file"))
async def test_checkpoint_port_save_and_read_handles_are_independent_wire_values(name, tmp_path):
    port = backend(name, tmp_path)
    value = Checkpoint("run", 1, {"items": ["first"]}, CheckpointStatus.RUNNING, 1.0, {"owners": ["host"]})
    await port.save(value)
    with pytest.raises(TypeError):
        value.state["items"].append("after-save")
    with pytest.raises(TypeError):
        value.metadata["owners"].append("after-save")
    first = await port.latest("run")
    assert first.state == {"items": ["first"]} and first.metadata == {"owners": ["host"]}
    with pytest.raises(TypeError):
        first.state["items"].append("after-read")
    with pytest.raises(TypeError):
        first.metadata["owners"].append("after-read")
    again = await port.at_version("run", 1)
    assert again.state == {"items": ["first"]} and again.metadata == {"owners": ["host"]}
    if name == "versioned_memory":
        await port.save(replace(value, version=2, state={"items": ["second"]}))
        assert (await port.at_version("run", 1)).state == {"items": ["first"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ("versioned_memory", "kv_memory", "kv_file"))
async def test_latest_terminal_checkpoint_never_restarts_an_earlier_running_snapshot(name, tmp_path):
    port = backend(name, tmp_path)
    checkpoint = Checkpointer(port)
    await checkpoint.snapshot("run", {"step": 1})
    await checkpoint.snapshot("run", {"step": 2}, status=CheckpointStatus.FAILED)
    assert await checkpoint.resume("run") is None
    terminal = await checkpoint.resume("run", include_terminal=True)
    assert terminal.status == CheckpointStatus.FAILED and terminal.version == 2
    assert terminal.state == {"step": 2}
    assert (await checkpoint.list_versions("run")) == ([1, 2] if name == "versioned_memory" else [2])
    assert (await port.at_version("run", 1) is not None) == (name == "versioned_memory")
    await checkpoint.delete("run")
    await checkpoint.delete("run")
    assert await checkpoint.resume("run", include_terminal=True) is None


@pytest.mark.asyncio
async def test_failed_checkpoint_write_does_not_advance_version_or_strand_run_lock():
    class FailOnce(InMemoryCheckpointStore):
        failed = False

        async def save(self, checkpoint):
            if not self.failed:
                self.failed = True
                raise OSError("controlled persistence failure before write")
            await super().save(checkpoint)

    checkpoint = Checkpointer(FailOnce())
    with pytest.raises(OSError, match="before write"):
        await checkpoint.snapshot("run", {"step": 1})
    assert await checkpoint.resume("run") is None
    import asyncio

    saved = await asyncio.wait_for(checkpoint.snapshot("run", {"step": 2}), timeout=1)
    assert saved.version == 1 and saved.state == {"step": 2}
