"""Reference storage survives restart and writes only new transcript chunks."""

import asyncio
import hashlib
import json
import stat
import subprocess
import sys

import pytest

from agentkit.core.contracts.sessions import SessionBundleRef
from agentkit.state.sessions import FileSessionBundleStore


@pytest.mark.asyncio
async def test_seventeen_mib_native_transcript_restores_by_reference_after_restart(tmp_path):
    home = tmp_path / "home"
    transcript = home / "sessions" / "native.jsonl"
    transcript.parent.mkdir(parents=True)
    data = b'{"message":"' + b"x" * (17 * 1024 * 1024 - 15) + b'"}\n'
    assert len(data) == 17 * 1024 * 1024
    transcript.write_bytes(data)
    (home / "auth.json").write_text("secret credential must not be bundled")
    store = FileSessionBundleStore(tmp_path / "store")
    reference = await store.save(home)
    assert isinstance(reference, SessionBundleRef)
    restored = tmp_path / "new-process-home"
    script = """
import asyncio,hashlib,json,sys
from pathlib import Path
from agentkit.state.sessions import FileSessionBundleStore,SessionBundleRef
async def work():
    await FileSessionBundleStore(Path(sys.argv[1])).restore(SessionBundleRef(sys.argv[2]),Path(sys.argv[3]))
    data=(Path(sys.argv[3])/'sessions'/'native.jsonl').read_bytes()
    print(json.dumps({'size':len(data),'digest':hashlib.sha256(data).hexdigest()}))
asyncio.run(work())
"""
    process = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(tmp_path / "store"), reference.key, str(restored)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {"size": len(data), "digest": hashlib.sha256(data).hexdigest()}
    assert stat.S_IMODE((restored / "sessions" / "native.jsonl").stat().st_mode) == 0o600
    assert (restored / "sessions" / "native.jsonl").read_bytes() == data
    assert not (restored / "auth.json").exists()
    assert len(reference.key) == 64


@pytest.mark.asyncio
async def test_second_slice_reuses_unchanged_chunks_and_previous_reference_stays_immutable(tmp_path):
    home = tmp_path / "home"
    file = home / "projects" / "project" / "thread.jsonl"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"a" * (2 * 65536))
    store = FileSessionBundleStore(tmp_path / "store")
    first = await store.save(home)
    before = set((tmp_path / "store" / "blobs").iterdir())
    with file.open("ab") as stream:
        stream.write(b'{"message":"second slice"}\n')
    second = await store.save(home, previous_ref=first)
    after = set((tmp_path / "store" / "blobs").iterdir())
    assert second != first and len(after - before) == 1
    assert await store.save(home, previous_ref=second) == second
    await store.restore(first, tmp_path / "old")
    await store.restore(second, tmp_path / "new")
    assert (tmp_path / "old" / "projects" / "project" / "thread.jsonl").read_bytes() == b"a" * (2 * 65536)
    assert (tmp_path / "new" / "projects" / "project" / "thread.jsonl").read_bytes() == file.read_bytes()


@pytest.mark.asyncio
async def test_corrupt_chunk_is_refused_without_partially_restoring_home(tmp_path):
    home = tmp_path / "home"
    (home / "sessions").mkdir(parents=True)
    (home / "sessions" / "thread.jsonl").write_bytes(b"original evidence")
    store = FileSessionBundleStore(tmp_path / "store")
    reference = await store.save(home)
    next((tmp_path / "store" / "blobs").iterdir()).write_bytes(b"corrupted evidence")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="digest"):
        await store.restore(reference, target)
    assert not target.exists()


@pytest.mark.asyncio
async def test_reference_traversal_and_source_symlinks_are_refused(tmp_path):
    store = FileSessionBundleStore(tmp_path / "store")
    with pytest.raises(ValueError, match="reference"):
        await store.restore(SessionBundleRef("../outside"), tmp_path / "target")
    home = tmp_path / "home"
    (home / "sessions").mkdir(parents=True)
    outside = tmp_path / "private.jsonl"
    outside.write_text("must not be read")
    (home / "sessions" / "link.jsonl").symlink_to(outside)
    with pytest.raises(PermissionError, match="symlink"):
        await store.save(home)
    assert not list((tmp_path / "store" / "blobs").iterdir())


@pytest.mark.parametrize("key", ["", "../outside", "a" * 63, "A" * 64, "g" * 64, 1])
def test_reference_rejects_non_digest_keys_before_storage_access(key):
    with pytest.raises(ValueError, match="reference"):
        SessionBundleRef(key)


@pytest.mark.asyncio
async def test_new_manifest_reference_cannot_escape_transcript_directories(tmp_path):
    store = FileSessionBundleStore(tmp_path / "store")
    manifest = json.dumps({"version": 1, "files": [{"path": "../auth.jsonl", "chunks": [], "size": 0}]}).encode()
    key = hashlib.sha256(manifest).hexdigest()
    path = tmp_path / "store" / "manifests" / key
    path.write_bytes(manifest)
    path.chmod(0o600)
    with pytest.raises(PermissionError, match="path"):
        await store.restore(SessionBundleRef(key), tmp_path / "target")
    assert not (tmp_path / "target").exists()


@pytest.mark.asyncio
async def test_session_store_refuses_an_unprotected_root_and_target_symlink(tmp_path):
    unprotected = tmp_path / "open-store"
    unprotected.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="private"):
        FileSessionBundleStore(unprotected)
    home = tmp_path / "home"
    (home / "sessions").mkdir(parents=True)
    (home / "sessions" / "thread.jsonl").write_text("evidence")
    store = FileSessionBundleStore(tmp_path / "store")
    reference = await store.save(home)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "target"
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="symlink"):
        await store.restore(reference, target)
    assert not list(outside.iterdir())
