from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentkit.models.providers.cli.isolation import IsolatedHome


def test_snapshot_roundtrip_excludes_credentials_and_preserves_private_permissions(tmp_path):
    source, restored = tmp_path / "source", tmp_path / "restored"
    (source / "projects" / "work").mkdir(parents=True)
    transcript = source / "projects" / "work" / "session.jsonl"
    transcript.write_bytes(b'{"text":"remember me"}\n')
    (source / "auth.json").write_text("never export")
    (source / "projects" / "settings.json").write_text("never export")
    restored.mkdir()
    bundle = IsolatedHome.snapshot(source)
    assert list(bundle) == ["projects/work/session.jsonl"]
    IsolatedHome.restore(restored, bundle)
    target = restored / "projects" / "work" / "session.jsonl"
    assert target.read_bytes() == transcript.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/sessions/x.jsonl",
        "../sessions/x.jsonl",
        "auth.json",
        "sessions/x.json",
        "sessions/../../x.jsonl",
        "sessions\\x.jsonl",
    ],
)
def test_restore_rejects_untrusted_paths_without_writing(tmp_path, name):
    with pytest.raises(PermissionError):
        IsolatedHome.restore(tmp_path, {name: "eA=="})
    assert not list(tmp_path.iterdir())


def test_restore_validates_entire_bundle_before_materializing(tmp_path):
    with pytest.raises(PermissionError):
        IsolatedHome.restore(tmp_path, {"sessions/good.jsonl": "eA==", "credentials.json": "eA=="})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("operation", ["snapshot", "restore"])
def test_session_bundle_has_total_size_limit(tmp_path, operation):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "one.jsonl").write_bytes(b"1234")
    (tmp_path / "sessions" / "two.jsonl").write_bytes(b"5678")
    with pytest.raises(ValueError, match="size limit"):
        if operation == "snapshot":
            IsolatedHome.snapshot(tmp_path, max_bytes=7)
        else:
            target = tmp_path / "target"
            target.mkdir()
            IsolatedHome.restore(
                target, {"sessions/one.jsonl": "MTIzNA==", "sessions/two.jsonl": "NTY3OA=="}, max_bytes=7
            )
            assert not list(target.iterdir())


def test_restore_rejects_encoded_allocation_before_decoding(tmp_path, monkeypatch):
    def unexpected_decode(*args, **kwargs):
        pytest.fail("oversized allocation reached base64 decoder")

    monkeypatch.setattr(base64, "b64decode", unexpected_decode)
    with pytest.raises(ValueError, match="size limit"):
        IsolatedHome.restore(tmp_path, {"sessions/x.jsonl": "A" * 10000}, max_bytes=8)


@pytest.mark.parametrize("operation", ["snapshot", "restore"])
def test_bundle_cannot_follow_directory_symlinks(tmp_path, operation):
    external = tmp_path / "external"
    external.mkdir()
    (external / "x.jsonl").write_text("private")
    home = tmp_path / "home"
    home.mkdir()
    (home / "sessions").symlink_to(external, target_is_directory=True)
    with pytest.raises(PermissionError):
        if operation == "snapshot":
            IsolatedHome.snapshot(home)
        else:
            IsolatedHome.restore(home, {"sessions/new.jsonl": "eA=="})
    assert not (external / "new.jsonl").exists()


def test_bundle_cannot_overwrite_existing_transcript(tmp_path):
    (tmp_path / "sessions").mkdir()
    target = tmp_path / "sessions" / "x.jsonl"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        IsolatedHome.restore(tmp_path, {"sessions/x.jsonl": "eA=="})
    assert target.read_text() == "original"


def test_owner_creation_and_sweeping_keep_all_live_owners(tmp_path):
    def create_owner(_):
        owner = IsolatedHome(tmp_path)
        IsolatedHome.sweep(tmp_path)
        return owner

    with ThreadPoolExecutor(max_workers=8) as pool:
        owners = list(pool.map(create_owner, range(20)))
    try:
        assert all(owner.root.exists() for owner in owners)
        IsolatedHome.sweep(tmp_path)
        assert all(owner.create().exists() for owner in owners)
    finally:
        for owner in owners:
            owner.close()
    assert not list(tmp_path.glob("owner-*"))


def test_crashed_creation_without_lock_is_swept(tmp_path):
    orphan = tmp_path / "owner-crashed"
    orphan.mkdir()
    (orphan / "auth.json").write_text("remove first")
    IsolatedHome.sweep(tmp_path)
    assert not orphan.exists()


def test_parent_symlink_rejected_and_closed_owner_cannot_create(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        IsolatedHome(link)
    with IsolatedHome(actual) as owner:
        assert owner.create().exists()
    with pytest.raises(RuntimeError):
        owner.create()


def test_credentials_removed_before_cleanup_retry(tmp_path, monkeypatch):
    from agentkit.models.providers.cli import isolation

    owner = IsolatedHome(tmp_path)
    (owner.root / "auth.json").write_text("secret")
    calls = []
    original = isolation.shutil.rmtree

    def remove(root):
        assert not (root / "auth.json").exists()
        calls.append(root)
        if len(calls) < 3:
            raise OSError("writer still closing")
        original(root)

    monkeypatch.setattr(isolation.shutil, "rmtree", remove)
    owner.close()
    assert len(calls) == 3


def test_cleanup_failure_is_bounded_and_releases_owner_lock(tmp_path, monkeypatch):
    from agentkit.models.providers.cli import isolation

    owner = IsolatedHome(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(isolation.shutil, "rmtree", lambda _: (_ for _ in ()).throw(OSError("locked")))
        with pytest.raises(OSError, match="bounded cleanup"):
            owner.close()
    IsolatedHome.sweep(tmp_path)
    assert not owner.root.exists()


@pytest.mark.parametrize("aliased", ["sessions/./x.jsonl", "sessions//x.jsonl"])
def test_bundle_aliases_are_rejected_before_any_file_is_written(tmp_path, aliased):
    with pytest.raises(PermissionError):
        IsolatedHome.restore(tmp_path, {"sessions/good.jsonl": "eA==", aliased: "eA=="})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("aliased", ["sessions/X.jsonl", "sessions/cafe\u0301.jsonl"])
def test_case_or_unicode_collisions_are_rejected_before_writing(tmp_path, aliased):
    original = "sessions/x.jsonl" if "X" in aliased else "sessions/caf\u00e9.jsonl"
    with pytest.raises(PermissionError, match="duplicate"):
        IsolatedHome.restore(tmp_path, {original: "eA==", aliased: "eA=="})
    assert not list(tmp_path.iterdir())


def test_without_posix_locking_cli_is_explicitly_unsupported(tmp_path, monkeypatch):
    from agentkit.models.providers.cli import isolation

    monkeypatch.setattr(isolation, "fcntl", None)
    with pytest.raises(NotImplementedError, match="POSIX"):
        IsolatedHome(tmp_path)


def test_api_adapter_import_remains_available_without_fcntl(tmp_path):
    import subprocess
    import sys

    script = """
import importlib
original = importlib.import_module
def without_fcntl(name, *args, **kwargs):
    if name == 'fcntl':
        raise ImportError('simulated non-POSIX platform')
    return original(name, *args, **kwargs)
importlib.import_module = without_fcntl
from agentkit.models.providers.frontiers.calls import ApiAdapter
from agentkit.models.providers.cli.isolation import IsolatedHome
assert ApiAdapter is not None
try:
    IsolatedHome()
except NotImplementedError:
    pass
else:
    raise AssertionError('CLI home unexpectedly supported')
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr


def test_cleanup_waits_for_background_writer_quiescence_after_removing_credentials(tmp_path, monkeypatch):
    import threading
    import time

    from agentkit.models.providers.cli import isolation

    owner = IsolatedHome(tmp_path)
    credential = owner.root / "auth.json"
    credential.write_text("fake-private-credential")
    attempted, quiesced = threading.Event(), threading.Event()
    original = isolation.shutil.rmtree
    attempts = []

    def writer():
        assert attempted.wait(1)
        assert not credential.exists()
        time.sleep(0.03)
        (owner.root / "final-cache-write").write_text("cache drained")
        quiesced.set()

    def remove(root):
        assert not credential.exists()
        attempts.append(root)
        attempted.set()
        if not quiesced.is_set():
            raise OSError("cache writer still draining")
        original(root)

    thread = threading.Thread(target=writer)
    thread.start()
    monkeypatch.setattr(isolation.shutil, "rmtree", remove)
    try:
        owner.close()
    finally:
        thread.join(timeout=1)
    assert quiesced.is_set() and not thread.is_alive()
    assert len(attempts) == 2 and not owner.root.exists()


def test_cleanup_persistent_failure_preserves_cause_and_has_fixed_retry_delay(tmp_path, monkeypatch):
    from agentkit.models.providers.cli import isolation

    owner = IsolatedHome(tmp_path)
    attempts, delays = [], []
    cause = OSError("persistent cleanup failure")

    def refuse(root):
        attempts.append(root)
        raise cause

    with monkeypatch.context() as patch:
        patch.setattr(isolation.shutil, "rmtree", refuse)
        patch.setattr(isolation.time, "sleep", delays.append)
        with pytest.raises(OSError, match="bounded cleanup") as failure:
            owner.close()
    assert failure.value.__cause__ is cause
    assert len(attempts) == 3 and delays == [0.05, 0.1]
    IsolatedHome.sweep(tmp_path)
    assert not owner.root.exists()
