"""Owned CLI homes with locks, credential-first teardown and orphan sweeping."""

from __future__ import annotations

import atexit
import base64
import importlib
import os
import shutil
import tempfile
import time
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

try:
    fcntl: Any = importlib.import_module("fcntl")
except ImportError:  # Windows can still import and use API-only calls.
    fcntl = None


def _remove(root: Path) -> None:
    for name in ("auth.json", ".credentials.json", "credentials.json"):
        for path in root.rglob(name):
            path.unlink(missing_ok=True)
    # Children have exited before this is called. A bounded retry tolerates
    # plugin-cache writers completing concurrently with the last drain.
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            shutil.rmtree(root)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt < 2:
                # A completed CLI can leave a cache writer briefly draining.
                # Give it bounded elapsed time rather than three busy retries.
                time.sleep(0.05 * (attempt + 1))
    if root.exists():
        raise OSError("could not remove owned CLI home after bounded cleanup") from last_error


@contextmanager
def _parent_lock(parent: Path) -> Iterator[None]:
    # Serialize publication of owner directories against orphan collection.
    # O_NOFOLLOW prevents a substituted lock from redirecting this open.
    if fcntl is None:
        raise NotImplementedError("isolated CLI homes require POSIX file locking")
    descriptor = os.open(parent / ".sweep.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class IsolatedHome:
    """One lock-owning process root. Never sweeps a live owner's directory."""

    def __init__(self, parent: Path | None = None) -> None:
        if fcntl is None or not hasattr(os, "getuid"):
            raise NotImplementedError("isolated CLI homes require POSIX file locking")
        self.parent = parent or Path(tempfile.gettempdir()) / f"agentkit-homes-{os.getuid()}"
        self.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.parent.is_symlink() or self.parent.stat().st_uid != os.getuid():
            raise ValueError("home parent must be owned by this user and cannot be a symlink")
        os.chmod(self.parent, 0o700)
        with _parent_lock(self.parent):
            self._sweep_locked(self.parent)
            self.root = Path(tempfile.mkdtemp(prefix="owner-", dir=self.parent))
            self._lock: IO[bytes] | None = (self.root / "owner.lock").open("w+b")
            fcntl.flock(self._lock, fcntl.LOCK_EX)
        atexit.register(self.close)

    def create(self) -> Path:
        if self._lock is None:
            raise RuntimeError("home owner is closed")
        return Path(tempfile.mkdtemp(prefix="call-", dir=self.root))

    @staticmethod
    def snapshot(home: Path, *, max_bytes: int = 16 * 1024 * 1024) -> dict[str, str]:
        """Export only native transcripts, with a bound before reading each file."""
        if max_bytes < 0 or home.is_symlink():
            raise ValueError("invalid native session snapshot root or limit")
        bundle: dict[str, str] = {}
        total = 0
        for directory in ("sessions", "projects"):
            for path in sorted((home / directory).rglob("*.jsonl")):
                relative = path.relative_to(home)
                if any((home / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
                    raise PermissionError("session snapshot cannot follow symlinks")
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "rb") as stream:
                    data = stream.read(max_bytes - total + 1)
                total += len(data)
                if total > max_bytes:
                    raise ValueError("native session bundle exceeds size limit")
                bundle[relative.as_posix()] = base64.b64encode(data).decode()
        return bundle

    @staticmethod
    def restore(home: Path, bundle: dict[str, str], *, max_bytes: int = 16 * 1024 * 1024) -> None:
        """Validate the whole bundle before creating files beneath an owned home."""
        if max_bytes < 0 or home.is_symlink():
            raise ValueError("invalid native session restore root or limit")
        total = 0
        validated: list[tuple[Path, bytes]] = []
        seen: set[str] = set()
        for name, encoded in bundle.items():
            relative = Path(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] not in ("sessions", "projects")
                or relative.suffix != ".jsonl"
                or "\\" in name
                or relative.as_posix() != name
            ):
                raise PermissionError("invalid native session bundle path")
            portable_name = unicodedata.normalize("NFC", name).casefold()
            if portable_name in seen:
                raise PermissionError("duplicate native session bundle path")
            seen.add(portable_name)
            # Bound allocation before decoding, including adversarial padding.
            if len(encoded) > 4 * ((max_bytes - total + 2) // 3):
                raise ValueError("native session bundle exceeds size limit")
            data = base64.b64decode(encoded, validate=True)
            total += len(data)
            if total > max_bytes:
                raise ValueError("native session bundle exceeds size limit")
            for i in range(1, len(relative.parts) + 1):
                if (home / Path(*relative.parts[:i])).is_symlink():
                    raise PermissionError("native session restore cannot follow symlinks")
            target = home / relative
            if target.exists():
                raise FileExistsError("native session restore cannot overwrite existing files")
            validated.append((target, data))
        for target, data in validated:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)

    def close(self) -> None:
        if self._lock is None:
            return
        lock, self._lock = self._lock, None
        try:
            _remove(self.root)
        finally:
            lock.close()
            atexit.unregister(self.close)

    @staticmethod
    def sweep(parent: Path) -> None:
        if parent.is_symlink() or parent.stat().st_uid != os.getuid():
            raise ValueError("home parent must be owned by this user and cannot be a symlink")
        with _parent_lock(parent):
            IsolatedHome._sweep_locked(parent)

    @staticmethod
    def _sweep_locked(parent: Path) -> None:
        for root in parent.glob("owner-*"):
            if root.is_symlink() or not root.is_dir():
                continue
            try:
                descriptor = os.open(root / "owner.lock", os.O_RDWR | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "r+b") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    _remove(root)
            except FileNotFoundError:
                # Publication is serialized by the parent lock. An owner
                # without its lock is a crashed creation, safe to collect.
                _remove(root)

    def __enter__(self) -> IsolatedHome:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
