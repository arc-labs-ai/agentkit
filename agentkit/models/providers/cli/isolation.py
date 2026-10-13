"""Owned CLI homes with locks, credential-first teardown and orphan sweeping."""

from __future__ import annotations

import asyncio
import atexit
import base64
import importlib
import inspect
import os
import shutil
import signal
import tempfile
import threading
import time
import unicodedata
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, ClassVar

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

    _active: ClassVar[weakref.WeakSet[IsolatedHome]] = weakref.WeakSet()
    _previous_signal_handlers: ClassVar[dict[int, Any]] = {}

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
        self._active.add(self)
        self._install_signal_cleanup()

    @classmethod
    def _install_signal_cleanup(cls) -> None:
        """Chain process signals so owned credential homes are removed first."""
        if threading.current_thread() is not threading.main_thread():
            return  # Python only permits process signal handlers on the main thread.
        for signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if signum is None or signum in cls._previous_signal_handlers:
                continue
            previous = signal.getsignal(signum)
            if previous == signal.SIG_IGN:
                continue
            signal.signal(signum, cls._handle_signal)
            cls._previous_signal_handlers[signum] = previous

    @classmethod
    def _handle_signal(cls, signum: int, frame: Any) -> None:
        previous = cls._previous_signal_handlers.get(signum, signal.SIG_DFL)
        for owner in tuple(cls._active):
            try:
                owner.close()
            except OSError:
                # Preserve signal delivery even if the filesystem is already
                # unavailable; the next owner sweep removes any orphan.
                pass
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        # SIG_IGN is never installed above, but retaining this branch makes
        # restoration behavior explicit if a host changes handlers at runtime.

    @classmethod
    def _restore_signal_cleanup(cls) -> None:
        if cls._active or threading.current_thread() is not threading.main_thread():
            return
        for signum, previous in tuple(cls._previous_signal_handlers.items()):
            if signal.getsignal(signum) == cls._handle_signal:
                signal.signal(signum, previous)
        cls._previous_signal_handlers.clear()

    def create(self) -> Path:
        if self._lock is None:
            raise RuntimeError("home owner is closed")
        return Path(tempfile.mkdtemp(prefix="call-", dir=self.root))

    @staticmethod
    def install_credentials(home: Path, mode: str, files: Mapping[str, bytes]) -> dict[str, bytes]:
        """Install only the native login files supported by each CLI, at mode 0600."""
        allowed = {
            "claude_code": {".credentials.json"},
            "codex": {"auth.json"},
        }.get(str(mode), set())
        if not allowed and files:
            raise ValueError("this CLI mode does not accept home credentials")
        if home.is_symlink() or not home.is_dir():
            raise ValueError("credential home must be an owned directory")
        if set(files) - allowed:
            raise ValueError("credential file is outside the mode allowlist")
        for name, data in files.items():
            if name not in allowed or "/" in name or "\\" in name or not isinstance(data, bytes):
                raise ValueError("invalid credential file")
            if (home / name).exists() or (home / name).is_symlink():
                raise FileExistsError("credential install cannot overwrite an existing file")
        installed: dict[str, bytes] = {}
        try:
            for name, data in files.items():
                target = home / name
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                os.chmod(target, 0o600)
                installed[name] = data
        except BaseException:
            for name in files:
                (home / name).unlink(missing_ok=True)
            raise
        return installed

    @staticmethod
    def materialize_images(home: Path, images: Any) -> tuple[Path, ...]:
        """Write bounded prompt images beneath the isolated home for Codex CLI."""
        suffixes = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
        paths: list[Path] = []
        total = 0
        try:
            for image in images:
                data = image.data
                mime = image.mime_type
                total += len(data)
                if mime not in suffixes or total > 40 * 1024 * 1024:
                    raise ValueError("unsupported or oversized prompt images")
                target = home / f"agentkit-image-{len(paths)}{suffixes[mime]}"
                paths.append(target)
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
            return tuple(paths)
        except BaseException:
            for path in paths:
                path.unlink(missing_ok=True)
            raise

    @staticmethod
    async def credential_lifecycle(
        home: Path,
        mode: str,
        files: Mapping[str, bytes],
        on_changed: Any = None,
        on_error: Any = None,
    ) -> Any:
        """Async context that returns rotated logins once and always removes them."""
        original = IsolatedHome.install_credentials(home, mode, files)

        class _Lifecycle:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                changed: dict[str, bytes] = {}
                try:
                    for name, old in original.items():
                        path = home / name
                        if path.is_symlink():
                            continue
                        if path.exists():
                            current = await asyncio.to_thread(path.read_bytes)
                            if current != old:
                                changed[name] = current
                    if changed and on_changed is not None:
                        result = on_changed(changed)
                        if inspect.isawaitable(result):
                            await result
                except Exception as callback_error:
                    if on_error is not None:
                        try:
                            result = on_error(callback_error)
                            if inspect.isawaitable(result):
                                await result
                        except Exception:
                            # Persistence and diagnostic hooks must never replace
                            # the model answer or cancellation outcome.
                            pass
                finally:
                    for name in original:
                        (home / name).unlink(missing_ok=True)

        return _Lifecycle()

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
            self._active.discard(self)
            self._restore_signal_cleanup()

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
