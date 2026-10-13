"""Private content-addressed transcript chunks; no inline bundle-size ceiling."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from agentkit.core.contracts.sessions import SessionBundleRef

_CHUNK_BYTES = 65536
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.is_symlink() or not path.is_dir() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("session bundle directories must be private and cannot be symlinks")


def _read_private(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if stat.S_IMODE(os.fstat(stream.fileno()).st_mode) & 0o077:
            raise PermissionError("session bundle files must be private")
        return stream.read()


class FileSessionBundleStore:
    """Chunked local storage for native ``sessions``/``projects`` JSONL files.

    Unchanged chunks are reused across slices; references are immutable and
    can be loaded by a fresh process. Credentials and unrelated home files
    are excluded. Hosts authorize access to the root and implement retention
    or garbage collection; this is not a checkpoint/job scheduler. Native
    transcript content is private persisted data, not event/log material.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        _private_directory(self.root)
        for name in ("blobs", "manifests"):
            _private_directory(self.root / name)

    def _path(self, group: str, key: str) -> Path:
        if not _DIGEST.fullmatch(key):
            raise ValueError("invalid session bundle reference or chunk digest")
        return self.root / group / key

    @staticmethod
    def _relative(name: str) -> Path:
        path = Path(name)
        if (
            path.is_absolute()
            or "\\" in name
            or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or len(path.parts) < 2
            or path.parts[0] not in ("sessions", "projects")
            or path.suffix != ".jsonl"
        ):
            raise PermissionError("invalid session transcript path")
        return path

    def _write(self, group: str, key: str, data: bytes) -> None:
        target = self._path(group, key)
        if target.exists():
            if _read_private(target) != data:
                raise ValueError("stored session bundle digest mismatch")
            return
        descriptor, temporary = tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    async def save(self, home: Path, *, previous_ref: SessionBundleRef | None = None) -> SessionBundleRef:
        return await asyncio.to_thread(self._save, home, previous_ref)

    def _save(self, home: Path, previous_ref: SessionBundleRef | None) -> SessionBundleRef:
        if home.is_symlink() or not home.is_dir():
            raise PermissionError("session source home cannot be a symlink")
        if previous_ref is not None:
            self._manifest(previous_ref)  # do not silently replace an invalid/missing prior reference
        files = []
        for folder in ("sessions", "projects"):
            for source in sorted((home / folder).rglob("*.jsonl")):
                relative = self._relative(source.relative_to(home).as_posix())
                if any((home / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
                    raise PermissionError("session source cannot follow symlinks")
                chunks, size = [], 0
                descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "rb") as stream:
                    while chunk := stream.read(_CHUNK_BYTES):
                        key = hashlib.sha256(chunk).hexdigest()
                        self._write("blobs", key, chunk)
                        chunks.append(key)
                        size += len(chunk)
                files.append({"path": relative.as_posix(), "chunks": chunks, "size": size})
        payload = json.dumps({"version": 1, "files": files}, sort_keys=True, separators=(",", ":")).encode()
        key = hashlib.sha256(payload).hexdigest()
        self._write("manifests", key, payload)
        return SessionBundleRef(key)

    def _manifest(self, reference: SessionBundleRef) -> dict[str, Any]:
        raw = _read_private(self._path("manifests", reference.key))
        if hashlib.sha256(raw).hexdigest() != reference.key:
            raise ValueError("session bundle manifest digest mismatch")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("files"), list):
            raise ValueError("invalid session bundle manifest")
        return value

    async def restore(self, reference: SessionBundleRef, home: Path) -> None:
        await asyncio.to_thread(self._restore, reference, home)

    def _restore(self, reference: SessionBundleRef, home: Path) -> None:
        manifest = self._manifest(reference)
        paths: set[str] = set()
        # Validate all references and chunks before creating any target files.
        for entry in manifest["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("invalid session bundle file entry")
            relative = self._relative(entry["path"])
            if entry["path"] in paths:
                raise ValueError("duplicate session transcript path")
            paths.add(entry["path"])
            if home.is_symlink() or any(
                (home / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)
            ):
                raise PermissionError("session restoration cannot follow symlinks")
            size = 0
            if not isinstance(entry.get("chunks"), list):
                raise ValueError("invalid session bundle chunk list")
            for key in entry["chunks"]:
                blob = _read_private(self._path("blobs", key))
                if hashlib.sha256(blob).hexdigest() != key:
                    raise ValueError("session bundle chunk digest mismatch")
                size += len(blob)
            if size != entry.get("size"):
                raise ValueError("session bundle file size mismatch")
        _private_directory(home)
        for entry in manifest["files"]:
            target = home / self._relative(entry["path"])
            _private_directory(target.parent)
            descriptor, temporary = tempfile.mkstemp(dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    for key in entry["chunks"]:
                        blob = _read_private(self._path("blobs", key))
                        if hashlib.sha256(blob).hexdigest() != key:
                            raise ValueError("session bundle chunk digest mismatch")
                        stream.write(blob)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)


__all__ = ["FileSessionBundleStore"]
