"""`FileStore` — the zero-dependency durable `StorePort` (JSON files; survives a process restart)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote


class FileStore:
    """Durable `StorePort` backed by JSON files under `base_dir` — survives a process restart, so a
    human-gate suspend or a crashed run resumes from disk. Zero-dependency (stdlib `json`/`pathlib`) and
    async-first: the blocking file I/O is bridged via `asyncio.to_thread` so it never stalls the loop.

    The reference durable adapter; a Postgres/Redis `StorePort` is the same shape behind an extra. Values
    must be JSON-serializable (which the loop/workflow checkpoints are). Single-flight is in-process (an
    `asyncio.Lock` per key); cross-process single-flight needs the real DB's transaction — documented.
    `ttl` is accepted but not enforced (no expiry sweeper); use a TTL-native backend for that.
    """

    def __init__(self, base_dir: str) -> None:
        self._root = Path(base_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def _safe(self, key: str) -> str:
        # percent-encode every non-alphanumeric byte → reversible and collision-free (so `a/b`, `a\b`,
        # `a:b` map to distinct filenames). `safe=""` encodes `/` and `:` too.
        return quote(key, safe="")

    def _kv_path(self, key: str) -> Path:
        return self._root / f"{self._safe(key)}.json"

    def _log_path(self, key: str) -> Path:
        return self._root / f"{self._safe(key)}.log"

    async def get(self, key: str) -> Any | None:
        path = self._kv_path(key)

        def _read() -> Any:
            return json.loads(path.read_text()) if path.exists() else None

        return await asyncio.to_thread(_read)

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        path = self._kv_path(key)
        await asyncio.to_thread(lambda: path.write_text(json.dumps(value)))

    async def delete(self, key: str) -> None:
        path = self._kv_path(key)
        await asyncio.to_thread(lambda: path.unlink(missing_ok=True))  # idempotent

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        existing = await self.get(key)
        if existing is not None:
            return existing
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            existing = await self.get(key)
            if existing is not None:
                return existing
            result = await fn()  # a raised fn propagates here, unwritten (never cache failure)
            await self.set(key, result)
            return result

    async def append(self, key: str, value: Any) -> None:
        path = self._log_path(key)

        def _append() -> None:
            with path.open("a") as f:
                f.write(json.dumps(value) + "\n")

        await asyncio.to_thread(_append)

    async def list(self, key: str) -> list[Any]:
        path = self._log_path(key)

        def _read() -> list[Any]:
            if not path.exists():
                return []
            return [json.loads(line) for line in path.read_text().splitlines() if line]

        return await asyncio.to_thread(_read)
