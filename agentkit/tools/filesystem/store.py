"""Durable virtual file trees over the host's StorePort."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from agentkit.core.contracts.ports import StorePort


class StoreFiles:
    """A FileTool/FileMemory backend with atomic tree mutations.

    The host supplies a fixed tenant ``namespace``; model paths never select
    another namespace or become operating-system paths. Each tree is one JSON
    document updated with the store's compare-and-set primitive, including
    rename and recursive delete. Durability and cross-process atomicity follow
    the supplied store: FileStore is process-local; Redis/Postgres provide
    their respective distributed guarantees. Concurrent FileStore users must
    share the same FileStore instance.
    """

    def __init__(
        self,
        store: StorePort,
        *,
        namespace: str,
        root: str = "/memories",
        max_bytes: int = 8_388_608,
        max_files: int = 4096,
    ) -> None:
        if not namespace or not namespace.strip() or "\x00" in namespace:
            raise ValueError("a nonempty host-controlled namespace is required")
        if max_bytes <= 0 or max_files <= 0:
            raise ValueError("file tree bounds must be positive")
        self.store = store
        self.namespace = namespace
        self._key = "files:" + json.dumps(namespace, ensure_ascii=True)
        self._root = self._canonical(root)
        if self._root == "/":
            raise ValueError("a non-root memory directory is required")
        self.max_bytes, self.max_files = max_bytes, max_files

    @staticmethod
    def _canonical(path: str) -> str:
        if not isinstance(path, str) or not path.startswith("/") or "\\" in path or "\x00" in path:
            raise ValueError("paths must be absolute virtual POSIX paths")
        if path != "/" and any(part in ("", ".", "..") for part in path[1:].split("/")):
            raise ValueError("path aliases and traversal are forbidden")
        return path

    def _path(self, path: str, *, listing: bool = False) -> str:
        path = self._canonical(path)
        if listing and path == "/":
            return path
        if path != self._root and not path.startswith(self._root + "/"):
            raise PermissionError("path is outside the memory root")
        return path

    def _tree(self, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            raise ValueError("stored file tree is not a string map")
        for path in value:
            if self._path(path) == self._root:
                raise ValueError("stored file tree contains a directory as a file")
        return dict(value)

    async def _mutate(self, change: Callable[[dict[str, str]], str]) -> str:
        for _ in range(64):
            previous = await self.store.get(self._key)
            tree = self._tree(previous)
            result = change(tree)
            if len(tree) > self.max_files or len(json.dumps(tree, ensure_ascii=False).encode()) > self.max_bytes:
                raise ValueError("file tree exceeds configured bounds")
            if await self.store.compare_and_set(self._key, previous, tree):
                return result
            await asyncio.sleep(0)
        raise RuntimeError("file tree update contention exceeded retry bound")

    async def view(self, path: str) -> str:
        path = self._path(path, listing=True)
        tree = self._tree(await self.store.get(self._key))
        if path in tree:
            return tree[path]
        children = sorted(key for key in tree if key.startswith(path.rstrip("/") + "/"))
        if children or path in ("/", self._root):
            return "\n".join(children)
        raise FileNotFoundError(path)

    async def create(self, path: str, file_text: str, *, overwrite: bool = False) -> str:
        path = self._path(path)
        if path == self._root or not isinstance(file_text, str):
            raise ValueError("a leaf file and string content are required")

        def change(tree: dict[str, str]) -> str:
            existed = path in tree
            if existed and not overwrite:
                raise FileExistsError(path)
            if any(key.startswith(path + "/") or path.startswith(key + "/") for key in tree):
                raise ValueError("file/directory collision")
            tree[path] = file_text
            return f"{'replaced' if existed else 'created'} {path}"

        return await self._mutate(change)

    async def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        path = self._path(path)
        if not old_str:
            raise ValueError("old_str must not be empty")

        def change(tree: dict[str, str]) -> str:
            if path not in tree:
                raise FileNotFoundError(path)
            if old_str not in tree[path]:
                raise ValueError(f"old_str not found in {path}")
            tree[path] = tree[path].replace(old_str, new_str, 1)
            return f"edited {path}"

        return await self._mutate(change)

    async def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        path = self._path(path)

        def change(tree: dict[str, str]) -> str:
            if path not in tree:
                raise FileNotFoundError(path)
            lines = tree[path].split("\n")
            if not isinstance(insert_line, int) or isinstance(insert_line, bool) or not 0 <= insert_line <= len(lines):
                raise ValueError("insert_line is outside the file")
            lines.insert(insert_line, insert_text.rstrip("\n"))
            tree[path] = "\n".join(lines)
            return f"inserted into {path} at line {insert_line}"

        return await self._mutate(change)

    async def delete(self, path: str) -> str:
        path = self._path(path)

        def change(tree: dict[str, str]) -> str:
            removed = [key for key in tree if key == path or key.startswith(path + "/")]
            if not removed:
                raise FileNotFoundError(path)
            for key in removed:
                del tree[key]
            return f"deleted {path}"

        return await self._mutate(change)

    async def rename(self, path: str, new_path: str) -> str:
        path, new_path = self._path(path), self._path(new_path)

        def change(tree: dict[str, str]) -> str:
            if path not in tree:
                raise FileNotFoundError(path)
            if (
                new_path == self._root
                or new_path in tree
                or any(key.startswith(new_path + "/") or new_path.startswith(key + "/") for key in tree)
            ):
                raise ValueError("rename target exists or collides with a directory")
            tree[new_path] = tree.pop(path)
            return f"renamed {path} -> {new_path}"

        return await self._mutate(change)


__all__ = ["StoreFiles"]
