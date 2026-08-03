"""File ops as a Tool — the agent-driven `memory(command=…)` file-tree the model edits explicitly.

This module is **only** the Tool half of agent-managed file memory: a `Tool` the LLM calls
to view/create/edit files via the Anthropic memory-tool command set —
`view` · `create` · `str_replace` · `insert` · `delete` · `rename`. The *read/score side*
of the same file tree (a uniform `MemorySource` over the backend) lives at
`agentkit.memory.FileMemory`; both wrap the same backend but answer different questions
("the model wants to edit a note" vs "what notes match this query?"). Register it like any tool:

    agent = Agent(name="reviewer", model=m, tools=[FileTool()])
    # the model then issues:  memory(command="view", path="/memories")
    #                         memory(command="create", path="/memories/bugs/race.md", file_text="…")

The default backend is an in-memory file tree (`InMemoryFiles`, deterministic, zero-dep);
inject a durable/FS backend for persistence across sessions. `FileTool` is a duck-typed tool
(`name`/`schema`/`run`/`side_effecting`), so `ToolRegistry`/`Agent` accept it directly.
"""

from __future__ import annotations

import posixpath
from typing import Any

from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import ToolSchema

_COMMANDS = ("view", "create", "str_replace", "insert", "delete", "rename")


class InMemoryFiles:
    """A minimal in-memory file tree (`path -> text`). The backend protocol is **async** so a real
    durable/filesystem backend does its blocking I/O off the loop (via `to_thread`) without stalling it —
    in-memory ops are instant but stay `async def` to keep the seam uniform (async-first)."""

    def __init__(self, root: str = "/memories") -> None:
        self._files: dict[str, str] = {}
        self._root = "/" + root.strip("/")  # the always-listable root, even when empty

    def _read(self, path: str) -> str:
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]

    async def view(self, path: str) -> str:
        if path in self._files:
            return self._files[path]  # a file → its contents
        prefix = path.rstrip("/") + "/"
        children = sorted(p for p in self._files if p.startswith(prefix))
        if children or path.rstrip("/") in ("", self._root):
            return "\n".join(children)  # a directory → its listing (maybe empty)
        raise FileNotFoundError(path)

    async def create(self, path: str, file_text: str, *, overwrite: bool = False) -> str:
        """Create a file. Refuses to silently clobber an existing path — pass
        ``overwrite=True`` to replace deliberately. An unconditional write
        would actively lie about what happened when the path already held a
        different note from a prior run. Pairs with ``rename``'s no-clobber
        semantics so all destructive writes are explicit."""
        existed = path in self._files
        if existed and not overwrite:
            raise FileExistsError(f"create {path!r}: file exists; pass overwrite=True to replace")
        self._files[path] = file_text
        return f"replaced {path}" if existed else f"created {path}"

    async def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        text = self._read(path)
        if old_str not in text:
            raise ValueError(f"old_str not found in {path}")
        self._files[path] = text.replace(old_str, new_str, 1)  # first occurrence only
        return f"edited {path}"

    async def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        lines = self._read(path).split("\n")  # split (not splitlines): preserves a trailing "" / \r\n
        if not 0 <= insert_line <= len(lines):
            raise ValueError(f"insert_line {insert_line} out of range (0..{len(lines)}) for {path}")
        lines.insert(insert_line, insert_text.rstrip("\n"))
        self._files[path] = "\n".join(lines)
        return f"inserted into {path} at line {insert_line}"

    async def delete(self, path: str) -> str:
        removed = [p for p in list(self._files) if p == path or p.startswith(path.rstrip("/") + "/")]
        if not removed:
            raise FileNotFoundError(path)
        for p in removed:
            del self._files[p]
        return f"deleted {path}"

    async def rename(self, path: str, new_path: str) -> str:
        if new_path in self._files:  # move semantics — refuse to silently clobber a note
            raise ValueError(f"rename target {new_path!r} already exists; delete it first")
        self._files[new_path] = self._read(path)
        del self._files[path]
        return f"renamed {path} -> {new_path}"


def _schema() -> ToolSchema:
    return ToolSchema(
        name="memory",
        description=(
            "Persistent note memory the agent manages itself: "
            "view/create/str_replace/insert/delete/rename files under a path."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "enum": list(_COMMANDS)},
                "path": {"type": "string"},
                "file_text": {"type": "string"},
                "overwrite": {
                    "type": "boolean",
                    "description": "When command=create, replace an existing file (default false: refuse-clobber).",
                },
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "insert_line": {"type": "integer"},
                "insert_text": {"type": "string"},
                "new_path": {"type": "string"},
            },
            "required": ["command", "path"],
        },
    )


class FileTool:
    """The `memory(command=…)` tool. Declares ``description`` and
    ``output_schema`` so ``isinstance(FileTool(), Tool)`` holds under the
    ``@runtime_checkable`` :class:`Tool` Protocol."""

    name = "memory"
    description = (
        "Persistent note memory the agent manages itself: view/create/"
        "str_replace/insert/delete/rename files under a path."
    )
    output_schema: dict[str, Any] | None = None
    side_effecting = True
    requires_approval = False
    caps: tuple[str, ...] = ()
    url_arg = None

    def __init__(self, backend: Any = None, *, root: str = "/memories") -> None:
        self.root = "/" + root.strip("/")  # canonical absolute root
        self._fs = backend if backend is not None else InMemoryFiles(self.root)
        self.schema = _schema()

    @staticmethod
    def _require(args: Any, cmd: str, *names: str) -> None:
        missing = [n for n in names if args.get(n) is None]
        if missing:
            raise ValueError(f"memory {cmd!r}: missing required arg(s) {missing}")

    def _confine(self, path: Any) -> str:
        """Normalize `path` and assert it stays under `self.root` — blocks `..` traversal and absolute
        escapes (`/etc/passwd`) before any command reaches the backend, so an injected filesystem backend
        can't be turned into an arbitrary read/write/delete primitive."""
        if not path:
            raise ValueError("memory: 'path' is required")
        base = path if str(path).startswith("/") else posixpath.join(self.root, str(path))
        norm = posixpath.normpath(base)
        if norm != self.root and not norm.startswith(self.root + "/"):
            raise PermissionError(f"memory path {path!r} escapes root {self.root!r}")
        return norm

    async def run(self, args: Any, ctx: Ctx | None = None) -> str:
        cmd = args.get("command")
        if cmd not in _COMMANDS:
            raise ValueError(f"unknown memory command {cmd!r} (expected one of {_COMMANDS})")
        path = self._confine(args.get("path"))
        if cmd == "view":
            return await self._fs.view(path)
        if cmd == "create":
            self._require(args, cmd, "file_text")
            # Pass through the optional ``overwrite`` flag so the model can
            # opt into replace, but default to refuse-clobber.
            overwrite = bool(args.get("overwrite", False))
            return await self._fs.create(path, args["file_text"], overwrite=overwrite)
        if cmd == "str_replace":
            self._require(args, cmd, "old_str", "new_str")
            return await self._fs.str_replace(path, args["old_str"], args["new_str"])
        if cmd == "insert":
            self._require(args, cmd, "insert_line", "insert_text")
            return await self._fs.insert(path, int(args["insert_line"]), args["insert_text"])
        if cmd == "delete":
            # Refuse to wipe the root path. Combined with the backend's
            # prefix-match deletion, ``delete("/memories")`` would erase every
            # file under root in one call. The model can still delete a
            # specific subpath or file.
            if path == self.root:
                raise PermissionError(
                    f"memory delete: refuses to wipe root path {self.root!r}; delete a specific subpath or file"
                )
            return await self._fs.delete(path)
        self._require(args, cmd, "new_path")  # rename
        return await self._fs.rename(path, self._confine(args.get("new_path")))  # confine both ends


__all__ = ["FileTool", "InMemoryFiles"]
