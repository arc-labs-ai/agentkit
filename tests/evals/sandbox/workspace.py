"""Workspace tools with explicit write/test grants and sandboxed test processes.

Model-authored code never executes in the credential-bearing runner process.
macOS sandbox-exec and Linux bubblewrap deny network access and writes outside
the disposable workspace and Python runtime. Unsupported platforms refuse
process execution.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentkit.tools import ToolRegistry
from agentkit.tools.declarations.function import FunctionTool
from agentkit.tools.execution.policy import ToolSet


class SandboxUnavailable(RuntimeError):
    """A workload requiring process isolation cannot run on this host."""


class WorkspaceSandbox:
    MAX_FILE_BYTES = 256 * 1024
    MAX_OUTPUT_BYTES = 64 * 1024

    def __init__(self, root: Path, *, test_timeout_s: float = 60) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.test_timeout_s = test_timeout_s
        # uv overlays can expose dependencies through the parent import path
        # while their isolated executable has only the overlay's own packages.
        # Carry only actual runtime package directories, never PYTHONPATH or
        # arbitrary project roots, into the credential-free child interpreter.
        self._runtime_sites = tuple(
            sorted(
                {
                    Path(entry).resolve()
                    for entry in sys.path
                    if entry and Path(entry).name in {"site-packages", "dist-packages"} and Path(entry).is_dir()
                }
            )
        )
        self._writes: set[str] = set()
        self._tests: set[str] = set()
        self.test_runs: list[dict[str, Any]] = []
        self.audit: list[dict[str, Any]] = []

    def _path(self, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or any(x.startswith(".env") for x in relative.parts):
            raise PermissionError("path is outside evaluation workspace permissions")
        target = self.root / relative
        for parent in (target, *target.parents):
            if parent == self.root.parent:
                break
            if parent.is_symlink():
                raise PermissionError("workspace tools do not follow symlinks")
        resolved = target.resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError("path escapes evaluation workspace")
        return resolved

    def allow_write_paths(self, paths: list[str]) -> None:
        self._writes = {self._path(p).relative_to(self.root).as_posix() for p in paths}

    def allow_test_paths(self, paths: list[str]) -> None:
        self._tests = {self._path(p).relative_to(self.root).as_posix() for p in paths}

    async def list_files(self) -> list[str]:
        """List available project files, excluding private grading and process scratch files."""
        return await asyncio.to_thread(
            lambda: sorted(
                p.relative_to(self.root).as_posix()
                for p in self.root.rglob("*")
                if p.is_file()
                and not p.is_symlink()
                and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(self.root).parts)
            )
        )

    async def read_file(self, path: str) -> str:
        """Read a project file by relative path. Files are bounded and paths cannot escape the workspace."""
        target = self._path(path)
        if ".grading" in target.relative_to(self.root).parts:
            raise PermissionError("private grader files are unavailable to the model")

        def read() -> str:
            with target.open("rb") as stream:
                data = stream.read(self.MAX_FILE_BYTES + 1)
            if len(data) > self.MAX_FILE_BYTES:
                raise ValueError("file exceeds bounded read size")
            return data.decode()

        self.audit.append({"operation": "read", "path": path})
        return await asyncio.to_thread(read)

    async def write_file(self, path: str, content: str) -> str:
        """Replace one explicitly approved source file. Test files and all other paths are immutable."""
        target = self._path(path)
        name = target.relative_to(self.root).as_posix()
        if name not in self._writes:
            raise PermissionError("file is not in this task's approved write grant")
        if len(content.encode()) > self.MAX_FILE_BYTES:
            raise ValueError("replacement exceeds file size bound")
        await asyncio.to_thread(target.write_text, content)
        self.audit.append({"operation": "write", "path": name})
        return "Updated " + name

    async def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace one exact, unique source fragment in an approved file.

        Read the file first. An absent or ambiguous fragment is refused; this
        avoids regenerating an entire module for a small repair.
        """
        target = self._path(path)
        name = target.relative_to(self.root).as_posix()
        if name not in self._writes:
            raise PermissionError("file is not in this task's approved write grant")
        if not old_text:
            raise ValueError("replacement requires a nonempty source fragment")
        content = await self.read_file(path)
        if content.count(old_text) != 1:
            raise ValueError("source fragment must match exactly once")
        return await self.write_file(path, content.replace(old_text, new_text, 1))

    @staticmethod
    @lru_cache(maxsize=1)
    def process_available() -> bool:
        if sys.platform == "darwin":
            return shutil.which("sandbox-exec") is not None
        executable = shutil.which("bwrap") if sys.platform.startswith("linux") else None
        if executable is None:
            return False
        try:
            probe = subprocess.run(
                [executable, "--unshare-all", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "/bin/true"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return probe.returncode == 0

    def _profile(self) -> str:
        allowed_reads = {
            self.root,
            *self._runtime_sites,
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path(sys.executable).resolve().parent,
            Path("/System"),
            Path("/usr/lib"),
            Path("/usr/share"),
            Path("/private/var/db/timezone"),
            Path("/dev"),
        }
        rules = [
            "(version 1)",
            "(deny default)",
            "(allow process-exec)",
            "(allow process-fork)",
            "(allow file-read-metadata)",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            "(allow signal (target self))",
        ]
        rules.extend(
            f"(allow file-read* file-map-executable (subpath {json.dumps(str(path))}))"
            for path in sorted(allowed_reads)
        )
        # Runtime path resolution needs ancestor directory descriptors, not
        # permission to read the files stored beneath those directories.
        ancestors = {parent for path in allowed_reads for parent in path.parents}
        rules.extend(f"(allow file-read* (literal {json.dumps(str(path))}))" for path in sorted(ancestors))
        # Model-authored test processes cannot rewrite source or acceptance
        # tests. Source edits occur only through the parent's approved tool.
        rules.append(f"(allow file-write* (subpath {json.dumps(str(self.root / '.process'))}))")
        rules.append('(allow file-write* (subpath "/dev"))')
        return "\n".join(rules)

    async def run_tests(self, paths: list[str]) -> dict[str, Any]:
        """Run approved pytest files in the isolated project. Returns exit code, bounded output and duration."""
        if not self.process_available():
            raise SandboxUnavailable("sandboxed Python tests require macOS sandbox-exec or Linux bubblewrap")
        if not paths or len(paths) > 32:
            raise ValueError("provide one to thirty-two approved test paths")
        selected = [self._path(p).relative_to(self.root).as_posix() for p in paths]
        if any(p not in self._tests for p in selected):
            raise PermissionError("test path is not approved for this workload")
        scratch = self.root / ".process"
        scratch.mkdir(exist_ok=True)
        environment = {
            "PATH": os.defpath,
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "AGENTKIT_SKIP_REAL_CLI": "1",
            "LANG": "en_US.UTF-8",
        }
        bootstrap = (
            "import json,sys; sys.path[:0] = [sys.argv[1], *json.loads(sys.argv[2])]; import pytest; "
            "raise SystemExit(pytest.main(['-q', '--import-mode=importlib', '-p', 'no:cacheprovider', "
            "*sys.argv[3:]]))"
        )
        python_command = [
            sys.executable,
            "-I",
            "-c",
            bootstrap,
            str(self.root),
            json.dumps([str(path) for path in self._runtime_sites]),
            *selected,
        ]
        if sys.platform == "darwin":
            command = [shutil.which("sandbox-exec") or "sandbox-exec", "-p", self._profile(), *python_command]
        else:
            command = self._bubblewrap_command(python_command)
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.root,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        stdout = process.stdout
        output = bytearray()
        truncated = False

        async def drain() -> None:
            nonlocal truncated
            while chunk := await stdout.read(8192):
                remaining = self.MAX_OUTPUT_BYTES - len(output)
                output.extend(chunk[: max(0, remaining)])
                truncated |= len(chunk) > remaining

        reader = asyncio.create_task(drain())
        try:
            async with asyncio.timeout(self.test_timeout_s):
                await process.wait()
                await reader
        except BaseException:
            import signal

            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            await reader
            raise
        result = {
            "return_code": process.returncode,
            "output": output.decode(errors="replace"),
            "duration_s": round(time.monotonic() - started, 3),
            "truncated": truncated,
            "paths": selected,
        }
        self.test_runs.append(result)
        return result

    def _bubblewrap_command(self, command: list[str]) -> list[str]:
        """Expose only system runtime files and this read-only task workspace.

        The user, PID, IPC, UTS and network namespaces prevent the test process
        from inspecting the credential-bearing runner or reaching host services.
        Its only writable mount is the bounded per-run scratch directory.
        """
        executable = shutil.which("bwrap")
        if executable is None:
            raise SandboxUnavailable("Linux process isolation requires bubblewrap")
        if not self.root.is_relative_to(Path("/tmp")):
            raise SandboxUnavailable("Linux process isolation requires the task workspace beneath /tmp")
        args = [executable, "--die-with-parent", "--unshare-all", "--tmpfs", "/"]
        for path in (Path("/usr"), Path("/etc")):
            args.extend(("--dir", str(path), "--ro-bind", str(path), str(path)))
        for path in (Path("/bin"), Path("/sbin"), Path("/lib"), Path("/lib64")):
            if path.is_symlink():
                args.extend(("--symlink", os.readlink(path), str(path)))
            elif path.exists():
                args.extend(("--dir", str(path), "--ro-bind", str(path), str(path)))
        prefix = Path(sys.prefix).resolve()
        if not prefix.is_relative_to(Path("/usr")):
            for parent in reversed(prefix.parents):
                if parent == Path("/"):
                    break
                args.extend(("--dir", str(parent)))
            args.extend(("--ro-bind", str(prefix), str(prefix)))
        args.extend(("--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"))
        for parent in reversed(self.root.parents):
            if parent in (Path("/"), Path("/tmp")):
                continue
            args.extend(("--dir", str(parent)))
        args.extend(("--dir", str(self.root), "--ro-bind", str(self.root), str(self.root)))
        scratch = self.root / ".process"
        args.extend(("--bind", str(scratch), str(scratch), "--chdir", str(self.root), "--clearenv"))
        for key, value in {
            "PATH": os.defpath,
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "AGENTKIT_SKIP_REAL_CLI": "1",
            "LANG": "en_US.UTF-8",
        }.items():
            args.extend(("--setenv", key, value))
        return [*args, *command]

    @property
    def registry(self) -> ToolRegistry:
        return ToolRegistry.from_tools(
            [
                FunctionTool.from_callable(self.list_files, side_effecting=False),
                FunctionTool.from_callable(self.read_file, side_effecting=False),
                FunctionTool.from_callable(self.write_file, side_effecting=True),
                FunctionTool.from_callable(self.edit_file, side_effecting=True),
                FunctionTool.from_callable(self.run_tests, side_effecting=False),
            ]
        )

    @property
    def tools(self) -> ToolSet:
        return ToolSet(self.registry)
