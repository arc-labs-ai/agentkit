"""Workspace tools with explicit write/test grants and sandboxed test processes.

Model-authored code never executes in the credential-bearing runner process.
macOS sandbox-exec and Linux bubblewrap deny network access and writes outside
the disposable workspace and Python runtime. Unsupported platforms refuse
process execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
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


def _minimal_roots(paths: object) -> list[Path]:
    """The shortest set of trees covering ``paths``, as bwrap mounts.

    Drops anything under /usr (already bound) and anything contained by a tree
    already in the result, so the same directory is never bound twice.
    """
    unique = {Path(p) for p in paths}  # type: ignore[union-attr]
    roots: list[Path] = []
    for path in sorted(unique, key=lambda p: len(p.parts)):
        if path == Path("/") or path.is_relative_to(Path("/usr")):
            continue
        if any(path.is_relative_to(root) for root in roots):
            continue
        roots.append(path)
    return roots


def _executable_chain() -> list[Path]:
    """``sys.executable`` and every path its symlink chain names, hop by hop.

    Resolving in one step is not enough. A venv's ``bin/python`` is a symlink
    whose target is an ABSOLUTE path written at creation time, and that path
    may itself run through a symlinked directory — which is how a CI runner
    lays out its temp tree. Measured on such a layout::

        bin/python -> /…/work/_temp/uv-python-dir/…/bin/python3.12   (declared)
        readlink -f   /…/real_temp/uv-python-dir/…/bin/python3.12     (resolved)

    ``sys.base_prefix`` reports only the RESOLVED form, so binding that leaves
    the declared target absent inside the sandbox and the symlink dangles. The
    kernel follows the declared path, so that is the one ``execvp`` complains
    about — naming the symlink, never the target it could not reach.

    Bounded iteration rather than ``while``: a symlink cycle is somebody else's
    bug and must not become an infinite loop here.
    """
    chain: list[Path] = []
    current = Path(sys.executable)
    seen: set[Path] = set()
    for _ in range(16):
        chain.append(current)
        if current in seen or not current.is_symlink():
            break
        seen.add(current)
        target = Path(os.readlink(current))
        current = target if target.is_absolute() else (current.parent / target)
    return chain


def _interpreter_roots() -> list[Path]:
    """Every directory tree the sandboxed interpreter needs to exist.

    Three paths, because under modern tooling they are three different places
    and binding only one leaves a dangling symlink:

    - ``sys.prefix`` — the venv the process is running in. Under
      ``uv run --with X`` this is an EPHEMERAL overlay venv, and its
      ``bin/python`` is a symlink rather than a binary.
    - ``sys.base_prefix`` — where the stdlib and the real interpreter live,
      and what that symlink points AT. With a uv-managed Python (CI sets
      ``UV_PYTHON_INSTALL_DIR``) this is outside /usr, so leaving it unbound
      makes the symlink dangle — which ``execvp`` reports as
      "No such file or directory", naming the symlink and not the missing
      target.
    - the resolved executable's own root, for layouts where neither of the
      above covers it.

    Paths under /usr are skipped because /usr is already bound, and a path
    contained by another returned path is skipped so the same tree is not
    bound twice.
    """
    # Both the resolved AND the unresolved form of each, because they can be
    # different directories and the sandbox needs whichever one the interpreter
    # is actually INVOKED by. A CI runner reaches its temp tree through a
    # symlink (`/home/runner/work/_temp` -> elsewhere), so resolving gives the
    # real path while `sys.executable` keeps the symlinked one — bwrap then
    # binds a directory nothing executes and `execvp` reports the path it was
    # given as missing. Measured: binding only the resolved form reproduces
    # `bwrap: execvp .../\_temp/.../bin/python: No such file or directory`
    # exactly, while a non-symlinked layout passes, which is why this only ever
    # failed on the runner.
    candidates = {
        Path(sys.prefix),
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix),
        Path(sys.base_prefix).resolve(),
    }
    # Every hop of the executable's symlink chain, in BOTH forms. This is what
    # covers the declared-target case above, which neither prefix reports.
    for hop in _executable_chain():
        candidates.add(hop.parent.parent)
        candidates.add(hop.resolve().parent.parent)
    return _minimal_roots(candidates)


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

    @staticmethod
    def signals_deliverable() -> bool:
        """Can this host signal a sandboxed process it started?

        Separate from :meth:`process_available`, because the two fail
        independently: the sandbox can work perfectly while the parent is
        unable to kill it. A `bwrap` shipped setuid-root — the packaging used
        where unprivileged user namespaces are restricted — starts its child as
        root, and an unprivileged parent then gets EPERM from `killpg`.

        That makes deadline and cancellation enforcement unavailable rather
        than merely untested, which is a fact worth stating out loud instead of
        discovering as a confusing EPERM inside a timeout handler.
        """
        if sys.platform == "darwin":
            return True
        executable = shutil.which("bwrap")
        if executable is None:
            return False
        try:
            probe = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                [executable, "--unshare-all", "--ro-bind", "/", "/", "--dev", "/dev",
                 "--proc", "/proc", "/bin/sleep", "5"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return False
        try:
            os.killpg(probe.pid, signal.SIGKILL)
        except PermissionError:
            return False
        except ProcessLookupError:
            return True
        finally:
            with contextlib.suppress(Exception):
                probe.wait(timeout=5)
        return True

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
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # The host will not let us signal our own sandbox. Distributions
                # ship `bwrap` setuid-root where unprivileged user namespaces
                # are restricted, and the child then starts as root, so the
                # unprivileged parent gets EPERM. Swallowed rather than raised
                # because the caller is already unwinding a timeout or a
                # cancellation, and replacing that with an EPERM would report
                # the symptom instead of the deadline that fired. Enforcement
                # genuinely is weaker on such a host; `signals_deliverable()`
                # is how a caller finds that out up front.
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
        args.extend(("--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"))
        # The interpreter mounts go in AFTER ``--tmpfs /tmp``, and that ordering
        # is the fix rather than a tidy-up. bwrap applies operations in
        # sequence, so a prefix bound BEFORE the tmpfs is shadowed by it the
        # moment the prefix lives under /tmp — which is exactly where
        # ``uv run --with ...`` puts the ephemeral overlay venv it builds
        # (measured: sys.prefix = /tmp/.cache/uv/builds-v0/.tmpXXXXXX). The
        # bind silently vanished and bwrap reported
        # ``execvp .../bin/python: No such file or directory``.
        # The site-packages directories go in with them. The bootstrap puts
        # them on ``sys.path``, which does nothing if the directory is not in
        # the sandbox's filesystem at all — the interpreter then starts
        # correctly and dies on ``import pytest``. macOS's profile has always
        # listed them in ``allowed_reads``; the bubblewrap path never bound
        # them, and nothing noticed because the only Linux hosts that reach
        # here are CI runners, where the earlier interpreter failure happened
        # first and masked it.
        for path in _minimal_roots([*_interpreter_roots(), *self._runtime_sites]):
            args.extend(("--ro-bind", str(path), str(path)))
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
