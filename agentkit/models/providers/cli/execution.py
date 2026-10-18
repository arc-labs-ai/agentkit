"""Typed process boundary. Host backends are trusted code, never caller evidence.

The shipped backend establishes no containment. Remote implementations must
project ordered stdout/stderr and stdin, a reliable exit status, tree termination
and cleanup onto this port. A disconnect must raise ExecutionOutcomeUnknown;
it must not synthesize a successful exit or a pre-dispatch failure.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import signal
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentkit.core.contracts.types import Mode
from agentkit.core.serialization._immutable import deep_freeze
from agentkit.models.contracts.constants import NativeToolPolicy
from agentkit.models.contracts.execution import ExecutionRequirement, ExecutionRequirements


class ExecutionOutcomeUnknown(RuntimeError):
    """The worker may have executed the invocation; automatic retry is unsafe."""


@dataclass(frozen=True)
class BackendReport:
    """Backend observations, not an attestation agentkit independently verified.

    No shipped verifier accepts host-declared guarantees as isolation evidence.
    """

    backend: str
    version: str
    guarantees: frozenset[ExecutionRequirement] = frozenset()


@dataclass(frozen=True)
class CliInvocation:
    """An adapter-built launch. Secrets never appear in its representation."""

    argv: tuple[str, ...] = field(repr=False)
    options: Mapping[str, Any] = field(repr=False)
    mode: Mode | None = None
    call_id: str = ""
    attempt_id: str | None = None
    execution: ExecutionRequirements = field(default_factory=ExecutionRequirements)
    native_tool_policy: NativeToolPolicy = NativeToolPolicy.WARN

    def __post_init__(self) -> None:
        if self.mode is not None:
            object.__setattr__(self, "mode", Mode(self.mode))
        object.__setattr__(self, "native_tool_policy", NativeToolPolicy(self.native_tool_policy))
        if not self.argv or any(not isinstance(arg, str) or "\0" in arg for arg in self.argv):
            raise ValueError("CLI invocation requires valid arguments")
        if set(self.options) - {"env", "cwd", "stdin", "stdout", "stderr", "start_new_session", "limit"}:
            raise ValueError("CLI invocation contains unsupported process configuration")
        environment = self.options.get("env", {})
        if not isinstance(environment, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
        ):
            raise ValueError("CLI invocation environment must map strings to strings")
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "options", deep_freeze(dict(self.options)))


class ExecutionProcess(Protocol):
    """Streams and lifecycle required by the existing CLI drivers."""

    stdin: asyncio.StreamWriter | None
    stdout: asyncio.StreamReader | None
    stderr: asyncio.StreamReader | None
    pid: int

    @property
    def returncode(self) -> int | None: ...
    async def wait(self) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    async def terminate_tree(self, grace_s: float) -> None: ...
    async def aclose(self) -> None: ...


class ExecutionBackend(Protocol):
    """Library port for trusted host execution, with no provider policy logic."""

    async def inspect(self) -> BackendReport: ...
    async def start(self, invocation: CliInvocation) -> ExecutionProcess: ...


class _Subprocess:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self.stdin, self.stdout, self.stderr = process.stdin, process.stdout, process.stderr
        self.pid = process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def wait(self) -> int:
        return await self._process.wait()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    async def terminate_tree(self, grace_s: float) -> None:
        from agentkit.models.providers.cli._common import _signal_tree

        # start() requires a new session, so its pgid is the original pid even
        # after the leader exits. Detached children remain outside this group.
        pgid = self.pid if hasattr(os, "killpg") and self.pid != os.getpgrp() else None
        _signal_tree(self._process, pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(self._process.wait(), grace_s)
        except TimeoutError:
            pass
        _signal_tree(self._process, pgid, signal.SIGKILL)
        await self._process.wait()

    async def aclose(self) -> None:
        # Sweep even after the leader exits: a child may still hold the pipes.
        await self.terminate_tree(0.1)


class SubprocessExecutionBackend:
    """Same-user subprocesses: private environment, no OS authority boundary."""

    async def inspect(self) -> BackendReport:
        return BackendReport("subprocess", f"1/python-{platform.python_version()}")

    async def start(self, invocation: CliInvocation) -> ExecutionProcess:
        from agentkit.models.providers.cli.capabilities import evaluate_cli_execution

        if invocation.mode is not None:
            report = evaluate_cli_execution(
                invocation.mode,
                await self.inspect(),
                invocation.execution,
                invocation.native_tool_policy,
            )
            if not report.compatible:
                raise ValueError("subprocess cannot establish the requested execution requirements")
        elif invocation.execution.required or invocation.native_tool_policy == NativeToolPolicy.DENY:
            raise ValueError("execution requirements require an identified provider mode")
        if invocation.options.get("start_new_session") is not True:
            raise ValueError("subprocess backend requires a private process group")
        process = await asyncio.create_subprocess_exec(*invocation.argv, **dict(invocation.options))
        return _Subprocess(process)


async def inspect_binary_version(executable: str) -> str | None:
    """Read a version with an empty home/environment, never native authentication.

    A version is diagnostic metadata, not proof of isolation or a binary pin.
    Unrecognized output stays unknown rather than becoming raw diagnostics.
    """
    from agentkit.models.providers.cli._common import _terminate

    with tempfile.TemporaryDirectory(prefix="agentkit-version-") as home:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "--version",
                env={
                    "PATH": os.environ.get("PATH", os.defpath),
                    "HOME": home,
                    "CLAUDE_CONFIG_DIR": home,
                    "CODEX_HOME": home,
                },
                cwd=home,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            output, _ = await asyncio.wait_for(process.communicate(), 5)
            match = re.fullmatch(rb"(?:codex-cli )?(\d+\.\d+\.\d+)(?: \(Claude Code\))?\s*", output)
            return match[1].decode("ascii") if process.returncode == 0 and match else None
        except (OSError, TimeoutError):
            return None
        finally:
            if process is not None and process.returncode is None:
                await _terminate(process, 0.1)
