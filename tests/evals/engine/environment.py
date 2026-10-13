"""Capture bounded runtime versions without recording credentials or the process environment."""

from __future__ import annotations

import asyncio
import os
import platform
from importlib.metadata import PackageNotFoundError, version

from agentkit.core.contracts.types import Mode


async def runtime_versions(modes: tuple[Mode, ...]) -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in ("arc-agentkit", "httpx", "mcp", "jsonschema"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    cli: dict[str, str] = {}
    if os.environ.get("AGENTKIT_SKIP_REAL_CLI") != "1":
        for mode, binary in ((Mode.CLAUDE_CODE, "claude"), (Mode.CODEX, "codex")):
            if mode not in modes:
                continue
            try:
                process = await asyncio.create_subprocess_exec(
                    binary, "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                )
            except OSError:
                cli[str(mode)] = "unavailable"
                continue
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(), timeout=3)
                cli[str(mode)] = stdout.decode(errors="replace").strip()[:200]
            except TimeoutError:
                process.kill()
                await process.communicate()
                cli[str(mode)] = "version probe timed out"
    return {"python": platform.python_version(), "platform": platform.system(), "packages": packages, "cli": cli}
