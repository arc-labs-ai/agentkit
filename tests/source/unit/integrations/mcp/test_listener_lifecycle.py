from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from agentkit.core.testing import make_test_ctx
from agentkit.integrations.mcp import serve_registry
from agentkit.integrations.mcp._transport import LoopbackMcpTransport
from agentkit.tools import FunctionTool, ToolRegistry


def echo(value: str) -> str:
    """Echo the supplied evidence string without modifying any state."""
    return value


def test_sequential_sse_listeners_do_not_inherit_previous_listener_shutdown() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.source.unit.integrations.mcp._listener_lifecycle_e2e"],
        cwd=Path(__file__).resolve().parents[5], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "ASGI callable returned" not in result.stderr
    assert "CancelledError" not in result.stderr
    assert json.loads(result.stdout) == {
        "completed": ["evidence-0", "evidence-1", "evidence-2"], "handlers_preserved": True,
    }


@pytest.mark.asyncio
async def test_cancelled_spec_entry_joins_listener_and_removes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    original = LoopbackMcpTransport.start

    async def delayed_start(self: LoopbackMcpTransport, app: Any) -> None:
        await original(self, app)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(LoopbackMcpTransport, "start", delayed_start)
    spec = serve_registry(ToolRegistry.from_tools([FunctionTool.from_callable(echo, side_effecting=False)]), name="cancelentry", ctx=make_test_ctx())
    task = asyncio.create_task(spec.__aenter__())
    await asyncio.wait_for(entered.wait(), 5)
    assert spec._transport is not None
    listener = spec._transport._task
    assert listener is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert listener.done()
    assert not spec._transport.running
    assert not spec.config_path.exists()


@pytest.mark.asyncio
async def test_cancelled_shutdown_waits_for_lifespan_before_rethrowing() -> None:
    from starlette.applications import Starlette

    closing = asyncio.Event()
    release = asyncio.Event()
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        yield
        closing.set()
        await release.wait()

    listener = LoopbackMcpTransport()
    await listener.start(Starlette(lifespan=lifespan))
    serving = listener._task
    assert serving is not None
    task = asyncio.create_task(listener.stop())
    await asyncio.wait_for(closing.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not serving.done(), "caller cancellation reached the owned listener"
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert serving.done()
    assert not listener.running
