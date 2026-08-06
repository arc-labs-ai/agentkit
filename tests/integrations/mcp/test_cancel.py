"""Cancellation propagates through ``MCPClient.call_tool``.

The v1 semantics are pragmatic: cancel is checked BEFORE dispatch so a
queued call never reaches the server. Mid-call cancellation is a
transport-level concern (closing the ``MCPClient`` context tears the
subprocess / http stream down). We verify the pre-dispatch check here
and confirm the session stays usable for follow-up calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentkit.testing.fakes.ctx import FakeCtx


class _CancellingCtx(FakeCtx):
    def __init__(self) -> None:
        super().__init__()
        self._cancel = False

    def trip(self) -> None:
        self._cancel = True

    def check_cancelled(self) -> None:
        if self._cancel:
            raise RuntimeError("run cancelled")


@pytest.mark.asyncio
async def test_call_tool_pre_dispatch_cancel(make_mcp_client: Any) -> None:
    async with make_mcp_client() as mcp_client:
        ctx = _CancellingCtx()
        ctx.trip()
        with pytest.raises(RuntimeError, match="cancelled"):
            await mcp_client.call_tool("add", {"a": 1, "b": 1}, ctx=ctx)
        # Session still usable for a follow-up call (cancel didn't wedge the transport).
        result = await mcp_client.call_tool("add", {"a": 2, "b": 3})
        texts = [getattr(b, "text", None) for b in result.content]
        assert "5" in texts
