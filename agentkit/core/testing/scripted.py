"""Scripted transports for deterministic ``CallRuntime`` contract tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agentkit.core.contracts.types import Mode
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.types import Binding, CallLimits, ModelSettings, SessionHandle, Turn, Workspace


class ScriptedAdapterExhausted(BaseException):
    """A runtime requested more transport turns than the script supplied."""


class ScriptedAdapter:
    """A finite sequence of transport facts; exhaustion fails loudly."""

    def __init__(self, turns: Iterable[Turn]) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    async def turn(
        self,
        binding: Binding,
        prompt: Any,
        *,
        settings: ModelSettings,
        limits: CallLimits,
        schema: dict[str, Any] | None,
        tools: Any,
        session: SessionHandle | None,
        cancel: Any,
        emit: Any,
        call_id: str,
        native_agents: Any = None,
        workspace: Workspace | None = None,
    ) -> Turn:
        self.requests.append(
            {
                "binding": binding,
                "prompt": prompt,
                "settings": settings,
                "limits": limits,
                "schema": schema,
                "tools": tools,
                "session": session,
                "cancel": cancel,
                "call_id": call_id,
                "native_agents": native_agents,
                "workspace": workspace,
            }
        )
        if not self.turns:
            raise ScriptedAdapterExhausted(f"script exhausted after {len(self.requests) - 1} transport turn(s)")
        return self.turns.pop(0)


def scripted_runtime(
    turns: Iterable[Turn],
    pricing: PriceTable,
    *,
    mode: Mode = Mode.OPENAI_API,
    **runtime_options: Any,
) -> CallRuntime:
    """Create a runtime wired to one scripted mode and an explicit price table."""
    mode = Mode(mode)
    return CallRuntime(
        adapters={mode: ScriptedAdapter(turns)},
        pricing=pricing,
        **runtime_options,
    )
