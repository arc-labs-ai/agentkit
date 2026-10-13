"""One-shot reasoning strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.constants import MindKind
from agentkit.models.contracts.types import Answer, Binding


@dataclass(frozen=True)
class SingleCallMind:
    name: MindKind = MindKind.SINGLE_CALL

    async def run(
        self, runtime: CallRuntime, binding: Binding, prompt: Any, tools: Any = None, **kwargs: Any
    ) -> Answer[Any]:
        if tools is not None and tools.schemas():
            raise ValueError("single-call reasoning does not execute tools; choose ReActMind")
        return await runtime.ask(binding, prompt, **kwargs)
