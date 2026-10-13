"""Reason/action strategy using the runtime's shared tool loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.constants import MindKind
from agentkit.models.contracts.types import Answer, Binding


@dataclass(frozen=True)
class ReActMind:
    name: MindKind = MindKind.REACT

    async def run(
        self, runtime: CallRuntime, binding: Binding, prompt: Any, tools: Any = None, **kwargs: Any
    ) -> Answer[Any]:
        return await runtime.run(binding, prompt, tools, **kwargs)
