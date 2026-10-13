"""Agent definition store contract and in-memory implementation."""

from __future__ import annotations

from typing import Protocol

from agentkit.agents.definitions.agent import AgentDefinition


class AgentStore(Protocol):
    async def get(self, ref: str) -> AgentDefinition: ...
    async def put(self, definition: AgentDefinition) -> AgentDefinition: ...



class MemoryAgentStore:
    def __init__(self) -> None:
        self.definitions: dict[str, AgentDefinition] = {}

    async def get(self, ref: str) -> AgentDefinition:
        return self.definitions[ref]

    async def put(self, definition: AgentDefinition) -> AgentDefinition:
        key = f"{definition.id}@{definition.version}"
        old = self.definitions.get(key)
        if old and old.digest != definition.digest:
            raise ValueError("agent version already exists with a different digest")
        self.definitions[key] = definition
        return definition
