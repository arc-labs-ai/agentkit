"""Prompt — the versioned, hand-authored system prompt seed.

agentkit's prompt management is intentionally minimal: the framework
ships the `Prompt` data type (id + version + template + render). Apps
own storage, discovery, and resolution: store prompts as code constants,
in config files, in a database — whatever fits. The framework's only
job is to define WHAT a prompt is, so attribution + versioning work
consistently across the agent roster."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prompt:
    """A versioned system prompt. The `version` label travels with every
    output the prompt produces (stamped via RequestBuilder onto
    `AgentResult` + traces), so a regression in quality maps back to a
    specific template version."""

    id: str
    version: str
    template: str
    inputs: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        return self.template.strip()


__all__ = ["Prompt"]
