"""Deliberate reasoning above models, without exposing private reasoning traces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from agentkit.agents.prompts.composition import Manifest, digest, inline_prompt
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.constants import MindKind
from agentkit.models.contracts.types import Answer, Binding


@dataclass(frozen=True)
class ChainOfThoughtMind:
    """Prompt-level deliberation, usable with any supported model mode.

    Reasoning effort remains a model setting. This strategy asks the model to
    solve intermediate steps internally and publishes only the requested final
    artifact; it neither requests nor logs a model's hidden chain of thought.
    """

    name: MindKind = MindKind.CHAIN_OF_THOUGHT
    guidance: str = (
        "Reason through the problem carefully in your internal analysis. "
        "Check dependencies and intermediate calculations before answering. "
        "Return only the requested final answer in the requested format."
    )

    async def run(
        self, runtime: CallRuntime, binding: Binding, prompt: Any, tools: Any = None, **kwargs: Any
    ) -> Answer[Any]:
        if tools is not None and tools.schemas():
            raise ValueError("chain-of-thought reasoning does not execute tools; choose ReActMind")
        prompt = inline_prompt(prompt) if isinstance(prompt, str) else prompt
        parts = (*prompt.manifest.parts, ("mind.chain_of_thought", "1", digest(self.guidance)))
        manifest = Manifest(parts, prompt.manifest.inputs_digest, digest([parts, prompt.manifest.inputs_digest]))
        composed = replace(
            prompt,
            system="\n\n".join(x for x in (prompt.system, self.guidance) if x),
            manifest=manifest,
            prefix_digest=digest([prompt.prefix_digest, self.guidance]),
        )
        return await runtime.ask(binding, composed, **kwargs)
