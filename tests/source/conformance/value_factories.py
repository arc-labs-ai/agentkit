"""Representative public values for the package-wide value invariant sweep."""

from agentkit.agents.definitions.agent import AgentDefinition, TriggeredSkill
from agentkit.agents.prompts.composition import ComposedPrompt, Manifest
from agentkit.agents.skills.resources import SkillDocument, SkillRef
from agentkit.integrations.mcp.gateway import ServerConfig
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.schema import OutputSpec
from agentkit.models.contracts.types import (
    Answer,
    Binding,
    CredentialRef,
    EffectiveSettings,
    Failure,
    ModelSettings,
    SessionHandle,
    Usage,
)
from agentkit.tools.execution.policy import OverflowPolicy


def credentials():
    return {"ANTHROPIC_API_KEY": "fake-test-credential"}


def binding():
    return Binding("openai_api", "test", CredentialRef(credentials))


def answer():
    return Answer(
        {"nested": [1]},
        "answer",
        ("commentary", "answer"),
        "answered",
        Usage(10, output=5),
        {"input_tokens": 10},
        None,
        "unpriced",
        EffectiveSettings(ModelSettings(effort="high")),
    )


FACTORIES = {
    "AgentDefinition": lambda: AgentDefinition(
        "agent", "1", "description", binding(), persona=(SkillRef("fs", "demo"),)
    ),
    "Answer": answer,
    "Binding": binding,
    "ComposedPrompt": lambda: ComposedPrompt(
        "rules", "task", Manifest((("part", "1", "digest"),), "input", "digest"), "prefix"
    ),
    "CredentialRef": lambda: CredentialRef(credentials),
    "EffectiveSettings": lambda: EffectiveSettings(ModelSettings(effort="high")),
    "agentkit.models.contracts.types.Failure": lambda: Failure("auth", mode_detail={"status": 401}),
    "OutputSpec": lambda: OutputSpec({"type": "object", "properties": {"n": {"type": "integer"}}}),
    "OverflowPolicy": lambda: OverflowPolicy(256),
    "PriceTable": lambda: PriceTable("1", "today", {"test": {"input_uncached": "1", "output": "5"}}),
    "ServerConfig": lambda: ServerConfig("public", url="https://public.example/mcp", env={"VAR": "value"}),
    "SessionHandle": lambda: SessionHandle(
        "openai_api", "run", history=({"role": "user", "content": "hi"},), raw_usage={"input_tokens": 10}
    ),
    "TriggeredSkill": lambda: TriggeredSkill(SkillRef("memory", "demo", "digest"), ("harness.*",)),
    "SkillDocument": lambda: SkillDocument(
        SkillRef("fs", "demo", "digest"), "demo", "description", "body", {"SKILL.md": ("digest", 20)}, {"nested": [1]}
    ),
}
