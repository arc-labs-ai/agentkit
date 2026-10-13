"""Versioned agent configuration and structural permissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from fnmatch import fnmatchcase

from agentkit.agents.prompts.composition import digest
from agentkit.agents.skills.resources import SkillRef
from agentkit.models.contracts.constants import MindKind, ModelSelection
from agentkit.models.contracts.types import Binding, CallEvent, CallLimits, ModelSettings


@dataclass(frozen=True)
class ToolGrant:
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def permits(self, name: str) -> bool:
        return not any(fnmatchcase(name, x) for x in self.deny) and any(fnmatchcase(name, x) for x in self.allow)



@dataclass(frozen=True)
class DelegationPolicy:
    allowed: tuple[str, ...] = ()
    max_depth: int = 1
    max_concurrency: int = 2
    budget_share: Decimal = Decimal("0.25")

    def __post_init__(self) -> None:
        if self.max_depth < 0 or self.max_concurrency < 1 or not 0 < self.budget_share < 1:
            raise ValueError("invalid delegation policy")



@dataclass(frozen=True)
class TriggeredSkill:
    """Approved skill activated only by explicit host-supplied event facts."""

    ref: SkillRef
    event_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.ref.digest or not self.event_kinds or any(not kind for kind in self.event_kinds):
            raise ValueError("triggered skill requires an approved digest and event kinds")

    def matches(self, event: CallEvent) -> bool:
        return any(fnmatchcase(event.kind, pattern) for pattern in self.event_kinds)



@dataclass(frozen=True)
class AgentDefinition:
    id: str
    version: str
    description: str
    model: Binding | ModelSelection = ModelSelection.INHERIT
    persona: tuple[SkillRef, ...] = ()
    discoverable_skills: tuple[SkillRef, ...] = ()
    triggered_skills: tuple[TriggeredSkill, ...] = ()
    tools: ToolGrant = ToolGrant()
    settings: ModelSettings = ModelSettings()
    limits: CallLimits = CallLimits(max_turns=10)
    delegation: DelegationPolicy = DelegationPolicy()
    mind: MindKind | str = MindKind.REACT

    def __post_init__(self) -> None:
        if not self.id or not self.version or len(self.description) > 1024:
            raise ValueError("agent requires id, version and bounded description")
        if not isinstance(self.model, Binding):
            object.__setattr__(self, "model", ModelSelection(self.model))
        if self.mind in tuple(MindKind):
            object.__setattr__(self, "mind", MindKind(self.mind))

    @property
    def digest(self) -> str:
        # Credential resolver identity and secrets never enter manifests.
        return digest(
            {
                "id": self.id,
                "mind": self.mind,
                "version": self.version,
                "description": self.description,
                "model": [self.model.mode, self.model.model] if isinstance(self.model, Binding) else self.model,
                "persona": [asdict(x) for x in self.persona],
                "discoverable": [asdict(x) for x in self.discoverable_skills],
                "triggered": [asdict(x) for x in self.triggered_skills],
                "tools": asdict(self.tools),
                "settings": asdict(self.settings),
                "limits": {k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(self.limits).items()},
                "delegation": {**asdict(self.delegation), "budget_share": str(self.delegation.budget_share)},
            }
        )
