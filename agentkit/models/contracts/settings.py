"""Pre-dispatch setting checks driven by host-overridable capability data."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentkit.core.contracts.types import Mode
from agentkit.models.contracts.constants import Effort
from agentkit.models.contracts.types import Binding, Degradation, EffectiveSettings, ModelSettings

_EFFORTS = tuple(Effort)
_PRESENTATION = {"reasoning_summary", "verbosity", "tier"}


@dataclass(frozen=True)
class CapabilityTable:
    """Profile values are null (any value) or a list of accepted values."""

    version: str
    date: str
    profiles: Mapping[str, Mapping[str, Any]] = field(hash=False)

    def __post_init__(self) -> None:
        from agentkit.core.serialization._immutable import deep_freeze

        object.__setattr__(self, "profiles", deep_freeze(dict(self.profiles)))

    @classmethod
    def load(cls, path: Path | str) -> CapabilityTable:
        return cls(**json.loads(Path(path).read_text()))

    def check(self, binding: Binding, requested: ModelSettings) -> EffectiveSettings:
        profile = self.profiles.get(f"{binding.mode}/{binding.model}", self.profiles.get(binding.mode, {}))
        applied: dict[str, Any] = {}
        degradations = []
        for name, value in asdict(requested).items():
            if value is None:
                continue
            if name not in profile:
                if name not in _PRESENTATION:
                    raise ValueError(f"{binding.mode}/{binding.model} has no verified support for {name}")
                degradations.append(Degradation(name, value, None, "unsupported presentation hint"))
                continue
            allowed = profile[name]
            if allowed is not None and value not in allowed:
                if name == "effort" and value in _EFFORTS and allowed:
                    candidates = [x for x in allowed if x in _EFFORTS]
                    if not candidates:
                        raise ValueError("capability profile contains no recognized efforts")
                    nearest = min(candidates, key=lambda x: abs(_EFFORTS.index(x) - _EFFORTS.index(value)))
                    degradations.append(Degradation(name, value, nearest, "nearest supported effort"))
                    value = nearest
                elif name in _PRESENTATION:
                    degradations.append(Degradation(name, value, None, "unsupported presentation value"))
                    continue
                else:
                    raise ValueError(f"unsupported {name}={value!r}")
            applied[name] = value
        if applied.get("temperature") is not None and not 0 <= applied["temperature"] <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if applied.get("top_p") is not None and not 0 <= applied["top_p"] <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if applied.get("max_output_tokens") is not None and applied["max_output_tokens"] <= 0:
            raise ValueError("max_output_tokens must be positive")
        return EffectiveSettings(ModelSettings(**applied), tuple(degradations))


DEFAULT_CAPABILITIES = CapabilityTable(
    "1",
    "2026-10-13",
    {
        Mode.CLAUDE_CODE: {"effort": ["low", "medium", "high"]},
        Mode.CODEX: {
            "effort": ["low", "medium", "high", "xhigh"],
            "verbosity": ["low", "medium", "high"],
            "reasoning_summary": ["auto", "concise", "detailed", "none"],
        },
        # Unverified models accept only the mode's required output cap. Hosts add
        # model profiles based on live probes instead of guessing at sampling.
        Mode.ANTHROPIC_API: {"max_output_tokens": None, "cache_ttl": ["5m", "1h"]},
        Mode.OPENAI_API: {"max_output_tokens": None},
        "openai_api/gpt-4o": {"temperature": None, "top_p": None, "max_output_tokens": None},
        "openai_api/gpt-4o-mini": {"temperature": None, "top_p": None, "max_output_tokens": None},
    },
)
