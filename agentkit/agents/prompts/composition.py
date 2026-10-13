"""Deterministic prompt composition with content-addressed manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agentkit.agents.prompts import Prompt


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PromptPart:
    prompt: Prompt
    kind: Literal["rules", "persona", "skill_index", "task", "context"] = "task"
    stable: bool = False
    trusted: bool = False


@dataclass(frozen=True)
class Manifest:
    parts: tuple[tuple[str, str, str], ...]
    inputs_digest: str
    digest: str


@dataclass(frozen=True)
class ComposedPrompt:
    system: str
    user: str
    manifest: Manifest
    prefix_digest: str


def compose(parts: Sequence[PromptPart], inputs: Mapping[str, Any], *, max_chars: int = 200_000) -> ComposedPrompt:
    """Render every part strictly, refuse overflow, and keep untrusted text in user content."""
    order = {"rules": 0, "persona": 1, "skill_index": 2, "task": 3, "context": 4}
    arranged = sorted(enumerate(parts), key=lambda x: (not x[1].stable, order[x[1].kind], x[0]))
    ids: set[str] = set()
    manifest_parts = []
    system: list[str] = []
    user: list[str] = []
    stable = []
    for _, part in arranged:
        if part.prompt.id in ids:
            raise ValueError(f"duplicate prompt part: {part.prompt.id}")
        ids.add(part.prompt.id)
        if part.kind == "rules" and not part.trusted:
            raise ValueError("rules must be engine-owned trusted parts")
        values = {key: inputs[key] for key in part.prompt.inputs if key in inputs}
        rendered = part.prompt.render(**values)
        manifest_parts.append((part.prompt.id, part.prompt.version, digest(rendered)))
        (system if part.trusted and part.kind in ("rules", "persona") else user).append(rendered)
        if part.stable:
            stable.append(rendered)
    s, u = "\n\n".join(system), "\n\n".join(user)
    if len(s) + len(u) > max_chars:
        raise ValueError("composed prompt exceeds size budget; pass context by reference")
    inp = digest(dict(inputs))
    rows = tuple(manifest_parts)
    return ComposedPrompt(s, u, Manifest(rows, inp, digest([rows, inp])), digest(stable))


def inline_prompt(text: str) -> ComposedPrompt:
    """Wrap literal text without interpreting JSON or code braces as placeholders."""
    manifest = Manifest((("inline", "1", digest(text)),), digest({}), digest(text))
    return ComposedPrompt("", text, manifest, digest([]))
