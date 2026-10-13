"""Deterministic prompt composition with content-addressed manifests."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agentkit.agents.prompts.prompt import Prompt


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
    images: tuple[ImagePart, ...] = ()


@dataclass(frozen=True)
class ImagePart:
    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        if self.mime_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            raise ValueError("unsupported image MIME type")
        if not self.data or len(self.data) > 20 * 1024 * 1024:
            raise ValueError("image must be non-empty and at most 20 MiB")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


def compose(
    parts: Sequence[PromptPart],
    inputs: Mapping[str, Any],
    *,
    max_chars: int = 200_000,
    persona_screening: Callable[[str], str] | None = None,
    images: Sequence[ImagePart] = (),
) -> ComposedPrompt:
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
        if part.kind == "persona" and not part.trusted and persona_screening is not None:
            screened = persona_screening(rendered)
            if not isinstance(screened, str) or not screened.strip():
                raise ValueError("persona screening must return non-empty text")
            rendered = screened
        manifest_parts.append((part.prompt.id, part.prompt.version, digest(rendered)))
        (system if part.trusted and part.kind in ("rules", "persona") else user).append(rendered)
        if part.stable:
            stable.append(rendered)
    s, u = "\n\n".join(system), "\n\n".join(user)
    if len(s) + len(u) > max_chars:
        raise ValueError("composed prompt exceeds size budget; pass context by reference")
    image_rows = tuple((f"image:{index}", "1", image.digest) for index, image in enumerate(images))
    inp = (
        digest({"inputs": dict(inputs), "images": [image.digest for image in images]})
        if images
        else digest(dict(inputs))
    )
    manifest_parts.extend(image_rows)
    rows = tuple(manifest_parts)
    return ComposedPrompt(s, u, Manifest(rows, inp, digest([rows, inp])), digest(stable), tuple(images))


def inline_prompt(text: str) -> ComposedPrompt:
    """Wrap literal text without interpreting JSON or code braces as placeholders."""
    manifest = Manifest((("inline", "1", digest(text)),), digest({}), digest(text))
    return ComposedPrompt("", text, manifest, digest([]))
