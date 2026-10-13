"""Portable Agent Skills sources with pinned file manifests."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentkit.agents.prompts.composition import digest
from agentkit.core.serialization._immutable import deep_freeze


@dataclass(frozen=True)
class SkillRef:
    source: str
    uri: str
    digest: str | None = None


@dataclass(frozen=True)
class SkillDocument:
    ref: SkillRef
    name: str
    description: str
    body: str
    files: Mapping[str, tuple[str, int]] = field(hash=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    unknown_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", deep_freeze(dict(self.files)))
        object.__setattr__(self, "metadata", deep_freeze(dict(self.metadata)))


class SkillSource(Protocol):
    async def list(self) -> tuple[SkillDocument, ...]: ...
    async def get(self, ref: SkillRef) -> SkillDocument: ...
    async def read(self, ref: SkillRef, path: str) -> str: ...


def parse_skill(
    text: str, ref: SkillRef, files: Mapping[str, tuple[str, int]], *, directory_name: str
) -> SkillDocument:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError('Skill loading requires pip install "arc-agentkit[calls]"') from exc
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    pieces = text.split("\n---", 1)
    if len(pieces) != 2:
        raise ValueError("unterminated skill frontmatter")
    data = yaml.safe_load(pieces[0][4:])
    if not isinstance(data, dict):
        raise ValueError("skill frontmatter must be a mapping")
    name, description = data.get("name", ""), data.get("description", "")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
        raise ValueError("invalid skill name")
    if name != directory_name:
        raise ValueError("skill name must match its directory")
    if not isinstance(description, str) or not 1 <= len(description) <= 1024:
        raise ValueError("skill description must be 1–1024 characters")
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("skill metadata must be a mapping")
    manifest_digest = digest(dict(files))
    if ref.digest is not None and ref.digest != manifest_digest:
        raise PermissionError("skill files changed since approval")
    unknown = tuple(
        sorted(set(data) - {"name", "description", "license", "compatibility", "metadata", "allowed-tools"})
    )
    return SkillDocument(
        SkillRef(ref.source, ref.uri, manifest_digest),
        name,
        description,
        pieces[1].strip(),
        files,
        metadata,
        unknown,
    )


class FileSkillSource:
    """No symlink escape or script execution; scripts are ordinary resources."""

    def __init__(self, root: Path, *, source_id: str = "filesystem", max_file_bytes: int = 200_000) -> None:
        self.root = root.resolve()
        self.source_id = source_id
        self.max_file_bytes = max_file_bytes

    def _directory(self, ref: SkillRef) -> Path:
        if ref.source != self.source_id:
            raise ValueError("wrong skill source")
        path = (self.root / ref.uri).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise PermissionError("skill path escapes source root")
        return path

    def _load(self, ref: SkillRef) -> SkillDocument:
        directory = self._directory(ref)
        files = {}
        skill_text = None
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise PermissionError("skill symlinks are not allowed")
            if path.is_file():
                blob = path.read_bytes()
                if len(blob) > self.max_file_bytes:
                    raise ValueError("skill resource exceeds size bound")
                relative = path.relative_to(directory).as_posix()
                files[relative] = (hashlib.sha256(blob).hexdigest(), len(blob))
                if relative == "SKILL.md":
                    skill_text = blob.decode("utf-8")
        if skill_text is None:
            raise FileNotFoundError(directory / "SKILL.md")
        return parse_skill(skill_text, ref, files, directory_name=directory.name)

    async def get(self, ref: SkillRef) -> SkillDocument:
        return await asyncio.to_thread(self._load, ref)

    async def list(self) -> tuple[SkillDocument, ...]:
        paths = await asyncio.to_thread(lambda: sorted(self.root.glob("*/SKILL.md")))
        return tuple([await self.get(SkillRef(self.source_id, p.parent.name)) for p in paths])

    async def read(self, ref: SkillRef, path: str) -> str:
        doc = await self.get(ref)  # verifies the entire approved manifest
        if path not in doc.files:
            raise PermissionError("resource is outside the pinned manifest")
        resource = self._directory(ref) / path
        if resource.is_symlink() or not resource.resolve().is_relative_to(self._directory(ref)):
            raise PermissionError("skill resource escapes approved directory")
        blob = await asyncio.to_thread(resource.read_bytes)
        if (hashlib.sha256(blob).hexdigest(), len(blob)) != doc.files[path]:
            raise PermissionError("skill resource changed since approval")
        return blob.decode("utf-8")


class MemorySkillSource:
    def __init__(
        self, documents: Mapping[str, SkillDocument], resources: Mapping[tuple[str, str], str] | None = None
    ) -> None:
        self.documents = dict(documents)
        self.resources = dict(resources or {})

    async def list(self) -> tuple[SkillDocument, ...]:
        return tuple(self.documents[k] for k in sorted(self.documents))

    async def get(self, ref: SkillRef) -> SkillDocument:
        doc = self.documents[ref.uri]
        if doc.ref.source != ref.source or (ref.digest and ref.digest != doc.ref.digest):
            raise PermissionError("skill source or digest changed")
        return doc

    async def read(self, ref: SkillRef, path: str) -> str:
        doc = await self.get(ref)
        value = self.resources[(ref.uri, path)]
        fingerprint = (hashlib.sha256(value.encode()).hexdigest(), len(value.encode()))
        if doc.files.get(path) != fingerprint:
            raise PermissionError("skill resource changed since approval")
        return value
