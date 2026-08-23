"""Prompt — the versioned, hand-authored system prompt seed.

agentkit's prompt management is intentionally minimal: the framework
ships the `Prompt` data type (id + version + template + render). Apps
own storage, discovery, and resolution: store prompts as code constants,
in config files, in a database — whatever fits. The framework's only
job is to define WHAT a prompt is, so attribution + versioning work
consistently across the agent roster."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    """The placeholder names ``template`` declares, e.g. ``("tenant", "tone")``
    for a template containing ``{tenant}`` and ``{tone}``. Declared on the value
    so a caller can validate a prompt against its call site before a run, and so
    a stored prompt carries its own contract."""

    def render(self, **values: Any) -> str:
        """Substitute the declared ``inputs`` and return the prompt text.

        With no ``inputs`` declared this is what it has always been — the
        stripped template — so every existing caller is unaffected.

        Substitution is a literal replacement of ``{name}`` for each DECLARED
        input, never ``str.format``. That is deliberate: system prompts are full
        of braces that are not placeholders — a JSON Schema, an example payload,
        a code fence — and ``format`` would either raise ``KeyError`` on them or
        silently eat the doubling a user added to escape them. Replacing only
        the names the prompt declared leaves every other brace untouched.

        Raises ``ValueError`` on a missing or unexpected input rather than
        rendering a half-filled prompt. A ``{tenant}`` that reaches the model
        unsubstituted is not a crash, which is exactly the problem: it is a
        plausible-looking prompt that quietly describes the wrong task.
        """
        if not self.inputs:
            # Nothing declared: no substitution, and no complaint about extra
            # kwargs would be wrong either — so refuse them, since passing
            # values to a prompt that declares none is a mistake at the call
            # site, not a no-op.
            if values:
                raise ValueError(
                    f"prompt {self.id!r} declares no inputs but render() got "
                    f"{sorted(values)}. Add them to `inputs=` or drop them."
                )
            return self.template.strip()

        declared = set(self.inputs)
        supplied = set(values)
        missing = sorted(declared - supplied)
        unexpected = sorted(supplied - declared)
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if unexpected:
                parts.append(f"unexpected {unexpected}")
            raise ValueError(
                f"prompt {self.id!r} v{self.version} declares inputs "
                f"{list(self.inputs)}; render() got " + " and ".join(parts)
            )

        text = self.template
        for name in self.inputs:
            text = text.replace("{" + name + "}", str(values[name]))
        return text.strip()


__all__ = ["Prompt"]
