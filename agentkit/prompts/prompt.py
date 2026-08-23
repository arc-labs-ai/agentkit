"""Prompt — the versioned, hand-authored system prompt seed.

agentkit's prompt management is intentionally minimal: the framework
ships the `Prompt` data type (id + version + template + bind + render).
Apps own storage, discovery, and resolution: store prompts as code
constants, in config files, in a database — whatever fits. The
framework's only job is to define WHAT a prompt is, so attribution +
versioning work consistently across the agent roster."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_EMPTY: Mapping[str, Any] = MappingProxyType({})


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

    bound: Mapping[str, Any] = _EMPTY
    """Values already bound to the declared ``inputs``. Normally set via
    :meth:`bind` rather than passed to the constructor. Because the values live
    on the *value*, a bound ``Prompt`` renders with no arguments — which is what
    lets every consumer in the framework (RequestBuilder, the CLI cognitions)
    keep calling ``render()`` with nothing while still shipping substituted
    text. Frozen to a read-only mapping in ``__post_init__``."""

    def __post_init__(self) -> None:
        # Copy-then-freeze: a Prompt is a value, so it must not alias a dict the
        # caller can mutate afterwards.
        frozen = MappingProxyType(dict(self.bound))
        self._reject_undeclared(frozen, verb="bound")
        object.__setattr__(self, "bound", frozen)

    def __hash__(self) -> int:
        """Hash on IDENTITY — id, version, template, inputs — not on ``bound``.

        A frozen dataclass normally derives ``__hash__`` from every compared
        field, and adding a mapping silently made ``Prompt`` UNHASHABLE:
        ``hash(prompt)`` raised ``TypeError: unhashable type: 'dict'`` where it
        had worked before. That breaks any caller keeping prompts in a dict or
        set, or behind an ``lru_cache`` — quietly, and only at the call site
        that happens to hash one.

        Excluding ``bound`` is sound rather than a workaround: ``__eq__`` still
        compares it, and the hash invariant only requires that equal objects
        hash equally. Two prompts that differ ONLY in their bindings collide,
        which is exactly what a hash bucket is for. It also means a bound and
        unbound copy of the same template share a bucket, which is the useful
        behaviour for a cache keyed on prompt identity.
        """
        return hash((self.id, self.version, self.template, self.inputs))

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor instead of restoring ``__dict__``.

        ``bound`` is a ``MappingProxyType`` so it cannot be mutated through the
        field — but a proxy cannot be PICKLED, and ``copy.deepcopy`` falls back
        to the pickle protocol for objects that do not override it. So BOTH
        ``pickle.dumps(prompt)`` and a plain ``deepcopy(prompt)`` raised
        ``TypeError: cannot pickle 'mappingproxy' object`` the moment anything
        was bound. deepcopy is the one that mattered: ``Checkpointer.snapshot``
        deep-copies state at the durable seam and the ReAct cognition
        deep-copies per drive, so a ``Prompt`` reachable from either would have
        taken the whole run down rather than just failing to copy.

        One hook covers both, which is why there is no ``__deepcopy__`` beside
        it: ``copy.deepcopy`` deep-copies the constructor ARGUMENTS it finds
        here, so a caller's mutable bound value is still copied rather than
        aliased, and ``__post_init__`` re-freezes on the way back in. A
        ``__deepcopy__`` was written first and deleted — a mutation proving it
        redundant survived, which is exactly what a redundant code path looks
        like."""
        return (
            _rebuild_prompt,
            (self.id, self.version, self.template, self.inputs, dict(self.bound)),
        )

    def _reject_undeclared(self, values: Mapping[str, Any], *, verb: str) -> None:
        """Undeclared names are a call-site mistake, not a no-op — almost always
        a renamed placeholder or a typo, and quietly accepting one renders the
        OLD template with none of the new values."""
        if not values:
            return
        if not self.inputs:
            raise ValueError(
                f"prompt {self.id!r} declares no inputs but got {sorted(values)} "
                f"{verb}. Add them to `inputs=` or drop them."
            )
        unexpected = sorted(set(values) - set(self.inputs))
        if unexpected:
            raise ValueError(
                f"prompt {self.id!r} v{self.version} declares inputs "
                f"{list(self.inputs)}; got unexpected {unexpected} {verb}"
            )

    def bind(self, **values: Any) -> Prompt:
        """Return a copy of this prompt with ``values`` bound to its inputs.

        The prompt is frozen, so this never mutates — it hands back a new value,
        which is the point: a bound prompt is still a pinnable, diff-able
        ``Prompt`` carrying the same ``id`` / ``version``, and can be handed
        anywhere an unbound one goes (``Agent(prompt=...)``, ``RequestBuilder``).

        Binding is incremental and last-write-wins, so partial binding at
        construction and the rest at the call site both work::

            p.bind(tenant="acme").bind(tone="terse")   # == p.bind(tenant="acme", tone="terse")

        An undeclared name raises ``ValueError`` HERE rather than at
        ``render()`` — the earlier the typo surfaces, the cheaper it is.
        """
        self._reject_undeclared(values, verb="bound")
        return Prompt(
            id=self.id,
            version=self.version,
            template=self.template,
            inputs=self.inputs,
            bound={**self.bound, **values},
        )

    def render(self, **values: Any) -> str:
        """Substitute the declared ``inputs`` and return the prompt text.

        Values come from :meth:`bind` and/or from ``**values`` here, with
        ``**values`` winning on a collision (a per-call override of a bound
        default). With no ``inputs`` declared this is what it has always been —
        the stripped template — so every existing caller is unaffected.

        Substitution is a literal replacement of ``{name}`` for each DECLARED
        input, never ``str.format``. That is deliberate: system prompts are full
        of braces that are not placeholders — a JSON Schema, an example payload,
        a code fence — and ``format`` would either raise ``KeyError`` on them or
        silently eat the doubling a user added to escape them. Replacing only
        the names the prompt declared leaves every other brace untouched.

        Raises ``ValueError`` when an input is neither bound nor supplied,
        rather than rendering a half-filled prompt. A ``{tenant}`` that reaches
        the model unsubstituted is not a crash, which is exactly the problem: it
        is a plausible-looking prompt that quietly describes the wrong task.
        """
        if not self.inputs:
            # Nothing declared: no substitution, and no complaint about extra
            # kwargs would be wrong either — so refuse them, since passing
            # values to a prompt that declares none is a mistake at the call
            # site, not a no-op.
            self._reject_undeclared(values, verb="passed to render()")
            return self.template.strip()

        resolved = {**self.bound, **values}
        declared = set(self.inputs)
        missing = sorted(declared - set(resolved))
        unexpected = sorted(set(resolved) - declared)
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"missing {missing}")
            if unexpected:
                parts.append(f"unexpected {unexpected}")
            hint = (
                " — bind them on the value (prompt.bind("
                + ", ".join(f"{n}=..." for n in missing)
                + ")) or pass them to render()"
                if missing
                else ""
            )
            raise ValueError(
                f"prompt {self.id!r} v{self.version} declares inputs "
                f"{list(self.inputs)}; render() got " + " and ".join(parts) + hint
            )

        text = self.template
        for name in self.inputs:
            text = text.replace("{" + name + "}", str(resolved[name]))
        return text.strip()


def _rebuild_prompt(
    id_: str, version: str, template: str, inputs: tuple[str, ...], bound: dict[str, Any]
) -> Prompt:
    """Module-level factory for ``Prompt.__reduce__`` — pickle needs a global."""
    return Prompt(id=id_, version=version, template=template, inputs=inputs, bound=bound)


__all__ = ["Prompt"]
