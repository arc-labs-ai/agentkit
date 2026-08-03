"""The cache-stable head of a WorkingContext: system prompt + grounding.

Frozen by construction so it stays cache-stable across turns — this
matches the KV-cache discipline (mutating any token in the prefix
invalidates the cache from that point on). All per-turn mutation
lands in ``WorkingContext.messages`` (the tail).

``PrefixContext`` exposes a single accessor (``as_messages()``) so
downstream code (RequestBuilder, providers) sees the same
``list[Message]`` shape it sees for the tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentkit.kernel.types import Message


@dataclass(frozen=True)
class PrefixContext:
    """The cache-stable head: system prompt + grounding chunks.

    Why frozen: the per-turn churn is the messages tail; the prefix
    is the shared cacheable prefix every turn re-uses. Mutating it
    after construction would invalidate every provider's prompt
    cache from that token forward — exactly the opposite of what we
    want from a "context engineering" boundary.

    ``schema_block`` is reserved for the structured-output
    prompt-injection track (Track B3). When set, it appears as a
    third pinned system message after the prompt and grounding.

    Equality is structural (dataclass-generated). Two prefixes built
    from the same (prompt, grounding, schema) are equal and hash
    equal — useful when keying a recall cache.
    """

    system_prompt: str = ""
    grounding: tuple[Message, ...] = field(default_factory=tuple)
    schema_block: str | None = None

    def as_messages(self) -> list[Message]:
        """The prefix as a flat message list — pinned system prompt,
        followed by the grounding messages in order, followed (when
        present) by the structured-output schema block.

        Returns a new list each call so a caller mutating the result
        can't accidentally aliasing-corrupt the frozen prefix."""
        out: list[Message] = []
        if self.system_prompt:
            out.append(Message("system", self.system_prompt))
        out.extend(self.grounding)
        if self.schema_block is not None:
            out.append(Message("system", self.schema_block, name="schema"))
        return out


__all__ = ["PrefixContext"]
