"""The two grounding seams ``RequestBuilder`` accepts — one flattened, one typed.

``Grounder`` returns TEXT. ``GroundingSource`` returns the ``MemoryItem``s the
text would have been made of, and lets the caller inspect, refuse and render
them. Both are supported; neither is deprecated, because the flattened one is
genuinely simpler when the grounding is a static fixture or an MCP prompt with
no provenance to preserve in the first place.

Why the typed one had to exist: by the time a ``Grounder``'s string reaches
``wc.prefix`` the item is gone — which source produced it, what it scored,
whether it was a recorded observation or a summary a MODEL wrote in an earlier
run. That last distinction is the one that matters, because **a memory a model
wrote is not evidence**, and once it is a string in the prefix nothing
downstream can tell it from a recorded fact. Any claim built on it inherits an
authority it never had. ``GroundingSource`` + ``admit`` is the only place that
rule can be enforced: after retrieval (so the item exists) and before the join
(so refusing it actually keeps it out of the prompt).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from agentkit.kernel.protocols import Ctx

# A TYPE-level dependency on ``agentkit.memory``, not a behavioural one: the
# builder reads ``.source`` and ``.content`` in the default renderer and
# otherwise passes items through untouched. ``agentkit.memory`` imports only
# ``agentkit.kernel``, so this direction cannot cycle — checked before adding
# it, because the reverse (memory importing capabilities) would.
from agentkit.memory.base import MemoryItem

Grounder = Callable[[Ctx, str], Awaitable[str]]
"""A grounder is anything async-callable as `(ctx, task) -> str`.

Why a callable and not a `MemorySource` plus k/where knobs: those knobs
are retrieval mechanics, not prompt-assembly concerns. RequestBuilder should
not know that the grounding came from a vector store at all — only
that *some* text is available for this task. The caller bakes the
retrieval policy (k, where filter, query derivation, even the choice
of source — vector index, MCP server, static fixtures) in once at
wiring time, then hands the bound callable to the RequestBuilder.

A `VectorMemory` (or any `MemorySource`) is adapted to a `Grounder`
by writing a small async wrapper that queries it and formats the
returned `MemoryItem`s:

    async def grounder(ctx, task):
        items = await mem.query(task, k=5, ctx=ctx)
        return "\\n".join(f"[{i.source}] {i.content}" for i in items)
    builder = RequestBuilder(prompt=..., grounder=grounder)

For role-specific tuning, the wrapper stays adjacent to the agent
definition without leaking into RequestBuilder's signature.

When the flattening in that wrapper is throwing away something the
application needs to act on, reach for `GroundingSource` instead — it
is the same seam with the items left intact."""


GroundingSource = Callable[[Ctx, str], Awaitable[Sequence[MemoryItem]]]
"""The typed grounding seam: async-callable as `(ctx, task) -> Sequence[MemoryItem]`.

Identical in spirit to `Grounder` — the caller still bakes the retrieval
policy (k, where, query derivation, which backend) in at wiring time and
RequestBuilder still treats the callable as opaque. The only difference is
what comes back, and that difference buys three things:

1. **The items are inspectable before rendering.** `RequestBuilder.admit`
   is a per-item predicate that runs before the join, so an application can
   refuse an item outright — most usefully the model-authored kind::

       admit=lambda item: item.metadata.get("tier") != "inferred"

   Filtering the rendered STRING instead cannot work: once joined, nothing
   identifies which span came from which item.

2. **The rendering is a policy.** `RequestBuilder.render` receives the
   admitted items, so numbered citations, score-annotated lines or a JSON
   block are a one-liner rather than a rewrite of the retrieval wrapper.

3. **The items can be recorded beside the prefix.** With
   `record_grounding=True` the admitted items land on
   `WorkingContext.scratchpad[GROUNDING_RECORD_KEY]`, so a finished run can
   say what it grounded ON, not only what it said. They land as JSON-safe
   `{"content", "source", "score", "metadata"}` dicts rather than live
   `MemoryItem`s, because the scratchpad is copied into the checkpoint blob
   and a record that cannot cross a resume is not an audit trail.

`agentkit.memory.as_grounding_source` adapts any `MemorySource` to this by
*not* flattening — strictly less code than `as_grounder`, which is the
usual sign the seam is at the right level."""


GroundingRender = Callable[[Sequence[MemoryItem]], str]
"""How admitted items become the text of the grounding block.

A pure, synchronous function on purpose. Rendering is string assembly over
data already in hand; making it async would invite a second retrieval round
trip into the middle of prompt assembly, where a failure has no honest
recovery and the cost is invisible to the budget pre-check."""


GroundingAdmit = Callable[[MemoryItem], bool]
"""Per-item admission predicate. `False` keeps the item out of the prompt
entirely — not merely out of the rendered text.

Deliberately one-item-at-a-time rather than a `Sequence -> Sequence` filter:
a whole-list hook is also a reordering and re-scoring hook, and ranking
belongs to the memory source (or its `Reranker`), which is the only layer
that knows what the scores mean. Keeping this shape means the order the
source returned is the order `render` sees."""


def render_grounding(items: Sequence[MemoryItem]) -> str:
    """The default rendering: one `[source] content` line per item.

    Byte-identical to what `agentkit.memory.as_grounder` has always produced,
    and that is the requirement rather than a coincidence — the flattened and
    typed seams have to put the SAME bytes in the prefix for the same items,
    or upgrading a deployment silently shifts every grounded prompt and
    invalidates every cached prefix at once.

    Sources stay bracket-prefixed because that attribution is the cheapest
    form of the provenance this module exists to preserve: the model can cite
    it, and a human reading the transcript can see which backend a claim came
    from. A bare `"\\n\\n".join(content)` reads more cleanly and throws that
    away.

    Exported as a named function rather than left inline so a caller wanting
    "the default, plus a header" composes with it instead of re-deriving the
    shape and drifting from it.
    """
    return "\n".join(f"[{i.source}] {i.content}" for i in items)


# The `WorkingContext.scratchpad` key under which `record_grounding=True`
# stores the admitted items. A module constant because it is a read/write
# rendezvous between the builder and the caller's post-run reporting — a
# literal on either side would be a typo away from a silently empty audit.
#
# NOTE the value is an unnamespaced, very ordinary word, so a caller who was
# already noting something of their own under "grounding" loses it on the
# first grounded turn. The framework's other reserved scratchpad key is
# namespaced (`SECRET_TAINT_KEY = "_agentkit_secret_taint"`); this one should
# probably follow, but the value is public surface and changing it is the
# integrator's call, not a review's.
GROUNDING_RECORD_KEY = "grounding"


# The `name=` value stamped on a grounding system message. It is a
# LABEL, not a matching key: `reground_every_turn=True` drops the stale
# block by rebuilding `PrefixContext.grounding` wholesale, so nothing in
# the framework ever looks this string up. It is a module constant so a
# downstream caller introspecting an assembled transcript can pick the
# grounding message out by the same name the builder wrote.
_GROUNDING_NAME = "grounding"


__all__ = [
    "GROUNDING_RECORD_KEY",
    "Grounder",
    "GroundingAdmit",
    "GroundingRender",
    "GroundingSource",
    "render_grounding",
]
