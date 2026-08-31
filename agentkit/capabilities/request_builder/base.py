"""RequestBuilder — the single seam where the four engineering disciplines meet.

The four disciplines are **prompt** engineering (what wording the model
sees), **memory** engineering (what past knowledge gets retrieved into
that wording), **context** engineering (how the transcript is bounded
when it grows), and **execution** engineering (how the request is
attributed, budgeted, and traced). Without this seam, every agent
would inline its own version of:

    1. render the system prompt
    2. (maybe) pull grounding from a retriever
    3. append the user task
    4. (maybe) compact the transcript
    5. send

That duplication is a source of small drifts — one agent would stamp
the prompt version, another would not; one would retrieve on every
turn, another only on the first. As prompts and memory grow in
importance, those small drifts become the primary failure modes.

`RequestBuilder` is the one place that owns steps 1–4. The LLM call itself
(step 5) stays where it belongs: behind the Invoker's middleware
chain. Callers build a `RequestBuilder` once, then call `build()` on
every turn — keeping the per-turn delta minimal and identical across
agents.

What the RequestBuilder does NOT do: it does not call the LLM, it does not
mutate budget counters (the Invoker does that authoritatively), and it
does not pick policies. The caller still chooses *which* compactor,
*which* retriever, and *how* to filter retrievals — RequestBuilder just
applies those choices in the right order, in one place.

Cache discipline. After the ``agentkit.context`` split, the system
prompt and grounding live in ``WorkingContext.prefix`` (frozen, cache-
stable). The user task and continuation turns land on
``WorkingContext.messages`` (the mutable tail). Compaction touches
ONLY the tail — the prefix is never rewritten mid-run, matching the
KV-cache discipline that says any token mutated in the prefix
invalidates the cache from that point onward.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentkit.capabilities.compaction import Compactor
from agentkit.capabilities.output_schema import SchemaAdapter
from agentkit.capabilities.request_builder.grounder import (
    _GROUNDING_NAME,
    GROUNDING_RECORD_KEY,
    Grounder,
    GroundingAdmit,
    GroundingRender,
    GroundingSource,
    render_grounding,
)
from agentkit.context import PrefixContext, WorkingContext
from agentkit.context.tokens import estimate_message_tokens
from agentkit.kernel._frozen import deep_freeze
from agentkit.kernel._json import dumps as _json_dumps
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message
from agentkit.memory.base import MemoryItem
from agentkit.prompts.prompt import Prompt

BudgetCheck = Callable[[int], None]
"""A budget pre-check hook: `(approx_tokens) -> None`. Raise to abort
`build()` before the messages are returned. The RequestBuilder does not
prescribe an exception type — the caller chooses what `BudgetExceeded`,
`PolicyDenied`, or similar looks like for their system."""


@dataclass(frozen=True)
class BuiltRequest:
    """The output of `RequestBuilder.build()`. Carries the messages ready
    to send AND the cross-cutting signals the caller's invoker and
    telemetry need:

    - `prompt_version` lets the caller stamp the agent's output for
      attribution (so a regression maps back to the exact template).
    - `approx_tokens` is a crude pre-call estimate the caller can use
      for budget pre-checks (the *authoritative* count comes back from
      the provider in `Usage`; this is just a guardrail).
    """

    messages: list[Message]
    prompt_version: str
    approx_tokens: int

    def __post_init__(self) -> None:
        """Frozen in name only until this ran: the assembled transcript could be appended to
        through a field advertised as immutable, after the token count and budget pre-check had
        already been computed from it."""
        object.__setattr__(self, "messages", deep_freeze(self.messages))

    def __hash__(self) -> int:
        """Identity is the prompt version and size. `messages` is excluded for two
        reasons: it is unhashable (see `_frozen.py`), and it grows without bound
        across a run — `len` is the O(1) discriminator that keeps successive turns
        off a single bucket."""
        return hash((self.prompt_version, self.approx_tokens, len(self.messages)))


@dataclass
class RequestBuilder:
    """Assembles the LLM input. One instance per agent role per run is
    typical — the seed prompt and grounding policy don't change inside
    a run, only the task and the growing transcript do.

    Required:
        prompt: the versioned seed system prompt (an
            `agentkit.prompts.Prompt`).

    Optional:
        grounder: an async callable `(ctx, task) -> str` injected on
            the first turn (or every turn if `reground_every_turn`).
            Empty string means "no grounding for this task" and
            produces no grounding message. The caller owns the policy
            (k, where, query derivation) — RequestBuilder just calls.
            "Turn" throughout means *one call to `build()`*, which is
            one user turn of a conversation — NOT one step of a tool
            loop. A `ReActCognition.drive()` calls `build()` exactly
            once and then appends tool traffic to the tail directly,
            so a 4-step ReAct run invokes the grounder once whatever
            `reground_every_turn` says (measured: 4 LLM calls, 1
            grounder call). The flag only bites when a caller reuses
            one `WorkingContext` across successive `agent.run(...)`
            calls.
        compactor: a `Compactor` to fold the transcript when it grows.
            Applied AFTER the new turn is appended, to the
            ``WorkingContext.messages`` tail. The cache-stable prefix
            is never touched — compactors must not pretend to. Kept-
            recent tail always includes the message the LLM is about
            to answer.
        reground_every_turn: whether grounding is retrieved once or
            refreshed on every turn. This is a genuine trade, not a
            compatibility shim — both settings are wrong for some
            workload, so the knob stays and the caller picks.

            False (the default) retrieves once, on the first turn, and
            pins that block for the life of the context. Turn 5 is
            answered with the evidence retrieved for turn 1's
            question. Measured on a 5-turn conversation over a
            5-fact handbook at k=1: 1 of 5 turns saw the fact that
            actually answered it, and the prefix was bit-identical on
            5 of 5 turns.

            True re-invokes the grounder each turn and rebuilds
            ``PrefixContext.grounding`` with the result, so the block
            tracks the current question — 4 of 5 turns saw their
            answering fact in the same probe (the fifth was a
            retrieval miss, not a staleness one). The price is that
            rewriting the prefix invalidates the provider's KV cache
            every turn: 1 of 5 prefixes matched turn 1. On a
            4,000-token grounded prefix over a 20-turn conversation
            with claude-sonnet-4-6, that is $0.2280 of fresh prefix
            reads against $0.0228 of cache reads — 10x on the prefix
            term, every turn, forever.

            Rule of thumb: leave it False when the grounding is a
            corpus the whole conversation shares (a handbook, a spec,
            a codebase digest) and True when each turn asks about
            something different and the retrieved evidence is the
            answer. The default is False because the auto-wired
            ``Agent(memory=...)`` path constructs this builder with no
            way to set the flag, so the default is what every
            auto-wired agent gets — and silently paying 10x on the
            prefix is a worse failure to hand someone by default than
            grounding they can see is stale in the transcript.
        budget_check: optional `(approx_tokens) -> None` callable
            invoked at the end of `build()`. Raise to abort — the
            RequestBuilder surfaces no judgement of its own about what's
            "too big," that policy lives entirely with the caller.
        grounding: the TYPED alternative to ``grounder`` — an async
            callable ``(ctx, task) -> Sequence[MemoryItem]``. Same seam,
            same place in the turn, same landing spot in the prefix; the
            difference is that the items arrive intact, so ``admit`` can
            refuse one and ``render`` can see its source and score.
            Mutually exclusive with ``grounder`` (see ``__post_init__``).
        admit: optional per-item predicate applied to what ``grounding``
            returned, BEFORE ``render``. A rejected item does not reach
            the prompt at all. The rule worth having it for is
            ``lambda i: i.metadata.get("tier") != "inferred"`` — a memory
            a model wrote is not evidence, and this is the only point in
            the pipeline where that is still checkable.
        render: optional ``Sequence[MemoryItem] -> str`` deciding how the
            admitted items become the grounding block. Defaults to
            ``render_grounding`` (``[source] content`` lines), which is
            byte-identical to what ``as_grounder`` produced.
        record_grounding: when True, the admitted items are stored on
            ``wc.scratchpad[GROUNDING_RECORD_KEY]`` so a finished run can
            report what it grounded ON, not only what it said. Off by
            default — an audit trail nobody reads is still a write into
            the caller's blackboard on every turn. Recorded as a tuple of
            JSON-safe ``{"content", "source", "score", "metadata"}`` dicts,
            not as live ``MemoryItem``s, because the scratchpad is durable
            state — see ``_record``. Rewritten (not appended to) on every
            re-ground, so it always describes the prefix as it stands.
    """

    prompt: Prompt
    grounder: Grounder | None = None
    compactor: Compactor | None = None
    reground_every_turn: bool = False
    budget_check: BudgetCheck | None = None
    # Appended AFTER ``budget_check`` rather than slotted next to ``grounder``
    # where they read better: the fields above are positional for anyone who
    # built a ``RequestBuilder(prompt, grounder, compactor)`` without keywords,
    # and re-ordering them would silently rebind those arguments.
    grounding: GroundingSource | None = None
    admit: GroundingAdmit | None = None
    render: GroundingRender | None = None
    record_grounding: bool = False

    def __post_init__(self) -> None:
        """Refuse the wirings that would silently do nothing.

        Two sources of truth for one slot is the failure this repo avoids
        everywhere, and grounding is the slot where it is least visible: both
        keywords produce a plausible prefix, so a run wired with both would
        look right while half the configuration was ignored. Refusing at
        CONSTRUCTION rather than resolving by precedence is deliberate — a
        precedence rule makes the ignored keyword invisible, and ``build()``
        is too late because by then a run is already in flight.

        ``admit`` / ``render`` / ``record_grounding`` are refused without
        ``grounding`` for the sharper version of the same reason. They are
        policies over ITEMS, and the string grounder has none; accepting them
        beside ``grounder=`` would mean an application believes it is refusing
        model-authored memories while every one of them reaches the prompt.
        That is not a cosmetic no-op, it is the exact failure ``admit`` exists
        to prevent, so it is worth a hard error rather than a warning.
        """
        if self.grounder is not None and self.grounding is not None:
            raise ValueError(
                "RequestBuilder got both grounder= and grounding= — they fill the same "
                "slot and would resolve silently. Pass the typed grounding= source (its "
                "items can be admitted, rendered and recorded), or the flattened "
                "grounder= callable, not both."
            )
        if self.grounding is None:
            dangling = [
                name
                for name, wired in (
                    ("admit", self.admit is not None),
                    ("render", self.render is not None),
                    ("record_grounding", self.record_grounding),
                )
                if wired
            ]
            if dangling:
                raise ValueError(
                    f"RequestBuilder got {', '.join(f'{n}=' for n in dangling)} without "
                    "grounding= — these are policies over MemoryItems and there are none "
                    "to apply them to. A grounder= callable has already flattened its "
                    "items to text; pass grounding= instead."
                )

    async def build(
        self,
        task: str,
        wc: WorkingContext,
        ctx: Ctx,
        *,
        output_adapter: SchemaAdapter[Any] | None = None,
    ) -> BuiltRequest:
        """Append a turn to ``wc`` and return the messages to send.

        First turn (empty prefix AND empty messages): the system
        prompt + (optional) grounding land in ``wc.prefix`` — the
        cache-stable head — and the user task is appended to
        ``wc.messages``. Subsequent turns: just append the user task
        (and re-ground if ``reground_every_turn``).

        ``output_adapter``: when supplied, the adapter's JSON Schema is
        rendered into ``wc.prefix.schema_block`` on the first turn — a
        cache-stable third pinned system message after the prompt and
        grounding. Never written into ``wc.messages`` (cache invariant).

        Mutates ``wc`` in place so the caller's running blackboard
        stays the source of truth for the transcript. The returned
        ``messages`` list is a fresh copy of ``wc.assembled()`` —
        safe to hand to the invoker without worrying about post-call
        mutation racing the next turn.
        """
        first_turn = not wc.messages and not wc.prefix.system_prompt and not wc.prefix.grounding

        with ctx.trace.span(
            "compose",
            "compose",
            **{"agentkit.prompt.version": self.prompt.version},
        ):
            if first_turn:
                # Lay down the cache-stable prefix exactly once: the
                # system prompt, then (when wired in) the grounding
                # block. Replacing the prefix in place rather than
                # appending to ``wc.messages`` is the cache discipline
                # — the prefix is frozen after this point, so the
                # provider's KV cache stays valid across turns.
                grounding_msgs, admitted = await self._ground(ctx, task)
                schema_block = (
                    _render_schema_block(output_adapter) if output_adapter is not None else None
                )
                wc.prefix = PrefixContext(
                    system_prompt=self.prompt.render(),
                    grounding=grounding_msgs,
                    schema_block=schema_block,
                )
                self._record(wc, admitted)
                wc.append(Message("user", task))
            else:
                if self.reground_every_turn and (
                    self.grounder is not None or self.grounding is not None
                ):
                    # Re-render the prefix in place: same system
                    # prompt, fresh grounding block (or none). The
                    # stale block is dropped because the whole
                    # ``grounding`` tuple is rebuilt from this turn's
                    # retrieval — there is no find-and-replace, and
                    # ``_GROUNDING_NAME`` is a label for the reader,
                    # not a matching key. Rebuilding rather than
                    # appending is what stops N turns accumulating N
                    # grounding blocks.
                    #
                    # This IS a cache invalidation by design — the
                    # caller asked for live-refresh grounding, and it
                    # costs 10x on the prefix term (measured: $0.2280
                    # vs $0.0228 for a 4,000-token prefix over 20
                    # turns of claude-sonnet-4-6). Callers who want
                    # the cache keep the default False and accept
                    # turn-1 evidence for the whole conversation.
                    grounding_msgs, admitted = await self._ground(ctx, task)
                    wc.prefix = PrefixContext(
                        system_prompt=wc.prefix.system_prompt or self.prompt.render(),
                        grounding=grounding_msgs,
                        schema_block=wc.prefix.schema_block,
                    )
                    # The record tracks the PREFIX, so it is overwritten rather
                    # than appended: the prefix now holds this turn's evidence
                    # only, and a record that accumulated would claim the run
                    # grounded on blocks that are no longer in the prompt.
                    self._record(wc, admitted)
                if task:
                    wc.append(Message("user", task))

            if self.compactor is not None:
                # Cache discipline: ONLY the mutable tail is folded.
                # The compactor's view is exactly what the caller's
                # historical contract was — a flat list of messages
                # to summarise — but limited to the tail so the
                # prefix stays bit-identical across turns.
                wc.messages = await self.compactor.compact(wc.messages, ctx)

            assembled = wc.assembled()
            approx = _approx_tokens(assembled)
            if self.budget_check is not None:
                # Caller's seam — raises here are intentional and propagate to the caller.
                self.budget_check(approx)

        return BuiltRequest(
            messages=assembled,
            prompt_version=self.prompt.version,
            approx_tokens=approx,
        )

    async def _ground(
        self, ctx: Ctx, task: str
    ) -> tuple[tuple[Message, ...], tuple[MemoryItem, ...] | None]:
        """Run whichever grounding seam is wired and return the prefix messages
        it produced, plus the admitted items when there were any to keep.

        One helper for both call sites (first turn, and every turn under
        ``reground_every_turn``) because the two used to be near-duplicate
        blocks that had already drifted once — the re-ground branch grew the
        "rebuild, don't append" comment while the first-turn branch did not.
        A second copy of the admit/render ordering is exactly the kind of drift
        that would let a rejected item into the prompt on turn 3 only.

        The second element is ``None`` — not ``()`` — when the flattened
        ``grounder`` ran: it means "nothing typed passed through here", which a
        caller must be able to tell from "the typed source admitted nothing".
        Only the latter is worth recording.

        The both-seams refusal is repeated here rather than left to
        ``__post_init__`` because ``RequestBuilder`` is a plain mutable
        dataclass: assigning ``builder.grounding = ...`` onto a builder wired
        with ``grounder=`` re-creates exactly the state the constructor refuses,
        and the branch below would have resolved it silently in the grounder's
        favour — dropping an ``admit`` predicate the caller believes is keeping
        model-authored memories out of the prompt. Constructor-time-only is not
        an invariant, it is a hope.
        """
        if self.grounder is not None and self.grounding is not None:
            raise ValueError(
                "RequestBuilder has both grounder= and grounding= set at build time — "
                "they fill the same slot and one would be resolved away silently. This "
                "state is refused at construction, so it can only come from assigning "
                "onto the builder afterwards; assign None to the one you do not want."
            )
        if self.grounder is not None:
            return self._as_messages(await self.grounder(ctx, task)), None
        if self.grounding is None:
            return (), None

        retrieved = await self.grounding(ctx, task)
        # ``admit`` is applied here, to ITEMS, and never to the rendered
        # string: once joined, nothing identifies which span came from which
        # item, so a text-level filter cannot enforce a per-item rule at all.
        # A predicate that raises propagates — failing OPEN would admit the
        # item, and the item this seam exists to refuse is a model-authored
        # memory being passed off as evidence.
        admitted = (
            tuple(retrieved)
            if self.admit is None
            else tuple(item for item in retrieved if self.admit(item))
        )
        if not admitted:
            # No admitted items means no grounding message — not an empty one,
            # and ``render`` is not consulted. A renderer that emits a fixed
            # "Relevant context:" header would otherwise pin a block asserting
            # evidence that does not exist, which is the authority inflation
            # this seam is here to stop. It also collapses three causes —
            # empty index, everything rejected, source returned [] — onto the
            # one behaviour the string grounder already had for an empty block.
            return (), admitted
        renderer = self.render if self.render is not None else render_grounding
        block = renderer(admitted)
        if not isinstance(block, str):
            # ``Message.content`` is annotated ``str`` and checked by nothing at
            # runtime, so a renderer returning a list would land in the frozen
            # prefix and surface as a serialisation error inside the provider
            # adapter — several layers and one await away from the callable
            # that caused it. One isinstance per turn buys the traceback.
            raise TypeError(
                f"RequestBuilder.render returned {type(block).__name__}, expected str — "
                "the grounding block is written straight into a system Message."
            )
        return self._as_messages(block), admitted

    @staticmethod
    def _as_messages(block: str) -> tuple[Message, ...]:
        """A non-empty block becomes exactly one pinned system message; an empty
        one becomes no message at all.

        Empty-means-absent is the contract both seams share, and it is why the
        typed path can short-circuit before ``render``: a caller reading a
        transcript never has to distinguish "grounded on nothing" from
        "grounding was not wired"."""
        if not block:
            return ()
        return (Message("system", f"Relevant context:\n{block}", name=_GROUNDING_NAME),)

    def _record(self, wc: WorkingContext, admitted: tuple[MemoryItem, ...] | None) -> None:
        """Stamp the admitted items onto the scratchpad, when asked.

        Gated on ``record_grounding`` so it is not mandatory work: a run that
        does not want an audit trail pays one attribute read, and nothing
        appears in the caller's blackboard that they did not ask for.

        ``admitted is None`` (the flattened ``grounder`` path) writes nothing
        even when the flag is set — but that combination is already refused in
        ``__post_init__``, so this is belt-and-braces against someone assigning
        ``builder.grounder = ...`` after construction rather than a reachable
        branch.

        Written as JSON-safe dicts, not as the live ``MemoryItem``s, because
        ``WorkingContext.scratchpad`` is durable state: ``ReActCognition._save``
        copies it verbatim into the checkpoint blob and
        ``PostgresCheckpointStore.save`` ``json.dumps`` that blob. Recording the
        dataclass tested green on ``InMemoryCheckpointStore`` and raised
        ``TypeError: Object of type MemoryItem is not JSON serializable`` on the
        Postgres one — the same failure ``PlanPolicy`` shipped once with ``Step``
        (see "Durable state is encoded, not stored raw" in the agents guide).
        Encoding is also what makes the record survive the resume it exists for:
        an audit trail that cannot cross a checkpoint is not an audit trail.

        ``metadata`` is passed through as the caller's own mapping rather than
        coerced. Whatever a backend put in there is governed by the same
        scratchpad rule as any other value the caller notes; what this method
        owes is that the FRAMEWORK's own shape is not the thing that breaks."""
        if not self.record_grounding or admitted is None:
            return
        # Recorded even when empty: "this run grounded on nothing" is a fact a
        # postmortem needs, and an absent key would be indistinguishable from
        # recording having been switched off.
        wc.note(GROUNDING_RECORD_KEY, tuple(_encode_item(i) for i in admitted))


def _encode_item(item: MemoryItem) -> dict[str, Any]:
    """One admitted ``MemoryItem`` as the JSON-safe record of it.

    Every field the typed seam exists to preserve is here — ``content``,
    ``source``, ``score``, ``metadata`` — so nothing about the provenance is
    lost by encoding; only the class is. ``dict(...)`` on the metadata un-aliases
    the item's frozen mapping so the recorded snapshot cannot be re-frozen or
    re-annotated through the record, and so the value that lands in the
    checkpoint blob is a plain ``dict`` rather than a ``FrozenDict`` subclass a
    decoder would have to know about.
    """
    return {
        "content": item.content,
        "source": item.source,
        "score": item.score,
        "metadata": dict(item.metadata),
    }


def _approx_tokens(messages: list[Message]) -> int:
    """Crude pre-call estimate, delegating to the ONE shared implementation.

    This used to be its own ``sum(len(m.content)) // 4``, with a docstring
    claiming it "matches the heuristic the compaction strategies use". It did,
    until that heuristic learned to count tool calls — and then it silently did
    not, which is the precise divergence a budget pre-check must never have: a
    caller pre-checks with this number and the compactor decides with another.
    Measured on the shared version before consolidation: a transcript of 80
    tool-call messages estimated at 20 tokens against a real ~81,000.

    One implementation, so the two cannot drift again.
    """
    return estimate_message_tokens(messages)


def _render_schema_block(adapter: SchemaAdapter[Any]) -> str:
    """Canonical prompt-injection form for the output schema.

    Lives in the cache-stable prefix (via ``PrefixContext.schema_block``),
    so it stays bit-identical across turns and the provider's KV cache
    keeps hitting. The two-line header before the schema is the
    "least-surprise" wording every adapter shares — adapters reach the
    Agent in different Python flavours but the model only ever sees the
    JSON Schema and this fixed preamble, so behaviour is uniform across
    Pydantic / dataclass / attrs / raw-schema callers.
    """
    schema = adapter.json_schema()
    return (
        f"Respond with JSON matching this schema named {adapter.name!r}. "
        "No prose, no code fences.\n\n"
        f"{_json_dumps(schema)}"
    )


__all__ = ["BudgetCheck", "BuiltRequest", "RequestBuilder"]
