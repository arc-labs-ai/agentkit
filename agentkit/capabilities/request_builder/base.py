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
from agentkit.capabilities.request_builder.grounder import _GROUNDING_NAME, Grounder
from agentkit.context import PrefixContext, WorkingContext
from agentkit.context.tokens import estimate_message_tokens
from agentkit.kernel._frozen import deep_freeze
from agentkit.kernel._json import dumps as _json_dumps
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message
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
        compactor: a `Compactor` to fold the transcript when it grows.
            Applied AFTER the new turn is appended, to the
            ``WorkingContext.messages`` tail. The cache-stable prefix
            is never touched — compactors must not pretend to. Kept-
            recent tail always includes the message the LLM is about
            to answer.
        reground_every_turn: when True, the grounder is invoked on
            every turn, and the previous grounding message is dropped
            before the new one is appended. The default (False)
            preserves the legacy first-turn-only behaviour, which
            keeps the prefix cacheable.
        budget_check: optional `(approx_tokens) -> None` callable
            invoked at the end of `build()`. Raise to abort — the
            RequestBuilder surfaces no judgement of its own about what's
            "too big," that policy lives entirely with the caller.
    """

    prompt: Prompt
    grounder: Grounder | None = None
    compactor: Compactor | None = None
    reground_every_turn: bool = False
    budget_check: BudgetCheck | None = None

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
                grounding_msgs: tuple[Message, ...] = ()
                if self.grounder is not None:
                    block = await self.grounder(ctx, task)
                    if block:
                        grounding_msgs = (
                            Message(
                                "system",
                                f"Relevant context:\n{block}",
                                name=_GROUNDING_NAME,
                            ),
                        )
                schema_block = (
                    _render_schema_block(output_adapter) if output_adapter is not None else None
                )
                wc.prefix = PrefixContext(
                    system_prompt=self.prompt.render(),
                    grounding=grounding_msgs,
                    schema_block=schema_block,
                )
                wc.append(Message("user", task))
            else:
                if self.reground_every_turn and self.grounder is not None:
                    # Re-render the prefix in place: same system
                    # prompt, fresh grounding block (or none). This
                    # IS a cache invalidation by design — the caller
                    # asked for live-refresh grounding. Per-turn
                    # callers that care about cache stability use the
                    # default ``reground_every_turn=False``.
                    block = await self.grounder(ctx, task)
                    grounding_msgs = (
                        (
                            Message(
                                "system",
                                f"Relevant context:\n{block}",
                                name=_GROUNDING_NAME,
                            ),
                        )
                        if block
                        else ()
                    )
                    wc.prefix = PrefixContext(
                        system_prompt=wc.prefix.system_prompt or self.prompt.render(),
                        grounding=grounding_msgs,
                        schema_block=wc.prefix.schema_block,
                    )
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
