"""The typed grounding seam — provenance that survives to the prompt.

The callable ``Grounder`` (``(ctx, task) -> str``) hands ``RequestBuilder`` a
string. By the time grounding reaches ``wc.prefix`` the item it came from is
gone: which source produced it, what it scored, whether it was a recorded
observation or a summary a model wrote in an earlier run. That last one is the
rule this file exists for — **a memory a model wrote is not evidence** — and it
is unenforceable once the memory is a string, because nothing downstream can
tell it from a recorded fact.

``grounding=`` takes a ``GroundingSource`` (``(ctx, task) -> Sequence[MemoryItem]``)
and keeps the items intact through two policy seams before the join:

    admit  — a per-item predicate; a rejected item never reaches the prompt
    render — how the admitted items become text

These tests pin, in order: the legacy ``grounder=`` path is untouched; the two
keywords cannot both be set; ``admit`` runs strictly before ``render``; a custom
``render`` is honoured; the DEFAULT render is byte-identical to what
``as_grounder`` produced, so migrating is provably behaviour-preserving; the
admitted items are readable after the run; and grounding still lands in
``wc.prefix`` and never in ``wc.messages`` — the KV-cache invariant that made
the prefix worth having.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any

import pytest

from agentkit.capabilities.request_builder import (
    GROUNDING_RECORD_KEY,
    RequestBuilder,
    render_grounding,
)
from agentkit.context import WorkingContext
from agentkit.memory import MemoryItem, as_grounder, as_grounding_source
from agentkit.prompts.prompt import Prompt
from agentkit.testing import FakeCtx, FakeGrounder, FakeMemory


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


SEED = Prompt(id="test.seed", version="seed-test-1", template="You are a test agent.")

# A recorded observation and a model-authored summary of a past run. The
# ``tier`` key is the application's own convention — the framework never reads
# it — which is the point: the item survives far enough for the application to.
OBSERVED = MemoryItem(
    content="Invoice 8812 was paid on 2026-03-04.",
    source="ledger",
    score=0.91,
    metadata={"tier": "observed"},
)
INFERRED = MemoryItem(
    content="The customer probably pays late.",
    source="run-summary",
    score=0.88,
    metadata={"tier": "inferred"},
)


def _source(*items: MemoryItem) -> Any:
    """A ``GroundingSource`` returning a fixed list — the smallest thing that
    satisfies the seam, so a failure points at RequestBuilder, not retrieval."""

    async def grounding(ctx: Any, task: str) -> list[MemoryItem]:
        del ctx, task
        return list(items)

    return grounding


# ── the legacy path is untouched ───────────────────────────────────────────


def test_legacy_grounder_keyword_still_works_and_warns_nothing() -> None:
    """``grounder=`` is not deprecated and must not behave as if it were.

    Every existing caller — including the auto-wired ``Agent(memory=...)``
    path — goes through this keyword. A ``DeprecationWarning`` here would fire
    under any suite that runs with ``-W error``, turning an additive change
    into a breaking one."""

    async def go() -> None:
        wc = WorkingContext()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            req = await RequestBuilder(prompt=SEED, grounder=FakeGrounder(block="[d] fact")).build(
                "q", wc, FakeCtx()
            )
        assert wc.prefix.grounding[0].content == "Relevant context:\n[d] fact"
        assert [m.role for m in req.messages] == ["system", "system", "user"]
        # No typed items were involved, so nothing is recorded.
        assert GROUNDING_RECORD_KEY not in wc.scratchpad

    _run(go())


def test_passing_both_grounder_and_grounding_is_refused_at_construction() -> None:
    """Two sources of truth for one slot, resolved silently, is the bug this
    repo avoids everywhere. Refuse at construction — not at ``build()``, which
    is a request into a live run, and not by precedence, which makes the
    ignored keyword invisible."""
    with pytest.raises(ValueError, match="grounder.*grounding"):
        RequestBuilder(prompt=SEED, grounder=FakeGrounder(), grounding=_source(OBSERVED))


def test_render_or_admit_without_a_grounding_source_is_refused() -> None:
    """``render``/``admit`` are policies over ITEMS; the string grounder has
    none to offer. Accepting them beside ``grounder=`` would silently drop an
    admission predicate the caller believes is filtering the prompt — the
    exact failure mode ``admit`` exists to prevent."""
    with pytest.raises(ValueError, match="admit"):
        RequestBuilder(prompt=SEED, grounder=FakeGrounder(), admit=lambda i: True)
    with pytest.raises(ValueError, match="render"):
        RequestBuilder(prompt=SEED, render=lambda items: "x")
    # Same reasoning for the audit trail: a caller who asked to record what the
    # run grounded on, and would have got an empty scratchpad forever.
    with pytest.raises(ValueError, match="record_grounding"):
        RequestBuilder(prompt=SEED, grounder=FakeGrounder(), record_grounding=True)


# ── the default render is behaviour-preserving ─────────────────────────────


def test_default_render_reproduces_as_grounder_byte_for_byte() -> None:
    """The migration proof. The same memory, reached through the old flattening
    adapter and the new typed one, must put the SAME bytes in the prefix —
    otherwise every prompt in every deployment shifts on upgrade and every
    cached prefix is invalidated at once."""

    async def go() -> None:
        mem = FakeMemory(items=[OBSERVED, INFERRED])
        old_wc = WorkingContext()
        await RequestBuilder(prompt=SEED, grounder=as_grounder(mem)).build("q", old_wc, FakeCtx())
        new_wc = WorkingContext()
        await RequestBuilder(prompt=SEED, grounding=as_grounding_source(mem)).build(
            "q", new_wc, FakeCtx()
        )
        assert new_wc.prefix.grounding == old_wc.prefix.grounding
        assert new_wc.assembled() == old_wc.assembled()

    _run(go())


def test_render_grounding_is_the_default_and_is_exported() -> None:
    """The default is a named function, not an inline lambda, so a caller who
    wants "the default plus one line" can compose with it instead of
    re-deriving the ``[source] content`` shape and drifting from it."""
    assert render_grounding([OBSERVED, INFERRED]) == (
        "[ledger] Invoice 8812 was paid on 2026-03-04.\n"
        "[run-summary] The customer probably pays late."
    )
    assert render_grounding([]) == ""


# ── admit runs before render ───────────────────────────────────────────────


def test_admit_runs_before_render_and_a_rejected_item_never_reaches_the_prompt() -> None:
    """The headline rule: a memory a model wrote is not evidence.

    ``admit`` must be applied to the items BEFORE they are joined, not to the
    rendered string afterwards — a post-hoc filter over text cannot tell which
    span came from which item. ``render`` is asserted to have never SEEN the
    rejected item, which is stronger than asserting it is absent from the
    output."""

    async def go() -> None:
        seen: list[list[MemoryItem]] = []

        def render(items: Any) -> str:
            seen.append(list(items))
            return " | ".join(i.content for i in items)

        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED,
            grounding=_source(OBSERVED, INFERRED),
            admit=lambda i: i.metadata.get("tier") != "inferred",
            render=render,
        ).build("q", wc, FakeCtx())
        assert seen == [[OBSERVED]]
        block = wc.prefix.grounding[0].content
        assert "Invoice 8812" in block
        assert "probably pays late" not in block
        assert all("probably pays late" not in m.content for m in wc.assembled())

    _run(go())


def test_admit_rejecting_everything_produces_no_grounding_message_at_all() -> None:
    """Not a blank section — no section.

    A "Relevant context:" header over nothing asserts evidence that does not
    exist, which is the authority inflation this whole seam is here to stop.
    ``render`` is not called either: a renderer that emits a fixed header would
    otherwise smuggle one in."""

    async def go() -> None:
        calls: list[int] = []

        def render(items: Any) -> str:
            calls.append(len(list(items)))
            return "EVIDENCE:\n"

        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED,
            grounding=_source(OBSERVED, INFERRED),
            admit=lambda i: False,
            render=render,
        ).build("q", wc, FakeCtx())
        assert wc.prefix.grounding == ()
        assert calls == []
        assert [m.role for m in wc.assembled()] == ["system", "user"]

    _run(go())


def test_admit_raising_aborts_the_build_rather_than_admitting() -> None:
    """An admission predicate that fails must not fail OPEN. Swallowing the
    exception and keeping the item is how a model-authored memory reaches the
    prompt with the authority of a recorded fact — the one outcome this seam
    exists to make impossible."""

    async def go() -> None:
        def admit(item: MemoryItem) -> bool:
            raise RuntimeError("policy backend down")

        wc = WorkingContext()
        with pytest.raises(RuntimeError, match="policy backend down"):
            await RequestBuilder(prompt=SEED, grounding=_source(OBSERVED), admit=admit).build(
                "q", wc, FakeCtx()
            )
        # The turn never landed: no half-built prefix, no orphan user message.
        assert wc.prefix.grounding == ()
        assert wc.messages == []

    _run(go())


# ── render is a policy ─────────────────────────────────────────────────────


def test_custom_render_is_a_policy_not_a_fixed_join() -> None:
    """The renderer sees scores and sources, so a caller can emit numbered
    citations the model can refer back to — impossible when the seam handed
    over a pre-joined string."""

    async def go() -> None:
        def render(items: Any) -> str:
            return "\n".join(
                f"[{n}] ({i.source}, score={i.score}) {i.content}"
                for n, i in enumerate(items, start=1)
            )

        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED, grounding=_source(OBSERVED, INFERRED), render=render
        ).build("q", wc, FakeCtx())
        assert wc.prefix.grounding[0].content == (
            "Relevant context:\n"
            "[1] (ledger, score=0.91) Invoice 8812 was paid on 2026-03-04.\n"
            "[2] (run-summary, score=0.88) The customer probably pays late."
        )

    _run(go())


def test_render_returning_an_empty_string_produces_no_grounding_message() -> None:
    """Same contract the string grounder already had: empty means "nothing to
    ground on this turn", not "an empty block"."""

    async def go() -> None:
        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED, grounding=_source(OBSERVED), render=lambda items: ""
        ).build("q", wc, FakeCtx())
        assert wc.prefix.grounding == ()

    _run(go())


def test_render_returning_a_non_string_is_refused_where_it_happened() -> None:
    """``Message.content`` is annotated ``str`` but not checked at runtime, so a
    renderer returning a list would sail into ``wc.prefix`` and surface as a
    JSON-serialisation error inside the provider adapter — several layers from
    the callable that caused it."""

    async def go() -> None:
        wc = WorkingContext()
        with pytest.raises(TypeError, match="render"):
            await RequestBuilder(
                prompt=SEED,
                grounding=_source(OBSERVED),
                render=lambda items: ["not", "a", "string"],  # type: ignore[return-value]
            ).build("q", wc, FakeCtx())

    _run(go())


# ── the cache-stable prefix invariant ──────────────────────────────────────


def test_grounding_lands_in_the_prefix_and_never_in_messages() -> None:
    """The invariant the typed seam must not spend. Grounding is a prefix
    concern: changing an early token throws away the provider's KV cache from
    that point on, so the retrieved block is pinned in the frozen head and the
    tail carries only per-turn churn."""

    async def go() -> None:
        wc = WorkingContext()
        builder = RequestBuilder(prompt=SEED, grounding=_source(OBSERVED, INFERRED))
        await builder.build("first", wc, FakeCtx())
        assert len(wc.prefix.grounding) == 1
        assert [m.role for m in wc.messages] == ["user"]
        assert all("Relevant context" not in m.content for m in wc.messages)
        prefix_before = wc.prefix
        await builder.build("second", wc, FakeCtx())
        # Bit-identical prefix across turns — the whole point.
        assert wc.prefix is prefix_before
        assert all("Relevant context" not in m.content for m in wc.messages)

    _run(go())


def test_reground_every_turn_rebuilds_the_typed_block_without_accumulating() -> None:
    """``reground_every_turn`` is a deliberate cache trade, and the typed seam
    inherits it unchanged: the grounding tuple is REBUILT each turn, so N turns
    cannot leave N stale blocks stacked in the prefix."""

    async def go() -> None:
        calls: list[str] = []

        async def grounding(ctx: Any, task: str) -> list[MemoryItem]:
            del ctx
            calls.append(task)
            return [MemoryItem(content=f"fact about {task}", source="s")]

        wc = WorkingContext()
        builder = RequestBuilder(prompt=SEED, grounding=grounding, reground_every_turn=True)
        await builder.build("first", wc, FakeCtx())
        await builder.build("second", wc, FakeCtx())
        assert calls == ["first", "second"]
        assert len(wc.prefix.grounding) == 1
        assert "fact about second" in wc.prefix.grounding[0].content
        assert "fact about first" not in wc.prefix.grounding[0].content

    _run(go())


# ── the items are recordable ───────────────────────────────────────────────


def test_recorded_items_are_readable_after_the_run() -> None:
    """A run should be able to say afterwards what it grounded ON, not only
    what it said. The record holds the ADMITTED items — the ones that actually
    reached the prompt — with their source, score and metadata intact, encoded
    as JSON-safe dicts so the trail survives the checkpoint it will be read
    after (see ``test_the_record_survives_a_json_serialising_checkpoint_store``)."""

    async def go() -> None:
        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED,
            grounding=_source(OBSERVED, INFERRED),
            admit=lambda i: i.metadata.get("tier") != "inferred",
            record_grounding=True,
        ).build("q", wc, FakeCtx())
        recorded = wc.scratchpad[GROUNDING_RECORD_KEY]
        assert recorded == (
            {
                "content": OBSERVED.content,
                "source": "ledger",
                "score": 0.91,
                "metadata": {"tier": "observed"},
            },
        )

    _run(go())


def test_recording_is_off_by_default() -> None:
    """Recording must not be mandatory work for a caller that does not want it
    — no scratchpad key appears unless it was asked for."""

    async def go() -> None:
        wc = WorkingContext()
        await RequestBuilder(prompt=SEED, grounding=_source(OBSERVED)).build("q", wc, FakeCtx())
        assert wc.scratchpad == {}

    _run(go())


def test_recording_an_empty_admission_still_records_the_emptiness() -> None:
    """"We grounded on nothing" is a fact worth recording. An absent key would
    be indistinguishable from "recording was off"."""

    async def go() -> None:
        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED,
            grounding=_source(OBSERVED),
            admit=lambda i: False,
            record_grounding=True,
        ).build("q", wc, FakeCtx())
        assert wc.scratchpad[GROUNDING_RECORD_KEY] == ()

    _run(go())


# ── edges of the source itself ─────────────────────────────────────────────


def test_a_source_returning_nothing_produces_no_grounding_message() -> None:
    """Empty index, wrong tenant scope, no hits — all the same shape, and all
    the same answer as the string grounder's empty block."""

    async def go() -> None:
        wc = WorkingContext()
        await RequestBuilder(prompt=SEED, grounding=_source()).build("q", wc, FakeCtx())
        assert wc.prefix.grounding == ()
        assert [m.role for m in wc.assembled()] == ["system", "user"]

    _run(go())


def test_a_source_that_raises_propagates() -> None:
    """RequestBuilder holds no judgement about retrieval failure. Swallowing it
    would silently answer an evidence-grounded question with no evidence; the
    caller who wired the source is the one who can decide to degrade."""

    async def go() -> None:
        async def boom(ctx: Any, task: str) -> list[MemoryItem]:
            raise TimeoutError("vector store timed out")

        with pytest.raises(TimeoutError):
            await RequestBuilder(prompt=SEED, grounding=boom).build(
                "q", WorkingContext(), FakeCtx()
            )

    _run(go())


def test_item_order_is_preserved_exactly_as_the_source_returned_it() -> None:
    """The source ranks; the builder does not re-rank. ``admit`` filters in
    place, so the renderer sees score order — which is what makes "cite [1]
    first" mean anything to the model."""

    async def go() -> None:
        items = [MemoryItem(content=f"c{n}", source="s", score=1.0 - n / 10) for n in range(5)]
        seen: list[list[MemoryItem]] = []

        def render(rendered: Any) -> str:
            seen.append(list(rendered))
            return "|".join(i.content for i in rendered)

        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED,
            grounding=_source(*items),
            admit=lambda i: i.content != "c2",
            render=render,
        ).build("q", wc, FakeCtx())
        assert [i.content for i in seen[0]] == ["c0", "c1", "c3", "c4"]
        assert wc.prefix.grounding[0].content.endswith("c0|c1|c3|c4")

    _run(go())


def test_identical_content_from_different_sources_both_survive_to_render() -> None:
    """Two backends holding the same passage are two provenances, not one fact.
    The builder must not collapse them — de-duplication is the composite
    memory's decision (where the surviving item can be stamped with what it
    absorbed), not a silent set() in prompt assembly."""

    async def go() -> None:
        a = MemoryItem(content="same text", source="ledger", score=0.9)
        b = MemoryItem(content="same text", source="wiki", score=0.7)
        wc = WorkingContext()
        await RequestBuilder(prompt=SEED, grounding=_source(a, b), record_grounding=True).build(
            "q", wc, FakeCtx()
        )
        assert [r["source"] for r in wc.scratchpad[GROUNDING_RECORD_KEY]] == ["ledger", "wiki"]
        assert wc.prefix.grounding[0].content == (
            "Relevant context:\n[ledger] same text\n[wiki] same text"
        )

    _run(go())


def test_a_large_item_set_is_not_silently_capped() -> None:
    """There is deliberately no cap here. A ceiling in prompt assembly would
    drop evidence the application's own ``admit`` had already approved, and it
    would do it invisibly. ``k`` on the source bounds retrieval and
    ``budget_check`` bounds the assembled request — both are the caller's."""

    async def go() -> None:
        items = [MemoryItem(content=f"fact-{n}", source="s") for n in range(500)]
        seen: list[int] = []
        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED,
            grounding=_source(*items),
            record_grounding=True,
            budget_check=seen.append,
        ).build("q", wc, FakeCtx())
        assert len(wc.scratchpad[GROUNDING_RECORD_KEY]) == 500
        assert "fact-499" in wc.prefix.grounding[0].content
        assert seen and seen[0] > 500

    _run(go())


# ── the memory adapter ─────────────────────────────────────────────────────


def test_as_grounding_source_does_not_flatten() -> None:
    """``as_grounder`` adapts a ``MemorySource`` by flattening to text;
    ``as_grounding_source`` adapts it by NOT flattening. Same ``k``/``where``
    binding at wiring time, one less lossy step."""

    async def go() -> None:
        mem = FakeMemory(items=[OBSERVED, INFERRED])
        source = as_grounding_source(mem, k=3, where={"tenant": "acme"})
        items = await source(FakeCtx(), "the task")
        assert list(items) == [OBSERVED, INFERRED]
        assert mem.queries == [("the task", 3, {"tenant": "acme"})]

    _run(go())


# ── review additions: the re-ground / record interaction, and the wire ──────
#
# The four tests below were added in review. Each kills a mutant the shipped
# suite let live: the first three because ``reground_every_turn`` and
# ``record_grounding`` were never exercised TOGETHER, and the last because the
# record was never taken anywhere durable.


def test_reground_refreshes_the_record_rather_than_leaving_turn_ones_behind() -> None:
    """The record must describe the prefix AS IT STANDS, not as it first stood.

    ``reground_every_turn`` rebuilds ``PrefixContext.grounding`` from this
    turn's retrieval, so a record still holding turn 1's items would claim the
    run grounded on evidence that is no longer in the prompt — the audit trail
    telling the opposite of the transcript, which is worse than no audit trail.

    Kills two mutants the shipped suite let live: dropping the ``_record`` call
    from the re-ground branch, and writing the record with ``setdefault``
    (first-write-wins) instead of ``note`` (last-write-wins)."""

    async def go() -> None:
        turn = {"n": 0}

        async def grounding(ctx: Any, task: str) -> list[MemoryItem]:
            del ctx, task
            turn["n"] += 1
            return [MemoryItem(content=f"evidence-{turn['n']}", source="s")]

        wc = WorkingContext()
        builder = RequestBuilder(
            prompt=SEED,
            grounding=grounding,
            reground_every_turn=True,
            record_grounding=True,
        )
        await builder.build("first", wc, FakeCtx())
        assert [r["content"] for r in wc.scratchpad[GROUNDING_RECORD_KEY]] == ["evidence-1"]
        await builder.build("second", wc, FakeCtx())
        # One entry, this turn's — not two, and not turn one's.
        assert [r["content"] for r in wc.scratchpad[GROUNDING_RECORD_KEY]] == ["evidence-2"]
        assert "evidence-2" in wc.prefix.grounding[0].content

    _run(go())


def test_reground_drops_the_stale_block_when_this_turn_admits_nothing() -> None:
    """The sharpest version of the staleness rule, and the one the shipped
    suite missed: on turn 2 the source returns something the application's
    ``admit`` refuses, so there is nothing to say. The prefix must go EMPTY.

    Keeping turn 1's block would leave the prompt asserting evidence retrieved
    for a different question — and would do it precisely when the application
    just said "none of this may be used", which is the authority inflation the
    typed seam exists to stop."""

    async def go() -> None:
        turn = {"n": 0}

        async def grounding(ctx: Any, task: str) -> list[MemoryItem]:
            del ctx, task
            turn["n"] += 1
            return [OBSERVED] if turn["n"] == 1 else [INFERRED]

        wc = WorkingContext()
        builder = RequestBuilder(
            prompt=SEED,
            grounding=grounding,
            admit=lambda i: i.metadata.get("tier") != "inferred",
            reground_every_turn=True,
            record_grounding=True,
        )
        await builder.build("first", wc, FakeCtx())
        assert len(wc.prefix.grounding) == 1
        await builder.build("second", wc, FakeCtx())
        assert wc.prefix.grounding == ()
        assert all("Invoice 8812" not in m.content for m in wc.assembled())
        assert wc.scratchpad[GROUNDING_RECORD_KEY] == ()

    _run(go())


def test_both_seams_set_after_construction_is_refused_at_build_not_resolved() -> None:
    """``RequestBuilder`` is a plain mutable dataclass, so the state
    ``__post_init__`` refuses is one assignment away from existing anyway.

    Before this was closed, ``builder.grounding = source`` on a
    ``grounder=``-wired builder resolved silently in the grounder's favour:
    observed on the shipped code, an ``admit=lambda i: False`` set alongside it
    was dropped and every item reached the prompt. A constructor-time-only
    check is not an invariant."""

    async def go() -> None:
        builder = RequestBuilder(prompt=SEED, grounder=FakeGrounder(block="[d] flattened"))
        builder.grounding = _source(OBSERVED)
        wc = WorkingContext()
        with pytest.raises(ValueError, match="grounder.*grounding"):
            await builder.build("q", wc, FakeCtx())
        assert wc.prefix.grounding == ()
        assert wc.messages == []

    _run(go())


def test_the_record_survives_a_json_serialising_checkpoint_store() -> None:
    """"Durable state is encoded, not stored raw" — the rule the agents guide
    states, and the one this record broke.

    ``ReActCognition._save`` copies ``wc.scratchpad`` verbatim into the
    checkpoint blob and ``PostgresCheckpointStore.save`` ``json.dumps`` it.
    Recording live ``MemoryItem``s therefore tested green on
    ``InMemoryCheckpointStore`` and raised ``TypeError: Object of type
    MemoryItem is not JSON serializable`` on the store anyone deploys — the
    same shape ``PlanPolicy`` once shipped with ``Step``.

    Round-tripping matters as much as dumping: the record is read AFTER a run,
    which for a suspended run means after a resume, so a shape that survives
    the write but not the read is no better."""

    async def go() -> None:
        wc = WorkingContext()
        await RequestBuilder(
            prompt=SEED,
            grounding=_source(OBSERVED, INFERRED),
            record_grounding=True,
        ).build("q", wc, FakeCtx())

        # Exactly what the Postgres checkpoint store does to the state blob.
        wire = json.dumps({"scratchpad": wc.scratchpad})
        restored = json.loads(wire)["scratchpad"][GROUNDING_RECORD_KEY]

        assert [r["source"] for r in restored] == ["ledger", "run-summary"]
        assert [r["score"] for r in restored] == [0.91, 0.88]
        assert restored[0]["metadata"] == {"tier": "observed"}
        assert restored[1]["content"] == INFERRED.content

    _run(go())
