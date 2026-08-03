"""RequestBuilder — the input-side seam where prompt + grounding + transcript + compaction
meet. These tests pin the contract:

    1. First turn appends system + grounding + user; subsequent turns append only user.
    2. The grounder callable is invoked with (ctx, task) — RequestBuilder treats it as
       opaque, no leaky `k` / `where` / `with_source` knobs in RequestBuilder's own
       signature.
    3. An empty grounding string does not add a stray system message.
    4. The compactor is invoked after the new turn is appended (not before — the kept
       tail must include the message the LLM is about to answer).
    5. BuiltRequest.prompt_version comes from the seed Prompt, for attribution.
    6. `reground_every_turn=True` re-invokes the grounder each turn and replaces the
       previous grounding system message in place (matched by `name="grounding"`).
    7. `budget_check(approx_tokens)` is invoked at the end and can raise to abort.

The tests use small async stubs rather than the real Retriever/Compactor so a failure
points at RequestBuilder's policy, not at an upstream capability's behavior.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.capabilities.request_builder import BuiltRequest, RequestBuilder
from agentkit.context import PrefixContext, WorkingContext
from agentkit.kernel.types import Message
from agentkit.prompts.prompt import Prompt
from agentkit.testing import FakeCompactor, FakeCtx, FakeGrounder


def _run(coro):
    return asyncio.run(coro)


SEED = Prompt(id="test.seed", version="seed-test-1", template="You are a test agent.")


# ── first-turn assembly ────────────────────────────────────────────────────


def test_first_turn_appends_system_then_user_without_grounder():
    """Minimum config: prompt only. First turn lays down the system
    prompt in ``wc.prefix`` (the cache-stable head) and the user task
    in ``wc.messages`` (the per-turn tail). The assembled view is the
    legacy ``[system, user]`` shape."""

    async def go() -> None:
        wc = WorkingContext()
        req = await RequestBuilder(prompt=SEED).build("hello", wc, FakeCtx())
        assert wc.prefix.system_prompt == "You are a test agent."
        assert wc.prefix.grounding == ()
        assert [m.role for m in wc.messages] == ["user"]
        assert wc.messages[0].content == "hello"
        assert [m.role for m in req.messages] == ["system", "user"]
        assert req.messages == wc.assembled()
        assert req.prompt_version == "seed-test-1"

    _run(go())


def test_first_turn_appends_system_grounding_user_with_grounder():
    """A wired-in grounder injects a grounding system message into
    ``wc.prefix.grounding`` alongside the system prompt. The
    assembled view orders them as [system_prompt, grounding, user] —
    three messages, in that order — preserving the legacy provider-
    facing shape."""

    async def go() -> None:
        wc = WorkingContext()
        grounder = FakeGrounder(block="[doc1] pricing fact")
        req = await RequestBuilder(prompt=SEED, grounder=grounder).build("price?", wc, FakeCtx())
        assert wc.prefix.system_prompt == "You are a test agent."
        assert len(wc.prefix.grounding) == 1
        assert wc.prefix.grounding[0].content.startswith("Relevant context:\n[doc1]")
        assert [m.role for m in wc.messages] == ["user"]
        assert wc.messages[0].content == "price?"
        assembled = req.messages
        assert [m.role for m in assembled] == ["system", "system", "user"]
        assert assembled[1].content.startswith("Relevant context:\n[doc1]")
        assert assembled[2].content == "price?"
        assert req.approx_tokens > 0

    _run(go())


def test_empty_grounding_does_not_add_message():
    """When the grounder returns "" (e.g. wrong tenant scope or empty
    index), the RequestBuilder does NOT add a placeholder grounding
    message — the prefix's grounding tuple stays empty."""

    async def go() -> None:
        wc = WorkingContext()
        await RequestBuilder(prompt=SEED, grounder=FakeGrounder(block="")).build(
            "anything", wc, FakeCtx()
        )
        assert wc.prefix.grounding == ()
        assert [m.role for m in wc.assembled()] == ["system", "user"]

    _run(go())


# ── subsequent turns ───────────────────────────────────────────────────────


def test_subsequent_turn_only_appends_user_task():
    """The system prompt is NOT re-rendered on continuation turns —
    that would corrupt the cached prefix and waste tokens. The grounder
    is NOT called again either (grounding is a first-turn injection by
    default). The prefix stays bit-identical; only the tail grows."""

    async def go() -> None:
        wc = WorkingContext()
        grounder = FakeGrounder(block="ignored")
        builder = RequestBuilder(prompt=SEED, grounder=grounder)
        await builder.build("first", wc, FakeCtx())
        first_call_count = len(grounder.calls)
        prefix_before = wc.prefix
        len_after_first = len(wc.messages)
        await builder.build("second", wc, FakeCtx())
        assert wc.prefix is prefix_before  # cache-stable
        assert len(wc.messages) == len_after_first + 1
        assert wc.messages[-1].role == "user"
        assert wc.messages[-1].content == "second"
        assert len(grounder.calls) == first_call_count  # grounder not re-invoked

    _run(go())


def test_subsequent_turn_with_empty_task_appends_nothing():
    """An empty task on a continuation turn is a no-op (the caller is
    asking the model to keep going, not to take a new instruction)."""

    async def go() -> None:
        wc = WorkingContext(
            prefix=PrefixContext(system_prompt="sys"),
            messages=[Message("user", "u")],
        )
        before_tail = list(wc.messages)
        before_prefix = wc.prefix
        await RequestBuilder(prompt=SEED).build("", wc, FakeCtx())
        assert wc.messages == before_tail
        assert wc.prefix is before_prefix

    _run(go())


# ── grounder is opaque to RequestBuilder ───────────────────────────────────


def test_grounder_receives_ctx_and_task_only():
    """RequestBuilder should call the grounder with exactly (ctx, task) — no
    extra knobs. The whole point of the callable-grounder design is
    that retrieval mechanics (k, where, etc.) stay inside the grounder
    where the caller bound them at wiring time."""

    async def go() -> None:
        ctx = FakeCtx()
        grounder = FakeGrounder(block="block")
        await RequestBuilder(prompt=SEED, grounder=grounder).build(
            "the task", WorkingContext(), ctx
        )
        assert grounder.calls == [(ctx, "the task")]

    _run(go())


def test_grounder_can_be_a_plain_async_function():
    """The grounder is a structural callable — no Protocol subclass
    required. A bare async function works just as well as a class with
    `__call__`, which keeps wire-up simple at the call site."""

    async def go() -> None:
        captured: list[str] = []

        async def my_grounder(ctx: Any, task: str) -> str:
            captured.append(task)
            return f"derived-from:{task}"

        wc = WorkingContext()
        await RequestBuilder(prompt=SEED, grounder=my_grounder).build("Q", wc, FakeCtx())
        assert captured == ["Q"]
        # The grounding lands in the prefix; assemble to see the full view.
        assert any("derived-from:Q" in m.content for m in wc.assembled())

    _run(go())


# ── compaction is the last step ────────────────────────────────────────────


def test_compactor_runs_after_new_turn_is_appended():
    """Compaction is applied to the tail AFTER the new turn lands —
    the cache-stable prefix is NEVER touched (that would invalidate
    the provider's KV cache from the prefix forward). The kept tail
    still includes the message the LLM is about to answer."""

    async def go() -> None:
        wc = WorkingContext()
        compactor = FakeCompactor(sentinel="[FOLDED]")
        req = await RequestBuilder(prompt=SEED, compactor=compactor).build("hello", wc, FakeCtx())
        assert compactor.called is True
        # Tail was folded to the sentinel; prefix untouched.
        assert [m.content for m in wc.messages] == ["[FOLDED]"]
        assert wc.prefix.system_prompt == "You are a test agent."
        # The assembled view the provider sees: prefix + folded tail.
        assert req.messages == [
            Message("system", "You are a test agent."),
            Message("system", "[FOLDED]"),
        ]

    _run(go())


# ── attribution / cost signals ─────────────────────────────────────────────


def test_built_request_carries_prompt_version_for_attribution():
    """The returned BuiltRequest stamps the seed Prompt's version
    so the caller can attribute the agent's output back to the exact
    template that produced it."""

    async def go() -> None:
        seed = Prompt(id="custom", version="2026-06-18-v2", template="...")
        req = await RequestBuilder(prompt=seed).build("x", WorkingContext(), FakeCtx())
        assert req.prompt_version == "2026-06-18-v2"
        assert isinstance(req, BuiltRequest)

    _run(go())


def test_build_records_trace_span_with_prompt_version_attr():
    """Trace spans stamp the prompt version as an attribute. This is
    what makes a compaction-quality regression mappable back to the
    bad template version."""

    async def go() -> None:
        ctx = FakeCtx()
        await RequestBuilder(prompt=SEED).build("hi", WorkingContext(), ctx)
        names = [name for name, _kind, _attrs in ctx.spans]
        assert "compose" in names
        compose_span = next(attrs for name, _kind, attrs in ctx.spans if name == "compose")
        assert compose_span["agentkit.prompt.version"] == "seed-test-1"

    _run(go())


# ── reground_every_turn: live-refresh grounding ───────────────────────────


def test_reground_every_turn_invokes_grounder_on_each_turn():
    """With `reground_every_turn=True`, the grounder is invoked on
    every `build()` call — the grounding message is regenerated to
    reflect the current task / scratchpad. This is the right knob when
    grounding depends on the latest turn."""

    async def go() -> None:
        wc = WorkingContext()
        grounder = FakeGrounder(block="ground:once")
        builder = RequestBuilder(prompt=SEED, grounder=grounder, reground_every_turn=True)
        await builder.build("first", wc, FakeCtx())
        first_calls = len(grounder.calls)
        await builder.build("second", wc, FakeCtx())
        assert len(grounder.calls) == first_calls + 1
        # The grounder is called with the new task on the second turn
        assert grounder.calls[-1][1] == "second"

    _run(go())


def test_reground_every_turn_replaces_previous_grounding_message():
    """The previous grounding system message is dropped from the
    prefix and the new one takes its place. The cache-stable prefix
    is INTENTIONALLY invalidated here — that's what the caller
    requested by setting ``reground_every_turn=True``. The transcript
    must NOT accumulate stale grounding blocks turn after turn."""

    async def go() -> None:
        wc = WorkingContext()

        @dataclass
        class _Sequence:
            blocks: list[str]
            calls: int = 0

            async def __call__(self, ctx: Any, task: str) -> str:
                block = self.blocks[self.calls]
                self.calls += 1
                return block

        grounder = _Sequence(blocks=["[v1] old fact", "[v2] new fact"])
        builder = RequestBuilder(prompt=SEED, grounder=grounder, reground_every_turn=True)
        await builder.build("first", wc, FakeCtx())
        await builder.build("second", wc, FakeCtx())

        # Exactly ONE grounding message remains — the latest — in the prefix.
        assert len(wc.prefix.grounding) == 1
        assert "[v2] new fact" in wc.prefix.grounding[0].content
        # Stale grounding text never leaks into the assembled view either.
        assembled = wc.assembled()
        assert all("[v1] old fact" not in m.content for m in assembled)

    _run(go())


def test_reground_every_turn_false_keeps_legacy_first_turn_only_behavior():
    """The default `reground_every_turn=False` matches the legacy semantics:
    grounder is called only on the first turn, and the grounding message persists
    untouched on continuation turns (keeping the prompt prefix cacheable)."""

    async def go() -> None:
        wc = WorkingContext()
        grounder = FakeGrounder(block="ground:initial")
        builder = RequestBuilder(prompt=SEED, grounder=grounder)  # default: False
        await builder.build("first", wc, FakeCtx())
        await builder.build("second", wc, FakeCtx())
        assert len(grounder.calls) == 1  # only called once

    _run(go())


# ── budget_check: pre-call abort hook ─────────────────────────────────────


def test_budget_check_is_invoked_with_approx_tokens():
    """The hook is called with the same `approx_tokens` value that
    appears on the returned `BuiltRequest`, so the caller's policy
    sees exactly what the invoker is about to send."""

    async def go() -> None:
        seen: list[int] = []

        def check(n: int) -> None:
            seen.append(n)

        req = await RequestBuilder(prompt=SEED, budget_check=check).build(
            "hello world", WorkingContext(), FakeCtx()
        )
        assert seen == [req.approx_tokens]
        assert req.approx_tokens > 0

    _run(go())


def test_budget_check_raise_aborts_build():
    """A raise inside `budget_check` propagates out of `build()` —
    RequestBuilder does not swallow it. This is the seam by which a caller
    enforces a hard pre-call budget cap without RequestBuilder having to
    know about budgets."""

    class BudgetExceeded(Exception):
        pass

    async def go() -> None:
        def check(n: int) -> None:
            raise BudgetExceeded(f"too big: {n}")

        with pytest.raises(BudgetExceeded):
            await RequestBuilder(prompt=SEED, budget_check=check).build(
                "hello world", WorkingContext(), FakeCtx()
            )

    _run(go())


def test_budget_check_none_is_a_noop():
    """No `budget_check` set → RequestBuilder behaves exactly as before."""

    async def go() -> None:
        await RequestBuilder(prompt=SEED).build("hi", WorkingContext(), FakeCtx())

    _run(go())  # no exception — no policy was installed
