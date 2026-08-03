"""TerminationCondition (ch18 / R11): per-turn delta evaluation, structured Stop, |/& composition,
reset, injectable clock, async predicate. Offline & deterministic."""

import asyncio

from agentkit.agents.control.termination import (
    ExternalTermination,
    FunctionalTermination,
    FunctionCall,
    MaxMessages,
    MaxTurns,
    SourceMatch,
    Stop,
    TextMention,
    Timeout,
    _Composite,
)
from agentkit.kernel.types import Message, ToolCall


def run(coro):
    return asyncio.run(coro)


def msg(role="assistant", content="", name=None, tool_calls=()):
    # Use keyword args defensively — Message field order has drifted before, and a positional
    # call here silently routed `tool_calls` into `name` (and vice versa), masking the actual
    # termination logic under "wrong field" symptoms.
    return Message(role=role, content=content, name=name, tool_calls=tuple(tool_calls))


def test_max_messages_counts_delta_then_terminates_and_resets():
    async def go():
        c = MaxMessages(3)
        assert await c([msg(), msg()]) is None and c.terminated is False
        stop = await c([msg()])
        assert (
            isinstance(stop, Stop)
            and stop.reason == "max_messages"
            and stop.detail == {"count": 3, "max": 3}
        )
        assert c.terminated
        assert (await c([msg()])).reason == "max_messages"  # stays terminated until reset
        await c.reset()
        assert c.terminated is False and c.count == 0

    run(go())


def test_max_turns():
    async def go():
        c = MaxTurns(2)
        assert await c([]) is None
        assert (await c([])).reason == "max_turns"

    run(go())


def test_text_mention_scoped_to_sources():
    async def go():
        c = TextMention("APPROVE", sources=["critic"])
        assert await c([msg(name="writer", content="APPROVE")]) is None  # wrong source ignored
        assert (await c([msg(name="critic", content="we APPROVE this")])).reason == "text_mention"

    run(go())


def test_function_call():
    async def go():
        c = FunctionCall("deploy")
        assert await c([msg(tool_calls=[ToolCall("1", "fetch", {})])]) is None
        assert (await c([msg(tool_calls=[ToolCall("2", "deploy", {})])])).reason == "function_call"

    run(go())


def test_source_match():
    async def go():
        assert (await SourceMatch("critic")([msg(name="critic")])).reason == "source_match"

    run(go())


def test_or_stops_if_any():
    async def go():
        c = MaxMessages(10) | TextMention("DONE")
        assert (await c([msg(content="DONE")])).reason == "text_mention"

    run(go())


def test_and_stops_only_when_all_across_turns():
    async def go():
        c = MaxTurns(2) & TextMention("DONE")
        assert await c([msg(content="DONE")]) is None  # text hit, but turns < 2 → not yet
        stop = await c([msg(content="x")])  # turn 2 reached; both now terminated
        assert (
            stop is not None
            and stop.reason == "all"
            and set(stop.detail["reasons"]) == {"max_turns", "text_mention"}
        )

    run(go())


def test_or_self_flattens():
    async def go():
        c = (MaxMessages(5) | TextMention("A")) | TextMention("B")
        assert isinstance(c, _Composite) and c.mode == "any" and len(c.conditions) == 3

    run(go())


def test_timeout_with_injectable_clock():
    async def go():
        now = [0.0]
        c = Timeout(5.0, clock=lambda: now[0])
        assert await c([]) is None
        now[0] = 6.0
        assert (await c([])).reason == "timeout"

    run(go())


def test_external_set():
    async def go():
        c = ExternalTermination()
        assert await c([]) is None
        c.set()
        assert (await c([])).reason == "external"

    run(go())


def test_functional_async_predicate():
    async def go():
        async def judge(delta):
            return any("ship" in (m.content or "") for m in delta)

        c = FunctionalTermination(judge, reason="judge_done")
        assert await c([msg(content="not yet")]) is None
        assert (await c([msg(content="ship it")])).reason == "judge_done"

    run(go())
