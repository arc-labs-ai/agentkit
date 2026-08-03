"""Framework-injected grounding (the get_context() step, ch13): an Agent wired with a
`RequestBuilder(grounder=...)` auto-injects retrieved context as a system grounding message
before the model call on the first turn. Deterministic via a stub grounder + a FakeLLM that
echoes the system prompt.

The OLD test API was `Agent(memory=retriever, memory_k=K)`. The grounding seam is now
`RequestBuilder(grounder=<async (ctx, query) -> str>)`, so these tests build a RequestBuilder
with a stub grounder and assert the same semantic claim: the grounded text appears in the
system block the LLM sees.
"""

import asyncio

from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import RequestBuilder
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Scope
from agentkit.prompts import Prompt
from agentkit.testing import FakeLLM, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


def _echo_system(*, system, user, model):
    return system  # res.output reveals what grounding was injected


def _stub_grounder(fact: str):
    async def grounder(ctx: Ctx, query: str) -> str:
        return f"{fact} (q={query})"

    return grounder


async def _empty_grounder(ctx: Ctx, query: str) -> str:
    return ""


def _prompt(text: str) -> Prompt:
    return Prompt(id="t", version="test", template=text)


def test_agent_injects_retrieved_context():
    builder = RequestBuilder(prompt=_prompt("base"), grounder=_stub_grounder("REDIS is a cache"))
    a = Agent(name="a", model="m", request_builder=builder)
    out = _run(
        a.run(
            "what is redis",
            make_test_ctx(llm=FakeLLM(_echo_system), scope=Scope(1, 2), correlation_id="r"),
        )
    ).output
    # The system the LLM sees is base prompt + grounding block. _flatten joins them by \n\n.
    assert "base" in out
    assert "REDIS is a cache" in out
    assert "q=what is redis" in out
    assert "Relevant context:" in out  # RequestBuilder wraps the grounded text in this prefix


def test_empty_retrieval_injects_nothing():
    builder = RequestBuilder(prompt=_prompt("base"), grounder=_empty_grounder)
    out = _run(
        Agent(name="a", model="m", request_builder=builder).run(
            "hi", make_test_ctx(llm=FakeLLM(_echo_system), scope=Scope(1, 2), correlation_id="r")
        )
    ).output
    # When the grounder returns "", no grounding system message is appended — the LLM sees
    # only the base prompt.
    assert out == "base"
    assert "Relevant context:" not in out


def test_tool_loop_agent_injects_on_fresh_start():
    """Same claim, but on a tool-loop Agent (`tools=[...]`). RequestBuilder-based grounding
    runs identically across the single-call and tool-loop regimes — they are the same `Agent`
    now."""
    from agentkit.tools import FunctionTool

    async def _noop(args, ctx):
        return "ok"

    tool = FunctionTool(
        name="noop",
        fn=_noop,
        description="A no-op tool used only to push Agent down the tool-loop code path.",
        side_effecting=False,
    )
    builder = RequestBuilder(prompt=_prompt("base"), grounder=_stub_grounder("FACT"))
    a = Agent(name="a", model="m", request_builder=builder, cognition=ReActCognition(tools=[tool]))
    out = _run(
        a.run("q", make_test_ctx(llm=FakeLLM(_echo_system), scope=Scope(1, 2), correlation_id="r"))
    ).output
    assert "FACT" in out and "base" in out
