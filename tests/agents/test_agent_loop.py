"""Agent tool-loop (the ReAct branch) end-to-end over a real Invoker + middleware chains.

`Agent(tools=...)` is the same pattern as a `tools=None` single-call agent, just with the
loop unrolled. These tests pin the loop-specific behaviour: max_iterations, structured-output
repair, durable resume, suspend/resume on approval, termination conditions, cancellation,
autonomy gating."""

import asyncio
import json

import pytest

from agentkit.adapters.store import InMemoryStore
from agentkit.agents import Agent, Suspended
from agentkit.agents.cognition import ReActCognition
from agentkit.agents.control.termination import FunctionCall, TextMention
from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.types import Scope, ToolCall
from agentkit.middlewares import meter, tracing
from agentkit.runtime import Budget, MeterExceeded
from agentkit.testing import FakeLLM, Turn, make_test_ctx
from agentkit.testing import RecordingTracer as Tracer
from agentkit.tools import FunctionTool, ToolRegistry


def _run(coro):
    return asyncio.run(coro)


def _registry(counter=None):
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            "fetch",
            lambda a, c: (counter and counter.__setitem__("n", counter["n"] + 1)) or {"s": 200},
            description="fetch a resource over the network for inspection",
            side_effecting=False,
        )
    )
    return reg


def test_two_step_tool_loop_metered_per_chat():
    llm = FakeLLM.script(
        [Turn(tool_calls=(ToolCall("c1", "fetch", {"url": "u"}),)), Turn(content='{"v": "ok"}')]
    )
    ctx = make_test_ctx(
        llm=llm,
        chat_middleware=[tracing(), meter()],
        tool_middleware=[tracing()],
        budget=Budget(max_cost_usd=10.0),
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="run-1",
    )
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry()), prompt="go")
    res = _run(loop.run("html", ctx))
    assert res.output == '{"v": "ok"}' and not res.partial
    assert llm.calls == 2 and ctx.budget.calls == 2  # meter middleware charged each chat


def test_multiple_tool_calls_in_one_turn_all_dispatched():
    """Model returns two tool calls in a single turn (a legitimate
    provider capability, and something models occasionally double-up
    on). Both must dispatch, both results must land in the transcript
    in order, and the loop continues cleanly to a final answer."""
    call_count = {"n": 0}
    counter = call_count  # counted by the _registry fake

    # Turn 1: two tool calls in one assistant message.
    # Turn 2: a final answer after both results come back.
    llm = FakeLLM.script(
        [
            Turn(
                tool_calls=(
                    ToolCall("c1", "fetch", {"url": "u1"}),
                    ToolCall("c2", "fetch", {"url": "u2"}),
                )
            ),
            Turn(content="done both"),
        ]
    )
    ctx = make_test_ctx(
        llm=llm,
        chat_middleware=[tracing(), meter()],
        tool_middleware=[tracing()],
        budget=Budget(max_cost_usd=10.0),
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="run-multi",
    )
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry(counter)))
    res = _run(loop.run("go", ctx))

    # Both fetches ran.
    assert call_count["n"] == 2
    # Two chat turns: one that emitted the two tool_calls, one that
    # produced the final answer after both tool_result messages
    # returned. Every tool dispatch does NOT count as a chat turn —
    # the meter charges chat calls, and there were exactly two.
    assert llm.calls == 2
    assert res.output == "done both"


def test_duplicate_tool_calls_in_one_turn_both_dispatched_by_default():
    """Model doubles up on the same tool with identical arguments in a
    single turn — two ToolCalls with the same ``name`` and same
    ``arguments``, distinct ids. Without ``idempotent()`` middleware
    wired, both dispatch (the default: no dedup). Regression check
    that agentkit doesn't implicitly dedupe on ``arguments`` — that
    behaviour is opt-in via ``idempotent()``."""
    call_count = {"n": 0}
    counter = call_count

    llm = FakeLLM.script(
        [
            Turn(
                tool_calls=(
                    ToolCall("c1", "fetch", {"url": "same-url"}),
                    ToolCall("c2", "fetch", {"url": "same-url"}),  # identical args
                )
            ),
            Turn(content="ok"),
        ]
    )
    ctx = make_test_ctx(
        llm=llm,
        chat_middleware=[tracing(), meter()],
        tool_middleware=[tracing()],  # NO idempotent — opt-in only
        budget=Budget(max_cost_usd=10.0),
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="run-dup",
    )
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry(counter)))
    res = _run(loop.run("go", ctx))

    # Two dispatches — no implicit dedup. If the caller wants
    # dedup, they wire ``idempotent()`` into ``tool_middleware``.
    assert call_count["n"] == 2
    assert res.output == "ok"


def test_tool_raising_cancelled_error_propagates_and_stops_the_run():
    """``_invoke_tool_safe`` catches every ``Exception`` and reflects
    the failure back to the model. But ``BaseException`` subclasses
    (``CancelledError``, ``KeyboardInterrupt``, ``SystemExit``) are
    NOT caught — cancellation and shutdown signals must continue to
    propagate so the run actually terminates instead of the model
    seeing "tool cancelled, retry?" and re-invoking a cancelled
    subsystem."""

    class _CancellingTool:
        name = "self_cancel"
        description = "a tool that raises CancelledError to simulate a cancelled sub-operation"
        schema = None
        output_schema = None
        side_effecting = False
        requires_approval = False

        async def run(self, args, ctx):
            raise asyncio.CancelledError("sub-operation was cancelled")

    reg = ToolRegistry()
    reg.register(_CancellingTool())

    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "self_cancel", {}),)),
            Turn(content="unreachable"),
        ]
    )

    with pytest.raises(asyncio.CancelledError):
        _run(
            Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).run(
                "t",
                make_test_ctx(
                    llm=llm,
                    chat_middleware=[tracing(), meter()],
                    tool_middleware=[tracing()],
                    budget=Budget(max_cost_usd=10.0),
                    trace=Tracer(),
                    scope=Scope(1, 2),
                    correlation_id="run-cancelling-tool",
                ),
            )
        )


def test_tool_raising_regular_exception_is_reflected_to_model():
    """Sanity companion: any regular ``Exception`` DOES get caught by
    ``_invoke_tool_safe`` and reflected back to the model as an
    ``ERROR:`` string, so the loop keeps running and the model can
    retry / pivot / give up. Only ``BaseException`` escapes."""

    class _BuggyTool:
        name = "buggy"
        description = "a tool that raises a normal exception to simulate a network error"
        schema = None
        output_schema = None
        side_effecting = False
        requires_approval = False

        async def run(self, args, ctx):
            raise RuntimeError("upstream flaked")

    reg = ToolRegistry()
    reg.register(_BuggyTool())

    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "buggy", {}),)),
            Turn(content="model saw the error and gave up"),
        ]
    )

    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-buggy-tool",
            ),
        )
    )

    # Run completed — the loop didn't crash on the tool's raise.
    assert res.output == "model saw the error and gave up"


def test_max_iterations_returns_partial():
    # `repeat_last=True` is the "never stops" script: one tool-call turn replayed
    # forever so the ceiling, not the script, is what ends the run. Without it a
    # 1-turn script now raises ScriptExhausted on call 2 — which is the point of
    # the flag, since a script running dry is otherwise the bug being hidden.
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "fetch", {}),))], repeat_last=True)
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry(), max_iterations=3))
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.partial and res.evals["stop_reason"] == "max_iterations" and llm.calls == 3


def test_max_iterations_partial_returns_last_assistant_text_not_tool_result():
    # ceiling hit right after a tool round — output must be the last assistant text, not the tool result
    llm = FakeLLM.script(
        [Turn(content="working on it", tool_calls=(ToolCall("c1", "fetch", {}),))],
        repeat_last=True,  # deliberate never-terminating script; max_iterations ends it
    )
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry(), max_iterations=2))
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.partial and res.evals["stop_reason"] == "max_iterations"
    assert res.output == "working on it"


def test_structured_output_repairs_once():
    llm = FakeLLM.script([Turn(content="not json"), Turn(content='{"ok": 1}')])
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry()), parse=json.loads)
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == '{"ok": 1}' and not res.partial and llm.calls == 2


def test_stream_event_order():
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "fetch", {}),)), Turn(content="done")])
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry()))

    async def _types():
        return [
            ev.type
            async for ev in loop.stream(
                "t",
                make_test_ctx(
                    llm=llm,
                    chat_middleware=[tracing(), meter()],
                    tool_middleware=[tracing()],
                    budget=Budget(max_cost_usd=10.0),
                    trace=Tracer(),
                    scope=Scope(1, 2),
                    correlation_id="run-1",
                ),
            )
        ]

    assert _run(_types()) == ["tool_call", "tool_result", "step", "message_delta", "final"]


def test_spans_nest_invoke_agent_chat_tool():
    spans = []
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "fetch", {}),)), Turn(content="done")])
    loop = Agent(name="analyzer", model="gemini/x", cognition=ReActCognition(tools=_registry()))
    _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(exporter=spans.append),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    names = [s.name for s in spans]
    assert "invoke_agent" in names
    assert names.count("chat") == 2
    assert "execute_tool" in names
    invoke = next(s for s in spans if s.name == "invoke_agent")
    assert invoke.attrs.get("gen_ai.agent.name") == "analyzer"
    chats = [s for s in spans if s.name == "chat"]
    assert all(s.attrs.get("gen_ai.request.model") == "gemini/x" for s in chats)
    tool_span = next(s for s in spans if s.name == "execute_tool")
    assert tool_span.attrs.get("gen_ai.tool.name") == "fetch"


def test_budget_exhaustion_mid_loop():
    from agentkit.kernel.types import Usage

    llm = FakeLLM.script(
        [Turn(tool_calls=(ToolCall("c1", "fetch", {}),), usage=Usage(0, 0, 0.01))],
        repeat_last=True,  # loops until the budget trips, not until the script ends
    )
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry()))
    with pytest.raises(MeterExceeded):
        _run(
            loop.run(
                "t",
                make_test_ctx(
                    llm=llm,
                    chat_middleware=[tracing(), meter()],
                    tool_middleware=[tracing()],
                    budget=Budget(max_cost_usd=0.015),
                    trace=Tracer(),
                    scope=Scope(1, 2),
                    correlation_id="run-1",
                ),
            )
        )


def test_durable_resume_skips_completed_steps():
    store = InMemoryStore()
    counter = {"n": 0}

    llm1 = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "fetch", {}),)), Turn(content="final")])
    loop1 = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry(counter)))

    async def _crash():
        async for ev in loop1.stream(
            "t",
            make_test_ctx(
                llm=llm1,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        ):
            if ev.type == "step":
                break

    _run(_crash())
    assert counter["n"] == 1 and llm1.calls == 1

    llm2 = FakeLLM.script([Turn(content="final")])
    loop2 = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry(counter)))
    res = _run(
        loop2.run(
            "t",
            make_test_ctx(
                llm=llm2,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == "final" and llm2.calls == 1 and counter["n"] == 1  # fetch not re-run


def test_checkpoint_is_cleared_on_terminal_completion():
    """Regression: a completed run leaves no checkpoint, so a later resume/re-stream of the same run_id
    can't replay a finished run (and storage is reclaimed)."""
    from agentkit.capabilities.checkpointer import ckpt_key

    store = InMemoryStore()
    counter = {"n": 0}
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "fetch", {}),)), Turn(content="final")])
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=_registry(counter))).run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == "final"
    assert _run(store.get(ckpt_key("run-1"))) is None  # checkpoint reclaimed at terminal


def test_approval_suspends_then_resume_executes():
    store = InMemoryStore()
    counter = {"n": 0}
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            "delete",
            lambda a, c: counter.__setitem__("n", counter["n"] + 1) or "gone",
            description="delete a resource by id; mutates the world (destructive)",
            side_effecting=True,
            requires_approval=True,
        )
    )

    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "delete", {}),)), Turn(content="done")])
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=reg))
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.partial and isinstance(res.evals["suspended"], Suspended) and counter["n"] == 0

    llm2 = FakeLLM.script([Turn(content="done")])
    loop2 = Agent(name="a", model="m", cognition=ReActCognition(tools=reg))
    res2 = _run(
        loop2.resume(
            "run-1",
            {"c1": "approve"},
            make_test_ctx(
                llm=llm2,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res2.output == "done" and counter["n"] == 1


def test_resume_reject_does_not_execute():
    store = InMemoryStore()
    counter = {"n": 0}
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            "delete",
            lambda a, c: counter.__setitem__("n", counter["n"] + 1) or "gone",
            description="delete a resource by id; mutates the world (destructive)",
            side_effecting=True,
            requires_approval=True,
        )
    )
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "delete", {}),)), Turn(content="done")])
    _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )

    llm2 = FakeLLM.script([Turn(content="ack")])
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).resume(
            "run-1",
            {"c1": "reject"},
            make_test_ctx(
                llm=llm2,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == "ack" and counter["n"] == 0


# ── Resume with mismatched / partial decision keys ──────────────────────────
#
# The resume contract: for each pending ToolCall, look up its id in
# ``decisions``. Missing id → default to "reject" (fail-closed).
# Extra ids in ``decisions`` → silently ignored (harmless noise).
# This is the safe default: an operator who submits a stale approval
# set never accidentally auto-approves.


def _one_pending_delete_run(store, correlation_id: str = "run-1"):
    """Drive an Agent to a suspended state with one pending delete call."""
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            "delete",
            lambda a, c: "gone",
            description="delete a resource by id; mutates the world (destructive)",
            side_effecting=True,
            requires_approval=True,
        )
    )
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "delete", {}),)), Turn(content="done")])
    _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id=correlation_id,
            ),
        )
    )
    return reg


def test_resume_with_unknown_decision_ids_defaults_pending_to_reject():
    """Operator submits decisions keyed by ids that don't match any
    pending ToolCall (typo, stale UI, race). The pending calls fall
    to the fail-closed default (reject), the extra ids are ignored,
    the run continues cleanly. No autonomous execution."""
    store = InMemoryStore()
    reg = _one_pending_delete_run(store)

    # Operator's decisions reference c99 (wrong id) — c1 is the actual
    # pending id.
    llm2 = FakeLLM.script([Turn(content="ack")])
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).resume(
            "run-1",
            {"c99": "approve", "c98": "approve"},  # neither matches c1
            make_test_ctx(
                llm=llm2,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )

    # Run completed (didn't hang / raise), but the delete was NOT
    # executed — c1 defaulted to reject because its id wasn't in
    # ``decisions``.
    assert res.output == "ack"


def test_resume_with_empty_decisions_rejects_all_pending():
    """Empty ``decisions`` dict → every pending ToolCall defaults to
    reject. Legal shape for "operator submitted the form with no
    approvals selected". No implicit approvals leak in."""
    store = InMemoryStore()
    reg = _one_pending_delete_run(store, correlation_id="run-2")

    llm2 = FakeLLM.script([Turn(content="rejected-all")])
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).resume(
            "run-2",
            {},  # no decisions at all
            make_test_ctx(
                llm=llm2,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-2",
            ),
        )
    )

    assert res.output == "rejected-all"


# ---- Phase-1 spine: termination (ch18) · cancellation (ch18) · autonomy gate (ch17) -----------


def _se_registry(counter=None, *, requires_approval=False):
    """A side-effecting tool (a "key step"), optionally requiring approval."""
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            "delete",
            lambda a, c: (counter and counter.__setitem__("n", counter["n"] + 1)) or "gone",
            description="delete a resource by id; mutates the world (destructive)",
            side_effecting=True,
            requires_approval=requires_approval,
        )
    )
    return reg


def test_termination_function_call_preempts_tool_execution():
    counter = {"n": 0}
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "fetch", {}),)), Turn(content="done")])
    loop = Agent(
        name="a",
        model="m",
        cognition=ReActCognition(tools=_registry(counter), termination=FunctionCall("fetch")),
    )
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.evals["stop_reason"] == "function_call" and not res.partial
    assert llm.calls == 1 and counter["n"] == 0  # stopped on the call, before executing the tool


def test_termination_text_mention_tags_the_stop():
    llm = FakeLLM.script([Turn(content="we are DONE here")])
    # termination is a tool-loop concept — give the Agent a (no-op) registry so the loop branch runs.
    loop = Agent(
        name="a",
        model="m",
        cognition=ReActCognition(tools=_registry(), termination=TextMention("DONE")),
    )
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == "we are DONE here" and res.evals["stop_reason"] == "text_mention"


def test_cancellation_aborts_before_first_chat():
    token = CancellationToken()
    token.cancel()
    ctx = make_test_ctx(
        llm=FakeLLM.script([Turn(content="never")]),
        chat_middleware=[tracing(), meter()],
        tool_middleware=[tracing()],
        budget=Budget(max_cost_usd=10.0),
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="run-1",
    )
    ctx.cancel = token
    with pytest.raises(Cancelled):
        _run(Agent(name="a", model="m", cognition=ReActCognition(tools=_registry())).run("t", ctx))


def test_manual_autonomy_gates_an_unapproved_tool():
    counter = {"n": 0}
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "fetch", {}),)), Turn(content="done")])
    ctx = make_test_ctx(
        llm=llm,
        chat_middleware=[tracing(), meter()],
        tool_middleware=[tracing()],
        store=InMemoryStore(),
        budget=Budget(max_cost_usd=10.0),
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="run-1",
    )
    ctx.autonomy = "manual"  # gate everything, even a plain non-approval tool
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=_registry(counter))).run("t", ctx)
    )
    assert res.partial and isinstance(res.evals["suspended"], Suspended) and counter["n"] == 0


def test_gated_autonomy_gates_side_effecting_key_step():
    counter = {"n": 0}
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "delete", {}),)), Turn(content="done")])
    ctx = make_test_ctx(
        llm=llm,
        chat_middleware=[tracing(), meter()],
        tool_middleware=[tracing()],
        store=InMemoryStore(),
        budget=Budget(max_cost_usd=10.0),
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="run-1",
    )
    ctx.autonomy = "gated"  # gate "key" (side-effecting) steps, sans approval flag
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=_se_registry(counter))).run(
            "t", ctx
        )
    )
    assert res.partial and isinstance(res.evals["suspended"], Suspended) and counter["n"] == 0


def test_auto_autonomy_runs_side_effecting_tool_without_gating():
    counter = {"n": 0}
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "delete", {}),)), Turn(content="done")])
    res = _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=_se_registry(counter))).run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert (
        res.output == "done" and counter["n"] == 1
    )  # AUTO ignores key_step; only honours requires_approval


# ---- single-call Agent regressions -----------------------------------------------------------


def test_resume_on_single_call_agent_raises():
    """Calling resume() on a `tools=None` Agent is a contract bug — surface it loudly.
    Single-call agents don't checkpoint and have no pending tool decisions to apply."""
    llm = FakeLLM("never")
    agent = Agent(name="a", model="m")
    with pytest.raises(RuntimeError, match="does not support resume"):
        _run(
            agent.resume(
                "run-1",
                {},
                make_test_ctx(
                    llm=llm,
                    chat_middleware=[tracing(), meter()],
                    tool_middleware=[tracing()],
                    store=InMemoryStore(),
                    budget=Budget(max_cost_usd=10.0),
                    trace=Tracer(),
                    scope=Scope(1, 2),
                    correlation_id="run-1",
                ),
            )
        )


# ── ReActCognition tool-error survivability ──────────────────────────────────
#
# ALL tool failures — hallucinated tool names, network errors, timeouts,
# validator errors, shape errors — must be reflected back to the model
# as a tool_result so the loop can recover. A tool exception must not
# propagate out of ``drive`` and abort the agent mid-loop with no
# ``final`` event.


def test_react_recovers_from_hallucinated_tool_name():
    """Model invents a tool that doesn't exist. The loop returns an
    ERROR message naming the available tools so the model can correct,
    and the run continues to a clean final."""
    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "ghost_tool", {"q": "x"}),)),
            Turn(content="recovered after seeing the error"),
        ]
    )
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=_registry()), prompt="go")
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == "recovered after seeing the error"
    assert not res.partial
    assert llm.calls == 2  # second turn ran — the loop survived


def test_react_recovers_from_generic_tool_exception():
    """A tool that raises an arbitrary Exception (network blip, timeout,
    internal bug) is reflected back to the model as a tool_result so
    the loop can pivot."""
    reg = ToolRegistry()

    def _boom(args, ctx):
        raise RuntimeError("simulated network blip")

    reg.register(FunctionTool("flaky", _boom, description="a flaky tool", side_effecting=False))

    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "flaky", {}),)),
            Turn(content="pivoted after tool failure"),
        ]
    )
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=reg), prompt="go")
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == "pivoted after tool failure"
    assert llm.calls == 2  # loop survived; model saw the error tool_result


def test_react_keeps_existing_tool_shape_error_recovery():
    """The ToolShapeError → "schema mismatch" recovery surface remains
    the most-trusted recovery path and must not regress."""
    from agentkit.tools.errors import ToolShapeError

    reg = ToolRegistry()

    def _bad_shape(args, ctx):
        raise ToolShapeError(expected={"type": "object"}, errors=["missing field 'x'"])

    reg.register(FunctionTool("bad", _bad_shape, description="bad shape", side_effecting=False))

    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "bad", {}),)),
            Turn(content="acknowledged shape failure"),
        ]
    )
    loop = Agent(name="a", model="m", cognition=ReActCognition(tools=reg), prompt="go")
    res = _run(
        loop.run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="run-1",
            ),
        )
    )
    assert res.output == "acknowledged shape failure"
    assert llm.calls == 2


# ── ReActCognition.resume hardening ──────────────────────────────────
#
# Three edges the resume path had to be tightened around: (1) an
# operator cancel that trips WHILE a multi-approve batch is being
# dispatched must abort mid-batch, not push every pending tool through
# first; (2) two concurrent runs sharing the same ReActCognition
# instance must NOT race on the termination condition's counters — the
# clone happens per-drive, and the resume path threads the same clone
# through ``_iterate``; (3) ``Suspended.pending`` is narrowed from
# ``tuple[Any, ...]`` to ``tuple[ToolCall, ...] | tuple[str, ...]``,
# so callers get a real type-error at the seam instead of a runtime
# surprise deep in the resume dispatch.


def test_resume_check_cancelled_before_dispatch():
    """An operator who cancels the run while decisions are in flight
    must abort BEFORE any pending tool fires. The dispatch loop reads
    ``ctx.check_cancelled()`` on entry and on every iteration; a
    tripped token raises ``Cancelled`` and no tool runs."""
    store = InMemoryStore()
    counter = {"n": 0}
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            "delete",
            lambda a, c: counter.__setitem__("n", counter["n"] + 1) or "gone",
            description="delete a resource by id; mutates the world (destructive)",
            side_effecting=True,
            requires_approval=True,
        )
    )

    # Drive to a suspended state with one pending delete.
    llm = FakeLLM.script([Turn(tool_calls=(ToolCall("c1", "delete", {}),)), Turn(content="done")])
    _run(
        Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).run(
            "t",
            make_test_ctx(
                llm=llm,
                chat_middleware=[tracing(), meter()],
                tool_middleware=[tracing()],
                store=store,
                budget=Budget(max_cost_usd=10.0),
                trace=Tracer(),
                scope=Scope(1, 2),
                correlation_id="cancel-run",
            ),
        )
    )
    assert counter["n"] == 0  # gated, not executed

    # Now resume with a tripped cancellation token AND an "approve"
    # decision — the check_cancelled at the top of resume() dispatch
    # must fire first, aborting BEFORE the approved tool is invoked.
    token = CancellationToken()
    token.cancel()
    llm2 = FakeLLM.script([Turn(content="never")])
    ctx2 = make_test_ctx(
        llm=llm2,
        chat_middleware=[tracing(), meter()],
        tool_middleware=[tracing()],
        store=store,
        budget=Budget(max_cost_usd=10.0),
        trace=Tracer(),
        scope=Scope(1, 2),
        correlation_id="cancel-run",
    )
    ctx2.cancel = token

    with pytest.raises(Cancelled):
        _run(
            Agent(name="a", model="m", cognition=ReActCognition(tools=reg)).resume(
                "cancel-run", {"c1": "approve"}, ctx2
            )
        )
    # The delete NEVER ran — cancel beat dispatch.
    assert counter["n"] == 0


def test_resume_uses_drive_local_termination_clone():
    """Two agents share a single ReActCognition instance (same
    termination object). Running them concurrently via ``asyncio.gather``
    must NOT share ``MaxTurns.turn`` counter state — each ``drive`` /
    ``resume`` clones ``self.termination`` into a drive-local variable
    before threading it into ``_iterate``. Without the clone, the two
    runs would race on the shared counter and one would stop early."""
    from agentkit.agents.control.termination import MaxTurns

    # Two-turn scripts each: the model tool-calls once, then answers.
    # MaxTurns(3) allows both turns comfortably. If the counter were
    # shared, the two concurrent runs would race and one would trip
    # max_turns after ~2 combined ticks.
    async def _run_pair():
        shared = ReActCognition(tools=_registry(), max_iterations=8, termination=MaxTurns(3))
        agent_a = Agent(name="a", model="m", cognition=shared)
        agent_b = Agent(name="b", model="m", cognition=shared)

        llm_a = FakeLLM.script(
            [Turn(tool_calls=(ToolCall("a1", "fetch", {}),)), Turn(content="A done")]
        )
        llm_b = FakeLLM.script(
            [Turn(tool_calls=(ToolCall("b1", "fetch", {}),)), Turn(content="B done")]
        )
        ctx_a = make_test_ctx(
            llm=llm_a,
            chat_middleware=[tracing(), meter()],
            tool_middleware=[tracing()],
            budget=Budget(max_cost_usd=10.0),
            trace=Tracer(),
            scope=Scope(1, 2),
            correlation_id="run-a",
        )
        ctx_b = make_test_ctx(
            llm=llm_b,
            chat_middleware=[tracing(), meter()],
            tool_middleware=[tracing()],
            budget=Budget(max_cost_usd=10.0),
            trace=Tracer(),
            scope=Scope(1, 2),
            correlation_id="run-b",
        )
        res_a, res_b = await asyncio.gather(agent_a.run("t", ctx_a), agent_b.run("t", ctx_b))
        return shared, res_a, res_b

    shared, res_a, res_b = _run(_run_pair())

    # Both runs completed with their own final answer — proof that
    # neither tripped the OTHER run's termination counter.
    assert res_a.output == "A done"
    assert res_b.output == "B done"
    # ``self.termination.turn`` stays 0 — the clones absorb the ticks,
    # not the original.
    assert shared.termination is not None
    assert shared.termination.turn == 0


def test_suspended_pending_type_matches_toolcall():
    """``Suspended.pending`` is a tuple of ``ToolCall``. Constructing a
    ``Suspended`` with a ``ToolCall`` tuple is the normal path; the
    type annotation catches drift at the seam, and a runtime
    ``isinstance`` assert here backstops that."""
    tc = ToolCall("c1", "delete", {"path": "/etc/hosts"})
    susp = Suspended(run_id="run-x", pending=(tc,))
    assert isinstance(susp.pending, tuple)
    assert len(susp.pending) == 1
    assert isinstance(susp.pending[0], ToolCall)
    assert susp.pending[0].id == "c1"
    assert susp.pending[0].name == "delete"
