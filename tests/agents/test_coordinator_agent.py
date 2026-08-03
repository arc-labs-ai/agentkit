"""Coordinator `Agent` + `RoundRobinPolicy` / `SelectorPolicy` (ch15 / R8): named children
take turns on a shared transcript until a `TerminationCondition` fires or the policy's
`max_turns` ceiling hits. Verifies it consumes the Phase-1 spine — termination (ch18),
cancellation (ch18), observation (ch23). Offline & deterministic via `asyncio.run`."""

import asyncio

import pytest

from agentkit import Agent, AgentResult
from agentkit.adapters.observer import CollectingObserver
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.control.termination import MaxMessages, TextMention
from agentkit.agents.policies.roundrobin import RoundRobinPolicy, marker_notes
from agentkit.agents.policies.selector_policy import SelectorPolicy, handoff_selector
from agentkit.capabilities import SummarizationCompactor
from agentkit.context import WorkingContext
from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.types import Scope, Usage
from agentkit.testing import FakeLLM, make_test_ctx


def _run(coro):
    return asyncio.run(coro)


def _coord(termination=None, *, max_turns=50, **kw):
    """Build a coordinator Agent with two named children + RoundRobinPolicy."""
    children = {"alice": Agent("alice", "m"), "bob": Agent("bob", "m")}
    return Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=children,
            policy=RoundRobinPolicy(max_turns=max_turns, **kw),
            termination=termination,
        ),
    )


def test_max_messages_stops_after_two_turns():
    coord = _coord(MaxMessages(2))
    res = _run(
        coord.run(
            "start",
            make_test_ctx(
                llm=FakeLLM("ok"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
        )
    )
    assert res.evals["stop_reason"] == "max_messages"
    assert len(res.evals["results"]) == 2  # one per agent turn
    transcript = res.evals["messages"]
    assert transcript[0].content == "start"  # transcript opens with the task
    assert [m.name for m in transcript[1:]] == ["alice", "bob"]


def test_unnamed_child_labelled_by_roster_key_not_turn():
    """Regression: a child agent without `.name` is labelled by its roster key in the
    children dict (matching `_name_of` fallback) — so handoff/source-match read a
    consistent name."""

    class _Nameless:  # an agent-like with no `.name`
        async def run(self, task, ctx):
            return AgentResult(output="hi", usage=Usage())

    # The nameless child is keyed as "1" in the children dict; the policy should label
    # its turn with that key (NOT the turn index "agent1"/"agent0").
    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children={"alice": Agent("alice", "m"), "1": _Nameless()},
            policy=RoundRobinPolicy(),
            termination=MaxMessages(2),
        ),
    )
    res = _run(
        coord.run(
            "start",
            make_test_ctx(
                llm=FakeLLM("ok"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
        )
    )
    assert [m.name for m in res.evals["messages"][1:]] == ["alice", "1"]


def test_text_mention_stops_on_first_match():
    res = _run(
        _coord(TextMention("APPROVE")).run(
            "start",
            make_test_ctx(
                llm=FakeLLM("we APPROVE this plan"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
        )
    )
    assert res.evals["stop_reason"] == "text_mention"
    assert len(res.evals["results"]) == 1  # stops the moment alice mentions it


def test_max_turns_ceiling_when_condition_never_fires():
    res = _run(
        _coord(TextMention("NEVER"), max_turns=3).run(
            "start",
            make_test_ctx(
                llm=FakeLLM("nothing here"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
        )
    )
    assert res.evals["stop_reason"] == "max_turns"
    assert len(res.evals["results"]) == 3  # ran exactly to the backstop


def test_emits_observations_per_turn_and_a_result():
    obs = CollectingObserver()
    _run(
        _coord(MaxMessages(2)).run(
            "start",
            make_test_ctx(llm=FakeLLM("ok"), observer=obs, scope=Scope(1, 2), correlation_id="r"),
        )
    )
    kinds = [o.kind for o in obs.items]
    assert kinds.count("summary") == 2  # one progress observation per turn
    assert kinds[-1] == "result"  # final result observation


def test_honors_cancellation_before_running_a_child():
    token = CancellationToken()
    token.cancel()
    with pytest.raises(Cancelled):
        _run(
            _coord(MaxMessages(2)).run(
                "start",
                make_test_ctx(
                    llm=FakeLLM("ok"),
                    cancel=token,
                    observer=CollectingObserver(),
                    scope=Scope(1, 2),
                    correlation_id="r",
                ),
            )
        )


def test_cancellation_mid_flight_propagates_to_children():
    """A cancel that trips WHILE the coordinator is dispatching children
    must raise on the next check-point — not merely block new work but
    unwind the run. The parent's ``ctx.cancel`` reference is shared
    into every ``ctx.child()``, so the child agents' own
    ``check_cancelled()`` calls see the trip on the next iteration."""
    token = CancellationToken()

    call_count = {"n": 0}

    class _TrippingLLM:
        """An LLM that trips the cancel token on its first call. The
        coordinator's next ``check_cancelled()`` — between turns —
        must observe the trip and raise ``Cancelled``. Simulates an
        operator's abort arriving mid-dispatch."""

        async def stream(self, **kw):
            call_count["n"] += 1
            token.cancel()
            # Yield one delta so the streaming primitive completes normally;
            # cancellation is observed on the loop's next boundary check.
            from agentkit.kernel.types import Delta, Usage

            yield Delta(text="partial", usage=Usage(input_tokens=1, output_tokens=1))

        async def chat(self, **kw):
            from agentkit.kernel.types import LLMResult, Usage

            call_count["n"] += 1
            token.cancel()
            return LLMResult(content="partial", usage=Usage(input_tokens=1, output_tokens=1))

    with pytest.raises(Cancelled):
        _run(
            _coord(MaxMessages(4)).run(
                "start",
                make_test_ctx(
                    llm=_TrippingLLM(),
                    cancel=token,
                    observer=CollectingObserver(),
                    scope=Scope(1, 2),
                    correlation_id="r-mid",
                ),
            )
        )
    # The LLM was invoked at least once — cancellation didn't happen
    # before any child dispatched. And the loop did NOT run to the
    # MaxMessages ceiling — cancellation cut it short.
    assert call_count["n"] >= 1
    assert call_count["n"] < 4


def test_cancel_token_is_shared_reference_across_ctx_child():
    """Structural invariant: ``ctx.child()`` shares the cancel token by
    reference, not by copy. Every descendant sees the same
    ``CancellationToken`` instance, so tripping it once cancels the
    whole tree — the load-bearing property for coordinator dispatch."""
    from agentkit.runtime import Budget, RunContext, Services

    token = CancellationToken()
    parent = RunContext(
        correlation_id="r",
        scope=Scope(1, 2),
        budget=Budget(),
        services=Services(),
        cancel=token,
    )
    child = parent.child()
    grandchild = child.child()

    # All three share the exact same token object — not equal-by-value
    # copies. ``token.cancel()`` on the parent's reference lands on
    # the child's and grandchild's.
    assert parent.cancel is token
    assert child.cancel is token
    assert grandchild.cancel is token

    token.cancel()
    with pytest.raises(Cancelled):
        parent.check_cancelled()
    with pytest.raises(Cancelled):
        child.check_cancelled()
    with pytest.raises(Cancelled):
        grandchild.check_cancelled()


# ---- SelectorPolicy (selector-driven who-speaks-next; subsumes handoff/swarm) ------------------


def _children():
    return {"alice": Agent("alice", "m"), "bob": Agent("bob", "m")}


def _selector_coord(selector, termination=MaxMessages(2)):
    return Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=_children(),
            policy=SelectorPolicy(selector=selector),
            termination=termination,
        ),
    )


def test_selector_picks_the_named_agent_every_turn():
    res = _run(
        _selector_coord(lambda t, a: "bob").run(
            "start",
            make_test_ctx(
                llm=FakeLLM("ok"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
        )
    )
    assert [m.name for m in res.evals["messages"][1:]] == ["bob", "bob"]


def test_selector_unknown_name_falls_back_to_round_robin():
    res = _run(
        _selector_coord(lambda t, a: "ghost").run(
            "start",
            make_test_ctx(
                llm=FakeLLM("ok"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
        )
    )
    # fallback never stalls the run
    assert [m.name for m in res.evals["messages"][1:]] == ["alice", "bob"]


def test_async_selector_is_awaited():
    async def pick(transcript, agents):
        return "bob"

    res = _run(
        _selector_coord(pick).run(
            "start",
            make_test_ctx(
                llm=FakeLLM("ok"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
        )
    )
    assert [m.name for m in res.evals["messages"][1:]] == ["bob", "bob"]


def test_handoff_selector_routes_by_marker_on_last_message():
    coord = _selector_coord(handoff_selector(default="alice"))
    res = _run(
        coord.run(
            "start HANDOFF:bob",
            make_test_ctx(
                llm=FakeLLM("ok HANDOFF:alice"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
        )
    )
    # turn 0: task names bob → bob speaks; turn 1: bob's reply hands off to alice
    assert [m.name for m in res.evals["messages"][1:]] == ["bob", "alice"]


# ---- shared blackboard: a WorkingContext as the team's transcript + scratchpad (ch13) ---------


def test_blackboard_is_the_transcript_and_accumulates():
    wc = WorkingContext()
    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=_children(),
            policy=RoundRobinPolicy(),
            termination=MaxMessages(2),
        ),
    )
    res = _run(
        coord.run(
            "start",
            make_test_ctx(
                llm=FakeLLM("ok"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
            context=wc,
        )
    )
    # caller's WorkingContext IS the live blackboard — they can inspect/fork after
    assert [m.content for m in wc.messages] == ["start", "ok", "ok"]
    assert [m.name for m in wc.messages[1:]] == ["alice", "bob"]
    # the returned transcript mirrors the live blackboard's contents
    assert [m.content for m in res.evals["messages"]] == [m.content for m in wc.messages]


def test_shared_scratchpad_is_rendered_to_children():
    wc = WorkingContext().note("topic", "redis")
    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children={"alice": Agent("alice", "m")},
            policy=RoundRobinPolicy(),
            termination=MaxMessages(1),
        ),
    )
    res = _run(
        coord.run(
            "start",
            make_test_ctx(
                llm=FakeLLM(lambda **kw: kw["user"]),  # echo the prompt seen
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
            context=wc,
        )
    )
    inner = res.evals["results"][0].output
    assert "Shared notes" in inner and "topic: redis" in inner


def test_note_parser_writes_child_notes_onto_the_blackboard():
    wc = WorkingContext()
    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children={"alice": Agent("alice", "m")},
            policy=RoundRobinPolicy(note_parser=marker_notes()),
            termination=MaxMessages(1),
        ),
    )
    _run(
        coord.run(
            "start",
            make_test_ctx(
                llm=FakeLLM("working on it NOTE:status=ok"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
            context=wc,
        )
    )
    assert wc.get("status") == "ok"


def test_notes_flow_from_one_child_to_the_next():
    wc = WorkingContext()

    def reply(**kw):
        return "NOTE:owner=alice\n" + kw["user"]  # emit a note, then echo the prompt seen

    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=_children(),
            policy=RoundRobinPolicy(note_parser=marker_notes()),
            termination=MaxMessages(2),
        ),
    )
    res = _run(
        coord.run(
            "start",
            make_test_ctx(
                llm=FakeLLM(reply),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
            context=wc,
        )
    )
    assert wc.get("owner") == "alice"  # alice's note landed on the blackboard
    assert "owner: alice" in res.evals["results"][1].output  # bob saw it via the scratchpad


def test_blackboard_is_compacted_on_cadence():
    wc = WorkingContext()
    compactor = SummarizationCompactor(summarizer=FakeLLM("SUMMARY"), max_tokens=0, keep_recent=1)
    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=_children(),
            policy=RoundRobinPolicy(compactor=compactor, compact_every=2),
            termination=MaxMessages(4),
        ),
    )
    res = _run(
        coord.run(
            "start",
            make_test_ctx(
                llm=FakeLLM("ok"),
                observer=CollectingObserver(),
                scope=Scope(1, 2),
                correlation_id="r",
            ),
            context=wc,
        )
    )
    assert any(
        m.role == "system" and "Summary" in m.content for m in wc.messages
    )  # earlier turns folded
    assert len(wc.messages) < 5  # 1 task + 4 replies would be 5 uncompacted
    # the returned transcript still mirrors the live blackboard after the re-alias compaction
    assert len(res.evals["messages"]) == len(wc.messages)
