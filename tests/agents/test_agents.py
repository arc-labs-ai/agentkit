"""Smoke tests for the `Agent` + `Policy` shape — the cross-cutting surface.
A leaf `Agent` is the single-call form; a coordinator `Agent` dispatches to
`children` per its `Policy`. `RunPolicy` (Rule of Two) is the run-wide safety
check."""

import asyncio
import copy
import json
import pickle

import pytest

from agentkit import Agent, RunPolicy
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies.plan import PlanPolicy, StaticPlanner, Step
from agentkit.kernel._frozen import FrozenDict
from agentkit.kernel.types import Scope
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import FunctionTool


def _run(coro):
    return asyncio.run(coro)


def test_agent_single_call_goes_through_invoker():
    ctx = make_test_ctx(llm=FakeLLM("answer"), scope=Scope(1, 2), correlation_id="r")
    res = _run(Agent(name="a", model="m", prompt="s").run("q", ctx))
    assert res.output == "answer"


def test_coordinator_plan_policy_runs_groups_in_order():
    ctx = make_test_ctx(llm=FakeLLM("x"), scope=Scope(1, 2), correlation_id="r")
    coord = Agent(
        name="orch",
        cognition=CoordinatorCognition(
            children={"ex": Agent("ex", "m"), "ju": Agent("ju", "m")},
            policy=PlanPolicy(
                planner=StaticPlanner([Step("ex", "c", group=0), Step("ju", "c", group=1)])
            ),
        ),
    )
    res = _run(coord.run("goal", ctx))
    # Two steps across two groups → two per-child results recorded in evals.
    results = res.evals["results"]
    assert len(results) == 2 and results[0].output == "x"


def _t(name, caps):
    return FunctionTool(
        name,
        lambda a, c: None,
        description="test stub tool used for RunPolicy capability checks",
        side_effecting=False,
        caps=caps,
    )


def test_run_policy_flags_lethal_trifecta():
    v = RunPolicy().check(
        [_t("a", ("private_data",)), _t("b", ("untrusted_content",)), _t("c", ("egress",))]
    )
    assert not v.allowed and "lethal trifecta" in v.reason


def test_run_policy_allows_two_of_three():
    assert RunPolicy().check([_t("a", ("private_data",)), _t("b", ("egress",))]).allowed


# ── Suspended handshake is fully immutable ──────────────────────────────────
#
# ``Suspended`` is the value the run yields when it pauses for human
# approval. The operator UI renders the ``pending`` items; the resume
# path threads them back verbatim. If the collection carrying those
# items were mutable, a stray ``suspended.pending.append(...)`` on
# either side would desync the operator's view from what actually
# runs after resume — the classic approve-then-modify attack surface.


def test_suspended_pending_is_a_frozen_tuple() -> None:
    """``pending`` is a tuple, not a list. Reassignment is blocked by
    the frozen dataclass; item assignment / append is blocked by the
    tuple itself. Two doors, both locked."""
    from dataclasses import FrozenInstanceError

    from agentkit.agents.result import Suspended
    from agentkit.kernel.types import ToolCall

    tc = ToolCall("call-1", "delete", {"path": "/etc/hosts"})
    susp = Suspended(run_id="run-x", pending=(tc,))

    assert isinstance(susp.pending, tuple)
    # Frozen shell — cannot rebind the whole collection.
    with pytest.raises(FrozenInstanceError):
        susp.pending = ()  # type: ignore[misc]
    # Frozen collection — cannot mutate items in-place either.
    with pytest.raises(TypeError):
        susp.pending[0] = ToolCall("call-2", "wipe", {})  # type: ignore[index]
    with pytest.raises(AttributeError):
        susp.pending.append(ToolCall("call-2", "wipe", {}))  # type: ignore[attr-defined]


def test_suspended_pending_survives_replace_for_intentional_edits() -> None:
    """``dataclasses.replace`` is the sanctioned rewrite path — a caller
    that legitimately needs to add or drop a pending item builds a
    fresh ``Suspended`` instead of mutating one in flight."""
    from dataclasses import replace

    from agentkit.agents.result import Suspended
    from agentkit.kernel.types import ToolCall

    original = Suspended(run_id="run-x", pending=(ToolCall("c1", "a", {}),))
    extended = replace(original, pending=(*original.pending, ToolCall("c2", "b", {})))

    assert len(original.pending) == 1  # untouched
    assert len(extended.pending) == 2
    assert original is not extended


# ── AgentResult / WorkflowResult behave like the values they are ──────────
#
# `AgentResult` is what every `agent.run()` returns, and it was unhashable for
# EVERY instance — `evals` is `field(default_factory=dict)`, so the generated
# all-fields hash reached a dict on the plainest result the framework builds.
# Measured before the fix::
#
#     hash(AgentResult(output="hi", usage=Usage()))              TypeError: unhashable type: 'dict'
#     hash(AgentResult(output="hi", usage=Usage(), evals={...})) TypeError: unhashable type: 'dict'
#
# Identical lines: it broke by TYPE, not by value, so there was no working case
# to compare against and nothing inside the framework hashes a result — the
# first caller to dedupe a fan-out's results through a `set`, or to memoize on
# one, would have been the one to find it. `WorkflowResult` is the same shape
# via its required `outputs` dict.
#
# The fix hashes an identity SUBSET and leaves `__eq__` alone: the hash
# invariant only requires EQUAL objects to hash equally, never that unequal
# ones differ. Unequal results sharing a bucket is what a bucket is for.


def _res(**kw):
    from agentkit.agents.result import AgentResult
    from agentkit.kernel.types import Usage

    kw.setdefault("output", "answer")
    kw.setdefault("usage", Usage(input_tokens=10, output_tokens=5, cost_usd=0.01))
    return AgentResult(**kw)


def test_agent_result_is_hashable_with_the_default_empty_evals() -> None:
    """The plainest result the framework can produce — no evals written at all
    — was unhashable. This is the line that shows the bug was unconditional."""
    assert isinstance(hash(_res()), int)


def test_agent_result_is_hashable_with_the_payloads_it_actually_carries() -> None:
    """Representative payloads, not minimal ones: nested `evals` as the tool
    loop and the coordinator policies write them, a `parsed` object from an
    output parser, and the `None`/empty edges."""

    class _ParsedModel:  # a Pydantic model is unhashable unless declared frozen
        __hash__ = None  # type: ignore[assignment]

    nested = {"stop_reason": "awaiting_decision", "children": [{"a": {"b": [1, 2]}}]}
    assert isinstance(hash(_res(evals=nested, parsed={"total": 3})), int)
    assert isinstance(hash(_res(evals={}, parsed=None)), int)
    assert isinstance(hash(_res(evals={"k": None}, parsed=_ParsedModel())), int)
    assert isinstance(hash(_res(output="", evals={"deep": {"x": {"y": []}}})), int)


def test_agent_result_hash_ignores_evals_and_parsed_while_eq_does_not() -> None:
    """The soundness argument, exercised. Two results that agree on text and
    spend but disagree on evals/parsed collide into one bucket, stay UNEQUAL,
    and both survive in a `set` — `__eq__` separates them there."""
    a = _res(evals={"stop_reason": "complete"}, parsed={"v": 1})
    b = _res(evals={"stop_reason": "max_turns"}, parsed={"v": 2})
    assert hash(a) == hash(b)
    assert a != b
    assert len({a, b}) == 2


def test_agent_result_hash_is_o1_in_the_evals_payload() -> None:
    """Proven STRUCTURALLY rather than by timing, so it cannot go flaky: a
    result carrying 100_000 eval keys hashes to the same number as one carrying
    a single key. That is only possible if the payload is never read."""
    huge = {f"k{i}": {"nested": [i]} for i in range(100_000)}
    assert hash(_res(evals={"k0": {"nested": [0]}})) == hash(_res(evals=huge))


def test_agent_result_hash_separates_the_parts_it_keeps() -> None:
    """The hashed subset has to earn its place — a hash that ignored these
    would still be CORRECT but would collapse every result into one bucket."""
    from agentkit.kernel.types import Usage

    base = _res()
    assert hash(base) != hash(_res(output="different"))
    assert hash(base) != hash(_res(usage=Usage(input_tokens=99)))
    assert hash(base) != hash(_res(partial=True))
    assert hash(base) != hash(_res(prompt_version="v2"))
    assert hash(base) != hash(_res(stop_reason="suspended"))


def test_agent_results_dedupe_through_a_set_the_way_a_fan_in_wants() -> None:
    """The caller this unlocks: a coordinator collects per-child results and
    wants the duplicates gone. Identical results collapse; a result that merely
    SHARES the hashed subset does not."""
    same = (_res(evals={"child": "a"}), _res(evals={"child": "a"}))
    assert len(set(same)) == 1
    assert len({*same, _res(evals={"child": "b"})}) == 2
    assert len({_res(), _res(output="other")}) == 2


def test_agent_result_works_as_a_memo_cache_key() -> None:
    """Keyed lookup by an EQUAL-but-distinct result object, which is what a
    memo table does — and the reason an identity hash would be wrong here."""
    cache = {_res(evals={"n": 1}): "rendered"}
    assert cache[_res(evals={"n": 1})] == "rendered"
    assert _res(output="unseen") not in cache


def test_agent_result_evals_are_a_dict_for_every_consumer() -> None:
    """POSITIVE CONTROL, and the constraint the freeze is under: `evals` is a
    `dict` SUBCLASS, not a `MappingProxyType`. It is JSON serialised and read
    back through `dataclasses.asdict`, neither of which a proxy survives.
    Passes before and after the freeze.

    This test USED to end with `r.evals["late"] = ...` — commented "the
    documented habit" — as a positive control that callers could annotate a
    result after construction. That contract is gone; the write half now lives
    in `test_agent_result_evals_refuse_post_construction_writes` below, which
    pins the refusal and shows the migration."""
    import dataclasses
    import json

    r = _res(evals={"stop_reason": "complete", "children": [{"n": 1}]})
    assert isinstance(r.evals, dict)
    assert r.evals["stop_reason"] == "complete"
    assert r.evals["children"][0]["n"] == 1
    assert len(r.evals) == 2 and set(r.evals) == {"stop_reason", "children"}
    assert dict(r.evals) == {"stop_reason": "complete", "children": [{"n": 1}]}
    assert r.evals == {"stop_reason": "complete", "children": [{"n": 1}]}  # eq vs a PLAIN dict
    assert json.loads(json.dumps(r.evals))["children"] == [{"n": 1}]
    assert dataclasses.asdict(r)["evals"]["stop_reason"] == "complete"


def test_agent_result_evals_refuse_post_construction_writes() -> None:
    """THE BREAKING CHANGE, stated as a test. `evals` is the audit record of a
    run: what it cost, why it stopped, which children failed. Rewriting it
    after the fact used to succeed silently, because `frozen=True` protected
    only the field REFERENCE::

        r.evals = {}            # FrozenInstanceError, as intended
        r.evals["cost"] = 0.0   # ...but this rewrote the record

    Migration — build a new value instead of editing the old one::

        r = dataclasses.replace(r, evals={**r.evals, "late": "note"})
    """
    import dataclasses

    r = _res(evals={"stop_reason": "complete"})

    with pytest.raises(TypeError, match="frozen value"):
        r.evals["late"] = {"written": "after construction"}
    with pytest.raises(TypeError, match="frozen value"):
        r.evals.update({"late": 1})  # the next thing a caller reaches for
    with pytest.raises(TypeError, match="frozen value"):
        del r.evals["stop_reason"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.evals = {}  # type: ignore[misc]  # unchanged — the reference was always frozen

    annotated = dataclasses.replace(r, evals={**r.evals, "late": "note"})
    assert annotated.evals == {"stop_reason": "complete", "late": "note"}
    assert r.evals == {"stop_reason": "complete"}  # the original is untouched
    assert isinstance(annotated.evals, FrozenDict)  # ...and the new one is frozen too


def test_agent_result_evals_are_frozen_all_the_way_down() -> None:
    """A shallow freeze is the same bug one level down and harder to see —
    `evals` is nested by construction (`cli_init`, `errors`, per-child result
    lists), so those are exactly the paths a caller reaches through."""
    r = _res(evals={"a": {"b": [{"c": 0}]}, "errors": [{"agent": "x"}]})

    with pytest.raises(TypeError, match="frozen value"):
        r.evals["a"]["b"][0]["c"] = 1
    with pytest.raises(TypeError, match="frozen value"):
        r.evals["errors"].append({"agent": "y"})
    assert r.evals["a"]["b"][0]["c"] == 0


def test_agent_result_does_not_alias_the_producers_dict() -> None:
    """Every producer builds a local `evals` dict and hands it to the
    constructor (`_agent_helpers.finish`, `claude_cli`, the four policies). If
    the result ALIASED that local, a producer writing one more key after
    returning would edit a result it had already handed out."""
    mine = {"stop_reason": "complete"}
    r = _res(evals=mine)

    mine["mutated"] = True  # the producer keeps using its own dict — legal
    assert r.evals == {"stop_reason": "complete"}
    assert "mutated" not in r.evals


def test_agent_result_tolerates_a_none_evals_the_way_the_serialiser_assumes() -> None:
    """`evals` is annotated `dict`, but `result_to_dict` guards `dict(r.evals or {})`
    — the framework does not trust the annotation here, and neither does the
    freeze. `deep_freeze` passes non-container leaves through, so a `None`
    payload stays `None` instead of raising inside `__post_init__` at the one
    seam every `agent.run()` goes through."""
    r = _res(evals=None)
    assert r.evals is None

    from agentkit.capabilities.checkpointer.persistence import dict_to_result, result_to_dict

    back = dict_to_result(result_to_dict(r))
    assert back.evals == {} and isinstance(back.evals, FrozenDict)


def test_agent_result_empty_and_default_evals_are_frozen_too() -> None:
    """The edge the default hides: `field(default_factory=dict)` means most
    results carry `{}`, and a freeze that only ran on a non-empty payload would
    leave the COMMON case writable."""
    assert isinstance(_res().evals, FrozenDict)
    assert isinstance(_res(evals={}).evals, FrozenDict)
    assert _res().evals == {}

    with pytest.raises(TypeError, match="frozen value"):
        _res().evals["first"] = 1


def test_agent_result_checkpoint_round_trip_is_unchanged() -> None:
    """POSITIVE CONTROL over the durable seam — `result_to_dict` /
    `dict_to_result` is how an `AgentResult` crosses a store, and it round-trips
    field-by-field with no hashing involved. Passes before and after."""
    from agentkit.capabilities.checkpointer.persistence import dict_to_result, result_to_dict

    r = _res(evals={"stop_reason": "awaiting_approval", "n": [1, 2]}, stop_reason="suspended")
    back = dict_to_result(result_to_dict(r))

    assert back == r
    assert back.evals == r.evals and isinstance(back.evals, dict)
    assert back.is_suspended and back.is_resumable


def test_agent_result_comes_back_from_a_checkpoint_frozen() -> None:
    """A value frozen on the way IN and mutable on the way OUT is only half
    fixed, and the restored copy is the more dangerous half: it is the one a
    resume path holds, long after the run that produced it ended.

    `dict_to_result` rebuilds through the CONSTRUCTOR, so `__post_init__` runs
    and the freeze is re-applied rather than merely inherited — which also
    means a record written before this change reads back frozen."""
    from agentkit.capabilities.checkpointer.persistence import dict_to_result, result_to_dict

    raw = result_to_dict(_res(evals={"stop_reason": "awaiting_approval", "n": [{"deep": 1}]}))
    assert json.dumps(raw)  # the record still crosses a serialising store

    back = dict_to_result(raw)
    assert isinstance(back.evals, FrozenDict)
    with pytest.raises(TypeError, match="frozen value"):
        back.evals["tampered"] = True
    with pytest.raises(TypeError, match="frozen value"):
        back.evals["n"][0]["deep"] = 2  # deep, not just at the top level


def test_agent_result_frozen_evals_survive_deepcopy_and_pickle() -> None:
    """POSITIVE CONTROL over the two paths a `FrozenDict` could plausibly break:
    both rebuild a dict subclass by repopulating it through `__setitem__` — the
    method the freeze blocks — so both go through `__reduce__` instead. The
    checkpointer deep-copies on every snapshot, so this is a live path, and the
    copy must come back FROZEN or the freeze leaks through `copy`."""
    r = _res(evals={"a": {"b": [1, {"c": 2}]}})

    for clone in (copy.deepcopy(r), pickle.loads(pickle.dumps(r))):
        assert clone == r and hash(clone) == hash(r)
        assert clone.evals == r.evals
        assert isinstance(clone.evals, FrozenDict)
        with pytest.raises(TypeError, match="frozen value"):
            clone.evals["a"]["b"][1]["c"] = 3


def test_agent_result_still_deepcopies_and_pickles() -> None:
    """POSITIVE CONTROL: both already worked (nothing here is a proxy) and must
    keep working — `Checkpointer.snapshot` deep-copies at the durable seam."""
    import copy
    import pickle

    r = _res(evals={"deep": {"a": [1, {"b": 2}]}}, parsed={"v": 1})
    assert copy.deepcopy(r) == r
    assert pickle.loads(pickle.dumps(r)) == r


def test_agent_result_hash_survives_every_round_trip() -> None:
    """The invariant across the seams a result actually crosses: deepcopy,
    pickle, and the durable checkpoint. Each returns an EQUAL result, so each
    must return an equally-hashing one or a rehydrated result would miss in a
    cache its original populated."""
    import copy
    import pickle

    from agentkit.capabilities.checkpointer.persistence import dict_to_result, result_to_dict

    r = _res(evals={"stop_reason": "complete", "n": [1, 2]}, prompt_version="v3")
    assert hash(copy.deepcopy(r)) == hash(r)
    assert hash(pickle.loads(pickle.dumps(r))) == hash(r)
    assert hash(dict_to_result(result_to_dict(r))) == hash(r)


def test_workflow_result_is_hashable_with_per_node_payloads() -> None:
    """`outputs` is a REQUIRED dict, so every `WorkflowResult` ever built was
    unhashable — including one whose nodes returned plain strings."""
    from agentkit.agents.result import Suspended, WorkflowResult
    from agentkit.kernel.types import Usage

    plain = WorkflowResult(outputs={"draft": "text"}, usage=Usage(), steps=1, stop_reason="complete")
    nested = WorkflowResult(
        outputs={"draft": {"sections": [{"body": ["a"]}]}, "review": None},
        usage=Usage(),
        steps=2,
        stop_reason="suspended",
        suspended=Suspended(run_id="r", pending=("gate",)),
    )
    assert isinstance(hash(plain), int)
    assert isinstance(hash(nested), int)
    assert isinstance(hash(WorkflowResult({}, Usage(), 0, "deadlock")), int)


def test_workflow_result_hash_keeps_node_names_and_drops_their_outputs() -> None:
    """Which nodes ran is the workflow's shape and is worth discriminating on;
    what they returned is opaque `Any`. Same-shape/different-output results
    collide, stay unequal, and both survive a `set`."""
    from agentkit.agents.result import WorkflowResult
    from agentkit.kernel.types import Usage

    a = WorkflowResult(outputs={"draft": {"v": 1}}, usage=Usage(), steps=1, stop_reason="complete")
    b = WorkflowResult(outputs={"draft": {"v": 2}}, usage=Usage(), steps=1, stop_reason="complete")
    other_node = WorkflowResult(
        outputs={"review": {"v": 1}}, usage=Usage(), steps=1, stop_reason="complete"
    )

    assert hash(a) == hash(b) and a != b and len({a, b}) == 2
    assert hash(a) != hash(other_node)
    assert hash(a) != hash(
        WorkflowResult(outputs={"draft": {"v": 1}}, usage=Usage(), steps=2, stop_reason="complete")
    )
    assert hash(a) != hash(
        WorkflowResult(outputs={"draft": {"v": 1}}, usage=Usage(), steps=1, stop_reason="max_steps")
    )


def test_workflow_result_hash_does_not_depend_on_node_insertion_order() -> None:
    """Why the node names go in as a `frozenset` and not a key tuple: dict
    equality ignores insertion order, so these two results are EQUAL, and an
    order-sensitive key tuple would hash them differently — the one thing a
    hash is not allowed to do."""
    from agentkit.agents.result import WorkflowResult
    from agentkit.kernel.types import Usage

    a = WorkflowResult(outputs={"a": 1, "b": 2}, usage=Usage(), steps=2, stop_reason="complete")
    b = WorkflowResult(outputs={"b": 2, "a": 1}, usage=Usage(), steps=2, stop_reason="complete")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_workflow_result_hash_is_o1_in_the_node_payload() -> None:
    """Structural proof again — the payload is never read, so a 100_000-key
    node output hashes exactly like a one-key one."""
    from agentkit.agents.result import WorkflowResult
    from agentkit.kernel.types import Usage

    huge = {f"k{i}": i for i in range(100_000)}
    assert hash(WorkflowResult({"n": {"k0": 0}}, Usage(), 1, "complete")) == hash(
        WorkflowResult({"n": huge}, Usage(), 1, "complete")
    )


def _wf(**kw):
    from agentkit.agents.result import WorkflowResult
    from agentkit.kernel.types import Usage

    kw.setdefault("outputs", {"draft": {"body": "x"}})
    kw.setdefault("usage", Usage())
    kw.setdefault("steps", 1)
    kw.setdefault("stop_reason", "complete")
    return WorkflowResult(**kw)


def test_workflow_result_outputs_are_a_dict_for_every_consumer() -> None:
    """POSITIVE CONTROL: node outputs are read by index and serialised, and the
    durable-resume contract depends on both. This test used to be titled
    "...stay a plain MUTABLE dict"; the mutable half is now
    `test_workflow_result_outputs_refuse_post_construction_writes`."""
    r = _wf()
    assert isinstance(r.outputs, dict)
    assert r.outputs["draft"]["body"] == "x"
    assert set(r.outputs) == {"draft"} and len(r.outputs) == 1
    assert json.dumps(r.outputs)
    assert r.outputs == {"draft": {"body": "x"}}  # eq against a PLAIN dict
    assert r == _wf()
    assert r != _wf(outputs={"draft": {"body": "y"}})
    assert hash(r) == hash(_wf(outputs={"draft": {"body": "y"}}))  # __hash__ unchanged


def test_workflow_result_outputs_refuse_post_construction_writes() -> None:
    """A node output is a RECORD of what that node produced, and the same map
    is what `Checkpointer.snapshot` persisted. Writing into it after the run
    edited a value the checkpointer had already recorded.

    Migration::

        res = dataclasses.replace(res, outputs={**res.outputs, "draft": new})
    """
    import dataclasses

    r = _wf()

    with pytest.raises(TypeError, match="frozen value"):
        r.outputs["draft"] = "rewritten"
    with pytest.raises(TypeError, match="frozen value"):
        r.outputs["draft"]["body"] = "rewritten"  # deep, not just the top level
    with pytest.raises(TypeError, match="frozen value"):
        r.outputs.setdefault("extra", 1)

    amended = dataclasses.replace(r, outputs={**r.outputs, "review": "ok"})
    assert amended.outputs == {"draft": {"body": "x"}, "review": "ok"}
    assert r.outputs == {"draft": {"body": "x"}}


def test_workflow_result_does_not_alias_the_engines_done_map() -> None:
    """The engine hands its LIVE `done` map to the constructor at four call
    sites in `workflow.py`, and hands the same map to the checkpointer. The
    result must be a copy, or the engine's own next write would edit a
    returned result."""
    done = {"draft": {"body": "x"}}
    r = _wf(outputs=done)

    done["draft"] = "engine kept going"  # legal: the engine owns `done`
    assert r.outputs == {"draft": {"body": "x"}}


# ── reground_every_turn is reachable from Agent ────────────────────────────


def test_agent_forwards_reground_every_turn_to_the_builder_it_makes():
    """`Agent(memory=...)` builds the `RequestBuilder` for you, so before this
    the flag was unreachable on the ergonomic path: flipping one bool meant
    hand-constructing the whole builder — prompt, grounder and all.

    The default stays `False` for a cost reason, not an inherited one. The
    grounded block sits in the cache-stable PREFIX, so re-grounding invalidates
    it every turn — measured over 20 turns on a 4k-token prefix, $0.0228 versus
    $0.2280, 10x forever. Flipping the default would have silently re-priced
    every auto-wired agent.
    """
    assert Agent(name="a", model="m").reground_every_turn is False
    assert Agent(name="a", model="m")._resolve_request_builder().reground_every_turn is False

    on = Agent(name="a", model="m", reground_every_turn=True)
    assert on._resolve_request_builder().reground_every_turn is True


def test_an_explicit_request_builder_keeps_its_own_reground_setting():
    """POSITIVE CONTROL. A caller who hand-built a `RequestBuilder` already had
    the knob; the Agent's field must not silently overwrite their choice. Only
    the auto-wired path forwards it."""
    from agentkit.capabilities.request_builder import RequestBuilder
    from agentkit.prompts import Prompt

    mine = RequestBuilder(
        prompt=Prompt(id="p", version="1", template="t"), reground_every_turn=True
    )
    agent = Agent(name="a", model="m", request_builder=mine, reground_every_turn=False)
    assert agent._resolve_request_builder() is mine
    assert agent._resolve_request_builder().reground_every_turn is True


def test_the_flag_actually_changes_how_often_memory_is_consulted():
    """Pins BEHAVIOUR, not the field. A forwarding bug that set the attribute
    but never reached the grounder would pass the tests above."""
    import asyncio

    from agentkit.context import WorkingContext
    from agentkit.memory.base import MemoryItem

    class _CountingMemory:
        name = "counter"

        def __init__(self):
            self.calls = 0

        async def query(self, query, *, k, ctx, where=None):
            self.calls += 1
            return [MemoryItem(content=f"fact {self.calls}", source="m", score=1.0)]

        async def write(self, items, *, ctx):
            return None

    def _turns(agent, n):
        mem = agent.memory
        builder = agent._resolve_request_builder()
        wc = WorkingContext()
        ctx = make_test_ctx(llm=FakeLLM("ok"))
        for i in range(n):
            asyncio.run(builder.build(f"q{i}", wc, ctx))
        return mem.calls

    once = Agent(name="a", model="m", memory=_CountingMemory())
    every = Agent(name="a", model="m", memory=_CountingMemory(), reground_every_turn=True)
    assert _turns(once, 3) == 1, "default must ground once for the whole run"
    assert _turns(every, 3) == 3, "the flag must reach the grounder, not just the field"
