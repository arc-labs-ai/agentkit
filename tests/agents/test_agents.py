"""Smoke tests for the `Agent` + `Policy` shape — the cross-cutting surface.
A leaf `Agent` is the single-call form; a coordinator `Agent` dispatches to
`children` per its `Policy`. `RunPolicy` (Rule of Two) is the run-wide safety
check."""

import asyncio

import pytest

from agentkit import Agent, RunPolicy
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.policies.plan import PlanPolicy, StaticPlanner, Step
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


def test_agent_result_evals_stay_a_plain_mutable_dict() -> None:
    """POSITIVE CONTROL, and the constraint this commit is under: `evals` is
    NOT frozen into a `MappingProxyType`. Callers write into it, it is JSON
    serialised, and `dataclasses.asdict` must keep working — a proxy is neither
    JSON-serialisable nor an `asdict` leaf. Passes before and after."""
    import dataclasses
    import json

    r = _res(evals={"stop_reason": "complete"})
    assert isinstance(r.evals, dict)
    r.evals["late"] = {"written": "after construction"}  # the documented habit
    assert json.dumps(r.evals)
    assert dataclasses.asdict(r)["evals"]["late"] == {"written": "after construction"}


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


def test_workflow_result_outputs_stay_a_plain_mutable_dict() -> None:
    """POSITIVE CONTROL: nothing was frozen. Node outputs are read by index and
    serialised; the durable-resume contract depends on them staying plain."""
    import json

    from agentkit.agents.result import WorkflowResult
    from agentkit.kernel.types import Usage

    r = WorkflowResult(outputs={"draft": {"body": "x"}}, usage=Usage(), steps=1, stop_reason="complete")
    assert isinstance(r.outputs, dict)
    assert r.outputs["draft"]["body"] == "x"
    assert json.dumps(r.outputs)
    assert r == WorkflowResult(
        outputs={"draft": {"body": "x"}}, usage=Usage(), steps=1, stop_reason="complete"
    )
    assert r != WorkflowResult(
        outputs={"draft": {"body": "y"}}, usage=Usage(), steps=1, stop_reason="complete"
    )
