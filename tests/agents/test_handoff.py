"""`Handoff` — the typed coordination verb for multi-agent coordinator Agents.

Covers:
  * dataclass shape (frozen / immutable) + render round-trip.
  * `parse_handoff` parse rules (last-marker-wins, target/reason/message split, no marker → None).
  * `handoff_tool(...)` builds a FunctionTool whose schema constrains `target` to an enum.
  * Tool execution returns the rendered Handoff marker; unknown targets fail closed.
  * `route_by_handoff` end-to-end on a coordinator `Agent` + `SelectorPolicy`:
      - marker-only path: agent A's reply names B → next turn is B.
      - tool-call path: agent A invokes the handoff_tool → tool result + agent echo route to B.

Offline & deterministic via `FakeLLM` and `FakeLLM.script`.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.agents import Agent, SelectorPolicy
from agentkit.agents.cognition import CoordinatorCognition, ReActCognition
from agentkit.agents.control import (
    HANDOFF_MARKER,
    Handoff,
    MaxMessages,
    handoff_tool,
    parse_handoff,
    route_by_handoff,
)
from agentkit.kernel.types import Scope, ToolCall
from agentkit.runtime import Budget
from agentkit.testing import FakeLLM, Turn, make_test_ctx
from agentkit.tools import FunctionTool


def _run(coro):
    return asyncio.run(coro)


_BUDGET = Budget(max_cost_usd=10.0)
_SCOPE = Scope(1, 2)


# ---- dataclass shape ---------------------------------------------------------------------------


def test_handoff_is_frozen():
    """The dataclass is immutable — coordination state is data, not a mutable handle."""
    h = Handoff(target="bob", reason="needs SQL help")
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        h.target = "alice"  # type: ignore[misc]


def test_handoff_defaults():
    h = Handoff(target="bob")
    assert h.target == "bob"
    assert h.reason == ""
    assert h.message == ""


def test_handoff_render_roundtrips_through_parse():
    """The marker emitted by `render()` is exactly what `parse_handoff` reads back."""
    h = Handoff(target="bob", reason="needs SQL help", message="The query is in scratchpad.")
    back = parse_handoff(h.render())
    assert back == h  # frozen dataclasses compare by value


# ---- parse_handoff rules -----------------------------------------------------------------------


def test_parse_handoff_bare_target():
    """`HANDOFF:bob` with nothing else → target only, empty reason/message."""
    h = parse_handoff("HANDOFF:bob")
    assert h == Handoff(target="bob")


def test_parse_handoff_target_and_reason():
    """Tokens AFTER the target on the SAME line collapse into `reason`."""
    h = parse_handoff("HANDOFF:bob explain this")
    assert h == Handoff(target="bob", reason="explain this")


def test_parse_handoff_target_reason_and_message():
    """Lines AFTER the marker line become the `message` payload (trailing/leading ws stripped)."""
    h = parse_handoff("HANDOFF:bob explain this\nHere's what I found so far.")
    assert h == Handoff(target="bob", reason="explain this", message="Here's what I found so far.")


def test_parse_handoff_no_marker_returns_none():
    assert parse_handoff("no marker here") is None
    assert parse_handoff("") is None


def test_parse_handoff_last_marker_wins():
    """If an agent thinks aloud and then commits to a handoff at the bottom, the bottom wins —
    matches the existing `handoff_selector`'s `rfind` behaviour so swapping selectors is safe."""
    text = "Maybe HANDOFF:alice would help\nActually, HANDOFF:bob is better"
    assert parse_handoff(text) == Handoff(target="bob", reason="is better")


def test_parse_handoff_marker_with_nothing_after_it_returns_none():
    """`HANDOFF:` followed by no token is not a valid handoff."""
    assert parse_handoff("HANDOFF:   ") is None
    assert parse_handoff("ending with HANDOFF:") is None


def test_parse_handoff_custom_marker():
    """The marker is configurable — useful when the default collides with prose."""
    assert parse_handoff("[[ROUTE:bob]] now", marker="ROUTE:") == Handoff(
        target="bob]]", reason="now"
    )


# ---- handoff_tool: schema + execution ----------------------------------------------------------


def test_handoff_tool_builds_function_tool_with_enum_target():
    """The schema constrains `target` to the roster — the model literally can't invent a name."""
    t = handoff_tool(["alice", "bob"])
    assert isinstance(t, FunctionTool)
    assert t.name == "handoff"
    assert t.side_effecting is False  # a coordination verb, not an external mutation
    assert t.idempotent is True
    props = t.schema.parameters["properties"]
    assert props["target"] == {"type": "string", "enum": ["alice", "bob"]}
    assert props["reason"] == {"type": "string"}
    assert props["message"] == {"type": "string"}
    assert t.schema.parameters["required"] == ["target"]


def test_handoff_tool_execution_returns_marker_string():
    """Tool execution emits the rendered `HANDOFF:<target> <reason>` marker so the model sees
    it on the next turn AND the coordinator selector can parse it from the final assistant
    reply."""
    t = handoff_tool(["alice", "bob"])
    out = _run(t.run({"target": "bob", "reason": "SQL"}, ctx=None))
    assert out.startswith(HANDOFF_MARKER + "bob")
    assert "SQL" in out
    # Round-trips through parse_handoff: the tool's output IS the marker.
    assert parse_handoff(out) == Handoff(target="bob", reason="SQL")


def test_handoff_tool_rejects_unknown_target():
    """If the model invents a target despite the enum, fail closed: no routable marker emitted."""
    t = handoff_tool(["alice", "bob"])
    out = _run(t.run({"target": "ghost", "reason": "x"}, ctx=None))
    assert "rejected" in out
    assert parse_handoff(out) is None  # nothing for the selector to route on


def test_handoff_tool_empty_targets_is_an_error():
    """An empty roster makes the tool useless — surface at construction time, not call time."""
    with pytest.raises(ValueError):
        handoff_tool([])


# ---- route_by_handoff: coordinator `Agent` + `SelectorPolicy` end-to-end -----------------------


def _children():
    return {"alice": Agent("alice", "m"), "bob": Agent("bob", "m")}


def _coord(children, *, termination=MaxMessages(2)):
    return Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=children,
            policy=SelectorPolicy(selector=route_by_handoff(default="alice")),
            termination=termination,
        ),
    )


def test_route_by_handoff_routes_to_named_target_from_marker():
    """End-to-end: agent A emits `HANDOFF:bob` in its reply → next turn is agent B."""
    # Task starts with HANDOFF:bob → first turn is bob. Bob then emits HANDOFF:alice → turn 2 alice.
    coord = _coord(_children())
    res = _run(
        coord.run(
            "start HANDOFF:bob",
            make_test_ctx(llm=FakeLLM("ok HANDOFF:alice"), scope=_SCOPE, budget=_BUDGET),
        )
    )
    assert [m.name for m in res.evals["messages"][1:]] == ["bob", "alice"]


def test_route_by_handoff_no_marker_falls_back_to_default():
    """Without a marker on the last message, the configured default speaks (deterministic routing)."""
    coord = _coord(_children())
    res = _run(
        coord.run(
            "just start",
            make_test_ctx(llm=FakeLLM("nothing special here"), scope=_SCOPE, budget=_BUDGET),
        )
    )
    # Turn 0: no marker on the task → default=alice. Turn 1: alice's reply has no marker → alice again.
    assert [m.name for m in res.evals["messages"][1:]] == ["alice", "alice"]


def test_route_by_handoff_no_marker_no_default_falls_back_to_round_robin():
    """`default=""` → return None → SelectorPolicy's own round-robin fallback. Never stalls."""
    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children=_children(),
            policy=SelectorPolicy(selector=route_by_handoff()),  # default=""
            termination=MaxMessages(2),
        ),
    )
    res = _run(
        coord.run("just start", make_test_ctx(llm=FakeLLM("nothing"), scope=_SCOPE, budget=_BUDGET))
    )
    # round-robin order
    assert [m.name for m in res.evals["messages"][1:]] == ["alice", "bob"]


def test_route_by_handoff_via_tool_call_routes_to_target():
    """Tool-call path: agent A is wired with `handoff_tool(["bob"])` and a scripted FakeLLM that
    (1) calls the tool, (2) emits a final reply that surfaces the handoff marker into the
    coordinator's transcript. The selector reads the marker off A's final assistant message and
    routes to B.

    This is the harder test — it exercises the tool registry, the tool-loop iteration, and the
    fact that the tool's result text propagates into the assistant's final reply so the
    coordinator-level selector (which only sees the coordinator transcript, not the agent's inner
    messages) can still route correctly."""
    tool = handoff_tool(["alice", "bob"])
    alice = Agent(name="alice", model="m", cognition=ReActCognition(tools=[tool]))
    bob = Agent(name="bob", model="m")

    # Single FakeLLM for all calls. Alice's two turns first (tool_call → final), then bob's turn.
    # FakeLLM.script replays in order across chat calls.
    llm = FakeLLM.script(
        [
            Turn(
                tool_calls=(ToolCall("c1", "handoff", {"target": "bob", "reason": "SQL help"}),),
            ),
            Turn(content="Handing off. HANDOFF:bob SQL help"),
            Turn(content="bob speaking, taking over"),
        ]
    )

    coord = Agent(
        name="team",
        cognition=CoordinatorCognition(
            children={"alice": alice, "bob": bob},
            policy=SelectorPolicy(selector=route_by_handoff(default="alice")),
            termination=MaxMessages(2),
        ),
    )
    res = _run(coord.run("start the task", make_test_ctx(llm=llm, scope=_SCOPE, budget=_BUDGET)))
    # Turn 0: no marker on task → default=alice; alice runs the tool loop and her final output
    # contains HANDOFF:bob. Turn 1: selector sees alice's marker → bob.
    assert [m.name for m in res.evals["messages"][1:]] == ["alice", "bob"]
    # Alice's reply (in the coordinator transcript) must carry the marker for the selector to route.
    assert "HANDOFF:bob" in res.evals["messages"][1].content


def test_handoff_tool_accepts_mappingproxy_arguments():
    """``handoff_tool`` extracts ``target``/``reason``/``message`` from
    a ``MappingProxyType`` — the read-only view a ``ToolCall`` wraps
    its arguments in. Matching only on ``dict`` would default every
    field to ``""`` and reject every handoff as ``"'' is not a known
    target"``, silently breaking the handoff-via-tool path."""
    import asyncio
    from types import MappingProxyType

    from agentkit.agents.control.handoff import handoff_tool

    tool = handoff_tool(targets=["specialist", "critic"])
    args = MappingProxyType({"target": "specialist", "reason": "domain expert", "message": ""})

    result = asyncio.run(tool.fn(args, None))
    assert "HANDOFF:specialist" in result
    assert "domain expert" in result
