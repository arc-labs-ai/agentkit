"""What the two fan-out sites do with a concurrency bound they cannot honour.

WHY THIS FILE EXISTS
--------------------
``ToolSet.max_concurrency`` and ``DelegationPolicy.max_concurrency`` are both
refused below one at construction, and
``tests/source/unit/core/runtime/test_budget_decimal_and_verdict.py`` plus
``tests/source/architecture/test_protocol_conformance.py`` already pin that
refusal and the primitive behind it (``asyncio.Semaphore(0)`` is born locked).
Neither of them asks the question this file asks: what happens at the site that
actually BUILDS the semaphore, when the value arriving there is not one the
constructor ever saw.

Two ways it is not:

* ``agentkit/core/runtime/calls.py`` reads the bound off a duck-typed policy
  object — ``getattr(tools, "max_concurrency", 4)`` — because an adapter or a
  host may hand the loop any object with ``schemas()``/``execute()`` (see
  ``test_cancelled_tool_batch_closes_pending_pairs_and_retains_confirmed_results``
  in ``tests/source/conformance/test_calls.py``, which does exactly that).
  ``ToolSet.__post_init__`` cannot speak for an object it never constructed, so
  the fan-out site is the ONLY gate on that path.
* A frozen, validated policy can still be written through
  ``object.__setattr__`` — which is not an exotic trick but how a dataclass
  writes its own frozen fields (``AgentDefinition.__post_init__`` uses it), and
  therefore how a rehydrating store can too.

The failure being excluded is the worst-shaped one in the framework: a
``Semaphore(0)`` admits nobody and nothing ever releases it, so the run does not
fail, it stops — no exception, no log line, no stack naming the configuration
that did it. The historical near-miss on the other side is just as bad in a
quieter way: ``calls.py`` used to clamp with ``max(1, ...)``, so a
misconfigured fan-out ran fully SERIALISED and answered normally, and the
caller's bound was silently replaced by one they never asked for.

Nothing here hangs to prove a hang. The refusal is asserted by what the run
reports, and the locked-semaphore property it exists to avoid is already pinned
on the primitive in the two files named above.
"""

from __future__ import annotations

import asyncio
import dataclasses
from decimal import Decimal
from typing import Any

import pytest

from agentkit.agents.definitions.agent import AgentDefinition, DelegationPolicy
from agentkit.agents.definitions.instructions import InstructionSet
from agentkit.agents.definitions.store import MemoryAgentStore
from agentkit.agents.execution.runner import AgentRunner
from agentkit.core.runtime.calls import CallRuntime
from agentkit.core.testing import make_test_ctx
from agentkit.models.contracts.constants import EventKind, FailureClass
from agentkit.models.contracts.types import Binding, CallLimits, Turn
from agentkit.tools import ToolRegistry
from agentkit.tools.execution.policy import ToolResult, ToolSet
from tests.source.conformance.test_calls import PRICES, ScriptedAdapter

_CALLS = ({"id": "a", "name": "first", "arguments": {}}, {"id": "b", "name": "second", "arguments": {}})


class _HostTools:
    """The duck-typed tool policy the runtime's fan-out actually supports.

    Deliberately NOT a ``ToolSet``: the point of every test using it is that
    ``calls.py`` reads ``max_concurrency`` off whatever object it was handed,
    so a constructor-time guard on one class cannot be the enforcement.
    """

    def __init__(self, **attributes: Any) -> None:
        self.executed: list[str] = []
        for name, value in attributes.items():
            setattr(self, name, value)

    def schemas(self) -> list[Any]:
        return []

    async def execute(self, name: str, *_: Any) -> ToolResult:
        self.executed.append(name)
        return ToolResult(({"type": "text", "text": f"{name} ran"},))


@pytest.mark.asyncio
async def test_a_fan_out_bound_of_zero_is_reported_instead_of_quietly_serialising_the_batch() -> None:
    """A bound of zero reaching the fan-out has exactly two honest outcomes, and
    the site used to choose neither: ``max(1, ...)`` turned it into one permit,
    so a policy that said "no concurrency at all" ran every tool one after
    another and the run ANSWERED. The caller is told nothing, their configured
    bound is gone, and the only visible symptom is latency.

    The alternative the clamp was hiding is worse: without it the same value
    builds ``Semaphore(0)``, which is locked from birth, and the batch waits
    forever. So the bound has to be refused at the fan-out, before any call is
    announced, with a reason a host can act on.

    Asserted on the duck-typed policy object because that is the shape the
    constructor cannot defend: this object was never validated by anything."""
    tools = _HostTools(max_concurrency=0)
    adapter = ScriptedAdapter(
        Turn(tool_calls=_CALLS, raw_usage={"input_tokens": 3, "output_tokens": 1}),
        # Scripted so the CLAMPED behaviour would succeed: serialise the batch,
        # come back for another turn and answer. The refusal has to be what
        # stops this run, not a transport that ran out of turns.
        Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )
    events: list[Any] = []

    # Bounded, and the bound is the whole point: a refusal is immediate, while a
    # bound that was TRUSTED builds `Semaphore(0)` and parks here with nothing
    # to release it. That is reported as a failed assertion in seconds rather
    # than as a suite-level timeout 60 seconds later, and nothing in this test
    # ever waits on the closed pool itself — the same discipline as
    # `test_the_refusal_is_what_a_zero_permit_pool_would_otherwise_do`.
    answer = await asyncio.wait_for(
        CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).run(
            Binding("openai_api", "test"), "diagnose", tools, events=events.append, call_id="zero-permits"
        ),
        timeout=10,
    )

    assert answer.outcome == "failed"
    assert answer.failure is not None
    assert answer.failure.classification == FailureClass.TOOL_ERROR
    assert "concurrency" in answer.failure.mode_detail["reason"]
    assert "0" in answer.failure.mode_detail["reason"], "the reason has to name the value that was refused"
    assert not tools.executed, "a refused fan-out must not dispatch anything"
    assert len(adapter.requests) == 1, "the run stopped at the refusal, not after another model turn"
    assert not [event for event in events if event.kind == EventKind.TOOL_CALL], (
        "the bound is policy, so it is refused before any call is announced to an observer"
    )


@pytest.mark.asyncio
async def test_one_permit_is_still_a_legitimate_bound_and_dispatches_every_call() -> None:
    """POSITIVE CONTROL for the refusal above, and the boundary it must not
    swallow. Serialised execution is a configuration a host chooses on purpose —
    tools that must not overlap — so ``1`` has to keep working. Without this
    test, "refuse anything that cannot fan out" could quietly widen into
    "refuse anything that cannot fan out IN PARALLEL" and the narrowest
    legitimate policy in the framework would start failing runs."""
    tools = _HostTools(max_concurrency=1)
    adapter = ScriptedAdapter(
        Turn(tool_calls=_CALLS, raw_usage={"input_tokens": 3, "output_tokens": 1}),
        Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )

    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).run(
        Binding("openai_api", "test"), "diagnose", tools, call_id="one-permit"
    )

    assert answer.outcome == "answered"
    assert tools.executed == ["first", "second"]


@pytest.mark.asyncio
async def test_a_policy_object_that_declares_no_bound_at_all_still_fans_out() -> None:
    """The other half of the duck-typed read: ``getattr(tools, "max_concurrency",
    4)`` has a DEFAULT, and an adapter-supplied policy that declares no bound
    relies on it. A refusal written as "the attribute must be present and valid"
    would break every such object — they are the majority of the objects this
    seam serves — so the default has to survive the guard that rejects zero."""
    tools = _HostTools()
    adapter = ScriptedAdapter(
        Turn(tool_calls=_CALLS, raw_usage={"input_tokens": 3, "output_tokens": 1}),
        Turn(("done",), raw_usage={"input_tokens": 1, "output_tokens": 1}),
    )

    answer = await CallRuntime(adapters={"openai_api": adapter}, pricing=PRICES).run(
        Binding("openai_api", "test"), "diagnose", tools, call_id="default-permits"
    )

    assert not hasattr(tools, "max_concurrency")
    assert answer.outcome == "answered"
    assert sorted(tools.executed) == ["first", "second"]


def test_a_validated_tool_policy_cannot_have_its_bound_rewritten_after_validation() -> None:
    """``ToolSet.__post_init__`` refuses a bound below one, and that refusal was
    worth nothing while the dataclass stayed mutable: ``tools.max_concurrency =
    0`` after construction never reaches ``__post_init__`` at all, so the value
    the fan-out reads was never the value anything checked.

    Frozen is the only form of that guarantee which holds for every field. It
    also states what ``ToolSet`` IS — the execution policy a host hands across
    the tool seam, alongside the already-frozen ``OverflowPolicy`` and
    ``ToolResult`` in the same module — rather than a scratch object a caller
    edits between turns."""
    tools = ToolSet(registry=ToolRegistry(), max_concurrency=2)

    with pytest.raises(dataclasses.FrozenInstanceError):
        tools.max_concurrency = 0  # type: ignore[misc]

    # Not a field-specific guard: the whole policy is a value, so the deadline
    # and the overflow bound cannot be rewritten behind the run either.
    with pytest.raises(dataclasses.FrozenInstanceError):
        tools.timeout_s = 0.0  # type: ignore[misc]

    assert tools.max_concurrency == 2
    # ...and `replace` remains the supported way to vary one, which is what
    # `with_registry` already does per invocation.
    assert dataclasses.replace(tools, max_concurrency=3).max_concurrency == 3


@pytest.mark.asyncio
async def test_freezing_the_tool_policy_leaves_deferred_discovery_state_working() -> None:
    """The one thing freezing could plausibly have broken. ``ToolSet`` carries
    per-invocation discovery state (``_loaded``) that ``search_tools`` writes,
    and a frozen dataclass forbids REBINDING a field, not mutating the object a
    field holds — so the set is still updated in place. Pinned because the
    obvious "fix" for a frozen value, swapping the set for a tuple, would turn
    search-before-use into a permanent refusal and hide every deferred tool for
    the rest of the run."""

    async def lookup_record(key: str) -> str:
        """Look up a record by its unique key and return its stored value."""
        return key

    tools = ToolSet(registry=ToolRegistry.from_tools([lookup_record]), deferred=("lookup_record",))

    assert [schema.name for schema in tools.schemas()] == ["search_tools", "call_loaded_tool"]
    await tools.execute("search_tools", {"query": "record"}, make_test_ctx())

    assert "lookup_record" in [schema.name for schema in tools.schemas()]


@pytest.mark.asyncio
async def test_the_delegation_fan_out_refuses_a_bound_written_past_the_frozen_policy() -> None:
    """The second fan-out site, and the one with no clamp to hide behind:
    ``runner.py`` builds ``asyncio.Semaphore(definition.delegation.max_concurrency)``
    straight from the policy. ``DelegationPolicy`` is frozen and validated, so
    a caller cannot type a zero — but ``object.__setattr__`` writes a frozen
    field anyway, which is how dataclasses write their own (see
    ``AgentDefinition.__post_init__``) and therefore how anything rehydrating a
    stored definition can too.

    Were the bound trusted there, that zero would become a delegation pool that
    admits no child and releases nothing: the first ``delegate`` call parks and
    the agent never answers. Refused before the model is dispatched instead —
    the adapter below is scripted with NO turns, so it would raise if the run
    reached it, which is how this test proves the refusal is upstream of
    dispatch rather than a failed delegation after the fact."""
    policy = DelegationPolicy(("alpha@1",))
    object.__setattr__(policy, "max_concurrency", 0)
    store = MemoryAgentStore()
    await store.put(AgentDefinition("alpha", "1", "A delegated worker."))
    definition = AgentDefinition(
        "coordinator",
        "1",
        "Coordinate authorized workers.",
        limits=CallLimits(max_cost_usd=Decimal("1")),
        delegation=policy,
    )

    async with CallRuntime(adapters={"openai_api": ScriptedAdapter()}, pricing=PRICES) as runtime:
        runner = AgentRunner(runtime, agents=store, tools=ToolRegistry())
        with pytest.raises(ValueError, match="concurrency"):
            await runner.run_agent(
                definition,
                InstructionSet("Delegate through the hosted tool."),
                {},
                binding=Binding("openai_api", "test"),
                call_id="delegation-zero-permits",
            )
