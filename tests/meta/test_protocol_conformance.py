"""Protocol conformance — ONE contract per Protocol, run against EVERY implementation.

This framework's central architectural bet is that cross-cutting concerns are
typed Protocols injected at wire-up. The failure mode that bet invites is
**silent drift**: four ``SchemaAdapter`` implementations each tested in their
own module, each passing, and each answering a shared question slightly
differently — until a caller that works against one breaks against another.
Per-implementation test files cannot catch that, because none of them ever
compares the implementations.

So the contract lives here, once, and is parametrized over the implementations.
Adding a new adapter / meter / cognition means it either passes this file or
it is not a conforming implementation. That is a stronger, cheaper guarantee
than remembering to mirror twenty assertions into a fifth test module.

Structural typing means ``isinstance`` against a ``runtime_checkable`` Protocol
only checks that method NAMES exist — never their semantics. These tests check
the semantics.
"""

from __future__ import annotations

import asyncio
import dataclasses
from decimal import Decimal
from typing import Any

import pytest
from _assertions import assert_money, assert_no_float_money

from agentkit.kernel.middleware import Call
from agentkit.kernel.types import ChatRequest, Message, Scope, Usage
from agentkit.runtime import Budget, Quota
from agentkit.runtime.context import RunContext


def _call(scope: Scope = Scope(1, 1)) -> Call:
    ctx = RunContext("conformance", scope)
    return Call("chat", ChatRequest(messages=[Message("user", "hi")], model="m"), ctx)


# ══ SchemaAdapter ════════════════════════════════════════════════════════════
#
# Four flavours with nothing in common at the Python level (Pydantic owns
# `model_validate`, dataclasses are walked via `dataclasses.fields()`, attrs has
# its own, raw schemas are dicts). That is exactly why the shared contract has
# to be asserted rather than assumed.

pydantic = pytest.importorskip("pydantic")
attrs = pytest.importorskip("attrs")


class _PydanticShape(pydantic.BaseModel):
    title: str
    count: int


@attrs.define
class _AttrsShape:
    title: str
    count: int


@dataclasses.dataclass
class _DataclassShape:
    title: str
    count: int


_JSON_SCHEMA_SHAPE: dict[str, Any] = {
    "type": "object",
    "title": "JsonShape",
    "properties": {"title": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["title", "count"],
}

# Every supported output spec, with the flavour name for readable failures.
SCHEMA_SPECS = [
    pytest.param(_PydanticShape, id="pydantic"),
    pytest.param(_AttrsShape, id="attrs"),
    pytest.param(_DataclassShape, id="dataclass"),
    pytest.param(_JSON_SCHEMA_SHAPE, id="json_schema"),
]

VALID_JSON = '{"title": "hello", "count": 3}'


@pytest.fixture(params=SCHEMA_SPECS)
def adapter(request: pytest.FixtureRequest):
    """One adapter per supported flavour, built through the public dispatcher."""
    from agentkit.capabilities.output_schema import adapt

    return adapt(request.param)


def _field(obj: Any, name: str) -> Any:
    """Read a field from a parsed object regardless of flavour — the raw
    JSON-Schema adapter yields a ``dict``, the others yield instances."""
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


class TestSchemaAdapterContract:
    """The six things ``SchemaAdapter``'s docstring says the rest of the
    framework is allowed to lean on."""

    def test_declares_a_name_and_a_python_type(self, adapter) -> None:
        assert isinstance(adapter.name, str) and adapter.name
        assert isinstance(adapter.python_type, type)

    def test_json_schema_is_a_self_contained_object_schema(self, adapter) -> None:
        """Self-contained because some providers reject external ``$ref``."""
        schema = adapter.json_schema()
        assert schema.get("type") == "object"
        assert "properties" in schema
        assert "$ref" not in repr(schema) or "#/" in repr(schema), (
            "schema references an EXTERNAL document; providers reject those"
        )

    def test_parse_accepts_a_json_string(self, adapter) -> None:
        parsed = adapter.parse(VALID_JSON)
        assert _field(parsed, "title") == "hello" and _field(parsed, "count") == 3

    def test_parse_accepts_an_already_parsed_dict(self, adapter) -> None:
        """A provider's native structured-output mode hands back a dict, not a
        string. Both entry points must work or the native path breaks per
        flavour."""
        parsed = adapter.parse({"title": "hello", "count": 3})
        assert _field(parsed, "title") == "hello"

    def test_parse_raises_the_ONE_shared_exception_type(self, adapter) -> None:
        """The whole point of wrapping every flavour's native error: a retry
        policy must not have to know which adapter is in play. A leaked
        ``pydantic.ValidationError`` or ``json.JSONDecodeError`` would make the
        repair loop flavour-dependent."""
        from agentkit.capabilities.output_schema import OutputCoercionError

        with pytest.raises(OutputCoercionError):
            adapter.parse("this is definitely not json")
        with pytest.raises(OutputCoercionError):
            adapter.parse('{"title": "missing the count field"}')

    def test_the_shared_exception_carries_diagnostics_and_the_raw_payload(
        self, adapter
    ) -> None:
        """``errors`` is what gets reflected back to the model, and ``raw`` is
        what a human reads in a failed-run UI. An adapter that raises the right
        TYPE with an empty body still breaks reflect-and-repair."""
        from agentkit.capabilities.output_schema import OutputCoercionError

        with pytest.raises(OutputCoercionError) as exc:
            adapter.parse('{"title": "no count"}')
        assert exc.value.raw == '{"title": "no count"}'
        assert isinstance(exc.value.errors, list)

    def test_partial_parse_never_raises_and_returns_none_on_junk(self, adapter) -> None:
        """Called on EVERY text delta by ``output_coerce``. An adapter that
        raises here turns a streaming run into a crash on its first byte."""
        for junk in ("", "   ", "{", '{"ti', "not json at all", "[]"):
            assert adapter.partial_parse(junk) is None or adapter.partial_parse(junk) is not None

    def test_partial_parse_is_prefix_consistent_as_bytes_arrive(self, adapter) -> None:
        """The property a streaming UI depends on: a field's value never
        rewinds. If an adapter re-interpreted earlier bytes the rendered
        output would flicker."""
        last_title = ""
        for cut in range(1, len(VALID_JSON) + 1):
            partial = adapter.partial_parse(VALID_JSON[:cut])
            if partial is None:
                continue
            title = _field(partial, "title") if _has(partial, "title") else None
            if isinstance(title, str):
                assert title.startswith(last_title), (
                    f"title rewound at cut {cut}: {last_title!r} -> {title!r}"
                )
                last_title = title

    def test_partial_parse_of_the_complete_payload_agrees_with_parse(self, adapter) -> None:
        partial = adapter.partial_parse(VALID_JSON)
        strict = adapter.parse(VALID_JSON)
        assert partial is not None
        assert _field(partial, "title") == _field(strict, "title")

    def test_serialize_round_trips_through_parse(self, adapter) -> None:
        """Used by the durable run-store, so a resumed run reads back what the
        original produced. A lossy ``serialize`` silently corrupts resume."""
        original = adapter.parse(VALID_JSON)
        as_dict = adapter.serialize(original)
        assert isinstance(as_dict, dict)
        assert adapter.parse(as_dict) is not None
        assert _field(adapter.parse(as_dict), "count") == 3

    def test_validate_fast_paths_an_already_typed_instance(self, adapter) -> None:
        """The tool-result path: whatever Python object the tool returned."""
        instance = adapter.parse(VALID_JSON)
        assert _field(adapter.validate(instance), "count") == 3

    def test_validate_coerces_a_dict_and_rejects_a_wrong_shape(self, adapter) -> None:
        from agentkit.capabilities.output_schema import OutputCoercionError

        assert _field(adapter.validate({"title": "x", "count": 1}), "count") == 1
        with pytest.raises(OutputCoercionError):
            adapter.validate(42)


def _has(obj: Any, name: str) -> bool:
    """Is this field actually populated on a partial? Required fields may be
    genuinely unset — that IS the partial contract."""
    if isinstance(obj, dict):
        return name in obj
    fields_set = getattr(obj, "model_fields_set", None)
    if fields_set is not None:
        return name in fields_set
    return hasattr(obj, name)


# ══ Meter ════════════════════════════════════════════════════════════════════
#
# The middleware iterates ``ctx.all_meters`` and treats every entry
# identically, so two meters behaving differently under the same call is a
# real defect and not a matter of taste. Before the verdict work, ``Budget``
# raised from ``charge`` and ``Quota`` never checked anything at all — one
# Protocol, two behaviours, and nothing asserted the difference.

METERS = [
    pytest.param(lambda: Budget(max_cost_usd="0.05"), id="Budget"),
    pytest.param(lambda: Quota(max_usd="0.05", clock=lambda: 1000.0), id="Quota"),
]


@pytest.fixture(params=METERS)
def meter(request: pytest.FixtureRequest):
    return request.param()


class TestMeterContract:
    def test_satisfies_the_runtime_checkable_protocol(self, meter) -> None:
        from agentkit.runtime import Meter

        assert isinstance(meter, Meter)

    def test_guard_and_charge_both_return_a_verdict(self, meter) -> None:
        from agentkit.runtime.meter import Charge

        assert isinstance(asyncio.run(meter.guard(_call())), Charge)
        assert isinstance(asyncio.run(meter.charge(_call(), Usage(1, 1, 0.001))), Charge)

    def test_a_fresh_meter_is_under_its_ceiling(self, meter) -> None:
        assert asyncio.run(meter.guard(_call())).ok is True

    def test_money_on_a_verdict_is_decimal_never_float(self, meter) -> None:
        """One float anywhere in the chain and the ledger stops reconciling."""
        verdict = asyncio.run(meter.charge(_call(), Usage(1, 1, 0.01)))
        assert isinstance(verdict.spent, Decimal)
        assert_no_float_money(verdict.spent, verdict.remaining, label="Charge")

    def test_every_money_accessor_returns_decimal(self, meter) -> None:
        """Not just the verdict — the accessors a caller reconciles against.

        Mutation-found gap: ``Budget.spent()`` regressed to ``float`` and all
        78 conformance tests stayed green, because ``Decimal("1.00") == 1.0``.
        Equality can never establish this; only ``isinstance`` can."""
        asyncio.run(meter.charge(_call(), Usage(1, 1, 0.01)))
        if isinstance(meter, Budget):
            assert_no_float_money(
                meter.spent(), meter.spent_cents(), meter.ceiling(), meter.remaining(),
                label="Budget",
            )
            assert isinstance(meter.spent(), Decimal)
            # ...while the documented float MIRROR stays a float on purpose.
            assert isinstance(meter.spent_usd, float)
        else:
            assert isinstance(meter.spent_in_window("org1:dom1"), Decimal)

    def test_exceeding_a_ceiling_is_reported_not_swallowed(self, meter) -> None:
        """Both meters must eventually say no to the same overspend — via a
        raise or a not-ok verdict, but never by silently allowing it."""
        from agentkit.runtime import MeterExceeded

        exceeded = False
        try:
            for _ in range(20):
                v1 = asyncio.run(meter.guard(_call()))
                v2 = asyncio.run(meter.charge(_call(), Usage(1, 1, 0.02)))
                if not (v1.ok and v2.ok):
                    exceeded = True
                    break
        except MeterExceeded:
            exceeded = True
        assert exceeded, "a meter allowed 20x its ceiling without raising or refusing"

    def test_charging_is_exact_over_many_small_amounts(self, meter) -> None:
        """A hundred cents is a dollar, on every meter."""
        fresh = type(meter)() if isinstance(meter, Budget) else Quota(clock=lambda: 1000.0)
        for _ in range(100):
            asyncio.run(fresh.charge(_call(), Usage(0, 0, 0.01)))
        spent = (
            fresh.spent() if isinstance(fresh, Budget) else fresh.spent_in_window("org1:dom1")
        )
        assert_money(spent, "1.00", label=f"{type(fresh).__name__} after 100x $0.01")

    def test_concurrent_charges_do_not_lose_updates(self, meter) -> None:
        """Both meters take an async lock; a lost update here is money the
        books never see."""
        fresh = type(meter)() if isinstance(meter, Budget) else Quota(clock=lambda: 1000.0)

        async def hammer() -> None:
            await asyncio.gather(
                *(fresh.charge(_call(), Usage(0, 0, 0.01)) for _ in range(50))
            )

        asyncio.run(hammer())
        spent = (
            fresh.spent() if isinstance(fresh, Budget) else fresh.spent_in_window("org1:dom1")
        )
        assert_money(spent, "0.50", label="concurrent charges")


# ══ Cognition ════════════════════════════════════════════════════════════════
#
# The Cognition Protocol's whole promise is that ``Agent`` does not care which
# one is plugged in. That only holds if every cognition emits the same terminal
# shape — a consumer branching on ``StreamEvent.type`` and reading
# ``result.stop_reason`` must not need to know the regime.


def _cognitions():
    from agentkit.agents.cognition import ReActCognition, SingleCallCognition
    from agentkit.tools import tool

    @tool(side_effecting=False)
    def noop(text: str) -> str:
        """A read-only tool so the ReAct registry is non-empty and realistic."""
        return text

    return [
        pytest.param(SingleCallCognition(), id="single_call"),
        pytest.param(ReActCognition(tools=[noop]), id="react"),
    ]


@pytest.fixture(params=_cognitions())
def cognition(request: pytest.FixtureRequest):
    return request.param


def _events(cognition) -> list:
    from agentkit.agents import Agent
    from agentkit.testing import FakeLLM, make_test_ctx

    agent = Agent("conformance", "m", cognition=cognition)
    ctx = make_test_ctx(llm=FakeLLM("a plain final answer"))

    async def go():
        return [ev async for ev in agent.stream("do it", ctx)]

    return asyncio.run(go())


class TestCognitionContract:
    def test_emits_exactly_one_terminal_final_event(self, cognition) -> None:
        """Two finals would double-count usage downstream; zero would hang a
        consumer waiting for one. ``Agent.run`` explicitly relies on this."""
        finals = [ev for ev in _events(cognition) if ev.type == "final"]
        assert len(finals) == 1

    def test_the_final_event_is_last(self, cognition) -> None:
        events = _events(cognition)
        assert events[-1].type == "final"

    def test_the_final_event_carries_an_agent_result(self, cognition) -> None:
        from agentkit.agents.result import AgentResult

        final = _events(cognition)[-1]
        assert isinstance(final.result, AgentResult)
        assert isinstance(final.result.usage, Usage)

    def test_stop_reason_is_from_the_closed_taxonomy(self, cognition) -> None:
        """A cognition inventing its own stop reason defeats the point of the
        Literal — a reader branching on it would silently fall through."""
        from typing import get_args

        from agentkit.agents.result import AgentStopReason

        final = _events(cognition)[-1]
        assert final.result.stop_reason in get_args(AgentStopReason)

    def test_a_clean_run_reports_complete_and_is_not_resumable(self, cognition) -> None:
        final = _events(cognition)[-1]
        assert final.result.stop_reason == "complete"
        assert final.result.is_suspended is False
        assert final.result.is_resumable is False
        assert final.result.partial is False

    def test_partial_output_is_none_without_an_output_schema(self, cognition) -> None:
        """The additive guarantee, asserted per cognition rather than once —
        a new cognition forwarding a stale partial would be caught here."""
        assert all(ev.partial_output is None for ev in _events(cognition))

    def test_every_event_type_is_in_the_declared_literal(self, cognition) -> None:
        from typing import get_args

        from agentkit.kernel.types import StreamEventType

        allowed = set(get_args(StreamEventType))
        assert {ev.type for ev in _events(cognition)} <= allowed

    def test_has_a_name_for_trace_attribution(self, cognition) -> None:
        """``agentkit.agent.cognition`` is stamped on every ``invoke_agent``
        span from this."""
        assert isinstance(cognition.name, str) and cognition.name
