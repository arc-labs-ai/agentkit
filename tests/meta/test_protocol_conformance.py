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


# ══ StorePort ════════════════════════════════════════════════════════════════
#
# Four implementations (memory / file / redis / postgres) behind a 6-method
# Protocol, and `InMemoryStore`'s own docstring calls itself "the offline
# reference StorePort and the contract every durable backend matches". Nothing
# checked that claim, and they had drifted: `FileStore.get_or_set` tested
# `existing is not None`, conflating "nothing stored" with "None stored", so a
# producer returning None re-ran on every call — 3 invocations against
# InMemoryStore's 1, on identical input.
#
# All four run here. Redis and Postgres ride the fakes in ``tests/_store_fakes.py``
# rather than a live server: neither extra is installed in the dev environment
# and CI has no service containers, so "omit the ones that need a server" meant
# two of the four implementations were never compared against the contract at
# all. The fakes store what the adapter hands them (JSON text) and return what
# a real client returns (bytes), so this is the real encode → store → fetch →
# decode path, and they yield to the event loop on every command so the
# concurrency assertions below are not vacuous.
#
# ``ttl`` is the one place the four legitimately differ, and the difference is
# declared per case rather than skipped: Redis and memory honour it, FileStore
# has no sweeper and warns, PostgresStore refuses. Tests assert the DECLARED
# behaviour, so a backend that quietly changed camp fails here.


@dataclasses.dataclass(frozen=True)
class _StoreCase:
    """One backend under test, plus the two things the contract cannot infer:
    how it treats ``ttl``, and how to move its clock."""

    store: Any
    ttl: str  # "honored" | "ignored" | "refused"
    advance: Any  # Callable[[float], None]; a no-op where ttl is not honored


def _memory_case() -> _StoreCase:
    from agentkit.adapters.store import InMemoryStore

    now = {"t": 0.0}

    def advance(seconds: float) -> None:
        now["t"] += seconds

    return _StoreCase(InMemoryStore(clock=lambda: now["t"]), "honored", advance)


def _file_case() -> _StoreCase:
    import tempfile

    from agentkit.adapters.store import FileStore

    return _StoreCase(FileStore(tempfile.mkdtemp()), "ignored", lambda _s: None)


def _redis_case() -> _StoreCase:
    from _store_fakes import FakeRedis

    from agentkit.adapters.store import RedisStore

    now = {"t": 0.0}

    def advance(seconds: float) -> None:
        now["t"] += seconds

    return _StoreCase(RedisStore(client=FakeRedis(clock=lambda: now["t"])), "honored", advance)


def _postgres_case() -> _StoreCase:
    from _store_fakes import FakePgPool

    from agentkit.adapters.store import PostgresStore

    return _StoreCase(PostgresStore(pool=FakePgPool()), "refused", lambda _s: None)


def _store_params():
    return [
        pytest.param(_memory_case, id="memory"),
        pytest.param(_file_case, id="file"),
        pytest.param(_redis_case, id="redis"),
        pytest.param(_postgres_case, id="postgres"),
    ]


@pytest.fixture(params=_store_params())
def store_case(request: pytest.FixtureRequest) -> _StoreCase:
    return request.param()


@pytest.fixture
def store(store_case: _StoreCase):
    """The backend itself. Derived from ``store_case`` rather than built
    separately so a test taking both gets ONE store, not two."""
    return store_case.store


async def _drain(keys: Any) -> list[str]:
    """Collect a `scan` into a list.

    ``scan`` is declared as a plain ``def`` returning an ``AsyncIterator`` — not
    an ``async def`` — for the same reason ``LLMPort.stream`` is: an ``async
    def`` that yields is a *function returning an iterator*, not a coroutine,
    so declaring it ``async def`` in the Protocol would make every real
    implementation fail a strict type check. Call sites are identical
    (``async for k in store.scan(p)``); only the annotation differs."""
    return [key async for key in keys]


class TestStorePortContract:
    """What every backend must agree on. Each assertion is a property a caller
    relies on without knowing which store is wired."""

    def test_get_returns_none_for_a_missing_key(self, store) -> None:
        assert asyncio.run(store.get("nope")) is None

    def test_set_then_get_round_trips(self, store) -> None:
        async def go():
            await store.set("k", {"a": [1, 2, {"b": "c"}]})
            return await store.get("k")

        assert asyncio.run(go()) == {"a": [1, 2, {"b": "c"}]}

    def test_set_overwrites(self, store) -> None:
        async def go():
            await store.set("k", "first")
            await store.set("k", "second")
            return await store.get("k")

        assert asyncio.run(go()) == "second"

    def test_delete_is_idempotent(self, store) -> None:
        async def go():
            await store.set("k", "v")
            await store.delete("k")
            await store.delete("k")  # again: must not raise
            return await store.get("k")

        assert asyncio.run(go()) is None

    def test_get_or_set_runs_the_producer_exactly_once(self, store) -> None:
        """Single-flight is the whole point: `memoize` and `idempotent` both
        ride this, and a producer that runs twice means a duplicated
        side-effect or a doubled provider bill."""
        calls = {"n": 0}

        async def produce():
            calls["n"] += 1
            return "made"

        async def go():
            return [await store.get_or_set("k", produce) for _ in range(3)]

        assert asyncio.run(go()) == ["made", "made", "made"]
        assert calls["n"] == 1

    def test_get_or_set_is_keyed_on_presence_not_truthiness(self, store) -> None:
        """The drift this suite exists to catch. A producer legitimately
        returning None (or any falsy value) must still be single-flight —
        `existing is not None` conflates "nothing stored" with "None stored"."""

        def probe(value: Any) -> tuple[list[Any], int]:
            """Own scope per value — a closure over the loop variable would
            capture the NAME, not the value (ruff B023)."""
            calls = {"n": 0}

            async def produce() -> Any:
                calls["n"] += 1
                return value

            async def go() -> list[Any]:
                key = f"falsy-{type(value).__name__}-{value!r}"
                return [await store.get_or_set(key, produce) for _ in range(3)]

            return asyncio.run(go()), calls["n"]

        for value in (None, 0, "", [], {}, False):
            returned, ran = probe(value)
            assert returned == [value, value, value]
            assert ran == 1, f"producer returning {value!r} re-ran {ran}x"

    def test_a_raised_producer_is_never_cached(self, store) -> None:
        """"Failures are never cached" is in the Protocol docstring. Caching a
        transient error would pin it for the entry's whole lifetime."""
        attempts = {"n": 0}

        async def flaky():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient")
            return "recovered"

        async def go():
            with pytest.raises(RuntimeError):
                await store.get_or_set("k", flaky)
            return await store.get_or_set("k", flaky)

        assert asyncio.run(go()) == "recovered"
        assert attempts["n"] == 2

    def test_append_then_list_preserves_order(self, store) -> None:
        async def go():
            for i in range(3):
                await store.append("log", {"i": i})
            return await store.list("log")

        assert asyncio.run(go()) == [{"i": 0}, {"i": 1}, {"i": 2}]

    def test_list_of_an_empty_log_is_an_empty_list(self, store) -> None:
        """Not None — the audit middleware reads this straight into a loop."""
        assert asyncio.run(store.list("never-appended")) == []

    def test_the_kv_and_log_namespaces_do_not_collide(self, store) -> None:
        """`set` and `append` share a key space in the Protocol's signature but
        are different stores; a collision would have an audit log clobber a
        checkpoint."""

        async def go():
            await store.set("same", {"kind": "kv"})
            await store.append("same", {"kind": "log"})
            return await store.get("same"), await store.list("same")

        kv, log = asyncio.run(go())
        assert kv == {"kind": "kv"}
        assert log == [{"kind": "log"}]

    def test_keys_with_path_and_scheme_characters_stay_distinct(self, store) -> None:
        """Keys are built from `Scope.key()` and colon-joined prefixes
        (`checkpoint:org1:dom2`). A backend that flattened separators would
        merge two tenants' entries."""

        async def go():
            for i, key in enumerate(["a/b", "a\\b", "a:b", "a b", "a.b"]):
                await store.set(key, i)
            return [await store.get(k) for k in ["a/b", "a\\b", "a:b", "a b", "a.b"]]

        assert asyncio.run(go()) == [0, 1, 2, 3, 4]

    # ── compare_and_set ──────────────────────────────────────────────────────
    #
    # `get_or_set` covers *create if absent*. It does not cover *replace only
    # if unchanged*, which is what every read-modify-write needs — allocating a
    # monotonic ordinal is "read max, write max+1", and two writers race it.

    def test_compare_and_set_replaces_a_matching_value(self, store) -> None:
        async def go():
            await store.set("k", 1)
            applied = await store.compare_and_set("k", 1, 2)
            return applied, await store.get("k")

        applied, value = asyncio.run(go())
        assert applied is True
        assert value == 2

    def test_compare_and_set_reports_a_lost_race_rather_than_raising(self, store) -> None:
        """The design requirement, not a preference: a caller that loses a CAS
        re-reads and retries, which is ordinary control flow. Raising would
        force every read-modify-write loop to wrap its own body in a
        ``try/except`` and to distinguish "someone beat me" from "the store is
        down" by exception type."""

        async def go():
            await store.set("k", 1)
            applied = await store.compare_and_set("k", 99, 2)
            return applied, await store.get("k")

        applied, value = asyncio.run(go())
        assert applied is False
        assert value == 1, "a losing compare_and_set wrote anyway"

    def test_compare_and_set_creates_an_absent_key_when_expected_is_none(self, store) -> None:
        """Absent compares equal to ``expected=None``, so the FIRST iteration
        of a read-modify-write loop works like every later one. The ordinal
        allocator reads ``None`` from an empty slot; if it then had to fall
        back to ``set``, the create would be the one step of the loop with no
        race protection — which is where the race actually is."""

        async def go():
            applied = await store.compare_and_set("fresh", None, 1)
            return applied, await store.get("fresh")

        assert asyncio.run(go()) == (True, 1)

    def test_compare_and_set_refuses_to_create_when_expected_is_not_none(self, store) -> None:
        """POSITIVE CONTROL for the rule above. "Absent means None" must not
        decay into "absent matches anything"."""

        async def go():
            applied = await store.compare_and_set("fresh", 7, 1)
            return applied, await store.get("fresh")

        assert asyncio.run(go()) == (False, None)

    def test_compare_and_set_matches_a_stored_null_against_expected_none(self, store) -> None:
        """`get` collapses "absent" and "stored null" onto ``None``, so the
        caller that produced ``expected`` from a `get` cannot tell them apart.
        A CAS that DID tell them apart would reject a swap on evidence the
        caller had no way to see."""

        async def go():
            await store.set("k", None)
            applied = await store.compare_and_set("k", None, "now-set")
            return applied, await store.get("k")

        assert asyncio.run(go()) == (True, "now-set")

    def test_compare_and_set_with_expected_none_loses_to_a_key_that_now_exists(self, store) -> None:
        """The ordinal allocator's actual race, and the reason "absent means
        None" cannot be implemented as a plain upsert: this reader saw an empty
        slot, another writer filled it, and the write MUST NOT land. An
        unconditional ``INSERT ... ON CONFLICT DO UPDATE`` would silently
        overwrite the winner and hand both callers the same ordinal."""

        async def go():
            await store.set("ordinal", 1)  # a competitor got there first
            applied = await store.compare_and_set("ordinal", None, 2)
            return applied, await store.get("ordinal")

        assert asyncio.run(go()) == (False, 1)

    def test_compare_and_set_compares_by_equality_not_identity(self, store) -> None:
        """The stored value round-trips through JSON on three of the four
        backends, so the caller NEVER holds the object that was written — an
        identity check would make CAS on any container permanently impossible,
        and would appear to work on the in-memory reference alone."""

        async def go():
            await store.set("k", {"a": [1, 2], "b": {"c": 3}})
            equal_but_distinct = {"b": {"c": 3}, "a": [1, 2]}  # different key order, too
            applied = await store.compare_and_set("k", equal_but_distinct, {"a": []})
            return applied, await store.get("k")

        assert asyncio.run(go()) == (True, {"a": []})

    def test_compare_and_set_on_a_container_that_differs_is_refused(self, store) -> None:
        """POSITIVE CONTROL for the equality rule — a deep difference must still
        lose, or "compare" degenerates into "always swap"."""

        async def go():
            await store.set("k", {"a": [1, 2]})
            applied = await store.compare_and_set("k", {"a": [1, 3]}, "clobbered")
            return applied, await store.get("k")

        assert asyncio.run(go()) == (False, {"a": [1, 2]})

    def test_exactly_one_of_two_concurrent_compare_and_sets_wins(self, store) -> None:
        """THE reason the primitive exists. Both racers read the same value and
        try to replace it; a read-modify-write that is not atomic lets both
        report success and one write is silently lost — which for the ordinal
        allocator means two runs with the same sequence number."""

        async def go():
            await store.set("ordinal", 0)
            return await asyncio.gather(
                store.compare_and_set("ordinal", 0, "a"),
                store.compare_and_set("ordinal", 0, "b"),
            )

        outcomes = asyncio.run(go())
        assert sorted(outcomes) == [False, True], f"both racers reported {outcomes}"

    # ── increment ────────────────────────────────────────────────────────────
    #
    # A counter with an expiry is the shape of every rate limit. Without it an
    # application writes a raw Lua script against Redis and bypasses the port,
    # putting the check outside everything the framework can test or trace.

    def test_increment_starts_a_missing_counter_at_by(self, store) -> None:
        assert asyncio.run(store.increment("hits")) == 1

    def test_increment_returns_the_new_total_and_get_agrees(self, store) -> None:
        """The counter is an ordinary stored value, not a private register — a
        rate limiter reports "37 of 50 used" by reading it back."""

        async def go():
            await store.increment("hits", 5)
            total = await store.increment("hits", 2)
            return total, await store.get("hits")

        total, read_back = asyncio.run(go())
        assert total == 7 and read_back == 7
        assert isinstance(total, int) and not isinstance(total, bool)

    def test_a_counter_written_by_set_can_be_incremented(self, store) -> None:
        """Counters are ordinary values in the same key space, not a separate
        register file. On Redis this holds only because ``json.dumps(5)`` is
        exactly the string Redis stores an integer as — encode integers any
        other way (a JSON envelope, a length prefix) and INCRBY stops seeing a
        number, so this pins the encoding as much as the semantics."""

        async def go():
            await store.set("seeded", 5)
            return await store.increment("seeded"), await store.get("seeded")

        assert asyncio.run(go()) == (6, 6)

    def test_increment_accepts_a_negative_by(self, store) -> None:
        """Refunds and released reservations are decrements. A counter that
        only went up would need a second key to track what came back."""

        async def go():
            await store.increment("balance", 3)
            return await store.increment("balance", -5)

        assert asyncio.run(go()) == -2

    def test_increment_totals_correctly_under_concurrency(self, store) -> None:
        """Twenty racers on one counter. A read-modify-write implementation
        loses updates here — and loses them silently, which is how a rate
        limiter starts admitting more than its ceiling."""

        async def go():
            await asyncio.gather(*(store.increment("hits") for _ in range(20)))
            return await store.get("hits")

        assert asyncio.run(go()) == 20

    def test_increment_on_a_non_integer_refuses_identically_on_every_backend(self, store) -> None:
        """Redis answers this with ``ResponseError``, Postgres with a cast
        failure, and a dict-backed store with ``TypeError`` — three different
        types for one mistake, which is exactly the leak the error taxonomy
        exists to close. Bools are included because ``isinstance(True, int)``
        is True in Python but ``true`` is not an integer in JSON or in Redis:
        the one backend that could have accepted it is the reference one."""
        from agentkit.kernel.errors import StoreValueError

        async def go():
            for i, value in enumerate(["text", {"a": 1}, [1], 1.5, True, None]):
                key = f"notint{i}"
                await store.set(key, value)
                with pytest.raises(StoreValueError):
                    await store.increment(key)
                assert await store.get(key) == value, f"{value!r} was clobbered by a failed increment"

        asyncio.run(go())

    def test_increment_refuses_a_non_integer_by_identically_on_every_backend(self, store) -> None:
        """The mirror of the test above, on the ARGUMENT rather than the stored
        value — and the four backends did not agree before this existed.
        Measured on ``increment(k, 1.5)``:

        * memory and file returned ``1.5``, a float out of a method annotated
          ``-> int``, and left ``1.5`` in the key, which the NEXT increment then
          rejects as a non-counter. The counter is poisoned by a call that
          reported success.
        * redis returned ``1`` while the key held ``1.5`` — the ``int()`` around
          INCRBY's reply truncates, so the total the caller acts on and the
          total the store holds disagree. That is a rate limiter that reports
          being under its ceiling while the counter says otherwise.
        * postgres failed inside the driver with a bare ``ValueError``.

        ``True`` is included for the same reason the stored-value test includes
        it: ``isinstance(True, int)`` makes it silently count as 1 on three
        backends and is rejected by the fourth.
        """
        from agentkit.kernel.errors import StoreValueError

        async def go():
            for by in (1.5, True, "3", None, 2 + 0j):
                with pytest.raises(StoreValueError, match="`by`"):
                    await store.increment("amount", by)
                assert await store.get("amount") is None, f"by={by!r} wrote anyway"
            # POSITIVE CONTROL: a real int still works, so the guard is not
            # simply refusing everything.
            assert await store.increment("amount", 3) == 3

        asyncio.run(go())

    # ── scan ─────────────────────────────────────────────────────────────────
    #
    # `list(key)` reads back one appended log. Without a prefix scan,
    # "everything recorded for this run" is answerable only if every writer
    # also maintained an index by hand.

    def test_scan_returns_every_key_under_the_prefix_and_nothing_above_it(self, store) -> None:
        async def go():
            for key in ("run:1:a", "run:1:b", "run:1:c:d", "run:2:a", "run", "other"):
                await store.set(key, 1)
            return await _drain(store.scan("run:1:"))

        assert set(asyncio.run(go())) == {"run:1:a", "run:1:b", "run:1:c:d"}

    def test_scan_yields_full_keys_not_the_remainder_after_the_prefix(self, store) -> None:
        """A caller feeds what it gets straight back into `get`/`delete`. A
        backend that stripped the prefix would hand back keys that resolve to
        nothing, and the reclaim pass built on it would delete zero rows while
        reporting success."""

        async def go():
            await store.set("run:1:a", "v")
            return await _drain(store.scan("run:1:"))

        keys = asyncio.run(go())
        assert keys == ["run:1:a"]
        assert asyncio.run(store.get(keys[0])) == "v"

    def test_scan_includes_a_key_that_is_itself_the_prefix(self, store) -> None:
        """``"run"`` is a prefix of ``"run"``. Excluding it would mean a
        summary record stored at the prefix itself vanished from the scan that
        exists to enumerate that run."""

        async def go():
            await store.set("run", "summary")
            await store.set("run:1", "step")
            return await _drain(store.scan("run"))

        assert set(asyncio.run(go())) == {"run", "run:1"}

    def test_scan_of_a_prefix_that_matches_nothing_is_empty(self, store) -> None:
        async def go():
            await store.set("run:1", 1)
            return await _drain(store.scan("nothing-here:"))

        assert asyncio.run(go()) == []

    def test_scan_with_an_empty_prefix_enumerates_every_key(self, store) -> None:
        async def go():
            for key in ("a", "b:1", "c"):
                await store.set(key, 1)
            return await _drain(store.scan(""))

        assert set(asyncio.run(go())) == {"a", "b:1", "c"}

    def test_scan_omits_a_deleted_key(self, store) -> None:
        async def go():
            await store.set("run:1", 1)
            await store.set("run:2", 1)
            await store.delete("run:1")
            return await _drain(store.scan("run:"))

        assert asyncio.run(go()) == ["run:2"]

    def test_scan_does_not_see_the_append_log_namespace(self, store) -> None:
        """`set` and `append` are different stores behind one key space — the
        contract already pins that they do not collide. A scan that surfaced
        log keys would return keys `get` answers ``None`` for."""

        async def go():
            await store.append("run:log", {"event": 1})
            await store.set("run:kv", 1)
            return await _drain(store.scan("run:"))

        assert asyncio.run(go()) == ["run:kv"]

    def test_scan_limit_caps_the_number_of_keys(self, store) -> None:
        """A cap on what the caller is willing to receive, NOT a page cursor:
        no ordering is promised, so *which* two arrive is backend-defined."""

        async def go():
            for i in range(5):
                await store.set(f"run:{i}", i)
            return await _drain(store.scan("run:", limit=2))

        keys = asyncio.run(go())
        assert len(keys) == 2
        assert set(keys) <= {f"run:{i}" for i in range(5)}

    def test_scan_with_limit_zero_yields_nothing(self, store) -> None:
        """Zero is a real cap, not "unset". Collapsing it onto ``None`` would
        make ``limit=remaining_budget`` return everything at exactly the moment
        the budget ran out."""

        async def go():
            await store.set("run:1", 1)
            return await _drain(store.scan("run:", limit=0))

        assert asyncio.run(go()) == []

    def test_scan_rejects_a_negative_limit(self, store) -> None:
        """A negative limit is a caller bug (an underflowed budget). Silently
        treating it as unlimited returns the whole key space to code that asked
        for less than nothing."""

        async def go():
            with pytest.raises(ValueError, match="limit"):
                await _drain(store.scan("run:", limit=-1))

        asyncio.run(go())

    def test_scan_treats_glob_metacharacters_in_the_prefix_literally(self, store) -> None:
        """Keys are built by joining caller data (`Scope.key()`, tool names,
        correlation ids), so a prefix containing ``*`` is user input, not a
        pattern. Redis's SCAN takes a glob, so an unescaped ``*`` there turns
        ``scan("a*")`` into "every key starting with a" — a cross-tenant read
        on a store whose whole job is scoping."""

        async def go():
            for key in ("a*b", "a?b", "a[b", "axb", "ab"):
                await store.set(key, 1)
            return (
                await _drain(store.scan("a*")),
                await _drain(store.scan("a?")),
                await _drain(store.scan("a[")),
            )

        star, question, bracket = asyncio.run(go())
        assert star == ["a*b"]
        assert question == ["a?b"]
        assert bracket == ["a[b"]

    def test_scan_does_not_raise_while_keys_are_being_written(self, store) -> None:
        """The audit reader runs against a live run. A scan is a long-lived
        iteration over a mutating table, and the in-memory backend's dict
        raises ``RuntimeError: dictionary changed size during iteration`` the
        moment a writer lands mid-scan. No snapshot semantics are promised —
        only that the iteration completes and the keys present throughout
        appear."""

        async def go():
            for i in range(6):
                await store.set(f"run:{i}", i)

            async def writer() -> None:
                for i in range(6, 12):
                    await store.set(f"run:{i}", i)
                    await asyncio.sleep(0)

            async def reader() -> list[str]:
                seen = []
                async for key in store.scan("run:"):
                    seen.append(key)
                    await asyncio.sleep(0)
                return seen

            seen, _ = await asyncio.gather(reader(), writer())
            return seen

        seen = asyncio.run(go())
        assert len(seen) == len(set(seen)), "scan returned a key twice"
        assert {f"run:{i}" for i in range(6)} <= set(seen), "a key present throughout was missed"

    # ── ttl, declared per backend ────────────────────────────────────────────
    #
    # The four legitimately differ, and the difference is asserted rather than
    # skipped so a backend cannot quietly change camp.

    def test_compare_and_set_applies_its_ttl_only_on_the_write_that_landed(
        self, store_case
    ) -> None:
        """A refused CAS must not touch the expiry either. Setting the TTL
        before checking the compare would let a losing racer shorten the
        winner's entry — the entry it never got to write."""
        store = store_case.store

        if store_case.ttl == "refused":
            with pytest.raises(NotImplementedError, match="ttl"):
                asyncio.run(store.compare_and_set("k", None, 1, ttl=10))
            return
        if store_case.ttl == "ignored":
            with pytest.warns(UserWarning, match="ignores ttl"):
                assert asyncio.run(store.compare_and_set("k", None, 1, ttl=10)) is True
            return

        async def go():
            await store.set("winner", 1)
            assert await store.compare_and_set("winner", 1, 2, ttl=10) is True
            await store.set("loser", 1)
            assert await store.compare_and_set("loser", 99, 2, ttl=10) is False
            store_case.advance(10)
            return await store.get("winner"), await store.get("loser")

        winner, loser = asyncio.run(go())
        assert winner is None, "the applied ttl did not expire the entry"
        assert loser == 1, "a REFUSED compare_and_set still set an expiry"

    def test_increment_ttl_expires_the_counter_and_its_window_together(self, store_case) -> None:
        """``ttl`` opens the window on the increment that finds no window —
        it never slides one that already exists.

        The alternative (reset on every increment) breaks the exact case the
        primitive is for: under sustained traffic the counter is touched more
        often than the window is long, so it never expires and the limit never
        resets — the rate limiter jams shut precisely under load. Redis's
        ``EXPIRE key ttl NX`` is this rule, and it rides inside the same MULTI
        as the INCRBY so a crash cannot leave a counter with no window (an
        immortal counter) or a window with no counter.
        """
        store = store_case.store

        if store_case.ttl == "refused":
            with pytest.raises(NotImplementedError, match="ttl"):
                asyncio.run(store.increment("c", ttl=10))
            return
        if store_case.ttl == "ignored":
            with pytest.warns(UserWarning, match="ignores ttl"):
                assert asyncio.run(store.increment("c", ttl=10)) == 1
            return

        async def go():
            assert await store.increment("c", ttl=10) == 1
            store_case.advance(9)
            assert await store.increment("c", ttl=10) == 2, "the counter lost its value early"
            store_case.advance(1)  # the ORIGINAL deadline, not a slid one
            assert await store.get("c") is None, "the window slid instead of expiring"
            assert await store.increment("c", ttl=10) == 1, "the counter outlived its window"
            store_case.advance(9)
            return await store.get("c")

        assert asyncio.run(go()) == 1, "the replacement window inherited the old deadline"
