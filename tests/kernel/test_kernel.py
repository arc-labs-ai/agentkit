"""L0 kernel: middleware composition order, resilience classification/retry, bounded concurrency."""

import asyncio
import datetime as _dt
import uuid as _uuid
from dataclasses import dataclass

import pytest

from agentkit.kernel.concurrency import gather_bounded
from agentkit.kernel.middleware import Call
from agentkit.kernel.middleware import chain as compose
from agentkit.kernel.resilience import (
    CircuitBreaker,
    CircuitOpen,
    ErrorClass,
    classify,
    idempotency_key,
    run_with_resilience,
    stable_hash,
)


def _run(coro):
    return asyncio.run(coro)


async def _nosleep(d):
    pass


def test_compose_orders_outer_to_inner_and_back():
    order = []

    def mw(tag):
        async def m(call, nxt):
            order.append(f"{tag}>")
            r = await nxt(call)
            order.append(f"<{tag}")
            return r

        return m

    async def terminal(call):
        order.append("T")
        return "done"

    handler = compose([mw("a"), mw("b")], terminal)
    out = _run(handler(Call("x", None, None)))
    assert out == "done"
    assert order == ["a>", "b>", "T", "<b", "<a"]  # a outermost, then b, then terminal


def test_compose_empty_returns_terminal():
    async def terminal(call):
        return 42

    assert _run(compose([], terminal)(Call("x", None, None))) == 42


def test_classify_transient_permanent_unknown():
    assert classify(TimeoutError("timed out")) is ErrorClass.TRANSIENT
    assert classify(ValueError("400 invalid")) is ErrorClass.PERMANENT
    assert classify(RuntimeError("weird")) is ErrorClass.UNKNOWN


def test_run_with_resilience_retries_transient_then_succeeds():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("timed out")
        return "ok"

    assert _run(run_with_resilience(fn, max_attempts=3, sleep=_nosleep)) == "ok"
    assert calls["n"] == 2


def test_run_with_resilience_fails_fast_on_permanent():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise ValueError("400 invalid")

    with pytest.raises(ValueError):
        _run(run_with_resilience(fn, max_attempts=5, sleep=_nosleep))
    assert calls["n"] == 1


def test_gather_bounded_preserves_order_and_bounds():
    sem = asyncio.Semaphore(2)

    async def job(i):
        await asyncio.sleep(0)
        return i * 10

    assert _run(gather_bounded([job(i) for i in range(5)], sem=sem)) == [0, 10, 20, 30, 40]


def test_circuit_breaker_opens_after_threshold():
    br = CircuitBreaker("x", fail_threshold=2)
    assert br.allow()
    br.record_failure()
    br.record_failure()  # → OPEN
    assert not br.allow()


def test_circuit_breaker_half_open_admits_exactly_one_probe():
    """Regression: once cooled-down, the breaker admits ONE probe; concurrent/subsequent callers are
    refused until that probe resolves — they must not all stampede the recovering dependency."""
    t = {"now": 0.0}
    br = CircuitBreaker("x", fail_threshold=1, cooldown=10.0, clock=lambda: t["now"])
    br.record_failure()  # → OPEN
    assert not br.allow()  # still cooling down
    t["now"] = 10.0  # cooldown elapsed
    assert br.allow()  # the single probe is admitted (→ half_open)
    assert not br.allow()  # a second caller is refused while the probe is in flight
    assert not br.allow()
    br.record_success()  # probe succeeded → CLOSED
    assert br.allow() and br.allow()  # fully recovered, all admitted


# ── stable_hash determinism ──────────────────────────────────────────────────
#
# ``_stable_default`` handles wire-relevant types explicitly and falls back to
# ``type-qualname + sorted __dict__``. This keeps hashes deterministic across
# runs / processes for any value — a plain-class ``str(o)`` fallback would
# include the memory address (``<Foo object at 0x...>``) and cause
# ``idempotency_key``, ``memoize``, ``semantic_memoize``, and audit
# fingerprints to silently drift.


def test_stable_hash_deterministic_for_primitives():
    """Same primitive value hashes the same on repeat."""
    a = {"q": "hello", "n": 42, "flag": True, "items": [1, 2, 3]}
    assert stable_hash(a) == stable_hash(a)


def test_stable_hash_handles_datetime_uuid_path_decimal():
    """Common non-JSON types (datetime, UUID, Path, Decimal) serialize
    deterministically."""
    import decimal
    import pathlib

    when = _dt.datetime(2026, 6, 28, 12, 0, 0, tzinfo=_dt.UTC)
    uid = _uuid.UUID("12345678-1234-5678-1234-567812345678")
    h1 = stable_hash(
        {"when": when, "id": uid, "p": pathlib.Path("/x/y"), "d": decimal.Decimal("1.5")}
    )
    h2 = stable_hash(
        {"when": when, "id": uid, "p": pathlib.Path("/x/y"), "d": decimal.Decimal("1.5")}
    )
    assert h1 == h2


def test_stable_hash_plain_class_no_memory_address():
    """Two distinct instances of the same plain class with the same state
    must hash identically — a ``str(o)`` fallback would include
    ``0x<addr>`` and yield different hashes for two ``Foo(42)`` values."""

    class Foo:
        def __init__(self, x: int) -> None:
            self.x = x

    f1, f2 = Foo(42), Foo(42)
    assert stable_hash(f1) == stable_hash(f2)


def test_stable_hash_plain_class_different_state_differs():
    """Conversely: different state → different hash. The fallback isn't a
    type-only constant — it incorporates sorted __dict__."""

    class Foo:
        def __init__(self, x: int) -> None:
            self.x = x

    assert stable_hash(Foo(42)) != stable_hash(Foo(43))


def test_stable_hash_set_order_independent():
    """Set / frozenset hash on sorted-element basis, so iteration order
    can't cause cache-key drift."""
    assert stable_hash({1, 2, 3}) == stable_hash({3, 2, 1})
    assert stable_hash(frozenset({"b", "a"})) == stable_hash(frozenset({"a", "b"}))


def test_stable_hash_dataclass_handled_as_dict():
    """Dataclass instances stable-serialize to their fields, so dataclass
    args to a tool round-trip cleanly through ``idempotency_key``."""

    @dataclass
    class Q:
        text: str
        max: int

    assert stable_hash(Q("hi", 3)) == stable_hash(Q("hi", 3))
    assert stable_hash(Q("hi", 3)) != stable_hash(Q("hi", 4))


def test_stable_hash_enum_value_used():
    """Enums serialize as ``{__enum__, value}`` so two members compare on
    value, not on default repr."""
    import enum as _e

    class K(_e.Enum):
        A = "a"
        B = "b"

    assert stable_hash(K.A) == stable_hash(K.A)
    assert stable_hash(K.A) != stable_hash(K.B)


def test_stable_hash_length_param():
    """``length=`` caps the hex output. The default is 16."""
    h = stable_hash({"x": 1}, length=8)
    assert len(h) == 8
    assert len(stable_hash({"x": 1})) == 16
    assert len(stable_hash({"x": 1}, length=24)) == 24


def test_idempotency_key_prefix_and_determinism():
    """The public ``idempotency_key`` prefixes ``idem:`` and is stable
    across calls for the same tuple."""
    k1 = idempotency_key("run1", "scope", "tool", {"arg": 1})
    k2 = idempotency_key("run1", "scope", "tool", {"arg": 1})
    assert k1 == k2
    assert k1.startswith("idem:")
    assert idempotency_key("run1", "scope", "tool", {"arg": 2}) != k1


def test_stable_hash_bytes_use_hex():
    """Bytes serialize to hex — deterministic and JSON-safe."""
    assert stable_hash(b"\x00\x01\xff") == stable_hash(b"\x00\x01\xff")
    assert stable_hash(b"\x00\x01\xff") != stable_hash(b"\x00\x01\xfe")


# ── classify priority + breaker hygiene ──────────────────────────────────────
#
# Three coupled invariants:
#   (a) classify TRANSIENT wins on collision — a ``ValidationError("timed
#       out")`` classifies as TRANSIENT so the retry loop actually retries.
#   (b) ``run_with_resilience`` only bumps the breaker on non-PERMANENT
#       errors (a 401 isn't an upstream health signal).
#   (c) When the breaker opens mid-retry, ``CircuitOpen`` chains the
#       original transient cause via ``__cause__`` so postmortems can see
#       what actually broke.


def test_classify_transient_wins_on_collision():
    """A ValidationError whose message contains 'timed out' must classify
    as TRANSIENT. A PERMANENT-first scan would match 'validation' in the
    class name and skip retrying a transient upstream problem."""

    class ValidationError(Exception):
        pass

    assert classify(ValidationError("request timed out")) is ErrorClass.TRANSIENT
    # Defensive parallel: a 5xx with 'invalid' in the message still
    # classifies as TRANSIENT.
    assert (
        classify(RuntimeError("503 ServiceUnavailable: upstream returned invalid response"))
        is ErrorClass.TRANSIENT
    )


def test_classify_pure_permanent_still_permanent():
    """A clean PERMANENT error with no TRANSIENT substring is still
    PERMANENT — the order change is collision-only, not a wholesale flip."""
    assert classify(ValueError("400 invalid")) is ErrorClass.PERMANENT
    assert classify(PermissionError("403 forbidden")) is ErrorClass.PERMANENT
    assert classify(RuntimeError("content filter triggered")) is ErrorClass.PERMANENT


def test_breaker_skips_permanent_errors():
    """A run of PERMANENT errors must NOT trip the breaker — those are
    contract failures, not health signals."""
    t = {"now": 0.0}

    def clk() -> float:
        return t["now"]

    br = CircuitBreaker(name="upstream", fail_threshold=3, cooldown=1.0, clock=clk)

    async def fn():
        raise ValueError("400 invalid")  # PERMANENT

    # Five permanent failures, each fails fast — but the breaker counter
    # never advances.
    for _ in range(5):
        with pytest.raises(ValueError):
            _run(run_with_resilience(fn, breaker=br, max_attempts=3, sleep=_nosleep))
    assert br.state == "closed"
    assert br._fails == 0


def test_breaker_still_trips_on_transient_errors():
    """Symmetric to the above: TRANSIENT failures DO count. This pins
    the breaker still works on the failures it's meant to track."""
    t = {"now": 0.0}

    def clk() -> float:
        return t["now"]

    br = CircuitBreaker(name="upstream", fail_threshold=2, cooldown=10.0, clock=clk)

    async def fn():
        raise TimeoutError("timed out")  # TRANSIENT

    # 3 attempts × TRANSIENT — first two bumps trip the breaker (threshold=2);
    # the third allow() refuses and raises CircuitOpen. Either escape
    # exception means the breaker did its job; what matters is the state.
    with pytest.raises((TimeoutError, CircuitOpen)):
        _run(run_with_resilience(fn, breaker=br, max_attempts=3, sleep=_nosleep))
    assert br.state == "open"


def test_circuit_open_preserves_original_cause():
    """When the breaker trips mid-retry-loop, ``CircuitOpen`` must chain
    the underlying transient via ``__cause__`` so callers can see what
    actually broke. Previously the cause was discarded and operators
    saw only ``CircuitOpen("circuit open: name")`` with no upstream."""
    t = {"now": 0.0}

    def clk() -> float:
        return t["now"]

    # Threshold=1 + half-open=False-on-second-allow makes the breaker
    # trip after the FIRST failure and refuse the second attempt.
    br = CircuitBreaker(name="upstream", fail_threshold=1, cooldown=60.0, clock=clk)

    async def fn():
        raise TimeoutError("upstream timed out")

    with pytest.raises(CircuitOpen) as ei:
        _run(run_with_resilience(fn, breaker=br, max_attempts=3, sleep=_nosleep))
    # The original TimeoutError must be reachable via __cause__.
    assert isinstance(ei.value.__cause__, TimeoutError)
    assert "upstream timed out" in str(ei.value.__cause__)


# ── Property tests: Usage arithmetic ─────────────────────────────────────────
#
# ``Usage`` is the run-wide accounting value. It's used inside merges
# (child usage + parent usage), inside caches (deltas fold into a
# terminal usage), and inside meters (per-turn accumulation). The
# arithmetic must obey algebraic laws so those callers can reorder
# and re-group additions without observable drift.

from hypothesis import given
from hypothesis import strategies as st

from agentkit.kernel.types import (
    ChatRequest,
    Chunk,
    Delta,
    LLMResult,
    Message,
    StreamEvent,
    ToolCall,
    ToolRequest,
    ToolSchema,
    Usage,
)

_usage_strategy = st.builds(
    Usage,
    input_tokens=st.integers(min_value=0, max_value=10_000_000),
    output_tokens=st.integers(min_value=0, max_value=10_000_000),
    cost_usd=st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    cache_read_tokens=st.integers(min_value=0, max_value=10_000_000),
    cache_write_tokens=st.integers(min_value=0, max_value=10_000_000),
)


@given(a=_usage_strategy, b=_usage_strategy, c=_usage_strategy)
def test_usage_add_is_associative(a: Usage, b: Usage, c: Usage) -> None:
    """(a + b) + c == a + (b + c). Meters batch charges into a running
    total; reordering charges (e.g. a fan-in from siblings arriving
    out of order) must not change the total.

    Integer counters are exactly associative. ``cost_usd`` rounds to
    6 decimals on every ``__add__``, so two adds can drift by up to
    5e-7 relative to the mathematically-equal single sum; use a
    single-digit-of-rounding-noise tolerance."""
    left = (a + b) + c
    right = a + (b + c)
    assert left.input_tokens == right.input_tokens
    assert left.output_tokens == right.output_tokens
    assert left.cache_read_tokens == right.cache_read_tokens
    assert left.cache_write_tokens == right.cache_write_tokens
    assert abs(left.cost_usd - right.cost_usd) < 5e-6


@given(a=_usage_strategy, b=_usage_strategy)
def test_usage_add_is_commutative(a: Usage, b: Usage) -> None:
    """a + b == b + a. A parent that observes two children finishing
    in either order lands on the same billed total."""
    l, r = a + b, b + a
    assert (l.input_tokens, l.output_tokens, l.cache_read_tokens, l.cache_write_tokens) == (
        r.input_tokens,
        r.output_tokens,
        r.cache_read_tokens,
        r.cache_write_tokens,
    )
    assert abs(l.cost_usd - r.cost_usd) < 1e-6


@given(a=_usage_strategy)
def test_usage_add_identity(a: Usage) -> None:
    """a + Usage() == a. A zero-usage stage (a human gate, a cached
    hit that already reimbursed its usage) must not perturb the
    running total."""
    result = a + Usage()
    assert result.input_tokens == a.input_tokens
    assert result.output_tokens == a.output_tokens
    assert result.cache_read_tokens == a.cache_read_tokens
    assert result.cache_write_tokens == a.cache_write_tokens
    assert abs(result.cost_usd - a.cost_usd) < 1e-6


@given(a=_usage_strategy)
def test_usage_add_returns_new_instance_not_mutating(a: Usage) -> None:
    """``__add__`` must return a fresh ``Usage`` — the operand instance
    stays byte-identical afterward. Otherwise a shared reference
    (e.g. one usage tracked across two meters) would corrupt its
    original observer."""
    input_snap, output_snap, cost_snap = a.input_tokens, a.output_tokens, a.cost_usd
    _ = a + Usage(input_tokens=100, output_tokens=200, cost_usd=0.5)
    assert a.input_tokens == input_snap
    assert a.output_tokens == output_snap
    assert a.cost_usd == cost_snap


def test_usage_total_tokens_matches_sum() -> None:
    u = Usage(input_tokens=100, output_tokens=250)
    assert u.total_tokens == 350


# ── Frozen kernel value types reject mutation ────────────────────────────────
#
# Every kernel value type is frozen. Cache adapters return the same
# instance on hits; observers hold references alongside audit sinks;
# middlewares thread the same result through the chain — a stray
# in-place assignment on any one of them would corrupt every
# subsequent reader. These tests are the guard rail: any attempt to
# mutate a field surfaces as ``FrozenInstanceError`` at write time,
# not as a mysterious downstream drift.


@pytest.mark.parametrize(
    "instance, field",
    [
        (Usage(), "input_tokens"),
        (Usage(), "cost_usd"),
        (Message("user", "hi"), "content"),
        (Message("user", "hi"), "role"),
        (ToolCall("id", "name"), "name"),
        (ToolSchema("s"), "name"),
        (LLMResult("hello"), "content"),
        (LLMResult("hello"), "usage"),
        (Delta(text="tok"), "text"),
        (Delta(text="tok"), "parsed"),
        (StreamEvent("final"), "type"),
        (StreamEvent("final"), "result"),
        (Chunk("id", "text"), "text"),
        (ChatRequest(messages=[Message("user", "hi")], model="m"), "model"),
        (ChatRequest(messages=[Message("user", "hi")], model="m"), "messages"),
        (ToolRequest(name="t", arguments={}, tool=None), "name"),
    ],
)
def test_kernel_value_types_are_frozen(instance, field) -> None:
    """Every kernel value type refuses in-place mutation of every field."""
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        setattr(instance, field, "clobber")


def test_frozen_llmresult_supports_dataclasses_replace() -> None:
    """``dataclasses.replace`` is the sanctioned rewrite path for
    frozen results — callers that need to update a field build a new
    instance instead of mutating the shared one."""
    from dataclasses import replace

    original = LLMResult("hello", model="a", usage=Usage(input_tokens=10))
    updated = replace(original, model="b")
    assert updated.model == "b"
    assert original.model == "a"  # snapshot unchanged
    assert updated.content == original.content == "hello"
    assert updated.usage is original.usage  # replace preserves untouched fields


def test_frozen_chatrequest_replace_swaps_model_without_touching_original() -> None:
    """The fallback middleware rewrites ``ChatRequest.model`` between
    attempts. The replace path must produce an independent request so
    the original stays intact for span attributes / audit records."""
    from dataclasses import replace

    req = ChatRequest(messages=[Message("user", "q")], model="gpt-4")
    swapped = replace(req, model="claude-sonnet")
    assert swapped.model == "claude-sonnet"
    assert req.model == "gpt-4"
    assert swapped.messages is req.messages  # unchanged fields share references


# ── ToolCall.arguments read-only mapping ─────────────────────────────────────
#
# A tool implementation that mutated ``tc.arguments`` (say
# ``args.pop("token")``) would corrupt every downstream reader — the
# ReAct approval snapshot, the idempotency hash, the audit trail.
# ``arguments`` is exposed as an immutable ``MappingProxyType`` view
# over a defensive copy of the caller's dict.


def test_toolcall_arguments_view_rejects_item_assignment() -> None:
    from types import MappingProxyType

    tc = ToolCall("c1", "search", {"q": "hello"})
    assert isinstance(tc.arguments, MappingProxyType)
    with pytest.raises(TypeError):
        tc.arguments["q"] = "changed"  # type: ignore[index]


def test_toolcall_arguments_view_isolated_from_source_dict() -> None:
    """Callers can mutate the original dict they passed in; the
    ToolCall's view stays pinned at construction-time contents."""
    source = {"q": "original"}
    tc = ToolCall("c1", "search", source)
    source["q"] = "changed"
    source["new"] = "extra"
    assert tc.arguments["q"] == "original"
    assert "new" not in tc.arguments


def test_toolcall_supports_deepcopy() -> None:
    """MappingProxyType is not natively deep-copyable in stdlib. The
    ``__deepcopy__`` hook unwraps to a dict, deepcopies, and rewraps
    so ``Checkpointer.snapshot`` (which deep-copies state that
    contains ToolCalls) works without a pickle detour."""
    import copy

    original = ToolCall("c1", "search", {"q": "hello", "nested": {"k": [1, 2, 3]}})
    clone = copy.deepcopy(original)

    assert clone == original
    assert clone is not original
    # The inner nested dict must be independently deep-copied too.
    assert clone.arguments["nested"] is not original.arguments["nested"]


def test_toolcall_arguments_hash_is_deterministic() -> None:
    """The idempotency-key hash routes ``ToolCall.arguments`` through
    ``stable_hash``. Two ToolCalls constructed with the same content
    must produce the same hash regardless of dict iteration order."""
    tc1 = ToolCall("c1", "search", {"a": 1, "b": 2, "c": 3})
    tc2 = ToolCall("c1", "search", {"c": 3, "b": 2, "a": 1})
    assert stable_hash({"tc": tc1}) == stable_hash({"tc": tc2})


# ── stable_hash: property + adversarial ──────────────────────────────────────


@given(
    st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-(2**53), max_value=2**53),
            st.text(max_size=50),
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=10),
            st.dictionaries(st.text(max_size=20), children, max_size=10),
        ),
        max_leaves=30,
    )
)
def test_stable_hash_deterministic_for_arbitrary_json(payload) -> None:
    """Any JSON-shaped payload hashes to the same value across
    repeated calls. This is the contract memoize / idempotency /
    audit fingerprints rely on."""
    assert stable_hash(payload) == stable_hash(payload)


@given(payload=st.dictionaries(st.text(max_size=10), st.integers(), min_size=2, max_size=10))
def test_stable_hash_ignores_dict_iteration_order(payload) -> None:
    """The encoder sorts keys — two dicts with the same entries but
    different insertion orders share a hash."""
    reversed_payload = dict(reversed(list(payload.items())))
    assert stable_hash(payload) == stable_hash(reversed_payload)


def test_stable_hash_uses_type_qualname_not_repr_for_bare_classes() -> None:
    """A class with no ``__dict__`` entries (e.g. an empty sentinel)
    falls back to a stable type-name marker. Two instances of the
    same empty class hash identically; instances of DIFFERENT empty
    classes with the same name in different modules don't collide
    because the qualname is namespace-aware."""

    class Sentinel:  # __dict__ is empty when no attrs are set
        pass

    a, b = Sentinel(), Sentinel()
    assert stable_hash(a) == stable_hash(b)


def test_stable_hash_handles_frozenset_and_set() -> None:
    """Set/frozenset are stored sorted for stability; the same members
    in different insertion orders share a hash."""
    a = frozenset(["x", "a", "m"])
    b = frozenset(["m", "a", "x"])
    assert stable_hash(a) == stable_hash(b)


def test_stable_hash_handles_nested_toolcall_with_frozen_arguments() -> None:
    """A ToolCall containing a MappingProxyType view routes through
    the encoder's Mapping branch. Regression: earlier the encoder
    fell through to the bare-type fallback and two identical
    ToolCalls hashed differently because of proxy identity."""
    tc = ToolCall("c1", "search", {"query": "hello", "k": 5})
    h1 = stable_hash({"tool_call": tc})
    h2 = stable_hash({"tool_call": tc})
    assert h1 == h2

    # And a fresh ToolCall with the same content produces the same hash.
    tc2 = ToolCall("c1", "search", {"query": "hello", "k": 5})
    assert stable_hash({"tool_call": tc2}) == h1


# ── classify: adversarial ordering ───────────────────────────────────────────


def test_classify_transient_wins_over_permanent_substring_collision() -> None:
    """A ``ValidationError('request timed out during validation')``
    contains both a PERMANENT marker (validation) and a TRANSIENT
    marker (timeout). The classifier scans TRANSIENT patterns first
    because the safer default is to retry — a PERMANENT-first order
    would fail-fast and drop legitimate retryable requests."""

    class ValidationError(Exception):
        pass

    exc = ValidationError("request timed out during validation")
    assert classify(exc) is ErrorClass.TRANSIENT


def test_classify_permanent_wins_when_no_transient_substring() -> None:
    """PERMANENT still fires when nothing transient matches — the
    ordering doesn't accidentally reclassify unambiguous 4xxs."""

    class ValidationError(Exception):
        pass

    exc = ValidationError("400 invalid request payload")
    assert classify(exc) is ErrorClass.PERMANENT


def test_classify_unknown_when_no_signal() -> None:
    """A vanilla ``RuntimeError`` with a neutral message classifies
    as UNKNOWN — the resilience loop treats UNKNOWN as retryable
    (conservative default), but the caller is still free to
    distinguish it from a categorised TRANSIENT."""
    assert classify(RuntimeError("something happened")) is ErrorClass.UNKNOWN


# ── CircuitBreaker: full state machine ───────────────────────────────────────


def test_circuit_breaker_closed_state_admits_freely() -> None:
    """A fresh breaker starts CLOSED and admits every caller until
    the fail count reaches the threshold."""
    br = CircuitBreaker("x", fail_threshold=3)
    for _ in range(10):
        assert br.allow()


def test_circuit_breaker_success_resets_fail_count() -> None:
    """A success mid-run resets the counter — so an occasional flake
    doesn't trip the breaker over the course of a long-running loop."""
    br = CircuitBreaker("x", fail_threshold=3)
    br.record_failure()
    br.record_failure()
    br.record_success()  # reset
    br.record_failure()
    br.record_failure()  # 2/3, still closed
    assert br.state == "closed"
    assert br.allow()


def test_circuit_breaker_half_open_failure_reopens() -> None:
    """After the cooldown a single probe is admitted; if that probe
    itself fails, the breaker opens again and the cooldown clock
    restarts. No stampede of failed probes."""
    t = {"now": 0.0}
    br = CircuitBreaker("x", fail_threshold=1, cooldown=5.0, clock=lambda: t["now"])
    br.record_failure()  # OPEN
    t["now"] = 5.0
    assert br.allow()  # probe admitted → half_open
    assert br.state == "half_open"
    br.record_failure()  # probe failed → back to OPEN
    assert br.state == "open"
    assert not br.allow()


# ── idempotency_key ──────────────────────────────────────────────────────────


def test_idempotency_key_deterministic_across_calls() -> None:
    """The same operation payload always produces the same idempotency
    key — memoize / rejection-ledger / audit fingerprints all rely on
    this invariant."""
    payload = {"tool": "search", "args": {"q": "hello"}}
    assert idempotency_key(payload) == idempotency_key(payload)


def test_idempotency_key_stable_across_dict_ordering() -> None:
    """Two payloads with the same entries in different insertion
    orders share an idempotency key — otherwise a retry that happened
    to iterate the dict differently would look like a NEW operation
    and re-execute a side-effecting call."""
    a = {"tool": "search", "args": {"q": "hello"}, "k": 5}
    b = {"k": 5, "args": {"q": "hello"}, "tool": "search"}
    assert idempotency_key(a) == idempotency_key(b)


def test_idempotency_key_diverges_on_content_change() -> None:
    """A single-byte change in the payload produces a distinct key.
    Otherwise a mutation would be silently de-duplicated against the
    previous call's result."""
    a = idempotency_key({"q": "hello"})
    b = idempotency_key({"q": "world"})
    assert a != b
