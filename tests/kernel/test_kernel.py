"""L0 kernel: middleware composition order, resilience classification/retry, bounded concurrency."""

import asyncio
import dataclasses
import datetime as _dt
import threading
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

from agentkit.kernel._frozen import FrozenDict
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
# ``arguments`` is an immutable ``FrozenDict`` over a deep copy of the
# caller's dict. It was a ``MappingProxyType`` until the payload freeze;
# the guarantee is identical and the serialisation behaviour is not —
# see ``test_frozen_request_payloads.py`` for that half.


def test_toolcall_arguments_view_rejects_item_assignment() -> None:
    tc = ToolCall("c1", "search", {"q": "hello"})
    assert isinstance(tc.arguments, FrozenDict)
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
    """``Checkpointer.snapshot`` deep-copies state that contains ToolCalls,
    so a ToolCall that cannot be deep-copied takes the durable path down
    with it. A ``MappingProxyType`` payload could not be, and ``ToolCall``
    carried a ``__deepcopy__`` hook to work around it; the hook is GONE
    (measured: ``'__deepcopy__' in ToolCall.__dict__`` is False) because
    ``FrozenDict`` defines its own. The clone comes back frozen at every
    level — measured ``FrozenDict / FrozenDict / FrozenList`` for the
    nested payload below."""
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


# ── ToolCall / Message hashability ───────────────────────────────────────────
#
# ``arguments`` is an immutable mapping — a ``MappingProxyType`` when this
# was written, a ``FrozenDict`` now — and NEITHER is hashable, so the
# immutability guarantee silently cost the frozen
# dataclass its generated ``__hash__``, and took ``Message``,
# ``PrefixContext`` and ``FrozenContext`` down with it. Measured before
# the fix::
#
#     hash(ToolCall("c1", "search", {"q": "hi"}))    TypeError: unhashable type: 'dict'
#     hash(Message("assistant", tool_calls=(tc,)))   TypeError: unhashable type: 'dict'
#
# ``ToolCall.__hash__`` hashes ``(id, name)`` only. These tests pin both
# halves of that trade: that hashing WORKS whatever the model put in
# ``arguments``, and that ``__eq__`` still sees the arguments, because
# dedup correctness (``WorkingContext.merge(mode="union")``) rides on
# equality, not on the hash.


def test_toolcall_is_hashable_with_nested_json_arguments() -> None:
    """Arguments come from decoded provider JSON, so nested lists/dicts are the
    NORMAL case, not an edge case. Hashability must not depend on what the
    model happened to emit."""
    tc = ToolCall("c1", "search", {"q": "hi", "filters": [{"k": [1, 2]}, None], "n": {"deep": {}}})
    assert isinstance(hash(tc), int)
    assert {tc} == {tc}


def test_toolcall_with_empty_arguments_is_hashable() -> None:
    """The degenerate payload — a no-arg tool — shares the code path."""
    assert isinstance(hash(ToolCall("c1", "ping")), int)


def test_message_carrying_tool_calls_is_hashable() -> None:
    """The failure that actually shipped was one level up: a Message hashes its
    ``tool_calls`` tuple, so an assistant turn that requested a tool was
    unhashable while a plain user turn was not."""
    tc = ToolCall("c1", "search", {"q": {"nested": ["json"]}})
    msg = Message("assistant", "", tool_calls=(tc,))
    assert isinstance(hash(msg), int)
    assert len({msg, Message("assistant", "", tool_calls=(tc,))}) == 1


def test_toolcall_hash_ignores_arguments_while_eq_does_not() -> None:
    """The whole design in one assertion. Hashing a SUBSET is sound because the
    invariant only requires equal objects to hash equally — so two calls that
    differ only in arguments may share a bucket, but must NOT compare equal, or
    union-mode dedup would silently drop a distinct tool call."""
    a = ToolCall("c1", "search", {"q": "alpha"})
    b = ToolCall("c1", "search", {"q": "beta"})
    assert hash(a) == hash(b)  # same bucket — deliberate
    assert a != b  # ...but not the same call
    assert len({a, b}) == 2  # so the set keeps both


def test_toolcall_hash_does_not_read_the_payload_at_all() -> None:
    """O(1), structurally proven rather than timed: a 100_000-key arguments
    dict hashes IDENTICALLY to a 1-key one, which is only possible if the
    payload is never walked. Measured 0.130 µs vs 0.128 µs; a
    ``stable_hash``-based hash measured 4.99 µs vs 87.8 ms on the same pair.
    The hash runs once per message per union merge, so payload-linear here
    would be a fan-in-sized regression."""
    small = ToolCall("c1", "search", {"q": "hi"})
    huge = ToolCall("c1", "search", {f"k{i}": {"v": [i, str(i)]} for i in range(100_000)})
    assert hash(small) == hash(huge)
    assert small != huge


def test_toolcall_hash_separates_distinct_ids_and_names() -> None:
    """The other side of the bucket trade: identity IS in the hash, so the
    common case — distinct provider-issued call ids — spreads across buckets
    instead of degrading a set to a linear scan."""
    base = ToolCall("c1", "search", {"q": "x"})
    assert hash(base) != hash(ToolCall("c2", "search", {"q": "x"}))
    assert hash(base) != hash(ToolCall("c1", "lookup", {"q": "x"}))


def test_toolcall_with_reused_id_but_different_name_is_unequal() -> None:
    """A provider echoing an id it already used (retry, replay, a buggy
    adapter) must not collapse two different tools into one entry."""
    a = ToolCall("c1", "search", {"q": "x"})
    b = ToolCall("c1", "delete_everything", {"q": "x"})
    assert a != b
    assert len({a, b}) == 2


def test_toolcall_survives_pickle_round_trip() -> None:
    """``__deepcopy__`` covered ``copy.deepcopy``; ``pickle`` had no hook and
    hit the same stdlib limitation head-on::

        pickle.dumps(tc)  TypeError: cannot pickle 'mappingproxy' object

    Tool calls reach durable stores (checkpointer, replay recorder) and a
    ``FrozenContext`` is advertised as shareable across an agent boundary —
    which means pickle once that boundary is a process. The round trip comes
    back FROZEN, not as a plain dict — and frozen ALL THE WAY DOWN, which the
    mappingproxy version never was (``clone.arguments["nested"]["k"].append(4)``
    used to work).

    ``ToolCall`` carries no ``__reduce__`` of its own any more, and neither
    does it need one: ``arguments`` is a ``FrozenDict``, which pickles through
    its own hook and comes back a ``FrozenDict``. That is what makes this test
    worth keeping rather than a formality — pickle uses the default protocol
    here, which restores state directly and never runs ``__post_init__``, so
    the assertions below are the only thing standing between a payload-side
    regression (``FrozenDict`` losing ``__reduce__``) and a tool call arriving
    from a durable store as an editable dict."""
    import pickle

    original = ToolCall("c1", "search", {"q": "hello", "nested": {"k": [1, 2, 3]}})
    clone = pickle.loads(pickle.dumps(original))

    assert clone == original
    assert hash(clone) == hash(original)
    assert isinstance(clone.arguments, FrozenDict)
    assert clone.arguments["nested"] == {"k": [1, 2, 3]}
    with pytest.raises(TypeError):
        clone.arguments["q"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        clone.arguments["nested"]["k"].append(4)


def test_toolcall_construction_and_access_are_unchanged() -> None:
    """POSITIVE CONTROL: adding ``__hash__`` must not disturb the constructor,
    the field defaults, or read access — passes identically before and after
    the fix. It also passed unchanged when the hand-written ``__reduce__`` that
    once sat beside ``__hash__`` was deleted, which is the point of a control:
    the surface a caller touches did not move."""
    tc = ToolCall("c1", "search", {"q": "hello"})
    assert (tc.id, tc.name) == ("c1", "search")
    assert tc.arguments["q"] == "hello"
    assert dict(tc.arguments) == {"q": "hello"}
    assert ToolCall("c2", "ping").arguments == {}
    assert tc == ToolCall("c1", "search", {"q": "hello"})


def test_toolcall_equality_still_compares_arguments() -> None:
    """POSITIVE CONTROL for the ``__eq__`` half specifically: ``hash=False``-style
    hashing must never leak into equality. Passes before and after."""
    assert ToolCall("c1", "s", {"a": 1}) != ToolCall("c1", "s", {"a": 2})
    assert ToolCall("c1", "s", {"a": 1}) == ToolCall("c1", "s", {"a": 1})
    # Insertion order is not identity — the underlying dicts compare equal.
    assert ToolCall("c1", "s", {"a": 1, "b": 2}) == ToolCall("c1", "s", {"b": 2, "a": 1})


# ── ChatRequest / ToolRequest / ToolSchema hashability ───────────────────────
#
# The same bug shape as ``ToolCall`` above, three more times: a
# ``frozen=True`` dataclass carrying a mutable payload, so the
# generated ``__hash__`` never ran. Measured before the fix::
#
#     hash(ChatRequest([Message("user", "hi")], "gpt-4"))     TypeError: unhashable type: 'list'
#     hash(ToolRequest("search", {"q": "hi"}, tool=None))     TypeError: unhashable type: 'dict'
#     hash(ToolSchema("search", "", {"type": "object"}))      TypeError: unhashable type: 'dict'
#
# ``ChatRequest`` is the worst of the three: ``messages`` is a ``list``,
# so EVERY request was unhashable, not only the ones carrying an
# awkward payload. Nothing in the framework hashed one — ``memoize``
# keys on ``stable_hash`` of the content, deliberately — so the break
# was caller-facing (a ``dict[ChatRequest, LLMResult]`` double, an
# ``lru_cache``, ``set(request.tools)`` to dedupe two registries) and
# surfaced only at the call site that hashed one.
#
# Each hash is a SUBSET: the call shape for ``ChatRequest``, the routing
# fields for ``ToolRequest``, the advertisement for ``ToolSchema``. The
# tests below pin both halves of that trade — that hashing works
# whatever the payload holds, and that ``__eq__`` still sees everything,
# because dedup correctness rides on equality, not on the hash.


def test_chatrequest_is_hashable_with_transcript_and_payload_dicts() -> None:
    """The representative request: a transcript, tool schemas, a structured
    output format and a provider cache hint — every one of them a container
    the generated hash choked on."""
    req = ChatRequest(
        messages=[Message("user", "hi"), Message("assistant", "", tool_calls=(ToolCall("c1", "s", {"q": [1]}),))],
        model="gpt-4",
        tools=[ToolSchema("search", "find things", {"type": "object", "properties": {"q": {"type": "string"}}})],
        response_format={"type": "json_schema", "schema": {"properties": {"a": {"type": "array"}}}},
        cache_hint={"prefix_tokens": 2048},
    )
    assert isinstance(hash(req), int)
    assert {req} == {req}


def test_chatrequest_with_empty_transcript_and_none_payloads_is_hashable() -> None:
    """The degenerate end: no messages, no tools, no response format. It shares
    the code path, and it is the shape a compaction middleware can transiently
    produce."""
    assert isinstance(hash(ChatRequest(messages=[], model="m")), int)


def test_chatrequest_hash_ignores_message_content_while_eq_does_not() -> None:
    """The whole design in one assertion. Hashing a subset is sound because the
    invariant only requires EQUAL objects to hash equally — so two requests of
    the same shape but different content share a bucket, and must NOT compare
    equal, or a request-keyed cache would serve one conversation's answer to
    another."""
    a = ChatRequest([Message("user", "alpha")], "m")
    b = ChatRequest([Message("user", "beta")], "m")
    assert hash(a) == hash(b)  # same bucket — deliberate
    assert a != b  # ...but not the same request
    assert len({a, b}) == 2  # so the set keeps both


def test_chatrequest_hash_does_not_read_the_transcript_or_the_payloads() -> None:
    """O(1) in payload size, proven structurally rather than by timing: a
    request whose single message carries a megabyte of text, a 10_000-key
    response format and a 10_000-key cache hint hashes IDENTICALLY to a
    one-character one. That equality is only possible if none of the three is
    ever walked. Measured 0.28 µs at every transcript size from 1 to 1000
    turns, against 273 µs for a messages-inclusive hash at 1000."""
    small = ChatRequest([Message("user", "x")], "m")
    huge = ChatRequest(
        [Message("user", "x" * 1_000_000)],
        "m",
        response_format={f"k{i}": [i] for i in range(10_000)},
        cache_hint={f"c{i}": {"v": i} for i in range(10_000)},
    )
    assert hash(small) == hash(huge)
    assert small != huge


def test_chatrequest_hash_separates_the_shapes_that_actually_vary() -> None:
    """The other side of the bucket trade. ``model`` and the sampling settings
    are the fields a fallback or a router rewrites, and the transcript LENGTH
    is what changes between successive turns of one loop — an appended message
    moves the request to a different bucket for O(1), which is why ``len`` is
    in the hash while the messages are not."""
    base = ChatRequest([Message("user", "q")], "gpt-4")
    assert hash(base) != hash(ChatRequest([Message("user", "q")], "claude-sonnet"))
    assert hash(base) != hash(ChatRequest([Message("user", "q"), Message("assistant", "a")], "gpt-4"))
    assert hash(base) != hash(ChatRequest([Message("user", "q")], "gpt-4", temperature=0.7))
    assert hash(base) != hash(ChatRequest([Message("user", "q")], "gpt-4", max_tokens=512))


def test_chatrequest_hashability_survives_an_unhashable_cache_hint() -> None:
    """``cache_hint`` is annotated ``Any`` and holds whatever the provider
    adapter wants. If the hash read it, hashability would depend on what a
    caller happened to put there — the bug, reintroduced one field over."""
    req = ChatRequest([Message("user", "hi")], "m", cache_hint={"blocks": [{"type": "text"}]})
    assert isinstance(hash(req), int)


def test_chatrequest_construction_and_field_access_are_unchanged() -> None:
    """POSITIVE CONTROL: adding ``__hash__`` must not disturb the constructor,
    the defaults or read access — passes identically before and after."""
    tools = [ToolSchema("search")]
    req = ChatRequest([Message("user", "hi")], "gpt-4", tools=tools, temperature=0.5, max_tokens=64)
    assert req.model == "gpt-4"
    assert req.messages[0].content == "hi"
    # The payload freeze COPIES, so ``is tools`` no longer holds — that is the
    # un-aliasing, not a regression. Everything a reader actually does with the
    # field is unchanged: it is still a ``list``, still ``==`` the input, still
    # indexable, and its elements are the same objects (a ToolSchema is a leaf
    # to ``deep_freeze``, so it is passed through, not rebuilt).
    assert req.tools is not tools
    assert isinstance(req.tools, list) and req.tools == tools
    assert req.tools[0] is tools[0]
    assert (req.temperature, req.max_tokens) == (0.5, 64)
    assert ChatRequest([], "m").tools is None and ChatRequest([], "m").response_format is None


def test_chatrequest_equality_still_compares_every_field() -> None:
    """POSITIVE CONTROL for the ``__eq__`` half: hashing a subset must never
    leak into equality. Passes before and after."""
    msgs = [Message("user", "hi")]
    assert ChatRequest(msgs, "m") == ChatRequest([Message("user", "hi")], "m")
    assert ChatRequest(msgs, "m") != ChatRequest(msgs, "other")
    assert ChatRequest(msgs, "m", tools=[ToolSchema("a")]) != ChatRequest(msgs, "m", tools=[ToolSchema("b")])
    assert ChatRequest(msgs, "m", response_format={"a": 1}) != ChatRequest(msgs, "m", response_format={"a": 2})
    assert ChatRequest(msgs, "m", cache_hint={"x": 1}) != ChatRequest(msgs, "m", cache_hint={"x": 2})


def test_toolrequest_is_hashable_with_nested_json_arguments() -> None:
    """Arguments are decoded provider JSON, so nested lists/dicts are the
    NORMAL case. Hashability must not depend on what the model emitted."""
    req = ToolRequest("search", {"q": "hi", "filters": [{"k": [1, 2]}, None]}, tool=None)
    assert isinstance(hash(req), int)
    assert {req} == {req}


def test_toolrequest_hash_ignores_arguments_while_eq_does_not() -> None:
    """Two calls to the same tool with different arguments share a bucket —
    deliberate — but must stay distinct, or a request-keyed set would drop a
    genuine second call."""
    a = ToolRequest("search", {"q": "alpha"}, tool=None)
    b = ToolRequest("search", {"q": "beta"}, tool=None)
    assert hash(a) == hash(b)
    assert a != b
    assert len({a, b}) == 2


def test_toolrequest_hash_does_not_read_the_arguments_at_all() -> None:
    """O(1), proven structurally rather than timed: a 100_000-key arguments
    dict hashes IDENTICALLY to a 1-key one. Measured 0.229 µs vs 0.237 µs;
    ``stable_hash`` on the same pair measured 3.28 µs vs 80.0 ms."""
    small = ToolRequest("search", {"q": "hi"}, tool=None)
    huge = ToolRequest("search", {f"k{i}": {"v": [i, str(i)]} for i in range(100_000)}, tool=None)
    assert hash(small) == hash(huge)
    assert small != huge


def test_toolrequest_hashability_does_not_depend_on_the_resolved_tool() -> None:
    """``tool`` is the resolved port, annotated ``Any``. A plain ``@dataclass``
    tool gets ``__eq__`` and therefore ``__hash__ = None`` — so reading it here
    would resurrect the same failure one indirection further from the call
    site, and only for some registries."""

    @dataclass
    class _UnhashableTool:  # eq=True by default → unhashable
        label: str

    tool = _UnhashableTool("search")
    with pytest.raises(TypeError):
        hash(tool)
    assert isinstance(hash(ToolRequest("search", {"q": "hi"}, tool=tool)), int)


def test_toolrequest_hash_separates_the_routing_fields() -> None:
    """The fields the chain routes on are the fields that spread the buckets:
    the tool name, the idempotency gate's ``side_effecting``, the egress
    guard's ``url_arg``."""
    base = ToolRequest("search", {"q": "x"}, tool=None)
    assert hash(base) != hash(ToolRequest("delete_everything", {"q": "x"}, tool=None))
    assert hash(base) != hash(ToolRequest("search", {"q": "x"}, tool=None, side_effecting=True))
    assert hash(base) != hash(ToolRequest("search", {"q": "x"}, tool=None, url_arg="url"))


def test_toolrequest_construction_access_and_equality_are_unchanged() -> None:
    """POSITIVE CONTROL: constructor, defaults, read access and equality all
    behave exactly as before — including that ``arguments`` is still a ``dict``
    by ``isinstance``, because the terminal hands it to ``tool.run(args, ctx)``
    and ``FunctionTool`` gates on ``isinstance(args, Mapping)``. It is a
    ``FrozenDict`` rather than a plain one since the payload freeze; a subclass
    is precisely what keeps that gate — and ``json``, and ``asdict`` —
    working."""
    sentinel = object()
    req = ToolRequest("search", {"q": "hi"}, tool=sentinel)
    assert (req.name, req.tool) == ("search", sentinel)
    assert req.arguments == {"q": "hi"} and isinstance(req.arguments, dict)
    assert (req.side_effecting, req.url_arg) == (False, None)
    assert req == ToolRequest("search", {"q": "hi"}, tool=sentinel)
    assert ToolRequest("s", {"a": 1}, tool=None) != ToolRequest("s", {"a": 2}, tool=None)
    assert ToolRequest("s", {"a": 1}, tool=None) != ToolRequest("s", {"a": 1}, tool=None, side_effecting=True)


def test_toolschema_is_hashable_with_a_real_json_schema_body() -> None:
    """A JSON Schema body nests dict and list by construction — ``properties``,
    ``required``, ``enum`` — so the empty schema was the ONLY hashable one
    before the fix, which is exactly the vacuous-pass trap."""
    schema = ToolSchema(
        "search",
        "find things",
        {
            "type": "object",
            "properties": {"q": {"type": "string"}, "mode": {"enum": ["fast", "deep"]}},
            "required": ["q"],
        },
    )
    assert isinstance(hash(schema), int)
    assert {schema} == {schema}


def test_toolschema_with_empty_parameters_is_hashable() -> None:
    """The degenerate payload — a no-argument tool — shares the code path."""
    assert isinstance(hash(ToolSchema("ping")), int)


def test_toolschema_hash_ignores_parameters_while_eq_does_not() -> None:
    """Two revisions of one tool's schema collide into a bucket — deliberate —
    and ``__eq__`` separates them there, so ``set(request.tools)`` deduping
    across registries stays exact rather than dropping a revision."""
    a = ToolSchema("search", "find things", {"properties": {"q": {"type": "string"}}})
    b = ToolSchema("search", "find things", {"properties": {"q": {"type": "integer"}}})
    assert hash(a) == hash(b)
    assert a != b
    assert len({a, b}) == 2


def test_toolschema_hash_does_not_read_the_schema_body() -> None:
    """O(1) in the schema size, proven structurally: a 20_000-key body hashes
    IDENTICALLY to a one-key one. Measured 0.192 µs for both; ``stable_hash``
    of a mere 20-property body measured 23.4 µs, and that cost would be paid
    per bucket probe rather than once per cache key."""
    small = ToolSchema("search", "d", {"type": "object"})
    huge = ToolSchema("search", "d", {f"p{i}": {"type": "string", "enum": [i]} for i in range(20_000)})
    assert hash(small) == hash(huge)
    assert small != huge


def test_toolschema_hash_separates_names_and_descriptions() -> None:
    """Identity IS in the hash: distinct tools spread across buckets instead of
    degrading a registry-sized set to a linear scan, and a rewritten
    description — the field that changes while the name does not — moves too,
    for the 0.063 µs it costs to include."""
    base = ToolSchema("search", "find things", {"type": "object"})
    assert hash(base) != hash(ToolSchema("lookup", "find things", {"type": "object"}))
    assert hash(base) != hash(ToolSchema("search", "find things FAST", {"type": "object"}))


def test_toolschema_set_dedupes_identical_advertisements() -> None:
    """The caller-facing use the missing hash blocked: two registries
    advertising the same tool collapse to one entry, by VALUE."""
    body = {"type": "object", "properties": {"q": {"type": "string"}}}
    pair = {ToolSchema("search", "find things", dict(body)), ToolSchema("search", "find things", dict(body))}
    assert len(pair) == 1


def test_toolschema_parameters_stay_a_plain_serialisable_dict() -> None:
    """POSITIVE CONTROL: ``parameters`` is frozen but still SERIALISABLE.

    Freezing it into a ``MappingProxyType`` was the option that would have
    broken every existing reader — a mappingproxy is not JSON-serialisable and
    ``dataclasses.asdict`` does not unwrap it — so the freeze is a ``dict``
    SUBCLASS instead. These three lines are what provider adapters do with a
    schema on every call (``openai_compat`` json-dumps ``schema.parameters``,
    ``anthropic`` sends it as ``input_schema``), and they pass identically
    before and after."""
    import json
    from dataclasses import asdict

    schema = ToolSchema("search", "find things", {"type": "object", "properties": {"q": {"type": "string"}}})
    assert isinstance(schema.parameters, dict)
    assert json.loads(json.dumps(schema.parameters)) == schema.parameters
    assert asdict(schema)["parameters"]["properties"] == {"q": {"type": "string"}}


def test_request_types_still_deepcopy_and_pickle_after_the_hash() -> None:
    """POSITIVE CONTROL for the other two thirds of the value contract. All
    three already round-tripped — only ``__hash__`` was missing — so this pins
    that the fix took nothing away. ``Checkpointer.snapshot`` deep-copies state
    at the durable seam, and the replay recorder pickles.

    ``ToolRequest.tool`` is left as ``None`` on purpose: a resolved port is a
    live object with no copy contract of its own, and it is not what this test
    is about."""
    import copy
    import pickle

    values = [
        ChatRequest(
            [Message("user", "hi")],
            "m",
            tools=[ToolSchema("search", "d", {"type": "object"})],
            response_format={"type": "json_object"},
        ),
        ToolRequest("search", {"q": "hi", "nested": {"k": [1, 2]}}, tool=None),
        ToolSchema("search", "find things", {"properties": {"q": {"type": "string"}}}),
    ]
    for original in values:
        for clone in (copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
            assert clone == original
            assert hash(clone) == hash(original)  # equal objects hash equally — across a boundary too


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
    """A ToolCall's frozen arguments hash by CONTENT, not by identity.

    The payload is a ``FrozenDict`` now, so stdlib ``json`` encodes it
    directly and the encoder's Mapping branch is not what carries this
    any more (measured: a spy in the ``default`` slot sees only
    ``ToolCall``). The regression being locked is unchanged — the encoder
    once fell through to the bare-type fallback and two identical
    ToolCalls hashed differently because of container identity."""
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


# ── CircuitBreaker: HALF_OPEN must always have an exit ───────────────────────
#
# ``allow()`` refused every caller while ``half_open`` and the only exits were
# ``record_success`` / ``record_failure``. But ``run_with_resilience``
# deliberately skips ``record_failure`` on PERMANENT errors, and a
# ``BaseException`` escapes its ``except Exception`` entirely — so the probe
# could resolve without ever reporting. Measured: one 401 on the post-cooldown
# probe left ``state == "half_open"`` and every later call against a fully
# healthy dependency raised ``CircuitOpen`` for the rest of the process
# lifetime (10_000 simulated seconds later, still refused).


def test_permanent_probe_failure_does_not_wedge_the_breaker():
    """The headline regression. A 401 on the post-cooldown probe must release
    the gate — and must NOT count toward the failure threshold."""
    t = {"now": 0.0}
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=10.0, clock=lambda: t["now"])

    async def transient():
        raise TimeoutError("timed out")

    with pytest.raises((TimeoutError, CircuitOpen)):
        _run(run_with_resilience(transient, breaker=br, max_attempts=1, sleep=_nosleep))
    assert br.state == "open"

    t["now"] += 11.0  # cooled down → the next call is the probe

    async def permanent():
        raise ValueError("401 unauthorized")

    fails_before = br._fails
    with pytest.raises(ValueError):  # the real error still reaches the caller
        _run(run_with_resilience(permanent, breaker=br, max_attempts=3, sleep=_nosleep))
    assert br.state != "half_open", "the breaker wedged in half_open"
    assert br._fails == fails_before, "a PERMANENT error was counted as a health failure"

    # The dependency is healthy — a call after the next cooldown must get
    # through. NOT the immediately-next call: releasing the probe restarts the
    # cooldown, because not doing so made the breaker stop braking entirely
    # (twenty calls reached a dead dependency in 2ms). The invariant here is
    # "does not wedge FOREVER", which is what the wedge bug violated.
    t["now"] += 11.0

    async def ok():
        return "healthy"

    assert _run(run_with_resilience(ok, breaker=br, max_attempts=1, sleep=_nosleep)) == "healthy"
    assert br.state == "closed"


def test_baseexception_probe_does_not_wedge_the_breaker():
    """Edge case — the probe raises ``CancelledError``, which
    ``except Exception`` never sees. It is not a health signal either, so it
    must release the gate rather than hold it forever."""
    t = {"now": 0.0}
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=10.0, clock=lambda: t["now"])
    br.record_failure()  # → OPEN
    t["now"] += 11.0

    async def cancelled():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _run(run_with_resilience(cancelled, breaker=br, max_attempts=1, sleep=_nosleep))
    assert br.state != "half_open"
    # A fresh probe after the next cooldown — releasing restarts it, so the
    # gate is not open again instantly. "Not wedged" is the invariant.
    t["now"] += 11.0
    assert br.allow() is True


def test_half_open_gate_self_releases_after_one_cooldown():
    """Structural backstop for callers that never report at all (a crash between
    ``allow()`` and the verdict, or a third party driving ``allow()`` directly).
    After one cooldown with no verdict the probe is presumed lost."""
    t = {"now": 0.0}
    br = CircuitBreaker("x", fail_threshold=1, cooldown=5.0, clock=lambda: t["now"])
    br.record_failure()  # → OPEN
    t["now"] = 5.0
    assert br.allow() is True and br.state == "half_open"  # the probe
    assert br.allow() is False  # in flight → nobody else
    t["now"] = 9.9
    assert br.allow() is False, "the probe deadline expired early"
    t["now"] = 10.0
    assert br.allow() is True, "the breaker wedged in half_open forever"
    assert br.allow() is False, "the replacement probe is still a SINGLE probe"


def test_successful_probe_still_closes_the_breaker():
    """POSITIVE CONTROL 1 for the release path: the documented recovery edge
    must survive. A "fix" that simply stopped honouring ``half_open`` — or that
    left the breaker open on success — fails here."""
    t = {"now": 0.0}
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=10.0, clock=lambda: t["now"])
    br.record_failure()
    t["now"] += 11.0

    async def ok():
        return "recovered"

    assert _run(run_with_resilience(ok, breaker=br, max_attempts=1, sleep=_nosleep)) == "recovered"
    assert br.state == "closed" and br._fails == 0


def test_transient_probe_failure_still_reopens_with_a_fresh_cooldown():
    """POSITIVE CONTROL 2: a TRANSIENT probe failure must still re-open the
    breaker AND restart the cooldown — the release path must not have turned
    every probe outcome into a free pass."""
    t = {"now": 0.0}
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=10.0, clock=lambda: t["now"])
    br.record_failure()
    t["now"] += 11.0
    assert br.allow() is True and br.state == "half_open"

    br.record_failure()  # the probe failed transiently
    assert br.state == "open"
    assert br.allow() is False, "a failed probe left the gate open"
    t["now"] += 5.0
    assert br.allow() is False, "the cooldown clock was not restarted by the failed probe"


def test_concurrent_probes_are_still_serialised_to_one():
    """Edge case — N callers race the post-cooldown gate. Exactly ONE becomes
    the probe; the rest see ``CircuitOpen``. The release path must not have
    widened the gate into a stampede."""
    t = {"now": 0.0}
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=10.0, clock=lambda: t["now"])
    br.record_failure()  # → OPEN
    t["now"] += 11.0

    started = []

    async def slow_ok(tag):
        started.append(tag)
        await asyncio.sleep(0.01)  # the probe is still in flight
        return tag

    async def go():
        async def one(tag):
            try:
                return await run_with_resilience(
                    lambda: slow_ok(tag), breaker=br, max_attempts=1, sleep=_nosleep
                )
            except CircuitOpen:
                return "refused"

        return await asyncio.gather(*(one(i) for i in range(5)))

    results = _run(go())
    assert len(started) == 1, f"{len(started)} callers stampeded the recovering dependency"
    assert results.count("refused") == 4


# ── backoff_delay must not overflow on an absurd attempt number ──────────────


def test_release_probe_grants_no_health_credit():
    """``release_probe`` is the NEUTRAL edge: it consumes nothing and grants
    nothing. Zeroing ``_fails`` would make it a success in disguise — the
    dependency would be treated as recovered on the strength of a 401, and the
    next real failure would need the whole threshold again.

    Not covered by the wedge tests, which assert on ``state``. Added after
    probing the fix: the property holds by construction, and nothing pinned it.
    """
    t = {"now": 0.0}
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=10.0, clock=lambda: t["now"])
    br.record_failure()
    t["now"] = 20.0
    assert br.allow() and br.state == "half_open"

    before = br._fails
    br.release_probe()

    assert br.state == "open"
    assert br._fails == before, "a neutral release must not look like a success"


def test_a_permanently_failing_dependency_is_probed_once_per_cooldown() -> None:
    """This test previously asserted the OPPOSITE, and it was wrong.

    ``release_probe`` deliberately did not restart the cooldown, on the
    reasoning that a permanently-failing dependency should keep surfacing its
    real 401 rather than hide behind a ``CircuitOpen``. The effect was a breaker
    that does not brake: with a 60s cooldown, twenty consecutive calls spanning
    2ms ALL reached the dead dependency, because an already-elapsed cooldown
    makes EVERY caller a fresh probe rather than just the next one.

    The premise was wrong anyway. ``run_with_resilience`` raises
    ``CircuitOpen(...) from last``, so the real error is already reachable as
    ``__cause__`` on the refused calls — nothing was being masked, the rate
    limiting was simply absent.
    """
    t = {"now": 0.0}
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=60.0, clock=lambda: t["now"])
    br.record_failure()
    t["now"] = 61.0

    reached = 0
    for _ in range(20):
        if br.allow():
            reached += 1
            br.release_probe()  # every call is a PERMANENT failure
        t["now"] += 0.0001  # 0.1ms apart — well inside one cooldown

    assert reached == 1, f"the breaker let {reached} calls through in 2ms"

    # ...and it is not wedged: after a cooldown, one more probe is admitted.
    t["now"] += 61.0
    assert br.allow()


def test_a_refused_call_still_carries_the_failure_that_opened_the_breaker() -> None:
    """The reason restarting the cooldown costs nothing: a refused call carries
    the original exception as ``__cause__``, so a caller diagnosing an outage
    is not left staring at a bare ``CircuitOpen``.

    My first version of this test used a 401 to open the breaker and never got
    a ``CircuitOpen`` at all — because PERMANENT errors deliberately do not
    count toward the threshold, so a 401 never opens it. Only transient
    failures do, which is exactly the shape a breaker is for.
    """
    t = {"now": 0.0}
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=60.0, clock=lambda: t["now"])

    async def transient():
        raise TimeoutError("upstream timed out")

    with pytest.raises(TimeoutError):
        _run(run_with_resilience(transient, breaker=br, max_attempts=1, sleep=_nosleep))
    assert br.state == "open"

    async def anything():
        return "unreachable"

    with pytest.raises(CircuitOpen) as exc:
        _run(run_with_resilience(anything, breaker=br, max_attempts=1, sleep=_nosleep))

    assert isinstance(exc.value.__cause__, TimeoutError)
    assert "timed out" in str(exc.value.__cause__)


def test_backoff_delay_survives_huge_attempt_numbers():
    """``2 ** (attempt - 1)`` is an arbitrary-precision int; multiplying it by
    the float ``base`` raised ``OverflowError: int too large to convert to
    float`` from ``attempt >= 1025``. A retry helper that crashes on its own
    arithmetic turns a recoverable error into an unrecoverable one."""
    from agentkit.kernel.resilience import backoff_delay

    for attempt in (1025, 5000, 10**6):
        d = backoff_delay(attempt, base=0.5, cap=30.0)
        assert 0.0 <= d <= 30.0


def test_backoff_delay_still_grows_then_saturates_at_cap():
    """POSITIVE CONTROL: the clamp must not have flattened the curve. Low
    attempts stay below the cap (so backoff still grows), and the clamped tail
    is identical to the pre-clamp saturated value."""
    from agentkit.kernel.resilience import backoff_delay

    class _Max:  # deterministic rng: always return the ceiling
        @staticmethod
        def uniform(_lo, hi):
            return hi

    assert backoff_delay(1, base=0.5, cap=30.0, rng=_Max) == 0.5  # 0.5 * 2**0
    assert backoff_delay(4, base=0.5, cap=30.0, rng=_Max) == 4.0  # 0.5 * 2**3
    assert backoff_delay(10, base=0.5, cap=30.0, rng=_Max) == 30.0  # saturated
    # The clamp is exact: everything past saturation is the cap, not a
    # truncated exponent that drifted below it.
    assert backoff_delay(1024, base=0.5, cap=30.0, rng=_Max) == 30.0
    assert backoff_delay(10**9, base=0.5, cap=30.0, rng=_Max) == 30.0


def test_every_state_transition_takes_the_lock() -> None:
    """The class used to say "for cross-thread sharing, serialize the breaker
    externally", pushing that onto every caller who shares one breaker across a
    thread pool — the obvious way to deploy it.

    This asserts the lock is TAKEN, rather than trying to demonstrate a torn
    counter. I tried the latter first and it does not work: on a GIL build
    `self._fails += 1` loses nothing even with 8 threads, 160,000 increments and
    `setswitchinterval(1e-6)` — measured `lost=0`. So a "lost updates" test
    passes against deliberately unlocked code and proves nothing.

    The lock still matters, for two reasons the GIL does not cover: a
    free-threaded build (3.13+) tears these genuinely, and `allow()`'s
    open→half_open transition spans several statements, so two threads can both
    decide they are "the one probe" and hit a dependency the gate exists to
    protect from exactly that.
    """
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=0.0)

    class _CountingLock:
        def __init__(self) -> None:
            self.acquisitions = 0
            self._inner = threading.Lock()

        def __enter__(self) -> None:
            self.acquisitions += 1
            self._inner.acquire()

        def __exit__(self, *exc: object) -> None:
            self._inner.release()

    counting = _CountingLock()
    br._lock = counting  # type: ignore[assignment]

    br.allow()
    br.record_failure()
    br.allow()
    br.record_success()
    br.release_probe()

    assert counting.acquisitions == 5, (
        f"every entry point must take the lock; saw {counting.acquisitions}"
    )


def test_the_half_open_gate_survives_concurrent_threads() -> None:
    """The gate must stay a gate under contention: admit, release, admit — and
    never end in a torn state. A crash or an impossible state here is the
    failure; the exact admit count is timing-dependent and not asserted."""
    br = CircuitBreaker("upstream", fail_threshold=1, cooldown=0.0)
    br.record_failure()
    admitted: list[int] = []
    guard = threading.Lock()

    def race() -> None:
        for _ in range(500):
            if br.allow():
                with guard:
                    admitted.append(1)
                br.release_probe()

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert admitted, "nothing was ever admitted — the gate is stuck"
    assert br.state in ("open", "half_open", "closed")


def test_the_lock_does_not_break_equality_or_repr() -> None:
    """A dataclass field holding a lock would otherwise leak into `==` and
    `repr` — hence `compare=False, repr=False`. Pinned because dropping either
    flag is an easy tidy-up that breaks value semantics."""
    import dataclasses

    a = CircuitBreaker("x", fail_threshold=1, cooldown=1.0)
    b = CircuitBreaker("x", fail_threshold=1, cooldown=1.0)
    assert a == b, "two identically-configured breakers must compare equal"

    # Assert the FLAGS, not the repr string: `clock=<built-in function
    # monotonic>` contains the substring "lock", so a naive
    # `"lock" not in repr(a)` fails against correct code. (It did.)
    (lock_field,) = [f for f in dataclasses.fields(a) if f.name == "_lock"]
    assert lock_field.compare is False, "a lock in `==` breaks value semantics"
    assert lock_field.repr is False, "a lock in `repr` is noise"


def test_retries_actually_back_off() -> None:
    """A refactor moved `await asleep(backoff_delay(...))` below `return result`
    in an `else:` block, where it was unreachable. Every retry became a hot loop
    with zero delay — the retry storm `backoff_delay` exists to prevent.

    `ruff` did not flag the dead code and no test covered it; the injected
    `sleep=` simply stopped being called. Measured before the fix: three
    attempts produced `sleep calls: []`.
    """
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    assert _run(run_with_resilience(flaky, max_attempts=3, sleep=record)) == "ok"
    assert attempts["n"] == 3
    assert len(slept) == 2, f"expected a backoff between each retry, got {slept}"
    assert all(d > 0 for d in slept)


def test_a_call_that_succeeds_first_time_does_not_sleep() -> None:
    """POSITIVE CONTROL: putting the backoff back on the retry path must not put
    it on the success path — a delay after every successful call would be a
    latency regression on the overwhelmingly common case."""
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    async def fine() -> str:
        return "ok"

    assert _run(run_with_resilience(fine, max_attempts=3, sleep=record)) == "ok"
    assert slept == []


def test_a_permanent_failure_does_not_sleep_before_giving_up() -> None:
    """POSITIVE CONTROL: a PERMANENT error raises immediately, so there must be
    no backoff before it — sleeping before a failure that can never succeed is
    pure added latency."""
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    async def permanent() -> str:
        raise ValueError("401 unauthorized")

    with pytest.raises(ValueError):
        _run(run_with_resilience(permanent, max_attempts=3, sleep=record))
    assert slept == []


def test_mutating_a_requests_messages_is_now_refused_outright() -> None:
    """`len(messages)` is in `__hash__`, so appending to the list used to move a
    request to a different bucket and silently un-find it in any set or dict.

    This test used to PIN that sharp edge — "discovered here rather than in
    someone's cache". The payload freeze closes it at the source instead:
    `messages` is a `FrozenList`, so the append raises where it happens rather
    than corrupting a lookup three layers away. The hash is byte-for-byte
    unchanged; what changed is that the input it depends on can no longer move.

    Nothing in the framework did this anyway — middleware REPLACES a request
    (`ctx.request = replace(...)`) rather than mutating it, which is why
    `len(messages)` was affordable in the first place and stays so.
    """
    req = ChatRequest(messages=[Message(role="user", content="hi")], model="m")
    seen = {req}
    before = hash(req)

    with pytest.raises(TypeError, match="frozen value"):
        req.messages.append(Message(role="user", content="more"))

    # The request is exactly as findable as it was a line ago — which is the
    # whole point of refusing rather than pinning.
    assert hash(req) == before
    assert req in seen
    assert req == ChatRequest(messages=[Message(role="user", content="hi")], model="m")

    # The supported rewrite still works and produces a DIFFERENT value, as it
    # always did — replacement was never the thing under threat.
    grown = dataclasses.replace(req, messages=[*req.messages, Message(role="user", content="more")])
    assert grown != req and len(grown.messages) == 2
