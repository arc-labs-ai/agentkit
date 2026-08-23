"""The four REQUEST-PATH value types whose "frozen" payloads are now actually frozen.

`test_frozen_payloads.py` covers the primitive (`FrozenDict` / `FrozenList` /
`deep_freeze`). This file covers wiring it into the types the `Invoker` carries
through the middleware chain:

    ToolCall.arguments          the model's requested call — approval snapshot,
                                idempotency key and audit trail all read it
    ToolSchema.parameters       a SHARED advertisement, reused every turn
    ChatRequest.messages/       the unit of work, mid-flight through a chain
      .tools/.response_format
    ToolRequest.arguments       what was authorised vs. what gets executed

`ToolCall` is the headline, and it is a MIGRATION rather than a new freeze.
`arguments` was already immutable — as a `MappingProxyType` — so nothing here
is about gaining immutability. It is about gaining it back without the four
things a proxy gives up: `json.dumps`, `dataclasses.asdict`, `isinstance(x,
dict)` and depth. `asdict` is the one that shipped broken, and it was
contagious: `asdict` deep-copies every leaf, a mappingproxy has no pickle
protocol, and a ToolCall does not have to be the ARGUMENT — only reachable from
it — so `asdict` of anything holding one raised.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
from dataclasses import asdict, replace
from typing import Any

import pytest

from agentkit.kernel._frozen import FrozenDict, FrozenList
from agentkit.kernel.types import ChatRequest, Message, ToolCall, ToolRequest, ToolSchema

# Decoded provider JSON, which is what these payloads actually are: nested,
# heterogeneous, and never flat.
DEEP: dict[str, Any] = {
    "query": "weather",
    "filters": {"lang": "en", "tags": ["a", {"nested": [1, 2]}]},
    "limit": 10,
}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"q": {"type": "string", "enum": ["a", "b"]}},
    "required": ["q"],
}


def _values() -> list[Any]:
    """One instance of each type, each carrying `DEEP` in its payload field."""
    return [
        ToolCall("c1", "search", dict(DEEP)),
        ToolRequest("search", dict(DEEP), tool=None),
        ToolSchema("search", "find things", dict(SCHEMA)),
        ChatRequest([Message("user", "hi")], "gpt-4", response_format=dict(DEEP)),
    ]


# ── the bug: a frozen record whose payload was not ─────────────────────────
#
# These are the tests that FAIL without the `__post_init__` freeze. Each one is
# a rewrite of a record that has already been read, hashed or sent.


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.__setitem__("query", "evil"), id="setitem"),
        pytest.param(lambda d: d.pop("query"), id="pop-a-secret"),
        pytest.param(lambda d: d.update({"query": "evil"}), id="update"),
        pytest.param(lambda d: d.setdefault("query", "evil"), id="setdefault"),
        pytest.param(lambda d: d.clear(), id="clear"),
    ],
)
def test_toolcall_arguments_refuse_every_mutation_route(mutate) -> None:
    """`args.pop("token")` inside a tool impl is the real one: the ReAct
    approval snapshot, the idempotency key and the audit record are all taken
    from this dict at DIFFERENT moments, so an in-place edit makes what was
    approved, what was hashed and what ran three different things."""
    tc = ToolCall("c1", "search", dict(DEEP))
    with pytest.raises(TypeError, match="frozen value"):
        mutate(tc.arguments)
    assert tc.arguments == DEEP


def test_toolrequest_arguments_refuse_mutation() -> None:
    """One layer down, with a shorter fuse: the chain reads these arguments
    BEFORE the tool does (`egress_audit` pulls out `arguments[url_arg]`,
    `memoize` folds the whole dict into the idempotency key) and then hands the
    same object to `tool.run(args, ctx)`."""
    req = ToolRequest("search", dict(DEEP), tool=None, url_arg="query")
    with pytest.raises(TypeError, match="frozen value"):
        req.arguments["query"] = "http://evil"
    assert req.arguments["query"] == "weather"


def test_toolschema_parameters_refuse_mutation() -> None:
    """A schema is SHARED, not per-call: `ToolRegistry.schemas()` hands the same
    object to every request for the life of the process, so one adapter
    "fixing up" a property type would rewrite what every other provider is
    advertised from then on."""
    schema = ToolSchema("search", "d", dict(SCHEMA))
    with pytest.raises(TypeError, match="frozen value"):
        schema.parameters["properties"]["q"]["type"] = "number"
    assert schema.parameters["properties"]["q"]["type"] == "string"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.messages.append(Message("user", "more")), id="messages-append"),
        pytest.param(lambda r: r.messages.__setitem__(0, Message("user", "x")), id="messages-setitem"),
        pytest.param(lambda r: r.messages.clear(), id="messages-clear"),
        pytest.param(lambda r: r.tools.append(ToolSchema("evil")), id="tools-append"),
        pytest.param(lambda r: r.response_format.__setitem__("query", "x"), id="response_format-setitem"),
    ],
)
def test_chatrequest_containers_refuse_mutation(mutate) -> None:
    """A ChatRequest is the unit of work the chain is MID-WAY through running.
    Editing its list in place rewrites what the retry middleware will re-send,
    what `tracing` snapshots as `list(call.request.messages)` after the call,
    and what `memoize` already keyed on — all against a request that has
    already gone out on the wire."""
    req = ChatRequest([Message("user", "hi")], "gpt-4", tools=[ToolSchema("search")], response_format=dict(DEEP))
    with pytest.raises(TypeError, match="frozen value"):
        mutate(req)
    assert len(req.messages) == 1 and len(req.tools) == 1


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(lambda: ToolCall("c1", "s", dict(DEEP)).arguments, id="ToolCall.arguments"),
        pytest.param(lambda: ToolRequest("s", dict(DEEP), tool=None).arguments, id="ToolRequest.arguments"),
        pytest.param(
            lambda: ChatRequest([], "m", response_format=dict(DEEP)).response_format,
            id="ChatRequest.response_format",
        ),
    ],
)
def test_the_freeze_reaches_the_bottom_of_decoded_json(payload) -> None:
    """A shallow freeze is the same bug one level down and harder to find —
    and one level down is exactly where a model's arguments live. Every one of
    these paths is inside a single tool call's arguments."""
    p = payload()
    with pytest.raises(TypeError, match="frozen value"):
        p["filters"]["lang"] = "de"
    with pytest.raises(TypeError, match="frozen value"):
        p["filters"]["tags"].append("c")
    with pytest.raises(TypeError, match="frozen value"):
        p["filters"]["tags"][1]["nested"].append(3)


def test_a_caller_editing_its_own_dict_cannot_reach_the_stored_payload() -> None:
    """Freezing in place would refuse mutation THROUGH the value while leaving
    the caller holding a live handle to the same object. `deep_freeze` copies,
    which is what actually un-aliases it — at every level, not just the top."""
    mine: dict[str, Any] = {"query": "original", "filters": {"tags": ["a"]}}
    tc = ToolCall("c1", "search", mine)
    tr = ToolRequest("search", mine, tool=None)
    ts = ToolSchema("search", "d", mine)
    cr = ChatRequest([], "m", response_format=mine)

    mine["query"] = "changed"
    mine["filters"]["tags"].append("b")
    mine["new"] = "extra"

    for stored in (tc.arguments, tr.arguments, ts.parameters, cr.response_format):
        assert stored["query"] == "original"
        assert stored["filters"]["tags"] == ["a"]
        assert "new" not in stored


def test_a_caller_editing_its_own_message_list_cannot_reach_the_request() -> None:
    msgs = [Message("user", "hi")]
    tools = [ToolSchema("search")]
    req = ChatRequest(msgs, "gpt-4", tools=tools)
    msgs.append(Message("user", "smuggled"))
    tools.append(ToolSchema("evil"))
    assert len(req.messages) == 1 and len(req.tools) == 1


# ── the ToolCall migration: what the mappingproxy cost ─────────────────────


def test_asdict_of_a_toolcall_no_longer_raises() -> None:
    """THE regression this migration exists for. Measured before::

        dataclasses.asdict(ToolCall("c1", "search", {"q": "hi"}))
        TypeError: cannot pickle 'mappingproxy' object

    `asdict` deep-copies every leaf value, and a mappingproxy is not a `dict`
    subclass so it never takes the mapping branch — it is treated as an opaque
    leaf and handed to `copy.deepcopy`, which has no protocol for it. A
    `FrozenDict` IS a `dict` subclass, so the mapping branch runs and the
    result is a plain, JSON-ready dict all the way down."""
    out = asdict(ToolCall("c1", "search", dict(DEEP)))
    assert out == {"id": "c1", "name": "search", "arguments": DEEP}
    assert json.dumps(out)
    assert type(out) is dict  # the OUTER dict asdict builds is always plain

    # SHARP EDGE, worth knowing: `asdict` rebuilds each nested container as
    # `type(obj)(...)`, so a `FrozenDict` field comes back as a `FrozenDict`.
    # The freeze survives the serialisation, which is the right default for a
    # durable record — but it means `asdict` is NOT the "give me a mutable
    # copy" escape hatch a caller might assume. `dict(tc.arguments)` is.
    assert isinstance(out["arguments"], FrozenDict)
    with pytest.raises(TypeError, match="frozen value"):
        out["arguments"]["query"] = "evil"
    editable = dict(tc_args := ToolCall("c1", "search", dict(DEEP)).arguments)
    editable["query"] = "fine to edit a copy"
    assert tc_args["query"] == "weather"


def test_asdict_survives_a_toolcall_reached_INDIRECTLY() -> None:
    """The failure was contagious, which is what made it expensive. A ToolCall
    never had to be the argument — only reachable from it — so `asdict` of any
    record holding one raised. `AgentResult` round-trips through `asdict` and
    an assistant `Message` carries `tool_calls`."""

    @dataclasses.dataclass(frozen=True)
    class Record:
        turn: int
        message: Message

    msg = Message("assistant", tool_calls=(ToolCall("c1", "search", dict(DEEP)),))
    out = asdict(Record(turn=1, message=msg))
    assert out["message"]["tool_calls"][0]["arguments"] == DEEP
    assert json.dumps(out)


def test_toolcall_arguments_are_json_serialisable_without_an_unwrap() -> None:
    """Four call sites carry a defensive `dict(tc.arguments)` because
    `json.dumps` refused a mappingproxy outright (`context/tokens.py`,
    `checkpointer/persistence.py`, and both provider adapters). Those unwraps
    are now belt-and-braces rather than load-bearing; this is the line that
    used to raise."""
    tc = ToolCall("c1", "search", dict(DEEP))
    assert json.loads(json.dumps(tc.arguments)) == DEEP


def test_toolcall_arguments_are_a_dict_by_isinstance() -> None:
    """`FunctionTool._invoke` gates on `isinstance(args, Mapping)` and
    `resilience.stable_hash` branches on `isinstance(o, dict)`. A mappingproxy
    is a `Mapping` but NOT a `dict`, so it took the slow/other branch
    everywhere; a subclass takes the same branch a plain dict does."""
    tc = ToolCall("c1", "search", dict(DEEP))
    assert isinstance(tc.arguments, dict)
    assert isinstance(tc.arguments, FrozenDict)
    assert isinstance(tc.arguments["filters"]["tags"], list)


# ── positive controls: pass BOTH before and after ──────────────────────────


def test_construction_and_field_access_are_unchanged() -> None:
    tc = ToolCall("c1", "search", dict(DEEP))
    assert (tc.id, tc.name) == ("c1", "search")
    assert tc.arguments["filters"]["tags"][0] == "a"

    tr = ToolRequest("search", dict(DEEP), tool=None, side_effecting=True, url_arg="query")
    assert (tr.name, tr.side_effecting, tr.url_arg) == ("search", True, "query")

    ts = ToolSchema("search", "find things", dict(SCHEMA))
    assert (ts.name, ts.description) == ("search", "find things")
    assert ts.parameters["required"] == ["q"]

    cr = ChatRequest([Message("user", "hi")], "gpt-4", temperature=0.5, max_tokens=64)
    assert cr.messages[-1].content == "hi"
    assert (cr.model, cr.temperature, cr.max_tokens) == ("gpt-4", 0.5, 64)


def test_equality_against_plain_dicts_and_lists_still_holds() -> None:
    """Every consumer compares payloads against plain literals. A `FrozenDict`
    is `==` a `dict` in both directions; a `tuple` would not have been."""
    tc = ToolCall("c1", "search", dict(DEEP))
    assert tc.arguments == DEEP and DEEP == tc.arguments
    cr = ChatRequest([Message("user", "hi")], "m", tools=[ToolSchema("search")])
    assert cr.messages == [Message("user", "hi")]
    assert cr.tools == [ToolSchema("search")]


def test_value_equality_and_hashing_are_untouched_by_the_freeze() -> None:
    """HARD CONSTRAINT. `__eq__` still compares payloads; `__hash__` still
    excludes them (fixed in ed06eee / 7e0d4cd and deliberately not revisited).
    Freezing must not have leaked into either."""
    for a, b in zip(_values(), _values(), strict=True):
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    # ...and unequal payloads are still unequal, i.e. equality did not get
    # narrowed to whatever the hash looks at.
    assert ToolCall("c1", "s", {"a": 1}) != ToolCall("c1", "s", {"a": 2})
    assert ToolSchema("s", "d", {"a": 1}) != ToolSchema("s", "d", {"a": 2})
    assert ToolRequest("s", {"a": 1}, None) != ToolRequest("s", {"a": 2}, None)
    assert ChatRequest([], "m", response_format={"a": 1}) != ChatRequest([], "m", response_format={"a": 2})
    # Equal-but-different-bucket is fine; equal-and-different-HASH is not.
    assert hash(ToolCall("c1", "s", {"a": 1})) == hash(ToolCall("c1", "s", {"a": 2}))


def test_the_frozen_shell_itself_still_rejects_field_assignment() -> None:
    """The half that always worked. `__post_init__` uses `object.__setattr__`
    to get past the freeze once, at construction — it must not leave it open."""
    tc = ToolCall("c1", "search", {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        tc.arguments = {}  # type: ignore[misc]


def test_payloads_json_serialise_and_index_and_iterate_like_containers() -> None:
    for v in _values():
        payload = getattr(v, "arguments", None) or getattr(v, "parameters", None) or v.response_format
        assert json.loads(json.dumps(payload)) == payload
        assert len(payload) == len(dict(payload))
        assert list(payload) == list(dict(payload))
        assert dict(payload) == payload and type(dict(payload)) is dict
        for k in payload:
            assert payload[k] == dict(payload)[k]


def test_chatrequest_message_list_reads_exactly_as_before() -> None:
    """The expressions actual consumers use, verbatim: `memoize` keys on
    `c.request.messages[-1].content`, `tracing` snapshots
    `list(call.request.messages)`, `compaction` reads
    `list(ctx.request.messages)`, and `MiddlewareContext.messages` returns
    `list(getattr(self._call.request, "messages", []) or [])`."""
    msgs = [Message("user", "one"), Message("assistant", "two"), Message("user", "three")]
    req = ChatRequest(msgs, "gpt-4", tools=[ToolSchema("search")])

    assert req.messages[-1].content == "three"
    assert list(req.messages) == msgs and type(list(req.messages)) is list
    assert len(req.messages) == 3
    assert [m.role for m in req.messages] == ["user", "assistant", "user"]
    assert req.messages[1:] == msgs[1:]
    assert isinstance(req.messages, (list, FrozenList))
    assert bool(req.messages) is True and bool(ChatRequest([], "m").messages) is False
    # A caller's copy is a normal, mutable list — the freeze is on the stored
    # payload, never on what a reader takes away.
    taken = list(req.messages)
    taken.append(Message("user", "four"))
    assert len(req.messages) == 3


def test_the_supported_rewrite_route_still_works() -> None:
    """Middleware REPLACES a request rather than editing one; that is the API
    the freeze pushes callers onto, so it had better be pleasant."""
    req = ChatRequest([Message("user", "hi")], "gpt-4")
    grown = replace(req, messages=[*req.messages, Message("assistant", "yo")])
    assert len(grown.messages) == 2 and len(req.messages) == 1
    assert isinstance(grown.messages, FrozenList)  # ...and the new one is frozen too
    assert replace(req, model="claude").messages is req.messages  # unchanged field, no re-walk


@pytest.mark.parametrize("value", _values(), ids=lambda v: type(v).__name__)
def test_every_value_deepcopies_and_pickles_and_stays_frozen(value: Any) -> None:
    """REGRESSION GUARD. `Checkpointer.snapshot` deep-copies state containing
    ToolCalls and the replay recorder pickles it, so a payload that could not
    survive either would surface as a broken save, not a broken copy. The copy
    must also come back FROZEN, or the guarantee leaks through any round
    trip."""
    for clone in (copy.deepcopy(value), copy.copy(value), pickle.loads(pickle.dumps(value))):
        assert clone == value
        assert hash(clone) == hash(value)
        payload = getattr(clone, "arguments", None) or getattr(clone, "parameters", None) or clone.response_format
        assert isinstance(payload, FrozenDict)
        with pytest.raises(TypeError, match="frozen value"):
            payload["query"] = "evil"
        # Deep, not just at the top — a copy that re-froze only one level would
        # pass every assertion above.
        with pytest.raises(TypeError, match="frozen value"):
            payload["filters"]["tags"].append("c") if "filters" in payload else payload["properties"].clear()


def test_a_chatrequest_round_trips_with_its_message_list_frozen() -> None:
    req = ChatRequest([Message("user", "hi")], "gpt-4", tools=[ToolSchema("search", "d", dict(SCHEMA))])
    for clone in (copy.deepcopy(req), pickle.loads(pickle.dumps(req))):
        assert clone == req
        assert isinstance(clone.messages, FrozenList) and isinstance(clone.tools, FrozenList)
        with pytest.raises(TypeError, match="frozen value"):
            clone.messages.append(Message("user", "more"))
        with pytest.raises(TypeError, match="frozen value"):
            clone.tools[0].parameters["type"] = "array"


# ── edge cases ─────────────────────────────────────────────────────────────


def test_empty_payloads_freeze_rather_than_slipping_through() -> None:
    """The degenerate case is the one a `if payload:` guard would skip, which
    is how a freeze ends up applying to every record except the ones a test
    happens to build."""
    assert isinstance(ToolCall("c1", "ping").arguments, FrozenDict)
    assert isinstance(ToolSchema("ping").parameters, FrozenDict)
    assert isinstance(ToolRequest("ping", {}, tool=None).arguments, FrozenDict)
    assert isinstance(ChatRequest([], "m").messages, FrozenList)
    with pytest.raises(TypeError, match="frozen value"):
        ToolCall("c1", "ping").arguments["injected"] = 1
    with pytest.raises(TypeError, match="frozen value"):
        ChatRequest([], "m").messages.append(Message("user", "injected"))


def test_none_payloads_stay_none() -> None:
    """`tools is None` ("advertise nothing") and `tools == []` mean different
    things to the provider adapters, so the optional fields must not be
    materialised into empty containers by the freeze."""
    req = ChatRequest([Message("user", "hi")], "m")
    assert req.tools is None and req.response_format is None
    assert req.cache_hint is None
    assert replace(req, tools=None).tools is None


def test_cache_hint_is_deliberately_not_frozen() -> None:
    """`cache_hint` is annotated `Any` and holds a provider object we do not
    own. `deep_freeze` passes non-container leaves through untouched rather
    than guessing a reconstructor — the same line it draws everywhere else.
    A dict hint DOES get frozen, because a dict is a container it understands;
    what is pinned here is that a foreign object survives identically."""

    class ProviderHint:
        pass

    hint = ProviderHint()
    assert ChatRequest([], "m", cache_hint=hint).cache_hint is hint


def test_a_payload_shared_between_two_values_is_not_walked_twice() -> None:
    """`deep_freeze` is idempotent, so a ToolCall's arguments handed on to the
    ToolRequest the invoker builds from it costs one isinstance check, not a
    second O(payload) walk. That path is real: `react.py` builds a ToolRequest
    with `arguments=tc.arguments`."""
    tc = ToolCall("c1", "search", dict(DEEP))
    req = ToolRequest("search", tc.arguments, tool=None)
    assert req.arguments is tc.arguments
    assert req.arguments["filters"] is tc.arguments["filters"]
    # ...and re-freezing on `replace` does not rebuild either.
    assert replace(tc, name="other").arguments is tc.arguments


def test_a_toolcall_arrives_frozen_however_it_is_reconstructed() -> None:
    """The invariant has to survive the trip, whichever route rebuilds it.

    This used to call `tc.__reduce__()` directly and assert the shape of the
    tuple it returned. That pinned an IMPLEMENTATION detail, and it broke the
    moment `ToolCall.__reduce__` was deleted as redundant — `arguments` is a
    `FrozenDict`, which carries its own `__reduce__`, so the hand-written hook
    stopped earning its place. The guarantee never changed; only the mechanism
    did, and the test was written against the mechanism.

    So assert the guarantee: however a ToolCall is rebuilt — through the
    constructor from a plain dict, through pickle, through deepcopy — its
    payload arrives frozen, all the way down.
    """
    tc = ToolCall("c1", "search", dict(DEEP))

    rebuilt = [
        ToolCall("c1", "search", dict(DEEP)),  # constructor, from a plain dict
        pickle.loads(pickle.dumps(tc)),  # across a process boundary
        copy.deepcopy(tc),  # the checkpointer's route
    ]
    for got in rebuilt:
        assert isinstance(got.arguments, FrozenDict)
        with pytest.raises(TypeError, match="frozen value"):
            got.arguments["injected"] = "forged"
        assert got == tc


def test_deeply_nested_schema_bodies_freeze_all_the_way_down() -> None:
    """A JSON Schema nests by construction, and the levels that matter —
    `properties.<name>.items.properties` — are four deep."""
    body = {"properties": {"a": {"items": {"properties": {"b": {"enum": ["x"]}}}}}}
    ts = ToolSchema("search", "d", body)
    with pytest.raises(TypeError, match="frozen value"):
        ts.parameters["properties"]["a"]["items"]["properties"]["b"]["enum"].append("y")
    assert ts.parameters["properties"]["a"]["items"]["properties"]["b"]["enum"] == ["x"]
