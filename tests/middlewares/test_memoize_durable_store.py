"""A `memoize()`d chat must survive a DURABLE store — the production wiring, not the test wiring.

`memoize()` on a chat chain stores the assembled `LLMResult`. `InMemoryStore` keeps the object by
reference, so every test in this directory passed; every durable adapter serializes with
`json.dumps` (`adapters/store/file.py:95`, `redis.py:59`, `postgres.py:86`), so every durable
adapter failed. Measured before the fix, one `memoize()`d chat, `FakeLLM("hi")`::

    InMemoryStore : ok
    FileStore     : TypeError: Object of type LLMResult is not JSON serializable

Note WHERE it failed: the first cache WRITE, on the MISS. Not a wrong answer on a hit — the run
crashed before it ever had a hit, and it crashed only for the people who wired the cache to
something that outlives the process, which is the only reason to wire a cache at all.

`JSONStore` below is a stand-in for `RedisStore` / `PostgresStore`. Neither's driver (`redis`,
`asyncpg`) is installed in this environment, so they are covered BY INSPECTION plus this
stand-in, which reproduces their exact value path: `json.dumps` on write (`redis.py`:59,
`postgres.py`:86), `json.loads` on read (`redis.py`:41, `postgres.py`:62). If that is the whole
of their serialization — and it is — a value that survives `JSONStore` survives them.

`FileStore` is exercised through a SECOND `FileStore` instance over the same directory wherever a
hit is asserted. One instance would prove less than it looks: `FileStore.get_or_set` returns the
value the producer handed it on a miss, so a same-instance "hit" can be served without the bytes
ever making the round trip through the disk. A fresh instance shares nothing but the directory,
which is what a second process has.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from agentkit.adapters.store.file import FileStore
from agentkit.adapters.store.memory import InMemoryStore
from agentkit.capabilities.output_schema import adapt
from agentkit.kernel.errors import StoreUnavailable
from agentkit.kernel.middleware import Call, Handler
from agentkit.kernel.types import ChatRequest, LLMResult, Message, ToolCall, ToolRequest, Usage
from agentkit.middlewares import memoize, output_coerce
from agentkit.middlewares.memoize import _scoped, default_key
from agentkit.runtime import Invoker
from agentkit.testing import FakeLLM, Turn, make_test_ctx

pydantic = pytest.importorskip("pydantic")


class Plan(pydantic.BaseModel):
    subject: str
    steps: list[str]


VALID = '{"subject": "ship", "steps": ["a", "b"]}'


# ── stores ─────────────────────────────────────────────────────────────────────────────────


class JSONStore:
    """`RedisStore` / `PostgresStore` reduced to the part that matters here: values go through
    `json.dumps` on the way in and `json.loads` on the way out. Single-flight is presence-keyed,
    matching both real adapters (and `InMemoryStore`, the declared reference contract)."""

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}

    async def get(self, key: str) -> Any:
        raw = self._kv.get(key)
        return None if raw is None else json.loads(raw)

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        self._kv[key] = json.dumps(value)

    async def delete(self, key: str) -> None:
        self._kv.pop(key, None)

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        if key in self._kv:
            return await self.get(key)
        result = await fn()
        await self.set(key, result, ttl=ttl)
        return result


class BrokenStore(InMemoryStore):
    """A store whose BACKEND is down, as opposed to one that cannot serialize a value. The two
    must not be confused: the codec's skip path exists for the second, and swallowing the first
    would turn "your cache is unreachable" into a silent per-call provider bill."""

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        await fn()
        raise StoreUnavailable("connection refused")


# ── wiring helpers ─────────────────────────────────────────────────────────────────────────


def _request() -> ChatRequest:
    # A fresh but EQUAL request per call — `default_key` hashes the request's fields, so two
    # equal requests are one cache entry, and reusing one object would also hide an
    # identity-keyed cache.
    return ChatRequest(messages=[Message(role="user", content="hi")], model="m")


def _twice(
    llm: Any,
    make_store: Callable[[], Any],
    *,
    middleware: list[Any] | None = None,
    adapter: Any = None,
) -> tuple[Any, Any]:
    """Run the same request twice, each call against a FRESHLY CONSTRUCTED store handle over the
    same durable backing — see the module docstring for why a shared handle proves less."""
    inv = Invoker(llm=llm, chat_middleware=middleware or [memoize()])
    meta = {"output_adapter": adapter} if adapter is not None else None

    async def go() -> tuple[Any, Any]:
        first = await inv.chat(
            _request(), make_test_ctx(invoker=inv, store=make_store()), meta=dict(meta or {}) or None
        )
        second = await inv.chat(
            _request(), make_test_ctx(invoker=inv, store=make_store()), meta=dict(meta or {}) or None
        )
        return first, second

    return asyncio.run(go())


def _file_store(tmp_path: Path) -> Callable[[], Any]:
    return lambda: FileStore(base_dir=str(tmp_path))


def _shared(store: Any) -> Callable[[], Any]:
    return lambda: store


# ── the bug ────────────────────────────────────────────────────────────────────────────────


def test_a_chat_write_to_a_durable_store_does_not_raise(tmp_path: Path) -> None:
    """The narrowest statement of the defect: ONE call, which is a cache MISS, which is a cache
    WRITE. Measured before the fix::

        TypeError: Object of type LLMResult is not JSON serializable

    raised out of `FileStore.set` → `json.dumps` and straight into the caller's run. No second
    call needed; the production configuration broke on its first request."""
    llm = FakeLLM("hi")
    inv = Invoker(llm=llm, chat_middleware=[memoize()])
    ctx = make_test_ctx(invoker=inv, store=FileStore(base_dir=str(tmp_path)))

    result = asyncio.run(inv.chat(_request(), ctx))

    assert result.content == "hi"
    assert llm.calls == 1


def test_a_chat_result_round_trips_through_filestore(tmp_path: Path) -> None:
    """Miss then hit, across two `FileStore` handles — the second reads the bytes off disk, which
    is what a restarted process does. The hit must be the same answer, not merely a hit."""
    llm = FakeLLM("hi", usage=Usage(10, 5, 0.0001, 7, 3))

    first, second = _twice(llm, _file_store(tmp_path))

    assert llm.calls == 1, "the second call was served by the provider — nothing was proven"
    assert second == first
    assert isinstance(second, LLMResult)
    assert (second.content, second.model, second.provider, second.finish_reason) == (
        "hi",
        "m",
        "fake",
        "stop",
    )


def test_a_chat_result_round_trips_through_a_json_serializing_store() -> None:
    """The same claim for the `RedisStore` / `PostgresStore` value path — see the module docstring
    for why this is a stand-in rather than the adapters themselves."""
    llm = FakeLLM("hi")
    store = JSONStore()

    first, second = _twice(llm, _shared(store))

    assert llm.calls == 1
    assert second == first
    assert isinstance(second, LLMResult)


def test_every_usage_field_survives_the_round_trip(tmp_path: Path) -> None:
    """All FIVE `Usage` fields, not the three a checkpoint carries.

    `persistence.usage_to_dict` — the obvious helper to reuse — drops the cache counters:
    measured, `usage_to_dict(Usage(1, 2, 0.5, 7, 3))` is `{'input': 1, 'output': 2, 'cost': 0.5}`.
    In a checkpoint that is a reporting detail. Here it is money: `meter`/`Budget` price a
    cache-read token differently from a fresh prompt token, so a hit reading back
    `cache_read_tokens=0` misreports spend on every replay — silently, and only on the durable
    path, which is the one that bills."""
    llm = FakeLLM("hi", usage=Usage(11, 22, 0.5, 7, 3))

    _, second = _twice(llm, _file_store(tmp_path))

    assert llm.calls == 1
    assert second.usage == Usage(11, 22, 0.5, 7, 3)
    assert (second.usage.cache_read_tokens, second.usage.cache_write_tokens) == (7, 3)


def test_tool_calls_come_back_frozen(tmp_path: Path) -> None:
    """A cached tool-requesting turn must be as immutable as a fresh one.

    `ToolCall.arguments` is a `FrozenDict`; JSON gives back a plain `dict`, and rebuilding through
    `ToolCall` is what re-freezes it (`__post_init__` deep-freezes, nested levels included). This
    is not cosmetic — the same `ToolCall` flows into the ReAct approval snapshot, the
    idempotency-key hash and the audit trail, all three of which assume nothing can mutate it
    underneath them. A replayed turn must not be the weak one."""
    call = ToolCall("c1", "search", {"q": "hi", "n": {"deep": 1}})
    llm = FakeLLM.script([Turn(content="", tool_calls=(call,))], repeat_last=True)

    first, second = _twice(llm, _file_store(tmp_path))

    assert llm.calls == 1
    assert second == first
    assert second.tool_calls[0] == call
    args = second.tool_calls[0].arguments
    assert type(args).__name__ == "FrozenDict"
    assert type(args["n"]).__name__ == "FrozenDict", "the nested level thawed"
    with pytest.raises(TypeError):
        args["q"] = "mutated"
    with pytest.raises(TypeError):
        args["n"]["deep"] = 2


# ── the `parsed` policy: carry it, or refuse the entry — never downgrade it ─────────────────


def test_a_json_native_parsed_round_trips_on_a_durable_store(tmp_path: Path) -> None:
    """A `parsed` made of JSON-native values is stored verbatim and reads back `==`. This is the
    case that MUST keep caching — refusing everything would be a correct-but-useless fix."""

    class DictAdapter:
        """An output adapter whose parse yields a plain dict."""

        def parse(self, text: str) -> Any:
            return json.loads(text)

        def schema(self) -> dict[str, Any]:
            return {}

        def partial_parse(self, text: str) -> Any:
            return None

    llm = FakeLLM(VALID)

    first, second = _twice(
        llm,
        _file_store(tmp_path),
        middleware=[memoize(), output_coerce()],
        adapter=DictAdapter(),
    )

    assert llm.calls == 1, "a JSON-native parsed was refused — the cache is now useless"
    assert second.parsed == first.parsed == {"subject": "ship", "steps": ["a", "b"]}
    assert second == first


def test_a_pydantic_parsed_is_refused_rather_than_dropped(tmp_path: Path) -> None:
    """The design decision, pinned.

    A Pydantic `parsed` cannot be rebuilt from JSON without importing a class name out of a cache
    entry, so on a serializing store the entry is REFUSED: the call runs, the caller gets the
    complete typed result, and nothing is stored. Both calls therefore reach the provider
    (`llm.calls == 2`) and both return a real `Plan`.

    The alternative — store the entry with `parsed` dropped to `None` — is the bug commit
    `5bb104a` fixed, one layer down: a miss returning `Plan(subject='ship', …)` and a hit
    returning `None`, from the same input, decided by cache state. An uncached call costs a
    provider round trip. A wrongly-cached one costs a wrong answer at the one call site that
    reads `.parsed`."""
    llm = FakeLLM(VALID)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first, second = _twice(
            llm,
            _file_store(tmp_path),
            middleware=[memoize(), output_coerce()],
            adapter=adapt(Plan),
        )

    assert llm.calls == 2, "a Pydantic parsed was cached — check what came back on the hit"
    assert isinstance(first.parsed, Plan) and isinstance(second.parsed, Plan)
    assert second.parsed == first.parsed == Plan(subject="ship", steps=["a", "b"])
    assert second == first, "the uncached path must return the same answer, not a degraded one"
    assert not list(tmp_path.glob("*memo*")), "something was written for an entry we refused"
    assert any("memoize()" in str(w.message) for w in caught), "the skip was silent"


def test_a_tuple_parsed_is_refused_rather_than_listified(tmp_path: Path) -> None:
    """The failure mode that REPORTS SUCCESS, which is the one worth a test.

    A tuple is not rejected by `json.dumps` — it is quietly rewritten. Measured::

        json.dumps({"steps": ("a", "b")})  -> '{"steps": ["a", "b"]}'
        json.loads(...) == {"steps": ("a", "b")}  -> False

    So a codec that let `parsed` through on the strength of "`json.dumps` didn't raise" would
    store a tuple and hand back a list on every hit: a cache that changes the TYPE of the answer.
    The walk is strict about which types round-trip unchanged, so this call is simply not cached."""

    class TupleAdapter:
        def parse(self, text: str) -> Any:
            return ("a", "b")

        def schema(self) -> dict[str, Any]:
            return {}

        def partial_parse(self, text: str) -> Any:
            return None

    llm = FakeLLM(VALID)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        first, second = _twice(
            llm,
            _file_store(tmp_path),
            middleware=[memoize(), output_coerce()],
            adapter=TupleAdapter(),
        )

    assert llm.calls == 2
    assert first.parsed == second.parsed == ("a", "b")
    assert isinstance(second.parsed, tuple), "the cache turned a tuple into a list"


def test_the_refusal_is_visible_to_a_programmatic_reader(tmp_path: Path) -> None:
    """The warning is latched to once per `memoize()` instance so a structured chain does not
    print it a thousand times — so `call.meta["cache_stored"] = False` is the channel that fires
    on EVERY refused call. A repo with a documented history of broad `except` clauses that report
    success does not get to skip a cache entry quietly."""
    seen: list[dict[str, Any]] = []

    async def recorder(call: Call, nxt: Handler) -> AsyncIterator[Any]:
        async for item in nxt(call):
            yield item
        seen.append(dict(call.meta))  # same Call object the chain shares

    llm = FakeLLM(VALID)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _twice(
            llm,
            _file_store(tmp_path),
            middleware=[recorder, memoize(), output_coerce()],
            adapter=adapt(Plan),
        )

    assert [m.get("cache_stored") for m in seen] == [False, False], "a refusal went unrecorded"
    assert sum("memoize()" in str(w.message) for w in caught) == 1, "the warning is not latched"


def test_an_object_store_still_carries_a_typed_parsed_through_a_hit() -> None:
    """`InMemoryStore` keeps values by reference, so the typed object survives a hit there exactly
    as it did before — the behaviour `test_memoize_preserves_parsed.py` pins. The refusal above is
    a property of a SERIALIZING store, not a new blanket rule about `parsed`."""
    llm = FakeLLM(VALID)
    store = InMemoryStore()

    first, second = _twice(
        llm, _shared(store), middleware=[memoize(), output_coerce()], adapter=adapt(Plan)
    )

    assert llm.calls == 1
    assert isinstance(second.parsed, Plan)
    assert second.parsed is first.parsed, "the object store handed back a rebuilt copy"
    assert second == first


# ── positive controls (must pass BEFORE and AFTER the fix) ─────────────────────────────────


def _lookup_tool() -> Any:
    class _ReadOnlyTool:
        side_effecting = False
        runs = 0

        async def run(self, arguments: dict[str, Any], ctx: Any) -> dict[str, Any]:
            _ReadOnlyTool.runs += 1
            return {"tool": "lookup", "n": _ReadOnlyTool.runs, **arguments}

    return _ReadOnlyTool()


@pytest.mark.parametrize("durable", ["file", "json"])
def test_tool_memoization_still_dedupes_on_a_durable_store(tmp_path: Path, durable: str) -> None:
    """A tool result is already JSON-shaped — it is whatever the tool returned — so the durable
    path worked for tools before this fix and must be untouched by it. The codec is chat-only for
    exactly this reason: wrapping a tool payload would change the format of every live tool entry
    to fix a problem tools never had."""
    tool = _lookup_tool()
    inv = Invoker(llm=None, tool_middleware=[memoize()])
    shared = JSONStore()

    def store() -> Any:
        return FileStore(base_dir=str(tmp_path)) if durable == "file" else shared

    async def go() -> tuple[Any, Any]:
        first = await inv.invoke_tool(
            ToolRequest("lookup", {"id": 7}, tool), make_test_ctx(invoker=inv, store=store())
        )
        second = await inv.invoke_tool(
            ToolRequest("lookup", {"id": 7}, tool), make_test_ctx(invoker=inv, store=store())
        )
        return first, second

    first, second = asyncio.run(go())

    assert type(tool).runs == 1, "the tool re-ran — the durable tool cache stopped deduping"
    assert first == second == {"tool": "lookup", "n": 1, "id": 7}


def test_a_failing_producer_is_never_cached_and_never_swallowed(tmp_path: Path) -> None:
    """`memoize` now catches `TypeError`/`ValueError` around `get_or_set` to detect a store that
    cannot serialize the value. That catch must not become a net under the PRODUCER: a `TypeError`
    raised by the provider call is a real failure of the run and has to propagate, unstored.

    Deliberately raised as a `TypeError` — the exact type the new handler looks for — because a
    handler that keys on the type alone rather than on "did the producer complete" would turn this
    provider crash into a silent, uncached, empty success.

    Not a both-ways control despite living next to them: it also fails pre-fix, for the incidental
    reason that the retry's cache write is the original defect."""
    llm = FakeLLM("hi", fail_times=1, fail_exc=TypeError("provider blew up"))
    inv = Invoker(llm=llm, chat_middleware=[memoize()])
    store = FileStore(base_dir=str(tmp_path))

    async def go() -> Any:
        with pytest.raises(TypeError, match="provider blew up"):
            await inv.chat(_request(), make_test_ctx(invoker=inv, store=store))
        # The failure was not cached: the retry produces a real answer.
        return await inv.chat(_request(), make_test_ctx(invoker=inv, store=store))

    assert asyncio.run(go()).content == "hi"
    assert llm.calls == 2


def test_an_unreachable_store_still_fails_the_run() -> None:
    """A backend that is DOWN is not a value that cannot be serialized, and the new handler must
    not conflate them. `StoreUnavailable` propagates exactly as it did before — a cache layer
    quietly absorbing "your store is unreachable" would leave an operator paying per call with
    nothing in the logs to explain it."""
    llm = FakeLLM("hi")
    inv = Invoker(llm=llm, chat_middleware=[memoize()])
    ctx = make_test_ctx(invoker=inv, store=BrokenStore())

    with pytest.raises(StoreUnavailable):
        asyncio.run(inv.chat(_request(), ctx))


def test_an_object_store_entry_written_before_the_codec_still_reads_back() -> None:
    """Back-compat, such as it is. No durable store can be holding a chat entry written by the old
    code — those writes RAISED — so the only pre-existing entries are `InMemoryStore` objects,
    which die with the process. The decoder is tolerant anyway: a raw `LLMResult` sitting at the
    key comes back untouched rather than being mistaken for a malformed envelope. Seeded here by
    hand because the situation is otherwise unreachable."""
    llm = FakeLLM("fresh")
    inv = Invoker(llm=llm, chat_middleware=[memoize()])
    store = InMemoryStore()
    seeded = LLMResult(content="from the old format", model="m", provider="fake")

    async def go() -> Any:
        ctx = make_test_ctx(invoker=inv, store=store)
        probe = Call(kind="chat", request=_request(), ctx=ctx)
        await store.set(_scoped(probe, default_key(probe)), seeded)
        return await inv.chat(_request(), ctx)

    result = asyncio.run(go())

    assert result.content == "from the old format"
    assert llm.calls == 0


def test_a_chat_hit_equals_the_miss_on_every_adapter(tmp_path: Path) -> None:
    """One assertion, three adapters: whatever the store does to the bytes, call 2 is call 1."""
    for make_store in (
        _shared(InMemoryStore()),
        _file_store(tmp_path),
        _shared(JSONStore()),
    ):
        llm = FakeLLM("hi", usage=Usage(3, 4, 0.02, 1, 2))
        first, second = _twice(llm, make_store)
        assert llm.calls == 1
        assert first == second


def test_a_tempfile_backed_filestore_is_actually_durable() -> None:
    """The trap this file is built to avoid: a "durable" store that is secretly in-process proves
    nothing. Two handles, two `_locks` tables, one directory — the second reads bytes."""
    with tempfile.TemporaryDirectory() as d:
        a, b = FileStore(base_dir=d), FileStore(base_dir=d)
        assert a is not b

        async def go() -> Any:
            await a.set("k", {"v": 1})
            return await b.get("k")

        assert asyncio.run(go()) == {"v": 1}


def test_a_tool_result_a_durable_store_cannot_encode_no_longer_kills_the_run(tmp_path: Path) -> None:
    """The deliberate widening, pinned. The codec is chat-only, but the "cannot serialize it →
    don't cache it" handler is not: a READ-ONLY tool that returns a live object (an ORM row, a
    Pydantic model) used to take the whole run down on the cache WRITE — measured,
    `TypeError: Object of type Plan is not JSON serializable` out of `FileStore.set`. The call had
    already succeeded; only the bookkeeping failed. Now the caller gets the object, the entry is
    skipped, and the skip is announced."""

    class _ObjectTool:
        side_effecting = False
        runs = 0

        async def run(self, arguments: dict[str, Any], ctx: Any) -> Any:
            _ObjectTool.runs += 1
            return Plan(subject="ship", steps=["a", "b"])

    tool = _ObjectTool()
    inv = Invoker(llm=None, tool_middleware=[memoize()])

    async def go() -> tuple[Any, Any]:
        first = await inv.invoke_tool(
            ToolRequest("build", {}, tool),
            make_test_ctx(invoker=inv, store=FileStore(base_dir=str(tmp_path))),
        )
        second = await inv.invoke_tool(
            ToolRequest("build", {}, tool),
            make_test_ctx(invoker=inv, store=FileStore(base_dir=str(tmp_path))),
        )
        return first, second

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first, second = asyncio.run(go())

    assert first == second == Plan(subject="ship", steps=["a", "b"])
    assert _ObjectTool.runs == 2, "an entry we could not store was somehow served back"
    assert any("memoize()" in str(w.message) for w in caught)
