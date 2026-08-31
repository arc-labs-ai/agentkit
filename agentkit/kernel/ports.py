"""L0 seams — the infra ports the core owns. The tracing seam is an injected `TracePort`
(no-op by default; observability is never a required dependency).

A seam is an *external system agentkit cannot implement itself* (a model, the world, a vector index,
a durable store, web search, the network, the wall clock). Features (idempotency, audit, checkpoint,
cache, quota) are NOT ports — they are middlewares/meters backed by `StorePort`. Pure `Protocol`s,
no third-party imports. A signature change here is a deliberate, reviewed event.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agentkit.kernel._frozen import deep_freeze

if TYPE_CHECKING:  # avoid import cycles at runtime
    from agentkit.kernel.types import Chunk, Delta, LLMResult, Message, Scope, ToolSchema


@runtime_checkable
class LLMPort(Protocol):
    def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        cache_hint: Any = None,
    ) -> AsyncIterator[Delta]: ...  # THE primitive — the Invoker's chat terminal iterates it

    async def chat(
        self,
        *,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        cache_hint: Any = None,
    ) -> LLMResult: ...  # derived: ≡ collect(stream(...)); kept for direct callers

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult: ...  # single-call convenience; a thin shim over chat() on real adapters


@runtime_checkable
class ToolPort(Protocol):
    name: str
    schema: Any  # optional ToolSchema; None → loop-invisible

    async def run(self, args: Any, ctx: Any) -> Any: ...  # typed I/O via Pydantic models on impls


@runtime_checkable
class VectorPort(Protocol):
    # search returns (score, chunk) so callers can threshold (RAG ignores, semantic-cache uses it).
    async def upsert(self, scope: Scope, chunks: list[Chunk]) -> None: ...
    async def search(
        self, scope: Scope, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[tuple[float, Chunk]]: ...


@runtime_checkable
class StorePort(Protocol):
    """The single KV seam that backs the former cache / idempotency / checkpoint / audit ports.
    `get_or_set` is single-flight; a producer that raises is NOT stored (failures are never cached).

    The last three methods are the *coordination* half. Without them the seam
    could express "create if absent" and "read one log back", and nothing else,
    so three ordinary shapes had to go around the port entirely:

    * **read-modify-write.** Allocating a monotonic ordinal is "read max, write
      max+1", and two writers race it. `get_or_set` cannot express "replace
      only if unchanged".
    * **a counter with an expiry** — the shape of every rate limit. The
      observed consequence was an application writing a raw Lua script straight
      at Redis, which put the check outside everything the framework can test,
      trace or meter.
    * **enumerating a prefix.** "Everything recorded for this run" was
      answerable only if every writer had also maintained an index by hand.
    """

    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None: ...
    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any: ...
    async def delete(self, key: str) -> None: ...  # idempotent remove (e.g. clear a checkpoint)
    async def append(self, key: str, value: Any) -> None: ...  # append-only log (audit)
    async def list(self, key: str) -> list[Any]: ...  # read an appended log back

    async def compare_and_set(
        self, key: str, expected: Any, value: Any, *, ttl: int | None = None
    ) -> bool:
        """Replace `key` with `value` only if it currently equals `expected`; report whether it applied.

        **Returns a bool rather than raising.** Losing a compare-and-set is an
        ordinary outcome that the caller retries — it is the *expected* half of
        an optimistic loop, not a fault. Raising would force every
        read-modify-write to wrap its own body in a ``try/except`` and to tell
        "someone beat me" apart from "the store is down" by exception type,
        which is precisely the distinction the return value already makes.

        Comparison is by EQUALITY, never identity: three of the four backends
        round-trip the value through JSON, so the caller never holds the object
        that was written and an identity check would make CAS on any container
        permanently impossible while appearing to work in memory.

        An absent key compares equal to ``expected=None``, so the first
        iteration of a read-modify-write loop behaves like every later one.
        That deliberately collapses "absent" and "stored null" — but so does
        `get`, which is where the caller's ``expected`` came from, so the
        distinction was never observable at the call site anyway.

        ``ttl`` applies only on the write that lands; a refused CAS leaves the
        existing expiry alone.
        """
        ...

    async def increment(self, key: str, by: int = 1, *, ttl: int | None = None) -> int:
        """Atomically add `by` to the integer at `key` (absent ⇒ 0) and return the new total.

        `by` may be negative — refunds and released reservations are
        decrements, and a counter that only went up would need a second key to
        track what came back.

        ``ttl`` opens a window on the increment that finds no window; it never
        slides one that already exists (Redis's ``EXPIRE key ttl NX``). The
        alternative — resetting on every increment — breaks the exact case the
        primitive is for: under sustained traffic the counter is touched more
        often than the window is long, so it never expires, and the limit never
        resets. The rate limiter would jam shut precisely under load. The
        counter and its window are set together, atomically, so a failure
        cannot leave a counter with no window (an immortal one) or a window
        with no counter.

        A key holding a non-integer raises `StoreValueError` — the same type
        and message on every backend, so the caller does not have to know
        whether Redis, Postgres or a dict answered. A stored ``null`` counts as
        a non-integer, NOT as absent, which is the one place this deliberately
        disagrees with `compare_and_set`: `compare_and_set`'s ``expected`` came
        out of a `get`, which cannot tell null from absent, so it has to accept
        both; `increment` is handed no such value and the durable backends
        genuinely cannot add to a JSON ``null``.
        """
        ...

    def scan(self, prefix: str, *, limit: int | None = None) -> AsyncIterator[str]:
        """Yield the KV keys beginning with `prefix` — full keys, in no promised order.

        Declared ``def`` returning an `AsyncIterator`, not ``async def``, for
        the same reason as `LLMPort.stream`: an ``async def`` that yields is a
        *function returning an iterator*, not a coroutine, so the ``async def``
        spelling would reject every real implementation under a strict type
        check. Call sites are identical — ``async for k in store.scan(p)``.

        Keys, not values, and never the append-log namespace: `list(key)` reads
        logs, and a scan that surfaced log keys would hand back keys `get`
        answers ``None`` for. A key equal to the prefix is included — a summary
        record stored at the prefix itself is part of "everything under it".

        ``limit`` is a cap on how many keys the caller is willing to receive,
        NOT a page cursor: with no ordering promised, *which* keys arrive under
        a limit is backend-defined. ``0`` is a real cap of zero and a negative
        limit raises `ValueError`.

        Iteration is concurrent-safe in the weak sense that matters: it must
        not raise when writers land mid-scan. No snapshot is promised — only
        that keys present throughout appear, exactly once.
        """
        ...


@dataclass(frozen=True)
class SearchHit:
    """One web-search result — every `SearchPort` adapter normalizes to this shape so callers
    (Researcher agents, retrieval middlewares) are provider-agnostic."""

    url: str
    title: str
    snippet: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A hit is FANNED OUT: the same object is unioned across providers,
        # re-ranked, cached, and handed to several middlewares, so a reranker
        # that did ``hit.metadata["score"] = ...`` would rewrite the record
        # every other consumer holds. Deep, because provider metadata nests
        # (``{"citations": [{"snippet": ...}]}``) and a shallow freeze just
        # moves the same bug one level down.
        #
        # Frozen dataclass — assign through ``object.__setattr__``.
        # Measured: 2.78 µs construction for a 3-key metadata dict against
        # 1.12 µs before, and 1.86 µs vs 1.14 µs for the default-empty case —
        # on a type produced at most `k` times per query, behind a network
        # call three orders of magnitude larger.
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))

    def __hash__(self) -> int:
        """Hash on the RESULT's identity — ``(url, title)`` — never on the rest.

        A frozen dataclass derives ``__hash__`` from every compared field, and
        ``metadata`` is a plain ``dict``, so the generated hash was dead on
        arrival. Measured before this fix::

            hash(SearchHit("https://x", "t", "s", 0.5, {"domain": "x"}))
            TypeError: unhashable type: 'dict'

        Hits are the type callers most obviously want in a ``set``: fanning the
        same query across two providers and unioning the results is the whole
        point of normalizing every adapter onto one shape, and that union
        raised on the default-constructed ``metadata`` of every hit.

        ``url`` + ``title`` is the identity because a web result IS its URL —
        that is what provider-crossing dedup keys on — and the title travels
        with it as the normalized record's stable half. ``score`` and
        ``snippet`` are excluded because they are QUERY-dependent, not
        result-dependent: the same document ranked 1st for one query and 4th
        for another, with a different snippet extracted each time, is one
        document and belongs in one bucket. ``metadata`` is excluded because it
        is arbitrary provider JSON — nested lists and dicts, unhashable by
        construction — and because a hash that reads it would be O(payload).

        Excluding them is sound rather than a workaround: ``__eq__`` still
        compares every field, and the hash invariant only requires that EQUAL
        objects hash equally — never that unequal ones differ. Two hits for one
        URL with different scores collide into one bucket and ``__eq__``
        separates them there; that is what a bucket is for.
        """
        return hash((self.url, self.title))


@runtime_checkable
class SearchPort(Protocol):
    # `where` is a metadata filter (date range, domain allowlist) — same pattern as VectorPort.
    async def search(
        self, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchHit]: ...


@dataclass(frozen=True)
class FetchResponse:
    """A text HTTP response. Binary fetch is out of scope — callers that need bytes use a different
    port. `fetched_at` is a unix timestamp from the implementation's `ClockPort` (real or fake)."""

    url: str
    status: int
    headers: dict[str, str]
    body: str
    content_type: str
    fetched_at: float

    def __post_init__(self) -> None:
        # ``headers`` only. ``body`` is a ``str`` — already immutable, and
        # ``deep_freeze`` would pass it straight through anyway — so there is
        # nothing to freeze there and no cost paid on the field that is
        # actually large. That asymmetry is the point: a fetched page is
        # unbounded (see ``__hash__``), and freezing walks the payload, so a
        # container body would have made every response O(page). It isn't one.
        #
        # ``headers`` is worth freezing despite being small because a
        # FetchResponse is CACHED: the same object is served to later callers,
        # and a middleware normalising ``resp.headers["etag"]`` in place would
        # rewrite the cached entry for everyone behind it.
        #
        # Measured: 3.47 µs construction for a 5-header response against
        # 1.23 µs before — paid once per network fetch, i.e. against a
        # milliseconds-scale operation.
        object.__setattr__(self, "headers", deep_freeze(self.headers))

    def __hash__(self) -> int:
        """Hash on the RESPONSE's identity — ``(url, status, content_type,
        fetched_at)`` — never on ``headers`` or ``body``.

        ``headers`` is a plain ``dict``, so the generated all-fields hash was
        unhashable from the first instance. Measured before this fix::

            hash(FetchResponse("https://x", 200, {"content-type": "text/html"},
                               "<html>hi</html>", "text/html", 1.7e9))
            TypeError: unhashable type: 'dict'

        ``fetched_at`` is in the key on purpose: two fetches of the same URL at
        different times are different responses (that is what makes this record
        cacheable at all), and the timestamp is the only field that separates
        them without reading the payload.

        ``body`` is excluded even though ``str`` IS hashable, and that is the
        load-bearing exclusion. A fetched page is unbounded — this is the type
        a crawler holds thousands of — and ``hash(str)`` is O(len) with no
        cached digest across instances, so a body-inclusive hash would make
        every ``set`` insertion cost a full scan of the document. Measured on
        this implementation: 0.23 µs for a 12-byte body and 0.23 µs for a
        4 MiB one — identical, because the body is never read — against 2.3 ms
        to hash that 4 MiB string once, a ~10_000× gap. CPython caches a
        string's hash on the string object, so a body-inclusive hash would be
        cheap on REPEATED hashes of the same instance and expensive on the
        first hash of each — which, for the crawler inserting each response
        into a set exactly once, is every one of them.

        Excluding them is sound rather than a workaround: ``__eq__`` still
        compares every field, and the hash invariant only requires that EQUAL
        objects hash equally. Two responses that differ only in body — a page
        re-fetched at the same timestamp after an edit — collide into one
        bucket, where ``__eq__`` tells them apart.
        """
        return hash((self.url, self.status, self.content_type, self.fetched_at))


@runtime_checkable
class FetchPort(Protocol):
    # The implementation enforces the URL allowlist + caching + rate limits; callers don't pass them.
    async def fetch(self, url: str, *, timeout_s: float = 10.0) -> FetchResponse: ...


@runtime_checkable
class ClockPort(Protocol):
    """Deterministic time seam — tests and replay need a fakeable wall clock. `now()` returns a
    unix timestamp; `sleep(s)` waits without blocking the loop. Use this instead of `time.time()` /
    `datetime.now()` / `asyncio.sleep` inside middlewares + retry/backoff loops."""

    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class CheckpointStatus(StrEnum):
    """Coarse durability gate for a `Checkpoint`. Auto-resume keys off
    this — a `suspended` checkpoint says "waiting on a human";
    `running` says "engine is in motion"; `done` / `failed` are
    terminal. Producers map their domain phases (e.g., the api's
    `RunPhase`) onto this taxonomy at snapshot time.

    Emission map (grep-able index — keep aligned with call sites):

    * ``RUNNING`` — the default snapshot status at every mid-run
      durable transition (``Checkpointer.snapshot(...)`` with no
      explicit status). Set by the ReAct cognition after each
      tool-loop iteration and by coordinator policies at each turn.
    * ``SUSPENDED`` — emitted by the ReAct cognition's human-approval
      path when a gated tool call needs a human decision
      (``agentkit/agents/cognition/react.py`` — the ``_save(...,
      status="suspended", pending=...)`` call site). The suspended
      snapshot carries the ``pending`` tool calls so ``resume()`` can
      apply per-call decisions. ``StrEnum`` equivalence means the
      string ``"suspended"`` at the emission site IS
      ``CheckpointStatus.SUSPENDED`` when the adapter round-trips it
      back (see ``tests/adapters/test_postgres_checkpoint.py`` for
      the wire-level round-trip). Coordinators + custom producers
      MUST use this status (not ``RUNNING``) when persisting a
      human-gate wait — auto-resume relies on it to distinguish
      "engine is chugging" from "waiting on the world".
    * ``DONE`` / ``FAILED`` — emitted by the producer at terminal
      transitions; ``Checkpointer.resume`` filters these by default
      so a "resume if any checkpoint exists" wiring cannot silently
      re-run a finished job.
    """

    RUNNING = "running"
    SUSPENDED = "suspended"
    DONE = "done"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """True if no further work is expected — auto-resume should
        NOT fire on a `done` or `failed` checkpoint."""
        return self in {CheckpointStatus.DONE, CheckpointStatus.FAILED}


@dataclass(frozen=True)
class Checkpoint:
    """A serialized snapshot of run state at a meaningful transition.

    `state` is opaque to the port — the application (a leaf `Agent`, a coordinator `Agent`,
    an application-defined state snapshot) interprets it. `version` is monotonic: v1 is the
    first save for a run, v2 the next, and so on. `status` is a `CheckpointStatus` and
    lets the port surface "is this run resumable?" without re-parsing `state`.
    `created_at` is a unix timestamp stamped by the producer's `ClockPort` (or
    `time.time()` when no clock is wired in) so replay and audit lines up across
    services."""

    run_id: str
    version: int
    state: dict[str, Any]
    created_at: float
    status: CheckpointStatus
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the payloads. This is the type the whole exercise is about.

        A Checkpoint is a DURABLE record — it has been written to a JSONB
        column — and ``frozen=True`` stopped at the field reference, so
        ``cp.state = {}`` raised while ``cp.state["turn"] = 99`` silently
        rewrote the run's snapshot in memory after the row was committed. The
        in-memory record and the durable row then disagree, and nothing says
        so: the next ``latest()`` on a real backend returns the row, the next
        ``latest()`` on ``InMemoryCheckpointStore`` returns the mutated object.

        This runs on EVERY construction, which is the half that matters most:
        ``PostgresCheckpointStore._row_to_checkpoint`` and
        ``StoreBackedCheckpointStore._from_dict`` both rebuild through this
        constructor, so a checkpoint comes back frozen on the way OUT of
        storage as well as on the way in. A record that is frozen on save and
        mutable on resume is only half fixed — the resume path is precisely
        where a caller reaches for ``cp.state.pop("pending")``.

        The durable path is unaffected, and that was the constraint this had to
        clear before it could ship. ``FrozenDict`` is a ``dict`` SUBCLASS
        (see ``_frozen.py`` for why ``MappingProxyType`` was rejected), so
        ``json.dumps(cp.state)`` produces BYTE-IDENTICAL output to the plain
        dict, ``dataclasses.asdict`` still walks it, ``_to_dict`` still passes
        it through verbatim, and equality against a plain-dict checkpoint still
        holds. Verified end to end against ``StoreBackedCheckpointStore``,
        ``InMemoryCheckpointStore`` and the Postgres row decoder in
        ``tests/kernel/test_frozen_value_payloads.py``.

        Cost is O(state), paid once per snapshot and measured BELOW the
        ``copy.deepcopy(state)`` that ``Checkpointer.snapshot`` already
        performs on the very same dict one line earlier. For a realistic
        20-turn transcript state (20 messages of 400 chars, each with a tool
        call, plus a 20-key scratchpad): construction goes 1.34 µs → 88.5 µs,
        against 146 µs for the deepcopy sitting directly above it. So the seam
        that produces checkpoints gets ~60% more expensive on a path that was
        already walking the payload twice — and it does that once per durable
        write, next to a database round trip. The freeze is not a new class of
        work at this seam, it is a second, cheaper walk.

        Frozen dataclass — assign through ``object.__setattr__``.
        """
        object.__setattr__(self, "state", deep_freeze(self.state))
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))

    def __hash__(self) -> int:
        """Hash on the DURABLE key — ``(run_id, version)`` — never on ``state``
        or ``metadata``.

        A frozen dataclass derives ``__hash__`` from every compared field, and
        two of these fields are plain ``dict``s, so every checkpoint ever built
        was unhashable. Measured before this fix::

            hash(Checkpoint("r", 1, {"turn": 3}, 1.7e9, CheckpointStatus.RUNNING))
            TypeError: unhashable type: 'dict'

        This is the clearest instance of the shape the value-type ratchet was
        built to catch: `Checkpoint` is declared frozen and is reasoned about
        everywhere as an immutable snapshot of a run, while ``cp.state["turn"]
        = 99`` rewrites the record in place through the "frozen" field. This
        commit does NOT close that half — see the note below — it closes the
        hashability half, which is the part that can be fixed without touching
        what every persistence path already reads.

        ``(run_id, version)`` is not a convenient subset, it is the record's
        actual primary key: it is the ``PRIMARY KEY (run_id, version)`` of the
        `agentkit_checkpoints` table in `adapters/checkpoint/postgres.py`, and
        the class docstring above already states the invariant that makes it
        one — ``version`` is monotonic per run, and a single producer is the
        authority over a ``run_id``. So two checkpoints hash equal exactly when
        they name the same durable row, which is the discrimination a caller
        holding checkpoints in a ``set``/``dict`` wants.

        ``state`` and ``metadata`` are excluded for two independent reasons,
        either of which alone would settle it:

        * They CANNOT be hashed. State is opaque application JSON — nested
          lists and dicts, unhashable by construction — so a value-inclusive
          hash would work only for the subset of runs whose state happens to
          be flat and scalar, i.e. it would fail by VALUE rather than by type,
          the worst failure mode for a value type.
        * They MUST NOT be hashed. A checkpoint's state is the biggest payload
          in the framework (a full transcript plus tool results), and it is
          hashed at the durable seam on every save. This hash is O(1) in state
          size and measurably so: 0.22 µs for a 1-key state and 0.20 µs for a
          100_000-key one (the gap is measurement noise — the payload is never
          read), against 3.4 µs / 25 ms for `stable_hash` over the same two, a
          ~114_000× gap on the large one.

        Excluding them is sound rather than a workaround: ``__eq__`` still
        compares every field, and the hash invariant only requires that EQUAL
        objects hash equally — never that unequal ones differ. Two snapshots of
        the same ``(run_id, version)`` that differ in state collide into one
        bucket and ``__eq__`` separates them there; that is what a bucket is
        for. (If you need a key that changes when the state changes, you want a
        CONTENT hash — `stable_hash` — not ``__hash__``.)

        Read-mutability of ``state`` — the other half named above — is now
        closed by ``__post_init__``, and NOT the way this docstring originally
        anticipated. It is not a ``MappingProxyType``: a checkpoint is
        serialised to durable storage (`json.dumps(cp.state)` into a JSONB
        column in the Postgres adapter, passed through verbatim by
        `StoreBackedCheckpointStore._to_dict`) and a mappingproxy is neither
        JSON-serialisable nor picklable, so that mechanism really would have
        traded a mutability bug for a broken persistence path. ``FrozenDict``
        is a ``dict`` subclass instead, which is what let the two halves land
        without a migration. Nothing about this hash changed.
        """
        return hash((self.run_id, self.version))


@runtime_checkable
class CheckpointPort(Protocol):
    """The durable-run-state seam. Distinct from `StorePort` (a generic KV) because the *thing*
    being persisted — a versioned, status-tagged snapshot — has its own access pattern: latest,
    at-version (time-travel), list-versions, and delete-all-for-run. A single producer is the
    authority over a `run_id`; two producers sharing a `run_id` would collide on `version`."""

    async def save(self, cp: Checkpoint) -> None: ...
    async def latest(self, run_id: str) -> Checkpoint | None: ...
    async def at_version(self, run_id: str, version: int) -> Checkpoint | None: ...
    async def list_versions(self, run_id: str) -> list[int]: ...
    async def delete(self, run_id: str) -> None: ...
