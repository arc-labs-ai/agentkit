"""L0 value types — the vocabulary every layer speaks. Pure stdlib dataclasses, no opinions.

These never change behaviour; they only carry data. `ChatRequest`/`ToolRequest` are the two units of
work the `Invoker` runs through a middleware chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from agentkit.kernel._frozen import deep_freeze

# Wire-level discriminator tag — a closed set typed as ``Literal``
# so consumers can exhaust the branch table with mypy.
StreamEventType = Literal["message_delta", "tool_call", "tool_result", "step", "interrupt", "final"]


class Operation(StrEnum):
    """The kind of unit of work a middleware is intercepting — match on it in a middleware
    (`if ctx.operation == Operation.MODEL_CALL: …`). A `str` enum, so `== "model_call"` also holds."""

    TOOL_CALL = "tool_call"  # a tool execution   (request is a ToolRequest; data = arguments)
    MODEL_CALL = "model_call"  # a chat / LLM call (request is a ChatRequest; data = messages)


@dataclass(frozen=True)
class Scope:
    """Tenant-isolation key. Threaded through every memory recall / cache key / meter / callback."""

    org_id: int | None = None
    domain_id: int | None = None

    def key(self) -> str:
        return f"org{self.org_id}:dom{self.domain_id}"


@dataclass(frozen=True)
class Usage:
    """Per-call resource consumption. Token counts are exact; ``cost_usd`` is a
    running APPROXIMATION — see the field note below.

    ``Budget`` accumulates a ``Usage`` for the token totals and keeps its own
    exact ``Decimal`` ledger for money. ``Budget.spent()`` is what a run
    reconciles against; summing ``usage.cost_usd`` is not.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0  # APPROXIMATE. ``__add__`` re-rounds to 6dp on every
    # addition, so this field is neither associative nor
    # identity-preserving: ``Usage(cost_usd=1.4296875) + Usage()``
    # yields 1.429688. It also rounds HALF_EVEN (Python's
    # ``round``) while ``Budget``'s ledger quantizes HALF_UP, so
    # on an exact tie the two differ by one unit in the last
    # place. Neither matters while ``pricing.cost()`` emits
    # 6-decimal values and ``Budget.spent()`` is the authority —
    # it matters a lot if you start summing THIS field and
    # expecting cents to balance. Pinned by
    # ``tests/runtime/test_money_properties.py``.
    cache_read_tokens: int = 0  # prompt-cache hits (billed cheaper) — for cache-aware costing
    cache_write_tokens: int = 0  # prompt-cache writes (billed once, sometimes a premium)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            round(self.cost_usd + other.cost_usd, 6),
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation the model requested. `id` is echoed back on the matching tool-result message.

    ``arguments`` is IMMUTABLE: the ToolCall flows into the ReAct approval
    snapshot AND into the idempotency-key hash AND into the audit trail — a
    tool impl that did ``args.pop("token")`` would have desynced all three.
    Callers that need to mutate copy first: ``dict(tc.arguments)``.

    The immutability was originally a ``MappingProxyType`` and is now a
    :class:`~agentkit.kernel._frozen.FrozenDict`. Same guarantee, without the
    stdlib limitation the proxy dragged in: a mappingproxy is not a ``dict``
    subclass, so ``json`` refused it, ``isinstance(args, dict)`` was False, and
    ``dataclasses.asdict`` — which deep-copies every leaf — blew up on a type
    with no pickle protocol. Measured on ``ToolCall("c1", "search", {...})``::

                                    MappingProxyType                       FrozenDict
        dataclasses.asdict(tc)      TypeError: cannot pickle 'mappingproxy' ok
        json.dumps(tc.arguments)    TypeError: not JSON serializable        ok
        isinstance(_, dict)         False                                   True

    The ``asdict`` line is the one that shipped broken, and it was contagious:
    a ToolCall does not have to be the argument, only reachable from it, so
    ``asdict`` of ANY dataclass holding a ToolCall raised. That is why
    ``resilience.stable_hash`` walks dataclass fields by hand instead of
    calling ``asdict`` (see the note at its ``_stable_default``) and why five
    other call sites carry a defensive ``dict(tc.arguments)`` unwrap
    (``anthropic``, ``openai_compat``, ``checkpointer.persistence``,
    ``context.tokens``, ``middlewares.memoize``), plus ``tools.function``'s
    ``dict(args)``, which is a snapshot rather than an unwrap. The count is
    spelled out because the previous "four" was wrong and nothing would have
    caught it.

    One consequence to know about: ``asdict`` rebuilds each nested container as
    ``type(obj)(...)``, so it hands back a ``FrozenDict``, not a plain one. The
    freeze survives serialisation, which is what a durable record wants — but
    ``asdict`` is therefore NOT the "give me an editable copy" escape hatch.
    ``dict(tc.arguments)`` is, and always was.

    ``deep_freeze`` is deep, so ``tc.arguments["filters"]["tags"].append(...)``
    is refused too — a shallow freeze leaves the same bug one level down, where
    decoded provider JSON actually lives. It also COPIES, so a caller that
    keeps editing the dict it passed in cannot reach the stored payload.

    Cost is O(payload) and it is NOT free — measured, min of five runs, on a
    realistic 3-key/2-level arguments dict::

        ToolCall(id, name, args)      1.21 us -> 3.97 us   (+2.8 us)
        ToolCall(id, name)  empty     1.22 us -> 1.48 us   (+0.3 us)
        copy.deepcopy(tool_call)      7.96 us -> 12.5 us
        pickle round trip             6.65 us -> 12.9 us

    Paid once per tool invocation, immediately before the tool does network or
    disk I/O, so it is ~3 us against a millisecond floor. The copy numbers grew
    because a ``FrozenDict`` deep-copies through its own ``__reduce__``/
    ``__deepcopy__`` rather than the C fast path (6.6 us vs 3.7 us for the same
    payload as a plain dict) — that is ``_frozen``'s price, and the
    checkpointer was already paying the 8 us.

    ``arguments`` being an unhashable container is also what would have cost
    every ``ToolCall`` — and every ``Message`` / ``PrefixContext`` /
    ``FrozenContext`` reachable from one — its hashability. See ``__hash__``;
    that decision is unchanged by this migration, and deliberately so."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclass — assign via object.__setattr__ to bypass the freeze.
        # ``deep_freeze`` short-circuits on an already-frozen payload, so the
        # re-freeze on every ``dataclasses.replace`` / unpickle costs one
        # isinstance check rather than a second walk.
        object.__setattr__(self, "arguments", deep_freeze(self.arguments))

    def __hash__(self) -> int:
        """Hash on IDENTITY — ``(id, name)`` — never on ``arguments``.

        A frozen dataclass derives ``__hash__`` from every compared field, and
        ``arguments`` is the immutable mapping above — unhashable then as a
        ``MappingProxyType``, unhashable now as a ``FrozenDict``, because
        neither refusing mutation nor being a ``dict`` subclass makes a mapping
        hashable. So the immutability guarantee cost hashability, silently and
        only at the call site that hashed one. Measured before this fix::

            hash(ToolCall("c1", "search", {"q": "hi"}))    TypeError: unhashable type: 'dict'
            hash(Message("assistant", tool_calls=(tc,)))   TypeError: unhashable type: 'dict'
            parent.merge(sibling, mode="union")            TypeError: unhashable type: 'dict'

        The third line is the one that shipped broken. ``WorkingContext.merge``
        with ``mode="union"`` deduplicates via ``set(self.messages)`` and is
        documented for a parent merging two siblings' contexts — it passed its
        tests because they use plain messages, and raised on the real
        coordinator fan-in path where every assistant turn carries tool calls.

        Excluding ``arguments`` is sound rather than a shortcut: ``__eq__``
        still compares it, and the hash invariant only requires that EQUAL
        objects hash equally — never that unequal ones differ. Two calls to the
        same tool with different arguments collide into one bucket, where
        ``__eq__`` separates them; that is what a bucket is for, and it is why
        union-mode dedup stays exact (a ``set`` consults ``__eq__``, the hash
        only chooses the bucket).

        It is also the only option that WORKS. Arguments are decoded provider
        JSON, so the values are routinely ``list`` / ``dict`` — unhashable
        themselves — and a value-inclusive hash would inherit that fragility:
        hashability would depend on what the model happened to emit. Hashing a
        stable serialisation instead (``resilience.stable_hash``) makes the
        hash O(payload), which this is not: measured 0.130 µs for a 1-key
        arguments dict and 0.128 µs for a 100_000-key one (the payload is never
        read), against 4.99 µs / 87.8 ms for ``stable_hash`` on the same two —
        a 675_000× gap on the large one, paid once per message per union merge.
        ``stable_hash`` remains the right tool where the CONTENT is the
        identity (memoize keys, idempotency keys); it is the wrong tool for
        ``__hash__``.
        """
        return hash((self.id, self.name))


@dataclass(frozen=True)
class ToolSchema:
    """JSON-schema advertisement of a tool — what the provider needs to *see* to call it.

    ``parameters`` is a ``dict`` SUBCLASS — a ``FrozenDict``, frozen deeply at
    construction. It is ``json.dumps``'d straight into provider payloads
    (``openai_compat`` sends ``schema.parameters or {}``, ``anthropic`` sends
    it as ``input_schema``) and read back through ``dataclasses.asdict``,
    neither of which survives a ``MappingProxyType`` — which is exactly why the
    freeze is a subclass and not a view. See ``kernel._frozen``.

    The freeze matters most here because a schema is SHARED, not per-call: a
    ``ToolRegistry`` hands the same ``ToolSchema`` object to every
    ``ChatRequest`` for the life of the process, so one adapter doing
    ``schema.parameters["properties"]["x"]["type"] = "number"`` to fix up one
    provider's payload would have silently rewritten what every other provider
    advertises from then on.

    This is the WORST relative cost of the four — 0.74 us before against
    12.3 us after for an 8-property schema, 16x — and the one that matters
    least, because a schema is built once and then reused forever. Every
    construction site is registration-time (``@tool`` decoration, ``as_tool``,
    ``handoff``, MCP discovery), and ``ToolRegistry.schemas()`` hands back the
    SAME cached ``ToolSchema`` objects on every turn rather than rebuilding
    them, so no per-turn path pays this. A 16x on a one-off is the right place
    to spend it; a 16x per turn would not have been.

    That mutable field is also what cost the frozen dataclass its generated
    ``__hash__``, and freezing does not give it back — a ``FrozenDict`` is
    still a mapping, so still unhashable. See ``__hash__``."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", deep_freeze(self.parameters))

    def __hash__(self) -> int:
        """Hash on the ADVERTISEMENT — ``(name, description)`` — never on the
        JSON Schema body.

        A frozen dataclass derives ``__hash__`` from every compared field, and
        ``parameters`` is a ``dict`` — a ``FrozenDict`` now, which is still a
        ``dict`` and so still unhashable — so the whole type was unhashable.
        Measured before this fix::

            hash(ToolSchema("search", "", {"type": "object"}))
            TypeError: unhashable type: 'dict'

        Nothing inside the framework hashed a ``ToolSchema`` — ``memoize``
        deliberately keys chat calls on ``sorted(t.name for t in tools)`` via
        ``stable_hash``, because there the CONTENT is the identity. The break
        was caller-facing and silent: ``set(request.tools)`` to dedupe two
        registries advertising the same tool, a ``dict[ToolSchema, Handler]``,
        an ``lru_cache`` over a schema-taking function. Each raises at the call
        site that happens to hash one, which is the worst place to find out.

        Excluding ``parameters`` is sound rather than a workaround: ``__eq__``
        still compares it, and the hash invariant only requires that EQUAL
        objects hash equally — never that unequal ones differ. Two revisions of
        one tool's schema land in the same bucket and ``__eq__`` separates them
        there; that is what a bucket is for, and a ``set`` still keeps both.

        It is also the only option that WORKS. A JSON Schema body nests
        ``dict``/``list`` by construction (``properties``, ``required``,
        ``enum``), so a content-inclusive hash would be unhashable for every
        realistic schema and hashable only for the empty one. Routing it
        through ``stable_hash`` instead would make ``__hash__`` O(schema):
        measured 0.192 µs here for both a ``{"type": "object"}`` stub and a
        20-property schema (the body is never read), against 23.4 µs for
        ``stable_hash`` on that same 20-property body — paid on every bucket
        probe, not once per cache key.

        ``description`` stays IN. It is a ``str``, so it is always hashable,
        and CPython caches a string's hash on the object — a 4000-character
        description measured the same 0.193 µs as a five-word one. It is also
        the field that varies while the name does not (a registry A/B-testing
        tool docs, a schema regenerated from a rewritten docstring), so it buys
        bucket spread for 0.063 µs: 0.193 µs with it against 0.130 µs for
        ``hash(self.name)`` alone.
        """
        return hash((self.name, self.description))


@dataclass(frozen=True)
class Message:
    """One turn in a chat transcript."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    name: str | None = None  # tool name on a result message
    tool_call_id: str | None = None  # set on a role="tool" result message
    tool_calls: tuple[ToolCall, ...] = ()  # set on an assistant turn that requested tools
    # Every field is hashable, so the dataclass-generated ``__hash__`` works —
    # but only because ``ToolCall`` defines one. Before that, a Message with
    # tool calls was unhashable while a plain one was not, which is how
    # ``merge(mode="union")`` passed its tests and failed in production. See
    # ``ToolCall.__hash__``.


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()  # calls the model requested this turn (empty = done)
    parsed: Any = None  # set by the response-coerce middleware when an output schema adapter
    # is wired on the run; stays None for unstructured calls. Kept on
    # LLMResult so the Agent's loop can lift it onto AgentResult.parsed
    # without re-running the adapter at the pattern layer.


@dataclass(frozen=True)
class StreamEvent:
    """A pattern-level run event emitted by `Agent.stream`/`*.stream` — a higher-level signal than a
    `Delta`: where `Delta` is one transport increment of a single LLM response, a `StreamEvent` marks a
    step in the *run* (a token `message_delta`, a `tool_call`/`tool_result`, an `interrupt`, the `final`
    result). `type` discriminates the payload."""

    type: StreamEventType
    text: str = ""
    tool_call: Any = None
    tool_result: Any = None
    result: Any = None  # AgentResult on a "final" event
    usage: Usage | None = None
    partial_output: Any = None  # the IN-PROGRESS typed object, forwarded verbatim from
    # ``Delta.partial`` on ``message_delta`` events. Non-None only
    # when BOTH an output schema is declared on the Agent (``output=``)
    # AND ``output_coerce()`` sits in the chat middleware chain — the
    # middleware is what runs the tolerant ``partial_parse``. Stays
    # ``None`` for every unstructured run, so a consumer reading only
    # ``text`` is unaffected.
    #
    # NOT named ``partial``: ``AgentResult.partial`` is a ``bool``
    # meaning "this run terminated incompletely", and the same code
    # paths emit both. ``ev.partial_output`` (a typed object) and
    # ``ev.result.partial`` (a failure flag) must not be confusable.
    #
    # CONSUMER CONTRACT — this object is built through the type's
    # bypass-init path (``model_construct`` / ``object.__new__``), so
    # REQUIRED FIELDS MAY BE UNSET. Never trust plain attribute
    # access; gate on what has actually arrived::
    #
    #     if ev.partial_output is not None:
    #         if "title" in ev.partial_output.model_fields_set:
    #             render(ev.partial_output.title)
    #
    # For dataclass / attrs shapes the equivalent guard is
    # ``getattr(obj, "title", None)`` — an unset field is genuinely
    # absent from ``__dict__``, so attribute access raises
    # ``AttributeError`` rather than returning a default.
    #
    # The value REFRESHES per event and is only stamped when the
    # partial changed (``output_coerce`` compares against the last one),
    # so consecutive ``message_delta``s may carry ``None`` even
    # mid-structured-stream. Hold the last non-None value; do not
    # treat a ``None`` as "the object went away".


@dataclass(frozen=True)
class Delta:
    """One streamed increment of an LLM response — the item the single `stream()` primitive yields.
    A `text` fragment and/or `tool_calls`; the **terminal** delta carries `usage`/`finish_reason`/
    `model`/`provider`. `assemble_deltas()` reduces a sequence back to an `LLMResult`, so a full
    chat result is just the collected stream."""

    text: str = ""
    model: str | None = None
    usage: Usage | None = None  # set on the terminal delta
    provider: str | None = None
    finish_reason: str | None = None  # set on the terminal delta
    tool_calls: tuple[ToolCall, ...] = ()  # tool calls revealed this delta (providers send them whole)
    parsed: Any = None  # set by the response-coerce middleware on a synthetic post-terminal
    # delta when an output schema adapter is wired; carries the typed
    # object so ``assemble_deltas`` can lift it onto ``LLMResult.parsed``.
    partial: Any = None  # set by the streaming response-coerce middleware on each text
    # delta when an output schema adapter is wired AND the adapter's
    # tolerant ``partial_parse`` produces a partial typed object that
    # differs from the previously-emitted partial. Distinct from
    # ``parsed`` (the strict-validated terminal result): ``partial``
    # may have unset required fields, refreshes per delta, and is a
    # per-delta concept — ``assemble_deltas`` does NOT lift it onto
    # ``LLMResult`` (only the final ``parsed`` survives assembly).


def assemble_deltas(deltas: list[Delta]) -> LLMResult:
    """Pure reducer: fold streamed `Delta`s into the final `LLMResult` (content concatenated; usage /
    finish_reason / model / provider / tool_calls / parsed taken from the deltas that carry them).

    ``Delta.partial`` is deliberately DROPPED here, and that is not an
    oversight — do not "fix" it. ``partial`` is a per-delta,
    in-progress value produced by a *tolerant* parse; ``parsed`` is the
    single strict-validated result. Lifting both onto ``LLMResult``
    would give the result type two competing typed fields where the
    tolerant one (with possibly-unset required fields) could shadow the
    strict one. This reducer only ever runs on a COMPLETE delta list,
    so by the time it is called there is no in-progress state left to
    carry: ``parsed`` is the answer.

    In-flight partials reach consumers through the streaming path
    instead — the cognitions forward ``Delta.partial`` onto
    ``StreamEvent.partial_output`` per ``message_delta``. See
    :class:`StreamEvent`.
    """
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage, finish, model, provider = Usage(), None, None, None
    parsed: Any = None
    for d in deltas:
        content += d.text
        if d.tool_calls:
            tool_calls = d.tool_calls
        if d.usage is not None:
            usage = d.usage
        if d.finish_reason is not None:
            finish = d.finish_reason
        model = d.model or model
        provider = d.provider or provider
        if d.parsed is not None:
            parsed = d.parsed
    return LLMResult(
        content=content,
        model=model,
        provider=provider,
        finish_reason=finish,
        usage=usage,
        tool_calls=tool_calls,
        parsed=parsed,
    )


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit for semantic memory / RAG (metadata carries source + scope tags)."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen in name only until this ran: ``chunk.metadata["scope"] = ...``
        # rewrote a retrieved value through a field advertised as immutable,
        # and scope tags in that dict are what tenant isolation reads.
        #
        # Found LATE, and the way it was found is the point: it was invisible to
        # ``tests/meta/test_value_type_invariants.py`` because that ratchet swept
        # ``agentkit.__all__`` and ``Chunk`` is not re-exported at top level. A
        # ratchet is only as wide as its enumeration — the sweep now walks the
        # package instead.
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))

    def __hash__(self) -> int:
        """A chunk IS its id and text. `metadata` is excluded (see `_frozen.py`): it
        carries scope tags that vary between retrievals of the same passage."""
        return hash((self.id, self.text))


# ---- the two units of work the Invoker runs through a middleware chain --------------------


@dataclass(frozen=True)
class ChatRequest:
    """Frozen value type. To rewrite a field mid-chain (fallback swaps
    the model, compaction rewrites messages), build a new instance
    via ``dataclasses.replace`` and reassign ``call.request`` — the
    ``Call`` reference is mutable; the ``ChatRequest`` it points at
    is not.

    "Frozen" now includes the three CONTAINER fields — ``messages``,
    ``tools``, ``response_format`` — not just the references to them. Before,
    ``frozen=True`` stopped at the reference and ``request.messages.append(m)``
    went straight through, which is a rewrite of the unit of work the chain is
    mid-way through executing: the retry middleware re-sends the request it
    kept, the tracing middleware snapshots ``list(call.request.messages)``
    AFTER the call, and ``memoize`` keys on the transcript — an in-place append
    desyncs all three against a request that already went out on the wire.
    Compaction shows the intended shape: it reads ``list(ctx.request.messages)``
    and REPLACES the request rather than editing the list.

    ``messages`` and ``tools`` are ``FrozenList``, ``response_format`` is a
    ``FrozenDict``; all three are still ``list``/``dict`` subclasses, so
    ``request.messages[-1].content``, ``len()``, iteration, slicing and
    ``json.dumps`` are untouched. ``None`` stays ``None`` — the optional fields
    are not materialised into empty containers, because ``tools is None``
    ("advertise nothing") and ``tools == []`` mean different things to the
    provider adapters.

    The freeze is DEEP, but a ``Message`` and a ``ToolSchema`` are dataclasses,
    not containers, so ``deep_freeze`` passes each one through as a leaf: the
    walk is O(number of messages) and flat in transcript BYTES. Measured, min
    of five::

        1 msg x 5 chars / x 5000 chars      3.24 us / 3.25 us
        10 msgs, 1 tool, response_format    1.74 us -> 6.67 us
        100 msgs                            26.4 us
        messages already frozen             0.12 us  (idempotent short-circuit)

    Once per turn, against an LLM round trip measured in hundreds of ms. The
    honest caveat: an agent loop rebuilds the request from a transcript that
    grows every turn, so this is O(turns) per turn and O(turns^2) over a run —
    ~26 us at turn 100. That is the same shape the transcript COPY the loop
    already does has, and it is why ``deep_freeze`` short-circuits: a caller
    that hands the same ``FrozenList`` back pays 0.12 us, not the walk.

    Not frozen: ``cache_hint``, which is annotated ``Any`` and is whatever the
    provider wanted. Freezing it would mean rewriting a provider object we do
    not own — the same line ``_frozen.deep_freeze`` draws at non-container
    leaves. It is passed through, never read by the framework."""

    messages: list[Message]
    model: str
    tools: list[ToolSchema] | None = None
    response_format: dict[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    cache_hint: Any = None  # provider prompt-cache hint (stable prefix)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", deep_freeze(self.messages))
        object.__setattr__(self, "tools", deep_freeze(self.tools))
        object.__setattr__(self, "response_format", deep_freeze(self.response_format))

    def __hash__(self) -> int:
        """Hash the CALL SHAPE — model, sampling settings, transcript LENGTH —
        never the transcript, the tool list or the payload dicts.

        Three of the fields are mutable containers, so the generated
        all-fields hash never ran. Measured before this fix::

            hash(ChatRequest([Message("user", "hi")], "gpt-4"))
            TypeError: unhashable type: 'list'

        Not even a payload-dependent failure like ``ToolCall``'s: ``messages``
        is a ``list``, so EVERY ChatRequest was unhashable, including the
        degenerate one. Anything a caller does with a request as a key — a
        ``dict[ChatRequest, LLMResult]`` test double, ``lru_cache`` over a
        request-taking helper, ``set()`` to spot a retried request — raised.

        ``messages`` is excluded for COST, not for hashability: ``Message`` and
        ``ToolCall`` gained ``__hash__`` in this same sweep, so
        ``tuple(self.messages)`` would hash fine. It would also make
        ``__hash__`` linear in the transcript, and a transcript is the one part
        of a request that grows without bound — measured 0.28 µs for this hash
        at every transcript size from 1 to 1000 turns, against 2.98 µs /
        27.4 µs / 273 µs for a messages-inclusive hash at 10 / 100 / 1000
        turns. An agent loop re-sends the whole transcript every turn, so a
        cache keyed on requests would pay O(turns) per probe and O(turns²) over
        a run, to re-derive what ``__eq__`` confirms exactly anyway.

        ``len(self.messages)`` is kept because it is O(1), is equal whenever
        the requests are equal, and is precisely what varies between the
        requests one run produces — successive turns of a loop differ by
        exactly one appended message, so the cheap discriminator spreads them
        across buckets that ``model`` alone would pile into one.

        ``tools`` is excluded on the same cost argument (a registry advertises
        the same list on every turn, so it discriminates nothing), and
        ``response_format`` / ``cache_hint`` because they cannot be hashed at
        all: the first is a decoded JSON schema body, the second is ``Any`` —
        provider-specific, in practice a dict. Hashing either would make
        hashability depend on what a caller happened to put there, which is
        the bug this method removes rather than a property to preserve.

        Excluding them is sound rather than a workaround: ``__eq__`` still
        compares every field, and the hash invariant only requires EQUAL
        objects to hash equally — never that unequal ones differ. Two requests
        differing only in message CONTENT collide into one bucket, where
        ``__eq__`` separates them; that is what a bucket is for, and a ``set``
        keeps both.
        """
        # ``len(self.messages)`` used to be a SHARP EDGE — a mutable list on a
        # frozen dataclass meant appending to it moved the request to a
        # different bucket, silently un-finding it in any set or dict. That
        # edge is now CLOSED at the source rather than documented: ``messages``
        # is a ``FrozenList``, so the append raises instead. The hash itself is
        # byte-for-byte unchanged — this comment is the only thing that moved,
        # and the ``len`` term stays for the reason it was chosen (O(1), equal
        # whenever the requests are equal, and precisely what varies between
        # successive turns of one loop).
        return hash((self.model, self.temperature, self.max_tokens, len(self.messages)))


@dataclass(frozen=True)
class ToolRequest:
    """The tool half of the unit of work — what the ``Invoker`` runs through
    the middleware chain before the terminal calls ``tool.run``.

    ``arguments`` is frozen deeply at construction, for the same reason as
    ``ToolCall.arguments`` one layer up but with a shorter fuse: this dict is
    read by the chain BEFORE the tool sees it (``egress_audit`` pulls the URL
    out of ``r.arguments[url_arg]``, ``memoize`` folds the whole dict into the
    idempotency key) and then handed to the tool impl as ``run(args, ctx)``. A
    tool that did ``args.pop("token")`` on the way in would have made the
    arguments that were AUTHORISED and the arguments that were EXECUTED two
    different things, with the audit trail recording the first.

    ``FrozenDict`` rather than a proxy because the same dict is
    ``json.dumps``'d into the audit record and hashed through
    ``stable_hash``. Cost measured at 1.05 us before against 4.31 us after for
    a realistic 3-key/2-level arguments dict — once per tool call, against the
    I/O the tool is about to do.

    ``tool`` is deliberately NOT frozen: it is the resolved ``ToolPort``, an
    object the registry built, and rewriting a caller's object is the line
    ``deep_freeze`` refuses to cross."""

    name: str
    arguments: dict[str, Any]
    tool: Any  # the resolved ToolPort (terminal calls tool.run)
    side_effecting: bool = False
    url_arg: str | None = None  # egress middleware checks this arg if set

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", deep_freeze(self.arguments))

    def __hash__(self) -> int:
        """Hash the ROUTING fields — ``(name, side_effecting, url_arg)`` —
        never ``arguments`` and never the resolved ``tool``.

        ``arguments`` is a ``dict`` — a ``FrozenDict`` now, which is still a
        ``dict`` and so still unhashable — so the generated all-fields hash
        never ran. Measured before this fix::

            hash(ToolRequest("search", {"q": "hi"}, tool=None))
            TypeError: unhashable type: 'dict'

        Same shape as ``ToolCall.__hash__`` one layer up, and for the same
        reason: arguments are decoded provider JSON, so their values are
        routinely nested ``list``/``dict``. A content-inclusive hash would be
        hashable only when the model happened to emit scalars — a trap, not a
        key. Routing them through ``stable_hash`` (what ``memoize`` does for
        its cache key, where the content genuinely IS the identity) would make
        ``__hash__`` O(arguments): measured 0.229 µs here for a 1-key
        arguments dict and 0.237 µs for a 100_000-key one (the payload is
        never read), against 3.28 µs / 80.0 ms for ``stable_hash`` on the same
        two.

        ``tool`` — the resolved ``ToolPort`` — is excluded for a second reason
        beyond cost: it is annotated ``Any`` and holds whatever the registry
        built, so hashing it would inherit that object's hashability. A
        ``@dataclass`` tool that defines ``__eq__`` has no ``__hash__`` at all,
        which would resurrect the exact failure this method removes, one
        indirection further away from the call site.

        ``side_effecting`` and ``url_arg`` stay in: both are cheap scalars and
        both are the fields the middleware chain routes on (the idempotency
        gate reads the first, the egress guard reads the second), so a request
        that is being treated differently by the chain also lands in a
        different bucket.

        Excluding the rest is sound rather than a workaround: ``__eq__`` still
        compares every field, and the hash invariant only requires EQUAL
        objects to hash equally — never that unequal ones differ. Two calls to
        the same tool with different arguments share a bucket, where ``__eq__``
        separates them; a ``set`` keeps both.
        """
        return hash((self.name, self.side_effecting, self.url_arg))
