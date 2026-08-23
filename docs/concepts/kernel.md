# Kernel

The kernel is agentkit's vocabulary: the small set of data shapes every
other part of the framework passes around, the interfaces to the outside
world, and the one way a cross-cutting concern wraps a call.

It contains no provider client, no policy, and no loop. Nothing in it
decides *how* to use a model — it only says what a message, a request, a
result, a cost and a tool call **are**.

## The problem it solves

Frameworks rot from the middle. A "core" package that imports a provider
SDK to define its result type has just made that SDK's shape load-bearing
for everything above it, and swapping it later means touching every
layer. The same happens with policy: once the core owns a retry loop,
every consumer inherits the core's opinion about retries.

agentkit's rule is narrow and enforced: `agentkit.kernel` imports no
provider, holds no policy, and grows no loop. That is what makes the
other packages substitutable rather than merely configurable. A tracing
adapter, a Postgres checkpoint store and a fake LLM all satisfy kernel
protocols, so a run wired with one is the same run wired with another.

The second thing it buys is a single vocabulary. There is one `Usage`
type, not one per layer; one `Message`; one `Scope`. When the meter, the
memoizer, the audit record and the provider adapter all talk about the
same `Usage` object, a number that appears in a trace is the same number
that appears on the invoice.

## The smallest example

Everything below runs offline — `FakeLLM` and `make_test_ctx` come from
`agentkit.testing` and need no API key.

```python
import asyncio

from agentkit import ChatRequest, Message
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM("42"))
    request = ChatRequest(
        messages=[Message("user", "what is 6 * 7?")],
        model="fake-model",
    )
    result = await ctx.invoker.chat(request, ctx)
    print(result.content)             # 42
    print(result.usage.total_tokens)  # 15
    print(result.provider)            # fake


asyncio.run(main())
```

`ChatRequest`, `Message`, `LLMResult` and `Usage` are kernel types. The
`Invoker` that ran the request is runtime; the `FakeLLM` behind it is an
adapter satisfying `LLMPort`. The kernel supplied the nouns and the
contract, and nothing else.

## How it works — four things, and nothing else

1. **Value types.** Frozen dataclasses for the shapes that cross a
   boundary. They carry data; they never do anything.
2. **Ports.** `Protocol`s describing an external system the framework
   cannot implement itself — a model, a tool, a vector index, a KV store,
   the web, the wall clock, durable storage.
3. **The middleware contract.** One `Call` envelope, one `Middleware`
   shape, one `chain()` function. Every cross-cutting concern in agentkit
   is written against exactly this.
4. **Primitives.** Cooperative cancellation, bounded fan-out,
   classify/backoff/circuit-breaker, and the reactive stream operators.

If you can predict which of those four a new idea belongs to, you can
predict where agentkit will put it.

## Value types

| Type | What it carries |
|---|---|
| `Scope` | The tenant key (`org_id`, `domain_id`). Threaded through every cache key, memory recall, meter and callback. |
| `Usage` | One call's consumption — input / output / cache-read / cache-write tokens, plus an approximate `cost_usd`. |
| `Message` | One turn of a transcript: `role`, `content`, and `tool_calls` on an assistant turn. |
| `ToolCall` | A tool invocation the model asked for: `id`, `name`, `arguments`. |
| `ToolSchema` | The JSON-Schema advertisement a provider needs in order to *offer* a tool. |
| `ChatRequest` | A unit of work: messages, model, tools, sampling settings. |
| `ToolRequest` | The other unit of work: tool name, arguments, the resolved tool, and the egress/idempotency routing flags. |
| `LLMResult` | An assembled model response — content, usage, `tool_calls`, and `parsed` when an output schema is wired. |
| `Delta` | One transport increment of a single model response. |
| `StreamEvent` | One step of a *run*: a token, a tool call, an interrupt, the final result. |
| `Operation` | `MODEL_CALL` or `TOOL_CALL` — the discriminator a middleware matches on. |

`Budget` is deliberately not here: a ceiling is a policy, so it lives in
[Runtime](runtime.md).

### `Usage.cost_usd` is an approximation; `Budget` keeps the ledger

`Usage.__add__` re-rounds `cost_usd` to six decimal places on every
addition, so the field is neither associative nor identity-preserving —
`Usage(cost_usd=1.4296875) + Usage()` gives `1.429688`. It also rounds
half-to-even while the budget's ledger quantizes half-up, so on an exact
tie the two differ in the last place.

None of that matters while you read it for display. It matters a lot if
you sum this field and expect cents to balance. `Budget.spent()` is a
`Decimal` and is the authority. See
[Runtime › money](runtime.md#money-is-decimal-and-the-mirror-is-not).

## Ports: the seams to the outside world

A port is an **external system agentkit cannot implement itself**.
Features are not ports — idempotency, audit, caching and quota are
middlewares and meters built over `StorePort`, not seams of their own.
That distinction is what keeps the list short:

| Port | The system behind it | Key methods |
|---|---|---|
| `LLMPort` | a chat model | `stream()`, `chat()`, `complete()` |
| `ToolPort` | anything the agent can execute | `run(args, ctx)` |
| `VectorPort` | a vector index | `upsert()`, `search()` |
| `StorePort` | a generic KV | `get`/`set`/`get_or_set`/`delete`/`append`/`list` |
| `SearchPort` | web search | `search()` |
| `FetchPort` | the network | `fetch()` |
| `ClockPort` | the wall clock | `now()`, `sleep()` |
| `CheckpointPort` | durable run state | `save`/`latest`/`at_version`/`list_versions`/`delete` |
| `TracePort` | operational tracing | `span()`, `current_span_id()`, `add_event_to_current_span()` |
| `ObserverPort` | the product-facing observation stream | `emit()`, `close()` |
| `MetricsPort` | counters and histograms | `add_counter()`, `record_histogram()` |

They are `Protocol`s rather than base classes because the implementations
have nothing in common at the Python level and should not be forced to
inherit from us. Structural typing means an object you already have —
your own HTTP client wrapper, a test double, a vendor SDK behind three
lines of glue — satisfies the port by having the right methods. Most are
`@runtime_checkable`, so `isinstance(x, LLMPort)` works for a wiring
assertion.

Two of them are worth calling out.

**`LLMPort.stream()` is the primitive.** `chat()` is defined as
"collect the stream", not the other way round. Everything in the
framework that consumes a model consumes a stream of `Delta`s and
reduces it, which is why a cache hit and a live call are indistinguishable
to the layers above.

**`CheckpointPort` is separate from `StorePort` on purpose.** A durable
run snapshot has its own access pattern — latest, at-version (time
travel), list-versions, delete-all-for-run — and a versioned,
status-tagged record does not fit a bare KV. One producer is the
authority over a `run_id`; two producers sharing one would collide on
`version`. See [Capabilities › Checkpointer](capabilities.md#checkpointer).

`ClockPort` exists so retry and backoff loops are testable: use it
instead of `time.time()` / `asyncio.sleep` anywhere inside a middleware.

There is one more structural protocol that is not a port: `Ctx`, in
`agentkit.kernel.protocols`. It is the slice of a `RunContext` that
capabilities and agents actually read, and it exists so those layers can
type-annotate the context without importing the runtime (which sits
above them). `RunContext` satisfies it by duck typing; so does a test
stub.

## The middleware contract

Both units of work — a chat turn and a tool execution — are a `Call`:

```python
from agentkit import Call, ChatRequest, Message
from agentkit.testing import make_test_ctx

call = Call(
    kind="chat",                                     # "chat" | "tool"
    request=ChatRequest(messages=[Message("user", "hi")], model="fake-model"),
    ctx=make_test_ctx(),                             # the RunContext
    meta={},                                         # per-call signals between middlewares
)
print(call.kind, type(call.request).__name__)        # chat ChatRequest
```

`Call.request` is a **mutable slot holding an immutable value**: you
cannot edit the `ChatRequest`, but you can point `call.request` at a new
one. That is how compaction rewrites a transcript and how fallback swaps
a model.

A handler yields the operation's items: `Delta`s for a chat, exactly one
item for a tool. A middleware is an async generator wrapping the next
handler, so it can act before, after, around, or *instead of* the rest of
the chain. `chain(middlewares, terminal)` folds a list into one handler,
with `middlewares[0]` outermost.

That single shape is what makes retry, fallback and memoize expressible
at all: retry re-invokes `next`, fallback rewrites the request and
re-invokes, memoize may skip `next` entirely. See
[Middlewares](middlewares.md) for the two authoring styles and the
shipped set.

`collect(stream, kind)` is the reducer that turns a handler's stream back
into a result — for a chat it is `assemble_deltas`, so **a chat result is
just its collected stream**.

## Concurrency and cancellation

- `CancellationToken` — a cooperative abort signal shared down a run
  tree. A parent calls `cancel()`; agent loops call
  `raise_if_cancelled()` at safe points and get `Cancelled`. This is
  distinct from a graceful `TerminationCondition` (which returns a stop)
  and from a ceiling (`MeterExceeded`).
- `gather_bounded(coros, sem=...)` — structured fan-out over
  `asyncio.TaskGroup`, bounded by a semaphore, preserving input order.
  A failure cancels the siblings and surfaces as an `ExceptionGroup`.
- `gather_best_effort(coros, sem=...)` — the same, except each slot is
  either a result **or** a `Failure` value. Use it when one child failing
  must not sink the batch.
- `run_agents(...)` — the fan-out agents actually use; it carves child
  contexts and per-actor budgets before dispatch.
- `run_sync(coro)` — the one sync bridge. If no loop is running it calls
  `asyncio.run`; if one is, it runs the coroutine on a fresh worker
  thread, so you get a blocking signature without "loop already running".

`contextvars` are copied into each task, so the active span and
correlation id propagate into fan-out.

!!! warning "An abort is bounded, not unconditional"
    When a fan-out aborts, it waits at most `SIBLING_CLEANUP_GRACE_S`
    (5 seconds) for already-cancelled siblings to run their `finally`
    blocks. Measured with one sibling swallowing `CancelledError` in a
    loop, an unbounded wait was still wedged at 130 seconds, and an outer
    `asyncio.wait_for` could not break it out — the await sat inside an
    `except BaseException` handler, which is a shielded region from the
    caller's point of view. Five seconds is a graceful-shutdown grace,
    not a deadline: releasing a semaphore or flushing a span is
    sub-millisecond.

`Failure` (from `agentkit.kernel.errors`) is the "error as data" type
these return: a category from `classify()`, a `source` naming the slot, a
`cause`, an optional `partial_output`, and `children` when it aggregates.
A `Failure` is a value a parent can route around; an exception is control
flow.

## Immutability: what "frozen" actually means here

Every public value type is `@dataclass(frozen=True)`. That alone stops at
the field *reference* — a frozen dataclass holding a plain `dict` still
lets you rewrite the dict. Twelve public types had exactly that hole:

```python
from agentkit import Checkpoint, CheckpointStatus

cp = Checkpoint(
    run_id="r1", version=1, state={"turn": 3},
    created_at=0.0, status=CheckpointStatus.RUNNING,
)
try:
    cp.state = {}          # always refused: the dataclass is frozen
except Exception as exc:
    print(type(exc).__name__)          # FrozenInstanceError
try:
    cp.state["turn"] = 99  # once silently rewrote a committed durable record
except TypeError:
    print("refused too, now")
```

The obvious fix is `MappingProxyType`, and it is the wrong one. Measured
against the four things these payloads have to do:

| | `json.dumps` | `dataclasses.asdict` | `deepcopy` | `pickle` | `isinstance(_, dict)` |
|---|---|---|---|---|---|
| `MappingProxyType` | TypeError | TypeError | TypeError | TypeError | `False` |
| `FrozenDict` | ok | ok | ok | ok | `True` |

So the container payloads are `dict` and `list` **subclasses** that refuse
mutation (`agentkit.kernel._frozen`). Serialisers, `isinstance` checks
and equality against plain dicts all keep working; `checkpoint.state` is
still `json.dumps`-able into a JSONB column, and an `AgentResult` still
round-trips through `dataclasses.asdict`.

```python
import copy
import json
import pickle

from agentkit import ChatRequest, Message, ToolCall

tc = ToolCall(id="c1", name="search", arguments={"q": "octopus", "filters": {"tags": ["marine"]}})

# The payload is a real dict, so everything that consumes dicts still works.
print(isinstance(tc.arguments, dict), json.dumps(tc.arguments))
print(tc == copy.deepcopy(tc), pickle.loads(pickle.dumps(tc)) == tc)

# ...but it refuses mutation, at every level.
try:
    tc.arguments["q"] = "squid"
except TypeError as exc:
    print("refused:", str(exc)[:46])
try:
    tc.arguments["filters"]["tags"].append("deep")
except TypeError:
    print("refused one level down too")

req = ChatRequest(messages=[Message("user", "hi")], model="fake-model")
try:
    req.messages.append(Message("user", "and again"))
except TypeError:
    print("a request in flight cannot be rewritten in place")
```

The freeze is **deep** and it **copies**. Deep, because a shallow freeze
leaves the same bug one level down, which is exactly where decoded
provider JSON lives. Copying, because a caller who keeps editing the dict
they passed in must not be able to reach the stored payload.

Why it matters per type:

- `ToolCall.arguments` flows into the approval snapshot, the idempotency
  key, and the audit trail. A tool doing `args.pop("token")` used to
  desync all three — the arguments that were *authorised* and the
  arguments that were *executed* became different things, with the audit
  recording the first.
- `ToolSchema.parameters` is **shared**, not per-call: a registry hands
  the same object to every request for the life of the process. One
  adapter patching it to fix up its own payload would silently rewrite
  what every other provider advertises.
- `ChatRequest.messages` is the unit of work the chain is *mid-way
  through executing*. Retry re-sends the request it kept, tracing
  snapshots the messages after the call, memoize keys on the transcript;
  an in-place append desyncs all three against a request already on the
  wire. The intended shape is compaction's: read the messages, build a
  new request with `dataclasses.replace`, reassign `call.request`.

Cost is O(payload) and paid once at construction — around 2.8 µs extra
for a realistic tool-call arguments dict, against the network call the
tool is about to make. `deep_freeze` short-circuits on an
already-frozen payload, so re-freezing on `dataclasses.replace` or
unpickle costs one `isinstance` check rather than a second walk.

### The consequence: hashing is on identity, not payload

A `FrozenDict` is still a `dict`, and dicts are unhashable — so freezing
does not hand back the generated `__hash__` a mutable payload cost. Every
affected type defines `__hash__` over an **identity subset** and leaves
`__eq__` comparing everything:

| Type | Hashed on |
|---|---|
| `ToolCall` | `(id, name)` |
| `ToolSchema` | `(name, description)` |
| `ToolRequest` | `(name, side_effecting, url_arg)` |
| `ChatRequest` | `(model, temperature, max_tokens, len(messages))` |
| `Prompt` | `(id, version, template, inputs)` |

This is sound rather than a shortcut: the hash invariant only requires
*equal* objects to hash equally, never that unequal ones differ. Two
calls to the same tool with different arguments share a bucket, where
`__eq__` separates them — which is what a bucket is for.

It is also the only option that works. Arguments and schema bodies are
decoded provider JSON, so their values are routinely nested `list`/`dict`;
a content-inclusive hash would be hashable only when the model happened
to emit scalars. Hashing a stable serialisation instead would make
`__hash__` O(payload): measured 0.130 µs for `ToolCall` at any size,
against 4.99 µs / 87.8 ms for `stable_hash` on a 1-key and a
100,000-key dict. `stable_hash` remains right where the *content* is the
identity — memoize keys, idempotency keys — and wrong for `__hash__`.

!!! note "The bug this fixed was not hypothetical"
    `WorkingContext.merge(mode="union")` deduplicates via
    `set(self.messages)`. It passed its tests, which used plain messages,
    and raised `TypeError: unhashable type: 'dict'` on the real
    coordinator fan-in path, where every assistant turn carries tool
    calls. `ChatRequest` was worse: `messages` is a `list`, so *every*
    request was unhashable, including the degenerate one.

## `Delta` vs `StreamEvent` — two levels of "streaming"

A `Delta` is one transport increment of a **single model response**. A
`StreamEvent` marks a step in the **run**: a token, a tool call, an
interrupt, the final result. `assemble_deltas()` reduces a list of
`Delta`s back to an `LLMResult`.

Both can carry an in-progress typed object when an output schema is
declared. `Delta.partial` is set by the `output_coerce()` middleware, and
the cognitions forward it verbatim onto `StreamEvent.partial_output` —
that is how an application streams a typed object through `Agent.stream`
alone. See [the recipe](../recipes/stream-typed-output.md).

`assemble_deltas` deliberately **drops** `partial`. It only ever runs on
a complete delta list, so by then there is no in-progress state left to
carry: `parsed` — the strict, validated object — is the answer. Lifting
both onto `LLMResult` would give the result type two competing typed
fields, where the tolerant one (which may have unset required fields)
could shadow the strict one.

!!! warning "A partial object may have unset required fields"
    `StreamEvent.partial_output` is built through the type's
    bypass-init path (`model_construct` / `object.__new__`), so plain
    attribute access is not safe. Gate on what has actually arrived:

    ```python
    from pydantic import BaseModel

    class Article(BaseModel):
        title: str
        body: str

    # What a tolerant partial parse hands back mid-stream:
    partial = Article.model_construct(title="Octopus cognition")
    print("title" in partial.model_fields_set)   # True  — safe to render
    print("body" in partial.model_fields_set)    # False — has not arrived
    ```

    So the guard on a `message_delta` event is
    `if ev.partial_output is not None and "title" in
    ev.partial_output.model_fields_set:` before you touch
    `ev.partial_output.title`.

    For dataclass and attrs shapes the equivalent guard is
    `getattr(obj, "title", None)` — an unset field is genuinely absent
    from `__dict__`, so attribute access raises `AttributeError` —
    see the worked example in
    [Capabilities › SchemaAdapter](capabilities.md#schemaadapter).

    The value also refreshes per event and is only stamped when the
    partial *changed*, so consecutive `message_delta`s may carry `None`
    mid-stream. Hold the last non-`None` value; do not read a `None` as
    "the object went away".

## Errors: one base, and a three-way retry verdict

Every typed error the framework raises inherits `AgentkitError`, so an
application that wants a single boundary can catch that one name:

```python
from agentkit import AgentkitError, StoreUnavailable

try:
    ...
except StoreUnavailable:
    ...          # the store could not be reached — degrade, or fail the run
except AgentkitError:
    ...          # anything else the framework raised, typed
```

`StoreUnavailable` is the one worth naming separately, because it is
infrastructure rather than logic: a `StorePort` operation could not reach
its backing store. Everything a memoize or checkpoint layer does on top
of a store has to decide what that means for the run.

!!! note "Not every error in the tree is an `AgentkitError`"
    The tool errors are the deliberate exception — `ToolArgumentError`
    subclasses `ValueError`, because the model sending a bad argument is
    a value problem and callers already catch `ValueError` around
    argument handling. `issubclass(ToolArgumentError, AgentkitError)` is
    `False`, on purpose.

`ErrorClass` is the separate question retry asks — not *what went wrong*
but *is trying again worth anything*:

| Member | Meaning | What `retry()` does |
|---|---|---|
| `transient` | a timeout, a 429, a 5xx | back off and try again |
| `permanent` | a 401, a malformed request | give up immediately |
| `unknown` | anything unrecognised | give up — guessing costs money |

`unknown` defaulting to "do not retry" is the conservative direction on
purpose: an unrecognised failure retried is a bill with no upside.
Measured — `classify(TimeoutError(...))` is `transient`,
`classify(ValueError(...))` is `unknown`.

## Two helpers worth knowing

`collect_one(stream)` reduces a one-item stream to its single value —
the tool seam's counterpart to `collect`, which assembles a chat stream's
deltas into an `LLMResult`. You need it when writing a middleware that
has to see a tool's result rather than pass the stream through.

`compose_failures(...)` aggregates child failures into one, returning
`None` when there are none. It is what a fan-out uses to answer "did any
of these fail, and what do I raise" without inventing its own shape.

## What bites people

- **`ctx.messages` inside a middleware is a copy.** Mutating it does
  nothing. Assign `ctx.request = ...` — that is the writable seam.
- **`dataclasses.asdict` gives you back frozen containers.** It rebuilds
  each nested container as `type(obj)(...)`, so a `FrozenDict` stays
  frozen through the round trip. Good for a durable record; not the
  "editable copy" escape hatch. `dict(tc.arguments)` is.
- **`ChatRequest.tools is None` and `tools == []` are different.** The
  first means "advertise nothing", the second means "here is an empty
  list"; provider adapters treat them differently, so the optional
  fields are never materialised into empty containers.
- **`cache_hint` is not frozen.** It is annotated `Any`, is whatever the
  provider wanted, and is passed through without ever being read by the
  framework. Freezing it would mean rewriting an object we do not own.
- **Never swallow `CancelledError`.** A middleware or tool that catches
  broad exceptions must re-raise it, or cooperative cancellation stops
  working for everything inside it.

## The invariants it enforces

1. **No provider dependency.** Importing `agentkit.kernel` never triggers
   an import of `httpx`, a vendor SDK, or a database driver.
2. **Values are immutable, payloads included.** Middleware constructs a
   new value and forwards; it never edits one in place.
3. **One vocabulary.** Every layer that talks about a usage or a scope
   reuses the kernel type. There is no parallel `Usage` in `runtime/` or
   `middlewares/`.
4. **Additive fields only.** A new field carries a default and changes
   nothing for a consumer that does not read it —
   `StreamEvent.partial_output` is `None` for every unstructured run.

!!! abstract "Where this fits in the four themes"
    The kernel sits *underneath* all four themes on the
    [landing page](../index.md) — it is the shared vocabulary they build
    on. **Cognition** speaks in `ChatRequest` / `LLMResult` /
    `StreamEvent`; **Control** uses `CancellationToken` and the
    concurrency primitives; **State** threads the `Message` / `Usage` /
    `Scope` value types; **Behaviour** *is* the `Call` + `Middleware` +
    `chain(...)` contract.

## Related

- [Runtime](runtime.md) — the per-request frame these values travel in.
- [Middlewares](middlewares.md) — writing against the `Call` contract.
- [Capabilities](capabilities.md) — the optional collaborators over these ports.
- [Adapters](adapters.md) — the implementations behind the ports.
- [Tools](tools.md) — what satisfies `ToolPort` in practice.
- [Testing](testing.md) — `FakeLLM`, `make_test_ctx` and the other doubles.
- [API › kernel](../api-reference/kernel.md) — the generated reference.
