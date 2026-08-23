# Adapters

Adapters are how agentkit talks to things it does not own — a model, a
database, a vector index, a tracing backend, the clock.

## The problem it solves

The moment a framework's core imports a vendor client, that vendor is in
your dependency tree forever. You inherit its transport, its retry
policy, its error taxonomy and its release cadence, and swapping it means
touching code you did not write. The version of this that actually hurts
is quieter: your test suite now needs an API key, so nobody runs it, so
the integration path rots.

agentkit's answer is a hard rule. `agentkit.kernel` defines the
**Port** protocols; `agentkit.adapters` implements them; nothing in the
core imports a provider client. Importing `agentkit.kernel` never
triggers an import of `httpx`, an SDK, or a database driver. Everything
outside is behind an opt-in extra, and every seam has an offline
implementation you can run in a unit test.

## The smallest thing that works

The same application code, two different storage backends, nothing else
changed:

```python
import asyncio
import tempfile

from agentkit import Agent
from agentkit.adapters.store import FileStore, InMemoryStore
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.types import ToolCall
from agentkit.middlewares import memoize
from agentkit.testing import FakeLLM, FakeTool, Turn, make_test_ctx


async def run_twice(store) -> int:
    """Application code. It never names a storage backend."""
    tool = FakeTool(name="lookup", responder={"expiry": "2026-08-01"})
    llm = FakeLLM.script(
        [
            Turn(tool_calls=(ToolCall("c1", "lookup", {"host": "example.com"}),)),
            Turn(content="done"),
            Turn(tool_calls=(ToolCall("c2", "lookup", {"host": "example.com"}),)),
            Turn(content="done"),
        ]
    )
    ctx = make_test_ctx(llm=llm, store=store, tool_middleware=[memoize()])
    agent = Agent("sre", model="fake-model", cognition=ReActCognition(tools=[tool]))
    await agent.run("check the cert", ctx)
    await agent.run("check the cert", ctx)
    return len(tool.calls)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        print(await run_twice(InMemoryStore()))   # 1 — cached
        print(await run_twice(FileStore(tmp)))    # 1 — cached, durably


asyncio.run(main())
```

## The pattern, before the catalogue

### A Port is a seam, not a feature

The distinction is the whole design, and it is easy to get wrong:

- A **Port** is an *external system agentkit cannot implement itself* —
  a model, a durable store, a vector index, web search, the network,
  the wall clock, a tracing backend.
- Idempotency, audit, caching, quota, retry, checkpointing are **not**
  ports. They are middlewares and meters *backed by* a port. There is
  no `CachePort`; there is `memoize()` riding on `StorePort`.

This is why `agentkit.kernel.ports` is small and stable. A signature
change there is a deliberate, reviewed event, because every adapter in
the ecosystem implements it.

### Ports are structural, not inherited

Every port is a `runtime_checkable` `Protocol`. Your adapter does not
subclass anything; it just has the right methods.

```python
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from agentkit.adapters.store import InMemoryStore
from agentkit.kernel.ports import StorePort


class CountingStore:
    """A StorePort that counts reads. Nothing is subclassed."""

    def __init__(self, inner: Any) -> None:
        self._inner, self.reads = inner, 0

    async def get(self, key: str) -> Any | None:
        self.reads += 1
        return await self._inner.get(key)

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        await self._inner.set(key, value, ttl=ttl)

    async def get_or_set(
        self, key: str, fn: Callable[[], Awaitable[Any]], *, ttl: int | None = None
    ) -> Any:
        return await self._inner.get_or_set(key, fn, ttl=ttl)

    async def delete(self, key: str) -> None:
        await self._inner.delete(key)

    async def append(self, key: str, value: Any) -> None:
        await self._inner.append(key, value)

    async def list(self, key: str) -> list[Any]:
        return await self._inner.list(key)


async def main() -> None:
    store = CountingStore(InMemoryStore())
    print(isinstance(store, StorePort))     # True
    await store.set("k", 1)
    print(await store.get("k"), store.reads)   # 1 1


asyncio.run(main())
```

### The ports

| Port | Module | What it abstracts |
| --- | --- | --- |
| `LLMPort` | `kernel.ports` | a model — `stream` is the primitive, `chat` ≡ collect(stream), `complete` is a shim |
| `ToolPort` | `kernel.ports` | one callable capability the loop can see |
| `StorePort` | `kernel.ports` | the single durable-KV seam |
| `VectorPort` | `kernel.ports` | a scope-isolated vector index |
| `SearchPort` | `kernel.ports` | web search, normalised to `SearchHit` |
| `FetchPort` | `kernel.ports` | a text HTTP fetch, returning `FetchResponse` |
| `ClockPort` | `kernel.ports` | wall time — `now()` / `sleep()` |
| `CheckpointPort` | `kernel.ports` | durable run state, keyed `(run_id, version)` |
| `ObserverPort` | `kernel.observation` | the product-facing observation channel |
| `TracePort` | `kernel.observation` | operational spans |
| `MetricsPort` | `kernel.metrics` | counters and histograms |
| `SamplerPort` | `kernel.sampling` | the head-sampling decision at span open |
| `ReplayStore` | `kernel.replay` | side-channel storage for full LLM payloads |

You wire them once, into `Services`, and every run in the process shares
them. Each seam has a default so `Services()` with no arguments is
fully usable: `NoopTrace`, `NoopObserver`, `NoopReplayStore`,
`NoopMetrics`, `AlwaysOnSampler`. Observability is never a required
dependency.

### One reference implementation per seam

Every family ships an in-process adapter that is not a toy — it is the
**contract** the durable backends are held to. `tests/meta/test_protocol_conformance.py`
runs one contract per Protocol against *every* implementation, so a new
adapter either passes that file or is not a conforming implementation.
That is what stops four adapters answering the same question four
slightly different ways.

## The catalogue

### `adapters.llm` — models

`CallableLLM` wraps anything you already have. Point it at a function
and you have an `LLMPort`:

```python
import asyncio

from agentkit import Agent
from agentkit.adapters.llm import CallableLLM
from agentkit.testing import make_test_ctx


def my_provider_sdk(*, system, user, model, **_kw):
    """Stand-in for whatever your provider SDK already returns."""
    return {
        "content": f"[{model}] {user.upper()}",
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }


async def main() -> None:
    ctx = make_test_ctx(llm=CallableLLM(my_provider_sdk))
    result = await Agent("shouty", model="my-model").run("hello", ctx)
    print(result.output)                                       # [my-model] HELLO
    print(result.usage.input_tokens, result.usage.output_tokens)   # 11 3


asyncio.run(main())
```

A sync `fn` runs off the loop via `asyncio.to_thread`; a coroutine is
awaited. Pass `chat_fn=` as well when your provider is tool-aware —
otherwise `chat()` flattens the transcript and reuses `complete()`.
Provider tool-calls are normalised to agentkit `ToolCall`s, and
`to_result=` / `cost_fn=` let you override the mapping and the pricing.

The HTTP provider clients live under `adapters.llm.providers`
(extra: `arc-agentkit[http]`). There is **one transport**, `httpx`, and
no per-provider SDKs. Two HTTP shapes cover the field:

- `OpenAICompatibleLLM` — OpenAI, DeepSeek, OpenRouter, Together, Groq,
  vLLM, anything else that speaks the same wire format, selected by
  `base_url`. Presets: `openai()`, `deepseek()`, `openrouter()`.
- `AnthropicLLM` — the Messages API, whose request shape genuinely
  differs (`system` is top-level, content is blocks, prompt caching is
  `cache_control` on a block). Preset: `claude()`.

These clients only do the call, the parse, and cache-aware cost.
Resilience, observability and result caching are the middleware chain's
job — see [Middlewares](middlewares.md). They map HTTP failures onto a
pre-classified `ProviderError` (rate-limit / 5xx / network → transient;
4xx → permanent, with `ProviderAuthError` for credentials) so
`retry()` and `fallback()` react correctly without their own taxonomy.

!!! note "An error can arrive inside a 200"

    Both providers can deliver a failure *inside* a successful
    response, part-way through a stream, long after headers.
    `raise_if_error_frame` turns those in-band SSE error frames into a
    raised `ProviderError`. Before it existed, the frame fell through
    every branch, the loop ended normally, and the caller received a
    **truncated answer presented as a complete one** —
    `finish_reason=None`, partial text, no exception. Nothing raised,
    so `retry()` never fired: the single most retryable provider
    failure there is (`overloaded_error`) was the one the resilience
    layer never saw.

`cache_hint` on a `ChatRequest` is where the KV-cache discipline from
[Context](context.md) lands: the Anthropic adapter turns a truthy
`cache_hint` into `cache_control: {"type": "ephemeral"}` on the system
block. The clients normalise usage onto one convention —
`input_tokens` is *fresh* input, `cache_read_tokens` are prompt-cache
hits, `cache_write_tokens` are cache creation — so
`providers.pricing.cost()` can bill the three at their different rates.

!!! warning "The price table will go stale"

    `adapters/llm/providers/pricing.py` holds best-effort public list
    prices, USD per 1M tokens as `(input, output, cache_read,
    cache_write)`. An unknown model costs `0.0` — no guessing. If
    money matters contractually, inject your own `pricing=` callable
    on the client.

### `adapters.store` — the one KV seam

`StorePort` backs what used to be four ports: cache, idempotency,
checkpoint and audit. The contract is small and precise:

```python
import asyncio

from agentkit.adapters.store import InMemoryStore


async def main() -> None:
    store = InMemoryStore()
    calls = {"n": 0}

    async def produce() -> str:
        calls["n"] += 1
        return "expensive"

    # Single-flight: 20 concurrent callers, ONE producer run.
    got = await asyncio.gather(*(store.get_or_set("shared", produce) for _ in range(20)))
    print(set(got), calls["n"])          # {'expensive'} 1

    # A producer that raises is NEVER stored. Failures are not cached.
    async def boom() -> str:
        raise RuntimeError("upstream 503")

    try:
        await store.get_or_set("flaky", boom)
    except RuntimeError as exc:
        print("raised:", exc)
    print("stored:", await store.get("flaky"))    # None

    # Append-only log — what the audit middleware rides on.
    await store.append("audit:run-1", {"step": 1})
    await store.append("audit:run-1", {"step": 2})
    print(await store.list("audit:run-1"))


asyncio.run(main())
```

| Adapter | Extra | Notes |
| --- | --- | --- |
| `InMemoryStore` | — | the reference implementation. Honours TTL (lazy expiry on read plus an amortised sweep every 256 writes, so a key nobody reads again is still reclaimed) |
| `FileStore` | — | zero-dependency durable JSON files; survives a restart. **Ignores TTL** and warns once |
| `RedisStore` | `redis` | the TTL-native backend |
| `PostgresStore` | `postgres` | `set(ttl=...)` raises `NotImplementedError` rather than silently dropping expiry |

`get_or_set` is keyed on **presence**, not truthiness, in every
adapter — a legitimately cached `None` is a hit, not a miss. That was a
real bug: a producer returning `None` re-ran on every call and
single-flight silently stopped holding.

### `adapters.vector` — retrieval

```python
import asyncio

from agentkit.adapters.vector import InMemoryVector
from agentkit.kernel.types import Chunk, Scope


async def main() -> None:
    vec = InMemoryVector()
    tenant_a = Scope(org_id=1)

    await vec.upsert(
        tenant_a,
        [Chunk(id="1", text="the TLS certificate expired", metadata={"doc": "runbook"})],
    )

    for score, chunk in await vec.search(tenant_a, "certificate", k=3):
        print(round(score, 3), chunk.id)          # 0.5 1

    # Scope is the isolation boundary, not a filter callers may forget.
    print(await vec.search(Scope(org_id=3), "certificate", k=3))    # []

    # `where` filters on chunk metadata.
    print(len(await vec.search(tenant_a, "certificate", k=3, where={"doc": "other"})))  # 0


asyncio.run(main())
```

`search` returns `(score, chunk)` pairs rather than bare chunks so a
caller can threshold: RAG ignores the score, the semantic-cache
middleware uses it.

- `InMemoryVector` — TF-cosine over tokenised text. Offline, deterministic.
- `PgVectorStore` (`postgres`) — the durable counterpart, same
  `upsert`/`search` contract and the *same* dependency-free
  feature-hashed embedding, so RAG behaves identically offline and in
  production. Ranks by cosine distance (`<=>`) and returns
  `score = 1 - distance`. Call `await init()` once to create the
  extension, table and index. Swap in a real embedding model by
  overriding `_embed`; the schema and queries are unchanged.

### `adapters.checkpoint` — durable run state

```python
import asyncio
import dataclasses
import json

from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.kernel.ports import Checkpoint, CheckpointStatus


async def main() -> None:
    store = InMemoryCheckpointStore()
    await store.save(Checkpoint("run-1", 1, {"turn": 1}, 1.7e9, CheckpointStatus.RUNNING))
    await store.save(Checkpoint("run-1", 2, {"turn": 2}, 1.7e9, CheckpointStatus.SUSPENDED))

    latest = await store.latest("run-1")
    print(latest.version, latest.status, latest.status.is_terminal())   # 2 suspended False
    print(await store.list_versions("run-1"))                          # [1, 2]

    # `state` is genuinely frozen — not just the field reference.
    try:
        latest.state["turn"] = 99
    except TypeError as exc:
        print(str(exc).split(".")[0])

    # ...and still serialises like the dict it is.
    print(json.dumps(latest.state), isinstance(latest.state, dict))    # {"turn": 2} True
    print(dataclasses.asdict(latest)["state"])                        # {'turn': 2}


asyncio.run(main())
```

That last part is the design decision worth understanding, because it
constrains every checkpoint adapter. `Checkpoint` is
`@dataclass(frozen=True)`, and `frozen` stopped at the field reference:
`cp.state = {}` raised while `cp.state["turn"] = 99` silently rewrote a
run's snapshot *after the row was committed*. The in-memory record and
the durable row then disagreed, and nothing said so.

The obvious fix — `MappingProxyType` — is the wrong one here. Measured
against the four things these payloads actually have to do:

| | `json.dumps` | `dataclasses.asdict` | `deepcopy` | `pickle` | `isinstance(_, dict)` |
| --- | --- | --- | --- | --- | --- |
| `MappingProxyType` | TypeError | TypeError | TypeError | TypeError | False |
| `FrozenDict` | ok | ok | ok | ok | True |

`Checkpoint.state` is `json.dumps`'d into a JSONB column and
`AgentResult` round-trips through `dataclasses.asdict`, so a proxy would
have traded a mutability bug for a data-path outage. `FrozenDict`
(`agentkit.kernel._frozen`) is a `dict` **subclass**, so every consumer
keeps working — serialisers, `isinstance` checks, equality against plain
dicts — while refusing mutation. The freeze runs on *every*
construction, including `PostgresCheckpointStore._row_to_checkpoint`, so
a checkpoint comes back frozen on the way **out** of storage as well as
in. A record frozen on save and mutable on resume is half fixed, and
resume is exactly where a caller reaches for `cp.state.pop("pending")`.

`CheckpointStatus` is the coarse durability gate auto-resume keys off:
`RUNNING` (engine in motion), `SUSPENDED` (waiting on a human), `DONE` /
`FAILED` (terminal, and filtered by `resume()` so a "resume if any
checkpoint exists" wiring cannot re-run a finished job). A coordinator
persisting a human-gate wait **must** use `SUSPENDED`, not `RUNNING`.

- `InMemoryCheckpointStore` — dict-backed, the contract every durable
  backend matches.
- `PostgresCheckpointStore` (`postgres`) — `PRIMARY KEY (run_id,
  version)`, which is also what `Checkpoint.__hash__` hashes.

### `adapters.observer` — the observation channel

Terminal sinks and cadence wrappers, composable in any order:

```python
import asyncio

from agentkit.adapters.observer import CollectingObserver, Hooks, PolicyObserver
from agentkit.testing import make_test_ctx


async def main() -> None:
    sink = CollectingObserver()
    seen: list[str] = []
    observer = Hooks(inner=PolicyObserver.result_only(sink)).on(
        "*", lambda obs: seen.append(obs.kind)
    )

    ctx = make_test_ctx(observer=observer)
    await ctx.emit("step", "planning", agent="analyst")
    await ctx.emit("result", "done", payload={"ok": True}, agent="analyst")
    await observer.close()

    print(seen)                              # ['step', 'result']
    print([o.kind for o in sink.items])      # ['result']


asyncio.run(main())
```

- `CollectingObserver` — every observation into `.items`, no
  backpressure. Tests and small in-process consumers.
- `QueueObserver(maxsize=...)` — bounded and non-blocking, with a
  never-drop-results rule; consume it with `async for obs in
  observer.stream()`.
- `PolicyObserver(inner, allow=...)` — a kind filter. Constructors:
  `.everything()`, `.summaries()`, `.result_only()`.
- `RollupObserver` — buffers and emits periodic summaries.
- `Hooks(inner=...)` — `on(stage, handler)` lifecycle subscription,
  where `stage` is an observation `kind` or `"*"`. Sync or async
  handlers, and an exception in a handler is swallowed: a hook must
  never destabilise the run it observes.

[Observability](observability.md) covers the channel itself — kinds,
cadence policy, and how observations relate to spans. What follows is
just the adapter surface.

`result` / `error` (the `CRITICAL_KINDS`) and the human-gate
`interrupt` are **always forwarded** by both cadence wrappers. A
cadence policy may reorder or delay, but it may not silence a result or
a human gate.

`close()` is part of `ObserverPort` for a reason: buffering observers
need a shutdown hook to flush their tail. `Hooks` chains in *front* of
another observer, so when it lacked `close()` the whole stack below it
lost its shutdown hook and a `Hooks → PolicyObserver → RollupObserver`
stack silently dropped the trailing summary at process exit. Unlike
`emit`, a failing inner `close` is **not** swallowed — hiding a flush
failure would lose the buffered tail without a trace.

### `adapters.observability` — the OTel bridge

Distinct from `adapters.observer`: that one is the product-facing
observation channel, this one is operational tracing and metrics.
Pure OTel SDK, no vendor dependency, behind `arc-agentkit[observability]`.

- `otel_tracer()` — a `TracePort`. Pass nothing in production and it
  pulls the global tracer configured by the standard
  `OTEL_SERVICE_NAME` / `OTEL_EXPORTER_OTLP_ENDPOINT` env vars; pass an
  explicit `tracer_provider` in tests.
- `otel_meter()` — a `MetricsPort`.
- `otel_exporter_otlp_http()` / `otel_metrics_exporter_otlp_http()` —
  one-line exporter setup.
- `otel_sampler(ratio)` — head sampling once traffic outpaces your
  backend.

See [Observability](observability.md) for the mental model and the
[OpenTelemetry recipe](../recipes/otel-tracing-and-metrics.md) for a
full wiring.

### `adapters.replay` — payloads out of the span

Spans carry the index; the replay store carries the bulk, so attribute
cardinality stays under the OTel SDK budget. `FileReplayStore` writes
each `ReplayRecord` as one JSON file under `<root>/<span_id>.json`,
atomically (temp file plus `os.replace`, so no reader ever sees a torn
file).

`ReplayStore.put` must never raise into a run — but best-effort was
never meant to be invisible. Every failure path logs (once per failure
class, so a full disk cannot flood the log) and bumps a counter. Alert
on `dropped_writes` / `failed_reads` if replay data matters; an ordinary
cache miss is neither logged above `DEBUG` nor counted.

Configuration — the `AGENTKIT_REPLAY_DIR` env var and the default
directory — is in [API › adapters](../api-reference/adapters.md).

## The model registry

Two problems wearing the same hat, so they share one table.

**The last mile.** `claude()` / `openai()` / `deepseek()` /
`openrouter()` all require an explicit `api_key=`, and the package read
no provider environment variable anywhere. Every application therefore
wrote the same bootstrap — read the key, pick the provider, handle the
optional-extra `ImportError`, degrade without failing silently. In one
production codebase that was ~710 lines and the only domain-free code in
its engine.

**The silent capability failure.** A model is a string.
`claude(model="claude-sonnet-4-6")` says nothing about what it can do,
so nothing catches a role bound to a model that lacks a needed
capability. That failure is silent *and* well-formed: an agent asked to
read images, bound to a model that cannot see, returns a structurally
valid answer citing evidence it never read.

Both are answered by knowing things about a model name. A `ModelEntry`
carries the provider that serves it and the capabilities it declares; a
`ProviderEntry` carries the environment variable and a lazily-imported
factory. Resolution and capability checking read the same rows.

### Capability is three-valued, and the third value is the point

```python
from agentkit.adapters.llm import (
    Capability,
    ModelCapabilities,
    ModelEntry,
    default_registry,
    normalize_model_name,
)

reg = default_registry()

print(reg.capabilities("claude-sonnet-4-6").tools)     # yes   (Capability.YES)
print(reg.capabilities("deepseek-reasoner").tools)     # no    (Capability.NO)
print(reg.capabilities("acme-llm-v3").tools)           # unknown
print(reg.capabilities("acme-llm-v3").all_unknown)     # True

# A dated release id from a provider response resolves to the family row.
print(normalize_model_name("anthropic/claude-sonnet-4-6-20250101"))
print(reg.capabilities("claude-sonnet-4-6-20250101").context_window)   # 200000

reg.register_model(
    ModelEntry(
        name="acme-llm-v3",
        provider="openai",
        capabilities=ModelCapabilities(
            tools=Capability.YES, streaming=Capability.YES, context_window=32_000
        ),
    )
)
print(reg.capabilities("acme-llm-v3").tools)           # yes
```

`Capability` is a three-valued `StrEnum` — `YES`, `NO`, `UNKNOWN` —
because two-state is what causes the bug:

- A `bool` forces every unrecognised model into "has it" (silent wrong
  answers) or "doesn't" (refuses every self-hosted model on day one).
- A `bool | None` is *worse* than either, because `if caps.vision:`
  reads `None` as absent and the type checker will not stop you.

A three-valued enum makes the third case impossible to ignore at the
read site. **Nothing is guessed**: an unregistered model reports every
capability as `UNKNOWN`, never `False` and never `True`. What to *do*
about `UNKNOWN` is the caller's policy.

The declarable capabilities are `tools`, `structured_output`,
`native_json_schema`, `streaming`, `vision`. `context_window` is
deliberately not among them — it is an `int`, not a tri-state, and is
requested as `min_context_window=`. Two distinctions worth reading
twice:

- `structured_output` means "can be steered to emit parseable JSON at
  all", via the native mode *or* the prompt-injection fallback.
  `native_json_schema` is the strict server-side
  `response_format={"type": "json_schema"}` mode specifically.
- `vision` is declared but not yet checkable end to end —
  `Message.content` is a `str` and both provider adapters map it to
  text only. It is there so the declaration is ready when that path
  lands, and so an application routing images outside the framework can
  still consult it.

`ModelCapabilities.get(name)` raises on an unrecognised *name*: a typo
in `requires=` is a programming error, not something that should degrade
into `UNKNOWN` and silently never check anything.

### Refusing a mismatch before any spend

```python
import warnings

from agentkit.adapters.llm import CapabilityMismatch, default_registry

reg = default_registry()

# A declared NO is fatal, always.
try:
    reg.check("deepseek-reasoner", ("tools",), subject="sre-agent")
except CapabilityMismatch as exc:
    print(str(exc).splitlines()[0])

# UNKNOWN is policy. Default: warn and continue.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    reg.check("acme-llm-v3", ("tools",), subject="sre-agent")
    print(caught[0].category.__name__)          # UserWarning

# A service that pins its models makes "we don't know" a deployment-time stop.
try:
    default_registry().check("acme-llm-v3", ("tools",), on_unknown="refuse")
except CapabilityMismatch:
    print("refused")

# A DERIVED requirement stays quiet on UNKNOWN.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    default_registry().check("acme-llm-v3", derived=("tools",))
    print(len(caught))                          # 0
```

`deepseek-reasoner` is the canonical example: it does not accept tools,
that is real and checkable today, and without a declaration you bind it
to a `ReActCognition` and the tools are simply ignored.

`requires=` is what the caller *declared*; `derived=` is what the
framework *inferred* from wiring (a tool-holding cognition implies
`tools`). Both raise on a declared `NO` — that is unambiguous and no
policy makes it acceptable — but only `requires` participates in the
`on_unknown` policy. The asymmetry is deliberate: nagging about a
derived requirement for every unregistered model name would put a
warning on essentially every development wiring and teach people to
filter the category out, at which point the real warnings go unread too.

`on_unknown` takes `"warn"` (default — every existing wiring keeps
working), `"refuse"` (the right setting for a production service that
pins its models), or `"allow"` (verified out of band).

### Resolving a provider from the environment

```python
from agentkit.adapters.llm import ProviderNotConfigured, default_registry

reg = default_registry()

print(reg.provider_for("claude-sonnet-4-6"))         # anthropic
print(reg.provider_for("gpt-4o-mini"))               # openai
print(reg.provider_for("meta-llama/llama-4"))        # openrouter
print(reg.provider_for("anthropic/claude-sonnet-9")) # anthropic
print(reg.provider_for("acme-llm-v3"))               # None

# Credentials come from the environment; `env` is injectable so a test
# never touches os.environ.
try:
    reg.resolve("claude-sonnet-4-6", env={})
except ProviderNotConfigured as exc:
    print(exc)

# Your own naming convention, without forking a prefix table.
reg.register_rule(lambda name: "openai" if name.startswith("acme-") else None)
print(reg.provider_for("acme-llm-v3"))               # openai
```

Lookup order: a registered `ModelEntry` always wins over any rule, and
rules run most-recently-registered first — each rule sees *every*
normalised candidate before the next rule is consulted. That ordering
is load-bearing. With candidates as the outer loop,
`provider_for("anthropic/claude-sonnet-9")` returned `openrouter`,
because the coarse "has a slash" rule matched the raw name before the
family rule was ever offered the bare `claude-sonnet-9`.

`normalize_model_name` produces the candidates, most specific first, by
stripping an OpenRouter-style `provider/` prefix and a trailing dated
release id (`claude-haiku-4-5-20251001`, `gpt-4o-mini-2024-07-18` — both
shapes, one pattern). Providers echo the dated id on every response, so
without this a `LLMResult.model` never matched a table row.

Four things this module commits to, each load-bearing:

1. **A missing extra degrades loudly.** No `httpx`, and you get
   `MissingProviderExtra` naming the extra. It is never absorbed into
   the fallback — a broken install must not masquerade as a missing
   credential, and a fake serving production traffic is the worst
   outcome in this problem space.
2. **Fallback is explicit.** `resolve()` with no credential raises
   unless you passed `fallback="fake"`. This differs from "warn and
   downgrade" on purpose: in a server process a `UserWarning` goes to a
   log nobody reads, and the result is fabricated completions served as
   real ones. When it *is* requested, the downgrade warns exactly once
   per registry.
3. **A credential never appears in a message.** Not in an error, not in
   a warning, not in a `repr`. `ProviderNotConfigured` names the
   variables *checked*, never their values — not even masked, because a
   masked credential is a leak's first half. Pinned by test.
4. **Empty string is absent.** A shell exporting `OPENAI_API_KEY=` is a
   misconfiguration; passing `""` to a provider produces a confusing
   401 far from the cause.

The module-level conveniences — `resolve_llm`, `model_capabilities`,
`require_capabilities`, `register_model`, `register_provider`,
`register_rule` — act on the process-wide `registry()`, built on first
use and shared, so an application's startup registrations are visible to
every later `Agent` construction. `default_registry()` returns a fresh,
independent instance; take one of those in a test rather than mutating
the shared one.

!!! warning "The built-in table will go stale"

    It is a convenience default, like the price table beside it. An
    application pins its own truth by registering rows over it —
    `register_model` replaces by name. The built-in *rules* only route
    a name to a provider when no row matches; they never invent a
    capability. A `gpt-` model nobody has heard of resolves to the
    OpenAI provider (correct) and reports every capability as
    `UNKNOWN` (also correct).

## Writing your own adapter

1. **Pick the port.** Read its Protocol in `agentkit/kernel/ports.py`
   (or `observation.py` / `metrics.py` / `sampling.py` / `replay.py`).
   If the thing you are wrapping is a *feature* rather than an external
   system, you probably want a middleware instead — see
   [Middlewares](middlewares.md).
2. **Implement the methods.** No base class. Protocols are structural;
   `isinstance(mine, StorePort)` is `True` once the methods exist.
3. **Read the reference implementation.** `InMemoryStore`,
   `InMemoryVector`, `InMemoryCheckpointStore` and `NoopObserver` are
   the contract, not toys. Behavioural details like presence-vs-truthiness
   in `get_or_set` and never-storing-a-raised-producer are part of the
   port, not of that adapter.
4. **Run the conformance suite.** `tests/meta/test_protocol_conformance.py`
   holds one contract per Protocol, parametrised over every
   implementation. Add yours to the parameter list.
5. **Keep the dependency optional.** If it needs a client library,
   import it lazily inside `__init__` or the first call, add an extra
   in `pyproject.toml`, and raise something that names the extra. Never
   import it at module scope in a package the core touches.
6. **Do not put policy in it.** Retry, tracing, caching, budgets and
   quotas are the middleware chain's job. An adapter that retries
   internally makes the framework's retry policy unobservable and
   double-retries under load.

## What bites people

!!! warning "`memoize()` on a chat chain: a typed `parsed` is not cached durably"

    Chat memoization used to store an `LLMResult` **object**.
    `InMemoryStore` holds objects, so it worked; every durable adapter
    serialises with `json.dumps`, so the same wiring raised
    `TypeError: Object of type LLMResult is not JSON serializable` on
    the first cache write against `FileStore`, `RedisStore` or
    `PostgresStore` — i.e. it failed for everyone who wired a cache
    that outlives the process. A chat result is now stored
    JSON-shaped and rebuilt on the way out (`tool_calls` come back as
    real `ToolCall`s with frozen `arguments`; all five `Usage` fields
    survive), so every store adapter caches a chat turn.

    One field cannot follow: `LLMResult.parsed`, the typed object
    `output_coerce()` produces. Rebuilding a Pydantic model or a
    dataclass from JSON would mean importing a class named in a cache
    entry. So the rule is **carry it or refuse the entry, never
    downgrade it**: a JSON-native `parsed` (dict/list/str/int/float/
    bool/`None`) round-trips, and anything else makes that call
    uncacheable on a serialising store — the call runs, the caller
    gets the complete typed result, nothing is stored, and a
    `UserWarning` plus `call.meta["cache_stored"] = False` say so.
    On `InMemoryStore` the typed object survives a hit as before.
    Storing a `parsed=None` copy instead would mean a miss returning
    `Plan(...)` and a hit returning `None` — a wrong answer beats a
    missed cache hit only in a benchmark.

    Tool-result memoization is unaffected on every adapter — a tool
    result is ordinary JSON and always was.

!!! warning "TTL is not uniform across store adapters"

    `InMemoryStore` honours it. `RedisStore` honours it natively.
    `FileStore` **ignores** it and warns once — an `idempotent()` key
    that never expires dedupes a legitimate retry of the same
    operation forever. `PostgresStore` raises `NotImplementedError`
    rather than accept a kwarg it cannot honour. Pick the backend for
    the key space, not the other way round.

!!! warning "Importing `adapters.llm.providers` requires the `http` extra"

    It raises `ImportError` at module load without `httpx`. This is
    why `ProviderEntry.factory` is a dotted `"module:attr"` string
    rather than a callable — registering a provider must cost no
    import, or the registry itself would become unimportable on a
    zero-dependency install, which is the opposite of what it is for.
    Importing `agentkit.adapters.llm` (the registry) is always safe.

!!! warning "An adapter that retries is an adapter you cannot observe"

    The provider clients deliberately do only call + parse + cost.
    Wrap them with `retry()`, `tracing()`, `meter()`, `fallback()`
    from the middleware chain instead. If your custom adapter hides a
    retry loop, the framework's own retry sits on top of it and your
    backoff multiplies.

!!! warning "`InMemoryVector` is not a toy, but it is not an embedding model"

    It ranks by TF-cosine over tokenised text, and `PgVectorStore`
    uses the same feature-hashed embedding so the two agree. That is
    the point — RAG behaves identically offline and in production. It
    is *not* semantic similarity. Override `_embed` with a real model
    before you conclude retrieval quality is bad.

## Related

- [Kernel](kernel.md) — the Port protocols and the value types adapters
  speak.
- [Runtime](runtime.md) — `Services`, where adapters get wired.
- [Middlewares](middlewares.md) — the policies that ride on top of a
  port.
- [Testing](testing.md) — the offline doubles for every port.
- [Observability](observability.md) — the observer / trace / metrics /
  replay seams in depth.
- [Memory](memory.md) — what `VectorPort` is usually holding.
- [Context](context.md) — where `cache_hint` comes from and why it
  matters.
- [Pick a provider from config](../recipes/provider-from-env.md) — the
  model registry end to end.
- [API › adapters](../api-reference/adapters.md) — the generated
  reference.
