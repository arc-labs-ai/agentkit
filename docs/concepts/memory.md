# Memory

Memory is what an agent reaches for before it answers.

!!! tip "Is this page for you?"

    **Reach for it when** answers need to be grounded in documents,
    past turns, or anything the model was not trained on.

    **Skip it for now if** everything the agent needs already fits
    in the prompt you hand it.

## The problem it solves

A model knows what it was trained on and what is in the current prompt.
Everything else — your refund policy, last week's incident report, what
this customer already asked — it does not have. Ask anyway and you get a
plausible answer that is not yours.

The fix is to fetch the relevant material and put it in the prompt. That
sounds like one job, and in practice it is five: a vector index for
semantic recall, a cache so you are not paying for the same lookup
twice, a tenant guard so org A never sees org B's documents, a
compaction step when the chunks are bigger than the budget, and often a
plain search tool that already exists. Wire those as five bespoke hooks
and every backend swap becomes surgery.

`agentkit.memory` is one Protocol with many backends. `MemorySource`
has two methods. Everything else in the package — vector store, files,
journal, cache, tenant guard, tool adapter, fan-out, fallback chain — is
either an implementation of it or a decorator over it, so they nest
arbitrarily and the agent is wired the same way regardless.

## The smallest thing that works

```python
import asyncio

from agentkit.adapters.vector.in_memory import InMemoryVector
from agentkit.kernel.types import Scope
from agentkit.memory import MemoryItem, VectorMemory
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM(), scope=Scope(org_id=1, domain_id=1))
    mem = VectorMemory(vector=InMemoryVector())

    await mem.write(
        [
            MemoryItem(content="Refunds are issued within 5 business days.", source="policy"),
            MemoryItem(content="Store credit never expires.", source="policy"),
        ],
        ctx=ctx,
    )

    for item in await mem.query("how long do refunds take", k=2, ctx=ctx):
        print(round(item.score, 3), item.source, "|", item.content)


asyncio.run(main())
```

```text
0.169 vector | Refunds are issued within 5 business days.
```

`InMemoryVector` is a deterministic TF-cosine stand-in — a real
embedding backend plugs into the same `VectorPort` seam. Note that
`source` on the returned item is `"vector"`, not `"policy"`: the source
label is stamped by the backend that produced the hit, not carried over
from what you wrote.

## Wiring it to an agent

```python
import asyncio

from agentkit import Agent
from agentkit.adapters.vector.in_memory import InMemoryVector
from agentkit.kernel.types import Scope
from agentkit.memory import MemoryItem, VectorMemory
from agentkit.testing import FakeLLM, make_test_ctx

memory = VectorMemory(vector=InMemoryVector())

# This fake echoes the system prefix back, so you can see what the model was sent.
llm = FakeLLM(lambda system, user, model: system)

agent = Agent(
    name="support",
    model="fake",
    prompt="Answer customer questions using the provided context.",
    memory=memory,
)


async def main() -> None:
    ctx = make_test_ctx(llm=llm, scope=Scope(org_id=1, domain_id=1))
    await memory.write(
        [MemoryItem(content="Refunds are issued within 5 business days.", source="policy")],
        ctx=ctx,
    )
    print((await agent.run("How long do refunds take?", ctx)).output)


asyncio.run(main())
```

```text
Answer customer questions using the provided context.

Relevant context:
[vector] Refunds are issued within 5 business days.
```

One keyword. What happened underneath is worth knowing, because it is
where you intervene when the defaults are wrong.

## How results reach the prompt

`Agent.memory` does not query anything by itself. When you set it and
did **not** pass an explicit `request_builder=`, the agent wraps it with
`as_grounder(memory)` and hands the result to a `RequestBuilder` as its
`grounder` — an async callable `(ctx, task) -> str`.

On the first turn the `RequestBuilder` calls that grounder with the user
task, and any non-empty text becomes a pinned system message
`"Relevant context:\n…"` in `WorkingContext.prefix` — the **cache-stable
head**, not the message tail. That placement is deliberate: the prefix is
frozen after the first turn, so the provider's KV cache stays valid for
the rest of the loop.

The half of that nobody enjoys: if you reuse one `WorkingContext` across
several `agent.run(...)` calls, turn 5 is answered with the evidence
retrieved for **turn 1's** question. Measured on a five-turn
conversation over a five-fact handbook at `k=1`, the answering fact was
in front of the model on 1 of 5 turns. Set `reground_every_turn=True` on
your own `RequestBuilder` to retrieve afresh each turn — 4 of 5 in the
same probe — and understand you are buying it with a cache invalidation
every turn: `$0.2280` against `$0.0228` for a 4,000-token prefix over 20
`claude-sonnet-4-6` turns.

Two things that catch people out. A "turn" is one `build()` call, so the
flag changes nothing inside a single ReAct tool loop — `drive()` grounds
once and then appends tool traffic to the tail (measured: 4 LLM calls, 1
grounder call). And the `memory=` shortcut on this page builds the
`RequestBuilder` for you and cannot pass the flag; to set it, build the
`RequestBuilder` yourself as below and pass `request_builder=`.

The default rendering is one `[source] content` line per item, `k=5`, no
`where` filter. To change any of that, build the grounder yourself:

```python
import asyncio

from agentkit import Agent
from agentkit.adapters.vector.in_memory import InMemoryVector
from agentkit.capabilities.request_builder import RequestBuilder
from agentkit.kernel.types import Scope
from agentkit.memory import MemoryItem, VectorMemory, as_grounder
from agentkit.prompts import Prompt
from agentkit.testing import FakeLLM, make_test_ctx

memory = VectorMemory(vector=InMemoryVector())

agent = Agent(
    name="support",
    model="fake",
    request_builder=RequestBuilder(
        prompt=Prompt(id="support", version="1", template="Answer using the context."),
        grounder=as_grounder(
            memory,
            k=3,
            where={"tier": "public"},
            format=lambda items: "\n".join(f"- {i.content} ({i.source})" for i in items),
        ),
    ),
)

llm = FakeLLM(lambda system, user, model: system)


async def main() -> None:
    ctx = make_test_ctx(llm=llm, scope=Scope(org_id=1, domain_id=1))
    await memory.write(
        [
            MemoryItem(content="Refunds take 5 business days.", source="x",
                       metadata={"tier": "public"}),
            MemoryItem(content="Internal escalation runbook.", source="x",
                       metadata={"tier": "internal"}),
        ],
        ctx=ctx,
    )
    print((await agent.run("How long do refunds take?", ctx)).output)


asyncio.run(main())
```

```text
Answer using the context.

Relevant context:
- Refunds take 5 business days. (vector)
```

The `where={"tier": "public"}` filter kept the internal runbook out. The
retrieval policy — `k`, the filter, the formatting, even which source —
is baked into the closure at wiring time; `RequestBuilder` never learns
that the text came from a vector store at all.

That is the auto-wired path. A cognition is free to do something else
entirely: `MemorySource` is just an object with a `query` method, and
nothing stops a custom cognition from querying it mid-loop, writing
results into `WorkingContext.scratchpad`, or interleaving recall with
tool calls.

!!! note "`scratchpad["memory"]`"
    `Agent`'s docstring mentions results being stamped onto
    `working_context.scratchpad["memory"]`. Nothing in the framework
    writes that key today — the grounding path above is the shipped
    mechanism. The scratchpad is still a memory *surface* in the other
    direction: `ScratchpadMemory` exposes `WorkingContext.scratchpad`
    as a queryable `MemorySource`.

## The contract

### `MemorySource`

Everything in this package is one interface with two methods: *find me
things* and *store these things*. That is the entire contract, and it is
small on purpose — a vector database, a folder of files, an MCP server
and an in-memory dict can all satisfy it, so the agent is wired the same
way no matter what is actually behind it.

```python
from agentkit.memory import MemorySource

print(sorted(MemorySource.__annotations__))
print(sorted(m for m in dir(MemorySource) if not m.startswith("_")))
```

```text
['name']
['query', 'write']
```

- `async query(query, *, k, ctx, where=None) -> list[MemoryItem]` — ask
  for the `k` items most relevant to `query`, optionally narrowed by a
  backend-specific `where` filter.
- `async write(items, *, ctx) -> None` — index items for later recall.
  Read-only backends implement this as a no-op rather than raising.
- `name` — a stable label, stamped onto every returned item's `source`
  and used for trace attribution.

It is a `Protocol`, and `@runtime_checkable`. There is no base class to
inherit and no `Agent` subclassing: a new backend is an object with
those three names.

The narrowness is the point. Every backend returns the same
`MemoryItem`, so the cognition consuming results never sees whether they
came from pgvector, a filesystem, or an HTTP search API.

!!! note "Memory and tools overlap, and answer different questions"
    A web-search tool is also a knowledge probe. The distinction is
    *who decides*: a [tool](tools.md) is what the **model** chooses to
    call mid-loop; memory is what the **cognition** fetches around an
    LLM call to ground it. `ToolMemory` bridges the two when one
    backend should serve both.

### `MemoryItem`

```python
import dataclasses

from agentkit.memory import MemoryItem

item = MemoryItem(content="Refunds take 5 days.", source="handbook", metadata={"path": "/a"})
print(item.content, "|", item.source, "|", item.score, "|", item.metadata)

try:
    item.metadata["path"] = "/b"
except TypeError as exc:
    print("frozen:", exc)

annotated = dataclasses.replace(item, metadata={**item.metadata, "chunk": 3})
print(annotated.metadata, "| still hashable:", hash(annotated) == hash(item))
```

```text
Refunds take 5 days. | handbook | None | {'path': '/a'}
frozen: this payload belongs to a frozen value and cannot be mutated in place. Build a new one instead: dataclasses.replace(obj, field={**obj.field, ...})
{'path': '/a', 'chunk': 3} | still hashable: True
```

Five fields: `content` (already-formatted text the agent reads),
`source` (which backend produced it), `id` (stable within that source,
`None` when the backend cannot supply one — see *Deduping the fan-out*
below), `score` (the backend's relevance signal, `None` when it does not
rank), `metadata` (backend-specific extras — chunk id, path, timestamp).

`id` is keyword-only, deliberately: it landed after the other four and a
positional insertion would have silently reassigned every existing
`MemoryItem(content, source, score)` call site.

Two things about it are not obvious and both were bugs first.

**`metadata` is deeply frozen at construction.** `frozen=True` on the
dataclass stopped at the field reference: `item.metadata = {}` raised
while `item.metadata["path"] = ...` went straight through. That matters
because fan-out invites exactly that habit — a `CompositeMemory` merges
items from several sources and a reranker stamps its own scores onto
them, and since backends pass `metadata` *through* to new items, one
stamp could land on an object another source still holds. `deep_freeze`
copies, so a passed-through payload is un-aliased at each hop, and it is
idempotent, so re-freezing an already-frozen payload costs one
`isinstance` check. Measured cost: 0.49 µs for empty metadata, ~7.6 µs at
344 B of JSON — against a backend call that is a network round trip.

**`__hash__` covers `(content, source, score)` and never `metadata`.**
Before that, `hash(MemoryItem(content="c", source="s"))` was a
`TypeError` — unhashable by *type*, so nothing could hash a recall result
at all, and callers wrote list scans where they wanted a `set`. Excluding
`metadata` is what makes dedup work: it is the field most likely to
differ between two records of the same passage (two chunk ids), and
`__eq__` still compares it, so two genuinely different records collide
into one bucket and are separated there.

### The shipped sources

| Source | Backed by | Ranks by | Writes |
|---|---|---|---|
| `VectorMemory` | a `VectorPort` (pgvector, in-memory) | backend similarity score | yes |
| `FileMemory` | a file tree (`InMemoryFiles` by default) | lowercase substring match count | yes |
| `JournalMemory` | a `MutationJournal` | recency (score is always `None`) | no-op |
| `ScratchpadMemory` | `WorkingContext.scratchpad` | unranked lexical KV match | yes |
| `ToolMemory` | any `Tool` | whatever the tool returns | no-op |

`FileMemory` is the read side of the same file tree the model edits
through [`FileTool`](tools.md); both wrap the same backend but answer
different questions — "what notes match this query?" versus "the model
wants to edit a note". `JournalMemory` and `ToolMemory` are read-only by
contract, so a memory write seam cannot break the journal's
"never rewrites" invariant or turn a mutating tool into a write sink.

### The compositions

`CompositeMemory` fans out to every source concurrently, merges, then
reranks and takes the top `k`. `SequentialMemory` tries sources in order
and stops once `k` items are collected.

```python
import asyncio

from agentkit.adapters.vector.in_memory import InMemoryVector
from agentkit.kernel.types import Scope
from agentkit.memory import CompositeMemory, MemoryItem, SequentialMemory, VectorMemory
from agentkit.testing import FakeLLM, FakeMemory, make_test_ctx

vector = VectorMemory(vector=InMemoryVector(), name="handbook")
cache = FakeMemory(
    name="cache",
    items=[MemoryItem(content="cached: refunds take 5 days", source="cache")],
)
empty_cache = FakeMemory(name="cache")


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM(), scope=Scope(org_id=1, domain_id=1))
    await vector.write(
        [MemoryItem(content="Refunds are issued within 5 business days.", source="x")],
        ctx=ctx,
    )

    fanout = CompositeMemory(sources=[cache, vector])
    print("composite:      ", [i.source for i in await fanout.query("refunds", k=5, ctx=ctx)])

    hot = SequentialMemory(sources=[cache, vector])
    print("sequential hit: ", [i.source for i in await hot.query("refunds", k=1, ctx=ctx)])

    cold = SequentialMemory(sources=[empty_cache, vector])
    print("sequential miss:", [i.source for i in await cold.query("refunds", k=1, ctx=ctx)])


asyncio.run(main())
```

```text
composite:       ['handbook', 'cache']
sequential hit:  ['cache']
sequential miss: ['handbook']
```

Reach for `CompositeMemory` when "ask everywhere and rank the union" is
right, and `SequentialMemory` when a cheap tier should answer before an
expensive one. Both implement `MemorySource`, so they nest: a
`CompositeMemory` of two `SequentialMemory` chains is a valid two-arm
rerank topology.

Note the ordering in the composite output. The default reranker,
`score_sort_rerank`, sorts by score descending with **`None` last but
never dropped** — a tool-wrapped source that does not rank still gets to
surface in the top-k. Wire a cross-encoder or an LLM judge by passing
any object with `async rerank(query, items, *, k)` as `reranker=`; that
shape is the `Reranker` Protocol.

Write semantics differ, and the difference is intentional:
`CompositeMemory.write` **broadcasts** to every source;
`SequentialMemory.write` writes to the **first** source only, because
writes belong at the cache tier in the typical setup.

A broadcast can partially fail, and that is reported rather than
flattened:

```python
import asyncio

from agentkit.memory import MemoryItem
from agentkit.memory.composite import CompositeMemory, CompositeWriteError
from agentkit.testing import FakeLLM, FakeMemory, make_test_ctx


class BrokenMemory(FakeMemory):
    async def write(self, items, *, ctx):
        raise RuntimeError("index is read-only")


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM())
    both = CompositeMemory(sources=[FakeMemory(name="cache"), BrokenMemory(name="index")])
    try:
        await both.write([MemoryItem(content="note", source="x")], ctx=ctx)
    except CompositeWriteError as exc:
        print(exc)
        print("accepted =", exc.accepted, "| failed =", list(exc.failed))


asyncio.run(main())
```

```text
CompositeMemory.write partial failure — accepted: ['cache']; failed: index: RuntimeError: index is read-only
accepted = ['cache'] | failed = ['index']
```

A naive `gather(return_exceptions=False)` would propagate the first
raising source verbatim, and the caller would have no signal that
another source's write **committed**. Postmortems need to know which
backends did.

#### Deduping the fan-out

Fanning out to several sources means the same fact can come back more
than once — and that is the *normal* case rather than bad luck, because
the journal a vector index was built from will happily return the same
row the index does. Merged blindly, one fact then occupies two of the
`k` slots the model actually reads.

```python
CompositeMemory(sources, dedupe="id")        # the default
CompositeMemory(sources, dedupe="content")   # for backends with no ids
CompositeMemory(sources, dedupe=None)        # concatenate, chosen rather than inherited
```

Two items are the same item if they share an `id` **or** their content
matches once stripped — the two relations are unioned, so a source that
supplies ids and one that does not can still agree on a fact.

Identity is deliberately strict about what counts as "having one":

- An `id` of `""` is **not** an identity. Admitting it would union every
  idless-but-not-quite-idless record from two backends into one, and the
  loser is deleted rather than ranked lower. Backends disagreed about
  this before it was pinned — `ToolMemory` normalised `""` to `None` and
  `VectorMemory` passed it through, so the same store deduped
  differently depending on which adapter the rows arrived through.
- Content that is **blank once stripped** carries no identity either, so
  `""`, `"   "` and `"\n\t"` do not collapse onto one another. They can
  still merge on a shared `id`. Without this a chunk that is whitespace
  after boilerplate stripping would swallow unrelated records that had
  perfectly good distinct ids.

On a collision the **higher score** survives, and the merged item is
stamped with `dedupe_sources` (every backend that agreed) and
`dedupe_count`. That two independent sources returned the same fact is
signal a reranker can use, and plain concatenation throws it away.

The stamp is **accumulative across nesting**. A `CompositeMemory` of two
`SequentialMemory` chains is a valid topology, and an inner composite's
stamp is absorbed by the outer one — so `dedupe_count` is the true
number of collapsed copies across the whole tree, and `dedupe_sources`
names every leaf backend that agreed rather than just the immediate
children.

Both keys are **transient per-query annotations, not record metadata**.
`VectorMemory.write` strips them rather than persisting them; a backend
that round-tripped them would inflate its own count on every pass.

### The decorators

Each is itself a `MemorySource`, single-responsibility, stackable in
whatever order matches your concerns.

- **`ScopedMemory`** — fail-loud tenant guard. Runs `enforce(ctx)` before
  every query and every write.
- **`CachedMemory`** — TTL cache over identical queries.
- **`CompactedMemory`** — shrinks each item's content through a
  `Compactor`, with an optional `max_items` cap. For when raw chunks
  would blow the prefix budget.
- **`ReadOnlyMemory`** — refuses (or deliberately drops) writes to a
  source that is read-only by policy rather than by backend.

#### Read-only by policy

Some sources an agent may read and must never extend: a curated
knowledge base, a registry an operator maintains, a corpus of recorded
facts. Nothing about the *backend* says so — the vector store behind a
curated KB takes an upsert exactly like any other. Before this
decorator the only thing protecting them was that nothing happened to
call `write`, which is a property of the code as it currently stands
rather than a rule. The first cognition taught to persist what it
learned would have written into the registry, and nothing would have
complained.

`ReadOnlyMemory` constrains exactly one verb.

```python
import asyncio

from agentkit.memory import CompositeMemory, MemoryItem, MemoryWriteRefused, ReadOnlyMemory
from agentkit.memory.composite import CompositeWriteError
from agentkit.testing import FakeLLM, FakeMemory, make_test_ctx

registry = FakeMemory(
    name="registry",
    items=[MemoryItem(content="eu-west-1 is the GDPR-resident region.", source="registry")],
)
curated = ReadOnlyMemory(inner=registry)
guess = [MemoryItem(content="I think it is eu-west-2.", source="agent")]


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM())

    print("name:", curated.name, "| accepts_writes:", curated.accepts_writes)
    print("recall:", [(i.source, i.content) for i in await curated.query("region", k=1, ctx=ctx)])

    try:
        await curated.write(guess, ctx=ctx)
    except MemoryWriteRefused as exc:
        print("refused:", exc.source, "| items:", exc.n)

    # Default policy inside a fan-out: the refusal is a failure.
    strict = CompositeMemory(sources=[FakeMemory(name="notes"), curated])
    try:
        await strict.write(guess, ctx=ctx)
    except CompositeWriteError as exc:
        print("strict :", exc.accepted, exc.refused, list(exc.failed))

    # "ignore": the writable member commits, the read-only one is bucketed apart.
    lenient_kb = ReadOnlyMemory(inner=registry, on_write="ignore")
    notes = FakeMemory(name="notes")
    await CompositeMemory(sources=[notes, lenient_kb]).write(guess, ctx=ctx)
    print("lenient: notes wrote", len(notes.writes), "| registry wrote", len(registry.writes))
    print("         refused_writes =", lenient_kb.refused_writes)


asyncio.run(main())
```

```text
name: registry | accepts_writes: False
recall: [('registry', 'eu-west-1 is the GDPR-resident region.')]
refused: registry | items: 1
strict : ['notes'] [] ['registry']
lenient: notes wrote 1 | registry wrote 0
         refused_writes = 1
```

Read the first two lines before the interesting part. `name` is
`registry`, not `read-only`, and the item comes back stamped
`source="registry"` as though the wrapper were not there: `query` is a
total pass-through — same `k`, same `where`, same items, no filtering
and no defensive copy. That is deliberate twice over. A read-only
source is meant to participate in recall exactly like every other
source, which is the whole reason to keep it around; and
`MemoryItem.source` is stamped from `name`, which the
[fan-out dedupe](#deduping-the-fan-out) treats as identity — a wrapper
that renamed the source would make the same fact stop merging depending
on whether it arrived through the wrapped path or the bare one.

**`on_write="refuse"`** (the default) raises `MemoryWriteRefused`,
carrying the source name and the item count. It subclasses
`AgentkitError`, so a run boundary already catching the framework's
taxonomy catches this too. `PermissionError` was the other candidate
and was rejected: `ScopedMemory` raises that for a tenant-boundary
violation, and "wrong tenant" and "right tenant, immutable source" want
different fixes at the catch site.

An empty write refuses as well — `write([])` reports `n=0` in the
exception rather than passing. The policy is about the attempt, not
about bytes moved: code that reaches `write` on a read-only source will
carry items the moment its input is non-empty, and letting the empty
call through moves the discovery from the first test run to production.
Nothing pays for that strictness, because both `CompositeMemory.write`
and `SequentialMemory.write` return on an empty list before reaching
any source.

**`on_write="ignore"`** exists because `CompositeMemory.write`
broadcasts to *every* source, and the `strict` line above is what that
costs under the default. The refusal is an exception like any other, so
it lands in `CompositeWriteError.failed` and the entire broadcast
raises — after `notes` has already committed, which is why it shows up
in `accepted`. One read-only member makes every write through that
composite an error, and that is a wrong report rather than a strict
one: nothing is broken.

Under `ignore` the source returns normally having written nothing, and
the composite files it in a **third bucket**. `CompositeWriteError`
carries `accepted`, `refused` and `failed`, and `refused` is not a
slice of `accepted` — `accepted` is what an operator reads to decide
which backends *not* to replay after a partial commit, and a source
that never committed does not belong on that list. Note what the
`lenient` line does **not** print: nothing failed, so no exception is
raised at all. `refused` only becomes visible when some *other* member
of the same fan-out fails, and then it appears in the message as
`refused (read-only): [...]`.

That is the honest cost of `ignore`: on the happy path the caller of
`write` is told nothing. "Silently succeeded" is the failure mode this
package keeps writing tests against, so the drop is accounted for three
ways out of band —

- `refused_writes` counts the turned-away calls on the instance (the
  last line of the example),
- a `memory.write_refused` observation lands on the run's timeline
  carrying the source, the item count and the policy,
- and `accepts_writes = False` is the marker the composite reads to
  bucket it.

`accepts_writes` is the one attribute these decorators agree on beyond
the Protocol, and it is read as `getattr(source, "accepts_writes",
True)` so every backend written before it existed keeps the permissive
default. It is deliberately **not** on `MemorySource`: that Protocol is
`@runtime_checkable`, and adding a non-method member would make
`isinstance(x, MemorySource)` `False` for every backend that predates
it. On `ReadOnlyMemory` it is a `ClassVar` rather than a field, so no
caller can pass `accepts_writes=True` to a read-only source.

It also survives nesting, which is what makes it worth having:

| Wrapper | `accepts_writes` is |
|---|---|
| `ScopedMemory`, `CachedMemory`, `CompactedMemory` | the wrapped source's |
| `CompositeMemory` | `any(...)` — true if at least one member can commit |
| `SequentialMemory` | the **first** source's, the only one `write` touches |

So `ScopedMemory(ReadOnlyMemory(kb))` inside a fan-out is still
bucketed as refused rather than accepted, and an all-read-only
`CompositeMemory` nested in an outer one reports `False` instead of
laundering itself into the outer `accepted`. The `SequentialMemory` row
is the sharp edge: a chain whose *cache tier* is read-only reports
`False` even when the vector tier behind it is perfectly writable. That
is correct — `SequentialMemory.write` only ever touches the first
source — but it reads as wrong if you think of the marker as being
about "the sources" rather than about the write target.

One nicety falls out of the same marker: `ScopedMemory.write` skips its
`memory.written` observation when the source beneath it does not accept
writes, so an `ignore` stack does not put a write on the operator's
timeline that never happened.

`ReadOnlyMemory` is a `MemorySource` like the rest, so it nests in both
directions and the order decides what you get.
`ReadOnlyMemory(ScopedMemory(kb))` refuses before the tenant check runs
— cheaper, and no `PermissionError` masking the real reason;
`ScopedMemory(ReadOnlyMemory(kb))` checks the tenant first.
Double-wrapping is harmless, since the outer refuses and the inner
never runs. And a typo in `on_write` is a `ValueError` at construction
rather than at the first write, because a policy that only surfaced
when something tried to write would lie dormant for exactly as long as
the bug this class replaces.

## Tenant scoping

Multi-tenant isolation happens on two levels, and you want both.

**At the backend.** `VectorMemory` passes `ctx.scope` straight through to
`VectorPort.upsert` / `VectorPort.search` as the bucket key, so a query
physically cannot see another tenant's chunks.

**At the framework boundary.** `ScopedMemory` refuses to call the inner
source at all unless the scope is populated. The default policy requires
**both** `org_id` and `domain_id` to be set — checked with explicit
`is not None`, because a valid tenant id may be `0` (the `root` org, the
`default` domain) and `bool(0)` would misclassify it as unscoped.

```python
import asyncio

from agentkit.adapters.vector.in_memory import InMemoryVector
from agentkit.kernel.types import Scope
from agentkit.memory import MemoryItem, ScopedMemory, VectorMemory
from agentkit.testing import FakeLLM, make_test_ctx

guarded = ScopedMemory(inner=VectorMemory(vector=InMemoryVector(), name="handbook"))


async def main() -> None:
    tenant = make_test_ctx(llm=FakeLLM(), scope=Scope(org_id=1, domain_id=1))
    await guarded.write([MemoryItem(content="Tenant 1 policy.", source="x")], ctx=tenant)
    print("in-tenant:   ", [i.content for i in await guarded.query("policy", k=1, ctx=tenant)])

    other = make_test_ctx(llm=FakeLLM(), scope=Scope(org_id=2, domain_id=1))
    print("other tenant:", await guarded.query("policy", k=1, ctx=other))
    print("guard name:  ", guarded.name)

    unscoped = make_test_ctx(llm=FakeLLM())  # Scope() — no org_id, no domain_id
    try:
        await guarded.query("policy", k=1, ctx=unscoped)
    except PermissionError:
        print("refused: unscoped run")


asyncio.run(main())
```

```text
in-tenant:    ['Tenant 1 policy.']
other tenant: []
guard name:   handbook
refused: unscoped run
```

A falsy return from `enforce` **or** any exception raised inside it both
become `PermissionError`, and the inner source is never touched. Pass
`enforce=` to pin a stricter policy ("must match this specific tenant",
"must be inside an approved checkpoint").

`ScopedMemory.name` mirrors the wrapped source — `handbook`, not
`scoped` — so attribution on `MemoryItem.source` is identical with and
without the guard. On the happy path the guard is invisible.

`ScopedMemory.write` also emits a `memory.written` observation carrying
`{"n": …, "source": …}`, best-effort. A memory write is the kind of side
effect operators want on the run timeline.

The tenant axis reaches the cache too. `CachedMemory`'s key is
`(scope_key, query, k, canonical-json(where))`. Without `scope_key` in
it, stacking `ScopedMemory(CachedMemory(inner))` would pass tenant B's
`enforce` check and then serve tenant A's cached items. And `where` is
serialised as canonical JSON rather than a `frozenset` of its items,
because `where={"tags": ["security", "finance"]}` breaks the set
fallback — lists are not hashable.

```python
import asyncio

from agentkit.kernel.types import Scope
from agentkit.memory import CachedMemory, MemoryItem, ScopedMemory
from agentkit.testing import FakeLLM, FakeMemory, make_test_ctx

slow = FakeMemory(name="handbook", items=[MemoryItem(content="Refunds take 5 days.", source="x")])
memory = ScopedMemory(inner=CachedMemory(inner=slow, ttl_seconds=30.0))


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM(), scope=Scope(org_id=1, domain_id=1))
    await memory.query("refunds", k=1, ctx=ctx)
    await memory.query("refunds", k=1, ctx=ctx)
    print("backend calls:", len(slow.queries))


asyncio.run(main())
```

```text
backend calls: 1
```

## Tool-backed memory

`ToolMemory` adapts any `Tool` into a `MemorySource`. The tool is called
with `{query_arg: query, "limit": k, **(where or {})}` — `where` keys
merge in last, so a caller-supplied `limit` overrides `k`, which is the
intuitive precedence.

```python
import asyncio

from agentkit.memory import ToolMemory
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import tool


@tool(side_effecting=False, idempotent=True)
def search_docs(query: str, limit: int = 5) -> list[dict]:
    """Search the internal engineering handbook and return matching passages."""
    corpus = [
        {"content": "Deploys freeze on Fridays.", "url": "/handbook/deploys", "score": 0.9},
        {"content": "Rollbacks need two approvers.", "url": "/handbook/rollback", "score": 0.7},
    ]
    return corpus[:limit]


probe = ToolMemory(tool=search_docs, name="handbook")


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM())
    for item in await probe.query("deploy policy", k=2, ctx=ctx):
        print(item.source, item.score, item.metadata, "|", item.content)
    await probe.write([], ctx=ctx)  # no-op by contract


asyncio.run(main())
```

```text
handbook 0.9 {'url': '/handbook/deploys'} | Deploys freeze on Fridays.
handbook 0.7 {'url': '/handbook/rollback'} | Rollbacks need two approvers.
```

`default_parse` handles four shapes and refuses to be clever about
anything else:

- a `dict` with a `"results"` list → unwrap and recurse
- `list[dict]` → each dict is `{content, score?, metadata?}`, with
  `text` and `snippet` accepted as content aliases and every remaining
  key preserved as metadata (that is where `url` above came from)
- `list[str]` → one item per string
- a `str` → split on blank lines

Anything else is wrapped as a single `str(result)` item — visible in
traces, not useful. If your tool returns something richer, pass
`result_to_items=`.

The same underlying callable can be registered as a `FunctionTool` for
LLM-decided calls **and** wrapped as a `ToolMemory` for
cognition-decided pre-fetch. That is the intended pattern, not a
conflict. But writes are a no-op by adapter contract: a tool that mutates
the world stays a tool.

## What bites people

!!! warning "`source` on write is not preserved"
    `MemoryItem.source` is stamped by whichever backend returned the
    item, using that backend's `name`. Setting it on the way in does
    nothing. If you need provenance to survive a round trip, put it in
    `metadata`.

!!! warning "`VectorMemory.write` consumes `metadata["id"]`"
    That key is popped and used as the chunk id (a fresh uuid otherwise),
    so it will not be in the metadata you get back. Every other key
    survives and stays available to `where=` filters.

!!! warning "`CachedMemory`'s scope guard rarely fires"
    It warns (or raises, under `strict_scope=True`) when
    `ctx.scope.key()` is empty. But `Scope()` — the zero-tenant default —
    returns the non-empty string `"orgNone:domNone"`, so with a real
    `RunContext` the guard never trips. Unscoped runs all share that one
    bucket. It is not dead — it does fire for a duck-typed context with no
    `.scope` attribute at all — but it cannot see the case that actually
    matters. Wrap with `ScopedMemory`, which does refuse, rather than
    relying on this.

    Widening it to treat `Scope()` as unscoped was tried and reverted:
    `Scope()` is the documented zero-tenant default, so warning on it fires
    at every legitimate single-tenant app on every query. The narrow guard
    plus `ScopedMemory` is the better trade.

!!! warning "The cache does not know about time-varying backends"
    `CachedMemory` is keyed on the query, not on the data. A write
    through the same decorator clears it; a write that reaches the
    backend by another path does not. `ttl_seconds` (default 60) is your
    only bound. It is also not thread-safe — agents are single-flow per
    loop.

!!! tip "Bound the fan-out"
    `CompositeMemory` gathers all sources at once, so a merged pool is
    up to `len(sources) * k` items before the rerank cut. For a wide
    tree, nest composites so the fan-out stays moderate at each level.

!!! warning "Annotate after construction and it will raise"
    Backends that used to do `item.metadata["path"] = p` now get a
    `TypeError`. The migration is one line:
    `dataclasses.replace(item, metadata={**item.metadata, "path": p})` —
    and `replace` re-runs `__post_init__`, so the rebuilt item is frozen
    too.

## Related

- [Capabilities](capabilities.md) — `RequestBuilder`, the `Grounder`
  seam, and the `Compactor` that `CompactedMemory` wraps.
- [Tools](tools.md) — the other outward-facing seam, and `FileTool`,
  whose file tree `FileMemory` reads.
- [Agents](agents.md) — where `memory=` is declared and how cognitions
  consume it.
- [Skills](skills.md) — bundling a prompt, a cognition and a memory into
  one reusable unit.
- [API › memory](../api-reference/memory.md) — the generated reference.
