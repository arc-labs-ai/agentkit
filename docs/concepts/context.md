# Context

`WorkingContext` is what an agent knows right now — the transcript it is
reasoning over, the notes it has taken, and the record of what it has
done.

## The problem it solves

An agent loop needs somewhere to put its state. The obvious answer is
"a list of messages", and it holds up until the second week: you want to
hand a sub-agent a briefing without letting its scribbles leak back into
the parent, you want to drop old turns without losing the system prompt,
and you want to know how big the transcript is *before* the provider
rejects it.

The expensive part is subtler. Providers cache the prompt **prefix**, and
a cache read is billed at a fraction of a fresh read — for
`claude-sonnet-4-6`, `$0.30` per Mtok against `$3.00`. Editing anything
near the top of the message list invalidates that cache from that token
forward. If your grounding chunks and your per-turn chatter live in the
same mutable list, you will invalidate the cache constantly without ever
noticing:

```python
from agentkit.adapters.llm.providers.pricing import cost
from agentkit.kernel.types import Usage

# A 20k-token grounding prefix, re-sent on every turn of a 30-turn run.
fresh = Usage(input_tokens=20_000, output_tokens=200)
cached = Usage(input_tokens=0, output_tokens=200, cache_read_tokens=20_000)

print(round(cost("claude-sonnet-4-6", fresh) * 30, 4))    # 1.89
print(round(cost("claude-sonnet-4-6", cached) * 30, 4))   # 0.27
```

`WorkingContext` separates the two so that stops being an accident.

## The smallest thing that works

```python
import asyncio

from agentkit.context import PrefixContext, WorkingContext
from agentkit.kernel.types import Message

ctx = WorkingContext(
    prefix=PrefixContext(system_prompt="You are a careful research assistant."),
)
ctx.append(Message("user", "What changed in the pricing table?"))
ctx.note("ticket", "ENG-4412")

print(ctx.size())                              # 1  — the tail only
print([m.role for m in ctx.assembled()])       # ['system', 'user']
print(ctx.get("ticket"))                       # ENG-4412
print(asyncio.run(ctx.tokens()))               # 25
```

`ctx.size()` counts the tail. `ctx.assembled()` is what the provider
actually sees: the prefix rendered to messages, then the tail.

!!! note "`WorkingContext` is not `RunContext`"

    `RunContext` (from [`agentkit.runtime`](runtime.md)) is *execution
    wiring* — correlation id, scope, budget, services, cancellation.
    `WorkingContext` is *what the agent knows*. A run has one
    `RunContext`; the agents inside it may hold many
    `WorkingContext`s.

## How it works

A `WorkingContext` has four axes, and they are orthogonal — you use the
ones you need and the rest stay empty.

| Axis | Field | Shape | Who uses it |
| --- | --- | --- | --- |
| Prefix | `prefix` | frozen `PrefixContext` | every LLM call |
| Transcript tail | `messages` | `list[Message]` | every LLM call |
| Scratchpad | `scratchpad` | `dict[str, Any]` | cross-step notes |
| Journal | `journal` | `MutationJournal` | hierarchical agents |

A single-shot chat agent uses the first three and never touches the
journal. A long-lived sub-agent reporting to a parent uses all four.

Long-term recall is deliberately **not** an axis. It lives on the agent
as `Agent.memory: MemorySource | None`, and the cognition folds recalled
items into the prompt through the `RequestBuilder` grounding seam. See
[Memory](memory.md).

### The prefix is a separate type because the cache is real

`PrefixContext` holds the cache-stable head: the system prompt, the
grounding chunks, and (when the agent declares an `output=` schema) the
rendered schema block. It is `@dataclass(frozen=True)`, and that is a
guarantee rather than a style choice:

```python
from agentkit.context import PrefixContext, WorkingContext
from agentkit.kernel.types import Message

prefix = PrefixContext(
    system_prompt="You are a careful research assistant.",
    grounding=(Message("system", "Q3 pricing: input $3/Mtok, output $15/Mtok."),),
)

ctx = WorkingContext(prefix=prefix)
ctx.append(Message("user", "What is the output price?"))

for m in ctx.assembled():
    print(m.role, "|", m.content)

try:
    prefix.system_prompt = "something else"
except Exception as exc:
    print(type(exc).__name__)     # FrozenInstanceError
```

Every mutator on `WorkingContext` — `append`, `extend`,
`clear_messages`, `note`, `update_scratchpad` — touches the tail or the
scratchpad. None of them can reach the prefix. To change the prefix you
build a new one, which is exactly the moment you *should* be thinking
about a cache miss.

`PrefixContext.as_messages()` renders the head in a fixed order —
pinned system prompt, then grounding in order, then `schema_block` (as a
`system` message named `schema`) when set. It returns a fresh list every
call, so a caller that mutates the result cannot corrupt the frozen
prefix by aliasing.

Downstream, the `cache_hint` field on `ChatRequest` is what turns this
into an actual provider cache: the Anthropic adapter maps a truthy
`cache_hint` onto `cache_control: {"type": "ephemeral"}` on the system
block. Equality on `PrefixContext` is structural, so two prefixes built
from the same `(system_prompt, grounding, schema_block)` are equal and
hash equal — usable as a recall-cache key. See
[Adapters](adapters.md).

## Composition: fork, merge, slice

### `fork()` — an independent copy

```python
from agentkit.context import WorkingContext
from agentkit.kernel.types import Message

parent = WorkingContext()
parent.append(Message("user", "Summarise the incident."))
parent.note("owner", "sre")

child = parent.fork()
child.append(Message("assistant", "Root cause: expired cert."))
child.note("owner", "me")

print(parent.size(), parent.get("owner"))   # 1 sre
print(child.size(), child.get("owner"))     # 2 me
```

The child gets its own message list, a `copy.deepcopy` of the
scratchpad, and a fresh journal seeded from the parent's entries. The
prefix is shared **by reference** — it is frozen, so sharing it is safe
and keeps the cache key identical across the fork. The token counter,
`limit` and `shared` flag are inherited; the lock is new.

### `merge()` — fan-in

```python
from agentkit.context import WorkingContext
from agentkit.kernel.types import Message

hit = Message("tool", "cert expired 2026-08-01", name="lookup")

a = WorkingContext().append(hit)
b = WorkingContext().append(hit)

union = WorkingContext()
union.merge(a, mode="union").merge(b, mode="union")
print(len(union.messages))    # 1

concat = WorkingContext()
concat.merge(a).merge(b)
print(len(concat.messages))   # 2
```

`mode="concat"` (the default) appends verbatim. `mode="union"`
deduplicates by value first, which is what a parent fanning two siblings
back in wants. "By value" means `__eq__`: `Message.__hash__` only picks
the bucket, so two tool-call messages sharing a call id but differing in
arguments collide and both survive, as they must. That hash exists only
because `ToolCall` defines one — before it did, this line raised
`TypeError: unhashable type: 'dict'` on every real fan-in.

Either mode also bulk-updates the scratchpad (last-write-wins) and
applies the other journal's entries without rewriting them, so author
attribution survives the merge.

### `slice()` — a narrowed view

A `ContextScope` is a predicate `(message, index) -> bool`.
`slice(scope)` materialises a new `WorkingContext` with only the
matching messages; prefix, scratchpad and journal are inherited
unchanged, and the copy is private (`shared=False`) by default.

```python
from agentkit.context import LastNTurns, Not, RoleFilter, WorkingContext
from agentkit.kernel.types import Message

ctx = WorkingContext()
ctx.append(Message("system", "House rules."))
for i in range(3):
    ctx.append(Message("user", f"q{i}"), Message("assistant", f"a{i}"))

print([m.content for m in ctx.slice(LastNTurns(1)).messages])
# ['House rules.', 'q2', 'a2']

print(len(ctx.slice(Not(RoleFilter(frozenset({"tool"})))).messages))
# 7
```

The built-in scopes, all frozen dataclasses (so hashable, picklable and
usable as cache keys for re-grounding decisions):

| Scope | Keeps |
| --- | --- |
| `LastNTurns(n)` | the last `n` turns — **plus every system message** |
| `RoleFilter(frozenset({...}))` | messages whose `role` is in the set |
| `Tagged(tag)` | messages whose `name` equals `tag` |
| `Since(checkpoint_index)` | messages with index ≥ the checkpoint |
| `AllOf((a, b, ...))` | messages every inner scope keeps |
| `AnyOf((a, b, ...))` | messages any inner scope keeps |
| `Not(scope)` | messages the inner scope drops |

`ContextScope` is a `runtime_checkable` Protocol, so your own predicate
class works anywhere the built-ins do — it needs one method,
`matches(message, index) -> bool`.

## The journal axis

`MutationJournal` is an append-only log with a watermark. It exists for
one job: letting a child stream progress upward without re-shipping
entries the parent has already absorbed.

```python
from agentkit.context import MutationJournal, WorkingContext

ctx: WorkingContext = WorkingContext()
ctx.journal.record({"kind": "row_written", "id": 1, "agent_id": "enricher-a"})
ctx.journal.record({"kind": "row_written", "id": 2, "agent_id": "enricher-a"})

print(ctx.journal.has_uncommitted(), ctx.journal.size())   # True 2
print(len(ctx.journal.diff()))                             # 2

ctx.journal.mark_committed()                # the parent absorbed them
print(ctx.journal.has_uncommitted())        # False

ctx.journal.record({"kind": "row_written", "id": 3, "agent_id": "enricher-a"})
print(len(ctx.journal.diff()))              # 1  — only the new one ships
print(len(ctx.journal.view()))              # 3  — the history is intact

# Re-hydrating from a snapshot: seed entries start ALREADY committed.
seeded: MutationJournal[dict] = MutationJournal([{"kind": "restored"}])
print(seeded.has_uncommitted())             # False
```

`JournalEntryT` is a `TypeVar`, so project code parameterises the
journal with its own mutation union — typically a discriminated union of
domain mutation kinds, each stamped with an `agent_id`. The journal
never rewrites an entry, which is what makes attribution survive a chain
of merges.

`apply()` trusts the caller's ack protocol: **do not double-apply the
same diff**. The journal has no idempotency of its own.

## Snapshots and diffs

`freeze()` returns a `FrozenContext`: a frozen prefix, a message tuple,
the scratchpad as a sorted `(key, value)` tuple (so equality is
deterministic), and a tuple of journal entries. Hand one across an agent
boundary when you want to brief a child without letting its writes leak
back.

```python
import copy
import pickle

from agentkit.context import WorkingContext
from agentkit.kernel.types import Message

ctx = WorkingContext()
ctx.append(Message("user", "hello"))
ctx.note("plan", {"steps": ["a", "b"]})     # a nested value — the ordinary case

snap = ctx.freeze()
print(hash(snap) == hash(ctx.freeze()))     # True
print(pickle.loads(pickle.dumps(snap)) == snap)   # True
print(copy.deepcopy(snap) == snap)                # True

ctx.append(Message("assistant", "hi"))
print(len(snap.messages), len(ctx.messages))      # 1 2
```

That snapshot is hashable **even though a scratchpad value is a dict**,
and the reason is worth knowing. `FrozenContext.__hash__` hashes the
prefix, the message tuple, and the scratchpad **keys** — never the
values, never the journal entries. The dataclass-generated all-fields
hash could not do this: `update_scratchpad({"plan": {...}})` is the
documented API, so `hash(...)` raised `TypeError: unhashable type:
'dict'` on ordinary use, and it failed *by value* rather than by type —
the same code path worked right up until someone stored a dict. Keys are
kept because they are `str` by construction and they are the part that
actually varies between snapshots a memoizer sees.

Excluding the values is sound rather than a dodge: `__eq__` still
compares every field, and the hash contract only requires *equal*
objects to hash equally. Two snapshots differing only in a scratchpad
value land in one bucket and `__eq__` separates them there.

The cost is structural and therefore linear: measured on tool-calling
transcripts, 5.2 µs at 10 messages, 41 µs at 100, 396 µs at 1000. Fine
once per snapshot; if you are hashing the same `FrozenContext` in a
tight loop, hold the key instead of recomputing it.

`diff(other)` computes `self - other` for replay and debugging:

```python
from agentkit.context import WorkingContext
from agentkit.kernel.types import Message

before = WorkingContext().append(Message("user", "q1")).note("stage", "plan")
after = before.fork().append(Message("assistant", "a1")).note("stage", "act")

d = after.diff(before)
print(len(d.messages_added), len(d.messages_removed))   # 1 0
print(d.scratchpad_changes)                             # {'stage': 'act'}
print(d.prefix_changed)                                 # False

try:
    d.scratchpad_changes["stage"] = "tampered"
except TypeError as exc:
    print(str(exc).split(".")[0])
    # this payload belongs to a frozen value and cannot be mutated in place
```

`ContextDiff.scratchpad_changes` is deeply frozen at construction. The
class always promised to be a value; leaving one plain dict in the
middle meant a delta could be edited after it had been computed,
reported and logged — and because `diff()` reads live scratchpad values,
a nested value in the diff *was* the live object, so a later
`update_scratchpad` retroactively changed what the diff said. Freezing
copies, which un-aliases it. Cost: 0.49 µs for the empty diff, ~7.6 µs
at 344 B of changed values, paid once in a debug helper.

The journal is deliberately absent from `ContextDiff` — it has its own
watermark-based `journal.diff()`, which is the right semantic for
streaming.

## Token counting

`ctx.tokens()` is `await`-able because the counter seam is async: a real
counter may call a provider-side endpoint or warm a lazy encoder. Two
in-process counters ship:

- `ApproxTokenCounter` (the default) — chars/4, free, no dependency,
  within ~30% for English.
- `TiktokenCounter` — provider-accurate when `tiktoken` is installed
  (`arc-agentkit[fast]`). Falls back to the approximation silently if
  the import or the encoding lookup fails; no `ImportError`, just less
  accuracy.

Both count the **same things** and differ only in the chars→tokens step.
They had to be made to agree:

```python
import asyncio

from agentkit.capabilities.compaction.base import _approx_tokens
from agentkit.context import ApproxTokenCounter, WorkingContext
from agentkit.context.tokens import estimate_message_tokens
from agentkit.kernel.types import Message, ToolCall

transcript = [
    Message("user", "check the cert"),
    Message(
        "assistant",
        "",
        tool_calls=(ToolCall("c1", "openssl", {"host": "example.com", "port": 443}),),
    ),
    Message("tool", "notAfter=2026-08-01", name="openssl", tool_call_id="c1"),
]

print(estimate_message_tokens(transcript))                        # 37
print(_approx_tokens(transcript))                                 # 37
print(asyncio.run(ApproxTokenCounter().estimate(transcript)))     # 37
print(asyncio.run(WorkingContext(messages=transcript).tokens()))  # 37

print(estimate_message_tokens([]))            # 0  — an empty transcript is free
print(estimate_message_tokens([transcript[1]]))   # 19 — content is "", but not free
```

Every in-process estimator now delegates to one function,
`estimate_message_tokens`. Before that, `ApproxTokenCounter` and the
compactors' `_approx_tokens` were independent copies of
`sum(len(m.content)) // 4`, and **both ignored `tool_calls` entirely**.
Measured on an 80-message tool-heavy transcript carrying 324,420
characters of tool arguments (~81k real tokens): both estimators
reported `0`, and `TruncationCompactor(max_tokens=1000)` kept 80 of 80
messages. The whole transcript went to the provider and died there as a
400, far from the cause. Tool-heavy *is* the normal agentic shape, so
"compaction never fires" was the default path.

What gets counted, and why:

- `content` — the text of the turn. A tool result arrives as
  `Message.content` on a `role="tool"` message, so results were always
  covered and are not double-counted.
- `name` / `tool_call_id` — echoed on the wire for every tool result.
  Small, but real.
- `tool_calls` — in their serialised wire form (id, name, JSON
  arguments), because that is what the provider bills. This is the term
  whose absence caused the bug.
- Four tokens of structural overhead per message *and* per tool call.
  Role markers, block delimiters and turn sentinels cost ~3-4 tokens
  each on every provider agentkit targets; without this term 500 empty
  messages estimated as 0 for what is genuinely ~2k tokens of
  scaffolding.

It is deliberately not inflated further: ~40 extra tokens on a 10-turn
chat against a 12,000-token default budget, and an empty transcript is
exactly 0 so budget pre-checks have a zero-cost starting point.

`WorkingContext.limit` is a declared ceiling, not an enforcement
mechanism — the caller's policy still owns the abort.

## Scope and tenant isolation

`Scope` lives in `agentkit.kernel.types`, not here, but it is the reason
several context operations are safe to share. It is the tenant key
threaded through every memory recall, cache key, meter and callback.

```python
import asyncio

from agentkit import Agent
from agentkit.adapters.store import InMemoryStore
from agentkit.kernel.types import Scope
from agentkit.middlewares import memoize
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    llm = FakeLLM("tenant-1 secret")
    store = InMemoryStore()             # ONE store, deliberately shared
    agent = Agent("analyst", model="fake-model")

    for scope in (Scope(org_id=1), Scope(org_id=999)):
        ctx = make_test_ctx(
            llm=llm, store=store, scope=scope, chat_middleware=[memoize()]
        )
        await agent.run("what is the number?", ctx)

    print(llm.calls)                    # 2 — one provider call per tenant
    print(Scope(org_id=1).key())        # org1:domNone


asyncio.run(main())
```

`memoize()` namespaces every key with `ctx.scope.key()` **before** it
reaches the store, so a cache entry cannot cross a tenant boundary
whatever the caller's `key=` callable returns. It did not always: the
key documented in the cheatsheet was the last message's text, and two
tenants asking the same question shared one cached LLM response.
Measured: tenant 999 received tenant 1's answer from a single provider
call. A tenant boundary that depends on every caller remembering is not
a boundary.

Cross-tenant sharing is still available — use a scope-less store or a
shared `Scope`. It just has to be an explicit act.

## Concurrency

`WorkingContext` is private by default. `fork()` gives you an
independent copy; that covers most fan-out.

For a team blackboard, pass the same context as `context=` and set
`shared=True`. The simple sync mutators (`append` / `extend` / `note` /
`update_scratchpad` / `clear_messages`) are GIL-atomic — two coroutines
on one event loop cannot interleave inside any single call, so they need
no lock.

The lock is for *sequences*. Acquire it when a series of mutations with
an `await` between them must look atomic to other coroutines, or when a
read-modify-write must not be lost:

```python
import asyncio

from agentkit.context import WorkingContext
from agentkit.kernel.types import Message


async def main() -> None:
    board = WorkingContext(shared=True)

    async def worker(name: str) -> None:
        async def add_pair(ctx: WorkingContext) -> None:
            ctx.append(Message("assistant", f"{name}: starting"))
            await asyncio.sleep(0)                 # a real await mid-sequence
            ctx.append(Message("assistant", f"{name}: done"))

        await board.apply_locked(add_pair)

    await asyncio.gather(*(worker(n) for n in ("a", "b", "c")))
    print([m.content for m in board.messages])
    # ['a: starting', 'a: done', 'b: starting', 'b: done', 'c: starting', 'c: done']


asyncio.run(main())
```

`await ctx.apply_locked(fn)` and `async with ctx.lock:` are equivalent;
`apply_locked` also accepts an async closure and awaits it inside the
lock. `shared=True` does **not** auto-lock the mutators — the lock is a
public primitive and the boundary is explicit on purpose.

## What bites people

!!! warning "`LastNTurns` loses its window inside a combinator"

    `LastNTurns` cannot decide membership from `(message, index)`
    alone — it needs the whole list to count back `n` turns, so it
    exposes a private `_surviving_indices` hook that `slice()` prefers.
    `AllOf` / `AnyOf` / `Not` do not forward that hook, and
    `LastNTurns.matches()` returns `True` for everything as a
    safe-by-default fallback. Wrap it in a combinator and the window
    silently disappears:

    ```python
    from agentkit.context import AllOf, LastNTurns, Not, RoleFilter, WorkingContext
    from agentkit.kernel.types import Message

    ctx = WorkingContext()
    ctx.append(Message("system", "House rules."))
    for i in range(3):
        ctx.append(Message("user", f"q{i}"), Message("assistant", f"a{i}"))

    print([m.content for m in ctx.slice(LastNTurns(2)).messages])
    # ['House rules.', 'q1', 'a1', 'q2', 'a2']    — windowed

    inside = AllOf((LastNTurns(2), Not(RoleFilter(frozenset({"system"})))))
    print([m.content for m in ctx.slice(inside).messages])
    # ['q0', 'a0', 'q1', 'a1', 'q2', 'a2']        — window GONE

    narrowed = ctx.slice(LastNTurns(2)).slice(Not(RoleFilter(frozenset({"system"}))))
    print([m.content for m in narrowed.messages])
    # ['q1', 'a1', 'q2', 'a2']                    — do this instead
    ```

    Window first, then filter the result.

!!! warning "`fork()` then `merge()` duplicates the shared history"

    `fork()` copies the parent's tail, so merging the child back in
    `concat` mode re-appends every message they already share. A
    parent with one message that forks a child, gets one reply, and
    merges ends up with three messages, two of them identical. Use
    `mode="union"`, or fork from a slice narrow enough that the
    overlap does not matter.

!!! warning "`freeze()` does not deep-freeze scratchpad values"

    The transcript axis of a `FrozenContext` is genuinely immutable —
    `messages` is a tuple of frozen `Message`s. The scratchpad is a
    tuple of `(key, value)` pairs, and the **values are the live
    objects**:

    ```python
    from agentkit.context import WorkingContext

    ctx = WorkingContext().note("plan", {"steps": ["a"]})
    snap = ctx.freeze()
    ctx.scratchpad["plan"]["steps"].append("b")
    print(dict(snap.scratchpad))     # {'plan': {'steps': ['a', 'b']}}
    ```

    `ContextDiff` deep-freezes its payload; `FrozenContext` does not.
    If you are handing a snapshot across an agent boundary and the
    scratchpad holds mutable structures, deep-copy them yourself
    first.

!!! warning "A `WorkingContext` is not a `RunContext`, and neither is a `Ctx`"

    Three similarly-named things. `WorkingContext` is reasoning state.
    `RunContext` is execution wiring. `Ctx` is the narrow Protocol a
    tool or middleware sees. Passing the wrong one is a type error at
    the call site, not a silent bug — but the names are close enough
    that it is worth reading twice.

!!! warning "Nothing here calls an LLM"

    This package is pure data operations: slice, fork, merge, freeze,
    diff, record, estimate. It does not compact, retrieve, or invoke.
    Those policies belong to their owners — `Compactor`,
    `MemorySource`, `Invoker` — and you compose them around a
    `WorkingContext` explicitly. If you are looking for "where does
    the transcript get shortened", that is
    [Capabilities](capabilities.md), not here.

## Related

- [Runtime](runtime.md) — `RunContext`, `Budget`, `Services`, the
  execution wiring `WorkingContext` sits beside.
- [Capabilities](capabilities.md) — `RequestBuilder` and `Compactor`,
  the policies that read and reshape a context.
- [Adapters](adapters.md) — where `cache_hint` becomes a real provider
  prompt cache, and where `Scope` becomes a real storage partition.
- [Memory](memory.md) — `Agent.memory` and how recalled items reach the
  prompt.
- [API › context](../api-reference/context.md) — the generated
  reference.
