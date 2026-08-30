# How do I make an agent answer from my documents?

Memory is what the agent can reach for. It is how an answer comes from
your handbook, your tickets, or your codebase instead of from whatever
the model absorbed in training.

## When you'd want this

Ask a general-purpose model a question about your refund policy and it
will answer. The answer will be well-written, specific, and about
somebody else's refund policy — and there is nothing in the reply that
says so.

Grounding fixes that by putting the relevant passages in front of the
model on the way in, with attribution, so the answer can cite what it
came from and you can tell when it came from nothing.

The seam is one Protocol, `MemorySource`, with a deliberately narrow
shape: `query(query, k, ctx)` and `write(items, ctx)`. Every backend —
vector store, files, an ordinary search tool, a fan-out over all three —
answers with the same `MemoryItem`, so the cognition consuming results
never learns where they came from.

## Working code

```python
"""Runs offline. FakeLLM stands in for the model; the retrieval is real."""

import asyncio

from agentkit import Agent, MemoryItem, Scope, VectorMemory
from agentkit.adapters.vector import InMemoryVector
from agentkit.testing import FakeLLM, make_test_ctx

HANDBOOK = [
    MemoryItem(
        content="Refunds are issued to the original payment method within 5 business days.",
        source="handbook",
        metadata={"id": "refunds", "team": "support"},
    ),
    MemoryItem(
        content="Orders over $500 need a manager's approval before they ship.",
        source="handbook",
        metadata={"id": "approvals", "team": "ops"},
    ),
]


async def main() -> None:
    memory = VectorMemory(vector=InMemoryVector(), name="handbook")
    ctx = make_test_ctx(
        llm=FakeLLM("Refunds go back to the original payment method in 5 business days."),
        scope=Scope(org_id="acme"),
    )
    await memory.write(HANDBOOK, ctx=ctx)

    # `memory=` is the whole wiring. The Agent builds a grounder from it.
    agent = Agent(
        name="support",
        model="claude-sonnet-4-6",
        prompt="Answer only from the grounding block. If it isn't there, say you don't know.",
        memory=memory,
    )
    print((await agent.run("How long do refunds take?", ctx)).output)

    # What the grounder retrieved and injected, with its scores:
    for item in await memory.query("How long do refunds take?", k=2, ctx=ctx):
        print(f"  {item.score:.2f} [{item.source}] {item.content}")

    # Isolation is the port's job, and it is real — a different tenant
    # searching the same store sees nothing.
    other = make_test_ctx(llm=FakeLLM("x"), scope=Scope(org_id="globex"))
    print("other tenant sees:", await memory.query("refunds", k=2, ctx=other))


asyncio.run(main())
```

Output:

```text
Refunds go back to the original payment method in 5 business days.
  0.13 [handbook] Refunds are issued to the original payment method within 5 business days.
other tenant sees: []
```

## What the model actually sees

Setting `memory=` on an `Agent` wires `as_grounder(memory)` into the
`RequestBuilder`. On the first turn the grounder runs, its results are
rendered as `[source] content` lines, and the block lands in the
**prefix** — the cache-stable head of the working context, alongside the
system prompt:

```text
Answer only from the grounding block. If it isn't there, say you don't know.

Relevant context:
[handbook] Refunds are issued to the original payment method within 5 business days.
```

Two consequences worth holding on to.

**The prefix is not rewritten mid-run.** That is the KV-cache
discipline: mutating a token in the prefix invalidates the provider's
cache from that point onwards. Grounding is fetched once by default, and
compaction only ever touches the tail. The cost of that default shows up
in a multi-turn conversation — turn 5 gets turn 1's evidence — and
[`reground_every_turn`](../concepts/capabilities.md#the-cache-stable-prefix)
is the knob that trades it back for a 10x prefix bill.

**The query is the task, verbatim.** `as_grounder` passes the user's
string straight to `memory.query(...)`. If you want query rewriting —
HyDE, decomposition, a cheap model rewriting the question first — wrap
the `MemorySource` rather than reaching inside the grounder.

## Taking control of retrieval

The auto-wiring is one `k` and one rendering. Build the
`RequestBuilder` yourself when you need more:

```python
from agentkit import Agent, Prompt, VectorMemory
from agentkit.adapters.vector import InMemoryVector
from agentkit.capabilities import RequestBuilder
from agentkit.memory import as_grounder

memory = VectorMemory(vector=InMemoryVector(), name="handbook")

agent = Agent(
    name="support",
    model="claude-sonnet-4-6",
    request_builder=RequestBuilder(
        prompt=Prompt(id="support", version="3", template="Answer from the sources."),
        grounder=as_grounder(
            memory,
            k=8,
            where={"team": "support"},                      # metadata filter
            format=lambda items: "\n".join(                 # your own rendering
                f"{i}. {it.content}  ({it.source})" for i, it in enumerate(items, 1)
            ),
        ),
        # Re-retrieve on every `build()` — i.e. every user turn of a
        # conversation that reuses one WorkingContext, NOT every step of a
        # tool loop. Costs a prefix cache miss per turn; worth it when each
        # question needs different evidence.
        reground_every_turn=True,
    ),
)
```

`request_builder=` overrides `prompt=` and `memory=` entirely — it is
the "I'll take it from here" seam.

## Composing sources

The backends are small and they nest:

| | What it is |
|---|---|
| `VectorMemory` | a `VectorPort` — the canonical RAG case |
| `FileMemory` | the read side of a file tree the agent maintains |
| `JournalMemory` | the run's own mutation journal, made queryable |
| `ScratchpadMemory` | notes the agents wrote this run |
| `ToolMemory` | any `Tool` adapted into a source (a search API, say) |
| `CompositeMemory` | fan out across several, merge and rerank |
| `SequentialMemory` | try in order, stop at the first hit (cache → fallback) |
| `ScopedMemory` | fail-loud tenant guard around any of the above |
| `CachedMemory` | TTL cache in front of any of the above |
| `CompactedMemory` | shrink each item's content through a `Compactor` |

```python
from agentkit import CompositeMemory, InMemoryFiles, VectorMemory
from agentkit.adapters.vector import InMemoryVector
from agentkit.memory import CachedMemory, FileMemory, score_sort_rerank

everything = CompositeMemory(
    [
        CachedMemory(VectorMemory(vector=InMemoryVector(), name="docs"), ttl_seconds=60.0),
        FileMemory(files=InMemoryFiles(), name="runbooks"),
    ],
    reranker=score_sort_rerank,   # a `Reranker`: (query, items, *, k) -> items
    name="all",
)
print(everything.name, [s.name for s in everything.sources])
```

`CompositeMemory` queries its sources concurrently and merges the
results; the `Reranker` decides the final order and cut. The default
sorts by the backends' own scores, which is only meaningful when the
backends score comparably — swap in your own for a cross-encoder.

## What bites people

- **`ScopedMemory` demands BOTH `org_id` and `domain_id` by default.**
  A `Scope(org_id="acme")` with no `domain_id` fails the default check
  with `PermissionError`, before the inner source is touched. That is
  the intent — a half-specified tenant is how cross-tenant leaks
  happen — but it surprises people who wrapped a source and had a
  working query stop working. Pass your own `enforce=` callable for a
  different policy.
- **`MemoryItem.metadata` is frozen.** `item.metadata["path"] = p`
  raises. Backends pass metadata *through* when they rebuild items, so
  a reranker stamping a score onto one item could otherwise land it on
  a copy another source still holds. The migration is one line:
  `dataclasses.replace(item, metadata={**item.metadata, "path": p})`.
- **`metadata["id"]` is consumed at write time.** `VectorMemory.write`
  pops `id` out of the metadata to use as the chunk id (falling back to
  a fresh uuid). It will not be in `metadata` when the item comes back.
  Write it deliberately if you want stable, idempotent upserts —
  re-writing the same id replaces rather than duplicating.
- **`InMemoryVector` is not an embedding model.** It is
  cosine-over-term-frequency: deterministic, dependency-free, and good
  enough to test wiring and tenant isolation with. Scores from it are
  not comparable to a real embedding store's, and it finds nothing for
  a paraphrase that shares no words. Production is `PgVectorStore`
  (extra: `arc-agentkit[postgres]`) or your own `VectorPort`.
- **An empty grounding block produces no message at all.** The grounder
  returning `""` means "nothing for this task", and `RequestBuilder`
  omits the block rather than injecting an empty header. So a prompt
  that says "answer only from the grounding block" needs to also say
  what to do when there isn't one.
- **Memory is not a tool, and the difference matters.** Memory is what
  the *cognition* fetches around a call; a tool is what the *model*
  decides to call mid-loop. Wrap a source as a tool (or a tool as a
  source, with `ToolMemory`) when you want the other behaviour — but
  pick deliberately, because "the model chose to search" and "we always
  search" produce very different bills and very different failures.

## Related

- [Give an agent a tool](define-a-tool.md) — the other way an agent
  reaches outside itself.
- [Keep a long conversation in budget](compact-a-long-conversation.md)
  — grounding grows the prompt; this is what bounds it.
- [Consume MCP tools from an agent](mcp-tools.md) — `mcp_resources()`
  returns a `MemorySource` like any other.
