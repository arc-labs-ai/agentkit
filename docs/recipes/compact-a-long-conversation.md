# How do I stop a long conversation blowing the context window?

A conversation that keeps going gets more expensive on every turn and
eventually stops fitting. Compaction is what you put in the way: a
policy that shrinks the transcript before it goes to the model.

## When you'd want this

Every turn resends the whole transcript. So turn 40 of a support
conversation costs roughly forty times turn 1, for the same answer —
and one turn later the provider returns a context-length error and the
run dies holding all of it.

Neither failure announces itself early. The bill creeps; the wall
arrives at once, usually on your longest and most valuable session.

Two knobs, and they solve different halves:

- **`compaction()` middleware** — bounds every chat request on the
  wire, whatever produced it. This is the safety net.
- **`RequestBuilder(compactor=...)`** — bounds one agent's working
  context as it accumulates. This is the policy.

Both take the same `Compactor`, so you pick the shape once.

## Working code

```python
"""Runs offline. Measures what actually goes on the wire, with and without."""

import asyncio

from agentkit import ChatRequest, Message, Scope, SlidingWindowCompactor
from agentkit.context.tokens import estimate_message_tokens
from agentkit.middlewares import compaction, meter, tracing
from agentkit.runtime import Budget, Invoker, RunContext, Services
from agentkit.testing import FakeLLM


async def main() -> None:
    history = [Message("system", "You are a terse assistant.")]
    for i in range(40):
        history.append(Message("user", f"question {i} " + "x" * 200))
        history.append(Message("assistant", f"answer {i} " + "y" * 200))
    print("transcript:", len(history), "messages,", estimate_message_tokens(history), "tokens")

    seen: dict[str, int] = {}

    def responder(*, system: str, user: str, model: str) -> str:
        seen["tokens"] = (len(system) + len(user)) // 4
        return "ok"

    for label, chain in (
        ("no compaction ", [tracing(), meter()]),
        # compaction sits AHEAD of meter, so the meter counts the tokens
        # that were really sent rather than the ones we started with.
        ("sliding window", [tracing(), compaction(SlidingWindowCompactor(keep_recent=6)), meter()]),
    ):
        ctx = RunContext(
            "run-1",
            Scope(),
            budget=Budget(),
            services=Services(invoker=Invoker(llm=FakeLLM(responder), chat_middleware=chain)),
        )
        await ctx.invoker.chat(ChatRequest(messages=history, model="claude-sonnet-4-6"), ctx)
        print(f"{label}: ~{seen['tokens']} tokens on the wire")


asyncio.run(main())
```

Output:

```text
transcript: 81 messages, 4545 tokens
no compaction : ~4451 tokens on the wire
sliding window: ~339 tokens on the wire
```

## Picking a compactor

Four ship. All of them preserve a leading `system` message verbatim —
that is the cache-stable prefix, and rewriting it would invalidate the
provider's KV cache from that token onward.

| Compactor | What it does | Costs |
|---|---|---|
| `SlidingWindowCompactor(keep_recent=N)` | keep the system message and the last N | nothing; drops the middle outright |
| `TruncationCompactor(max_tokens=, keep_recent=)` | drop oldest until under budget | nothing; the middle is gone |
| `SummarizationCompactor(summarizer=, ...)` | replace the middle with an LLM summary | one extra model call when it fires |
| `ImportanceFilteringCompactor(filterer=, ...)` | ask a model which turns matter, keep those | one extra model call when it fires |

The first two are free and lossy; the second two cost a call and keep
the gist. Point the LLM-backed ones at a **cheap** model — the whole
point is spending a little to save a lot:

```python
import asyncio

from agentkit import Message, SummarizationCompactor
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    compactor = SummarizationCompactor(
        summarizer=FakeLLM("Earlier: the user asked about 26 unrelated things."),
        model="claude-haiku-4-5",   # a cheap model, not the one doing the work
        max_tokens=800,             # below this, compact() is a no-op
        keep_recent=4,              # verbatim tail
    )
    transcript = [Message("system", "Be terse.")] + [
        Message("user" if i % 2 == 0 else "assistant", f"m{i} " + "z" * 400) for i in range(30)
    ]
    for m in await compactor.compact(transcript, make_test_ctx(llm=FakeLLM("x"))):
        print(f"  {m.role:9} {m.content[:48]}")


asyncio.run(main())
```

Output:

```text
  system    Be terse.
  system    [Summary of 26 earlier turns]
Earlier: the user 
  user      m26 zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
  assistant m27 zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
  user      m28 zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
  assistant m29 zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
```

## Where to put it

**In the chat chain**, ahead of `meter()`:

```python
from agentkit import SlidingWindowCompactor
from agentkit.middlewares import compaction, meter, retry, tracing
from agentkit.runtime import Invoker
from agentkit.testing import FakeLLM

invoker = Invoker(
    llm=FakeLLM("ok"),
    chat_middleware=[
        tracing(),
        compaction(SlidingWindowCompactor(keep_recent=20)),  # shrink first…
        meter(),                                             # …then count
        retry(),
    ],
)
```

Order matters for a reason you can measure: put `meter()` first and it
charges the budget for tokens that were never sent.

**On one agent**, through its `RequestBuilder`:

```python
from agentkit import Agent, Prompt, SlidingWindowCompactor
from agentkit.capabilities import RequestBuilder

agent = Agent(
    name="support",
    model="claude-sonnet-4-6",
    request_builder=RequestBuilder(
        prompt=Prompt(id="support", version="2", template="Answer support questions."),
        compactor=SlidingWindowCompactor(keep_recent=4),
    ),
)
```

The builder compacts **after** appending the new turn and only ever
touches `WorkingContext.messages` — the tail. The prefix (system prompt,
grounding, output schema) is read, never rewritten.

## Counting tokens

`ApproxTokenCounter` is the default: characters ÷ 4, no dependency, and
it counts tool calls as well as text. `TiktokenCounter` is exact for
OpenAI-family models and needs `tiktoken` installed. Both satisfy the
same `TokenCounter` protocol and both are `async`:

```python
import asyncio

from agentkit import ApproxTokenCounter, Message, WorkingContext


async def main() -> None:
    print(await ApproxTokenCounter().estimate([Message("user", "hello world")]))

    wc = WorkingContext(limit=8_000)   # the ceiling this context reports against
    wc.append(Message("user", "hello"))
    print(wc.size(), await wc.tokens())


asyncio.run(main())
```

## Gotchas

- **Compaction is not on by default.** There is no hidden chain to
  override — the app owns the middleware list. An agent that has never
  been given a compactor will grow until the provider refuses.
- **A tool result is never orphaned from its call.** Every built-in
  compactor walks the kept-tail boundary *backwards* over leading `tool`
  messages, because a `tool` message whose originating assistant
  tool-call was dropped is a provider 400. So `keep_recent=4` may
  actually keep five or six messages. That is deliberate.
- **An empty result is refused.** If a compactor returns `[]`, the
  middleware keeps the original messages and drops a
  `context.compaction_rejected` event on the span. Sending zero messages
  is a 400 or a hallucination depending on the provider, and a
  compaction is an optimisation — it must not be able to break the run.
- **The middleware decides *whether*, and it uses its own estimator.**
  `compaction()` measures the request with the shared approximate
  counter before asking the compactor to act. A `TiktokenCounter` wired
  elsewhere does not change that decision.
- **Summarising loses specifics first.** An order number, a file path,
  a chosen option — exactly the things a later turn needs. If a run
  depends on details from turn 3, write them to
  `WorkingContext.scratchpad` (or a `MemorySource`) rather than trusting
  a summary to carry them.
- **`max_tokens` is a floor for acting, not a ceiling on the result.**
  `compact()` is a no-op below `max_tokens`; above it, the compactor
  does its one pass and returns. `SummarizationCompactor` does not loop
  until it fits, so a `keep_recent` tail that is on its own bigger than
  the window still doesn't fit.

## Related

- [Cap spend with Budget and Quota](spend-budget-and-quota.md) — the
  other half of the cost story; compaction reduces the tokens, the
  budget stops the run.
- [Make an agent answer from my documents](ground-with-memory.md) —
  grounding grows the prefix, which compaction deliberately never
  touches.
- [Write a custom middleware](custom-middleware.md) — `compaction()` is
  a `BaseMiddleware` transform; writing your own is the same shape.
