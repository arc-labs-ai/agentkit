# How do I stream a typed object as it's being generated?

## When you'd want this

You declared `output=MyModel` on an agent and you want to render it
**while the model is still writing it** — a title that appears before
the body finishes, a list of steps that grows item by item, a form
that fills in field by field. Waiting for `AgentResult.parsed` means
waiting for the whole response; the UI sits blank for the full
generation.

The framework parses partial structured output on every delta. This
recipe is how you receive it: `StreamEvent.partial_output`, off
`Agent.stream`, with no private attribute access.

!!! warning "You must wire `output_coerce()` into the chat chain"
    The partial pipe lives in the `output_coerce()` middleware. With
    `output=` set but that middleware missing, `AgentResult.parsed`
    still works — the cognition runs the parser itself — and
    `partial_output` is silently `None` forever. Nothing looks broken.
    The Agent emits a one-shot `UserWarning` when it spots that
    wiring; don't filter it.

## Working code

```python
"""Requires ANTHROPIC_API_KEY in the environment."""

import asyncio

from pydantic import BaseModel

from agentkit import Agent, Scope
from agentkit.adapters.llm import resolve_llm
from agentkit.middlewares import output_coerce, tracing
from agentkit.runtime import Invoker, RunContext, Services


class Article(BaseModel):
    title: str
    body: str


async def main() -> None:
    invoker = Invoker(
        llm=resolve_llm("claude-sonnet-4-6"),
        # output_coerce() is what runs the tolerant partial parse on each
        # delta. Without it there are no partials.
        chat_middleware=[tracing(), output_coerce()],
    )
    ctx = RunContext("run-1", Scope(), services=Services(invoker=invoker))
    agent = Agent(
        name="writer",
        model="claude-sonnet-4-6",
        prompt="Write a short article. Return JSON matching the schema.",
        output=Article,
    )

    seen_title = False
    async for ev in agent.stream("Write about octopus cognition.", ctx):
        partial = ev.partial_output
        if partial is None:
            continue

        # REQUIRED FIELDS MAY BE UNSET. The partial is built through the
        # type's bypass-init path, so plain attribute access can raise.
        if not seen_title and "title" in partial.model_fields_set:
            print(f"# {partial.title}")
            seen_title = True
        if "body" in partial.model_fields_set:
            print(f"\r{len(partial.body)} chars…", end="")

    # The strict, fully-validated object is still on the final result.
    result = await agent.run("Write about octopus cognition.", ctx)
    print(f"\n\nfinal: {result.parsed!r}")


asyncio.run(main())
```

## What's happening

`output_coerce()` sees each text delta, appends it to a running
buffer, and calls the adapter's tolerant `partial_parse` on the whole
buffer. When the result differs from the last one it stamps it onto
`Delta.partial`. Both chat cognitions forward that verbatim onto
`StreamEvent.partial_output`. The cognition neither parses nor
interprets it — it is a pass-through, which is why an agent with no
output schema is completely unaffected.

## The consumer contract

Three properties, each of which will bite you if you assume otherwise.

**Required fields may be unset.** The partial is constructed via
`model_construct` (Pydantic) or `object.__new__` + `setattr`
(dataclass / attrs), so a missing required field is genuinely absent.
Gate on `model_fields_set` for Pydantic, or `getattr(obj, name, None)`
for the others. `partial.body` on an object that hasn't reached `body`
yet raises `AttributeError`.

**`None` does not mean "the object went away".** The middleware only
stamps when the partial *changed*, so consecutive `message_delta`
events mid-stream can carry `None`. Hold the last non-`None` value.

**String fields grow character by character.** An unterminated
`"Shi` yields `title="Shi"`. Values are prefix-consistent across
partials, so appending is safe, but don't treat an early value as
final.

## Gotchas

**The `final` event carries no partial.** `output_coerce` emits the
strict `parsed` value on a synthetic post-terminal delta with no text,
so no partial is computed for it. Read `ev.result.parsed` on the
`final` event; `ev.partial_output` there is legitimately `None`.

**Don't confuse it with `AgentResult.partial`.** That is a `bool`
meaning "this run terminated incompletely". `StreamEvent.partial_output`
is the typed object. The names are deliberately different for exactly
this reason.

**Repair still works with the middleware wired.** `output_coerce`
strict-parses at end-of-stream and re-raises on failure; the
cognitions catch that and route it into the existing reflect-and-retry
loop, so a malformed first response is re-prompted rather than
aborting the run. (Before this was fixed, adding `output_coerce()` to
the chain broke repair — which is why older examples omitted it.)

**Partial parsing is O(n²) in response length.** The middleware
re-parses the whole accumulated buffer on every text delta. For very
long structured outputs that is real CPU. If it shows up in a profile,
sample the partials rather than rendering every one.

## Related

- [Cap spend with Budget and Quota](spend-budget-and-quota.md)
- [Concepts: Kernel](../concepts/kernel.md) — `Delta`, `StreamEvent`,
  `assemble_deltas`
