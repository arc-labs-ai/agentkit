"""Example 05 — streaming a typed object while the model is still writing it.

Demonstrates:

- `output=Article` declares a Pydantic output schema on the `Agent`. Its JSON
  Schema goes into the cache-stable prompt prefix and `AgentResult.parsed`
  carries the strict, fully-validated object at the end.
- `output_coerce()` in the chat chain is what makes PARTIALS flow. On every
  text delta it runs the adapter's tolerant `partial_parse` over the buffer
  so far and, when the result changed, stamps it onto `Delta.partial`. Both
  chat cognitions forward that verbatim onto `StreamEvent.partial_output`.
  Without this middleware `AgentResult.parsed` still works and
  `partial_output` is silently `None` forever — so the `Agent` emits a
  one-shot warning when it sees that wiring.
- `StreamEvent.partial_output` is the in-progress object, off the PUBLIC
  `Agent.stream`. No `_resolve_request_builder()`, no `_output_adapter`, no
  hand-built `ChatRequest`, no direct `ctx.invoker.stream`.

The consumer contract, which this script demonstrates rather than describes:

- REQUIRED FIELDS MAY BE UNSET. The partial is built through the type's
  bypass-init path, so `partial.body` can raise `AttributeError`. Always gate
  on `model_fields_set` (Pydantic) or `getattr(obj, name, None)` (dataclass /
  attrs).
- `None` does NOT mean "the object went away" — the middleware only stamps
  when the partial changed, so consecutive deltas can carry `None`.
- String fields grow character by character and are prefix-consistent, so
  appending is safe but an early value is not final.

Runs against `FakeLLM` — no API key needed. Swap in a real provider with
`resolve_llm("claude-sonnet-4-6")` (reads `ANTHROPIC_API_KEY`).
"""

from __future__ import annotations

import asyncio

from agentkit import Agent
from agentkit.middlewares import output_coerce, tracing
from agentkit.testing import FakeLLM, make_test_ctx

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover — the example needs one optional dep
    raise SystemExit("this example needs pydantic: uv sync --all-extras") from None


class Article(BaseModel):
    """Two fields, one short and one long — so the title is final while the
    body is still arriving. That asymmetry is the whole reason to stream a
    partial rather than wait for the result."""

    title: str
    body: str


ARTICLE_JSON = (
    '{"title": "Eight Arms, Nine Brains", '
    '"body": "An octopus distributes most of its neurons into its arms, '
    'so each one solves problems with a striking degree of independence."}'
)


async def main() -> None:
    ctx = make_test_ctx(
        llm=FakeLLM(ARTICLE_JSON),
        # output_coerce() is the partial pipe. Drop it and this example still
        # prints a final result — it just never prints a single partial.
        chat_middleware=[tracing(), output_coerce()],
    )
    agent = Agent(
        name="writer",
        model="claude-sonnet-4-6",
        prompt="Write a short article. Return JSON matching the schema.",
        output=Article,
    )

    print("── streaming ───────────────────────────────────────────")
    title_shown = False
    async for ev in agent.stream("Write about octopus cognition.", ctx):
        partial = ev.partial_output
        if partial is None:
            continue  # not "gone" — just unchanged since the last stamp

        fields = partial.model_fields_set
        if not title_shown and "title" in fields and partial.title.endswith("Brains"):
            print(f"title settled early: {partial.title!r}")
            title_shown = True
        if "body" in fields:
            print(f"  body so far: {len(partial.body):>3} chars", end="\r")

    print("\n\n── final ───────────────────────────────────────────────")
    result = await agent.run("Write about octopus cognition.", ctx)
    print(f"stop_reason : {result.stop_reason}")
    print(f"parsed      : {type(result.parsed).__name__}")
    print(f"title       : {result.parsed.title}")
    print(f"body        : {result.parsed.body[:60]}…")

    print("\n── the negative: no schema, nothing changes ────────────")
    plain = Agent(name="plain", model="claude-sonnet-4-6")
    events = [ev async for ev in plain.stream("hi", make_test_ctx(llm=FakeLLM("hello there")))]
    assert all(ev.partial_output is None for ev in events)
    print(f"{len(events)} events, every partial_output is None — additive, as promised")


if __name__ == "__main__":
    asyncio.run(main())
