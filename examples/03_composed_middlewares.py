"""Example 03 — composing the chat middleware chain.

The `Invoker` walks a user-supplied list of middlewares to a terminal LLM
call. Order matters: the first entry is OUTERMOST (sees the request first,
sees the response last); the last entry sits closest to the LLM.

This example wires:

    chat_middleware = [
        tracing(),       # OUTERMOST: opens one span for the whole call,
                         # so retries + coercion time show up as child events
                         # on the same span. Nothing observability-side
                         # should be BELOW tracing, or those events land on
                         # the wrong span.
        retry(...),      # MIDDLE: re-invokes on pre-stream failures
                         # (rate-limits, connect errors, `CircuitOpen`).
                         # Once the LLM has yielded the first delta the
                         # retry commits and streams through — you can't
                         # un-send tokens. Sits inside tracing so each
                         # retry attempt is a `retry.attempt` event on the
                         # active span.
        output_coerce(), # INNERMOST: collects the assembled stream and runs
                         # the SchemaAdapter's `parse` on it, landing the
                         # typed object on `LLMResult.parsed`. Sits closest
                         # to the LLM because it must see the raw content
                         # BEFORE anything above transforms it. When parsing
                         # fails it re-raises `OutputCoercionError` — the
                         # retry above catches it and can reflect the
                         # diagnostics back to the model on the next attempt.
    ]

The request flows top-to-bottom (tracing -> retry -> output_coerce -> LLM);
the response bubbles bottom-to-top.

Run:
    uv run python examples/03_composed_middlewares.py
"""

from __future__ import annotations

import asyncio

from agentkit import ChatRequest, Message
from agentkit.middlewares import output_coerce, retry, tracing
from agentkit.testing import FakeLLM, make_test_ctx


async def main() -> None:
    # This FakeLLM fails once pre-stream, then succeeds — enough to exercise
    # the retry middleware without cluttering the output.
    llm = FakeLLM(
        '{"answer": 42}',
        fail_times=1,
        fail_exc=TimeoutError("simulated transient upstream timeout"),
    )

    ctx = make_test_ctx(
        llm=llm,
        chat_middleware=[
            tracing(),
            retry(max_attempts=3, sleep=lambda _s: asyncio.sleep(0)),  # no real sleep in tests
            output_coerce(),
        ],
    )

    # We drive the invoker directly here to keep the example about the
    # middleware chain rather than about the Agent surface. An Agent would
    # produce the exact same chain of calls under the hood.
    req = ChatRequest(
        messages=[Message("user", "Give me a JSON answer")],
        model="gpt-4o-mini",
    )
    result = await ctx.invoker.chat(req, ctx)

    print(f"llm attempts:  {llm.calls}       # 2 = one failure + one success")
    print(f"content:       {result.content!r}")
    print(f"finish:        {result.finish_reason}")
    print(f"usage.cost:    {result.usage.cost_usd}")


# Expected output:
#   llm attempts:  2       # 2 = one failure + one success
#   content:       '{"answer": 42}'
#   finish:        stop
#   usage.cost:    0.0001


if __name__ == "__main__":
    asyncio.run(main())
