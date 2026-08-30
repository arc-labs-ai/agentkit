# How do I test an agent without calling a model?

An agent is mostly ordinary code — a prompt, some tools, some wiring —
with one non-deterministic part. Swap that one part for a script and
the rest tests like anything else: fast, offline, and the same answer
every time.

## When you'd want this

The alternative is a suite that calls a real provider, and it fails in
all four ways at once. It is slow (seconds per assertion), it costs
money on every CI run, it is flaky in a way that trains people to
re-run red builds, and it cannot make the interesting cases happen —
you cannot ask a real model to please return malformed JSON, or to
please time out.

`agentkit.testing` gives you the seam. `FakeLLM` scripts the model;
`make_test_ctx` builds a **real** `RunContext` around it, so everything
between your code and the provider is the production code path.

!!! note "`agentkit.testing` is deliberately not re-exported at the top level"
    There is no `from agentkit import FakeLLM`. Test doubles live behind
    `agentkit.testing` precisely so a production wiring cannot pick one
    up by accident.

## Working code

```python
"""A real pytest module. Run: uv run pytest test_support_agent.py"""

import pytest

from agentkit import Agent, Budget, tool
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.types import ToolCall
from agentkit.middlewares import meter
from agentkit.testing import FakeLLM, FakeTool, RecordingTracer, Turn, make_test_ctx


@tool(side_effecting=False, idempotent=True)
async def lookup_order(order_id: str) -> str:
    """Look up an order by its id and return its status line.

    Used by the support agent whenever an order number appears."""
    return f"order {order_id}: shipped"


def support_agent(tools: list) -> Agent:
    return Agent(
        name="support",
        model="claude-sonnet-4-6",
        prompt="Answer order questions. Use the tools; never guess a status.",
        cognition=ReActCognition(tools=tools),
    )


@pytest.mark.asyncio
async def test_it_answers_from_the_tool_not_from_memory() -> None:
    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall(id="c1", name="lookup_order", arguments={"order_id": "A-771"}),)),
        Turn(content="Order A-771 has shipped."),
    ])
    result = await support_agent([lookup_order]).run("Where is A-771?", make_test_ctx(llm=llm))

    assert result.output == "Order A-771 has shipped."
    assert result.stop_reason == "complete"
    assert llm.calls == 2          # one turn to call the tool, one to answer


@pytest.mark.asyncio
async def test_it_passes_the_order_id_through_verbatim() -> None:
    probe = FakeTool(name="lookup_order", responder="order A-771: shipped")
    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall(id="c1", name="lookup_order", arguments={"order_id": "A-771"}),)),
        Turn(content="Shipped."),
    ])
    await support_agent([probe]).run("Where is A-771?", make_test_ctx(llm=llm))

    assert probe.calls == [{"order_id": "A-771"}]   # every call, in order


@pytest.mark.asyncio
async def test_a_transient_provider_error_is_not_swallowed() -> None:
    llm = FakeLLM("never reached", fail_times=1)
    with pytest.raises(TimeoutError):
        await support_agent([lookup_order]).run("Where is A-771?", make_test_ctx(llm=llm))


@pytest.mark.asyncio
async def test_the_run_is_charged_to_the_budget() -> None:
    budget = Budget(max_cost_usd="1.00")
    # meter() is what charges the budget. Without it the ledger stays at
    # zero and this assertion is the one that tells you.
    ctx = make_test_ctx(llm=FakeLLM("Shipped."), budget=budget, chat_middleware=[meter()])
    await support_agent([lookup_order]).run("Where is A-771?", ctx)

    assert budget.calls == 1
    assert budget.spent() > 0


@pytest.mark.asyncio
async def test_the_agent_span_is_opened() -> None:
    tracer = RecordingTracer()
    ctx = make_test_ctx(llm=FakeLLM("Shipped."), trace=tracer)
    await support_agent([lookup_order]).run("Where is A-771?", ctx)

    assert "invoke_agent" in [s.name for s in tracer.spans]
```

All five pass in well under a second, with no network and no key.

## The two pieces

**`FakeLLM`** is an `LLMPort`. Several ways to say what it returns:

```python
from agentkit.kernel.types import ToolCall, Usage
from agentkit.testing import FakeLLM, Turn

FakeLLM("always this string")                       # one fixed answer
FakeLLM({"refund": "5 days", "ship": "Tuesday"})    # first key found in the user text wins
FakeLLM(lambda *, system, user, model: user.upper())  # a callable, for assertions on the prompt

FakeLLM.script([                                    # a multi-step tool loop
    Turn(tool_calls=(ToolCall(id="c1", name="search", arguments={"q": "octopus"}),)),
    Turn(content="Octopuses are clever.", usage=Usage(120, 45, 0.002)),
])

FakeLLM("...", fail_times=2)                        # raise twice, then succeed — for retry tests
FakeLLM("...", fail_exc=ValueError("bad key"))      # choose the exception
FakeLLM("...", delay=0.05)                          # for timeout / cancellation tests
```

It streams in 8-character chunks, so partial-output and
delta-assembly paths are genuinely exercised rather than
short-circuited — `chat()` is `assemble_deltas(...)` over its own
`stream()`. `llm.calls` counts invocations. A script that runs out
raises `ScriptExhausted`, naming how many turns it had and which turn
was asked for: a loop that goes one iteration further than you expected
is the finding, and it used to be swallowed by replaying the last
`Turn`. Pass `FakeLLM.script([...], repeat_last=True)` when the
unbounded loop is deliberate and a ceiling is what ends the run.

**`make_test_ctx(...)`** builds a real `RunContext`, not a stub. Every
knob is a keyword:

| Keyword | For testing |
|---|---|
| `llm=` / `invoker=` | the model seam (`llm=` is wrapped in an `Invoker` for you) |
| `chat_middleware=` / `tool_middleware=` | metering, retry, compaction, your own |
| `budget=` / `meters=` | ceilings and quotas |
| `scope=` | tenant isolation |
| `checkpointer=` / `store=` / `vector=` | durability and retrieval |
| `asker=` | human-in-the-loop: setting it makes a gated call **park** instead of suspend |
| `trace=` / `observer=` | assertions on spans and observations |
| `cancel=` | cooperative cancellation |
| `autonomy=` | `"auto"` / `"gated"` / `"manual"` |
| `correlation_id=` | the run id, for resume tests |

Leaving `llm=` and `invoker=` unset gives you a context with
`services.invoker = None` — fine for testing a capability that never
calls a model, and an `AttributeError` the moment something does.

## The other doubles

| Double | Records / does |
|---|---|
| `FakeTool` | `Tool`-shaped; `.calls` is every args dict it received, in order |
| `FakeMemory` | `MemorySource`-shaped, with canned items |
| `FakeClock` | a `ClockPort` you advance by hand — deadlines without `sleep` |
| `FakeFetch` / `FakeSearch` | canned HTTP / search responses |
| `FakeCompactor` / `FakeGrounder` | capability stand-ins for `RequestBuilder` tests |
| `RecordingTracer` | a `TracePort` that keeps every `RecordingSpan` on `.spans` |
| `FakeCtx` | a minimal ctx that records spans, when you don't want a real one |

`RecordingTracer` is the one to reach for when the behaviour you care
about is observability: span names, the attributes the `tracing()`
middleware stamped, the events a tool dropped.

## What to actually assert

Testing "the model said the right thing" is testing the model. The
things worth pinning are the ones your code decides:

- **which tool got called, with which arguments** — `FakeTool.calls`;
- **how many model calls a task took** — `llm.calls`, which is how you
  catch a loop that silently doubled;
- **the stop reason** — `complete` vs `max_iterations` vs
  `budget_exhausted` vs `suspended` are four different bugs;
- **the ledger** — `budget.spent()`, `budget.calls`, `budget.usage`;
- **the typed output** — `result.parsed`, and the repair path when the
  first response is malformed (script a `Turn` with bad JSON, then a
  good one);
- **tenant isolation** — run the same query under two `Scope`s and
  assert the second sees nothing;
- **suspend and resume** — `result.is_suspended`, then `agent.resume`
  from a freshly built agent and context.

## What bites people

- **`make_test_ctx` is not a fake.** It builds a real `RunContext` with
  real `Services` and a real `Invoker`. That is the point: a test that
  passes tells you the production wiring works. `FakeCtx` (a
  span-recording stub) and `NullCtx` (a production null object that
  records nothing) are different things for different jobs.
- **No middleware is wired by default.** `chat_middleware` defaults to
  `()`. So a budget assertion fails at zero without `meter()`, a
  `partial_output` assertion is `None` forever without
  `output_coerce()`, and a retry test never retries without `retry()`.
  Wire the chain the test is about.
- **`fail_times` raises before the first delta.** That makes it a
  *pre-stream* failure, which is the one `retry()` can genuinely
  re-invoke. A provider that dies mid-stream is a different failure and
  needs a different double.
- **A `FakeLLM` script is per instance, not per test.** `llm.calls` and
  the turn index keep advancing. Build a fresh one per test unless the
  carry-over is what you are testing.
- **Scripted `Turn`s are consumed by `stream()`, and everything goes
  through `stream()`.** Compactors and grounders that call the same
  `FakeLLM` will eat turns your agent was going to use. Give them their
  own instance.
- **An `async def` test needs `pytest-asyncio`, and how you mark it
  depends on the mode.** This repo runs `asyncio_mode = "strict"`, so
  every async test carries `@pytest.mark.asyncio` — as above. Under
  `asyncio_mode = "auto"` you can drop the marker. Under neither,
  pytest skips the test with a warning and the suite reports green.

## Related

- [Give an agent a tool](define-a-tool.md) — `FakeTool` is how you
  assert on what a tool received.
- [Cap spend with Budget and Quota](spend-budget-and-quota.md) — what
  `budget.spent()` means, and why `meter()` has to be wired.
- [Stream a typed object](stream-typed-output.md) — testing partials
  needs `output_coerce()` in the chain.
