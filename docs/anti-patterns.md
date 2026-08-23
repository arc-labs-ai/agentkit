# Anti-patterns

Fifteen mistakes we see over and over. Each is a **Don't** with a short
"why this breaks", followed by a **Do** with the fix. Read this once
before you write your first non-trivial agent; skim it again the day
after you deploy.

---

### Don't: mix up the argument order — `agent.run(ctx, task)`

This is the number-one first-day bug. The signature is
`agent.run(task, ctx)`. Passing a `RunContext` where a `str` is
expected will either raise `TypeError` immediately or (worse) pass a
context repr into the model's user turn.

### Do: `await agent.run(task, ctx)` — task first, ctx second.

```python
result = await agent.run("Brief me on octopus cognition.", ctx)
```

Same for `agent.stream(task, ctx)`. `agent.resume` is the odd one out
— `agent.resume(run_id, decisions, ctx)`.

---

### Don't: reuse one `RunContext` across concurrent agent runs

`RunContext.budget` and `RunContext.cancel` are shared references.
Firing two `agent.run(task, ctx)` calls in parallel over the *same*
top-level ctx means both contend the budget's async lock, and a
cancel on one cancels the other. That's fine for a coordinator (it
*wants* subtree cancel) — but not for two independent user requests.

### Do: one top-level `RunContext` per external request. For a child of the same run, use `ctx.child()`.

```python
# One top-level per request:
ctx_req1 = RunContext("req-1", scope, budget=Budget(max_cost_usd=1.0), services=svc)
ctx_req2 = RunContext("req-2", scope, budget=Budget(max_cost_usd=1.0), services=svc)
await asyncio.gather(agent.run(task_a, ctx_req1), agent.run(task_b, ctx_req2))

# Coordinated fan-out uses ctx.child(), which SHARES budget + cancel by design:
await run_agents([(a, "sub-1"), (b, "sub-2")], parent_ctx)  # each runs under parent_ctx.child()
```

---

### Don't: put a side-effecting tool under `autonomy="auto"` and hope

`"auto"` gates *only* tools that explicitly opt in with
`requires_approval=True`. A `side_effecting=True` tool is NOT gated
under `"auto"`. If you meant "ask a human before this mutates the
world," you meant `"gated"`.

### Do: set `RunContext.autonomy="gated"` on any run that has side-effecting tools.

```python
ctx = RunContext(
    correlation_id="run-42",
    scope=scope,
    autonomy="gated",   # every @tool(side_effecting=True) suspends before firing
    services=svc,
)
```

Escalate to `"manual"` if you want the human to approve *every* tool
call, side-effecting or not.

---

### Don't: skip `side_effecting=` on `@tool`

The framework can't guess whether a tool mutates the world, and the
approval-gate + idempotency middleware both depend on the answer.
Forgetting the keyword raises `ToolDefinitionError` **at decoration
time**, not at call time — the error IS the fix.

### Do: state the flag explicitly on every `@tool`.

```python
@tool(side_effecting=False)                     # read-only lookup
async def search(query: str) -> str:
    """Search the web for `query`. Returns bulleted hits."""
    ...

@tool(side_effecting=True, idempotent=False)    # mutates + not safe to retry
async def publish(title: str) -> str:
    """Publish `title` to the team wiki. Not idempotent."""
    ...
```

`idempotent=True` is independent — set it on any tool the retry
middleware can safely re-invoke.

---

### Don't: import cognitions from the top-level `agentkit` package

They aren't re-exported there. `from agentkit import ReActCognition`
gives you `ImportError`. This is deliberate — cognitions are one
family among many and the top-level package is already dense.

### Do: `from agentkit.agents.cognition import ReActCognition`

```python
from agentkit.agents.cognition import (
    SingleCallCognition,
    ReActCognition,
    CoordinatorCognition,
    ClaudeCliCognition,
)
```

The [cheatsheet](cheatsheet.md#cognition) shows the full menu.

---

### Don't: mutate `Suspended.pending`

`Suspended` is a frozen dataclass and `.pending` is a `tuple` on
purpose. The operator UI renders the pending items and the resume
path threads them back verbatim; a mutable list would let a stray
`.append(...)` desync the two ends of the handshake. Attempting to
reassign raises `FrozenInstanceError`.

### Do: build a fresh `Suspended` if you need to synthesize one in a test; otherwise treat it as read-only.

```python
susp = result.evals["suspended"]
assert isinstance(susp, Suspended)
decisions = {tc.id: "approve" for tc in susp.pending}       # read
final = await agent.resume(susp.run_id, decisions, ctx)     # thread the id back
```

---

### Don't: construct `SummarizationCompactor()` with no arguments

`summarizer=` is a required positional-only parameter — a
`SummarizationCompactor` without an LLM to summarize with is a
no-op with a misleading name. Constructing without it raises
`TypeError` at construction time.

### Do: pass an `LLMPort` — same shape as the one on your `Invoker`.

```python
from agentkit import SummarizationCompactor
from agentkit.adapters.llm import providers
import os

summarizer = providers.claude(api_key=os.environ["ANTHROPIC_API_KEY"],
                              model="claude-3-5-haiku-latest")
compactor = SummarizationCompactor(summarizer=summarizer, max_tokens=8_000)
```

For dep-free compaction, use `SlidingWindowCompactor(keep_recent=10)`
or `TruncationCompactor(max_tokens=12_000)` — no LLM required.

---

### Don't: set `Budget(max_cost_usd=0.01)` and expect it to catch the very first spend

The meter's check is **strict greater-than**: a call whose spend lands
exactly on `max_cost_usd` completes; the *next* call is where the run
halts with `MeterExceeded`. The first spend always passes.

### Do: size the budget so the "next call" ceiling matches what you actually want to stop.

```python
# If a chat call costs ~$0.01 and you want to allow at most 3 completed calls:
Budget(max_cost_usd=0.03, max_calls=3)
```

If you need first-call blocking too, guard your own precheck before
`agent.run(...)`.

---

### Don't: wire `FakeLLM` / `make_test_ctx` into production code

Test doubles under `agentkit.testing.*` exist for your unit tests and
any mock adapters you write. They're deliberately not re-exported
from the top-level `agentkit` package — so a `from agentkit import
FakeLLM` shape can't accidentally pin a fake into a real request
path. Wiring them into a production `Invoker` is a real outage story.

### Do: use a real `LLMPort` in production wiring.

```python
# claude / openai / deepseek / openrouter presets — batteries-included Chat.
from agentkit import claude
async with claude(api_key=..., model="claude-sonnet-4-6") as chat:
    result = await chat("hi", system="Answer briefly.")

# Or ClaudeCliCognition for a local CLI with no API key management.
from agentkit.agents.cognition import ClaudeCliCognition
agent = Agent(name="a", prompt="...", cognition=ClaudeCliCognition(...))

# Or wire your own LLMPort — three methods: stream (an async iterator),
# async def chat, async def complete. Extend as needed.
class MyProviderLLM:
    async def chat(self, req: ChatRequest, ctx: Ctx) -> LLMResult: ...
    async def stream(self, req: ChatRequest, ctx: Ctx) -> AsyncIterator[StreamEvent]: ...
```

`FakeLLM` and friends stay under `agentkit.testing.*` — a boundary
you can grep for.

---

### Don't: import cognitions from the top-level and then wonder why `Cognition` isn't `Protocol`-friendly

`Cognition` IS a `Protocol` — it lives at
`agentkit.agents.cognition.Cognition`. But a random `class MyCog:
async def drive(...): ...` isn't automatically recognised as one
until you import the Protocol and inherit for `mypy --strict`, or
duck-type check at wire-up. Missing the type import silently degrades
your type coverage.

### Do: inherit the Protocol explicitly when you add a cognition.

```python
from agentkit.agents.cognition import Cognition

class MyCognition(Cognition):
    async def drive(self, agent, task, ctx, context):
        ...
        yield final_event
```

Runtime works either way; type-check coverage doesn't.

---

### Don't: run two workers on the same `run_id` against the same `CheckpointPort`

The checkpoint version numbering is monotonic per-run and the port
assumes a single writer. Two workers on the same `run_id` will
collide on `version` — one write will lose, the other will
overwrite a snapshot mid-flight, and `resume(...)` will hydrate
garbage.

### Do: enforce a lease at the driver, one worker per `run_id`.

```python
# Pseudocode — however your infra names it:
lease = await lock_manager.acquire(f"run-lease/{run_id}", ttl=300)
try:
    await agent.run(task, ctx)   # or agent.resume(...)
finally:
    await lease.release()
```

The next release will surface the collision as a clear error; today,
the contract is on you.

---

### Don't: rely on `agent.resume(...)` for a non-`ReActCognition` cognition

Suspend/resume + checkpoint state live only in `ReActCognition`.
Calling `agent.resume(...)` on an agent whose cognition is
`SingleCallCognition`, `CoordinatorCognition`, `ClaudeCliCognition`,
or your own custom cognition raises `RuntimeError` explicitly — the
message names the offending cognition.

### Do: keep `agent.resume(...)` on `ReActCognition` agents; suspend other cognitions at the workflow layer.

```python
# HITL at the tool-loop level:
agent = Agent(name="briefer", prompt="...", cognition=ReActCognition(tools=[...]))

# HITL at the workflow level — a human_gate node in a Workflow suspends
# the workflow (not an individual cognition) with the same Suspended shape.
```

---

### Don't: set an empty `agent.prompt` and expect useful output

`Agent(name="x", model="m")` (no prompt) constructs cleanly and runs,
but the model has nothing to anchor on. You'll get plausible-looking
gibberish that's very hard to debug because the trace shows the
system message as `""`.

### Do: give every agent a purposeful system prompt.

```python
agent = Agent(
    name="briefer",
    model="claude-sonnet-4-6",
    prompt="You are a terse briefer. Cite every claim with a URL + year.",
    cognition=ReActCognition(tools=[search]),
)
```

For anything real, promote the string to a `Prompt(id=..., version=...)`
so template drift shows up as a version bump in the trace and in
`AgentResult.prompt_version`.

---

### Don't: swallow `Cancelled` in a broad `except`

The kernel's `Cancelled` (from `agentkit.kernel.concurrency`) is
raised by `ctx.check_cancelled()` when the token has flipped. A
naive `try: ... except Exception: pass` in a tool or middleware
turns cooperative cancel into "the tool didn't stop". Every safe
point in the tree suddenly no-ops.

### Do: let `Cancelled` propagate — or re-raise it deliberately.

```python
try:
    result = await risky_side_effecting_call()
except Cancelled:
    raise                          # never swallow cancellation
except Exception as exc:
    logger.warning("recovering: %s", exc)
    result = fallback_value
```

Same rule for the middleware chain: any middleware that catches
broadly must re-raise `asyncio.CancelledError`.

---

### Don't: put `meter()` after `retry()` in the chat chain

Middleware order is deterministic and outer-first. `retry` re-invokes
`next` on transient failure. If `retry` sits *outside* `meter`, a
retry loop hides its intermediate attempts from the meter — you'll
overspend by the retry factor without the budget noticing.

### Do: `tracing → compaction → meter → fallback → retry → memoize`.

```python
from agentkit import SlidingWindowCompactor
from agentkit.middlewares import (
    tracing, meter, retry, fallback, compaction,
)

chat_middleware = [
    tracing(),                                          # outermost — one span across the whole call
    compaction(SlidingWindowCompactor(keep_recent=10)), # shrink transcripts before meter counts them
    meter(),                                            # guard + charge every ATTEMPT (including retries)
    fallback(models=["gpt-4o", "gpt-4o-mini"]),         # rewrite request.model on hard failures
    retry(),                                            # re-invoke on transient failures
]
```

`meter` above `retry` means every retried attempt gets charged, which
is what you want: your provider bill counts retries; your budget had
better too.

---

## Related

- [Cheatsheet](cheatsheet.md) — the same primitives, invocation-form.
- [Recipes](recipes/index.md) — the same primitives, problem-form.
- [Concepts › Agents](concepts/agents.md) — the mental model of
  `Agent` / `Cognition` and why the split is load-bearing.
