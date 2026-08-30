# Features agentkit needs to support a CLI-driven coding agent

Written against `arc-agentkit` as installed, after reading every module. The premise is a single constraint
with wide consequences: **the agent is the Claude Code CLI, and the CLI owns its own loop.** Nothing in the
calling application drives turns.

That constraint is well supported in one direction and barely supported in the other. `ClaudeCliCognition`
launches, streams, budgets and records a CLI dispatch, and it does so better than a hand-written subprocess
would. What it does not yet do is let an application **give the CLI anything of its own** — its tools, its
approvals, its guardrails, its sub-agents — or **test one offline**. Every gap below is on that side.

Each section states the failure it prevents, the surface, where it lands in the tree, and the tests that
would make it real. Two are unblocking; three are the difference between a working integration and a
maintained one.

---

## The one-paragraph summary

`ClaudeCliCognition` already carries the knobs — `mcp_config`, `permission_prompt_tool`, `settings`,
`agents`, `tools`, `config_dir`. Every one of them is a **seam agentkit exposes and does not fill**. The
five features below fill them: serve a `ToolRegistry` over MCP, bridge `Asker` to the permission prompt,
generate hook settings so middleware reaches native tools, project `Skill` into CLI sub-agents, and give the
CLI path an offline double. None requires changing the cognition.

---

## F1 · Serve a `ToolRegistry` as an MCP server

**Priority: blocking.** Nothing custom reaches a CLI agent without it.

### The gap

`agentkit.integrations.mcp` is a **client**. It consumes a server and adapts its tools into agentkit
`Tool`s, which is the right thing for an agent that runs its own loop. The CLI is the opposite case: it
*is* the loop, and the only way to hand it a tool is `--mcp-config`, which `ClaudeCliCognition` already
accepts and nothing in agentkit can produce.

Everything needed to describe a tool is already here and already correct — `FunctionTool` derives a
`ToolSchema` from a signature and docstring, `ToolRegistry` holds them, `ToolArgumentError` refuses a bad
call. **Only the transport is missing.**

### The surface

```python
from agentkit.integrations.mcp import serve_registry, McpServerSpec

spec: McpServerSpec = serve_registry(
    registry,                 # ToolRegistry | Sequence[Tool]
    name="engine",            # the server's name; tools appear as mcp__engine__<tool>
    ctx=ctx,                  # the Ctx handed to every tool.run(args, ctx)
    transport="stdio",        # "stdio" | "http"
)

cognition = ClaudeCliCognition(
    mcp_config=(spec.config_path,),
    strict_mcp_config=True,
    tools=("",),              # every tool comes from the server
)
```

`McpServerSpec` carries the written config path, the server name, the tool names as the CLI will see them
(`mcp__engine__run_check`), and an async context manager owning the process or task.

### What it has to get right

- **`ToolSchema.parameters` maps to MCP `inputSchema` unchanged.** Both are JSON Schema. A translation step
  is a second description of one thing and will drift.
- **`ToolArgumentError` becomes an MCP tool error, not a transport error.** The distinction is the whole
  value of that type: a bad call is reflected to the model, which is the only party that can fix it. A
  transport error kills the session instead.
- **`requires_approval` maps to the MCP annotation**, so a tool declaring it still pauses under F2.
- **`side_effecting` and `caps` travel too**, or `RunPolicy`'s Rule-of-Two check silently stops applying
  the moment tools move behind MCP — which is exactly when the check matters most.
- **The config is written to a real path**, because `--mcp-config` takes files or inline JSON and a caller
  should not be assembling either by hand.

### Tests that make it real

- a registry with one tool produces a config the CLI accepts, and the tool appears in the session
- a call with an unexpected argument comes back as a tool error the model can read, and the session survives
- a call with a missing required argument does the same
- the schema the server advertises is `ToolSchema.parameters` byte-for-byte
- `strict_mcp_config=True` with this server leaves exactly these tools and no others
- a tool raising an ordinary exception fails that call and not the session

---

## F2 · Bridge `Asker` / `HumanGate` to `--permission-prompt-tool`

**Priority: blocking for any human-in-the-loop feature.**

### The gap

Every HITL primitive agentkit owns — `Elicitation`, `Decision`, `HumanGate`, `SignalChannel`,
`approval_deadline_s` — hangs off `ReActCognition`. `ClaudeCliCognition` has none of them; a search for
`asker`, `approval` or `human` in that module returns nothing.

**But the seam exists and is already wired.** `permission_prompt_tool` → `--permission-prompt-tool`, which
the CLI calls, as an MCP tool, whenever it needs approval. That is the same shape as `HumanGate`: an action
arrives, somebody decides, the answer comes back. Nothing new has to be invented — the two ends just need
joining.

### The surface

```python
from agentkit.integrations.mcp import serve_permission_gate

gate = serve_permission_gate(
    asker=ctx.asker,             # or a HumanGate
    autonomy=ctx.autonomy,       # "auto" answers without asking
    deadline_s=120,              # bounded wait; an expiry degrades rather than hangs
)

cognition = ClaudeCliCognition(
    mcp_config=(gate.config_path,),
    permission_prompt_tool=gate.tool_name,
    strict_mcp_config=True,
)
```

### What it has to get right

- **The wait is bounded.** `approval_deadline_s` exists on the ReAct path for a reason: an unbounded wait
  is a hung build holding a worktree. An expiry has to **degrade** — deny and record — rather than hang.
- **`autonomy` is honoured identically to every other pattern.** `HumanGate`'s own claim is that autonomy is
  set once per run and honoured uniformly; a CLI path that decided independently would break that claim in
  the one place it is hardest to notice.
- **A decision is recorded as a decision**, with who and when, so a run can be audited afterwards.
- **A denial reaches the model as a refusal it can act on**, not as a dead end. Same discipline as
  `ToolArgumentError`.

### Tests that make it real

- an approval request under `autonomy="auto"` is answered without an `Asker` being consulted
- a request under `autonomy="ask"` reaches the `Asker` and the answer reaches the CLI
- a deadline that expires denies, records, and lets the run continue
- a denial is visible in the stream as a refusal
- no `Asker` wired plus `autonomy="ask"` fails loudly at construction, not at the first prompt

---

## F3 · Make the middleware chain reach native CLI tools

**Priority: high. This is the architectural one.**

### The gap

agentkit states it in its own source, in `_charge_meters`:

> *The CLI bypasses the `Invoker`, so the `meter()` middleware never sees this usage and every meter on the
> context stays at zero. That is how a documented safety mechanism ends up doing nothing.*

Metering was patched by hand for exactly that reason. **Nothing else was.** So for every native CLI tool
call — `Write`, `Edit`, `Bash`, `WebFetch` — these do not run:

| middleware | what stops applying |
|---|---|
| `egress` | default-deny URL checking. A `WebFetch` reaches anywhere |
| `guard` | every input guard |
| `audit` | one record per tool call. There are no records |
| `memoize` | idempotency and single-flight |
| `Guardrail.check_url` | SSRF and allowlist |
| `RunPolicy` | the Rule-of-Two capability check |

A caller reading agentkit's middleware documentation and wiring `ClaudeCliCognition` gets a session where
**none of it applies**, with nothing saying so. That is the same failure `_charge_meters` was written to fix,
five more times.

### Three ways to close it, and they are not equivalent

**(a) Serve everything over MCP; `tools=('',)`.** Every call comes back through agentkit code, so the chain
applies naturally and F1 is the whole fix. Cost: the CLI's own tools are good — its `Edit` in particular —
and reimplementing them behind MCP is real work and a real quality loss.

**(b) Generate hook settings.** `ClaudeCliCognition` already takes `settings` → `--settings`, and Claude Code
settings carry `PreToolUse` hooks. A hook can call back into the chain and refuse.

```python
from agentkit.integrations.claude_cli import hook_settings

settings = hook_settings(
    middleware=tool_chain,       # the same chain the Invoker would use
    ctx=ctx,
    tools=("Write", "Edit", "Bash"),
)
cognition = ClaudeCliCognition(settings=settings.path, ...)
```

Keeps the CLI's tools and makes the chain apply. Cost: the refusal now lives in a generated hook script,
which is a second execution path to keep correct.

**(c) Do nothing, and say so.** Document the bypass at the top of `ClaudeCliCognition` with the same
directness `_charge_meters` uses, and have `__post_init__` **warn** when a `ctx` carrying tool middleware is
used with native tools enabled.

### The recommendation

**(c) unconditionally, plus (b).** (c) is a day's work and removes a whole class of silent
mis-configuration; the warning is the part that matters, because the failure is invisible by construction.
(b) is the general fix and keeps the CLI's tool quality. (a) is a caller's choice, not a framework feature —
F1 already makes it possible.

### Tests that make it real

- a `ctx` with tool middleware plus native tools emits a warning naming the middlewares that will not apply
- the same configuration with `tools=('',)` emits none
- a `PreToolUse` hook generated from a chain refuses a write outside an allowlist, and the CLI reports the
  refusal
- a hook that itself fails does not take the session down

---

## F4 · Project a `Skill` into a CLI sub-agent

**Priority: medium.**

### The gap

`ClaudeCliCognition` takes `agents: dict[str, Any] | None` — sub-agent definitions the CLI can delegate to.
agentkit has `Skill`, described as *"the missing primitive between `Tool` and `Agent` — prompt + cognition +
memory as one wirable unit"*. Those are the same idea at two ends of a wire, and nothing joins them.

Without it, an application that has expressed a reviewer as a `Skill` has to restate it as a CLI agent
definition by hand — the second description of one thing that every other part of agentkit is careful to
avoid.

### The surface

```python
from agentkit.integrations.claude_cli import as_cli_agents

cognition = ClaudeCliCognition(
    agents=as_cli_agents([reviewer_skill, repairer_skill]),
)
```

### What it has to get right

- **A skill's tool restriction survives the projection.** A reviewer that is read-only because of its tool
  list must not become a sub-agent with the parent's tools.
- **The prompt travels whole**, not its first line. (See the note at the end of this document.)
- **A skill whose cognition cannot be expressed as a CLI sub-agent is refused at construction**, rather than
  projected into something that looks similar and behaves differently.

---

## F5 · An offline double for the CLI path

**Priority: high. Without it, nothing on this path can be tested without spending.**

### The gap

`agentkit.testing.fakes` is thorough — `FakeLLM`, `FakeMemory`, `FakeTool`, `FakeClock`, `FakeFetch`,
`FakeSearch`, `FakeCompactor`, `FakeGrounder`, `RecordingTracer`, `make_test_ctx`. Every one of them fakes a
**port**. `FakeLLM.script([...])` even replays a multi-step tool loop.

None of them helps here, because the CLI is not behind an `LLMPort`. It is a subprocess emitting
stream-json. So a test of anything CLI-shaped either spends real money or stands up a real `claude` binary.

`FakeLLM`'s own docstring draws the distinction this needs: a *script* is "a finite, ordered claim about how
many turns the run takes", and asking for one more raises rather than repeating. The CLI double wants the
same discipline over recorded sessions.

### The surface

```python
from agentkit.testing.fakes import FakeClaudeCli

cli = FakeClaudeCli.replay(Path("sessions/adds_endpoint.jsonl"))
cognition = ClaudeCliCognition(claude_bin=cli.binary)   # or an injected transport seam
```

Two shapes, and both are worth having:

- **`replay(path)`** — a recorded stream-json session, replayed event for event. This is what makes a real
  dispatch reusable as a fixture.
- **`script([...])`** — assembled turns, for a case nobody has recorded yet: a refusal, a timeout, a session
  limit, a malformed final answer.

### What it has to get right

- **The double sits at the same seam the real one does**, so a test exercises the parsing, the budget
  charging and the event mapping rather than skipping them. A double that returns finished `AgentResult`s
  tests nothing that has ever gone wrong.
- **A recording replays byte-identically twice.** No clock, no randomness.
- **Exhaustion raises**, matching `FakeLLM`'s reasoning: a recording is a claim about how many turns a run
  takes, and asking for one more contradicts it.
- **Malformed sessions are constructible.** Every bug worth a regression test on this path is a session that
  came back wrong.

---

## Two smaller things worth fixing while in there

**A tool's model-facing description is the first line of its docstring and nothing else.** Everything after
it is read by people editing the source and by nobody else, while still sitting in the docstring looking as
though it had been delivered. This cost a real defect: the sentence a model most needed — *your own reading
of the code is not a substitute for this call* — was in the second paragraph and never shipped. Either take
the whole docstring, or refuse a multi-paragraph docstring at decoration time so the author has to choose
out loud.

**`ClaudeCliCognition` has no `caps`.** `RunPolicy` refuses a tool set that can reach private data, untrusted
content and egress at once. A CLI session with `Read`, `WebFetch` and `Bash` is that trifecta exactly, and
`RunPolicy` cannot see it. Declaring capability tags for the built-in CLI tools would let the same check
apply on both paths.

---

## Suggested order

```
F1  serve a ToolRegistry over MCP        ─── unblocks everything custom
F2  bridge Asker to the permission tool  ─── unblocks every HITL feature
F5  the offline CLI double               ─── unblocks testing F1 and F2 without spending
F3c warn on bypassed middleware          ─── one day, removes a silent failure class
F3b generate hook settings               ─── the general fix for native tools
F4  Skill → CLI sub-agents               ─── needed once sub-agents are real
```

F1, F2 and F5 are the set that makes a CLI-driven application buildable at all. F3c is the cheapest thing on
the list and prevents the failure that is hardest to see. F3b and F4 follow when there is something to
apply them to.

---

# Part II · Memory, stores, and composition

A second pass, reading the memory and store surfaces rather than their summaries. The first part was about
a constraint the framework had not met yet; this part is about places where a well-built seam stops one step
short of the thing an application actually has to do.

The memory design is genuinely good — one `MemorySource` protocol, backends that are small, decorators that
nest because each *is* a `MemorySource`. Composition is better than most: `Workflow` and coordinator `Agent`
nest in both directions over one spine, with `as_tool` closing the loop. So none of what follows is a
redesign. Each is a missing piece at a boundary that already exists.

---

## M1 · `MemoryItem` has no identity, and `CompositeMemory` therefore cannot dedupe

**Priority: high. It is a correctness gap, not an ergonomic one.**

### The gap

`CompositeMemory` is documented as *"parallel fan-out + merge + optional rerank"*. The merge is:

```python
merged = [item for batch in batches for item in batch]
```

Concatenation. There is no dedupe, and there could not be: `MemoryItem` is `content · source · score ·
metadata` with **no id**. So two sources holding the same fact return it twice, a reranker scores both, and
the top-k a model sees is one fact occupying two slots.

That is worst in exactly the composition the class exists for. Ask a vector store and a journal the same
question and the overlap is not incidental — it is the normal case, because the journal is often what the
vector store was built from.

The absent id costs three more things: an item cannot be **updated**, cannot be **deleted**, and cannot be
**correlated** across a query and a later write. A registry of recurring facts — the shape any
learn-across-runs feature takes — needs all three.

### The surface

```python
@dataclass(frozen=True)
class MemoryItem:
    content: str
    source: str
    id: str | None = None      # stable within a source; None keeps every existing caller working
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
CompositeMemory(sources, dedupe="id")        # default when every item carries one
CompositeMemory(sources, dedupe="content")   # digest of content, for backends that cannot supply ids
CompositeMemory(sources, dedupe=None)        # today's behaviour, chosen rather than inherited
```

On a collision, keep the **higher score**, and record the sources that agreed in metadata — that two
independent sources returned the same fact is signal a reranker should be able to use, and concatenation
throws it away.

### Tests

- two sources returning one fact with one id yield one item
- the surviving item keeps the higher score and names both sources
- `dedupe=None` reproduces today's output exactly
- ids absent everywhere falls back to content digest rather than silently not deduping

---

## M2 · No `ReadOnlyMemory`, so a source that must never be written has no way to say so

**Priority: medium. Cheap, and it closes a real hole.**

`ScopedMemory` enforces tenant boundaries, `CompactedMemory` bounds size, `CachedMemory` avoids repeat
queries. There is no decorator that refuses `write`.

Every application has at least one source that is read-only *by policy* rather than by backend: a curated
knowledge base, a registry an operator maintains, a corpus of recorded facts an agent may consult and must
not extend. Today the only protection is that nothing happens to call `write`, which is a property of the
code as it currently stands rather than a rule.

```python
ReadOnlyMemory(registry)                          # write raises MemoryWriteRefused
ReadOnlyMemory(registry, on_write="ignore")       # for a composite where one source is read-only
```

The second form matters more than it looks: `CompositeMemory.write` fans out to every source, and one
read-only member should not have to make the whole composite unwritable. `CompositeWriteError` already
distinguishes accepted from failed, so the pieces are there.

---

## M3 · Provenance is flattened to a string before it reaches a prompt

**Priority: high for any application that has to defend what it grounded on.**

### The gap

```python
Grounder = Callable[[Ctx, str], Awaitable[str]]
```

`RequestBuilder` takes a `Grounder`, and grounding lands in `wc.prefix` — correctly, never in `messages`, so
the cache invariant holds. But by the time it gets there it is **text**. Which source it came from, what
score it had, whether it was an observation or something a model wrote in an earlier run: all of it is gone
at the boundary where it starts to matter.

That makes one rule unenforceable at the only place it could be enforced: *a memory a model wrote is not
evidence*. Once a summary of a past run is a string in the prefix, nothing downstream can tell it from a
recorded fact, and any claim built on it inherits an authority it never had.

The class of applications this affects is not small — anything that has to answer *what was this conclusion
based on* after the fact.

### The surface

Keep the callable seam and add a typed one beside it, rather than replacing it:

```python
GroundingSource = Callable[[Ctx, str], Awaitable[Sequence[MemoryItem]]]

RequestBuilder(
    grounding=source,
    render=lambda items: "\n\n".join(i.content for i in items),   # default
    admit=lambda item: item.metadata.get("tier") != "inferred",   # optional predicate
)
```

Three things follow, and each is worth having on its own: the **items are inspectable before rendering**, so
an application can refuse one; the **rendering is a policy** rather than a fixed join; and the **items can be
recorded beside the prefix**, so a run can say afterwards what it grounded on rather than only what it said.

`memory.grounder` already adapts a `MemorySource` to a `Grounder` by flattening — it would adapt to the
typed seam by not flattening, which is less code.

---

## S1 · `StorePort` has no compare-and-set, no atomic counter, and no key scan

**Priority: high. Three primitives, and each is load-bearing for a different feature.**

### The gap

`StorePort` is `get · set · get_or_set · delete · append · list`. That is a good minimal key-value surface
and it is missing the three operations that anything coordinating across processes needs.

**No compare-and-set.** `get_or_set` covers *create if absent*. It does not cover *replace only if unchanged*,
which is what every read-modify-write needs. Allocating a monotonic ordinal — the shape any versioning
feature takes — is `read max, write max+1`, and two writers race it. With four adapters including Postgres
and Redis, both of which have the primitive natively, the port is the only thing in the way.

**No atomic increment.** A counter with an expiry — the shape of every rate limit — cannot be expressed. The
consequence is concrete and observable: an application needing one writes a raw Lua script against Redis and
bypasses the port entirely, which is a store adapter that exists and is not reached through the store seam.
The check itself then sits outside everything the framework knows how to test, trace or meter.

**No key scan.** `list(key)` reads back one appended log. There is no way to enumerate keys under a prefix,
so *everything recorded for this run* is answerable only if every writer also maintained an index by hand —
and an index maintained by hand beside the data it indexes is the classic pair that drifts.

### The surface

```python
class StorePort(Protocol):
    ...
    async def compare_and_set(self, key: str, expected: Any, value: Any, *, ttl: int | None = None) -> bool
    async def increment(self, key: str, by: int = 1, *, ttl: int | None = None) -> int
    async def scan(self, prefix: str, *, limit: int | None = None) -> AsyncIterator[str]
```

`compare_and_set` returns whether it applied rather than raising: a lost race is an ordinary outcome that a
caller retries, not an error.

### Why it belongs on the port and not in each application

All four adapters can implement all three. Memory: trivially. File: with the lock the adapter already has.
Postgres: `UPDATE ... WHERE value = $expected`, a sequence, and a prefix query. Redis: `WATCH`/`MULTI` or a
small script, `INCR` with `PEXPIRE`, and `SCAN`. Leaving them out does not remove the need — it relocates it
into every application, once per application, untested each time.

### Tests

- two concurrent `compare_and_set` calls on one key: exactly one returns `True`
- `increment` under concurrency totals correctly
- `increment` with a ttl expires the counter and the window together, not one and then the other
- `scan` returns every key under a prefix and nothing above it
- the same contract suite runs against all four adapters

---

## C1 · Neither control model covers a graph whose shape is decided at runtime

**Priority: high, and it is the composition gap rather than a missing convenience.**

### The gap

The two models are complementary and well built.

**`Workflow`** is explicit: `.agent`, `.fn`, `.tool`, `.coordinator`, `.human_gate`, `.subworkflow`,
`.route`, with typed outputs threaded to dependents and durable resume. Its nodes are **authored**, so the
graph is a fact about the source.

**`PlanPolicy`** is a supervisor over a list of `Step`s, groups sequential and steps within a group
concurrent. The list *is* constructible at runtime — so this is the dynamic path — but a Step dispatches to
a **named child**, and the policy has no routes, no sub-workflows and no typed outputs threaded between
steps.

So an application whose graph depends on data it computed a moment ago — N items discovered at runtime, each
needing the same treatment, with a gate after the group — has to choose between a model that cannot size
itself and one that cannot express the rest of the structure. That is not an exotic case; it is what any
plan-then-execute application looks like.

### The surface

A node that expands at runtime, keeping everything else Workflow already provides:

```python
wf.map(
    "implement",
    over=lambda done: done["plan"].requirements,   # runtime-sized
    each=lambda item: wf_agent_for(item),          # one node per element
    after="plan",
    bounded_by=4,                                  # the tree semaphore, as everywhere else
)
wf.human_gate("review", after="implement")
```

The expansion has to reuse the existing spine rather than sit beside it: `gather_bounded` and
`ctx.semaphore()` for concurrency, `ctx.check_cancelled()` between elements, and the `done` map keyed so a
resume can tell which elements finished. That last one is the hard part and the reason this belongs in the
framework rather than in an application: **resume across a dynamically sized node** needs the expansion to
be deterministic given the same inputs, and the checkpoint to record the expansion rather than only its
results.

### Tests

- a map over three elements runs three nodes and threads each output to the dependent
- concurrency is bounded by the semaphore, and a wave of N provably overlaps rather than serialising
- a failure in one element behaves like any other node failure, `best_effort` included
- a resume after two of three completed runs only the third
- an empty collection is a node that completes rather than a graph that hangs

---

## C2 · No composable retry-with-recurrence-detection

**Priority: medium.**

`run_with_resilience` retries on **transient** failure — classify, backoff, circuit breaker. That is the
right shape for a flaky call and the wrong shape for a *semantic* one: an attempt that completed, produced
an answer, and did not achieve the goal.

`LedgerPolicy` has the missing half — a progress ledger re-derived each round asking *satisfied? looping?* —
but it is welded into a multi-agent supervisor. The pattern underneath is general and worth extracting:

> attempt · fingerprint the outcome · attempt again on a *different* fingerprint · stop on a repeat.

Retry counts are the wrong bound for this, and it is worth saying why: three attempts producing three
different failures is progress, and two producing the same one is not. A count cannot tell those apart, so a
count is either too tight for the first case or too loose for the second.

```python
await attempt_until_stuck(
    lambda: run_one(),
    fingerprint=lambda outcome: outcome.failure_signature,
    on_repeat="escalate",       # or "stop"
    max_attempts=4,             # a backstop, not the bound
)
```

---

## Improvements to things that already work

**`Quota` is per-process.** `max_rpm`, `max_tpm`, `max_usd`, `window` — with no store behind it, so it holds
until a second worker starts, and then holds nothing while continuing to report that it does. This is the
same failure `_charge_meters` documents about the CLI bypassing the Invoker, in a different place: a
documented safety mechanism doing nothing. Given S1's `increment`, a store-backed `Quota` is small.

**`CachedMemory` has no invalidation.** Fine for a query cache, wrong for anything writable — a write to the
underlying source leaves the cache serving what was there before, indefinitely. A ttl and an explicit
`invalidate(query)` would cover both.

**`SequentialMemory` and `CompositeMemory` have no per-source budget.** One slow backend delays every query
through the composite; `CompositeMemory`'s docstring mentions the tree semaphore but not a deadline. A
per-source timeout that degrades — return what came back, record what did not — matches how the framework
handles a failing judge and a misbehaving tracer everywhere else.

**Prompt versioning is deliberately out of scope, and the boundary is worth restating.**
`agentkit.prompts` says the framework defines what a prompt is and applications own storage, discovery and
validation. That is a defensible line. It is worth saying in the docstring what an application is expected
to build on the far side of it — a digest over template *and* declared inputs, and cases per guardrail —
because every application that gets there arrives at the same shape and each one discovers it separately.

---

## Revised order

```
S1   store: compare_and_set · increment · scan     ─── three primitives, several features each
M1   MemoryItem.id + CompositeMemory dedupe        ─── correctness, not ergonomics
C1   Workflow.map                                  ─── the composition gap
M3   typed grounding                               ─── provenance survives to the prefix
M2   ReadOnlyMemory                                ─── an afternoon
C2   attempt_until_stuck                           ─── extract from LedgerPolicy
     store-backed Quota, CachedMemory invalidation, per-source timeouts
```

S1 first because three separate features are waiting on it and every adapter can already do all three. M1
next because it is the only item here that produces a *wrong answer* rather than a missing capability.
