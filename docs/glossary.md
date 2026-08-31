# Glossary

Every framework invents words. This page is the plain-English version of
agentkit's, written for someone who has not read the rest of the docs
yet.

Each entry is a one-line answer first, then the detail, then a link to
the page that goes deep. You should be able to read any entry without
having read any other one.

---

## The main pieces

### Agent

**A model, a prompt, and the things it is allowed to do — bundled into
one object you can run.**

You give an agent a task as a string, and it gives you back a result. In
between it may call the model once, or many times, and it may use tools
along the way. Everything else in agentkit exists to control what happens
in that "in between".

→ [Agents](concepts/agents.md)

### Cognition

**The part that decides how many times to call the model, and what to do
between calls.**

This is the piece most frameworks leave implicit. Asking a model one
question and returning its answer is one strategy. Letting it request a
tool, running that tool, handing back the result and asking again — over
and over until it stops asking — is a different strategy. Coordinating a
team of child agents is a third.

In agentkit that strategy is a separate object, so you can swap it
without touching your prompt, your tools, or anything downstream:

- `SingleCallCognition` — ask once, return the answer.
- `ReActCognition` — the tool loop (see **ReAct** below).
- `CoordinatorCognition` — drive several child agents.
- `ClaudeCliCognition` — hand the whole job to a local `claude` CLI.

→ [Agents](concepts/agents.md)

### ReAct

**Let the model use tools by looping: it asks for one, you run it, you
tell it what happened, and it decides what to do next.**

The name is from a 2022 research paper and stands for *Reason + Act*. The
loop is simple: the model produces either a final answer or a request to
call a tool. If it is a tool request, the framework runs the tool, appends
the result to the conversation, and calls the model again. It repeats
until the model answers or hits a limit you set.

Almost every "AI agent" you have used works this way underneath.

→ [Agents](concepts/agents.md)

### Tool

**A Python function the model is allowed to ask for by name.**

The model cannot run code. It can only produce text — including text that
says "please call `search` with `query='octopus cognition'`". A tool is
the function that text refers to, plus a description of its arguments
that gets sent to the model so it knows the function exists.

→ [Tools](concepts/tools.md)

### Workflow

**A pipeline you write out step by step, instead of letting the model
decide the order.**

Use it when you already know the sequence — fetch, then summarise, then
file a ticket — and you want the steps, their dependencies, and their
failure behaviour to be explicit rather than up to the model.

→ [Agents](concepts/agents.md)

### Map node

**A workflow step whose width is not known until the run gets there.**

Every other node is authored, so the graph is a fact about your source
file. `wf.map(name, over=, each=)` is the exception: `over=` is handed
the outputs so far and returns a collection, and one element node runs
per item in it. You could already fan out inside a plain function node;
what you could not do is make that fan-out *resumable*, which is why
this belongs in the engine. The price is that a resumed run needs the
same expansion — if `over=` returns a different collection on the second
pass that is a changed graph, and it raises `MapExpansionChanged` rather
than threading outputs into the wrong slots.

→ [Recipe: the workflow graph](recipes/workflow-graph.md)

### Skill

**A saved recipe for an agent, so you can stamp out copies of it.**

Prompt, cognition, tools and memory bundled under a name. Useful when you
need the same specialised agent in several places, or want to hand it to
another agent as a tool.

→ [Skills](concepts/skills.md)

---

## How a run is controlled

### RunContext

**One object carrying everything that is true for this particular run.**

Who it belongs to, what it may spend, how to cancel it, and which
external services it can reach. It is passed to nearly every call, which
is why those things can be enforced without any global state.

→ [Runtime](concepts/runtime.md)

### Budget

**A spending limit for one run.**

The thing that stops an agent looping forty times against a confused
model and handing you a $217 bill. It tracks money and call counts, and
either refuses further work or reports that it is out, depending on how
you configure it.

→ [Runtime](concepts/runtime.md)

### Quota

**A rate limit per customer, over a rolling time window.**

Where a budget caps one run, a quota caps a tenant across many runs — so
many requests per minute, so many tokens, so many dollars.

→ [Runtime](concepts/runtime.md)

### Meter

**The thing that actually does the counting.**

A budget or quota is the *limit*. A meter is what watches calls go past
and charges them against it.

→ [Observability](concepts/observability.md)

### Scope and tenant

**Which customer this run belongs to.**

If your app serves more than one organisation, every cache key, memory
lookup and stored record has to be tagged with whose data it is, or one
customer's documents end up in another's answers. `Scope` is that tag,
and it is threaded through everything that stores or retrieves anything.

→ [Context](concepts/context.md)

### Autonomy

**How much the agent is allowed to do without asking a human.**

Set it to `gated` and any tool marked as changing the world will pause
the run and wait for a person to approve it.

→ [Agents](concepts/agents.md)

### Suspend and resume

**Pausing a run to the point where the process can die, and picking it up
later.**

When an agent needs a human decision, it does not sit in memory holding a
connection open. It writes down where it got to, ends, and returns a
result that says "suspended". Later — possibly in a different process,
possibly the next day — you hand back the answer and it continues.

→ [Agents](concepts/agents.md)

### Approval decision

**The record of one permission prompt: what was asked, what was decided,
and *what did the deciding*.**

A boolean cannot hold that last part, and it is the part an auditor
needs. *Allowed because a person looked at it* and *allowed because the
run's tier does not gate this call* are the same `True`. So are *denied
because the reviewer said no* and *denied because nobody answered in
sixty seconds* — and the second is not a refusal, it is a timeout that
degraded into one. So `ApprovalDecision.source` is a closed set naming
which happened: `asker`, `auto_allow`, `autonomy`, `timeout`, `error`.
`asked` is kept separate from `source == "asker"`, because a prompt can
reach a human and then expire, and whether somebody was interrupted is a
fact about what the run cost a person rather than about the verdict.

→ [Integrations](concepts/integrations.md)

### Elicitation

**Asking a person for a *value*, not just a yes or no.**

Approval answers "may this run?". Elicitation answers "what is the code
they just texted you?" — a typed request for information, with a deadline
and a record of who answered.

→ [Recipe: elicit a value](recipes/elicit-a-value-from-a-human.md)

### Cooperative cancellation

**Stopping a run by asking rather than by force.**

The framework sets a flag; the agent checks it at safe points — between
steps, between tool calls — and stops cleanly. This means cleanup still
runs and nothing is left half-done. The cost is that a cancel is not
instant: it takes effect at the next check.

→ [Runtime](concepts/runtime.md)

---

## Keeping state

### Checkpoint / Checkpointer

**A saved snapshot of a run, so a crash does not cost you everything.**

If a twenty-minute research run dies at minute eighteen, you want to
restart from minute eighteen, not from zero. A checkpointer writes that
snapshot to durable storage after each successful step.

→ [Capabilities](concepts/capabilities.md)

### WorkingContext

**What the agent knows right now** — the conversation so far, plus any
notes it has taken.

→ [Context](concepts/context.md)

### Compaction / Compactor

**Shrinking the conversation when it gets too long to send.**

Models accept a limited amount of text. Once a conversation outgrows it
you must drop or summarise something. A compactor is the policy that
decides what goes: keep the last N turns, summarise the middle, drop the
least important.

→ [Capabilities](concepts/capabilities.md)

### Memory

**Things the agent looks up before it answers.**

Your documents, past conversations, anything the model was not trained
on. Usually backed by a search index.

→ [Memory](concepts/memory.md)

### Dedupe and memory identity

**Recognising that two sources returned the same fact, so it does not
occupy two of the few slots the model actually reads.**

Asking several sources at once means the same passage comes back twice,
and that is the normal case rather than bad luck: the journal a vector
index was built from will happily return the row the index does.
`MemoryItem.id` is the backend's own identifier for a record — a chunk
id, a row key — and `CompositeMemory` merges on it by default, or on the
content once stripped for backends that have no ids. The surviving copy
is the higher-scored one, stamped with how many copies collapsed and
which backends agreed; that two independent sources said the same thing
is signal a reranker can use, and plain concatenation throws it away.

→ [Memory](concepts/memory.md)

### Read-only memory

**A source the agent may read and must never extend — enforced, rather
than assumed.**

A curated knowledge base, an operator-maintained registry. Nothing about
the *backend* says it is read-only: the vector store behind a curated KB
takes an upsert exactly like any other. So the only thing protecting it
is that no code happens to call `write` today, which is a property of
the code as it currently stands and not a rule. `ReadOnlyMemory`
constrains exactly one verb — `query` passes straight through, `write`
raises `MemoryWriteRefused`. `on_write="ignore"` drops the write instead,
so one read-only member does not make a whole fan-out unwritable; it is
the dangerous option, and it is counted and reported rather than silent.

→ [Memory](concepts/memory.md)

### Grounding / Grounder

**Fetching relevant material and putting it in the prompt before the
model sees the question.**

The difference between asking a model "what is our refund policy?" and
asking it the same question with your actual refund policy pasted above.

→ [Capabilities](concepts/capabilities.md)

### Grounding source

**Grounding that reaches the prompt builder as records rather than as
one flattened string.**

A `Grounder` returns text, so by the time retrieved material reaches the
prompt, which source it came from, what it scored, and whether it was a
recorded fact or a summary a model wrote in an earlier run are all gone
— flattened at exactly the boundary where they start to matter. That
leaves one rule unenforceable at the only place it could be enforced: *a
memory a model wrote is not evidence*. A `GroundingSource` hands over
`MemoryItem`s instead, so an `admit` predicate can veto one before it is
rendered, the formatting is a policy you choose (`render_grounding` is
the default), and the admitted items can be recorded beside the prefix.
The callable form still works unchanged; passing both is refused rather
than quietly resolved.

→ [Capabilities](concepts/capabilities.md)

### Token

**The unit models read, write and bill in — roughly ¾ of a word.**

Every limit you will meet is counted in these: context windows, rate
limits, and your invoice.

---

## How the plumbing fits together

### Port

**A description of something agentkit needs but cannot provide itself.**

A model, a database, a vector index, the clock. agentkit says what shape
that thing must have — which methods, taking what — and stays out of the
business of implementing it.

→ [Adapters](concepts/adapters.md)

### Adapter

**An actual implementation of a port.**

The port says "a store must be able to get and set keys". The adapter is
the one backed by Postgres, or by a dict in memory, or by Redis. Your
application code talks to the port, so swapping the adapter changes one
line at startup and nothing else.

→ [Adapters](concepts/adapters.md)

### Compare-and-set, increment and scan

**The three store operations that let two runs touch one key without
racing each other.**

Get and set express "cache this". They cannot express *changing
something that is already there*. Allocating the next ordinal is "read
the max, write max+1", and two writers race it: both read `4`, both
write `5`, and one ordinal is handed to two runs.
`compare_and_set(key, expected, value)` writes only if the key still
holds what you read, and *returns* whether it applied rather than
raising — losing is the expected half of an optimistic loop, not a
fault. `increment(key, by, ttl=...)` is a counter with an expiry, which
is the shape of every rate limit, as one atomic step. `scan(prefix)`
enumerates the keys under a prefix, so "everything recorded for this
run" no longer needs a hand-maintained index beside it that drifts.
Every shipped store adapter implements all three.

→ [Adapters](concepts/adapters.md)

### Seam

**A deliberate place where one piece can be swapped for another.**

Used throughout these docs as shorthand for "this is a boundary you are
meant to be able to cut at". Every port is a seam.

### Protocol

**Python's way of saying "anything with these methods counts".**

From `typing`. Your class does not inherit from anything or register
anywhere — if it has the right methods, it fits. This is why you can hand
agentkit an object it has never heard of.

### Middleware

**A wrapper that adds one concern to every call, without you editing the
call sites.**

Retry, caching, cost tracking, tracing, a safety check. Each is written
once and layered on. They nest like an onion: the outermost sees the call
first and the response last.

→ [Middlewares](concepts/middlewares.md)

### Invoker

**The thing that actually makes the call, after walking the middleware
chain.**

You rarely construct one directly; the agent does it for you.

→ [Runtime](concepts/runtime.md)

### Frozen / immutable

**A value that cannot be changed after it is created.**

agentkit freezes the things that get recorded — tool arguments, audit
records, results — so that what you read in a log is what actually
happened, and not something a later line of code edited.

→ [Kernel](concepts/kernel.md)

### Kernel

**The shared vocabulary — the data shapes every other part passes
around.**

What a message is, what a result is, what a cost is. Deliberately tiny,
and deliberately importing nothing, so everything above it can be
swapped.

→ [Kernel](concepts/kernel.md)

---

## Watching what happened

### Observation

**A human-meaningful event you can show a user** — "searching the
handbook", "waiting for approval".

→ [Observability](concepts/observability.md)

### Trace / span

**The engineer-facing timeline of a run.**

A span is one timed segment — one model call, one tool execution — with
timing and metadata attached. Nested spans form a trace, which is what
you look at to find out why something took nine seconds. This is the
OpenTelemetry vocabulary, and agentkit uses it as-is.

→ [Observability](concepts/observability.md)

### Replay

**Storing the full request and response payloads somewhere other than the
trace**, because traces have size limits and prompts are large.

→ [Observability](concepts/observability.md)

### Sampler

**The rule that decides which runs get traced.**

Recording every span of every run is expensive at volume, so you record
a fraction. The sampler is what picks. It only ever drops *traces* —
your cost and usage numbers are always counted in full.

→ [Observability](concepts/observability.md)

### Delta

**One small piece of a streaming answer.**

Models emit their reply gradually rather than all at once. Each fragment
that arrives is a delta; joined together in order, they are the complete
response. This is what lets a UI show text appearing as it is written.

→ [Kernel](concepts/kernel.md)

---

## Words that show up in passing

### Fan-out

**Starting many pieces of work at once instead of one after another.**

### Structured concurrency

**Running things in parallel with a rule: no task outlives the block that
started it.** If one fails, its siblings are cancelled rather than left
running unattended.

→ [Runtime](concepts/runtime.md)

### Backpressure

**Making a fast producer wait for a slow consumer**, instead of letting
work pile up until you run out of memory.

### Idempotent / idempotency

**Safe to run twice.** Reading a file is idempotent. Charging a card is
not — so if a retry happens, the framework needs a key that lets it
recognise "I already did this one".

### Memoize

**Cache the answer so an identical call does not run twice.**

→ [Middlewares](concepts/middlewares.md)

### Egress

**Outbound network access from a tool** — and the check that decides
whether a given URL is allowed.

### Guardrail

**A check that can veto** — refusing a URL, blocking an output.

### Lethal trifecta

**One run that can reach private data, take in untrusted content, and
send something outbound — the combination that turns a crafted input
into an exfiltration.**

Any two of the three are ordinary. All three together mean that text the
agent merely *read* can instruct it to fetch something sensitive and
post it somewhere. Tools declare which legs they supply as `caps`, and
`RunPolicy` refuses a set that assembles all three without a human gate.
Two things surprise people. One tool can supply two legs — `WebFetch`
both ingests untrusted content and reaches the network — so you do not
need three dangerous-looking tools. And a Claude CLI session now
declares capabilities of its own, where the default "every built-in
tool" is the full trifecta.

→ [Agents](concepts/agents.md)

### Evaluator

**Something that scores an answer**, so "did this get better?" has a
number behind it rather than an opinion. Can be plain deterministic
checks, a second model asked to judge, or both.

→ [Capabilities](concepts/capabilities.md)

### Reranker

**A second pass that re-sorts search results before they reach the
prompt.**

The first pass is fast and approximate; it fetches more candidates than
you need. The reranker orders them properly and keeps the best few, so
the model reads the most relevant material rather than merely the
closest match.

→ [Memory](concepts/memory.md)

### Schema

**A machine-readable description of a data shape.**

Sent to the model so it knows what arguments a tool takes, and used on
the way back to confirm what it produced matches what you asked for.

### Coercion

**Turning the model's JSON into the actual Python object you annotated.**

Without it, a parameter you declared as an `Enum` arrives as a plain
string and your code breaks on `.value`.

→ [Tools](concepts/tools.md)

### Handoff

**One agent passing the task to a peer**, rather than answering itself.

### Coordinator and Policy

**A coordinator runs several child agents. A policy decides whose turn it
is** — round-robin, a model picking the next speaker, or a plan written
up front.

→ [Agents](concepts/agents.md)

### Termination

**The rule for when a loop has gone on long enough** — a turn cap, a
deadline, or a condition you write.

### Recurrence-bounded retry

**Stopping when an attempt fails the *same way* as an earlier one,
rather than after N tries.**

A retry count is the right bound for a call that failed to complete — a
timeout, a 502 — and the wrong one for an attempt that finished,
produced an answer, and did not achieve the goal. Three attempts with
three different failures is progress; two with the same failure is a
circle. A count cannot tell those apart, so it is either too tight for
the first or too loose for the second. `attempt_until_stuck`
fingerprints each outcome and stops when a signature it has already seen
comes back — every signature seen, not just the previous one, because an
A, B, A, B oscillation never repeats consecutively and would otherwise
read as progress forever. It raises `Stuck`; `max_attempts` stays on as
a backstop for the case where the signatures keep genuinely differing.

→ [Kernel](concepts/kernel.md)

### MCP (Model Context Protocol)

**A standard way for a program to publish tools, documents and prompts so
any AI application can use them.**

There are thousands of published MCP servers. Speaking the protocol once
means you can use all of them, rather than writing an adapter per
service.

That is only half of it. agentkit speaks the protocol in both
directions: as a *client* consuming other people's servers, and — see
the next entry — as a *server* publishing your own tools.

→ [Integrations](concepts/integrations.md)

### Serving tools over MCP

**Publishing a `ToolRegistry` of your own as an MCP server, so a program
you do not control can call your Python functions.**

It exists because `ClaudeCliCognition` hands the whole loop to the
`claude` binary, and that binary cannot import your code. The only way
it calls your `deploy()` is if something serves it. Everything needed to
*describe* an agentkit tool was already there and already correct — only
the wire was missing, so a service wrote MCP JSON by hand and then owned
the job of keeping a hand-written schema in step with the Python
signature it claimed to describe. `serve_registry` sends the existing
schema out unchanged, and carries `side_effecting`, `requires_approval`
and `caps` across with it, so a policy check on your side still sees
what the tools can do.

→ [Integrations](concepts/integrations.md)

### Bearer token and the loopback fence

**Binding a server to `127.0.0.1` keeps it off the network. It does not
keep it away from everything else running on the same machine.**

Both of agentkit's MCP servers bind loopback and, by default, also
require a token the listener generates and puts in the config document
the CLI reads. Loopback alone used to *be* the containment, and that
argument holds exactly as long as nothing untrusted shares the host —
the wrong assumption here, because the point of the CLI cognition is
that it runs `Bash` in that same namespace. Be honest about what the
token buys: it turns *anything on this host* into *anything that can
read this process's temp directory*, which is why the config file is
`0600` and is removed along with the listener. `auth="none"` restores
the unauthenticated behaviour, as something you ask for by name.

→ [Integrations](concepts/integrations.md)

### Hook (PreToolUse)

**A callback the Claude CLI makes out to a script of yours before it
runs one of its own tools.**

It is the one place your middleware can still say no. When the CLI owns
the loop, its `Write`, `Bash` and `WebFetch` never pass through your
`Invoker`, so your egress checks, guards and audit records are wired,
documented, and doing nothing. `hook_settings` generates CLI settings
whose `PreToolUse` hook runs that same chain. The cost is real: the
refusal now lives in a generated script, which is a second execution
path to keep correct, and a middleware that needs your live `ctx` cannot
run in a separate process at all — those are refused when the settings
are generated, rather than accepted and then silently skipped.

→ [The Claude CLI](concepts/claude-cli.md)

### Sub-agent

**A named helper the Claude CLI can delegate to, defined in its
configuration rather than in your process.**

It is the CLI's word for what agentkit calls a `Skill` — a prompt, a
cognition, tools and memory under a name — which is why `as_cli_agents`
projects one into the other instead of asking you to restate it as a
second description of one thing. A skill's tool restriction survives the
projection, and that is a security property rather than an ergonomic
one: a reviewer that is read-only *because of its tool list* must not
arrive as a sub-agent holding the parent's tools. A skill that cannot be
expressed this way is refused by name with `SkillNotProjectable`, rather
than projected into something that looks similar and behaves
differently.

→ [The Claude CLI](concepts/claude-cli.md)

---

## Related

- [Why agentkit](why.md) — the case for the design these words describe.
- [Getting started](getting-started.md) — install and run something.
- [Cheatsheet](cheatsheet.md) — the same vocabulary as code.
