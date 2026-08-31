# Capabilities

A capability is an optional collaborator you can wire into an agent —
something that assembles its prompt, shrinks its context, vetoes its
output, scores it, saves it, or turns it into a typed object.

None of them are required. An agent with no capabilities is a valid
agent. Each one is a small typed seam with implementations shipped in
`agentkit.capabilities`, and the point of the seam is that swapping an
implementation never means editing the loop.

!!! tip "Is this page for you?"

    **Reach for it when** you need grounding, compaction,
    checkpointing, typed output, a guardrail or an evaluator.

    **Skip it for now if** your agent takes a prompt and answers —
    every capability here is optional, and none of them are on by
    default.

## The problem it solves

Compaction, grounding, guardrails and durability are the wrong things to
bake into a base class. Every one of them has a context-specific
trade-off — which model summarises, which safety policy applies, which
storage backend, which output flavour — and every one of them is
something you will want to change without touching the agent that uses
it.

Bake them in and you get the familiar shape: a base class with fourteen
constructor arguments, half of which are `None`, and a subclass per
combination. Concretely, the failures this avoids:

- A transcript grows past the model's window and the provider returns a
  400 in production, because compaction was a `if len(msgs) > 40:` inline
  in one agent's loop and nowhere else.
- A run crashes on turn nine of a twelve-turn job and there is nothing to
  resume from, because durability was a field on a class that happened
  not to have it.
- Your safety review asks "what checks the URLs the agent fetches?" and
  the answer is "each tool, differently".

Capabilities keep the loop small and the swap surface flat.

## The smallest thing that works

A `RequestBuilder` is the capability you meet first: it is the one place
that turns a prompt, retrieved evidence and a growing transcript into the
messages you send.

```python
import asyncio

from agentkit import Prompt, RequestBuilder, SlidingWindowCompactor, WorkingContext
from agentkit.testing import make_test_ctx


async def main() -> None:
    builder = RequestBuilder(
        prompt=Prompt(id="briefer", version="1.0.0", template="You brief analysts."),
        compactor=SlidingWindowCompactor(keep_recent=4),
    )
    wc = WorkingContext()
    ctx = make_test_ctx()

    built = await builder.build("summarise Q3", wc, ctx)
    print(built.prompt_version, built.approx_tokens)          # 1.0.0 15
    print([(m.role, m.content) for m in built.messages])


asyncio.run(main())
```

Call `build()` again on the same `WorkingContext` and it appends the next
turn, re-grounds if you asked it to, and folds the transcript through the
compactor — the per-turn delta stays minimal and identical across every
agent in your roster.

## The capabilities that ship

| Capability | What it does | Kind |
|---|---|---|
| `RequestBuilder` | Assemble the messages: prompt + grounding + task + compaction. | class — configure the shipped one |
| `Grounder` | Attach retrieved evidence to a request. | a callable type alias |
| `Compactor` | Shrink a transcript before it hits the model's window. | `Protocol` — 4 implementations ship |
| `Guardrail` | Check outbound URLs, redact, and frame tool output as untrusted. | class |
| `Evaluator` | Score an output with code checks or an LLM judge. | class |
| `Checkpointer` | Snapshot run state so a crash or a human gate can be resumed. | facade over `CheckpointPort` |
| `SchemaAdapter` | Coerce free-form model output into a typed object. | `Protocol` — 4 flavours ship |

Two of them are literal `Protocol`s, so any object with the right shape
satisfies them without inheriting anything. The rest are concrete classes
you configure or subclass, and `Checkpointer` is a special case worth
understanding: the substitutable seam is not the capability but the
**port** underneath it. You swap `CheckpointPort`, not `Checkpointer`.

## How they plug in

The thing to expect — and the thing that surprises most people first —
is that you do **not** pass capabilities to `Agent(...)`.

That is deliberate. A capability is not part of what an agent *is*; it
is something that happens to a request on its way out or on its way
back. Making them constructor arguments would mean a constructor that
grows a parameter every time the framework learns a new trick, and an
agent that has to know about every one of them.

So each capability attaches at whichever edge of the loop it actually
acts on. Where that is depends on what the capability does:

- a `Compactor` and a `Grounder` are configured on a `RequestBuilder`,
  which the agent uses to build every request;
- a `Compactor` can *also* run as the `compaction()` middleware, on the
  assembled request instead (see the warning below);
- a `Guardrail` is handed to the `egress()` middleware, which checks URLs
  before a side-effecting tool runs;
- a `Checkpointer` is resolved by the cognition or policy that suspends
  and resumes;
- a `SchemaAdapter` is built for you when you declare `output=` on an
  `Agent`, and read by the `output_coerce()` middleware.

## `RequestBuilder`

The single seam where four disciplines meet: **prompt** engineering (the
wording), **memory** engineering (what gets retrieved into it),
**context** engineering (how a growing transcript is bounded), and
**execution** engineering (how the request is attributed and budgeted).

Without it, every agent inlines its own version of *render the system
prompt → maybe retrieve → append the task → maybe compact → send*. That
duplication drifts: one agent stamps the prompt version, another does
not; one retrieves every turn, another only on the first. Those small
drifts become the primary failure modes as prompts and memory grow.

It owns steps one through four. The model call itself stays behind the
invoker's middleware chain. It does not call the LLM, does not touch
budget counters, and does not pick policies — the caller still chooses
*which* compactor and *which* retriever; the builder applies those
choices in the right order, in one place.

`build()` returns a `BuiltRequest` carrying `messages`, `prompt_version`
(so a regression maps back to an exact template revision) and
`approx_tokens` (a crude pre-call estimate for a budget pre-check — the
authoritative count comes back from the provider in `Usage`).

### The cache-stable prefix

This is the part that surprises people. A `WorkingContext` has two
halves:

- a **prefix** — the system prompt, the grounding block, and the output
  schema block. Written once, on the first turn, and never rewritten.
- a **tail** — `wc.messages`, the user task and every continuation turn.

Compaction touches **only the tail**. That is the KV-cache discipline:
any token mutated in the prefix invalidates the provider's cache from
that point onward, and the prefix is exactly the expensive, stable part.

`reground_every_turn` is where you choose which half of that to give up,
and there is no free answer:

| | `False` (default) | `True` |
| --- | --- | --- |
| Grounder invoked | once, on turn 1 | every turn |
| Turn 5 is answered with | evidence retrieved for **turn 1's** question | evidence for turn 5's question |
| Prefix bit-identical to turn 1 | 5 of 5 turns | 1 of 5 turns |
| Prefix cost, 4k tokens × 20 turns | `$0.0228` (cache reads) | `$0.2280` (fresh reads) |

Measured on a five-turn conversation against a five-fact handbook at
`k=1`: the default put the answering fact in front of the model on **1
of 5 turns**, `reground_every_turn=True` on 4 of 5 (the fifth was a
retrieval miss, not a staleness one). So it costs 10x on the prefix term
to stop answering turn 5 with turn 1's evidence.

Leave it `False` when the grounding is a corpus the whole conversation
shares — a handbook, a spec, a codebase digest. Set it `True` when each
turn asks about something different and the retrieved evidence *is* the
answer.

!!! warning "A \"turn\" here is a `build()` call, not a tool-loop step"

    `reground_every_turn=True` does **not** re-retrieve as a ReAct agent
    works through its tools. `ReActCognition.drive()` calls `build()`
    exactly once and appends tool traffic straight to the tail, so a
    four-step ReAct run invokes the grounder once either way (measured:
    4 LLM calls, 1 grounder call). The flag only does anything when you
    reuse one `WorkingContext` across successive `agent.run(...)` calls
    — a multi-turn conversation.

    Note also that the auto-wired `Agent(memory=...)` path builds its
    `RequestBuilder` for you and cannot pass this flag. To set it, build
    the `RequestBuilder` yourself and pass `request_builder=`.

`budget_check` is a `(approx_tokens) -> None` hook you can raise from to
abort `build()` before the messages are returned. The builder offers no
judgement of its own about what is "too big"; that policy is yours.

### `Grounder`

`Grounder` is a type alias, not a class: anything async-callable as
`(ctx, task) -> str`. Returning an empty string means "no grounding for
this task" and produces no grounding message.

#### Keeping the provenance — `GroundingSource`

A `Grounder` returns **text**, which means that by the time retrieved
material reaches the prompt, everything about where it came from is gone.
Which source, what score, whether it was a recorded fact or a summary a
model wrote in an earlier run — all flattened at exactly the boundary
where it starts to matter.

That makes one rule unenforceable at the only place it could be
enforced: **a memory a model wrote is not evidence.** Once a summary of a
past run is a string in the prefix, nothing downstream can tell it from a
recorded fact, and any claim built on it inherits an authority it never
had.

So there is a typed seam beside the callable one:

```python
RequestBuilder(
    grounding=source,                                             # -> Sequence[MemoryItem]
    render=lambda items: "\n\n".join(i.content for i in items),   # the default
    admit=lambda item: item.metadata.get("tier") != "inferred",   # optional veto
)
```

Three things follow, each worth having alone. The **items are inspectable
before rendering**, so an application can refuse one. The **rendering is
a policy** rather than a fixed join. And the items can be **recorded
beside the prefix**, so a run can say afterwards what it grounded on
rather than only what it said.

`grounder=` is unchanged and un-deprecated — every existing caller keeps
working. Passing **both** is refused at construction rather than silently
resolved, because two sources of truth for one slot is the bug this
framework avoids everywhere else.

`admit` runs before `render`, so a rejected item never reaches the
prompt at all.

The three seams have names you can annotate against — `GroundingSource`
for the retrieval callable, `GroundingAdmit` for the predicate, and
`GroundingRender` for the formatter. `render_grounding` is the default
renderer, exported so a custom one can fall back to it for the items it
does not want to treat specially.

!!! note "The record is encoded, not stored live"

    `record_grounding=True` writes the admitted items to the scratchpad —
    and the scratchpad is copied verbatim into every checkpoint. Storing
    live `MemoryItem` objects there worked in memory and raised
    `TypeError: Object of type MemoryItem is not JSON serializable` on
    any durable store, which is to say the audit trail broke on exactly
    the runs long enough to need one. They are encoded on the way in.

It is a callable rather than a memory object plus `k`/`where` knobs
because those knobs are retrieval mechanics, not prompt-assembly
concerns. The builder should not know the text came from a vector store
at all — only that *some* text is available. You bake the retrieval
policy in once at wiring time:

```python
import asyncio

from agentkit import InMemoryFiles, Prompt, RequestBuilder, WorkingContext
from agentkit.memory import FileMemory
from agentkit.testing import make_test_ctx

files = InMemoryFiles()
mem = FileMemory(files=files)


async def grounder(ctx, task):
    """The retrieval policy — k, filters, formatting — lives here, once."""
    items = await mem.query(task, k=5, ctx=ctx)
    return "\n".join(f"[{i.source}] {i.content}" for i in items)


async def main() -> None:
    await files.create("/notes.md", "Octopuses have nine brains.")
    builder = RequestBuilder(
        prompt=Prompt(id="briefer", version="1.0.0", template="You brief analysts."),
        grounder=grounder,
    )
    built = await builder.build("octopuses", WorkingContext(), make_test_ctx())
    for m in built.messages:
        print(m.role, "|", m.content)
    # system | You brief analysts.
    # system | Relevant context:
    #          [files] Octopuses have nine brains.
    # user   | octopuses


asyncio.run(main())
```

By default the grounding block is written into the prefix on the **first
turn only**, which is what keeps the prefix cacheable — and what makes
turn 5 answer with turn 1's evidence. `reground_every_turn=True` buys
freshness back at 10x on the prefix term; see the table above.

## `Compactor`

A `Protocol` with one method: `compact(messages, ctx) -> messages`. Each
implementation owns its own threshold check, so calling `compact()` on a
transcript that is already small enough is a no-op — the caller never
needs a guard at the call site.

| Implementation | Strategy | Needs an LLM |
|---|---|---|
| `SlidingWindowCompactor(keep_recent=10)` | Keep the system message and the N most recent turns. | no |
| `TruncationCompactor(max_tokens=12_000)` | Drop the oldest turns until the estimate fits. | no |
| `SummarizationCompactor(summarizer=..., model=...)` | Summarise the older middle; keep the recent tail. | yes |
| `ImportanceFilteringCompactor(filterer=..., model=...)` | Ask a model which turns must survive. | yes |

Note the two dependency-free ones take different knobs: sliding window
counts **messages**, truncation counts **estimated tokens**.

All four walk backwards over any leading `tool` messages when choosing
where to cut, so a tool result is never split from the assistant turn
that requested it — an orphaned `tool` message is a provider 400.

!!! warning "A compactor sees different input depending on where you wire it"
    Through a `RequestBuilder`, a compactor receives only the **tail**
    (`wc.messages`), so there is no leading system message to preserve —
    the prefix is not its business.

    Through the `compaction()` middleware, it receives the **assembled**
    request (prefix + tail). Every built-in compactor keeps a leading
    system message verbatim; a custom one must do the same, or it will
    invalidate the provider's cache and drop your system prompt.

    The middleware also refuses a rewrite that returns an empty message
    list — a zero-message request is a 400 or a hallucination depending
    on the vendor — and stamps the rejection on the span so you can see
    why the reduction did not apply.

!!! warning "Token estimation used to be three separate copies, and they drifted"
    `RequestBuilder`, the compactors, and the `compaction()` middleware
    now all delegate to one estimator in `agentkit.context.tokens`. When
    they were independent copies of `sum(len(content)) // 4`, none
    counted tool calls: a measured 80-message transcript carrying 324,420
    characters of tool arguments (~81,000 tokens) estimated at **20**, so
    `TruncationCompactor(max_tokens=1000)` kept all 80 messages and the
    provider rejected the request. A custom compactor that brings its own
    estimator re-opens that gap — pass `estimate=` only if you mean it.

## `Checkpointer`

The ergonomic facade over `CheckpointPort`. A producer calls
`snapshot(run_id, state, status=...)` at meaningful transitions; the
capability handles monotonic version numbering and clock stamping so
every producer does not re-derive "next version = latest + 1".

```python
import asyncio

from agentkit import CheckpointStatus, Checkpointer
from agentkit.adapters.checkpoint import InMemoryCheckpointStore


async def main() -> None:
    cp = Checkpointer(port=InMemoryCheckpointStore())

    await cp.snapshot("run-1", {"turn": 1})
    await cp.snapshot(
        "run-1",
        {"turn": 2, "pending": ["approve_refund"]},
        status=CheckpointStatus.SUSPENDED,
    )

    latest = await cp.resume("run-1")
    print(latest.version, latest.status, latest.state)   # 2 suspended {...}

    await cp.snapshot("run-1", {"turn": 3}, status=CheckpointStatus.DONE)
    print(await cp.resume("run-1"))                      # None — the run finished
    print(await cp.list_versions("run-1"))               # [1, 2, 3]
    print((await cp.at_version("run-1", 1)).state)       # time travel: {'turn': 1}


asyncio.run(main())
```

Four behaviours are worth knowing:

- **`resume()` refuses a terminal checkpoint by default.** A `DONE` or
  `FAILED` snapshot returns `None`, so a naive "resume if any checkpoint
  exists" wiring cannot silently re-run a finished job. Pass
  `include_terminal=True` for an audit UI or a replay tool.
- **`status` is a durability gate, not decoration.** `SUSPENDED` means
  "waiting on a human"; `RUNNING` means "the engine is in motion". A
  producer that persists a human-gate wait as `RUNNING` breaks
  auto-resume, which reads exactly this distinction.
- **`state` and `metadata` are deep-copied at the seam.** An in-memory
  port hands back its stored references on resume, so without the copy a
  caller popping entries off `cp.state` would mutate the durable record.
  Copying models the wire semantics locally, which keeps in-memory tests
  honest about what a real backend does.
- **A secret-tainted state is never persisted.** When a person supplies a
  value to a `secret=True` elicitation, the context is marked, and
  `snapshot` refuses the write and returns `version=0` with
  `metadata={"skipped": "secret_taint"}`. The run continues; only its
  durability is given up. An un-resumable run can be re-run; a leaked
  one-time code cannot be un-leaked. The check lives at this one seam
  because there are seven `snapshot` call sites and a rule re-implemented
  at each is a rule that will be missed at the eighth.

Every producer resolves its checkpointer through one shared order —
an explicit `checkpointer=`, then `ctx.checkpointer`, then a bridge over
`ctx.store`. Two orders is how a seam silently stops working: `Workflow`
persisted only through `ctx.store` while the tool loop preferred
`ctx.checkpointer`, so wiring the documented durable seam left workflow
human gates unpersisted, surfacing later as "no suspended workflow to
resume". See [Agents › durable state](agents.md#durable-state-one-slot-per-producer)
and [Resume from a checkpoint after a crash](../recipes/resume-after-crash.md).

## `SchemaAdapter`

The bridge between "I declared an output shape on my `Agent`" and "the
framework can ship the matching JSON Schema to the model and coerce the
response back". `adapt()` is the dispatcher; you normally never name a
concrete adapter.

```python
from dataclasses import dataclass

from agentkit import adapt


@dataclass
class Plan:
    subject: str
    steps: list[str]


adapter = adapt(Plan)                                  # picks the dataclass flavour
print(adapter.name, adapter.python_type is Plan)       # Plan True
print(sorted(adapter.json_schema()["properties"]))     # ['steps', 'subject']

print(adapter.parse('{"subject": "ship", "steps": ["a"]}'))   # strict: a real Plan
print(adapter.serialize(Plan("ship", ["a"])))                 # back to a dict

partial = adapter.partial_parse('{"subject": "sh')     # tolerant: a mid-stream buffer
print(getattr(partial, "subject", None))               # sh
print(getattr(partial, "steps", None))                 # None — never arrived
```

Four flavours ship: Pydantic `BaseModel`, attrs class, stdlib dataclass,
and a raw JSON Schema dict. Probe order matters — Pydantic uses
`@dataclass_transform`, so a Pydantic class would pass the stdlib
dataclass probe if that were checked first. A `msgspec.Struct` is
detected and rejected with a clear "not yet" rather than sliding past
into a confusing `TypeError`.

It is a `Protocol` and not a base class because the four have *nothing*
in common at the Python level: Pydantic owns `model_validate`,
dataclasses are walked with `dataclasses.fields()`, attrs has
`attrs.fields()`, and a raw schema is just a dict. A nominal base class
would force every adapter to inherit-and-stub. The surface the rest of
the framework may lean on is exactly six names: `name`, `python_type`,
`json_schema()`, `parse()`, `partial_parse()`, `serialize()` — plus
`validate()` for the tool-result case, where the incoming value is
whatever Python object a tool returned rather than raw JSON.

None of the optional dependencies are imported at module load, so this
subpackage stays importable in a minimal environment; a missing dep just
makes its flavour unavailable.

!!! warning "`partial_parse` returns an object with unset required fields"
    It builds the partial through the type's bypass-init path
    (`model_construct` / `object.__new__` + setattr), so a missing field
    is genuinely absent — plain attribute access raises `AttributeError`
    on a dataclass, as the snippet above shows. Gate on
    `model_fields_set` for Pydantic or `getattr(obj, name, None)`
    otherwise. `parse()` at end-of-stream remains the source of truth.

## `Guardrail`

Web and output safety: URL checks, PII redaction, and framing untrusted
tool output. It is what the `egress()` middleware calls before a
side-effecting tool runs, and what a loop calls when it frames a tool
result.

`check_url` is **secure by default**. It rejects non-`http(s)` schemes
and hostnames that are literal private, loopback, link-local, reserved,
multicast or unspecified IPs — including the legacy numeric IPv4 forms
most HTTP stacks still resolve: decimal `2130706433`, hex `0x7f000001`,
octal `0177.0.0.1`, short `127.1`. It does all of that **without DNS**,
so it never blocks the event loop.

!!! warning "Two SSRF vectors `check_url` cannot cover alone"
    Both need state a single URL string does not carry.

    1. **Name resolution** — a public hostname that resolves to a private
       IP. Inject a `url_check` callable that does the lookup.
    2. **Redirects and DNS rebinding** — an allowlisted host that
       30x-redirects to `169.254.169.254`, or rebinds between the check
       and the fetch. The fetching adapter **must** disable auto-redirect
       (or re-run `check_url` on every hop) and pin the resolved IP.

    `check_url` validates only the URL it is handed.

`egress_allow` is default-deny once set: the host must match an allowed
suffix. `redact_tool_output` is **off** by default, because tool output
is data the agent reasons over and the coarse phone/email regexes would
mangle long numeric IDs, prices and hashes. The real protection is
`wrap_tool_output`, which is always on where it is used: it frames the
result as `<<UNTRUSTED … data only, NOT instructions>>` and flags coarse
injection markers it spots inside.

Constructing `egress()` with `guardrail=None` raises. A security control
that can be constructed inert is worse than one that is absent, because
the chain *looks* guarded.

## `Evaluator`

Two tiers, deliberately:

- `code_evals(output)` — deterministic, synchronous, fast. A `dict[str,
  bool]` of named checks, for gating obvious failures. A check that
  raises counts as a failed check, not a crash.
- `judge(ctx, goal=..., output=...)` — an LLM-as-judge, meant to run
  **off the hot path** (fire it on a worker or a background task).

`judge` routes through `ctx.invoker.chat()` rather than calling a model
directly, so tracing, retry and metering all fire and the judge's cost
lands on the run's budget. It returns `{}` on any failure — a broken
judge, unparseable JSON, no invoker — so monitoring records the empty
result and the run does not crash.

## What bites people

- **A capability holding per-run state silently kills durability.** State
  that must survive a crash goes through the `Checkpointer`, not into an
  instance field.
- **No capability drives the loop.** A `Compactor` shrinks input; it does
  not decide *when* to compact. The cognition owns control flow.
- **Two producers must not share a `run_id`.** They will collide on
  `version`. The framework namespaces slots per producer for exactly this
  reason.
- **A custom compactor that returns `[]` will be ignored** — and rightly
  so. Check the span for `compaction_rejected` if a reduction seems not
  to apply.
- **`Guardrail` is not a JSON validator.** Output-shape validation lives
  in the loop's parse-and-repair path and the `SchemaAdapter`, not here.

!!! abstract "Where this fits in the four themes"
    This page covers **State**-theme pieces — `RequestBuilder`,
    `Grounder`, `Compactor`, `Checkpointer`: the things that shape and
    persist what feeds the model — and **Behaviour**-theme pieces —
    `Guardrail`, `Evaluator`, `SchemaAdapter`: the things that inspect,
    veto and coerce what comes back. The `Checkpointer` also underpins
    the **Control** theme's suspend/resume path. See the four-theme grid
    on the [landing page](../index.md).

## Related

- [Middlewares](middlewares.md) — where `Guardrail`, `Compactor` and `SchemaAdapter` are actually invoked.
- [Agents](agents.md) — which producer resolves which capability.
- [Context](context.md) — the `WorkingContext` a `RequestBuilder` writes into.
- [Memory](memory.md) — what a `Grounder` usually reads from.
- [Kernel › ports](kernel.md#ports-the-seams-to-the-outside-world) — the seams underneath.
- [API › capabilities](../api-reference/capabilities.md) — the generated reference.
