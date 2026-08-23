# Mental models

Four worked scenarios that take a plausible product, build it out of
agentkit primitives, and then ask what has to be true for it not to
break.

They are the longest documents on this site — 500 to 800 lines each —
and they are the only place that shows the framework carrying real
weight. The [Tutorial](../tutorial.md) shows you a working agent; these
show you what a working agent looks like at 10,000 rows, five tenants,
or a fan-out of children where one of them fails.

!!! note "These are design narratives, not copy-paste code"
    The code blocks are sketches and expected-value snapshots. They
    contain deliberate placeholders (`state={"prefix": <serialized
    prompt prefix>, …}`) because their job is to show you the shape of
    what comes back, not to run. For runnable code, use the
    [Tutorial](../tutorial.md), the [Recipes](../recipes/index.md) or
    the [Examples](../examples.md).

Each scenario was chosen to stress a *different* set of framework
guarantees, so between them they cover the four themes without
overlapping much.

## The four scenarios

<div class="grid cards" markdown>

-   __01 · Multi-tenant "Chat with Docs"__

    ---

    Many customers served from one process, and no customer's document
    ever reaching another's context — even with a warm cache, even
    under concurrent load.

    **Exercises:** `Scope`, `ScopedMemory`, `CachedMemory`,
    `VectorMemory`, per-tenant `Quota`, `FunctionTool.caps`

    [:octicons-arrow-right-24: Read it](01-multi-tenant-chat-with-docs.md)

-   __02 · Autonomous DevOps investigator__

    ---

    An oncall AI that reads production logs and proposes a rollback —
    and structurally cannot apply one without a named human saying yes.

    **Exercises:** `RunPolicy` (the lethal trifecta),
    `requires_approval`, `Autonomy` tiers, `Suspended` + `resume`,
    `CancellationToken`, termination composition

    [:octicons-arrow-right-24: Read it](02-autonomous-devops-investigator.md)

-   __03 · Long-running data enrichment__

    ---

    10,000 rows, two hours, a cheap worker that gets OOM-killed at row
    4,100 — and a resume that neither re-pays for finished rows nor
    overshoots the cap.

    **Exercises:** `Workflow`, `Checkpointer`, `Budget` under
    concurrency, `Usage.__add__` associativity, `idempotency_key` +
    `StorePort.get_or_set`, `retry`, `Failure`

    [:octicons-arrow-right-24: Read it](03-long-running-data-enrichment.md)

-   __04 · Multi-agent coordinated research__

    ---

    A planner, five parallel researchers, a synthesizer and a critic,
    streaming to one live canvas — isolated where isolation matters,
    shared where it doesn't.

    **Exercises:** `CoordinatorCognition`, `Skill.as_agent()`,
    `SignalChannel` + `MergeInbox`, `ActorBudget`, `run_agents` /
    `gather_bounded`, the `merge` stream operator

    [:octicons-arrow-right-24: Read it](04-multi-agent-coordinated-research.md)

</div>

## Which one to read

**If you are building something**, read the one whose shape matches
your product. They are independent; there is no reading order.

| If your thing is… | Read |
|---|---|
| RAG or chat over documents, especially for more than one customer | [01](01-multi-tenant-chat-with-docs.md) |
| An agent with real-world authority — deploys, payments, deletions | [02](02-autonomous-devops-investigator.md) |
| A batch job, a pipeline, anything that runs for hours | [03](03-long-running-data-enrichment.md) |
| A team of agents, or one agent that spawns others | [04](04-multi-agent-coordinated-research.md) |

**If you are evaluating the framework**, read
[02](02-autonomous-devops-investigator.md) first. It is the one that
most clearly shows what agentkit is for: the safety story from an
untrusted alert all the way to a human clicking approve, with the
failure analysis of every step in between.

**If you are learning the framework**, read them after the
[Tutorial](../tutorial.md) and at least one
[Concepts](../concepts/kernel.md) page. They assume you know what a
cognition and a middleware chain are, and spend their length on
consequences rather than definitions.

**If you are changing the framework**, read the one that covers the
code you are touching — the section below maps directories to
documents — and treat its correctness checklist as a pre-commit list.

| Touching | Read |
|---|---|
| `memory/`, `context/` | 01 |
| `agents/control/`, `RunPolicy` | 02 |
| `capabilities/checkpointer/`, `runtime/meter.py`, `Workflow` | 03 |
| `agents/policies/`, `agents/cognition/`, `agents/control/channel.py`, `skills/` | 04 |
| `kernel/` | all four — the kernel is load-bearing everywhere |

## What is inside each one

The four documents share a structure, so once you have read one you can
navigate the others.

- **Problem** and **user experience** — the product, in a paragraph, and
  what a person actually sees. Concrete enough to disagree with.
- **How it actually works end-to-end** — the longest section. The run,
  traced through the primitives, in order.
- **Composition** — the wiring as a diagram: which object holds which,
  and where the seams are.
- **The primitives it exercises**, and — more usefully —
  **what it deliberately doesn't use**, with the reason. The second
  table is the better answer to "do I need this primitive?" than any
  feature list.
- **Where it can fail** — enumerated failure modes with the mechanism,
  not a hand-wave.
- **Expected output on a successful run** — what the `AgentResult`,
  the `Checkpoint`, the budget ledger and the signal queues actually
  contain. This is the section to read if you want to know what a
  primitive returns without reading its source.
- **Invariants** — property by property, each with the concrete failure
  if it slips and the test that locks it in. Where there is no test
  yet, the row says "gap" rather than pretending.
- **Verification protocol** and **correctness checklist** — the greps,
  files and one-off scripts to run before you commit a change near
  those invariants, plus a snapshot of what they printed when the
  document was written.
- **Design tensions** — the decisions that were genuinely live during
  design, recorded so a future maintainer sees why the code is shaped
  the way it is instead of assuming it is arbitrary.
- **Reverse view** — "if X breaks in production, which framework
  invariant failed?" The map from a user-visible bug back to the
  contract that should have prevented it.

## Adding a fifth

New scenarios are welcome if they expose invariants the four here do
not. Before adding one, check whether your feature genuinely stresses
primitives beyond scope + HITL + durability + coordination. If it is a
variant of an existing shape, extend the existing document instead —
the reverse-view section is designed to accumulate.
