# How do I run a fixed multi-step pipeline?

Some processes are not a conversation. Draft, then review, then a
person signs off, then publish — in that order, every time. A
`Workflow` is how you write that down as a graph instead of hoping a
model follows instructions.

## When you'd want this

An agent loop decides its own path. That is the right shape when the
path genuinely varies, and the wrong shape when it doesn't: a model
told "always run the checks before deploying" will, on some fraction of
runs, not. There is no retry policy for "it skipped a step", because
nothing recorded that a step existed.

Reach for a `Workflow` when:

- the order is yours to decide, not the model's;
- independent steps should run **concurrently** and you want that for
  free rather than hand-written;
- a step is a plain Python function or a tool call, not a model call at
  all;
- a **human has to sign off** partway through, and the run must survive
  the wait — including a process restart.

Nodes can still be agents. The graph is explicit control; what runs
inside a node can be as emergent as you like.

## Working code

```python
"""Runs offline: FakeLLM stands in for the model, so no API key is needed."""

import asyncio
import dataclasses

from agentkit import Agent, Workflow
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.capabilities import Checkpointer
from agentkit.testing import FakeLLM, make_test_ctx


def build() -> Workflow:
    drafter = Agent(name="drafter", model="claude-sonnet-4-6", prompt="Draft a short release note.")
    critic = Agent(name="critic", model="claude-sonnet-4-6", prompt="Name one problem with the draft.")

    wf = Workflow("release-notes", max_steps=20)
    wf.agent("draft", drafter)
    wf.agent("critique", critic, after="draft")
    wf.human_gate("approve", after="critique")
    wf.fn("publish", lambda inputs: f"PUBLISHED: {inputs['approve']}", after="approve")
    return wf


async def main() -> None:
    ctx = make_test_ctx(
        llm=FakeLLM("a fine sentence"),
        checkpointer=Checkpointer(port=InMemoryCheckpointStore()),
        correlation_id="wf-1",
    )
    wf = build()

    # Runs draft -> critique, then stops at the gate and checkpoints.
    paused = await wf.run("Announce v2.", ctx)
    print(paused.stop_reason, paused.suspended.pending)
    print("done so far:", sorted(paused.outputs))

    # A different process, hours later: same run_id, same checkpointer.
    final = await wf.resume("wf-1", {"approve": "approved by dana"}, ctx)
    print(final.stop_reason, "steps:", final.steps)
    print("publish ->", final.outputs["publish"])

    # `outputs` is frozen. Amending it means building a new result.
    try:
        final.outputs["publish"] = "tampered"
    except TypeError as exc:
        print("frozen:", exc)
    amended = dataclasses.replace(final, outputs={**final.outputs, "note": "reviewed"})
    print("amended:", sorted(amended.outputs))


asyncio.run(main())
```

Output:

```text
suspended ('approve',)
done so far: ['critique', 'draft']
complete steps: 4
publish -> PUBLISHED: approved by dana
frozen: this payload belongs to a frozen value and cannot be mutated in place. Build a new one instead: dataclasses.replace(obj, field={**obj.field, ...})
amended: ['approve', 'critique', 'draft', 'note', 'publish']
```

## How it works

You declare **nodes** and their data dependencies (`after=`). The
engine derives the schedule from those; you never write the order.

Each pass round the loop is a **wave**: every node whose dependencies
are all satisfied runs, concurrently, bounded by the run's semaphore
(`Budget.max_concurrency`, default 8). A node receives a dict of its
dependencies' outputs — `{dep_name: output}` — and returns its own
output, which is threaded onward. Usage from every node is merged into
one `WorkflowResult.usage`.

Six node kinds:

| Kind | What runs |
|---|---|
| `wf.agent(name, agent)` | a leaf `Agent`, on a `ctx.child()` |
| `wf.coordinator(name, agent)` | a coordinator `Agent` — a whole team as one node |
| `wf.fn(name, f)` | a plain function, `f(inputs)` or `f(inputs, goal)`, sync or async |
| `wf.tool(name, tool)` | a `Tool`, through the invoker's tool middleware |
| `wf.human_gate(name)` | suspend, checkpoint, wait for a decision |
| `wf.subworkflow(name, child)` | another `Workflow`, nested |

`fn` nodes are how you keep the deterministic parts deterministic. If a
step is "parse this JSON and pick the highest score", that is a
function, and putting it in the graph means it is visible in the trace
alongside the model calls.

## Branching and loop-back

`route(from_, when=..., to=...)` adds a conditional edge, evaluated on a
node's output after its wave completes. Route forward to branch; route
back to an ancestor to loop:

```python
import asyncio

from agentkit import Agent, Workflow
from agentkit.testing import FakeLLM, make_test_ctx


def revise_until_clean() -> Workflow:
    drafter = Agent(name="drafter", model="claude-sonnet-4-6", prompt="Draft it.")
    critic = Agent(name="critic", model="claude-sonnet-4-6", prompt="Reply LGTM or a fix.")

    wf = Workflow("revise", max_steps=8)   # the backstop that bounds the cycle
    wf.agent("draft", drafter)
    wf.agent("review", critic, after="draft")
    # Not approved? Re-run `draft` — and everything downstream of it.
    wf.route("review", when=lambda out: "LGTM" not in out, to="draft")
    return wf


async def main() -> None:
    ctx = make_test_ctx(llm=FakeLLM("needs work"))
    result = await revise_until_clean().run("Announce v2.", ctx)
    print(result.stop_reason, result.steps)     # max_steps 8


asyncio.run(main())
```

A loop-back clears the target node **and its whole forward closure**,
so every node that depended on the re-run node re-runs too. `max_steps`
is the never-hang backstop, checked once before each wave. It can
overshoot by up to (widest wave − 1), because a wave that has started
runs whole rather than dropping siblings mid-flight — three root nodes
with `max_steps=2` report `steps=3`. Bounded, not exact.

## Durable resume across a gate

A `human_gate` node suspends the run: it checkpoints
`{goal, done, steps}` through the same `Checkpointer` seam the agent
loop uses, emits an `interrupt` observation, and returns a
`WorkflowResult` with `stop_reason="suspended"` and a `Suspended`
naming the pending gate.

`wf.resume(run_id, {gate_name: decision}, ctx)` continues from there.
The decision value **becomes the gate node's output**, so downstream
nodes can read it — which is why the `publish` node above can print who
approved. On a terminal result the checkpoint is reclaimed, so a naive
"resume if anything exists" wiring cannot replay a finished run.

`run(..., on_existing=...)` covers the other lifecycle questions:

| `on_existing` | Behaviour |
|---|---|
| `"start_fresh"` (default) | ignore any checkpoint and run from step 1 |
| `"resume"` | replay from a resumable checkpoint if there is one |
| `"fail"` | raise `CheckpointerError` if this `run_id` has ANY persisted state |

`"fail"` is the idempotency guard: use it when a queue may deliver the
same job twice.

## Gotchas

- **A gate with no durable seam suspends into a void.** With neither
  `Services(checkpointer=...)` nor `Services(store=...)` wired, the
  gate still returns a well-formed `Suspended` — and every later
  `resume` raises `no suspended workflow ... to resume`. The engine
  emits a `UserWarning` at the moment of suspend for exactly this
  reason; don't filter it.
- **`WorkflowResult.outputs` is frozen, deeply.** Writing into it
  raises `TypeError`, because the same map is what the checkpointer
  persisted — a post-hoc edit would rewrite a durable record. Use
  `dataclasses.replace(res, outputs={**res.outputs, ...})`.
- **A node output that crosses a gate must be serializable.** The
  built-in kinds are fine (`agent`/`coordinator` return `str`,
  `subworkflow` returns the child's `outputs` dict). A custom `fn` or
  `tool` node returning a live object works against `InMemoryStore` and
  fails against a real one — the failure appears on the resume, in
  another process, long after the code that caused it.
- **An `fn` node returns just its output.** `wf.fn` supplies the
  `Usage()` itself — you do not return a tuple. If a node ever hands
  the engine the wrong shape it raises naming that node, rather than
  failing as a tuple-unpacking error somewhere inside the wave loop.
- **A `tool` node cannot downgrade a tool's safety flags.**
  `side_effecting=` and `url_arg=` on `wf.tool(...)` are escalate-only:
  they can mark a tool as side-effecting or URL-bearing, never
  un-mark one that declared itself so. A graph author should not be
  able to opt a tool out of the egress check its author asked for.
- **No incremental stream.** `Workflow.run` returns a
  `WorkflowResult`; progress arrives as `ctx.emit(...)` observations
  (`run_start`, `summary` per node, `interrupt` at a gate), not as a
  token stream. Wire an `ObserverPort` if you need a progress UI.
- **`deadlock` is a real stop reason.** If no pending node has all its
  dependencies satisfied — usually a typo in `after=` — the run stops
  with `stop_reason="deadlock"` and the pending set in the emitted
  payload, rather than hanging.

## Related

- [Split work across several agents](multi-agent-coordination.md) — the
  emergent counterpart, where a policy decides who goes next.
- [Human-in-the-loop tool approval](hitl-tool-approval.md) — the same
  suspend/resume primitive, inside a tool loop instead of a graph.
- [Resume from a checkpoint after a crash](resume-after-crash.md) — the
  checkpointer these gates write through.
