# How do I resume an agent from a checkpoint after a process crash?

A long job should survive the machine it started on. When the worker
dies halfway through, the next one should carry on from where it got
to rather than starting the whole thing again.

## When you'd want this

The worker running a long tool loop dies mid-flight — OOM, deploy,
node reboot, transient upstream flap. You want the next worker to pick
up exactly where the last one left off, not re-run the tool calls that
already succeeded and not lose the transcript. That's what a
`Checkpointer` is for.

The mechanism is the same one that powers human-in-the-loop suspend —
`ReActCognition` snapshots state through the `Checkpointer` after every
successful tool-loop iteration. If a fresh `Agent` runs against the
same `run_id` and the same `CheckpointPort`, it hydrates from the
snapshot and continues from the next iteration.

!!! note "Assumes `ANTHROPIC_API_KEY` in the environment"
    Wired via `providers.claude(...)` on the `Invoker`. Swap for
    `providers.openai` (and set `OPENAI_API_KEY`) if that's what you
    have — the checkpoint plumbing is LLM-agnostic.

    To run it with **no key at all**, replace the `providers.claude(...)`
    call with a scripted `FakeLLM` from `agentkit.testing`:

    ```python
    from agentkit.kernel.types import ToolCall
    from agentkit.testing import FakeLLM, Turn

    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall(id="c1", name="search", arguments={"query": "octopus cognition"}),)),
        Turn(content="Octopuses distribute cognition across their arms."),
    ])
    ```

    Every other line is identical — that substitution is how this snippet
    is verified.

## Working code

```python
"""Requires ANTHROPIC_API_KEY in the environment."""

import asyncio
import os

from agentkit import Agent, Scope, tool
from agentkit.adapters.checkpoint import InMemoryCheckpointStore
from agentkit.adapters.llm import providers
from agentkit.agents.cognition import ReActCognition
from agentkit.capabilities import Checkpointer
from agentkit.runtime import Invoker, RunContext, Services


@tool(side_effecting=False)
async def search(query: str) -> str:
    """Web search for `query`. Returns one bulleted hit."""
    return "- 'Distributed cognition in cephalopods' (science.org, 2023)"


def build_agent() -> Agent:
    return Agent(
        name="briefer",
        model="claude-sonnet-4-6",
        prompt="Research the question. Use `search` exactly once before answering.",
        cognition=ReActCognition(tools=[search]),
    )


def build_ctx(checkpointer: Checkpointer, run_id: str) -> RunContext:
    llm = providers.claude(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )
    services = Services(invoker=Invoker(llm=llm), checkpointer=checkpointer)
    return RunContext(correlation_id=run_id, scope=Scope(), services=services)


async def main() -> None:
    port = InMemoryCheckpointStore()
    checkpointer = Checkpointer(port=port)
    run_id = "run-42"

    # ── attempt 1: worker aborts after the first tool-loop iteration ──
    # A "step" event fires right AFTER the cognition has snapshotted state
    # for that iteration. Breaking on `step` is the cleanest way to model
    # a worker that crashed just after saving its progress.
    ctx = build_ctx(checkpointer, run_id)
    agent = build_agent()
    async for ev in agent.stream("Brief me on octopus cognition.", ctx):
        if ev.type == "step":
            print(f"[crash after] {ev.text}")
            break  # simulate worker exit

    # NOT ``list_versions(run_id)``: the tool loop namespaces its slot per
    # agent, so the versions live under ``ReActCognition.checkpoint_slot(...)``.
    slot = ReActCognition.checkpoint_slot(run_id, agent.name)
    print(f"[checkpoint] versions saved: {await port.list_versions(slot)}")

    # ── attempt 2: a fresh worker picks up ────────────────────────────
    # Rebuilding the Agent + Ctx models a process restart — no in-memory
    # state carries over. The Checkpointer + run_id is the ONLY link.
    ctx2 = build_ctx(checkpointer, run_id)
    agent2 = build_agent()
    final = await agent2.run("Brief me on octopus cognition.", ctx2)
    print(f"[resumed] {final.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
```

## How it works

`Checkpointer` is a thin facade over `CheckpointPort`. On every
`snapshot(...)` it bumps the version monotonically and stamps
`created_at` from an injected `ClockPort` (or `time.time()` as a
fallback). `resume(run_id)` returns the latest **resumable** checkpoint
— by default it filters out `DONE` / `FAILED` snapshots so a naive
"resume if any checkpoint exists" wiring cannot silently re-run a
finished job.

The `ReActCognition`'s `drive(...)` calls `_load(ctx, run_id)` at
entry. When a non-suspended snapshot exists, it calls `rehydrate` to
reconstruct the `WorkingContext`, the accrued `Usage`, and the next
iteration index — then continues from there. When a **suspended**
snapshot exists, `drive` refuses to run and expects
`agent.resume(run_id, decisions, ctx)` to be called instead.

The `run_id` you pass to `agent.resume(...)` is
`RunContext.correlation_id`, and two `RunContext`s sharing a
`correlation_id` and a `Checkpointer` are the same run for this
purpose.

**The storage slot is not that id.** Each producer namespaces its own,
so a coordinator and its children cannot clobber one another:

| Producer | Slot |
|---|---|
| Tool loop | `ReActCognition.checkpoint_slot(run_id, agent_name)` → `{run_id}:agent:{name}` |
| `PlanPolicy` gate | `{run_id}:plan` |
| Coordinator policies, `Workflow` | `{run_id}` |

So `port.list_versions(run_id)` returns `[]` for a leaf agent's run —
ask for the slot instead. `resume()` re-derives it for you; only
direct `CheckpointPort` introspection needs to know.

## Gotchas

- **One producer per `run_id`.** The version numbering is monotonic
  per-run, and the port assumes a single writer. Two workers running
  the same `run_id` will collide on `version` and one write will lose.
  The next release will surface this as a clear error; today, the
  contract is on you.
- **`InMemoryCheckpointStore` doesn't survive process restarts.** For
  real durable resume, use `PostgresCheckpointStore` (extra:
  `arc-agentkit[postgres]`) or write your own `CheckpointPort` impl.
- **Terminal snapshots are hidden by default.** `resume(run_id)`
  filters `DONE` / `FAILED`. If you're building an audit view that
  needs to inspect a finished run's terminal state, pass
  `include_terminal=True`.
- **The cognition clears the checkpoint on success.** When a run
  reaches a `final` (not a suspend), `_clear(ctx, run_id)` deletes all
  versions for that run. If you need to keep them for post-hoc replay,
  wrap the `CheckpointPort` with an adapter that soft-deletes.

## Related

- [Human-in-the-loop tool approval](hitl-tool-approval.md) — the
  checkpointer's other job.
- [Concepts · Capabilities](../concepts/capabilities.md) — where
  `Checkpointer` fits alongside `RequestBuilder`, `Compactor`,
  `Guardrail`, `Evaluator`.
- `PostgresCheckpointStore` in `agentkit.adapters.checkpoint` — the
  production backend.
