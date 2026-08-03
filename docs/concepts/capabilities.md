# Capabilities

**What this is.** A capability is an *optional collaborator* an agent
can be wired with — a `RequestBuilder` that assembles a prompt from
sources, a `Compactor` that shrinks a growing context window, a
`Guardrail` that vetoes a response, an `Evaluator` that scores it, a
`Checkpointer` that snapshots the run so a crash can be resumed.
Each capability is a small, typed Protocol; each has one or more
implementations shipped in `agentkit.capabilities`.

**Why it exists.** Compaction, grounding, guardrails, and durability
are *pathologically* the wrong things to bake into a base class. They
have context-specific tradeoffs (which model? which safety policy?
which storage?), and you almost always want to swap one implementation
for another without editing the loop. Capabilities are how agentkit
keeps the loop small and the swap surface flat.

## The capabilities shipped

| Capability        | What it does                                                                     | Ships |
|-------------------|----------------------------------------------------------------------------------|-------|
| `RequestBuilder`  | Compose a `ChatRequest` from prompt + memory + tools + budget.                   | Yes   |
| `Grounder`        | Attach retrieved evidence (memory hits) into the request.                        | Yes   |
| `Compactor`       | Shrink a `WorkingContext` before it hits the model's window.                     | 4 impls |
| `Guardrail`       | Approve / block / edit an outbound message.                                      | Yes   |
| `Evaluator`       | Score a candidate output (LLM-judge or rule-based).                              | Yes   |
| `Checkpointer`    | Snapshot run state so a crash can be resumed at the last step.                   | Yes   |
| `SchemaAdapter`   | Coerce free-form model output into a typed shape (Pydantic / dataclass / dict).  | Yes   |

## How capabilities plug in

Capabilities are not `Agent` constructor kwargs. They wire in at the
edges of the loop: a `Compactor` feeds a `RequestBuilder`, a
`Guardrail` runs inside the middleware chain, a `Checkpointer` is
handed to the cognition or policy that suspends/resumes.

The shipped compactor strategies are
`ImportanceFilteringCompactor`, `SlidingWindowCompactor`,
`SummarizationCompactor`, and `TruncationCompactor` — all in
`agentkit`. `Guardrail`, `Evaluator`, and `Checkpointer` ship as
Protocols in `agentkit.capabilities`; implementations are yours to
plug in.

```python
from agentkit import (
    RequestBuilder,
    SummarizationCompactor,
)

builder = RequestBuilder(
    compactor=SummarizationCompactor(target_tokens=8000),
)
```

None of these are required; an agent with none of them is still a
valid agent.

## The invariants it enforces

1. **Every capability is a Protocol.** No `abstract base class`
   inheritance; you implement the interface and hand it in.
2. **Capabilities are stateless per run.** State that has to survive
   goes through `Checkpointer` — an in-memory field in a capability
   silently kills durability.
3. **No capability drives the loop.** A `Compactor` shrinks input; it
   doesn't decide when to compact. The cognition owns control flow.

## API

Full generated reference lives at
[API › capabilities](../api-reference/capabilities.md).
