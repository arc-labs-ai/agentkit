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

Capabilities are wired at agent construction, not discovered at
runtime:

```python
from agentkit import Agent, ReActCognition
from agentkit.capabilities import (
    LLMCompactor, ContentGuardrail, InMemoryCheckpointer,
)

agent = Agent(
    name="researcher",
    cognition=ReActCognition(),
    compactor=LLMCompactor(model="claude-haiku-4-5", target_tokens=8000),
    guardrail=ContentGuardrail(...),
    checkpointer=InMemoryCheckpointer(),
)
```

None of these are required; an `Agent` built with none of them is
still a valid agent.

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
