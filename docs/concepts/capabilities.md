# Capabilities

!!! abstract "Where this fits in the four themes"
    This page covers primitives from the **State** theme
    (`RequestBuilder`, `Grounder`, `Compactor` and its four
    implementations, `Checkpointer` — the pieces that shape and
    persist the state feeding the model) and the **Behaviour** theme
    (`Guardrail`, `Evaluator`, `SchemaAdapter` — the pieces that
    inspect, veto, and coerce what the model produces). The
    `Checkpointer` also underpins the **Control** theme's HITL
    suspend/resume path. See the four-theme grid on the
    [landing](../index.md).

**What this is.** A capability is an *optional collaborator* an agent
can be wired with — a `RequestBuilder` that assembles a prompt from
sources, a `Compactor` that shrinks a growing context window, a
`Guardrail` that vetoes a response, an `Evaluator` that scores it, a
`Checkpointer` that snapshots the run so a crash can be resumed.
Each is a small, typed seam with one or more implementations shipped
in `agentkit.capabilities`. Two of them — `Compactor` and
`SchemaAdapter` — are literal `Protocol`s, so any object with the right
shape satisfies them. The rest are concrete base classes you subclass
or configure:

| Capability | Kind | Substituted by |
|---|---|---|
| `Compactor` | `Protocol` | structural typing — any matching object |
| `SchemaAdapter` | `Protocol` | structural typing |
| `RequestBuilder` | class | subclass, or configure the shipped one |
| `Guardrail` | class | subclass |
| `Evaluator` | class | subclass (`code_evals` / `judge`) |
| `Checkpointer` | class over `CheckpointPort` | swap the **port** |

`Checkpointer` is the one worth understanding: it is a thin facade, and
the substitutable seam beneath it is `CheckpointPort` — in-memory
today, Postgres or Redis tomorrow — so you swap the port, not the
capability.

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
`agentkit`. `Guardrail` and `Evaluator` ship as base classes to subclass;
`Checkpointer` is a concrete facade whose `CheckpointPort` is the part
you swap.

```python
from agentkit.capabilities import SlidingWindowCompactor, TruncationCompactor

# Dep-free strategies — take a budget directly.
compactor = SlidingWindowCompactor(keep_recent=10)
# or:
compactor = TruncationCompactor(max_tokens=12_000)

# Hand it to a RequestBuilder as `compactor=`; the RequestBuilder
# folds the transcript through it before every LLM call.
```

The LLM-driven strategies (`SummarizationCompactor`,
`ImportanceFilteringCompactor`) also require a `summarizer=` /
`filterer=` LLM port to do their work.

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
