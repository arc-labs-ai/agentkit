# Mental Models for Checking agentkit Correctness

Four use cases, each a plausible product built on agentkit, each chosen
to stress a **different set of framework invariants**. The docs are
not implementation guides — they are the *thinking* that lets us look at
agentkit and say "the framework must guarantee X because of Y" instead
of just "the tests pass".

Read a doc when you're about to touch code near the primitives it covers.
The correctness checklist in each doc is a "grep before you commit" list.

## The four use cases

| Doc | Use case | Invariants stressed |
|---|---|---|
| **[01](01-multi-tenant-chat-with-docs.md)** | Multi-tenant "Chat with Docs" SaaS | Scope-partitioned caches, `ScopedMemory` gate, tenant `Quota`, in-process isolation |
| **[02](02-autonomous-devops-investigator.md)** | Autonomous DevOps Investigator | `RunPolicy` lethal-trifecta, `requires_approval` HITL, `Autonomy` tiers, `CancellationToken` propagation |
| **[03](03-long-running-data-enrichment.md)** | Long-running data enrichment (10k rows) | `Checkpointer` deep-copy semantics, resume-through-crash, `Usage.__add__` associativity, `idempotency_key` determinism |
| **[04](04-multi-agent-coordinated-research.md)** | Multi-agent coordinated research report | `Skill.as_agent` per-run isolation, frozen `SignalEnvelope`, per-drive termination clones, streaming via `merge` |

## What each doc gives you

- **Composition sketch** — how the agentkit primitives fit together for
  that shape of feature.
- **Invariants table** — property-by-property, with the concrete
  failure mode if the invariant slips + the existing test that locks
  it in (or "gap" if we don't have coverage yet).
- **Correctness checklist** — the greps / files / patterns to verify
  when touching code near the invariants.
- **Design tensions** — the decisions that were live during design, so
  a future maintainer sees why the code is shaped the way it is.
- **Reverse view** — "if X breaks in production, which framework invariant
  failed?" — the map from user-visible bug to framework contract.

## When to read which doc

- **Touching `memory/` or `context/`** → doc 01 (tenant + memory story).
- **Touching `agents/control/` or `RunPolicy`** → doc 02 (security + HITL).
- **Touching `capabilities/checkpointer/`, `runtime/meter.py`, or `Workflow`**
  → doc 03 (durability + cost).
- **Touching `agents/policies/`, `agents/cognition/`, `agents/control/channel.py`,
  `skills/`** → doc 04 (multi-agent coordination).
- **Touching `kernel/`** → all four (kernel is load-bearing everywhere).

## Adding a fifth use case

New use cases are welcome if they expose invariants the four here don't.
Before adding: check if your feature genuinely stresses primitives
beyond scope+HITL+durability+coordination. If it's a variant of an
existing shape, extend the existing doc instead — the reverse-view
section is designed to accumulate.
