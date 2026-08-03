# Kernel

**What this is.** `agentkit.kernel` is the opinion-free foundation every
other package sits on. It defines the value types the whole framework
speaks (`ChatRequest`, `LLMResult`, `Message`, `Scope`, `Usage`), the
`Port` protocols that let you swap providers, the `middleware`
contract that every cross-cutting concern conforms to, and the small
library of concurrency and resilience primitives that make async code
safe to compose.

**Why it exists.** Every framework that lets its "core" import a
provider client eventually smuggles opinions into places you can't
easily replace. agentkit's rule is that the kernel imports *no
provider*, holds *no policy*, and grows *no loop*. It is the shared
vocabulary; policy is what higher layers add.

## What lives in the kernel

- **Value types** — immutable dataclasses / models for the shapes that
  cross every boundary: `Message`, `ToolCall`, `ToolSchema`,
  `ChatRequest`, `ToolRequest`, `LLMResult`, `StreamEvent`, `Usage`,
  `Scope`, `Operation`. (`Budget` is a runtime concern, not a kernel
  type — see [Runtime](runtime.md).)
- **Ports** — Protocols for external services: `LLMPort`, `StorePort`,
  `TracePort`, `ObserverPort`. Every adapter (in `agentkit.adapters/`)
  implements one of these.
- **Middleware contract** — the `Call` envelope, `Middleware` Protocol,
  and `chain(...)` function that turn a list of middlewares plus a
  terminal handler into a single chain.
- **Concurrency** — `CancellationToken`, `gather_bounded`,
  `gather_best_effort`, `run_agents`, and `run_sync`.
- **Observation** — the shape of the trace / metric events every layer
  emits, so observers stay adapter-agnostic.

## The invariants it enforces

1. **No provider dependency.** Importing `agentkit.kernel` never
   triggers an import of `httpx`, an SDK, or a database driver.
2. **Values are immutable.** Kernel dataclasses are frozen where
   possible; middleware cannot mutate a `Call` in place — it constructs
   a new one and forwards.
3. **One vocabulary.** Every layer that talks about a "usage" or a
   "budget" reuses the kernel type. There is no parallel `Usage` type
   in `runtime/` or `middlewares/`.

## Related deep dive

See the internal
[mental models](https://github.com/arc-labs-ai/agentkit/tree/main/docs/mental-models)
for the reasoning behind these invariants and the concrete failure
modes when they slip. The kernel is load-bearing in all four models.

## API

Full generated reference lives at
[API › kernel](../api-reference/kernel.md).
