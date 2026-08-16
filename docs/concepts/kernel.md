# Kernel

!!! abstract "Where this fits in the four themes"
    The kernel is *underneath* all four themes — it is the shared
    vocabulary every theme builds on. **Cognition** speaks in
    `ChatRequest` / `LLMResult` / `StreamEvent`; **Control** uses
    `CancellationToken` and the concurrency primitives; **State**
    threads the `Message` / `Usage` / `Scope` value types; **Behaviour**
    is the `Call` + `Middleware` + `chain(...)` contract itself.
    See the four-theme grid on the [landing](../index.md).

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

## `Delta` vs `StreamEvent` — two levels of "streaming"

A `Delta` is one transport increment of a **single LLM response**. A
`StreamEvent` marks a step in the **run**: a token, a tool call, an
interrupt, the final result. `assemble_deltas()` reduces a list of
`Delta`s back to an `LLMResult`, so a chat result is just the
collected stream.

Both carry an in-progress typed object when an output schema is
declared: `Delta.partial` is set by the `output_coerce()` middleware,
and the cognitions forward it verbatim onto
`StreamEvent.partial_output` — that is how an application streams a
typed object through `Agent.stream` alone. See
[the recipe](../recipes/stream-typed-output.md).

`assemble_deltas` deliberately **drops** `partial`. It only ever runs
on a complete delta list, so by then there is no in-progress state
left to carry: `parsed` (the strict, validated object) is the answer.
Lifting both onto `LLMResult` would give the result type two competing
typed fields where the tolerant one — which may have unset required
fields — could shadow the strict one.

## The invariants it enforces

1. **No provider dependency.** Importing `agentkit.kernel` never
   triggers an import of `httpx`, an SDK, or a database driver.
2. **Values are immutable.** Kernel dataclasses are frozen where
   possible; middleware cannot mutate a `Call` in place — it constructs
   a new one and forwards.
3. **One vocabulary.** Every layer that talks about a "usage" or a
   "budget" reuses the kernel type. There is no parallel `Usage` type
   in `runtime/` or `middlewares/`.
4. **Additive fields only.** A new field on a value type carries a
   default and changes nothing for a consumer that doesn't read it.
   `StreamEvent.partial_output` is `None` for every unstructured run.

## Related deep dive

See the internal
[mental models](https://github.com/arc-labs-ai/agentkit/tree/main/docs/mental-models)
for the reasoning behind these invariants and the concrete failure
modes when they slip. The kernel is load-bearing in all four models.

## API

Full generated reference lives at
[API › kernel](../api-reference/kernel.md).
