# Examples

End-to-end demos live in the
[`examples/`](https://github.com/arc-labs/agentkit/tree/main/examples)
folder in the repo. Each example is a small, self-contained project
that runs against a real LLM provider and demonstrates one shape of
composition.

!!! note "Examples are being populated"
    The `examples/` folder is being built out alongside this docs site.
    If the folder is empty when you clone, check back after the next
    release, or open an issue asking for the shape you want to see.

## What to look for

- **Batteries-included chat.** The shortest possible program — one
  `Chat`, one prompt, one response.
- **Agent + tools.** A `ReActCognition` agent with two or three
  `@tool`-decorated functions.
- **Coordinated multi-agent.** Planner → parallel researchers →
  synthesizer, coordinated by a `Coordinator` cognition over a
  `SignalChannel`.
- **Durable long run.** A run with a `Checkpointer` wired in, crash-and-
  resume tested.
- **Multi-tenant chat.** `ScopedMemory` + tenant `Quota` in a single
  process serving many customers.

Each example ships with a short README explaining the invariants it
exercises and the Concepts page that covers the primitives it uses.

## Running an example

```bash
git clone https://github.com/arc-labs/agentkit
cd agentkit
uv sync
export ANTHROPIC_API_KEY=sk-...
uv run python examples/01_chat_hello.py
```

## Related deep dives

The
[mental models](https://github.com/arc-labs/agentkit/tree/main/docs/mental-models)
walk through four longer scenarios that stress different framework
invariants — read them alongside the examples for the *why* behind the
composition each demo picks.
