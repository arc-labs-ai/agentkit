# Examples

End-to-end demos live in the
[`examples/`](https://github.com/arc-labs-ai/agentkit/tree/main/examples)
folder in the repo. Each example is a small, self-contained project
that runs against a real LLM provider and demonstrates one shape of
composition. The folder currently ships three:
`01_single_agent.py`, `02_streaming_and_tools.py`, and
`03_composed_middlewares.py`.

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
git clone https://github.com/arc-labs-ai/agentkit
cd agentkit
uv sync
export ANTHROPIC_API_KEY=sk-...
uv run python examples/01_single_agent.py
```

## Related deep dives

The
[mental models](https://github.com/arc-labs-ai/agentkit/tree/main/docs/mental-models)
walk through four longer scenarios that stress different framework
invariants — read them alongside the examples for the *why* behind the
composition each demo picks.
