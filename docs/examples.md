# Examples

End-to-end demos live in the
[`examples/`](https://github.com/arc-labs-ai/agentkit/tree/main/examples)
folder in the repo. Each is a small, self-contained script that runs
against a `FakeLLM` — no API keys required — and demonstrates one
shape of composition. The folder currently ships three:

- **`01_single_agent.py`** — the shortest useful program. One
  `Agent`, one prompt, one call. Prints the typed `AgentResult`.
- **`02_streaming_and_tools.py`** — a `ReActCognition` agent with
  two `@tool`-decorated functions, driven token-by-token through
  `Agent.stream(...)`.
- **`03_composed_middlewares.py`** — a custom middleware chain
  (`tracing → meter → retry`) wired into an `Invoker`, showing the
  seams the `Agent` normally composes for you.

## Running an example

```bash
git clone https://github.com/arc-labs-ai/agentkit
cd agentkit
uv sync
uv run python examples/01_single_agent.py
```

Every example uses `FakeLLM` from `agentkit.testing`, so `uv sync`
is the only prerequisite. To point one at a real provider, swap
`FakeLLM(...)` for one of the batteries-included preset clients
(`claude(api_key=...)`, `openai(api_key=...)`, etc.) and provide
the corresponding API key.

## Related deep dives

The
[mental models](https://github.com/arc-labs-ai/agentkit/tree/main/docs/mental-models)
walk through four longer scenarios that stress different framework
invariants — read them alongside the examples for the *why* behind
the composition each demo picks.
