# Examples

End-to-end demos live in the
[`examples/`](https://github.com/arc-labs-ai/agentkit/tree/main/examples)
folder in the repo. Six shipped scripts, each self-contained:

- **`01_single_agent.py`** — the shortest useful program. One
  `Agent`, one prompt, one call. Prints the typed `AgentResult`.
- **`02_streaming_and_tools.py`** — a `ReActCognition` agent with
  two `@tool`-decorated functions, driven token-by-token through
  `Agent.stream(...)`.
- **`03_composed_middlewares.py`** — a custom middleware chain
  (`tracing → meter → retry`) wired into an `Invoker`, showing the
  seams the `Agent` normally composes for you.
- **`04_claude_cli.py`** — delegates the whole agent loop to a
  locally-installed `claude` CLI via `ClaudeCliCognition`. Uses
  whatever auth the CLI already resolved (`~/.claude/` login,
  `CLAUDE_CODE_OAUTH_TOKEN`, or `ANTHROPIC_API_KEY`) — no key on
  agentkit's side. Skips cleanly if `claude` isn't on PATH.
- **`05_streaming_typed_output.py`** — renders a typed object *while
  the model is writing it*, via `StreamEvent.partial_output` off the
  public `Agent.stream`. Shows the consumer contract (required fields
  may be unset) and the negative case: an agent with no output schema
  is completely unaffected.
- **`06_hitl_budget_and_capabilities.py`** — three things that
  compose: a capability mismatch refused at construction before any
  spend; a gated tool call that **parks in place** on an injected
  `Asker` instead of unwinding; and a budget that stops on a verdict
  and writes a current checkpoint, so the spend is recoverable.

Examples 01-03, 05, and 06 run against `FakeLLM` — no API keys or
external binaries required (05 needs `pydantic`, so `uv sync
--all-extras`). Example 04 requires the `claude` CLI installed (from
Anthropic) and an already-authenticated Claude Code session.

## Running an example

```bash
git clone https://github.com/arc-labs-ai/agentkit
cd agentkit
uv sync
uv run python examples/01_single_agent.py
```

For 01-03, 05 and 06, `uv sync --all-extras` is the only
prerequisite. To point them at a
real provider, swap `FakeLLM(...)` for one of the batteries-included
preset clients (`claude(api_key=...)`, `openai(api_key=...)`, etc.)
and provide the corresponding API key. Example 04 works out of the
box once you have `claude` installed and logged in.

## Related deep dives

The
[mental models](https://github.com/arc-labs-ai/agentkit/tree/main/docs/mental-models)
walk through four longer scenarios that stress different framework
invariants — read them alongside the examples for the *why* behind
the composition each demo picks.
