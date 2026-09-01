# Examples

Six self-contained scripts live in the
[`examples/`](https://github.com/arc-labs-ai/agentkit/tree/main/examples)
folder of the repo. Unlike the [Tutorial](tutorial.md), which builds one
agent across six steps, each of these is a standalone demonstration of
one thing you can read top to bottom.

Five of the six run offline against `FakeLLM` — no API key, no network.

| Script | Shows | Needs |
|---|---|---|
| `01_single_agent.py` | The shortest useful program: one `Agent`, one prompt, one call, a typed `AgentResult`. | nothing |
| `02_streaming_and_tools.py` | A `ReActCognition` agent with two `@tool` functions, driven token-by-token through `Agent.stream(...)`. | nothing |
| `03_composed_middlewares.py` | A hand-built `tracing → meter → retry` chain on an `Invoker` — the seams the `Agent` normally composes for you. | nothing |
| `04_claude_cli.py` | Delegating the whole loop to a locally-installed `claude` CLI via `ClaudeCliCognition`. | the `claude` CLI |
| `05_streaming_typed_output.py` | Rendering a typed object *while the model is writing it*, via `StreamEvent.partial_output`. | `pydantic` |
| `06_hitl_budget_and_capabilities.py` | Three things composing: a capability mismatch refused at construction before any spend, a gated tool that **parks in place** on an injected `Asker`, and a budget that stops on a verdict and writes a recoverable checkpoint. | nothing |
| `07_codex_cli.py` | The same delegation to a locally-installed `codex` CLI via `CodexCliCognition` — and the difference that matters: containment is an OS **sandbox**, not a tool list, shown by running one instruction under two of them. | the `codex` CLI |

Three of them are worth calling out because they demonstrate behaviour
the docs otherwise only assert:

- **05** shows the negative case as well as the positive one: an agent
  with no output schema is completely unaffected by `output_coerce()`,
  and partials only flow when that middleware is in the chain.
- **06**'s gated tool *parks* rather than suspending. With an `Asker`
  wired into `Services`, the cognition awaits the human in place instead
  of checkpointing and unwinding — the other half of the HITL story from
  [tutorial step 5](tutorial.md#step-5-make-it-ask-before-it-acts).
- **07** runs the *same* instruction under `sandbox="read-only"` and
  `sandbox="workspace-write"` and prints whether the file appeared. The
  [Codex CLI page](concepts/codex-cli.md) claims the sandbox is real
  containment rather than a declared policy; this is the claim executing.

## Running them

```bash
git clone https://github.com/arc-labs-ai/agentkit
cd agentkit
uv sync
uv run python examples/01_single_agent.py
```

`uv sync` is enough for every example except `04` and `07`. It installs
the dev group, which pulls in `pydantic` (transitively, via the `mcp`
extra) — that is what `05` needs.

`04_claude_cli.py` additionally needs the `claude` CLI on your `PATH`
and an authenticated Claude Code session; it uses whatever auth the CLI
already resolved (`~/.claude/` login, `CLAUDE_CODE_OAUTH_TOKEN`, or
`ANTHROPIC_API_KEY`) rather than a key held by agentkit. If `claude`
isn't installed it prints a note and exits cleanly, so it is safe in CI.

`07_codex_cli.py` is the same arrangement for the `codex` CLI, using
whatever auth *it* resolved (`$CODEX_HOME` login, `CODEX_API_KEY`, or
`OPENAI_API_KEY`). It also initialises a throwaway git repo, because
`codex` refuses to run outside one unless told to skip the check. Same
clean exit when the binary is missing.

To point any of the others at a real provider, replace the `FakeLLM(...)`
construction with a preset — `claude(api_key=...)`, `openai(api_key=...)`
— and supply the matching key. Nothing else in the script changes;
[Getting started](getting-started.md#the-same-agent-against-a-real-provider)
shows that wiring in full.

## Going deeper

The [mental models](mental-models/README.md) are the next level up: four
worked product scenarios — multi-tenant RAG, an autonomous DevOps
investigator, a 10,000-row batch job, and a coordinated research team —
that put these primitives under real load and name what breaks when each
guarantee slips. Read them alongside the examples for the *why* behind
the composition each script picks.
