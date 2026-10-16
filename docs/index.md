# agentkit

agentkit is a model-neutral Python framework for building agent harnesses. It has one model execution path: `CallRuntime`. `AgentRunner` composes agent instructions, context, tools, skills, delegation, and verification on top of that runtime.

The host application owns workflow orchestration, durable business state, and domain-specific grading. agentkit supplies the model call, tool loop, events, continuation, accounting, and hooks that a host can compose into its harness.

## The runtime and the harness

Use `CallRuntime.ask()` for a single request or `CallRuntime.run()` for a bounded tool loop. Use `AgentRunner.run_agent()` when you want agent definitions, minds, prompt composition, scoped tools, and memory providers. Both call the same runtime and return the same `Answer` and `Usage` contracts.

```python
import asyncio

from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, Mode, PriceTable


async def main() -> None:
    # Supply an adapter configured for the selected mode/model.
    adapter = ...  # A provider adapter configured for the selected model.
    async with CallRuntime(adapters={Mode.OPENAI_API: adapter}, pricing=PriceTable.default()) as runtime:
        runner = AgentRunner(runtime)
        answer = await runner.run_agent(
            AgentDefinition(
                id="assistant",
                version="1",
                description="Answer clearly and accurately.",
            ),
            InstructionSet("What does this service do?"),
            {},
            binding=Binding(Mode.OPENAI_API, "your-configured-model"),
            call_id="docs-example",
        )
        print(answer.text)


asyncio.run(main())
```

The package supports Anthropic API, OpenAI API, Claude Code CLI, and Codex CLI adapters. Runtime behavior belongs in the shared runtime; adapters translate provider requests and responses. Provider-specific capability limits are surfaced explicitly.

## What plugs into the harness

- **Minds** define reasoning strategy independently of model transport.
- **Tools** are declared with side-effect metadata, validated, scoped, and dispatched by the current runtime path.
- **Context** travels with messages and can be transformed by a context policy between slices or before resume.
- **Memory providers** contribute untrusted context during prompt composition.
- **Skills** provide approved resources and can emit lifecycle events.
- **EventHub** exposes call and agent events to host subscribers. Subscribers can opt into isolated failure handling.
- **Pricing, limits, output validation, retries, and continuation** are configured through the runtime call contract.

Runnable examples live under `tests/examples`; offline tests live under `tests/source`, and provider-backed evaluations live under `tests/evals`.

## Development checks

```sh
uv run pytest -q
uv run ruff check agentkit tests
```

Provider-backed checks are opt-in and documented in `tests/evals/README.md`. Their reports state which modes, models, and scenarios were actually exercised.

## Project structure

| Path | Responsibility |
| --- | --- |
| `agentkit/agents` | Agent definitions, minds, prompt composition, and `AgentRunner` |
| `agentkit/core/runtime` | `CallRuntime`, call lifecycle, accounting, and events |
| `agentkit/models/providers/frontiers` | Direct API adapters |
| `agentkit/models/providers/cli` | Installed Claude Code and Codex CLI adapters |
| `agentkit/tools` | Tool definitions, execution, policy, and hosted tools |
| `agentkit/state/context` | Immutable working context, context policies, and memory providers |
| `tests/source` | Offline source and conformance tests |
| `tests/evals` | Opt-in real-provider evaluations |
| `tests/examples` | Runnable usage examples |
