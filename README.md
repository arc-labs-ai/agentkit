<div align="center">

# agentkit

**A model-neutral runtime and harness foundation for Python agents.**

[![CI](https://github.com/arc-labs-ai/agentkit/actions/workflows/ci.yml/badge.svg)](https://github.com/arc-labs-ai/agentkit/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/arc-agentkit)](https://pypi.org/project/arc-agentkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/arc-agentkit)](https://pypi.org/project/arc-agentkit/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)

[Docs](https://arc-labs-ai.github.io/agentkit/) · [Quickstart](#quickstart) · [Examples](./tests/examples) · [Changelog](./CHANGELOG.md)

</div>

---

## What agentkit provides

Agentkit has one execution runtime: `CallRuntime`. The `AgentRunner` builds a host harness on that runtime, while minds define how an agent reasons. Providers implement the transport contract; tools, context policies, scoped memory, and EventHub subscribers plug into the harness.

The runtime owns retries, limits, pricing, structured output, continuation, and call events. `AgentRunner` owns agent definitions, prompt composition, tool grants, skills, delegation, memory recall, and verification. Hosts own workflow orchestration, durable business state, and domain grading.

The package core has no required runtime dependencies. Install Python 3.12 or newer:

```bash
pip install arc-agentkit
```

Optional extras provide HTTP model providers, MCP, persistence adapters, and OpenTelemetry integration:

```bash
pip install "arc-agentkit[calls]"           # HTTP/CLI calls, schemas and MCP tools
pip install "arc-agentkit[postgres]"         # Postgres and pgvector adapters
pip install "arc-agentkit[redis]"            # Redis adapter
pip install "arc-agentkit[observability]"    # OpenTelemetry exporters
```

## Quickstart

A small host wires an adapter, a price table, and an agent definition. The same `AgentRunner` works with API and CLI bindings supported by the runtime.

```python
import asyncio

from agentkit.agents import AgentDefinition, AgentRunner, InstructionSet
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, Mode, PriceTable


async def main():
    prices = PriceTable.default()
    async with CallRuntime(pricing=prices) as runtime:
        runner = AgentRunner(runtime)
        answer = await runner.run_agent(
            AgentDefinition("answerer", "1", "Answer briefly."),
            InstructionSet("What is 6 times 7?"),
            {},
            binding=Binding(Mode.OPENAI_API, "gpt-4.1-mini"),
            call_id="quickstart-1",
        )
        print(answer.text)


asyncio.run(main())
```

See the [`tests/examples/` index](./tests/examples/README.md) for examples in separate directories, including the [support harness](./tests/examples/support_harness) and [plugin execution](./tests/examples/plugin_execution/README.md) with independent tools, skills and hooks.

## Package boundaries

| Package | Responsibility |
| --- | --- |
| `models` | Provider contracts, transport adapters, capabilities, schemas, and pricing |
| `agents` | Agent definitions, minds, prompts, skill resources, and `AgentRunner` |
| `core` | CallRuntime, tool execution, events, serialization, and shared value contracts |
| `state` | Working and frozen context, context policies, and event journals |
| `tools` | Tool declarations, registries, execution policy, and filesystem tools |
| `integrations` | MCP and optional observability adapters |

Validation lives under `tests/source`, live provider evaluations under `tests/evals`, and examples under `tests/examples`.
