# Recipes

Grouped by the problem you're trying to solve, not by the primitive
you're trying to use. Each recipe is a self-contained page: the
question, one runnable script, an explanation, gotchas.

If you want a linear walkthrough, use the [Tutorial](../tutorial.md);
if you want the mental model of a primitive, use the
[Concepts](../concepts/kernel.md); if you're about to *write* code and
want the tightest possible menu, use the
[Cheatsheet](../cheatsheet.md). This page is what you reach for after
you've hit one specific wall.

## I need to wire a provider from configuration

Read the key from the environment, pick the provider from the model
name, and refuse a model that can't do the job — before any spend.
One registry: `resolve_llm`, `from_env`, `Capability`, `requires=`.

[Pick a provider from config, catch a bad model :octicons-arrow-right-24:](provider-from-env.md)

## I need to render a typed object while it's still being written

`output=MyModel` gives you `AgentResult.parsed` at the end.
`StreamEvent.partial_output` gives you the in-progress object on every
delta, through `Agent.stream` alone.

[Stream a typed object as it generates :octicons-arrow-right-24:](stream-typed-output.md)

## I need to control cost

Hard ceiling on a single run and a rolling window per tenant.
`Budget`, `Quota`, `MeterExceeded`, `meter()` middleware.

[Cap spend with Budget and Quota :octicons-arrow-right-24:](spend-budget-and-quota.md)

## I need to pause for a human

Any tool that mutates the world — publishing, spending, sending —
should not run without a human saying yes. Real pause, snapshot,
resume from a fresh process.

[Human-in-the-loop tool approval :octicons-arrow-right-24:](hitl-tool-approval.md)

## I need a person to supply a *value*, not just say yes

A one-time code mid-journey, nowhere near a tool call. Parkable in
place (your live state survives), deadlined (no abandoned-tab hang),
typed (who answered, and when), and secret-safe.

[Elicit a value from a human :octicons-arrow-right-24:](elicit-a-value-from-a-human.md)

## I need to survive a crash

The worker died mid-flight. A fresh worker picks up where the last
one left off — same `run_id`, same `CheckpointPort`, hydrated
transcript.

[Resume from a checkpoint after a crash :octicons-arrow-right-24:](resume-after-crash.md)

## I need parallel agents that fail-fast together

Fan out N children under a shared budget + cancel. One failure
cancels the siblings; `best_effort=True` isolates failures instead.

[Parallel agents with cooperative cancellation :octicons-arrow-right-24:](parallel-agents-with-cancellation.md)

## I need to plug in a custom concern

Redact a secret before it goes on the wire. Time every call. Cache
on a custom key. Two shapes: `BaseMiddleware` for
transform/guard/observe; raw `(call, next)` for
resilience/caching/instrumentation.

[Write a custom middleware :octicons-arrow-right-24:](custom-middleware.md)

## I need OpenTelemetry

Real traces and metrics in Tempo / Jaeger / Datadog / Honeycomb —
any OTLP backend. `TracePort` + `MetricsPort`, one adapter each in
`arc-agentkit[observability]`.

[Wire OpenTelemetry :octicons-arrow-right-24:](otel-tracing-and-metrics.md)

## I need to use my local Claude Code (no API key)

`ClaudeCliCognition` subprocesses your locally-installed `claude`
CLI per run. Auth is the CLI's problem; you get its full tool surface
and permission modes.

[Plug the claude CLI into FastAPI code-gen :octicons-arrow-right-24:](claude-cli-fastapi-code-gen.md)

## I need to consume any MCP server

`MCPClient(StdioServer(...))` plus three adapters that surface an
MCP server's tools, resources, and prompts through agentkit's
canonical `Tool`, `MemorySource`, and `Prompt` Protocols.

[Consume MCP tools from an agent :octicons-arrow-right-24:](mcp-tools.md)
