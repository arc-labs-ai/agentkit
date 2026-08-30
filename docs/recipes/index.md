# Recipes

One page per wall you can hit. Each is the same shape: the question in
the title, why you'd be here, one runnable script, what's actually
happening, and the things that bite people.

Every snippet on these pages runs. Where a specific provider isn't the
point of the page, the model is a `FakeLLM` from `agentkit.testing`, so
you can paste and run with no API key. Where a key really is required,
the page says so on its first line.

If you want a linear walkthrough instead, use the
[Tutorial](../tutorial.md); for the mental model of a primitive, the
[Concepts](../concepts/kernel.md); for the tightest possible menu while
you're already writing code, the [Cheatsheet](../cheatsheet.md).

Unfamiliar word? The [glossary](../glossary.md) defines every term
these recipes use in a sentence of plain English.

## Making the agent do something

<div class="grid cards" markdown>

-   __Give an agent a tool__

    ---

    Turn a Python function into something the model can call, and
    declare up front which calls mutate the world. Schemas, `ctx`
    injection, result validation, an agent as a tool.

    [:octicons-arrow-right-24: Define and register a tool](define-a-tool.md)

-   __Answer from my documents__

    ---

    Retrieval and grounding, so the answer comes from your handbook
    rather than from training data that resembles it. One
    `MemorySource` Protocol, ten backends that nest.

    [:octicons-arrow-right-24: Ground an agent in memory](ground-with-memory.md)

-   __Stream a typed object__

    ---

    `output=MyModel` gives you `AgentResult.parsed` at the end;
    `StreamEvent.partial_output` gives you the object filling in field
    by field, on every delta.

    [:octicons-arrow-right-24: Stream typed output](stream-typed-output.md)

-   __Consume an MCP server__

    ---

    Point at any published MCP server and get its tools, resources, and
    prompts through agentkit's canonical `Tool`, `MemorySource`, and
    `Prompt` Protocols.

    [:octicons-arrow-right-24: Use MCP tools](mcp-tools.md)

</div>

## Structuring the work

<div class="grid cards" markdown>

-   __Run a fixed multi-step pipeline__

    ---

    A `Workflow` graph: declared dependencies, concurrent waves,
    branching, bounded loop-back, human gates with durable resume.
    Explicit control, when the order is yours to decide.

    [:octicons-arrow-right-24: Author a workflow](workflow-graph.md)

-   __Split work across several agents__

    ---

    Coordinators, the four policies, `Handoff` routing, composable
    termination conditions. Emergent control, when who-goes-next
    depends on what was said.

    [:octicons-arrow-right-24: Coordinate a team](multi-agent-coordination.md)

-   __Run agents in parallel, cancel on failure__

    ---

    `run_agents` under a shared budget and cancel token. One failure
    stops the siblings; `best_effort=True` isolates failures into
    `Failure` objects instead.

    [:octicons-arrow-right-24: Fan out with cancellation](parallel-agents-with-cancellation.md)

</div>

## Keeping a person in the loop

<div class="grid cards" markdown>

-   __Pause a tool for approval__

    ---

    Approve, reject, or rewrite the arguments — with the run suspended
    to a checkpoint so an entirely fresh process can pick it up.

    [:octicons-arrow-right-24: Gate a tool on a human](hitl-tool-approval.md)

-   __Ask a person for a value__

    ---

    Not yes/no: a one-time code, mid-run, nowhere near a tool call.
    Parkable in place, deadlined, typed, and secret-safe.

    [:octicons-arrow-right-24: Elicit a value](elicit-a-value-from-a-human.md)

</div>

## Controlling cost and behaviour

<div class="grid cards" markdown>

-   __Cap spend__

    ---

    `Budget` for a hard per-run ceiling, `Quota` for a rolling
    per-tenant window, and `on_exceeded="stop"` to make exhaustion
    recoverable rather than fatal.

    [:octicons-arrow-right-24: Budget and Quota](spend-budget-and-quota.md)

-   __Keep a long conversation in budget__

    ---

    Four compactors and two places to put them, so turn 40 doesn't cost
    forty times turn 1 and then stop fitting at all.

    [:octicons-arrow-right-24: Compact a transcript](compact-a-long-conversation.md)

-   __Pick a provider from config__

    ---

    One model string decides the client, the credential, and whether
    that model can do the job at all — refused at construction, before
    any spend.

    [:octicons-arrow-right-24: Resolve a provider](provider-from-env.md)

-   __Write a custom middleware__

    ---

    Redact a secret, time a call, cache on your own key. Two shapes:
    `BaseMiddleware` to transform/guard/observe, raw `(call, next)` to
    retry, skip, or wrap.

    [:octicons-arrow-right-24: Write a middleware](custom-middleware.md)

</div>

## Running it in production

<div class="grid cards" markdown>

-   __Survive a crash__

    ---

    Same `run_id`, same `CheckpointPort`, hydrated transcript — the next
    worker continues from the last completed iteration rather than
    starting over.

    [:octicons-arrow-right-24: Resume after a crash](resume-after-crash.md)

-   __Wire OpenTelemetry__

    ---

    `TracePort` and `MetricsPort` into Tempo, Jaeger, Datadog,
    Honeycomb — anything that speaks OTLP. One adapter each in
    `arc-agentkit[observability]`.

    [:octicons-arrow-right-24: Wire OTel](otel-tracing-and-metrics.md)

-   __Test an agentkit app__

    ---

    `FakeLLM` and `make_test_ctx`: offline, deterministic, and able to
    produce on cue the failures a real provider never will when you
    want them.

    [:octicons-arrow-right-24: Test without a model](test-an-agentkit-app.md)

-   __Use the local `claude` CLI__

    ---

    `ClaudeCliCognition` subprocesses your installed CLI — its auth, its
    tools, its permission modes — behind a FastAPI endpoint, with the
    spend still on your `Budget`.

    [:octicons-arrow-right-24: Drive the claude CLI](claude-cli-fastapi-code-gen.md)

</div>
