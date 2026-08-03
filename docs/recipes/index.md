# Recipes

Recipes are focused answers to specific questions. Each one is a
self-contained page: the question, one runnable script, an explanation
of the primitives involved, and the gotchas. If you want a linear
walkthrough, use the [Tutorial](../tutorial.md); if you want the mental
model of a primitive, use the [Concepts](../concepts/kernel.md); this
section is what you reach for after you've written some code and hit
one specific wall.

<div class="grid cards" markdown>

-   __Human-in-the-loop tool approval__

    ---

    How do I pause a tool for human approval? `side_effecting=True`,
    `Autonomy.GATED`, `Suspended`, `agent.resume(...)`.

    [:octicons-arrow-right-24: Read](hitl-tool-approval.md)

-   __Resume after a crash__

    ---

    How do I pick a run back up after the worker process died?
    `Checkpointer` + one run id.

    [:octicons-arrow-right-24: Read](resume-after-crash.md)

-   __Cap spend with Budget and Quota__

    ---

    How do I put a hard ceiling on a run's cost, and a rolling cap on a
    tenant's spend? `Budget`, `Quota`, `MeterExceeded`.

    [:octicons-arrow-right-24: Read](spend-budget-and-quota.md)

-   __Parallel agents with cancellation__

    ---

    How do I run agents in parallel, and how do I make sure one
    failure cancels the rest? `run_agents`, `CancellationToken`,
    cooperative cancel.

    [:octicons-arrow-right-24: Read](parallel-agents-with-cancellation.md)

-   __Write a custom middleware__

    ---

    How do I add my own cross-cutting concern? `BaseMiddleware` for
    transform/observe, raw `(call, next)` for resilience/caching.

    [:octicons-arrow-right-24: Read](custom-middleware.md)

-   __Wire OpenTelemetry__

    ---

    How do I hook agentkit up to a real trace + metrics backend?
    `TracePort` / `MetricsPort`, `otel_tracer`, span attributes.

    [:octicons-arrow-right-24: Read](otel-tracing-and-metrics.md)

</div>
