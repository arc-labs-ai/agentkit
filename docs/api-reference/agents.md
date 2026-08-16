# `agentkit.agents`

The `Agent`, `Workflow`, and `Cognition` machinery, plus the control
primitives (signals, `RunPolicy`, `Handoff`, `ActorBudget`) and
policies (`RoundRobinPolicy`, `SelectorPolicy`, `PlanPolicy`,
`LedgerPolicy`).

See the [Agents concept](../concepts/agents.md) for the mental model.

::: agentkit.agents
    options:
      show_root_heading: false
      show_source: false
      members_order: source

## Human-in-the-loop

`Elicitation` / `Decision` / `Asker` — pausing a run for a person as a
**value request**, parkable in place, deadlined, and typed. See the
[recipe](../recipes/elicit-a-value-from-a-human.md).

::: agentkit.agents.control.elicitation
    options:
      show_root_heading: false
      show_source: false
      members_order: source
