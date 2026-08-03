# Prompts

**What this is.** `agentkit.prompts` is a small, deliberately-boring
package that owns the *shape* of a prompt — a versioned `Prompt`
value type — plus a couple of built-in helpers. It does **not** own
templating engines, evaluators, or A/B routing; those live in higher
layers (or your app) and consume the `Prompt` shape.

**Why it exists.** Prompt drift is a top source of silent regressions
in agent systems. Keeping prompts as *values* — pinnable, diff-able,
and threaded through the same middleware chain as any other call —
means a change to a system prompt is a code change with the same
review surface as a change to a function signature.

## What a `Prompt` is

A `Prompt` bundles:

- **`name`** — a stable identifier used for tracing and evaluation
  reports.
- **`version`** — an integer or semver string; every rendered prompt
  carries this into the trace envelope.
- **`template`** / **`render(inputs)`** — the pure function from
  inputs to a system + user string pair.
- **`inputs_schema`** *(optional)* — a schema describing what the
  template requires, so misuse fails fast at build time rather than at
  inference time.

## Built-ins

`agentkit.prompts.builtin` ships a small number of vetted prompts used
across the framework's own agents (e.g. a default system prompt for
`ReActCognition`). They exist so a new project can start without
authoring prompts, and so upstream improvements benefit every
consumer.

## The invariants it enforces

1. **Prompts are values.** A `Prompt` is immutable; mutating one is a
   contract violation.
2. **Version travels with output.** Every LLM call carries
   `prompt.name` and `prompt.version` on its trace envelope so a
   regression can be localised.
3. **Rendering is pure.** `render(inputs)` cannot do I/O or read
   ambient state; if you need to inject data, put it in `inputs`.

## API

Full generated reference lives at
[API › prompts](../api-reference/prompts.md).
