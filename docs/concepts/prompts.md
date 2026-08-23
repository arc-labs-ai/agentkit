# Prompts

!!! abstract "Where this fits in the four themes"
    `Prompt` is a **State**-theme primitive — the versioned template
    that anchors every chat call. Its `id` + `version` travel with
    every LLM call through the middleware chain (the **Behaviour**
    theme) so a regression can be localised to a specific template
    revision. See the four-theme grid on the
    [landing](../index.md).

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

- **`id`** — a stable identifier used for tracing and evaluation
  reports.
- **`version`** — a semver string; every rendered prompt carries this
  into the trace envelope.
- **`template`** — the template string.
- **`inputs`** — a `tuple[str, ...]` of the placeholder names the
  template declares; the NAMES live on the `Prompt` value so a stored
  prompt carries its own contract, while the VALUES are passed to
  `render()`.
- **`render(**values)`** — substitutes each declared `{name}` and
  returns the text. A missing or unexpected input raises `ValueError`
  rather than rendering a half-filled prompt: `{tenant}` reaching the
  model unsubstituted is not a crash, which is the problem — it is a
  plausible-looking prompt that quietly describes the wrong task.

  Substitution is a literal replacement of the declared names, **not**
  `str.format`. System prompts are full of braces that are not
  placeholders — a JSON Schema, an example payload, a code fence — and
  `format` would raise on them or eat a user's escaping:

  ```python
  p = Prompt(id="x", version="1", inputs=("kind",),
             template='Return {"type": "object"} for {kind}.')
  p.render(kind="invoices")   # 'Return {"type": "object"} for invoices.'
  ```

  A prompt that declares no `inputs` renders exactly as before —
  the stripped template, no arguments.

## Built-ins

`agentkit.prompts.builtin` ships two vetted prompts used by the
Compactor middleware: `COMPACTION_IMPORTANCE` (importance scoring)
and `COMPACTION_SUMMARY` (summary generation). There is no built-in
ReAct system prompt — cognitions author their own.

## The invariants it enforces

1. **Prompts are values.** A `Prompt` is immutable; mutating one is a
   contract violation.
2. **Version travels with output.** Every LLM call carries
   `prompt.id` and `prompt.version` on its trace envelope so a
   regression can be localised.
3. **Rendering is pure.** `render()` cannot do I/O or read ambient
   state; if you need to inject data, declare it in `inputs` and bind
   it on the `Prompt` value.

## API

Full generated reference lives at
[API › prompts](../api-reference/prompts.md).
