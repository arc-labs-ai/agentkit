# Prompts

A `Prompt` is a system prompt with an id and a version attached, so that
when quality changes you can say *which wording* changed it.

That is the whole package. `agentkit.prompts` is deliberately boring: it
owns the shape of a prompt and nothing else. No templating engine, no
prompt store, no A/B router — those live in your app, or in a layer above,
and consume this shape.

## The problem it solves

Prompt drift is a top source of silent regressions in agent systems. A
teammate softens one sentence in a system prompt to fix one bad output,
and three weeks later a different behaviour is worse and nobody can point
at the change. There is no stack trace for "the wording moved".

Treating a prompt as a **value** — pinnable, diff-able, and carried
through the same middleware chain as every other call — makes a prompt
edit a code change with the same review surface as a change to a function
signature. The `version` label travels onto the trace and onto
`AgentResult`, so a regression localises to a specific template revision
rather than to "sometime last sprint".

The second thing it fixes is quieter and worse: a placeholder that
reaches the model unsubstituted. `{tenant}` arriving literally is not a
crash — it is a plausible-looking prompt that describes the wrong task,
and the model will happily answer it.

## The smallest example

```python
import asyncio

from agentkit import Agent, Prompt
from agentkit.testing import FakeLLM, make_test_ctx

BRIEFER = Prompt(
    id="briefer",
    version="1.0.0",
    inputs=("tenant",),
    template='Brief tenant {tenant}. Reply as {"type": "object"}.',
)


async def main() -> None:
    seen: list[str] = []
    llm = FakeLLM(lambda system, user, model: seen.append(system) or "done")
    ctx = make_test_ctx(llm=llm)

    agent = Agent(name="briefer", model="fake-model", prompt=BRIEFER.bind(tenant="acme"))
    await agent.run("summarise Q3", ctx)
    print(seen[0])       # Brief tenant acme. Reply as {"type": "object"}.

    # Handing the agent the UNBOUND prompt is refused at construction.
    try:
        Agent(name="briefer", model="fake-model", prompt=BRIEFER)
    except ValueError as exc:
        print(str(exc)[:60])


asyncio.run(main())
```

## What a `Prompt` is

| Field | What it carries |
|---|---|
| `id` | A stable identifier, used in tracing and evaluation reports. |
| `version` | A version label; travels into the trace envelope on every call. |
| `template` | The template string. |
| `inputs` | A `tuple[str, ...]` of the placeholder names the template declares. |
| `bound` | The values bound to those names — a read-only mapping. |

The names live on the **value**, not at the call site, so a stored prompt
carries its own contract: you can validate a prompt against a call site
before a run, and a prompt loaded from a database still knows what it
needs.

Two methods:

- **`bind(**values)`** — returns a **new** `Prompt` with those values
  bound. Same `id`, same `version`, same `inputs`, so a bound prompt is
  still pinnable, diff-able and attributable, and goes anywhere an
  unbound one goes.
- **`render(**values)`** — substitutes each declared name and returns the
  text, taking values from `bound` and from `**values`, with the latter
  winning on a collision.

## Binding, end to end

Values live on the prompt because **nothing in the framework passes them
at call time**. `RequestBuilder` and both CLI cognitions call `render()`
with no arguments. Binding is what makes `inputs` usable through an
`Agent` at all.

```python
from agentkit import Prompt

BRIEFER = Prompt(
    id="briefer",
    version="1.0.0",
    inputs=("tenant", "tone"),
    template='Brief tenant {tenant} in a {tone} voice. Reply as {"type": "object"}.',
)

# Binding is incremental, immutable, and last-write-wins.
half = BRIEFER.bind(tenant="acme")
print(half.render(tone="terse"))
print(half.bind(tenant="globex").bind(tone="warm").render())
print(BRIEFER.bound)                       # {} — the original never changed

# A missing input is refused rather than shipped half-filled.
try:
    half.render()
except ValueError as exc:
    print(str(exc)[:80])

# So is an undeclared name — at bind() time, not at render().
try:
    BRIEFER.bind(tenat="acme")
except ValueError as exc:
    print(str(exc)[:80])
```

That last one is the cheap catch: a renamed or mistyped placeholder is
caught where you wrote it, not on the first drive. Quietly accepting it
would render the *old* template with none of the new values.

### An unbound prompt is refused at construction

`Agent.check_prompt()` runs from `__post_init__`, beside the model
capability check, and for the same reason. An unbound prompt is not a
*maybe* problem — nothing passes values later, so it is a guaranteed
`ValueError` on the first drive. The only question is whether you find
out before or after the run starts, and the alternative to refusing is
shipping a literal `{tenant}` to the model.

!!! warning "`Agent` is a mutable dataclass"
    Assigning `agent.prompt` after construction bypasses that check.
    Re-assert with `agent.check_prompt()`, exactly as with
    `agent.check_capabilities()`.

A plain `str` prompt, a `None` prompt, and a `Prompt` that declares no
`inputs` are all unaffected.

## Why not `str.format`

Substitution is a **literal replacement of the declared names**, never
`str.format`. System prompts are full of braces that are not
placeholders — a JSON Schema, an example payload, a code fence — and
`format` would either raise `KeyError` on them or silently eat the
doubling someone added to escape them.

```python
import copy
import pickle

from agentkit import Prompt

p = Prompt(id="x", version="1", inputs=("kind",),
           template='Return {"type": "object"} for {kind}.')

print(p.render(kind="invoices"))   # Return {"type": "object"} for invoices.

bound = p.bind(kind="invoices")
print(hash(bound) == hash(p))                      # True — hashed on identity
print(copy.deepcopy(bound).render())               # deep-copyable
print(pickle.loads(pickle.dumps(bound)).render())  # and picklable
```

A prompt that declares no `inputs` renders exactly as it always did: the
stripped template, no arguments, no `bind()` needed.

## A `Prompt` is a value all the way down

Three properties that are easy to assume and were, at one point, not
true:

- **`bound` is frozen deeply, and it is a copy.** A caller cannot keep
  editing the dict they bound, and nested containers inside it are frozen
  too — a shallow copy would leave the same bug one level down.
- **It is deep-copyable and picklable.** `bound` is a `dict` subclass
  rather than a `MappingProxyType`, and that matters concretely: the
  proxy made a bound prompt unpicklable and therefore
  un-deep-copyable — which the checkpointer does on every snapshot — and
  invisible to `json.dumps` and `dataclasses.asdict`. See
  [Kernel › immutability](kernel.md#immutability-what-frozen-actually-means-here).
- **It is hashable, on identity.** `hash()` covers
  `(id, version, template, inputs)` and deliberately not `bound`, so a
  bound and an unbound copy of the same template share a bucket — the
  useful behaviour for a cache keyed on prompt identity. `__eq__` still
  compares the bindings, so a `set` keeps both.

## Built-ins

`agentkit.prompts.builtin` ships two vetted seed prompts used by the
LLM-driven compaction strategies: `COMPACTION_SUMMARY` (for
`SummarizationCompactor`) and `COMPACTION_IMPORTANCE` (for
`ImportanceFilteringCompactor`).

They live there rather than as string literals inside the compactors for
three reasons. They get one source of truth, so an operator tuning
compaction wording bumps a version instead of patching a file. They get
attribution, so a compaction summary lands in traces with a version
stamp. And a compaction *strategy* (when and how to fold a transcript) is
kept separate from a compaction *prompt* (wording).

There is no built-in ReAct system prompt — cognitions author their own.

## What bites people

- **Nothing passes values at call time.** If a template declares
  `inputs`, bind them, or the run refuses. This is the single most common
  surprise on this page.
- **Undeclared names are refused, not ignored** — at `bind()`, and also
  at `render()` for a prompt that declares no inputs. Passing values to a
  prompt that declares none is a mistake at the call site, not a no-op.
- **`bind()` never mutates.** A module-level prompt shared by several
  call sites cannot pick up another one's values. If you expected
  in-place binding, you will silently render the unbound template.
- **`render()` strips the result.** Leading and trailing whitespace in a
  template does not survive.
- **The version label is yours to bump.** Nothing computes it from the
  template, so a wording change with an unchanged version defeats the
  point of the field.

## The invariants it enforces

1. **Prompts are values.** A `Prompt` is immutable, payload included.
2. **Version travels with output.** Every LLM call carries `prompt.id`
   and `prompt.version` on its trace envelope so a regression can be
   localised.
3. **Rendering is pure.** `render()` does no I/O and reads no ambient
   state. To inject data, declare it in `inputs` and bind it.
4. **Binding does not mutate.** `bind()` returns a new value; the
   original stays as it was.

!!! abstract "Where this fits in the four themes"
    `Prompt` is a **State**-theme primitive — the versioned template that
    anchors every chat call. Its `id` and `version` travel with every LLM
    call through the middleware chain (the **Behaviour** theme), so a
    regression can be localised to a specific template revision. See the
    four-theme grid on the [landing page](../index.md).

## Related

- [Capabilities › RequestBuilder](capabilities.md#requestbuilder) — where a prompt is rendered into a request.
- [Agents](agents.md) — where `check_prompt()` runs.
- [Context](context.md) — the cache-stable prefix a rendered prompt lands in.
- [API › prompts](../api-reference/prompts.md) — the generated reference.
