# How do I pick a provider from configuration, and catch a bad model before I pay for it?

## When you'd want this

Two problems that turn out to be one problem.

**The last mile.** Every explicit factory — `claude()`, `openai()`,
`deepseek()`, `openrouter()` — requires an `api_key=`. So every
application writes the same bootstrap: read the key, pick the
provider from the model name, handle the optional-extra `ImportError`,
decide what to do when nothing is configured. Measured in one
production codebase that was ~710 lines, and the only domain-free
code in its engine.

**The silent capability failure.** A model is a string.
`claude(model="claude-sonnet-4-6")` says nothing about what it can do,
so nothing catches a role bound to a model that lacks a needed
capability. An agent asked to read images, bound to a model that
cannot see, returns a structurally valid answer citing evidence it
never read. Nothing downstream can tell.

Both are answered by knowing things about a model name, so both read
one table: the **model registry**.

## Working code

```python
"""Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY, or …) in the environment."""

import asyncio

from agentkit import Agent, Scope
from agentkit.adapters.llm import (
    Capability,
    ModelCapabilities,
    ModelEntry,
    register_model,
    resolve_llm,
)
from agentkit.client import from_env
from agentkit.runtime import Invoker, RunContext, Services


async def main() -> None:
    # 1. The one-liner. Provider picked from the model name, credential read
    #    from the environment. No api_key= anywhere in application code.
    async with from_env("claude-sonnet-4-6") as chat:
        print((await chat("Say hi in five words.")).content)

    # 2. The port, for when you're wiring an Invoker yourself.
    invoker = Invoker(llm=resolve_llm("claude-sonnet-4-6"))
    ctx = RunContext("run-1", Scope(), services=Services(invoker=invoker))

    # 3. Declare what the ROLE needs. Refused at construction — before spend.
    reader = Agent(
        name="ocr",
        model="claude-sonnet-4-6",
        prompt="Describe the attached image.",
        requires=("vision",),
        min_context_window=100_000,
    )
    print(await reader.run("Describe it.", ctx))

    # 4. Declare your own model. Registering a row turns "we don't know"
    #    into a real check — including a real refusal.
    register_model(
        ModelEntry(
            name="acme-internal-v3",
            provider="openai",  # OpenAI-compatible endpoint
            capabilities=ModelCapabilities(
                tools=Capability.YES,
                vision=Capability.NO,
                context_window=32_000,
            ),
        )
    )
    Agent("ocr2", "acme-internal-v3", requires=("vision",))  # CapabilityMismatch


asyncio.run(main())
```

## Resolution: what happens, in order

1. An explicit `provider=` wins.
2. Otherwise a registered `ModelEntry` names the provider.
3. Otherwise the name→provider **rules** run, in registration order.
   Register your own so an internal convention routes without forking
   a prefix table:

   ```python
   from agentkit.adapters.llm import register_rule
   register_rule(lambda name: "openai" if name.startswith("acme-") else None)
   ```
4. The provider's environment variables are checked in order
   (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
   `OPENROUTER_API_KEY`). An empty string counts as absent.
5. The factory is imported **lazily**, by dotted path. Registering a
   provider costs no import, so the registry stays usable on a
   zero-dependency install.

Dated release ids normalise: `claude-haiku-4-5-20251001` and
`anthropic/claude-sonnet-4-6` both resolve to their family row.

## Capabilities: declared, never guessed

`Capability` is **three-valued** — `YES`, `NO`, `UNKNOWN` — and the
third value is the whole point. A `bool` forces every unrecognised
model into either "has it" (silent wrong answers) or "doesn't"
(refuses everything custom).

| Declared | What happens |
|---|---|
| `NO` | Always raises `CapabilityMismatch`, naming the capability and the model |
| `YES` | Silent |
| `UNKNOWN` | Governed by `on_unknown_capability` on the Agent |

`on_unknown_capability` is `"warn"` by default: say so once, and
continue. Refusing by default would break every self-hosted,
fine-tuned, or brand-new model name on upgrade. Set `"refuse"` in a
production service that pins its models — it turns "we don't know"
into a deployment-time stop:

```python
Agent("ocr", "our-finetune-v2", requires=("vision",), on_unknown_capability="refuse")
```

Declarable capabilities: `tools`, `structured_output`,
`native_json_schema`, `streaming`, `vision`, plus `context_window`
(an int, requested via `min_context_window=`).

**A tool-using cognition implies `tools` automatically.** You don't
declare it. That derived requirement only ever fires against a model
declared `tools=NO` — it stays silent on `UNKNOWN`, so a made-up model
name in development doesn't produce a warning nobody can act on.

## Gotchas

**Fallback is opt-in.** `resolve_llm("...")` with no credential
**raises** `ProviderNotConfigured`. Pass `fallback="fake"` to degrade
to a canned LLM instead, which warns exactly once. This is deliberate:
in a server process a `UserWarning` goes to a log nobody reads, and
the outcome is fabricated completions served as real answers. The
escape hatch stays one keyword away for a notebook; it just isn't the
accident.

**A missing extra is fatal, never a downgrade.** No `httpx` installed
raises `MissingProviderExtra` naming the pip extra — even when
`fallback="fake"` was requested. A broken install must not masquerade
as a missing credential.

**Credentials never appear in a message.** Not in an error, not in a
warning, not in a `repr`. Pinned by test.

**The built-in table will go stale.** It is a convenience default,
like the price table beside it. `register_model` replaces by name, so
pin your own truth over it at startup.

**`Agent` is a mutable dataclass.** Swapping `agent.model` after
construction bypasses `__post_init__`. Re-assert with
`agent.check_capabilities()`.

**The explicit factories are untouched.** `claude(api_key=...)` still
works exactly as before. This is a layer above them, not a
replacement.

## Related

- [Cap spend with Budget and Quota](spend-budget-and-quota.md) — an
  unregistered model also has no price, so it silently costs `$0.00`
- [Stream a typed object](stream-typed-output.md) —
  `native_json_schema` is what decides whether provider-native strict
  mode gets wired
