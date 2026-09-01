# The Codex CLI

The second of the two places where your code does not own the loop.

`CodexCliCognition` hands the whole agent loop to a locally installed `codex`
binary. The CLI decides how many turns to take, which of its own tools to call,
and when it is finished. Your application starts it and reads what comes back.

!!! tip "Is this page for you?"

    **Reach for it when** you want Codex's coding ability inside an agentkit run
    — its `shell`, its `apply_patch`, its OS-level sandbox — without managing API
    keys or reimplementing any of it.

    **Read [The Claude CLI](claude-cli.md) first if** you have not, because the
    two cognitions are deliberately parallel and that page makes the shared
    argument once. This page is mostly about where they differ.

## The shape of it

```python
from pathlib import Path

from agentkit import Agent
from agentkit.agents.cognition import CodexCliCognition

agent = Agent(
    name="local",
    prompt="You are a concise assistant.",
    cognition=CodexCliCognition(
        model="gpt-5-codex",
        working_dir=Path("/tmp/sandbox"),
        sandbox="read-only",        # what the session may DO
        ask_for_approval="never",   # ...and whether it may pause to ask a human
    ),
)
```

Auth is entirely the CLI's problem: whatever `CODEX_API_KEY`, `OPENAI_API_KEY`
or `$CODEX_HOME` login the CLI would find on its own is what this uses.
agentkit never touches an API key here.

Everything a caller relies on is the same as the Claude cognition: exactly one
terminal `final` event on every path, the same `AgentResult`, the same
stop-reason taxonomy, the same `spawn=` seam for tests, the same `evals` keys
where the same fact exists (`session_id`, `cli_duration_ms`,
`cli_return_code`).

## The four places they differ

The two cognitions are parallel wherever the *binaries* are. Where they are
not, papering over it would mean a field that silently does nothing — so each
difference is a visible difference in the API.

| | `ClaudeCliCognition` | `CodexCliCognition` |
|---|---|---|
| containment | `tools=` — a tool allow-list | `sandbox=` — an OS sandbox |
| system prompt | `--append-system-prompt` | no flag; prepended, or a replaced instructions file |
| cost | reported by the CLI | computed from a price table |
| in-flight budget cap | `--max-budget-usd` | none |
| a session | one held process, many turns | one thread, resumed per turn |
| middleware escape hatch | `hook_settings` | none — Codex has no pre-tool hook |

### Containment is a sandbox, not a tool list

Claude Code restricts what a session *has*. Codex gives every session the same
small toolbox — `shell`, `apply_patch`, `update_plan`, and `web_search` when
you ask for it — and restricts what those tools may *do*:

- `sandbox="read-only"` (the CLI's own default for `codex exec`) — can read the
  workspace, cannot write, no network.
- `sandbox="workspace-write"` — can also write inside the workspace and any
  `add_dirs`. Still no network unless you pass `network_access=True`.
- `sandbox="danger-full-access"` — no containment at all.

`ask_for_approval` is the *other* half and is not a substitute for it. In a
service it should be `"never"` — there is nobody at the terminal to answer —
which makes `sandbox` the only thing standing between the model and the
machine.

!!! note "This is a stronger guarantee than a tool list, on one axis"

    A tool allow-list is enforced by the CLI deciding not to offer a tool. A
    sandbox is enforced by the operating system. The Codex cognition has no way
    to make your middleware apply to `shell` (see below) — and the thing that
    *does* contain `shell` does not depend on the model's cooperation.

### There is no system-prompt flag

Codex has no `--append-system-prompt`. So `agent.prompt` reaches the model one
of two ways, and both are stated rather than guessed at:

- `system_prompt_mode="prepend"` (the default) puts it at the top of the first
  user message, above the task, separated by a rule. Codex's own base
  instructions — tool guidance, the sandbox explanation, the `apply_patch`
  format — stay in place.
- `system_prompt_mode="replace"` writes it to a file passed as
  `-c experimental_instructions_file=…`, which REPLACES those base instructions
  the way `claude --system-prompt` replaces Claude Code's. The config key is
  marked experimental by the CLI, so this is the mode that will break first on
  an upgrade — deliberate, since the alternative is silently not replacing
  anything.

In a [session](#a-session-is-a-resumed-thread), the prepended prompt goes out
on the **first turn only**. Sending it again on turn three would put three
copies in the transcript and read to the model as escalating emphasis.

### The CLI does not report cost

Claude Code's result payload carries `total_cost_usd`. Codex's `turn.completed`
carries token counts and nothing else. So `Usage.cost_usd` is **computed**:

```python
from agentkit.agents.cognition import CodexCliCognition

cognition = CodexCliCognition(
    model="gpt-5-codex",
    pricing=lambda model, usage: usage.input_tokens / 1e6 * 1.25
    + usage.output_tokens / 1e6 * 10.0,
)
```

Without a `pricing=` the framework's own best-effort table is used, and it
returns `0.00` for a model it has never heard of — which is most Codex models.
Every result carries `evals["cost_source"] == "estimated"` so nobody reads the
field as a billed number.

One more consequence, and it is the one to watch: **there is no
`--max-budget-usd`.** An exhausted budget refuses to spawn (with the resumable
`budget_exhausted` stop reason) and what a run cost is charged afterwards, but
nothing hands the CLI a ceiling it can stop itself against mid-flight. A run
that starts with a cent of headroom can spend five dollars.

### A session is a resumed thread

`codex exec` is one-shot. Its continuation seam is `codex exec resume <id>`, so
`CodexCliSession` spawns per turn and threads the conversation through the id it
learned from turn one:

```python
import asyncio

from agentkit.agents.cognition import CodexCliCognition


async def main() -> None:
    cognition = CodexCliCognition(model="gpt-5-codex")
    async with cognition.session() as chat:
        async for _ in chat.turn("Summarise README.md"):
            ...
        async for _ in chat.turn("Now list the risks you skipped"):
            ...
        print(chat.session_id, chat.turns_taken)


asyncio.run(main())
```

That costs a CLI warm-up per turn, where the Claude session pays it once. It
buys three things the held-process design cannot have:

- **A cancelled turn does not end the conversation.** The thread is on disk; the
  next `turn()` resumes it. `ClaudeCliSession` has to declare itself over,
  because its process *was* the conversation.
- **Per-turn structured output works.** `--output-schema` is chosen at spawn and
  every turn is its own spawn, so an `output=`-carrying agent may be passed to
  any turn. The Claude session has to refuse it.
- **A failed turn costs one turn.**

`ephemeral=True` is refused by `session()`: nothing is written, so there is
nothing to resume.

## The thing to understand first: the chain does not apply

Same edge as the Claude page, and here it is sharper, because there is no hook
to fall back on. When the CLI runs `shell` or `apply_patch`, the call never
passes through your `Invoker`, and none of this runs:

| middleware | what stops applying |
|---|---|
| `egress` | default-deny URL checking |
| `guard` | every input guard |
| `audit` | one record per tool call. There are no records |
| `memoize` | idempotency and single-flight |
| `Guardrail.check_url` | SSRF and allowlist |

So the cognition warns, once per instance, naming the middlewares that will not
apply and the sandbox that is actually in force. Two ways to close the gap and
they are not equivalent:

- **Serve your own tools over MCP** (below). Those calls come back through
  agentkit, so the chain applies to *them*. `shell` is still outside it.
- **Contain `shell` with the sandbox.** `sandbox="read-only"` means an `egress`
  guard has nothing to guard: there is no network.

There is deliberately no `codex_hook_settings`. Codex's extension points are
`notify` (which fires after the fact and cannot refuse) and execpolicy `.rules`
files (static patterns, not a Python chain against a live `ctx`). A function
that generated one of those and called it middleware coverage would be the
silent non-application this whole area exists to avoid.

## Capabilities: the Rule-of-Two applies here too

[`RunPolicy`](agents.md) refuses a tool set that reaches private data, untrusted
content and egress at once. A Codex session's tags come off the sandbox rather
than a tool table:

```python
from agentkit.agents.control.safety import RunPolicy
from agentkit.agents.cognition import CodexCliCognition

for kwargs in [{"sandbox": "read-only"}, {"sandbox": "read-only", "web_search": True}]:
    cognition = CodexCliCognition(**kwargs)
    print(kwargs, "->", cognition.caps, RunPolicy().check([cognition]).allowed)
```

```text
{'sandbox': 'read-only'} -> ('private_data',) True
{'sandbox': 'read-only', 'web_search': True} -> ('egress', 'private_data', 'untrusted_content') False
```

The second row is the one to pause on. **A read-only sandbox with web search on
is the full trifecta** — it reads like the safest configuration Codex has,
because the agent cannot write anything, and it combines private-data access,
untrusted content and egress in one run.

Three more notes on the tagging:

- **Read-only is about writes, not about reading.** Every mode carries
  `private_data`; a read-only agent is not blind.
- **`network_access=True` adds `egress`** to `workspace-write`, because that is
  exactly what it turns on.
- **MCP tools contribute nothing here**, because their caps live on the server's
  own tool definitions. Pass those tools' agentkit-side objects alongside the
  cognition: `RunPolicy(mode="deny").check([cognition, *tools])`.

## Giving Codex your own tools

`serve_registry` already advertises an agentkit `ToolRegistry` over MCP with its
schemas, `requires_approval` flags and `caps` intact. What it writes is Claude
Code's `--mcp-config` document; Codex reads MCP servers out of `config.toml`.
`as_codex_mcp` is that translation:

```python
from agentkit.agents.cognition import CodexCliCognition
from agentkit.integrations.codex_cli import as_codex_mcp
from agentkit.integrations.mcp import serve_registry


async def wire(registry, ctx):
    spec = serve_registry(registry, name="engine", ctx=ctx)
    async with spec:
        cognition = CodexCliCognition(model="gpt-5-codex", **as_codex_mcp(spec))
        return cognition
```

`spec.codex_kwargs()` is the same thing as a method, next to `cli_kwargs()`.

One thing there is not a rename. Codex has no header field for a bearer token —
it has `bearer_token_env_var`, which names an *environment variable* the CLI
reads at connect time. So `as_codex_mcp` returns an `env=` entry alongside
`mcp_servers=`, and the token travels in the child's environment rather than in
the argv, which is world-readable in `ps` output on most systems.

Unlike `cli_kwargs`, there is no `builtin_tools=` switch. Codex has no tool
allow-list, so "only OUR tools" is not something a flag can say — every session
has `shell`. What contains it is `sandbox=`.

## Testing it

[`FakeCodexCli`](testing.md) sits at the spawn seam, so a test still exercises
the real parsing, the real token split, the real cost computation and the real
meter charge:

```python
from agentkit.agents.cognition import CodexCliCognition
from agentkit.testing import FakeCodexCli, codex_turn

cli = FakeCodexCli.script(codex_turn(text="done", usage=(1200, 1000, 40)))
cognition = CodexCliCognition(spawn=cli)
print(type(cognition).__name__, cli.spawns)
```

```text
CodexCliCognition 0
```

`codex_turn` builds one complete, well-formed turn — `thread.started`,
`turn.started`, the items, `turn.completed` — because the order carries meaning
the cognition depends on and hand-writing it in every test is where a test ends
up asserting against a stream the binary would never emit.
`FakeCodexCli.answering("one", "two")` is the shorthand for a multi-turn
session, where a spawn is a turn.

## What bites people

- **`sandbox=None` is not "no sandbox", it is the CLI's default** — read-only
  for `codex exec`, plus whatever the operator's `config.toml` says. Set it
  explicitly in a service: "whatever the machine happens to be configured for"
  is not a containment decision.
- **`ask_for_approval="never"` alone is not containment.** It removes the human
  and leaves the sandbox as the only control. Set both.
- **A read-only sandbox with `web_search=True` is the lethal trifecta**, and it
  is the most innocuous-looking configuration Codex has.
- **`usage.cost_usd` is an estimate and is `0.00` for an unpriced model.** Check
  `evals["cost_source"]`, and pass `pricing=` if the number matters.
- **The budget is not an in-flight ceiling.** It refuses before and charges
  after; nothing stops a run mid-flight.
- **Codex refuses to run outside a git repository.** A scratch-directory
  workspace needs `skip_git_repo_check=True`.
- **The middleware chain does not apply to `shell`, and there is no hook that
  can make it.** The sandbox is the control.

## Related

- [The Claude CLI](claude-cli.md) — the sibling cognition, and the shared
  argument for why any of this is shaped the way it is.
- [Integrations (MCP)](integrations.md) — `serve_registry`, which
  `as_codex_mcp` points at Codex.
- [Agents](agents.md) — the cognitions alongside each other, and `RunPolicy`.
- [Middlewares](middlewares.md) — the chain that does *not* reach native tools.
- [Testing](testing.md) — the CLI doubles in full.
