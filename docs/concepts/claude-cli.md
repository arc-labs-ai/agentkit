# The Claude CLI

Everywhere else in agentkit, your code owns the loop. This is one of two places
it does not — the other is [The Codex CLI](codex-cli.md), which is deliberately
parallel to this page and differs only where the binaries do.

`ClaudeCliCognition` hands the whole agent loop to a locally installed `claude`
binary. The CLI decides how many turns to take, which of its own tools to call,
and when it is finished. Your application starts it and reads what comes back.

!!! tip "Is this page for you?"

    **Reach for it when** you want Claude Code's coding ability inside an
    agentkit run — its `Edit`, its `Bash`, its planning — without managing API
    keys or reimplementing any of it.

    **Skip it for now if** your agent talks to a provider through an `LLMPort`.
    Everything on this page is about the case where it does not.

## The problem it solves

The CLI is genuinely good at what it does, and a subprocess wrapper around it is
easy to write badly. `ClaudeCliCognition` handles the stream-json parsing, maps
it onto agentkit's `StreamEvent`s, charges the budget, honours cancellation, and
returns the same `AgentResult` every other cognition returns. That part has
worked for a while.

The harder problem is the other direction. **The CLI owns the loop, so anything
your application knows has to reach it as configuration** — before the run
starts, in a form the binary understands. Your tools. Your approval policy. Your
guardrails. Your sub-agents.

Every one of those was a knob `ClaudeCliCognition` exposed and nothing could
fill. `mcp_config` took a path no agentkit code could produce.
`permission_prompt_tool` named an MCP tool that did not exist. `settings` took a
file nobody generated. `agents` took definitions you had to restate by hand.

This page is what fills them.

## The five seams

| You want the CLI to… | Reach for | Lands as |
|---|---|---|
| call **your** tools | [`serve_registry`](integrations.md#the-other-direction-agentkit-as-an-mcp-server) | `--mcp-config` |
| ask **your** reviewer before acting | [`ApprovalServer`](integrations.md#the-other-direction-agentkit-as-an-mcp-server) | `--permission-prompt-tool` |
| respect **your** middleware on its *own* tools | `hook_settings` | `--settings` |
| delegate to **your** skills | `as_cli_agents` | `--agents` |
| run in a test, offline and free | [`FakeClaudeCli`](testing.md#fakeclaudecli-testing-the-cli-path-without-spending) | the spawn seam |

The first two are MCP servers, so they live with the [MCP
integration](integrations.md). The middle two are plain configuration
generators in `agentkit.integrations.claude_cli`. The last is a test double.

## The thing to understand first: the chain does not apply

This is the sharpest edge on this page, and it is invisible unless someone tells
you.

agentkit says it in its own source, in `_charge_meters`:

> *The CLI bypasses the `Invoker`, so the `meter()` middleware never sees this
> usage and every meter on the context stays at zero. That is how a documented
> safety mechanism ends up doing nothing.*

Metering was patched by hand for exactly that reason. **Nothing else was.** So
when the CLI runs one of its own tools — `Write`, `Edit`, `Bash`, `WebFetch` —
the call never passes through your `Invoker`, and none of this runs:

| middleware | what stops applying |
|---|---|
| `egress` | default-deny URL checking. A `WebFetch` reaches anywhere |
| `guard` | every input guard |
| `audit` | one record per tool call. There are no records |
| `memoize` | idempotency and single-flight |
| `Guardrail.check_url` | SSRF and allowlist |

A caller who reads the [middleware](middlewares.md) documentation, wires a
chain, and then uses this cognition gets a session where **none of it applies**.

So the cognition warns. If your `ctx` carries tool middleware and native tools
are enabled, you get a warning **naming the middlewares that will not apply** —
not a vague caution you can learn to ignore:

```text
UserWarning: ClaudeCliCognition: the tool middleware on this context
(egress, audit, memoize) is not applied to the CLI's built-in tools …
```

It fires once per cognition instance. Not per turn, because a warning that
repeats is noise, and noise is what teaches people to add a `filterwarnings`
line — which is how the *next* silent misconfiguration gets through. Not per
process either, because two cognitions in one service are usually two different
configurations, and the second one is the one nobody has audited.

Three ways to close the gap, and they are not equivalent:

- **Serve everything over MCP** (`tools=("",)` plus `serve_registry`). Every
  call comes back through agentkit code, so the chain applies naturally. The
  cost is real: the CLI's own tools are good — its `Edit` in particular — and
  reimplementing them behind MCP is work and a quality loss.
- **Generate hooks** (`hook_settings`, below). Keeps the CLI's tools and makes
  the chain apply. The cost is that the refusal now lives in a generated script:
  a second execution path to keep correct.
- **Accept it, knowingly.** The warning is what makes this a choice rather than
  an accident.

## `hook_settings` — make the chain reach native tools

Claude Code settings carry `PreToolUse` hooks: before the CLI runs a tool, it
calls out to a script, which can refuse. `hook_settings` generates one from the
same middleware chain your `Invoker` would use.

```python
from agentkit.integrations.claude_cli import hook_settings

async with hook_settings(
    middleware=tool_chain,          # the chain the Invoker would use
    ctx=ctx,
    tools=("Write", "Edit", "Bash"),
) as settings:
    cognition = ClaudeCliCognition(settings=settings.path)
```

Three properties are worth knowing because they live in a file a *different
process* reads, where nothing in your test suite can see them:

**It fails closed.** A payload the hook cannot parse is a deny, not a pass.
That sounds obvious and was not: emitting `{}` means "no opinion", which under
`bypassPermissions` runs the tool. A payload with no `tool_name` used to become
`ToolRequest(name="")`, so name-keyed guards matched nothing and `Egress` passed
— a missing URL is not a blocked URL. Measured before the fix:
`allowed=True` for `http://169.254.169.254/`.

**The matcher is anchored.** `^(Write|Edit)$`, not `(Write|Edit)`. Unanchored,
an `Edit` matcher also fires for `NotebookEdit` — a guard applying where nobody
chose, with a deny reason naming a tool the caller never listed.

**The CLI's own hook timeout is the loosest of the three deadlines.** A hook the
CLI times out is a *non-blocking* error: it prints and then **runs the tool**.
So the inner deadlines must fire first, or a chain that is merely slow becomes
the fail-open the whole design exists to close.

`settings.decisions` records what the hook decided, so a run can be audited
afterwards.

!!! warning "Not every middleware can work here"

    The hook runs in a separate process, so it cannot hold your live `ctx` — no
    `Invoker`, no store handle, no cancel token. Middlewares that need those
    cannot apply, and `hook_settings` refuses them at generation time rather
    than accepting one that would silently do nothing. That refusal is the
    point: the whole reason this feature exists is that silent non-application
    is the failure mode.

## `as_cli_agents` — project a Skill into a CLI sub-agent

The CLI can delegate to sub-agents defined in its configuration. A
[`Skill`](skills.md) is agentkit's name for the same idea: a prompt, a
cognition, tools and memory bundled under a name. They are two ends of one wire,
and restating one as the other by hand is the second description of one thing
that the rest of agentkit is careful to avoid.

```python
from agentkit.integrations.claude_cli import as_cli_agents

cognition = ClaudeCliCognition(
    agents=as_cli_agents([reviewer_skill, repairer_skill]),
)
```

**A skill's tool restriction survives the projection.** This is a security
property, not an ergonomic one: a reviewer that is read-only *because of its tool
list* must not arrive as a sub-agent holding the parent's tools. The whole
prompt travels too, not its first line.

**A skill that cannot be expressed as a CLI sub-agent is refused by name at
construction**, raising `SkillNotProjectable`, rather than projected into
something that looks similar and behaves differently. A skill carrying custom
agentkit tools is the common case: those cannot reach a CLI sub-agent directly,
because the CLI only knows its own tool names and whatever MCP serves it. Serve
them with `serve_registry` and the sub-agent can reach them by their
`mcp__<server>__<tool>` names.

## Capabilities: the Rule-of-Two applies here too

[`RunPolicy`](agents.md) refuses a tool set that reaches private data, untrusted
content and egress at once — the *lethal trifecta*, where a crafted input can
instruct an agent to read something sensitive and send it somewhere.

A CLI session is exactly that shape, and `RunPolicy` could not see it, because
the cognition declared no capabilities. It does now:

```python
from agentkit.agents import RunPolicy
from agentkit.agents.cognition import ClaudeCliCognition


class _Session:
    """RunPolicy reads `.caps` off each entry; a CLI session is one entry."""

    def __init__(self, caps: tuple[str, ...]) -> None:
        self.name, self.caps = "cli-session", caps


for tools in [("Read",), ("Read", "WebFetch")]:
    cognition = ClaudeCliCognition(tools=tools)
    verdict = RunPolicy().check([_Session(cognition.caps)])
    print(tools, "->", verdict.allowed)
```

```text
('Read',) -> True
('Read', 'WebFetch') -> False
```

Two results there are worth pausing on.

**`WebFetch` supplies two legs on its own** — it ingests untrusted content *and*
reaches the network — so `Read` plus `WebFetch` is already the full trifecta
without a `Bash` in sight. The intuition that you need three dangerous-looking
tools is wrong.

**`Task` and `SlashCommand` count as the whole trifecta by themselves.** They
are indirections to the full tool set, so tagging them narrowly would let a
caller launder the trifecta through one innocuous-looking name.

The default `tools=None` means every built-in tool, which is the trifecta.
That is not a bug in the tagging — it is an accurate description of what an
unrestricted CLI session can do.

## Testing it

[`FakeClaudeCli`](testing.md#fakeclaudecli-testing-the-cli-path-without-spending)
sits at the spawn seam, so a test still exercises the real parsing, the real
budget charging and the real event mapping — which is where every bug on this
path has actually lived.

```python
from agentkit.testing import FakeClaudeCli
from agentkit.agents.cognition import ClaudeCliCognition

cli = FakeClaudeCli.script([
    {"type": "result", "subtype": "success", "result": "done"},
])
cognition = ClaudeCliCognition(spawn=cli)
print(type(cognition).__name__, cli.spawns)
```

```text
ClaudeCliCognition 0
```

It found a production bug with its first hostile input: a line that is not valid
UTF-8 raised `UnicodeDecodeError`, which is not a `JSONDecodeError`, so it
escaped the parser's handler — and because the reader stopped before the
`result` payload, a completed run was charged $0.00.

## What bites people

- **The middleware chain does not apply to native tools.** The warning above is
  the only thing standing between you and a session where your `egress` guard is
  wired, documented, and doing nothing.
- **`tools=None` is not "no tools", it is "all of them"** — and therefore the
  full capability trifecta.
- **A denial has to reach the model as a refusal it can act on**, not as a dead
  end. Both the approval gate and the hook path are written that way; a custom
  `Asker` that raises instead of returning a `Decision` breaks it.
- **The CLI's hook timeout is a fail-open**, not a fail-closed. If you override
  the generated deadlines, keep the CLI's the loosest.
- **`serve_registry` and `ApprovalServer` require a generated bearer token** by
  default, and `cli_kwargs()` carries it. Loopback alone was the containment
  before, which held only as long as nothing untrusted shared the host — and the
  point of this cognition is that the CLI runs `Bash` in that same namespace.
  `auth="none"` restores the old behaviour if you want it, by name.

## Related

- [The Codex CLI](codex-cli.md) — the sibling cognition. Same contract, and four
  real differences: an OS sandbox instead of a tool list, no system-prompt flag,
  no CLI-reported cost, and a session that resumes a thread rather than holding
  a process.
- [Integrations (MCP)](integrations.md) — `serve_registry` and `ApprovalServer`,
  the two servers this page points at.
- [Agents](agents.md) — `ClaudeCliCognition` alongside the other three
  cognitions, and `RunPolicy`.
- [Skills](skills.md) — what `as_cli_agents` projects.
- [Middlewares](middlewares.md) — the chain that does *not* reach native tools.
- [Testing](testing.md) — `FakeClaudeCli` in full.
- [Plug the claude CLI into FastAPI code-gen](../recipes/claude-cli-fastapi-code-gen.md)
  — the runnable end-to-end version.
