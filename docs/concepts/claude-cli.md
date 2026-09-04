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

## Which credentials the CLI actually uses

Setting `config_dir=` to isolate a tenant does not, on its own, isolate a
tenant. The CLI prefers an ambient API key over a signed-in profile, and it
says so itself on stderr:

```
⚠ claude.ai connectors are disabled because ANTHROPIC_API_KEY or another auth
  source is set and takes precedence over your claude.ai login
```

So a service that exports `ANTHROPIC_API_KEY` for one purpose and passes
`config_dir=` for another silently runs every tenant on the shared key. The
isolation you asked for is exactly the thing that does not happen, and the only
signal is a line on a stream the cognition surfaces only when a run fails.

`env_policy` makes the choice explicit:

| Policy | The child gets | Use it when |
|---|---|---|
| `"inherit"` | `os.environ`, verbatim | you want the ambient credentials — the historical behaviour |
| `"profile"` | `os.environ` minus the ambient auth and session variables | the CLI should authenticate from `config_dir` / its own login |
| `"isolated"` | `PATH`, `HOME`, proxies, TLS, locale — plus whatever you pass in `env=` | a run should be reproducible across machines |

**`config_dir=` implies `"profile"`.** Pointing the CLI at a configuration
directory is the statement of intent, and inheriting a key that overrides it is
never what that caller meant. Pass `env_policy="inherit"` to opt back out. A
strip is warned about once, naming the variables, so a run that genuinely
wanted the ambient key is told where it went.

`env=` is applied last, over everything the policy produced — including a name
the policy just removed. That is how per-tenant credentials work:

```python
ClaudeCliCognition(
    config_dir=member_dir,                      # → env_policy="profile"
    env={"ANTHROPIC_API_KEY": tenant_key},      # ...this key, not the ambient one
)
```

The Codex cognition has the same field with the same three values. Measured
against codex 0.152.1, an ambient `OPENAI_API_KEY` was *ignored* in favour of
the `CODEX_HOME` login — the opposite precedence — so there the policy buys
determinism rather than fixing a live override.

## `native_tool_policy` — refuse what the chain cannot govern

The warning below tells you the middleware does not reach native CLI tools.
For a service, a warning is not a control:

```python
ClaudeCliCognition(native_tool_policy="deny", tools=("",), mcp_config=[...])
```

`"deny"` refuses **at construction** if the session would hold any native tool,
so a policy violation is a startup failure rather than something nobody
notices. It is satisfiable here: `tools=("",)` plus `mcp_config=` routes every
call back through the `Invoker`, where the chain does apply.

It is *not* satisfiable on Codex — every Codex session has `shell`, and there
is no allow-list and no `PreToolUse` hook — so `CodexCliCognition(native_tool_policy="deny")`
raises unconditionally and says why. That asymmetry is the honest answer to "no
ungovernable tool may run", not an oversight.

Defaults stay at `"warn"`, because `"deny"` changes whether existing code
constructs at all.

## Liveness: `timeouts`

A CLI that stops making progress does not announce it. Measured against
claude 2.1.236 with an invalid `ANTHROPIC_API_KEY`: **no stdout, no stderr, no
exit, for over 45 seconds.** The process was alive, the run looked active, and
nothing in the event stream said otherwise.

You can wrap `drive` in `asyncio.wait_for` — that works, and both cognitions
honour it. What it cannot do is tell you *which* kind of stuck you have, and
that distinction is the whole diagnosis:

| Bound | Measures | A crossing means | Stop reason |
|---|---|---|---|
| `startup` | spawn → first stdout line | the binary never came up: auth, config, a bad flag | `startup_timeout` |
| `first_event` | spawn → first line a UI could show | it came up; the provider is not answering | `first_event_timeout` |
| `idle` | gap between consecutive stdout lines | it went quiet mid-answer; a tool call is stuck | `idle_timeout` |
| `total` | wall clock for the run | the work is simply too big | `total_timeout` |

Three more typed reasons come from the process rather than the clock:

| Reason | Means |
|---|---|
| `malformed_output` | the stream ended without the CLI's own end-of-turn payload |
| `process_crashed` | a negative return code — death by signal, not a chosen exit |
| `schema_rejected` | the CLI refused the schema we handed it at startup |

`malformed_output` fixed a silent wrong answer. A truncated stream — a killed
process, a full disk, a broken pipe — used to report `stop_reason="complete"`,
`partial=False`, handing back half an answer as a finished one. It hid so well
because a *successful* terminal payload also leaves `stop_reason` at `None`, so
"the CLI finished" and "the stream stopped" were the same value. Tracking the
terminal payload explicitly is what separates them.

All four map to `failed` in the closed taxonomy — **not** `expired`, which in
this framework means a human-gate deadline passed and the run *degraded and
continued*. Nothing continues here; the process group is killed and there is no
answer. Not `terminated` either: nobody chose this, a bound did.

When a `startup_timeout`'s stderr carries an authentication failure, the reason
sharpens to `authentication_failed`, with `evals["timeout_kind"]` still naming
the bound underneath. That refinement only ever renames an already-failed run —
it cannot mask a different failure, and the marker list deliberately excludes
the `ANTHROPIC_API_KEY ... takes precedence` line, which the CLI prints on runs
that *work*.

### Defaults are asymmetric on purpose

```python
CliTimeouts(startup=120.0, first_event=None, idle=None, total=None)   # the default
```

Only `startup` is on. It measures local work — boot, read config, resolve auth
— so it cannot false-positive on model latency, and 120s is roughly forty times
the observed warm-up.

**`idle` is the one that bites.** The CLI runs its own tools in its own process,
so a `Bash(npm install)` or a long test suite is *minutes* of legitimate silence
on stdout. An `idle` tuned to model latency will kill working runs. Set it above
the longest tool call the session can make, or leave it off.

```python
ClaudeCliCognition(timeouts=CliTimeouts.production())  # all four bounds set
ClaudeCliCognition(timeouts=CliTimeouts.off())         # the old behaviour, by name
```

`CodexCliCognition` takes the identical field with the identical defaults.

### Cancelling a CLI that has gone quiet

`ctx.check_cancelled()` is polled inside the read loop, which used to mean it
was only checked **between lines**. A CLI mid-`Bash(npm install)` produces no
output for minutes, so a tripped token sat unnoticed until the CLI spoke again
— measured at 83s against the real binary, and *indefinitely* against a process
that had genuinely hung.

The read now surfaces every 250ms while the stream is silent, so a stop button
lands in well under a second on all four paths: both cognitions, `drive` and
sessions alike.

Three things this deliberately is not:

- **Not `asyncio.CancelledError`.** That has always worked — it arrives
  mid-`await` and unwinds immediately. This is the *cooperative* token, the one
  a service's stop button actually holds.
- **Not a `CliTimeouts` field.** Those are bounds that *end* a run, and
  `CliTimeouts.off()` sets every one to `None`. A cancellation cadence living
  there would mean "no timeouts" silently also meant "the stop button waits for
  the CLI", which is the bug rather than the fix.
- **Not a tax on runs that cannot use it.** With no `ctx` there is no token, so
  the loop stays on its allocation-free path. Polling costs ~1.7µs per line —
  roughly what parsing that line's JSON costs, or 35ms across a 20,000-line
  streamed run.

A poll tick is not progress, so it cannot hold the `idle` bound open on a
process that has genuinely stopped talking, and it never lengthens a bound: a
50ms `startup` still fires at 50ms.

### Two ways a run could hang after the CLI was done

Both were found while verifying the cancellation fix against the real
binaries, and both sat *downstream* of every bound — which is what made them
hard to see. The liveness bounds cover the stream; these were after it.

**stderr was read without a bound.** `await proc.stderr.read()` waits for EOF,
and EOF means *every* writer has closed the pipe — not just the process
agentkit spawned. `codex exec` leaves a `codex-code-mode-host` helper behind
holding an inherited stderr, so the read never returned. It ran from the
driver's `finally`, so it hung the run past the timeouts, past cancellation,
with the answer already parsed and never delivered. Roughly **one run in
three**. Stderr is now drained in chunks under a 2s bound, which keeps whatever
arrived rather than discarding it the way a single cancelled `read()` would.

**One long line killed the run.** `create_subprocess_exec` defaults its reader
to **64 KiB**, and both CLIs speak NDJSON where a single line can carry a whole
tool result — the contents of a file the agent just read. Past that the reader
raised `ValueError: Separator is not found, and chunk exceed the limit` and the
run came back `parse_failed` while the CLI exited 0 with the answer in hand.
The limit is now 8 MiB, which is a ceiling rather than a reservation: the
buffer only grows as far as a line actually needs.

Past *that* ceiling the run still fails — it has to, since skipping the line
would silently drop data — but it now says so in terms you can act on:

```
stop_reason : output_line_too_long   (taxonomy: failed)
error       : CliLineTooLong: the CLI emitted a single stdout line larger than
              the 8,388,608-byte reader limit, so neither that line nor the
              rest of the run could be assembled. That is ONE NDJSON payload,
              not the whole stream: in practice it is a tool result carrying a
              very large file. Have the agent read the file in parts, or narrow
              what the tool returns.
```

Deliberately **not** `parse_failed`. Nothing was malformed — the line was valid
JSON that was simply too big to assemble — and `parse_failed` sends an operator
hunting a corrupt payload. The byte count is read off the reader that actually
enforced it, not from the constant, so a caller-supplied `spawn=` with its own
`limit` reports its own number rather than a figure nothing is applying.

If you supply your own `spawn=`, accept `**kwargs` — `limit` is passed
positionally alongside `start_new_session`, and the shipped fakes already
ignore what they don't use.

### A session's system prompt is set at spawn

`agent.prompt` **is** the system prompt. Pass the agent when you open the
session and it reaches the CLI:

```python
async with cognition.session(agent=agent) as chat:
    ...
```

This did not used to work. The session built its argv with `system_prompt=""`
and nothing ever supplied one, so an agent handed to a session kept its schema,
its meters and its name — and silently lost the instructions that say what it
*is*. Measured, the same agent through both entry points:

```
cognition=cog   ->  "ARRR, 2 + 2 be 4, matey!..."
cognition=chat  ->  "4"
```

That second line is the swap `ClaudeCliSession.drive` exists to enable — use a
session *as* an agent's cognition and keep the process warm. It changed the
model's behaviour with no error and nothing missing from the result, which
debugs like a model ignoring its instructions rather than like wiring.

**A later turn cannot change it.** `--system-prompt` is fixed at spawn, exactly
like `--json-schema`, so a turn whose agent carries a *different* prompt is
refused with an explanation rather than answered as something else:

```
the system prompt is a process-level flag on the CLI, so it cannot be changed
per turn. This session was started with 'You are a pirate...'. Pass the agent
when you OPEN the session (cognition.session(agent=...)) so its prompt is set
at spawn, or use ClaudeCliCognition.drive() for a per-run prompt.
```

A refusal costs the turn, not the conversation — nothing was written, so the
CLI never heard about it and the next turn continues normally.

The check compares against what the **process** was spawned with, not against
"does this turn carry a prompt", so the ordinary `session(agent=a)` then
`turn(t, agent=a)` pattern is not refused. Both sides are compared *rendered*,
so a `Prompt` rebuilt per turn — a different object, even a different version,
with the same text — is accepted.

Large prompts take the same file transport `drive` uses, so the argv ceiling
applies here too.

### On a session, the bounds are per turn

A `ClaudeCliSession` holds one process across many turns, so `total` had to
mean something a copy from `drive` could not supply. A session exists in order
to outlive its turns — bounding the whole conversation would kill the thing the
class is for — so every bound is rebuilt per turn and measured from the write
to stdin:

| Bound | On a session, measures |
|---|---|
| `startup` | this turn's write → its first stdout line |
| `first_event` | this turn's write → its first line of *content* |
| `idle` | gap between consecutive lines within the turn |
| `total` | wall clock for the longest single **turn**, not the conversation |

`startup` therefore applies to every turn rather than only the first. That is
the right shape anyway: the first turn pays the CLI's warm-up and later ones
answer sooner, so a bound that clears the first clears the rest with room.

A crossed bound ends the **session**, exactly as a cancel does and for the same
reason — no protocol message retracts a half-finished turn, so the process is
killed and the conversation goes with it. The next `turn()` reports
`session_closed` rather than silently starting a fresh conversation with no
history.

An in-flight `interrupt()` counts as liveness, not as silence. Its
acknowledgement arrives on the same stdout the turn reader is draining, so a
stop button cannot trip an `idle` bound on a turn that is working fine.

`CodexCliSession` needs none of this: it re-spawns per turn and resumes by
thread id, so it runs through the same driver `drive` does and inherits every
bound already.

## Structured output that failed, as data

When a run that declared `output=Invoice` doesn't produce one, the reason used
to arrive as one English sentence in `evals["structured_output_error"]` — built
by joining the adapter's per-field diagnostics. The structure was there and got
thrown away, so an application that wanted to mark the offending field in a form
had to parse the sentence back apart.

```python
from agentkit.agents.cognition import StructuredOutputFailure

result = await agent.run(task, ctx)
failure = StructuredOutputFailure.of(result.evals)
if failure is not None:
    for v in failure.violations:
        form.mark_invalid(v.path, v.message)     # $.lines[0].qty, "…valid integer"
```

Paths are JSONPath (`$.lines[0].qty`), not the adapters' dotted form
(`lines.0.qty`) — the dotted form is ambiguous the moment a mapping has a
numeric-looking key, and `$`-rooted paths are what you can hand to `jq`, a UI,
or an error-grouping key.

Four kinds, because each has a different fix:

| `kind` | Means | Carries violations |
|---|---|---|
| `missing` | schema given, no structured payload came back | no |
| `undecodable` | there was a payload and it wasn't JSON | no (keeps `raw_excerpt`) |
| `schema_mismatch` | it was JSON and didn't fit the declared type | **yes** |
| `retries_exhausted` | the CLI re-prompted *itself* and gave up | no |

`retries_exhausted` is Claude-specific and carries no violations on purpose:
the CLI validates against `--json-schema` internally, and agentkit never sees
the intermediate attempts. Claiming per-field diagnostics there would be
inventing them.

`evals["structured_output_error"]` is still the same string it always was.
The typed form lives alongside it in `evals["structured_output_failure"]` as a
**dict** — `evals` is deep-frozen, checkpointed and serialised, so a dataclass
in there would produce a result that can't round-trip through JSON. `.of()` is
the typed view.

### It renders a repair prompt; it does not send one

`failure.repair_prompt()` gives compact, model-readable feedback naming exactly
the fields to fix. Neither cognition retries on it automatically, and that is
deliberate for Claude: the CLI **already** re-prompts itself against
`--json-schema` and reports `error_max_structured_output_retries` when it gives
up. An agentkit-side loop on top would be a second retry layer over an opaque
one — double the spend, double the latency, and usage accounting that no longer
maps to attempts. Run the repair turn yourself when you want it.

## Secrets: argv is world-readable

Both `mcp_config=` and `settings=` accept **inline JSON**, and that is the
convenient form this class advertises. Wired the documented way, it put a
bearer token into the process argument list:

```
$ ps -eo args | grep claude
claude -p ... --mcp-config {"mcpServers":{"s":{"headers":{"Authorization":"Bearer sk-…"}}}}
```

Any local account can read that. It is the same exposure as the prompt-in-argv
problem, except the payload *is* the credential.

An inline blob is now written to a **0600 file** and replaced by its path, so
the secret never enters argv. A value that is already a path is passed through
untouched — a path is a reference, not a value. The file is removed when the
run ends, however it ends — including when the spawn itself fails, since a
credential left in `/tmp` would be strictly worse than the argv it replaced:
argv at least dies with the process.

A **session** gets the same treatment, and it is the case that matters more.
`drive` leaks for one turn; a session holds its process — and so its argument
list — for the whole conversation. The scratch directory is owned by the
session rather than by a turn, because the CLI may re-read an `--mcp-config`
file at any point while it is alive, and it is torn down by `close()` after the
process is gone.

Nothing about your configuration changes; it is a change of transport. It also
lifts an `ARG_MAX` ceiling on large MCP configurations for free.

### A large system prompt goes to a file too

Same ceiling, different payload. The kernel copies argv **and the environment**
into the new process image under two caps:

| | Cap | Applies to |
|---|---|---|
| darwin | `ARG_MAX` = 1 MiB | argv + envp **combined** |
| Linux | the same total, **plus** `MAX_ARG_STRLEN` = 128 KiB | any **single** argument |

Measured here: one 1,000,000-byte argument spawns fine, twenty 100,000-byte
ones do not — so on macOS it is a shared pot, and your environment spends from
it too. Linux's per-argument cap is the tighter one and the one to design
against: a prompt that works on a laptop can be rejected outright on a CI
runner.

The system prompt was the last payload still travelling that way, and the one
most likely to grow — the task is usually a sentence, while the system prompt
is where retrieved context or a compiled instruction set ends up. Measured
through the cognition: **2,000,000 bytes came back `spawn_failed` with
`OSError: [Errno 7] Argument list too long`**, thrown before the binary ran, so
with no stderr, no CLI diagnostic, and an error naming "the argument list"
rather than which of the four things in it was to blame.

Above 32 KiB the prompt is now written to a 0600 file and passed as
`--system-prompt-file` / `--append-system-prompt-file`. Neither flag appears in
`claude --help` on 2.1.236 and both work — verified not merely accepted but
*honoured*, by handing the CLI a persona through each and getting it back in
the answer.

Below that threshold nothing changes. Always writing a file would make every
run depend on a writable temp dir to do something that currently needs no
filesystem at all — trading a bug that bites large prompts for a failure mode
that could bite every locked-down sandbox.

Two payloads are still stuck inline: `--agents` and `--json-schema`. The CLI
offers no file form for either (`--agents-file` and `--json-schema-file` are
both rejected as unknown options), so they share what is left of the pot — and
vacating the system prompt is what gives them room.

`CodexCliCognition` never had this problem: `system_prompt_mode="prepend"` puts
the text in the first user message, which travels on stdin, and `"replace"`
already wrote a file for `-c experimental_instructions_file=…`.

### Codex cannot do this, and says so

`codex exec` has no config-file flag, so `-c key=value` is the only way to set
an override and agentkit **cannot** move it out of argv. A secret-looking key
in `config_overrides` or `mcp_servers` therefore raises a warning at
construction pointing at the mechanism that does work — Codex's own answer:

```python
CodexCliCognition(
    mcp_servers={"svc": {"url": "...", "bearer_token_env_var": "MY_TOKEN"}},
    env={"MY_TOKEN": token},          # the secret travels in the environment
)
```

`*_env_var` keys *name* a variable to read the token from, so they are the fix
rather than the leak and are never warned about.

### Diagnostics are redacted before they are stored

`evals["stderr"]` is the CLI's stderr, and an `AgentResult` is checkpointed,
logged and fanned out to observers — so a CLI that prints a failing request's
`Authorization` header would have it persisted. Credential-shaped runs
(`sk-…`, `Bearer …`, `ghp_…`, `AKIA…`, JWTs, and `key=value` pairs whose key
names a secret) are replaced before the string reaches `evals`.

This is **defence in depth, not a boundary.** It cannot know every credential
format, so do not conclude that redacted output is safe to publish. It is a
cheap way to stop the common shapes reaching durable storage.

Redaction is structure-preserving: `api_key=[redacted]` keeps the key, because
knowing *which* credential was involved is what an operator needs and the name
of a secret is not the secret. The patterns all require a credential-shaped
prefix or a long opaque run — never a bare word — because a redactor that eats
ordinary diagnostics (`claude-opus-4-5` is exactly the shape a naive rule
swallows) is one that gets switched off wholesale.

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

## Version drift, and the canary for it

Three shipped bugs in this codebase were the same bug: a CLI moved a flag, and
nothing noticed until a run died with a usage message.

| Drift | Effect |
|---|---|
| `codex exec --ask-for-approval` removed | exit 2, no thread started |
| `codex --search` moved to the parent command | exit 2, every `web_search=True` run |
| legacy `task_complete` vocabulary | complete runs read as `malformed_output` |

`tests/agents/cognition/test_cli_version_matrix.py` builds argv for ~20
representative configurations and asks the **installed binary** to parse each
one. The prompt is empty and stdin closes immediately, so a CLI that gets past
parsing has nothing to do and exits — argument parsing happens before any work,
so the whole matrix runs in seconds with no model spend.

**Why not a hand-maintained flag → version table?** Because it rots, and it is
wrong in both directions. Measured against claude 2.1.236: `--max-turns` and
`--permission-prompt-tool` are *absent* from `--help` and work perfectly. A
preflight that rejected flags missing from the help text would have broken two
working configurations while still missing `--search`, which *is* documented —
on a different command. The only ground truth is the binary.

Only argument *rejection* counts as a failure there. A run that starts and then
fails for its own reasons — no credentials, a sandbox denial, an unreachable
MCP server — has already proved the point: the flags parsed. Asserting more
would make the file fail for reasons unrelated to drift, which is how a canary
gets deleted.

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
- **A `config_dir` alone does not isolate credentials.** It switches
  `env_policy` to `"profile"` for you, but if you had been relying on an
  ambient `ANTHROPIC_API_KEY` reaching the child, it no longer does. The
  warning names what was stripped.
- **Inline `mcp_config`/`settings` used to leak into `ps`.** They no longer do,
  but the Codex cognition's `-c` overrides still can — it has nowhere to move
  them to. Use `bearer_token_env_var` + `env=` there.
- **`idle` is not a model-latency bound.** It has to clear the longest *tool
  call* the session can make. This is the single easiest way to turn this
  feature into killed working runs.
- **`native_tool_policy="deny"` is a construction-time refusal**, not a runtime
  filter. It cannot be satisfied at all on the Codex cognition.
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
