# How do I plug the `claude` CLI into a FastAPI code-generation endpoint?

## When you'd want this

You have the [`claude` CLI](https://docs.claude.com/en/docs/claude-code)
installed locally and already logged in (`claude login` set up your
OAuth), and you want a small HTTP endpoint that runs Claude — with tool
access and permission modes — inside an agentkit run without you
managing API keys, provider quotas, or a second SDK. `ClaudeCliCognition`
subprocesses the local CLI per run, streams its stream-JSON output as
agentkit `StreamEvent`s, and hands you back the usual
`AgentResult` — usage, session id, cost estimate.

Reach for this when:

- You want local-first code-gen against `~/.claude` OAuth instead of
  distributing API keys to your server.
- You need the CLI's built-in tools (`Read`, `Bash`, `Edit`, ...) and
  its permission modes (`acceptEdits`, `plan`, `bypassPermissions`).
- You want the CLI's cost estimate surfaced on `Usage.cost_usd`
  automatically.

## Working code

```python
"""Streaming code-gen endpoint over the local `claude` CLI.

Run: `uv run --with fastapi --with uvicorn uvicorn app:app`
Then: `curl -N -X POST localhost:8000/generate -d 'a python quicksort'`
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from agentkit import Agent, Scope
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.runtime import RunContext, Services

app = FastAPI()

# One cognition, shared across requests. It's stateless — every drive
# spawns a fresh subprocess. The class-level semaphore (`max_concurrent`)
# bounds parallel spawns so we don't hit the SDK's ~200-parallel hang.
cognition = ClaudeCliCognition(
    model="claude-opus-4-5",
    permission_mode="acceptEdits",
    working_dir=Path("/tmp/codegen-sandbox"),
    # Isolate CLI settings/auth per deployment. Concurrent writes to
    # ~/.claude.json can corrupt it — a dedicated `config_dir` avoids that.
    config_dir=Path("/var/lib/mysvc/claude"),
    allowed_tools=("Read", "Write", "Edit", "Bash"),
    max_concurrent=8,
)

agent = Agent(
    name="codegen",
    prompt="You are a terse code generator. Produce runnable code only.",
    cognition=cognition,
)


@app.post("/generate")
async def generate(request: Request) -> StreamingResponse:
    prompt = (await request.body()).decode() or "hello, world"

    async def sse_stream():
        # Minimal RunContext — no persistence, no tracing, just a run id.
        ctx = RunContext(correlation_id="req-1", scope=Scope(), services=Services())
        async for ev in agent.stream(prompt, ctx):
            if ev.type == "message_delta":
                yield f"data: {ev.text}\n\n"
            elif ev.type == "tool_call":
                yield f"event: tool_call\ndata: {ev.tool_call.name}\n\n"
            elif ev.type == "final":
                yield (
                    f"event: done\n"
                    f"data: session_id={ev.result.evals.get('session_id', '')}"
                    f" cost_usd={ev.result.usage.cost_usd:.4f}\n\n"
                )

    return StreamingResponse(sse_stream(), media_type="text/event-stream")
```

Every request spawns one `claude -p "<prompt>" --output-format
stream-json --verbose` subprocess with your model, system prompt,
allowed tool list, permission mode, and working directory threaded
through. Auth is entirely the CLI's problem: whatever
`CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, or
`~/.claude/`-stored OAuth the CLI would find on its own is what your
endpoint uses.

## How it works

`ClaudeCliCognition.drive(...)` maps the CLI's stream-JSON output onto
the standard `StreamEvent` stream:

| CLI event                        | agentkit event                    |
|----------------------------------|-----------------------------------|
| `{"type": "assistant", ...text}` | `StreamEvent("message_delta")`    |
| `{"type": "assistant", ...tool_use}` | `StreamEvent("tool_call")`    |
| `{"type": "user", ...tool_result}` | `StreamEvent("tool_result")`    |
| `{"type": "result", ...}` (final)| `StreamEvent("final", result=…)`  |

Cost, session id, and duration are lifted onto `AgentResult`:

```python
final = await agent.run("write a REST client", ctx)
final.usage.cost_usd            # CLI-side estimate (see below)
final.evals["session_id"]       # pass back as resume_session_id= (NOT session_id=)
final.evals["cli_duration_ms"]  # end-to-end CLI wall time
```

## Typed output from the CLI

`output=` works here exactly as it does on any other agent. The schema goes
out as `--json-schema`, the CLI validates its own final answer against it
(re-prompting itself on a mismatch), and the validated value comes back
through the same `SchemaAdapter` as a real Python object:

```python
class Invoice(BaseModel):
    vendor: str
    total: float

agent = Agent(name="extract", output=Invoice, cognition=ClaudeCliCognition())
result = await agent.run("Pull the vendor and total out of invoice.pdf", ctx)

result.parsed          # Invoice(vendor='ACME', total=42.5)
result.evals["structured_output"]   # the raw validated dict
```

Three outcomes, and only the first is a success — the other two set
`partial=True` and `stop_reason="invalid_output"`:

| Outcome | `evals["stop_reason"]` |
|---|---|
| A conforming value | *(absent — the run is clean)* |
| The CLI exhausted its retries | `error_max_structured_output_retries` |
| Exit 0 but no value at all | `structured_output_missing` |
| A value the *Python* type rejects | `structured_output_mismatch` |

The last one is worth knowing: the CLI validates against JSON Schema, which
is looser than most Python types. `evals["structured_output_error"]` carries
the per-field diagnostics.

Pass `json_schema=` on the cognition to send a shape that isn't the agent's
Python type; it overrides `output=`.

## Spend

The CLI bypasses the `Invoker`, so the `meter()` middleware never sees its
usage. That used to mean CLI spend was invisible: a $50 run against a $1
`Budget` completed happily and the ledger read `$0.00`. Both ends are wired
now.

```python
ctx = RunContext("run-1", Scope(), services=Services(), budget=Budget(max_cost_usd=2.50))
result = await agent.run("Refactor the auth module", ctx)

ctx.budget.spent()      # Decimal('0.750000') — what the CLI reported
ctx.budget.remaining()  # Decimal('1.750000')
```

- **Before the spawn**, the run's *remaining* headroom goes out as
  `--max-budget-usd`, so the CLI stops itself mid-flight instead of being
  audited after the money is gone. An already-exhausted budget doesn't spawn
  at all — the result carries `stop_reason="budget_exhausted"`, which is
  resumable: raise the ceiling and run again.
- **After the run**, the reported cost and tokens are charged to every meter
  on the context (`Budget`, any `Quota`) and to `ctx.actor_budget`.
- A ceiling crossed by *this* run is recorded in `evals["meter_error"]`, never
  raised — the money is spent and the answer exists; throwing it away helps
  nobody.
- `meter_spend=False` opts a run out of both ends (a warm-up call, an eval
  harness with its own accounting).

`Usage.cost_usd` remains a CLI-side estimate, so the ledger it feeds is an
estimate too.

## One process, many turns

`drive()` spawns a subprocess per turn. That costs two to five seconds of CLI
warm-up **every** time — and the turns share no context, so a follow-up
question has no idea what was just discussed. Measured on the same two-turn
conversation:

| | wall | turn 2 answers |
|---|---|---|
| `session()` | 9.7s | `8137` |
| `drive()` ×2 | 16.1s | *"I don't have a record of you asking me to remember a number"* |

A session holds the process open and feeds turns over stdin, so the CLI keeps
its own conversation context:

```python
async with cognition.session() as chat:
    async for ev in chat.turn("Summarise README.md"):
        ...
    async for ev in chat.turn("Now list the risks you skipped"):
        ...
print(chat.session_id)      # for a later --resume
```

A session is also `Cognition`-shaped, so it can *be* an agent's cognition and
consecutive `agent.run(...)` calls share one process and one conversation.

Every per-turn contract is unchanged — exactly one terminal `final` event, the
same stop reasons, the same metering — because both paths run the same
finaliser. What differs is what a shared process implies:

- **Turns are serialised.** One stdin and one transcript, so a second
  concurrent `turn()` waits rather than interleaving two conversations.
- **A dead process stays dead.** If the CLI exits mid-session, that turn
  reports `stop_reason="session_closed"` and later turns refuse rather than
  silently starting a fresh conversation with no history.
- **Cancelling a turn ends the session.** No protocol message retracts a
  half-finished turn, so the process is terminated — better than a context
  holding half an answer nobody saw.
- **`output=` is not per-turn.** `--json-schema` is fixed at spawn. Use
  `drive()` for a typed run, or set `json_schema=` on the cognition before
  opening the session; asking per turn is refused with that message rather
  than silently returning prose.

## Streaming tokens

By default the CLI emits one `assistant` message per **completed** content
block, so `message_delta` arrives per paragraph. For a UI with a cursor in it,
turn on partial messages:

```python
ClaudeCliCognition(partial_messages=True)   # --include-partial-messages
```

Each provider token then arrives as its own `message_delta`. The completed
block still arrives too — the cognition uses it to accumulate
`AgentResult.output` and does *not* re-emit it, so a consumer sees each token
exactly once and the final text is written once. Thinking deltas stream the
same way and stay out of `output` (they land in `evals["thinking"]`).

## Diagnostics in the result

```python
result.evals["cli_init"]      # model, mcp_servers, mcp_server_errors, plugin_errors, version
result.evals["api_retries"]   # one entry per provider retry (also emitted live as `step` events)
result.evals["cli_duration_ms"]
```

`mcp_server_errors` and `plugin_errors` appear **only when non-empty** — the
CLI validates each `--mcp-config` entry, skips invalid ones and runs anyway,
exiting cleanly. Their presence is the signal, which makes them the CI gate
the CLI docs recommend:

```python
if result.evals.get("cli_init", {}).get("mcp_server_errors"):
    raise SystemExit("an MCP server did not load")
```

## Running it as a service

Four flags matter once this is behind an API rather than on your laptop.

```python
ClaudeCliCognition(
    bare=True,                    # --bare
    stable_prompt_prefix=True,    # --exclude-dynamic-system-prompt-sections
    mcp_config=("/etc/mysvc/mcp.json",),
    strict_mcp_config=True,       # only the servers YOU declared
    no_session_persistence=True,  # nothing written to disk
    fallback_model=("claude-sonnet-4-6",),
    add_dirs=(Path("/srv/shared-templates"),),
    settings='{"apiKeyHelper": "/usr/local/bin/fetch-key"}',
)
```

- **`bare=True`** skips auto-discovery of hooks, skills, commands, subagents,
  plugins, MCP servers, auto memory and `CLAUDE.md`. Without it, a `-p`
  session runs the hooks in the working directory's `.claude/settings.json`
  and connects the servers in its `.mcp.json` — including a repository you
  just cloned to review. The CLI docs call bare "the recommended mode for
  scripted and SDK calls". **It also never reads OAuth credentials or the
  keychain**, so set `ANTHROPIC_API_KEY` (or an `apiKeyHelper` in `settings`);
  the cognition warns if neither is present, because the CLI's own error
  ("Not logged in · Please run /login") points at exactly the wrong fix on a
  machine where `claude` works fine interactively.
- **`stable_prompt_prefix=True`** moves per-machine sections (cwd,
  environment, memory paths) out of the system prompt and into the first user
  message, so the cache-stable prefix is byte-identical across users and
  machines running the same task. Documented for precisely this: "Use with
  `-p` for scripted, multi-user workloads."
- **`strict_mcp_config=True`** ignores every MCP configuration except the ones
  you passed. It's refused without `mcp_config=`, since on its own it would
  just leave the session with no servers.
- **`no_session_persistence=True`** keeps sessions off disk — a containment
  control in a multi-tenant service, not an optimisation.

## Configuration walkthrough

```python
ClaudeCliCognition(
    claude_bin="claude",              # override for a specific install path
    model="claude-opus-4-5",           # -> --model
    working_dir=Path("/tmp/sandbox"),  # subprocess cwd; --add-dir surface
    config_dir=Path("/var/claude"),    # CLAUDE_CONFIG_DIR — isolated auth
    system_prompt_mode="append",       # agent.prompt -> --append-system-prompt
    tools=("Read", "Grep"),            # -> --tools Read Grep   (what EXISTS)
    allowed_tools=("Read", "Grep"),    # -> --allowed-tools     (what runs unprompted)
    disallowed_tools=("Bash",),        # -> --disallowed-tools Bash
    permission_mode="acceptEdits",     # -> --permission-mode acceptEdits
    max_turns=6,                       # -> --max-turns 6
    resume_session_id=prior_evals["session_id"],  # -> --resume  (continue that run)
    extra_args=("--fallback-model", "claude-sonnet-4-6"),
    terminate_grace_s=5.0,             # SIGTERM grace before SIGKILL
    max_concurrent=8,                  # class-level BoundedSemaphore
)
```

## Gotchas

- **`tools` and `allowed_tools` are different flags.** `--tools` decides
  which built-in tools the session *has*; `--allowed-tools` decides which
  run *without a permission prompt*. Listing three tools in
  `allowed_tools` alone leaves every other tool — Bash included —
  available and merely prompting. If you mean a sandbox, set `tools=`.
- **`agent.prompt` is APPENDED, not substituted.** `--system-prompt`
  replaces Claude Code's entire system prompt, tool guidance included, so
  a one-line persona used to turn a capable coding agent into a chat model
  holding tools it no longer knew how to drive. Pass
  `system_prompt_mode="replace"` if you genuinely want that.
- **`session_id` names a session; `resume_session_id` continues one.**
  `--session-id` assigns a UUID to a *new* conversation. To pick up a
  previous run, pass its `evals["session_id"]` as `resume_session_id=`.
  A non-UUID `session_id` is refused at construction.
- **Cold start is real.** First subprocess per config_dir is 2–5s of
  CLI warmup even before the model call. Second-and-later calls are
  faster if you keep the same `config_dir`.
- **`~/.claude.json` corrupts under concurrent writes.** The default
  `~/.claude` is a single-writer store; running two production
  workers against the same home directory will eventually mangle it.
  Point `config_dir=Path(...)` at a per-worker directory in
  production, and treat it as owned by the process.
- **`Usage.cost_usd` is a CLI estimate.** The CLI computes it from
  published per-token prices; provider invoices can drift from it.
  Surface it as an estimate in your UI, not a bill.
- **`agent.model` is ignored.** Only `ClaudeCliCognition.model` is
  passed to `--model`. If both are set, the cognition wins; if neither
  is set, the CLI picks its default.
- **`ctx.check_cancelled()` fires between events, not mid-event.**
  On cancel the cognition sends SIGTERM (waits `terminate_grace_s`)
  then SIGKILL, and still emits a terminal `final` with `partial=True`
  and `evals["stop_reason"] == "cancelled"`. Long uninterruptible
  tool calls inside the CLI can only be killed, not "asked" to stop.

## Related

- [Concepts · Agents](../concepts/agents.md) — how a `Cognition` slots
  into an `Agent`.
- [Human-in-the-loop tool approval](hitl-tool-approval.md) — the
  agentkit-side gate that runs OUTSIDE this cognition (the CLI has
  its own permission model).
- [Cap spend with Budget and Quota](spend-budget-and-quota.md) —
  wire cost limits over the top of `ClaudeCliCognition`.
