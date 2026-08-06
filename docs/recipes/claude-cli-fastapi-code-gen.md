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
final.evals["session_id"]       # for CLI --session-id resumption
final.evals["cli_duration_ms"]  # end-to-end CLI wall time
```

## Configuration walkthrough

```python
ClaudeCliCognition(
    claude_bin="claude",              # override for a specific install path
    model="claude-opus-4-5",           # -> --model
    working_dir=Path("/tmp/sandbox"),  # subprocess cwd; --add-dir surface
    config_dir=Path("/var/claude"),    # CLAUDE_CONFIG_DIR — isolated auth
    allowed_tools=("Read", "Grep"),    # -> --allowed-tools Read,Grep
    disallowed_tools=("Bash",),        # -> --disallowed-tools Bash
    permission_mode="acceptEdits",     # -> --permission-mode acceptEdits
    max_turns=6,                       # -> --max-turns 6
    session_id="ex-abcd-...",          # -> --session-id (for resume)
    extra_args=("--append-system-prompt", "no comments"),
    terminate_grace_s=5.0,             # SIGTERM grace before SIGKILL
    max_concurrent=8,                  # class-level BoundedSemaphore
)
```

## Gotchas

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
