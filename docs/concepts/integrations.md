# Integrations

An integration lets an agent use software that was never written for
agentkit — and, in one case, lets other software use agentkit.

Today there is exactly one: the **Model Context Protocol** (MCP), under
`agentkit.integrations.mcp`.

## What MCP is, and why a framework bothers

MCP is a small wire protocol for the thing every agent framework
reinvents: "here are some tools you can call, some documents you can
read, and some prompts you can use." A *server* implements that
protocol; a *client* — your agent — talks to it. The server can be a
subprocess on the same machine or an HTTP endpoint on another
continent, and it can be written in any language.

The reason this matters is arithmetic. There are thousands of published
MCP servers — filesystem, git, GitHub, Postgres, Slack, search, ticket
systems — plus whatever internal ones your company already runs. Without
a protocol, using any of them from Python means writing and maintaining
an adapter per server: a schema by hand, an argument-marshalling layer,
an error path, a test. That work does not compose and it does not stop.
With a protocol, one adapter serves all of them.

So `agentkit.integrations.mcp` is a translation layer with three
directions and one reversal:

| MCP concept | becomes an agentkit | via |
|---|---|---|
| tool | `Tool` | `mcp_tools(client)` |
| resource | `MemorySource` | `mcp_resources(client)` |
| prompt | `Prompt` | `mcp_prompts(client)` |
| *(the reversal)* | agentkit **is** the server | `ApprovalServer` |

Each integration is opt-in and gated behind a `pyproject` extra so the
core stays dependency-free:

```bash
pip install "arc-agentkit[mcp]"
```

Importing `agentkit.integrations.mcp` without the extra raises
`ImportError` immediately, with the install line in the message, rather
than failing somewhere deep in a transport later.

!!! tip "There is a recipe for the common case"
    [Consume MCP tools from an agent](../recipes/mcp-tools.md) is the
    copy-paste task page. This page is the layer above it: what the
    pieces are, which transport applies, what is cached and for how
    long, and what the server side is for.

## The smallest working example

This spawns a real MCP server as a subprocess, adapts its tools, and
calls one. It needs no API key, no network, and no third-party server —
the server is six lines of `mcp` and runs under the same interpreter.

```python
"""Spawn a tiny MCP server and call its tool as an agentkit Tool."""

import asyncio
import logging
import sys

from agentkit.integrations.mcp import MCPClient, StdioServer, mcp_tools
from agentkit.testing import make_test_ctx

logging.disable(logging.INFO)  # the mcp SDK narrates every request at INFO

# A complete MCP server, run via `python -c` so this file stands alone.
# In real use `command` is `uvx`, `npx`, or a compiled binary.
SERVER = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("demo", log_level="ERROR")

@mcp.tool()
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

mcp.run()
"""


async def main() -> None:
    server = StdioServer(command=sys.executable, args=("-c", SERVER))
    async with MCPClient(server) as mcp:
        tools = await mcp_tools(mcp, prefix="demo_")
        tool = tools[0]
        print(tool.name, "|", tool.description)
        print("params:", tool.schema.parameters["properties"])
        print("side_effecting:", tool.side_effecting)

        ctx = make_test_ctx()
        print("result:", await tool.run({"a": 2, "b": 3}, ctx))


asyncio.run(main())
```

```text
demo_add | Add two integers.
params: {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
side_effecting: True
result: 5
```

The returned objects satisfy the `Tool` Protocol structurally — no
inheritance, no registration — so they drop into a cognition unchanged:

```python
"""The same server, driven by a scripted FakeLLM instead of a real one."""

import asyncio
import logging
import sys

from agentkit import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.integrations.mcp import MCPClient, StdioServer, mcp_tools
from agentkit.kernel.types import ToolCall
from agentkit.testing import FakeLLM, Turn, make_test_ctx

logging.disable(logging.INFO)

SERVER = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("demo", log_level="ERROR")

@mcp.tool()
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

mcp.run()
"""


async def main() -> None:
    llm = FakeLLM.script([
        Turn(tool_calls=(ToolCall(id="1", name="demo_add", arguments={"a": 2, "b": 3}),)),
        Turn(content="2 + 3 is 5."),
    ])
    ctx = make_test_ctx(llm=llm)
    async with MCPClient(StdioServer(command=sys.executable, args=("-c", SERVER))) as mcp:
        agent = Agent(
            name="calc",
            model="fake",
            prompt="Use the tools.",
            cognition=ReActCognition(tools=await mcp_tools(mcp, prefix="demo_")),
        )
        result = await agent.run("What is 2 + 3?", ctx)
        print("output:", result.output, "| stop_reason:", result.stop_reason)


asyncio.run(main())
```

```text
output: 2 + 3 is 5. | stop_reason: complete
```

Swap `FakeLLM` for a real provider and nothing about the MCP wiring
changes.

## How it works

### `MCPClient` owns the connection

The raw MCP SDK gives you a two-level context stack: open a transport,
then open and initialize a session on top of it. `MCPClient` collapses
that into one `async with`, using an `AsyncExitStack` so a failure
half-way through entering unwinds what was already opened — a spawned
subprocess or an HTTP client does not leak because `initialize()`
raised.

After entry, `client.session` is the live `mcp.ClientSession`. Before
entry — or after exit — reading it raises `RuntimeError` with the
message *"MCPClient not entered — use `async with MCPClient(...) as c:`"*.

On exit the client drops its caches as well as its session, so a torn-down
client can never hand back tool metadata for a server it is no longer
connected to.

### Two transports

```python
from agentkit.integrations.mcp import StdioServer, StreamableHttpServer

local = StdioServer(
    command="uvx",
    args=("mcp-server-time", "--local-timezone", "UTC"),
    env={"TZ": "UTC"},            # overlays on os.environ in the subprocess
    cwd="/srv/workdir",
)

remote = StreamableHttpServer(
    url="https://mcp.example.com/mcp",
    headers={"Authorization": "Bearer …"},
    timeout_s=30.0,
)
```

| | `StdioServer` | `StreamableHttpServer` |
|---|---|---|
| Shape | spawns a subprocess, speaks over its stdin/stdout | HTTP POST with SSE responses |
| Use when | the server is a CLI you can run locally (`uvx`, `npx`, a binary) | the server is hosted, shared, or behind auth |
| Lifetime | one process per `MCPClient` context | one HTTP client per `MCPClient` context |
| Auth | environment variables via `env` | `headers` |

Most published community servers are stdio. Hosted and multi-tenant
servers are streamable HTTP.

Both configs are frozen dataclasses, and both deep-freeze their one
mutable field in `__post_init__` — `env` for stdio, `headers` for HTTP.
The reason is stated plainly in the source: `env` is handed to a
subprocess, and a later in-place edit would change what the *next* spawn
inherits while the config object still reads as the one that was
reviewed; `headers` can carry credentials, and "a value that can be
edited after review is one that can be edited after the review that
approved it." Neither field participates in `__hash__` — identity is the
process (`command`, `args`, `cwd`) or the endpoint (`url`, `timeout_s`).

!!! note "The HTTP transport moved underneath this adapter"
    Upstream renamed *and* re-signatured the streamable-HTTP transport:
    `streamablehttp_client(url, headers=…, timeout=…)` became
    `streamable_http_client(url, http_client=…)`, moving ownership of
    the `httpx.AsyncClient` to the caller. The old spelling still works
    but emits a `DeprecationWarning` — which any caller running under
    `-W error` (this project's own test suite included) turns into a
    hard failure raised from *inside* the transport, where it reads as a
    connection problem rather than a deprecation.

    `client.py` prefers the new spelling and keeps the old as the floor
    of the supported `mcp>=1.28,<2` range. On the new path it builds the
    `httpx.AsyncClient` itself and enters it on the exit stack **above**
    the transport, so unwinding tears the transport down first and it
    never writes into a closed client.

### Caching, and its 30-second TTL

`list_tools()` and `list_resources()` are cached client-side for
`_DEFAULT_TTL_S = 30.0` seconds. The MCP spec allows a server to
advertise its own cache TTL on list responses, but the current Python
client does not surface that field, so the adapter applies a fixed
client-side value instead of inventing one per server.

Two ways out of the cache:

```python
async def refresh(client):
    # A server that just registered a new tool won't be visible for up
    # to 30s otherwise.
    return await client.list_tools(force_refresh=True)
```

…and exiting the client context, which clears both caches.

`list_prompts()` is **not** cached — prompts are fetched fresh every call.

There is a second, independent 30-second cache inside `mcp_resources()`:
the returned `MemorySource` caches its own resource list so a burst of
`query()` calls does not re-list on every one. It is a separate layer, so
a `force_refresh` on the client does not reach through it.

### Resources become memory

MCP resources are named documents the server exposes for reading.
`mcp_resources()` presents them as a queryable `MemorySource`: on
`query()` it lists resources, filters by substring match against
name / description / URI, reads the top-k matches, and returns
`MemoryItem`s.

The ranking is deliberately coarse in v1 — substring match, ties broken
by first-seen order. If you want vector-quality ranking, wrap the source
in `CachedMemory` / `CompactedMemory` or put it in a `CompositeMemory`
next to a `VectorMemory`.

Writes raise `NotImplementedError`. MCP resources are server-authored;
a client-side write path would be a protocol violation.

### Prompts become `Prompt` objects

MCP prompts come in two flavours and only one survives adaptation:

- **Static** (no arguments, or all optional) — rendered eagerly at
  adaptation time and stored as `Prompt.template`, with
  `version="mcp:1"`.
- **Argumented** (any argument with `required=True`) — dropped with a
  logged warning, because there is no template to materialise without
  the arguments. Reach for `MCPClient.get_prompt(name, args)` directly
  instead.

A prompt whose render *fails* is also skipped with a warning rather than
sinking the whole batch.

```python
"""Resources and prompts from the same server."""

import asyncio
import logging
import sys

from agentkit.integrations.mcp import MCPClient, StdioServer, mcp_prompts, mcp_resources
from agentkit.testing import make_test_ctx

logging.disable(logging.INFO)

SERVER = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("demo", log_level="ERROR")

@mcp.resource("docs://runbook")
def runbook() -> str:
    "On-call runbook."
    return "Restart the ingest worker, then check the queue depth."

@mcp.prompt()
def triage() -> str:
    "Static triage prompt."
    return "You triage incidents. Be terse."

@mcp.prompt()
def review(diff: str) -> str:
    "Needs an argument."
    return f"Review this diff: {diff}"

mcp.run()
"""


async def main() -> None:
    ctx = make_test_ctx()
    async with MCPClient(StdioServer(command=sys.executable, args=("-c", SERVER))) as mcp:
        memory = mcp_resources(mcp, name="runbooks")
        items = await memory.query("runbook", k=1, ctx=ctx)
        print("memory:", items[0].content)
        print("metadata:", items[0].metadata)

        prompts = await mcp_prompts(mcp)
        print("static:", {k: (v.version, v.template) for k, v in prompts.items()})
        print("argumented, fetched directly:", await mcp.get_prompt("review", {"diff": "-a +b"}))


asyncio.run(main())
```

```text
mcp_prompts: dropping 'review' — MCP prompt requires arguments, cannot render statically. Use MCPClient.get_prompt(name, args) directly.
memory: Restart the ingest worker, then check the queue depth.
metadata: {'uri': 'docs://runbook', 'name': 'runbook', 'mimeType': 'text/plain'}
static: {'triage': ('mcp:1', 'You triage incidents. Be terse.')}
argumented, fetched directly: Review this diff: -a +b
```

The warning on the first line is the drop, working as designed.

## The other direction: agentkit as an MCP server

`ApprovalServer` is the one place the arrow points the other way.

The problem it solves is specific. `ClaudeCliCognition` delegates the
whole agent loop to the Claude CLI, and the CLI owns its own
permissions. That left a service two options, both bad:
`bypassPermissions` (the agent may do anything, unattended) or
`dontAsk` (anything not pre-approved is denied outright and the run just
fails). agentkit already had the missing middle — `Asker`, the injected
human transport behind its own human-in-the-loop path — but nothing
connected the two.

The CLI's seam is `--permission-prompt-tool`: name an MCP tool and the
CLI calls it instead of prompting a terminal, sending the tool name and
arguments and expecting an allow/deny back. So `ApprovalServer` *is* an
MCP server. Each prompt becomes an `Elicitation`, the application's
`Asker` answers it, and the answer maps back onto the CLI's wire shape.

Because an `Asker` may await a person for as long as it likes, the CLI
turn parks in place — the same behaviour agentkit's own tool loop gets
from the same protocol.

```python
"""agentkit as the MCP server — driven here by agentkit's own MCP client."""

import asyncio
import logging

from agentkit.agents.control.elicitation import Decision, Elicitation
from agentkit.integrations.mcp import ApprovalServer, MCPClient, StreamableHttpServer

logging.disable(logging.INFO)


class AutoReviewer:
    """A stand-in Asker. A real one awaits a person over a socket."""

    async def ask(self, request: Elicitation) -> Decision:
        path = (request.tool_call or {}).get("arguments", {}).get("file_path", "")
        if path.startswith("/etc/"):
            return Decision(kind="reject", actor="ci", note="/etc is out of scope")
        return Decision(kind="approve", actor="ci")


async def main() -> None:
    async with ApprovalServer(
        asker=AutoReviewer(), timeout_s=30.0, auto_allow=("Read",)
    ) as approvals:
        print("tool_name: ", approvals.tool_name)
        print("cli_kwargs:", approvals.cli_kwargs())

        # Ask it the questions the CLI would ask.
        async with MCPClient(StreamableHttpServer(url=approvals.url)) as client:
            print("tools:     ", [t.name for t in await client.list_tools()])
            for args in (
                {"tool_name": "Read", "input": {"file_path": "/etc/passwd"}},
                {"tool_name": "Write", "input": {"file_path": "/etc/passwd"}},
                {"tool_name": "Write", "input": {"file_path": "out.txt"}},
            ):
                res = await client.call_tool("approve", args)
                print(f"  {args['tool_name']:6} -> {res.content[0].text}")
        print("prompts_seen:", approvals.prompts_seen)


asyncio.run(main())
```

```text
tool_name:  mcp__agentkit_approvals__approve
cli_kwargs: {'mcp_config': ('{"mcpServers": {"agentkit_approvals": {"type": "http", "url": "http://127.0.0.1:64619/mcp"}}}',), 'strict_mcp_config': True, 'permission_prompt_tool': 'mcp__agentkit_approvals__approve'}
tools:      ['approve']
  Read   -> {"behavior": "allow", "updatedInput": {"file_path": "/etc/passwd"}}
  Write  -> {"behavior": "deny", "message": "/etc is out of scope"}
  Write  -> {"behavior": "allow", "updatedInput": {"file_path": "out.txt"}}
prompts_seen: 3
```

The port is ephemeral — `ApprovalServer` asks the OS for a free loopback
port at `start()`, so you will see a different one. Picking a fixed port
collides the moment two agents run on one host.

!!! note "What this example does and does not prove"
    Every line above ran: the server bound a loopback port, served a
    real MCP session, and answered three real tool calls. What it does
    **not** exercise is the Claude CLI itself — the binary that would
    normally be the client. The wire values (`tool_name`, `mcp_config`,
    `cli_kwargs()`) are the real ones the CLI is handed; that it accepts
    them is verified by this project's own tests against the binary, not
    by this snippet.

In real use you never call `approve` yourself. You hand
`cli_kwargs()` straight to the cognition:

```python
from agentkit import Agent
from agentkit.agents.cognition.claude_cli import ClaudeCliCognition
from agentkit.integrations.mcp import ApprovalServer


async def run(asker, task, ctx):
    async with ApprovalServer(asker=asker, timeout_s=120.0) as approvals:
        cognition = ClaudeCliCognition(model="claude-sonnet-4-6", **approvals.cli_kwargs())
        return await Agent(name="dev", cognition=cognition).run(task, ctx)
```

### The decision table

`ApprovalServer._decide` never raises. It runs inside an MCP request
handler, and an exception there reaches the CLI as a *malformed result*
— which it reports as a broken permission system rather than a denial,
leaving the model free to retry a call nobody approved. So a transport
failure is a **deny with the reason attached**.

| `Decision.kind` | CLI sees | Note |
|---|---|---|
| `approve`, `value` | `allow` with the original arguments | |
| `modify` (dict `value`) | `allow` with the **edited** arguments | approve-with-changes; the model is not told they changed |
| `reject` (or anything else) | `deny` with `note` | |
| `expired` | `deny`, message names the deadline | fires from `timeout_s`, enforced by the server |

Two details worth internalising:

- **The default has to be deny.** An approval gate that fails open is
  not a gate. But the message says *which* denial it was, because "the
  reviewer said no" and "nobody answered in 60s" call for different
  fixes.
- **`timeout_s` is enforced by the server, not trusted to the `Asker`.**
  The `Asker` protocol allows an implementation to wait forever, and a
  queue worker holding a CLI subprocess open indefinitely is a resource
  leak with a model attached. `timeout_s=None` waits forever — right for
  an interactive UI, wrong for a worker.

### `auto_allow`, and why it exists

`auto_allow` names tools approved without asking. It is not a
convenience knob. The CLI prompts for reads too, and a person clicking
"yes" on forty `Read` calls is not oversight — it is habituation, the
thing that makes the fortieth prompt, the one that mattered, get the
same reflexive yes.

It is also blunt: the match is on tool name only. In the run above,
`Read /etc/passwd` was allowed without reaching the reviewer, while
`Write /etc/passwd` was denied — because `Read` is in `auto_allow` and
arguments are not consulted. Put only tools you would allow with *any*
arguments in that tuple.

## Gotchas

- **Every MCP tool is `side_effecting=True`.** MCP has no standard
  read-only marker, so the adapter assumes the worst. `ReActCognition`
  passes `side_effecting` to `should_gate` as `key_step`, so under
  `autonomy="gated"` *every* MCP tool call pauses for a human — even a
  pure lookup. Under the default `"auto"` none of them do. If you know a
  server is read-only, wrap its tools and flip the flag.
- **`requires_approval` is `False` on every adapted tool.** The adapter
  will not force a gate the run's autonomy tier did not ask for; a tool
  author's `requires_approval=True` is honoured even under `"auto"`, and
  claiming that on someone else's server would be a guess.
- **Name collisions are yours to avoid.** Two servers both exposing
  `search` will shadow each other in one cognition. That is what
  `prefix=` is for: `mcp_tools(fs, prefix="fs_")`,
  `mcp_tools(gh, prefix="gh_")`. The prefix lands on `name` *and* on
  `schema.name`, so the model sees the prefixed name too.
- **Cancellation is pre-dispatch, not mid-call.** `call_tool(..., ctx=ctx)`
  calls `ctx.check_cancelled()` before it hits the transport, so a run
  cancelled while queued never reaches the server. To stop a call
  already in flight you must exit the `MCPClient` context, which tears
  down the transport — and the session is not reusable afterwards.
- **Non-text tool output is dropped with a marker.** MCP tools return a
  list of text / image / audio / resource blocks. Text is concatenated;
  anything else becomes `[<type> content omitted]` so the agent sees
  something attributable rather than a silent hole. `isError=True`
  results come back as `"ERROR: …"` strings, not exceptions.
- **`read_resource()` drops binary blobs** for the same reason — it
  returns the human/model-facing text.
- **One session per context.** Servers that keep per-session state (open
  file handles, a DB transaction) lose it if you split calls across
  independent `MCPClient` contexts.
- **`ApprovalServer` binds loopback with no authentication.** Anything
  able to reach that port can answer permission prompts on the agent's
  behalf. Loopback-only *is* the containment; do not bind it to a
  routable address.
- **The `mcp` extra brings `pydantic`** (and, transitively, `uvicorn`
  and `starlette`, which is why `ApprovalServer` adds no new
  dependency). If your install was pydantic-free by design, the extra
  ends that.

## Related

- [Consume MCP tools from an agent](../recipes/mcp-tools.md) — the
  task-shaped version of the client half.
- [Human-in-the-loop tool approval](../recipes/hitl-tool-approval.md) —
  agentkit's own gate, the one `ApprovalServer` borrows the `Asker`
  from.
- [Elicit a value from a human](../recipes/elicit-a-value-from-a-human.md)
  — `Elicitation` / `Decision` in full.
- [Plug the claude CLI into FastAPI code-gen](../recipes/claude-cli-fastapi-code-gen.md)
  — `ClaudeCliCognition` end to end.
- [Concepts · Agents](agents.md) — where `Tool`, `MemorySource` and
  `Prompt` plug in.
- The protocol itself: <https://modelcontextprotocol.io>
