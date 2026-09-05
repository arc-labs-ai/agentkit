# Integrations

An integration lets an agent use software that was never written for
agentkit — and, where the arrow reverses, lets other software use
agentkit.

Today there is exactly one: the **Model Context Protocol** (MCP), under
`agentkit.integrations.mcp`.

!!! tip "Is this page for you?"

    **Reach for it when** you want your agent to use MCP servers, or
    you want to expose your own tools to something else that speaks
    MCP.

    **Skip it for now if** your tools are ordinary Python functions
    and nothing outside your process needs to call them.

## The problem it solves

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
directions and two reversals:

| MCP concept | becomes an agentkit | via |
|---|---|---|
| tool | `Tool` | `mcp_tools(client)` |
| resource | `MemorySource` | `mcp_resources(client)` |
| prompt | `Prompt` | `mcp_prompts(client)` |
| *(reversed)* your `Asker` | an MCP **server** that answers permission prompts | `ApprovalServer` |
| *(reversed)* your `ToolRegistry` | an MCP **server** that runs your tools | `serve_registry` |

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

## The smallest thing that works

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
Both fields carry the same risk, so both get the same treatment.

`env` is handed to a subprocess. Edit it in place later and the *next*
spawn inherits something different, while the config object still reads
as the one that was reviewed.

`headers` can carry credentials. As the source puts it: "a value that can
be edited after review is one that can be edited after the review that
approved it."

Neither field participates in `__hash__`. Identity is the process
(`command`, `args`, `cwd`) or the endpoint (`url`, `timeout_s`).

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

Two things here are servers, and both exist for one reason.
`ClaudeCliCognition` hands the whole agent loop to the Claude CLI, and
the CLI owns that loop: you cannot step into it, wrap a tool call, or
answer a prompt from inside. Anything you want it to do differently has
to be handed to it on the way in — and the seams it offers are MCP
servers.

| what you want | the CLI's seam | agentkit's server |
|---|---|---|
| a human to approve each action | `--permission-prompt-tool` | `ApprovalServer` |
| the loop to call *your* tools | `--mcp-config` | `serve_registry` |

Both run on the same `LoopbackMcpTransport`: one bind-and-wait loop,
one shutdown that genuinely releases the port, one
`mcp__<server>__<tool>` spelling. That is worth a sentence because the
second server was written when the first already had all of it, and two
copies of a bind-and-wait loop is two places to fix "handed the CLI a
URL that was not listening yet" — only one of which would have been
fixed.

`ApprovalServer` is the narrower of the two, and it came first.

The problem it solves is specific. The CLI owns its own permissions,
which left a service two options, both bad:
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

The returned CLI configuration also disables ambient `user`, `project`, and
`local` setting sources while leaving subscription authentication intact. This
is part of the approval guarantee: a global `permissions.allow = ["Write"]`
would otherwise settle the call before the permission-prompt tool could see it.

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
        async with MCPClient(
            StreamableHttpServer(url=approvals.url, headers=approvals.auth_headers)
        ) as client:
            print("tools:     ", [t.name for t in await client.list_tools()])
            for args in (
                {"tool_name": "Read", "input": {"file_path": "/etc/passwd"}},
                {"tool_name": "Write", "input": {"file_path": "/etc/passwd"}},
                {"tool_name": "Write", "input": {"file_path": "out.txt"}},
            ):
                res = await client.call_tool("approve", args)
                print(f"  {args['tool_name']:6} -> {res.content[0].text}")
        print("prompts_seen:", approvals.prompts_seen)
        for d in approvals.decisions:
            print(f"  {d.tool:6} allowed={d.allowed!s:5} "
                  f"source={d.source:10} asked={d.asked}")


asyncio.run(main())
```

```text
tool_name:  mcp__agentkit_approvals__approve
cli_kwargs: {'mcp_config': ('{"mcpServers": {"agentkit_approvals": {"type": "http", "url": "http://127.0.0.1:65162/mcp", "headers": {"Authorization": "Bearer <43-char token>"}}}}',), 'strict_mcp_config': True, 'permission_prompt_tool': 'mcp__agentkit_approvals__approve'}
tools:      ['approve']
  Read   -> {"behavior": "allow", "updatedInput": {"file_path": "/etc/passwd"}}
  Write  -> {"behavior": "deny", "message": "/etc is out of scope"}
  Write  -> {"behavior": "allow", "updatedInput": {"file_path": "out.txt"}}
prompts_seen: 3
  Read   allowed=True  source=auto_allow asked=False
  Write  allowed=False source=asker      asked=True
  Write  allowed=True  source=asker      asked=True
```

Those last three lines are the point of `decisions`, and they are the
reason a count could not do the job. The `Read` was allowed because it
was on `auto_allow` — nobody was consulted, `asked=False`. The two
`Write`s reached the reviewer. `prompts_seen` is `3` for all of them
and says which is which about none of them.

`source` is a closed set for the same reason:

| `source` | what happened |
|---|---|
| `asker` | a person was consulted and answered |
| `auto_allow` | the tool was pre-approved; no `Asker` was invoked |
| `autonomy` | the run's tier did not gate this call |
| `timeout` | nobody answered in time — denied, and **not** a refusal |
| `error` | the `Asker` raised; denied, and **not** a policy decision |

The two that matter most are the ones a boolean collapses. *Allowed
because a person said so* and *allowed because the tier does not gate*
are the facts an auditor most needs apart. And a `timeout` that degraded
to a deny must not read afterwards as a human who said no.

`asked` is separate from `source == "asker"` on purpose: a prompt can
reach a human and then expire. Whether somebody was interrupted is a
fact about what the run cost a person, and it is not recoverable from
the verdict.

Every decision is also emitted on `ctx.emit` as `gate.check` as it
happens, best-effort — the in-memory append comes first, because a
refusal that happened matters more than a record of it that did not.
`arguments` is held on the record but deliberately kept off the
broadcast.

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

That is worth stating plainly because the mistake is easy to make in the
safe-sounding direction: it is tempting to read `auto_allow=("Read",)` as
"reading is fine", when it means "reading **anything** is fine, including
`~/.ssh/id_rsa`".

#### Narrowing it by argument

`auto_allow_when(tool_name, arguments) -> bool` is the opt-in filter for
when a tool is *mostly* safe:

```python
from agentkit.agents.control.elicitation import Decision, Elicitation
from agentkit.integrations.mcp.approvals import ApprovalServer


class Reviewer:
    async def ask(self, request: Elicitation) -> Decision:
        return Decision(kind="approve", actor="ops")


server = ApprovalServer(
    asker=Reviewer(),
    auto_allow=("Read",),
    auto_allow_when=lambda tool, args: not str(
        args.get("file_path", "")
    ).startswith("/etc/"),
)
```

```text
Read README.md    auto-allowed: True
Read /etc/passwd  auto-allowed: False   # falls through to the reviewer
```

Two properties make this safe to put on an approval seam:

- **It can only subtract.** The gate is `tool_name in auto_allow` **and**
  the predicate, so the predicate filters an already-approved set. A
  predicate returning `True` for a tool that is not in `auto_allow`
  cannot let it through — it is never a second way in.
- **A raising predicate falls through to the reviewer.** A bug in your
  filter costs a prompt, not an approval.

Leaving it unset (the default) keeps the old behaviour exactly.

### Serving your own tools: `serve_registry`

`ApprovalServer` answers questions about what the CLI wants to do.
`serve_registry` gives it something to do that it did not ship with.

The gap here was narrow and total. Everything needed to *describe* an
agentkit tool to a model was already present and already correct:
`FunctionTool` derives a `ToolSchema` from a signature and a docstring,
`ToolRegistry` holds them and refuses a duplicate name, and
`ToolArgumentError` refuses a bad call with a message naming the tool,
the offending arguments and the accepted set. `ClaudeCliCognition`
already accepted `mcp_config`. Only the wire was missing — nothing in
agentkit could produce the document that flag reads. So a service that
wanted the CLI to run its own `deploy()` wrote MCP JSON by hand, stood
a server up beside it, and then owned the job of keeping a hand-written
schema in step with the Python signature it claimed to describe.

`serve_registry` takes the registry you already have and hands back an
`McpServerSpec`: the config file, the qualified tool names, the safety
declarations, and — for the HTTP transport — the listener.

```python
"""agentkit as the MCP server: a ToolRegistry the Claude CLI can call.

Driven here by agentkit's own MCP client over a real loopback socket,
because that is the only way to show the error contract holding.
"""

import asyncio
import json
import logging

from agentkit.integrations.mcp import MCPClient, StreamableHttpServer, serve_registry
from agentkit.testing import make_test_ctx
from agentkit.tools import ToolRegistry, tool

logging.disable(logging.INFO)


@tool(side_effecting=False)
def run_check(name: str, strict: bool = False) -> str:
    """Run the named check and report whether it passed."""
    return f"{name}:{'strict' if strict else 'lax'}:ok"


@tool(side_effecting=True, requires_approval=True, caps=("egress",))
def deploy(target: str) -> str:
    """Deploy the current build to the named target environment."""
    return f"deployed to {target}"


async def main() -> None:
    ctx = make_test_ctx()
    spec = serve_registry(ToolRegistry.from_tools([run_check, deploy]), name="engine", ctx=ctx)

    print("tool_names:       ", spec.tool_names)
    print("requires_approval:", spec.requires_approval)
    print("auto_approve:     ", spec.auto_approve)
    print("caps:             ", spec.caps)
    print("cli_kwargs keys:  ", sorted(spec.cli_kwargs()))

    async with spec:
        async with MCPClient(
        StreamableHttpServer(
            url=spec.url, headers=spec.auth_headers, timeout_s=10.0
        )
    ) as client:
            listed = {t.name: t for t in await client.list_tools()}
            print("advertised:       ", sorted(listed))
            assert run_check.schema is not None
            print(
                "schema unchanged: ",
                json.dumps(listed["run_check"].inputSchema)
                == json.dumps(dict(run_check.schema.parameters)),
            )
            print("deploy _meta:     ", listed["deploy"].meta["agentkit"])
            print("deploy read-only: ", listed["deploy"].annotations.readOnlyHint)

            ok = await client.call_tool("run_check", {"name": "lint"})
            print("ok:               ", ok.content[0].text)

            bad = await client.call_tool("run_check", {"name": "lint", "verbose": True})
            print("bad isError:      ", bad.isError)
            print("bad text:         ", bad.content[0].text)

            after = await client.call_tool("run_check", {"name": "after"})
            print("session survived: ", after.content[0].text)

    print("calls_seen:       ", spec.calls_seen)


asyncio.run(main())
```

```text
tool_names:        ('mcp__engine__deploy', 'mcp__engine__run_check')
requires_approval: ('mcp__engine__deploy',)
auto_approve:      ('mcp__engine__run_check',)
caps:              ('egress',)
cli_kwargs keys:   ['mcp_config', 'strict_mcp_config']
advertised:        ['deploy', 'run_check']
schema unchanged:  True
deploy _meta:      {'side_effecting': True, 'requires_approval': True, 'caps': ['egress']}
deploy read-only:  False
ok:                lint:lax:ok
bad isError:       True
bad text:          tool 'run_check' call rejected: unexpected argument(s) ['verbose']. Accepted arguments: ['name', 'strict']
session survived:  after:lax:ok
calls_seen:        3
```

Three lines in that output are the whole design, and each has its own
subsection below: `schema unchanged: True`, the pair
`bad isError: True` / `session survived`, and `deploy _meta`. The last
line is the humbler one — `calls_seen` exists because a session that
reports zero either never called a tool or never *reached* the server,
and those are different bugs to go looking for.

!!! note "What this example does and does not prove"
    Every line above ran: a real loopback listener, a real MCP session
    over a real socket, three real tool calls. What it does **not**
    exercise is the Claude CLI — the binary that would normally be the
    client. The config file and `cli_kwargs()` are the real values it
    is handed; that it accepts them, connects, and runs the tool body
    back in this process is covered by this project's own tests against
    the binary, not by this snippet.

In real use you never call the tools yourself. You hand `cli_kwargs()`
to the cognition, exactly as with `ApprovalServer`:

```python
"""The shape a real caller writes."""

from agentkit import Agent
from agentkit.agents.cognition.claude_cli import ClaudeCliCognition
from agentkit.integrations.mcp import serve_registry


async def run(registry, task, ctx):
    spec = serve_registry(registry, name="engine", ctx=ctx, timeout_s=120.0)
    async with spec:
        cognition = ClaudeCliCognition(
            model="claude-sonnet-4-6",
            allowed_tools=spec.auto_approve,        # the rest still prompt
            **spec.cli_kwargs(builtin_tools=False),  # only OUR tools
        )
        return await Agent(name="dev", cognition=cognition).run(task, ctx)
```

`strict_mcp_config=True` is always in there. Without it the CLI also
loads whatever MCP servers the working directory or the user's home
configuration happen to define — the difference between "these tools"
and "these tools plus a teammate's `.mcp.json`". `builtin_tools=False`
is a *separate* decision and is off by default, because it disables the
CLI's own Read, Grep and Bash, and that is a statement about what the
session can do at all rather than about MCP wiring. A flag named
`strict_mcp_config` should not quietly make it.

#### The schema is not translated

`ToolSchema.parameters` goes out as MCP `inputSchema` unchanged — one
`dict()` unwrap of the `FrozenDict` and nothing else. No key renaming,
no shape massaging, no defaults filled in. That is what the
`schema unchanged: True` line above is asserting, and the project's own
test asserts it as *byte* equality rather than `==`, because two dicts
can compare equal with different key order and the bytes are what the
model is prompted with.

Both sides are already JSON Schema, so a translation step would be a
second description of one thing — and second descriptions drift. The
drift is not cosmetic. It surfaces as the model being shown a schema
the tool does not validate against: it writes a call that matches the
documentation it was given, gets rejected, and has nothing to work with.

This is also why the module builds a lowlevel MCP `Server` rather than
a `FastMCP` one. `FastMCP` derives `inputSchema` from a Python
signature via pydantic and validates arguments against that derived
model *before* the handler runs. Both halves are wrong here: the schema
already exists, and pydantic's rejection would replace
`ToolArgumentError`'s message — which names the tool, the offending
arguments and the accepted set — with a generic validation dump. The
handler is registered `validate_input=False` for the same reason.
agentkit's tools already check their own calls and their diagnosis is
the better one.

Two consequences worth knowing before they surprise you:

- **A tool whose `schema` is `None` is dropped, not advertised.** The
  `Tool` Protocol calls that loop-invisible, and MCP has no way to say
  "callable but undescribed" — `inputSchema` is required. Inventing an
  empty schema would advertise something the model would then call
  blind.
- **Tool names are sanitised, server names are not.** MCP's own
  grammar would allow `run.check`, but the CLI addresses a served tool
  as `mcp__engine__run.check` and callers then paste that into
  `--allowed-tools`, shell arguments and log greps — so agentkit
  applies a narrower rule than MCP's and replaces anything outside
  `[A-Za-z0-9_-]` with `_`, keeping the qualified name one glob-free,
  quote-free token. The model only ever sees the sanitised form, so the
  rename is invisible to it; *you* write the qualified name into
  `allowed_tools` by hand, which is why `spec.mcp_names` publishes the
  mapping rather than leaving it implicit. The **server** name is
  refused rather than rewritten, because you typed that one yourself
  and are holding it in a string somewhere else. Two tool names that
  collide after sanitising (`run.check` and `run check`) raise instead
  of shadowing: `ToolRegistry` refuses a duplicate name for a reason,
  and sanitising would otherwise re-open that hole one layer down.

#### A bad call is a tool error, not a transport error

This is the distinction the module exists to preserve, so it is worth
being concrete about what the alternative costs.

An exception raised out of an MCP request handler is a **transport**
error. It ends the session — and the session is the whole registry. One
tool raising `RuntimeError("the database is on fire")` would take every
other tool down with it mid-run, and what reaches the model is "the
tool system is broken", which is not something it can route around. A
`ToolArgumentError` arriving that way is worse still: the model
authored the bad call and is the only party that can fix it, and it
never gets told what was wrong.

So the dispatch handler never raises. Every failure comes back as an
`isError` **result** on the call, which is reflected into the
transcript:

| what happened | what the model reads |
|---|---|
| `ToolArgumentError` | the message verbatim — tool, bad arguments, accepted set |
| any other exception | `tool 'x' failed: RuntimeError: the database is on fire` |
| no such tool | the name, plus the sorted list of ones that do exist |
| the run was cancelled before dispatch | `the run was cancelled; 'x' was not run` |
| the run was cancelled mid-call | `the run was cancelled while 'x' was running` |

The two cancellation rows are not bookkeeping. `ctx.check_cancelled()`
runs *before* the tool body, because the CLI knows nothing about
agentkit's cancellation seam and will keep driving tools for a run
somebody stopped — a side-effecting tool firing after someone pressed
stop is the failure that check exists to prevent. And `Cancelled` gets
its own clause because it subclasses `RuntimeError`: without one it
would fall into the generic row above and be reported as an ordinary
failure, which is an invitation to retry work that was deliberately
abandoned.

#### The declarations travel, so the Rule of Two still applies

`side_effecting`, `requires_approval` and `caps` go out twice: once as
MCP `ToolAnnotations`, and once verbatim in `_meta` under the
`agentkit` key — `META_KEY` in `agentkit.integrations.mcp.serve`,
namespaced because `_meta` is a shared bag and an unqualified `caps`
would collide with whatever the next middleware wants to attach. Twice,
because annotations are *hints*, and the MCP spec tells clients not to
make decisions on hints from a server they do not trust. `_meta` is
where a cooperating
agentkit-side reader — a `RunPolicy` check, an audit trail — gets the
declarations back without re-inferring them from three booleans that
were lossy on the way out.

`readOnlyHint` is `not (side_effecting or requires_approval)`, not the
obvious `not side_effecting`. `readOnlyHint` is precisely the hint a
client consults to decide it may run something *without asking*, so a
read-only tool that nonetheless declares `requires_approval` — reading
a customer record, say — must not advertise itself as free. That is the
exact case where an approval requirement matters most and is least
visible, which is why `deploy read-only: False` above is a computed
answer rather than a copy.

Because the tags survive the crossing, the lethal-trifecta check still
sees the set it needs to see:

```python
"""The Rule-of-Two check still applies to tools that moved behind MCP."""

import asyncio

from agentkit.agents.control.safety import RunPolicy
from agentkit.integrations.mcp import serve_registry, stdio_command
from agentkit.testing import make_test_ctx
from agentkit.tools import ToolRegistry, tool


@tool(side_effecting=False, caps=("private_data",))
def read_record(customer: str) -> str:
    """Read one customer record out of the private store."""
    return customer


@tool(side_effecting=False, caps=("untrusted_content",))
def fetch_page(url: str) -> str:
    """Fetch a page of untrusted web content and return its text."""
    return url


@tool(side_effecting=True, caps=("egress",))
def notify(channel: str, text: str) -> str:
    """Post a message to the named chat channel — the egress leg."""
    return f"sent to {channel}"


async def main() -> None:
    registry = ToolRegistry.from_tools([read_record, fetch_page, notify])
    spec = serve_registry(
        registry,
        name="engine",
        ctx=make_test_ctx(),
        transport="stdio",
        command=stdio_command("myapp.mcp_server"),
    )
    async with spec:
        verdict = RunPolicy().check(registry.tools())
        print("spec.caps:", spec.caps)
        print("allowed:  ", verdict.allowed)
        print("reason:   ", verdict.reason)


asyncio.run(main())
```

```text
spec.caps: ('egress', 'private_data', 'untrusted_content')
allowed:   False
reason:    lethal trifecta: this tool set combines private-data access, untrusted-content ingestion, and egress in one run — require a human gate or split the run
```

Be honest about where that check runs: on your side, before you start
the CLI. Nothing on the wire enforces it, and `serve_registry` will
serve a trifecta happily if you ask it to. What it guarantees is that
the tags are still *there* to be checked — which is the part that used
to vanish at the boundary, silently, exactly when the tools are
furthest from the caller.

`spec.requires_approval` and `spec.auto_approve` are the same
declarations in the shape `allowed_tools` wants: qualified names, split
by whether the tool asked for a human. Splatting `auto_approve` into
`allowed_tools` is what actually makes the CLI prompt for the others —
`requires_approval=True` is a statement the CLI reads, not a check this
server performs.

!!! warning "Loopback, plus a generated bearer token"
    The HTTP transport binds `127.0.0.1` and, by default, **requires a
    bearer token** that the listener generates. `cli_kwargs()` puts it
    in the config document's `headers`; `spec.auth_headers` is the same
    credential for a non-CLI client. Nothing accepts a caller-supplied
    token — one that a caller can set is one that ends up a constant in
    somebody's config file.

    Loopback alone used to *be* the containment, and that argument holds
    exactly as long as nothing untrusted shares the host. It is the
    wrong assumption for the case this exists to serve: the point of
    `ClaudeCliCognition` is that the CLI runs `Bash` — a build, a
    package install, a script out of a repository the agent was pointed
    at — inside the same network namespace as the server holding the
    agent's own tools. The trust boundary loopback assumes is precisely
    the boundary the tool set is designed to cross.

    A token is a weaker fence than a `0700` directory, and the honest
    statement is that it turns *anything on the host* into *anything
    that can read this process's temp directory*. That is why the config
    file is `0600` and why `stop()` removes it together with the
    listener — the file is now a live credential, so a stale one
    outliving its server would be worse than untidy.

    `auth="none"` restores the old unauthenticated behaviour. It is
    still supported, and now something a caller asks for by name rather
    than gets by default.

#### Which transport, and when

`transport="http"` (the default) keeps the tools in **this** process.
The spec reserves a loopback port, writes a config naming it, and
`async with` starts a uvicorn listener. Your registry, its closures and
its `ctx` are all right here, so a tool can hold a database handle, an
open connection, a half-built object — anything that would not survive
being reconstructed elsewhere.

`transport="stdio"` is a different shape, and the difference is not
cosmetic: the CLI **spawns** a stdio server itself. The tools are
rebuilt in a fresh interpreter, and this process's registry, closures
and `ctx` do not cross that boundary. So `command=` is required, and
passing `transport="stdio"` without it raises immediately with that
explanation rather than writing a config the CLI would fail on a minute
later.

```python
"""transport="stdio": the parent half writes a config naming a command."""

import asyncio
import json

from agentkit.integrations.mcp import serve_registry, stdio_command
from agentkit.testing import make_test_ctx
from agentkit.tools import tool


@tool(side_effecting=False)
def run_check(name: str) -> str:
    """Run the named check and report whether it passed."""
    return f"{name}:ok"


async def main() -> None:
    spec = serve_registry(
        [run_check],                       # a bare sequence works; it is registered for you
        name="engine",
        ctx=make_test_ctx(),
        transport="stdio",
        command=stdio_command("myapp.mcp_server"),
        env={"PYTHONPATH": "/srv/app"},
    )
    async with spec:
        entry = json.loads(spec.config_path.read_text())["mcpServers"]["engine"]
        print("type:", entry["type"])
        print("args:", entry["args"], "(command is sys.executable)")
        print("env: ", entry["env"])
        print("url: ", spec.url)


asyncio.run(main())
```

```text
type: stdio
args: ['-m', 'myapp.mcp_server'] (command is sys.executable)
env:  {'PYTHONPATH': '/srv/app'}
url:  None
```

The command's `main` is the other half, and it calls
`serve_registry_stdio` — the same advertising, the same error contract,
differing only in which pipe carries it:

```python
"""myapp/mcp_server.py — the child the CLI spawns."""

import asyncio

from agentkit.integrations.mcp import serve_registry_stdio
from agentkit.testing import make_test_ctx
from agentkit.tools import tool


@tool(side_effecting=False)
def run_check(name: str) -> str:
    """Run the named check and report whether it passed."""
    return f"{name}:ok"


async def serve() -> None:
    # Serves on this process's stdin/stdout until the peer closes it.
    # A real child builds its own ctx; make_test_ctx keeps this runnable.
    await serve_registry_stdio([run_check], name="engine", ctx=make_test_ctx())


def main() -> None:
    """What `python -m myapp.mcp_server` calls."""
    asyncio.run(serve())
```

No config file is written on that side, and that is not an oversight: a
process that will be *spawned by* the config cannot also be the process
that authors it.

`stdio_command("myapp.mcp_server")` builds `(sys.executable, "-m",
"myapp.mcp_server")` rather than starting from `"python"`, because the
CLI spawns the child with the *caller's* `PATH`, not the virtualenv's,
and a bare `python` there is whatever the system ships — which will not
have agentkit installed. `env=` is often the only way that fresh
interpreter can import the package your tools live in.

`spec.url` is `None` for stdio. There is no endpoint to have, and
returning a plausible-looking one would be a lie a caller could paste
somewhere.

Reach for HTTP unless the tools genuinely have no in-process state.
The process boundary is real work — everything the tools need has to be
reachable from a fresh interpreter running under the CLI's environment
— and it buys you nothing that loopback does not already give you.

#### Pointing Codex at the same server

`spec.cli_kwargs()` writes Claude Code's `--mcp-config` shape. Codex reads
MCP servers out of `config.toml` instead, addressed as
`mcp_servers.<name>.<key>`, so the same spec has a second projection:

```python
from agentkit.agents.cognition import CodexCliCognition
from agentkit.integrations.codex_cli import as_codex_mcp


async def wire(spec):
    async with spec:
        return CodexCliCognition(model="gpt-5-codex", **as_codex_mcp(spec))
```

`spec.codex_kwargs()` is the same thing as a method, next to `cli_kwargs()`.

One part is not a rename. Codex has no header field for a bearer token — it
has `bearer_token_env_var`, naming an environment variable it reads at connect
time. So the projection also returns an `env=` entry, and the credential
travels in the child's environment rather than in the argv, which is
world-readable in `ps` output on most systems.

There is no `builtin_tools=` switch on the Codex side, because Codex has no
tool allow-list: every session has `shell` whatever you serve it. What contains
that is `sandbox=`, covered on [The Codex CLI](codex-cli.md).

#### The lifecycle, and re-entering it

`serve_registry` is synchronous on purpose. It reserves the port and
writes the config file, so the returned spec is complete enough to
build a `ClaudeCliCognition` *before* anything is serving.
`async with spec` then starts the listener; leaving the block stops it
and deletes the config.

The port is reserved by binding a socket and **holding** it, not by
asking the OS for a free port and closing it again. `--mcp-config`
wants a URL written to a file before the CLI is spawned, so the port
has to be known before anything serves on it, and the ask-then-rebind
version leaves a window that a second agent on the same host wins often
enough to matter. It also keeps uvicorn's own bind-failure path out of
the picture, which is worth knowing about: that path is
`logger.error(...); sys.exit(3)` raised *inside* the serve task, so a
port collision propagates a `SystemExit` out of `asyncio.run` — a
library taking down its host process over a busy port, while the
awaiting caller gets a bare `CancelledError` with the port named
nowhere. Binding early raises the real
`OSError: [Errno 48] Address already in use` in your own frame, with
the endpoint appended to the message.

`start()` and `stop()` are both idempotent, and a spec is re-enterable
— a retry loop can `async with spec` a second time. That took a fix to
actually be true. `stop()` deletes the config file, and nothing used to
put it back, so a second entry brought the listener up on the same port
and left `cli_kwargs()` naming a path that no longer existed: a live
server the CLI could not be pointed at, reported as success. `start()`
now rewrites the document.

Deleting the config on the way out is deliberate. A stale one outlives
the port it names, and the next reader gets a URL pointing at nothing —
or worse, at whatever bound that port next. Only a directory the spec
created is removed; a `config_path=` you supplied is yours to keep.

#### Two timeouts that do not read the same

```python
"""A tool's own TimeoutError is not this server's deadline firing."""

import asyncio
import logging

from agentkit.integrations.mcp import MCPClient, StreamableHttpServer, serve_registry
from agentkit.testing import make_test_ctx
from agentkit.tools import tool

logging.disable(logging.INFO)


@tool(side_effecting=False)
async def slow_build() -> str:
    """Wait far past the server's deadline, so only the deadline can end it."""
    await asyncio.sleep(3600)
    return "never"


@tool(side_effecting=False)
def call_upstream(url: str) -> str:
    """Read the named upstream, failing the way a real client fails."""
    raise TimeoutError("upstream read timed out after 3s")


async def main() -> None:
    spec = serve_registry(
        [slow_build, call_upstream], name="engine", ctx=make_test_ctx(), timeout_s=0.05
    )
    async with spec:
        async with MCPClient(
        StreamableHttpServer(
            url=spec.url, headers=spec.auth_headers, timeout_s=10.0
        )
    ) as client:
            ours = await client.call_tool("slow_build", {})
            theirs = await client.call_tool("call_upstream", {"url": "http://x"})
    print("our deadline:", ours.content[0].text)
    print("their error: ", theirs.content[0].text)


asyncio.run(main())
```

```text
our deadline: tool 'slow_build' did not return within 0.05s and was abandoned
their error:  tool 'call_upstream' failed: TimeoutError: upstream read timed out after 3s
```

Those are deliberately different sentences, and the difference is a
bug fix rather than a flourish. The handler used to use
`asyncio.wait_for`, which collapses "our deadline fired" and "the tool
raised a `TimeoutError` of its own" into one indistinguishable
exception — so every one of them was reported as the deadline. With the
default `timeout_s=None`, a tool whose upstream read timed out came
back to the model as *"did not return within Nones and was
abandoned"*: false twice over — nothing was abandoned, and there was no
deadline — with the one message naming what actually failed thrown
away. `asyncio.timeout` keeps them apart, because `deadline.expired()`
is true only when that scope is what did the cancelling. A model cannot
repair a call it has been lied to about.

## What bites people

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
- **Both servers require a generated bearer token, and the token is
  the fence rather than loopback.** Without it, anything able to reach
  `ApprovalServer`'s port could answer permission prompts on the
  agent's behalf, and anything able to reach a `serve_registry` port
  could *run your tools*, `requires_approval` ones included — that flag
  is a declaration the CLI reads, not a check the server performs. The
  credential is checked on HTTP requests and every other ASGI scope is
  refused outright. Still do not give `host=` a routable address:
  the token narrows who can call, not who can reach.
- **`ApprovalDecision.reason` is unredacted text from outside this
  library** — a reviewer's free-text note, or the message of an
  exception raised by your own `Asker` — and unlike `arguments` it IS
  broadcast on `ctx.emit`. Plan retention for it. Nothing here redacts
  anything, deliberately: a library that half-redacted would hand you
  an audit trail nobody can trust to be complete.
- **`serve_registry(timeout_s=None)` is the default, and it means
  forever.** A tool that never returns parks the CLI turn with no
  signal anywhere: the CLI waits on the MCP call, agentkit waits on the
  tool, nothing times out. A default deadline would be the worse
  mistake — it would kill a legitimately slow tool, a long build or a
  human-in-the-loop approval, and read as a flake — so the caller has
  to say which of those they have.
- **A served tool with no `schema` is silently absent.** It is dropped
  rather than advertised with an invented empty `inputSchema`, so a
  tool you registered can simply not be there. `spec.tool_names` is the
  authoritative list of what the CLI will actually see.
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
- [The Codex CLI](codex-cli.md) — `as_codex_mcp`, and the two Claude-side
  adapters (`hook_settings`, `as_cli_agents`) that have no Codex counterpart
  and why.
- [Concepts · Agents](agents.md) — where `Tool`, `MemorySource` and
  `Prompt` plug in.
- The protocol itself: <https://modelcontextprotocol.io>
