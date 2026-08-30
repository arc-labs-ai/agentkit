# How do I consume an MCP server from an agentkit agent?

Somebody has already written the integration you need — for GitHub, for
your database, for a filesystem — and published it as an MCP server.
This is how you point an agent at it instead of writing your own.

## When you'd want this

The [Model Context Protocol](https://modelcontextprotocol.io) ecosystem
ships thousands of pre-built servers — filesystem, git, GitHub, search,
databases, custom internal APIs — as language-agnostic subprocesses or
HTTP endpoints. Rather than wrapping each one in a bespoke agentkit
`Tool`, this integration lets you point at any MCP server and expose
its tools, resources, and prompts through the framework's canonical
`Tool`, `MemorySource`, and `Prompt` Protocols.

Install the extra:

```bash
pip install "arc-agentkit[mcp]"
```

Reach for this when:

- You want an off-the-shelf MCP server's tools inside a
  `ReActCognition` loop without hand-writing an adapter each time.
- You want the server's exposed resources to become a `MemorySource`
  the agent can query for grounding.
- You want the server's authored prompts as versioned `Prompt`
  objects your agent can wire.

## Working code

```python
"""Connect to an MCP server and hand its tools to an agent.

Needs the `mcp` extra and `uvx` on PATH. The model is a `FakeLLM` scripted
to call the server's tool once, so this runs with no API key — swap `llm=`
for `resolve_llm("claude-sonnet-4-6")` and nothing else changes.
"""

from __future__ import annotations

import asyncio

from agentkit import Agent, Scope
from agentkit.agents.cognition import ReActCognition
from agentkit.integrations.mcp import (
    MCPClient,
    StdioServer,
    mcp_prompts,
    mcp_resources,
    mcp_tools,
)
from agentkit.kernel.types import ToolCall
from agentkit.runtime import Invoker, RunContext, Services
from agentkit.testing import FakeLLM, Turn


async def main() -> None:
    # Point at any local MCP server. `uvx` / `npx` / a compiled binary —
    # the client just spawns the command and speaks the protocol over
    # stdio. Swap for `StreamableHttpServer(url=...)` for hosted servers.
    server = StdioServer(
        command="uvx",
        args=("mcp-server-time", "--local-timezone", "UTC"),
    )

    async with MCPClient(server) as mcp:
        tools = await mcp_tools(mcp, prefix="time_")
        print("adapted:", [t.name for t in tools])

        # Resources and prompts are SEPARATE, OPTIONAL MCP capabilities. A
        # tools-only server (this one) answers "Method not found" for both,
        # so ask for them defensively rather than assuming.
        memory = mcp_resources(mcp, name="time_docs")
        try:
            prompts = await mcp_prompts(mcp)
        except Exception:  # McpError: this server publishes no prompts
            prompts = {}

        agent = Agent(
            name="clock",
            model="claude-sonnet-4-6",
            prompt=prompts.get("system") or "Answer with the current time when asked.",
            cognition=ReActCognition(tools=tools),
            memory=memory if prompts else None,
        )

        llm = FakeLLM.script([
            Turn(tool_calls=(ToolCall(
                id="c1",
                name="time_get_current_time",
                arguments={"timezone": "Asia/Tokyo"},
            ),)),
            Turn(content="It is just past noon in Tokyo."),
        ])
        # The Invoker is not optional: the tool loop calls
        # `ctx.invoker.stream(...)` on every iteration, and a bare
        # `Services()` leaves it `None`.
        ctx = RunContext(
            correlation_id="mcp-1",
            scope=Scope(),
            services=Services(invoker=Invoker(llm=llm)),
            autonomy="auto",
        )
        print((await agent.run("What time is it in Tokyo?", ctx)).output)


asyncio.run(main())
```

`mcp_tools()` returns a list satisfying agentkit's `Tool` Protocol —
drop it straight into `ReActCognition(tools=...)`. `mcp_resources()`
returns a `MemorySource`; assign it to `Agent.memory` and the default
grounder will fold results into the prompt. `mcp_prompts()` returns
`{name: Prompt}` for any server-authored prompts that don't require
arguments.

## The three seams

Tools, resources, and prompts are three **independent, optional** MCP
capabilities. Most community servers implement only tools — the time
server above is one of them, which is why the snippet guards the other
two. Each adapter is a one-liner against whichever ones your server has:

```python
# tools ────────────────────────────────────────────────────────────
tools = await mcp_tools(client, prefix="fs_")
for t in tools:
    print(t.name, t.description, t.schema.parameters)
    # t is a `Tool` — usable in ReActCognition unchanged.

# resources ────────────────────────────────────────────────────────
memory = mcp_resources(client, name="my_docs")
items = await memory.query("release notes", k=3, ctx=ctx)
# items: list[MemoryItem(content=..., source="my_docs", metadata={"uri": ...})]

# prompts ──────────────────────────────────────────────────────────
prompts = await mcp_prompts(client)
# {name: Prompt(id=name, version="mcp:1", template=<rendered>)}
```

## What bites people

- **Stateful vs stateless servers.** Some MCP servers keep per-session
  state (open file handles, DB transactions). `MCPClient` opens exactly
  one session per `async with` block; if you split calls across
  independent `MCPClient` contexts, that state does NOT carry over.
- **Tool-list caching.** `MCPClient.list_tools()` caches for 30s. Call
  `force_refresh=True` when a live server just registered a new tool
  and you want the model to see it in the current turn.
- **Cancellation is pre-dispatch, not mid-call.**
  `MCPClient.call_tool(..., ctx=ctx)` checks `ctx.check_cancelled()`
  BEFORE hitting the transport. To cancel a mid-flight call, close the
  `MCPClient` context — the transport tear-down will terminate the
  server's outbound stream. The session itself is not usable after that.
- **Every MCP tool defaults to `side_effecting=True`.** MCP has no
  standard "read-only" marker, so we assume the worst. Under
  `Autonomy.GATED` this means every MCP tool call surfaces a human
  approval interrupt — set `Autonomy.AUTO` if that's not what you want,
  or wrap the returned tools to flip the flag.
- **Argumented prompts are dropped in v1.** `mcp_prompts()` can only
  materialize static prompts. Use `MCPClient.get_prompt(name, args)`
  directly for prompts that require args.
- **A capability the server doesn't implement raises, it doesn't return
  empty.** `mcp_prompts()` and `MCPClient.list_resources()` on a
  tools-only server raise `mcp.shared.exceptions.McpError: Method not
  found`, from inside the `async with` block, which surfaces to you
  wrapped in an `ExceptionGroup` from the transport's task group. Ask
  for a surface only when you know the server has it, or guard the call.
- **`mcp_resources()` is lazy; `mcp_tools()` and `mcp_prompts()` are
  not.** The first returns a `MemorySource` and touches the server only
  on `query(...)`, so a missing resources capability surfaces later,
  from wherever the grounder runs. The other two hit the server
  immediately.
- **The `mcp` extra pulls in `pydantic`.** If your existing setup was
  pydantic-free by design, the extra will end that. It comes in via
  the upstream `mcp` package's wire schema.

## Related

- [Concepts · Integrations](../concepts/integrations.md) — what MCP is,
  the two transports, the caching behaviour, and the server side
  (`ApprovalServer`). This recipe is the task; that page is the model.
- [Concepts · Agents](../concepts/agents.md) — how `Tool` / `Memory`
  slot into an `Agent`.
- [Human-in-the-loop tool approval](hitl-tool-approval.md) — the
  agentkit-side gate that fires before an MCP tool runs when
  `Autonomy` allows it.
- The MCP spec: <https://modelcontextprotocol.io>
