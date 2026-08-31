# How do I give the Claude CLI my own tools?

The CLI owns its loop, so you cannot hand it a Python function. You hand
it a **server** — and `serve_registry` builds one from a `ToolRegistry`
you already have.

## When you'd want this

`ClaudeCliCognition` delegates the whole agent loop to the `claude`
binary, which is good at editing code and knows nothing about your
application. The moment the task needs something only your process can do
— close a ticket, look up a customer, record a build verdict — the CLI
has no way to call it.

Its seam for that is `--mcp-config`, which `ClaudeCliCognition` has
always accepted and nothing in agentkit could produce. Everything needed
to describe a tool was already correct: `FunctionTool` derives a
`ToolSchema` from your signature and docstring, `ToolRegistry` holds
them, `ToolArgumentError` refuses a bad call. Only the transport was
missing.

Reach for this when:

- the CLI needs a capability that lives in your process, not on disk;
- you want your tool middleware to apply to it, which it does, because
  the call comes back through agentkit code (see
  [the bypass warning](../concepts/claude-cli.md#the-thing-to-understand-first-the-chain-does-not-apply)
  for what happens with the CLI's *own* tools);
- you want `tools=("",)` — every tool the session has is one you wrote.

Needs the `mcp` extra:

```bash
pip install "arc-agentkit[mcp]"
```

## Working code

```python
"""Serve a ToolRegistry to an MCP client. No API key, no `claude` binary.

The client here is agentkit's own `MCPClient`, standing in for the CLI so
the whole thing runs offline. Pointing the real CLI at it is one line —
`ClaudeCliCognition(**spec.cli_kwargs())` — and is shown below.
"""

import asyncio

from agentkit.integrations.mcp import MCPClient, StreamableHttpServer, serve_registry
from agentkit.testing import make_test_ctx
from agentkit.tools import ToolRegistry, tool


@tool(side_effecting=True)
async def close_slice(slice_id: str, verdict: str) -> str:
    """Record a slice of work as finished with the given verdict.

    Only the deterministic runner may call this — it is the call the whole
    build gate rests on, so it is marked side-effecting and travels to the
    CLI with that flag intact.
    """
    return f"{slice_id}: {verdict}"


async def main() -> None:
    ctx = make_test_ctx()
    registry = ToolRegistry.from_tools([close_slice])

    spec = serve_registry(registry, name="engine", ctx=ctx)
    async with spec:
        print("tool names the CLI sees:", spec.tool_names)

        async with MCPClient(
            StreamableHttpServer(url=spec.url, headers=spec.auth_headers)
        ) as client:
            served = await client.list_tools()
            print("advertised:", [t.name for t in served])

            ok = await client.call_tool(
                "close_slice", {"slice_id": "s-1", "verdict": "pass"}
            )
            print("result:  ", ok.content[0].text)

            bad = await client.call_tool("close_slice", {"slice_id": "s-1"})
            print("bad call:", bad.content[0].text[:60], "...")
            print("session survived:", len(await client.list_tools()) == 1)


asyncio.run(main())
```

```text
tool names the CLI sees: ('mcp__engine__close_slice',)
advertised: ['close_slice']
result:   s-1: pass
bad call: tool 'close_slice' call rejected: missing required argument( ...
session survived: True
```

Those last two lines are the part worth keeping. A call the model got
wrong comes back as a **tool error it can read and retry**, not as a
transport failure that kills the session — because the model is the only
party that can fix a bad call, and a dead session gives it no chance to.

## Pointing the real CLI at it

```python
cognition = ClaudeCliCognition(
    **spec.cli_kwargs(),   # mcp_config + strict_mcp_config, credential included
    tools=("",),           # no built-in tools; everything comes from your server
)
```

`cli_kwargs()` exists so you never assemble the config document or its
`Authorization` header by hand. `strict_mcp_config=True` means the
session gets your server and nothing else — no user-level MCP config
leaking in from the developer's machine.

`tools=("",)` is the other half of that, and worth understanding: it
disables the CLI's built-in tools entirely, so every call in the session
comes back through your middleware chain. That is the strongest version
of this recipe and it costs you the CLI's own `Edit`, which is good. Most
applications want some of both.

## What travels with a tool, and why it matters

The schema is `ToolSchema.parameters` **byte-for-byte** — no translation
step, because a translation step is a second description of one thing and
will drift.

Three flags travel with it, and the reason is `RunPolicy`:

| declaration | why the CLI side needs it |
|---|---|
| `side_effecting` | the approval gate and idempotency both read it |
| `caps` | `RunPolicy`'s Rule-of-Two check |
| `requires_approval` | maps to the MCP annotation, so the tool still pauses |

Without those, moving your tools behind MCP would silently switch off the
capability check — at exactly the moment it matters most, because a
served registry is reachable by anything that can reach the port.

## What bites people

- **`timeout_s=None` is the default and it means forever.** A tool that
  never returns parks the CLI turn with no signal anywhere: the CLI waits
  on the MCP call, agentkit waits on the tool. A default deadline would
  be the worse mistake — it would kill a legitimately slow build or a
  human-in-the-loop approval and read as a flake — so the caller says
  which of those they have.
- **A tool with no `schema` is silently absent.** It is dropped rather
  than advertised with an invented empty `inputSchema`, so a tool you
  registered can simply not be there. `spec.tool_names` is the
  authoritative list of what the CLI will actually see — assert on it.
- **`transport="stdio"` rebuilds your tools in a fresh process.** The
  registry's closures and the live `ctx` do not survive that boundary, so
  any tool that closes over run state has to use the HTTP transport.
  `serve_registry` refuses rather than letting you find out at runtime.
- **The server is authenticated but still loopback.** The token narrows
  who can call; it does not narrow who can reach. Do not give `host=` a
  routable address.

## Related

- [The Claude CLI](../concepts/claude-cli.md) — all five seams, and the
  middleware bypass this one avoids.
- [Integrations (MCP)](../concepts/integrations.md) — the server in full,
  including the auth surface and both transports.
- [Human-in-the-loop tool approval](hitl-tool-approval.md) — the
  companion server, for permission prompts rather than tools.
- [Plug the claude CLI into FastAPI code-gen](claude-cli-fastapi-code-gen.md)
  — the same wiring inside a real worker.
