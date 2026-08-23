"""``ApprovalServer`` — route the Claude CLI's permission prompts to an ``Asker``.

``ClaudeCliCognition`` delegates the whole loop to the CLI, which owns its own
permissions. That left a service two options, both bad: ``bypassPermissions``
(the agent may do anything, unattended) or ``dontAsk`` (anything not
pre-approved is denied outright, and the run just fails). agentkit already has
the missing middle — :class:`~agentkit.agents.control.elicitation.Asker`, the
injected human transport behind its own HITL path — but nothing connected the
two.

The CLI's seam for this is ``--permission-prompt-tool``: name an MCP tool and
the CLI calls it instead of prompting a terminal, sending the tool name and
arguments and expecting an allow/deny back. So this module *is* an MCP server —
agentkit as the callee, for once — that turns each prompt into an
:class:`~agentkit.agents.control.elicitation.Elicitation`, awaits the
application's ``Asker``, and maps the answer back:

    async with ApprovalServer(asker=my_asker) as approvals:
        cognition = ClaudeCliCognition(
            model="claude-sonnet-4-6",
            **approvals.cli_kwargs(),
        )
        result = await Agent(name="dev", cognition=cognition).run(task, ctx)

Because the ``Asker`` may await a person for as long as it likes, the CLI turn
parks in place — the same behaviour agentkit's own tool loop gets from the same
protocol.

Requires the ``mcp`` extra (``pip install "arc-agentkit[mcp]"``); ``uvicorn``
and ``starlette`` arrive with it, so this adds no new dependency.

.. warning::
   The server binds ``127.0.0.1`` on an ephemeral port with no authentication:
   anything able to reach that port can answer permission prompts on this
   agent's behalf. Loopback-only is the containment. Do not bind it to a
   routable address.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.agents.control.elicitation import Decision, Elicitation

if TYPE_CHECKING:  # pragma: no cover — typing only
    from agentkit.agents.control.elicitation import Asker

# The MCP server name. It appears in the tool's fully-qualified name
# (``mcp__<server>__<tool>``), which the caller passes to
# ``--permission-prompt-tool``, so it is part of the wire contract rather than
# a label.
SERVER_NAME = "agentkit_approvals"
TOOL_NAME = "approve"


def _free_port() -> int:
    """An ephemeral loopback port the OS says is free.

    There is an unavoidable race between closing this socket and the server
    binding it. Asking the OS beats picking a fixed port, which collides the
    moment two agents run on one host.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _allow(updated_input: dict[str, Any]) -> str:
    """The CLI's allow shape. ``updatedInput`` is what the tool actually runs
    with, so passing the arguments through unchanged is an approval and
    replacing them is an approve-with-changes."""
    return json.dumps({"behavior": "allow", "updatedInput": updated_input})


def _deny(message: str) -> str:
    """The CLI's deny shape. ``message`` reaches the MODEL, which may adapt —
    so a useful reason ("that path is out of scope") is worth more than "no"."""
    return json.dumps({"behavior": "deny", "message": message})


@dataclass
class ApprovalServer:
    """An in-process MCP server that answers the CLI's permission prompts.

    ``asker`` is the application's human transport. ``timeout_s`` bounds how
    long a single prompt waits before it is denied — ``None`` waits forever,
    which is right for an interactive UI and wrong for a queue worker.

    ``auto_allow`` names tools to approve without asking. It exists because the
    CLI prompts for reads too, and a person clicking "yes" on forty ``Read``
    calls is not oversight, it is habituation — the thing that makes the
    fortieth prompt, the one that mattered, get the same reflexive yes.
    """

    asker: Asker
    timeout_s: float | None = None
    auto_allow: tuple[str, ...] = ()
    run_id: str = ""
    agent: str = ""
    host: str = "127.0.0.1"
    port: int = 0  # 0 = ask the OS for a free one at start()

    _server: Any = field(default=None, init=False, repr=False)
    _task: asyncio.Task[Any] | None = field(default=None, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)

    # ---- lifecycle -------------------------------------------------------------------------

    async def __aenter__(self) -> ApprovalServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Bind the port and serve. Idempotent."""
        if self._task is not None:
            return
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover — exercised by the extra-less env
            raise ImportError(
                "ApprovalServer needs the 'mcp' extra: pip install 'arc-agentkit[mcp]'"
            ) from exc

        if not self.port:
            self.port = _free_port()

        config = uvicorn.Config(
            self.build_mcp().streamable_http_app(),
            host=self.host,
            port=self.port,
            log_level="error",  # the app's own logs are the interesting ones
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        # Wait for the bind to actually happen: handing the CLI a URL that is
        # not listening yet turns into a startup timeout 30 seconds later.
        while not self._server.started:
            if self._task.done():  # the bind failed — surface THAT, not a hang
                await self._task
            await asyncio.sleep(0.01)

    def build_mcp(self) -> Any:
        """The ``FastMCP`` app this server serves. Public so the wire format
        can be pinned without opening a socket.

        ``stateless_http`` because each permission prompt is a complete
        question: there is no session state worth resuming, and a stateful
        server would keep one per CLI process for no benefit.
        """
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP(SERVER_NAME, stateless_http=True)

        # ``structured_output=False`` is load-bearing, not tidiness. The CLI
        # rejects anything but ONE text block: "Permission prompt tool returned
        # an invalid result. Expected a single text block param with
        # type='text' and a string text value." FastMCP's default adds a
        # ``structuredContent`` field alongside the text, and every prompt then
        # fails with exactly that, while the tool itself looks like it ran —
        # measured against the real binary before this line existed.
        @mcp.tool(name=TOOL_NAME, structured_output=False)
        async def approve(tool_name: str, input: dict[str, Any]) -> str:  # noqa: A002
            """Decide whether the agent may run this tool call."""
            return await self._decide(tool_name, input)

        return mcp

    async def stop(self) -> None:
        """Shut the server down and wait for the port to be released."""
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, timeout=5.0)
            self._task = None
        self._server = None

    # ---- wiring ----------------------------------------------------------------------------

    @property
    def url(self) -> str:
        """The MCP endpoint. ``/mcp`` is FastMCP's streamable-http mount."""
        return f"http://{self.host}:{self.port}/mcp"

    @property
    def mcp_config(self) -> str:
        """The ``--mcp-config`` value, inline rather than a temp file so there
        is no path to clean up and no window where a stale file points at a
        dead port."""
        return json.dumps({"mcpServers": {SERVER_NAME: {"type": "http", "url": self.url}}})

    @property
    def tool_name(self) -> str:
        """The fully-qualified MCP tool name for ``--permission-prompt-tool``."""
        return f"mcp__{SERVER_NAME}__{TOOL_NAME}"

    @property
    def prompts_seen(self) -> int:
        """How many permission prompts this server has answered. A run that
        reports zero either never needed permission or never reached the
        server — worth being able to tell apart."""
        return self._seen

    def cli_kwargs(self) -> dict[str, Any]:
        """The ``ClaudeCliCognition`` fields that wire this server in.

        ``strict_mcp_config=True`` is included deliberately: without it the CLI
        also loads whatever MCP servers the working directory or the user's
        home configuration happen to define, which is not what a service
        wiring an approval gate is asking for.
        """
        return {
            "mcp_config": (self.mcp_config,),
            "strict_mcp_config": True,
            "permission_prompt_tool": self.tool_name,
        }

    # ---- the decision ----------------------------------------------------------------------

    async def _decide(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """One permission prompt → one ``Asker`` round trip → allow or deny.

        Never raises. This runs inside an MCP request handler, and an exception
        here reaches the CLI as a malformed result — which it reports as a
        broken permission system rather than a denial, leaving the model to
        retry a call nobody approved. A failure is a DENY with the reason
        attached, so the model sees it and can adapt.
        """
        self._seen += 1
        if tool_name in self.auto_allow:
            return _allow(arguments)

        request = Elicitation(
            id=f"cli-approval-{self._seen}",
            prompt=f"Claude Code wants to run {tool_name}.",
            kind="approval",
            choices=("approve", "deny"),
            # The arguments ARE the thing being approved — a path, a shell
            # command — so they travel as the tool_call rather than being
            # flattened into the prompt, where a UI could not render them.
            tool_call={"name": tool_name, "arguments": arguments},
            deadline_s=self.timeout_s,
            run_id=self.run_id,
            agent=self.agent,
        )

        try:
            decision = await self._ask(request)
        except Exception as exc:  # noqa: BLE001 — see the docstring
            return _deny(f"the approval transport failed ({type(exc).__name__}: {exc})")

        return self._render(decision, tool_name, arguments)

    async def _ask(self, request: Elicitation) -> Decision:
        """Await the human, bounded by ``timeout_s``.

        The timeout is enforced HERE rather than trusted to the ``Asker``: the
        protocol allows an implementation to wait forever, and a queue worker
        holding a CLI subprocess open indefinitely is a resource leak with a
        model attached.
        """
        if self.timeout_s is None:
            return await self.asker.ask(request)
        try:
            return await asyncio.wait_for(self.asker.ask(request), timeout=self.timeout_s)
        except TimeoutError:
            return Decision(kind="expired", note=f"no answer within {self.timeout_s}s")

    def _render(
        self, decision: Decision, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Map a :class:`Decision` onto the CLI's allow/deny shape.

        ``modify`` becomes an approve-with-changes, which is the CLI's own
        ``updatedInput`` semantics: the tool runs with the arguments the person
        edited, and the model is not told they changed. That is how a reviewer
        redirects a write to a sandbox path without derailing the run.

        Everything that is not an approval denies, including ``expired``. The
        default has to be deny — an approval gate that fails open is not a
        gate — and the message says WHICH of them it was, since "the reviewer
        said no" and "nobody answered in 60s" call for different fixes.
        """
        if decision.kind == "modify" and isinstance(decision.value, dict):
            return _allow(decision.value)
        if decision.kind in ("approve", "value"):
            return _allow(arguments)
        if decision.kind == "expired":
            return _deny(
                decision.note or f"no approval for {tool_name} arrived before the deadline"
            )
        return _deny(decision.note or f"a reviewer declined the {tool_name} call")


__all__ = ["SERVER_NAME", "TOOL_NAME", "ApprovalServer"]
