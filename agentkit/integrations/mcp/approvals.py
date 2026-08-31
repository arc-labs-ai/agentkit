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

    async with ApprovalServer(asker=my_asker, autonomy=ctx.autonomy) as approvals:
        cognition = ClaudeCliCognition(
            model="claude-sonnet-4-6",
            **approvals.cli_kwargs(),
        )
        result = await Agent(name="dev", cognition=cognition).run(task, ctx)

Because the ``Asker`` may await a person for as long as it likes, the CLI turn
parks in place — the same behaviour agentkit's own tool loop gets from the same
protocol.

``autonomy`` is the run's tier and it is honoured through the SAME
:func:`~agentkit.agents.control.gate.should_gate` every other pattern calls,
never a table of tiers restated here. ``HumanGate``'s claim is that autonomy is
set once per run and honoured uniformly, and this is the place a divergence
would be hardest to notice: the prompts are answered by a server the operator
wired once and then stopped watching, so a CLI path deciding for itself shows
up as "the reviewer was asked about things they thought were automatic" — or,
worse, the reverse — with nothing in the run pointing at why.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit.agents.control.elicitation import Decision, Elicitation
from agentkit.agents.control.gate import Autonomy, should_gate
from agentkit.kernel.protocols import AutonomyLiteral

if TYPE_CHECKING:  # pragma: no cover — typing only
    from agentkit.agents.control.elicitation import Asker

# The MCP server name. It appears in the tool's fully-qualified name
# (``mcp__<server>__<tool>``), which the caller passes to
# ``--permission-prompt-tool``, so it is part of the wire contract rather than
# a label.
SERVER_NAME = "agentkit_approvals"
TOOL_NAME = "approve"

# The tiers, spelled once. Only used to tell a REAL tier from a typo — the
# policy itself is ``should_gate``'s and is not restated here. ``Autonomy`` is a
# ``StrEnum``, so an ``Autonomy`` member compares equal to its literal and
# passes this check without a cast.
_TIERS: tuple[str, ...] = tuple(t.value for t in Autonomy)


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
    replacing them is an approve-with-changes.

    Serialisation is guarded because ``_decide``'s contract is that it never
    raises, and this is the only line in it that can. ``json.dumps`` raises on
    a value it cannot encode, and an exception escaping an MCP request handler
    reaches the CLI as a BROKEN PERMISSION SYSTEM rather than as a denial —
    the model is told the gate is malfunctioning and retries a call nobody
    approved. Every other failure in this module lands as a deny with a reason;
    an unencodable argument has to as well, or the one path that fails open is
    the one that was about to allow.
    """
    try:
        return json.dumps(
            {"behavior": "allow", "updatedInput": updated_input}, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        # ``allow_nan=False`` is deliberate: Python's json emits bare ``NaN`` /
        # ``Infinity``, which is not JSON, and the CLI's own parser rejects the
        # result — an allow that arrives as a parse error is the same broken
        # gate by a slower route.
        return _deny(
            f"the approval could not be encoded for the CLI "
            f"({type(exc).__name__}: {exc}), so the call was not allowed"
        )


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

    .. warning::
       ``auto_allow`` matches on the tool NAME and **ignores the arguments
       entirely**. Allow-listing ``Read`` allow-lists reading *anything the
       agent can reach*, not "safe reads". Measured against this server with
       ``auto_allow=("Read",)`` and a reviewer that denies everything::

           Read   /etc/passwd      -> allow      (reviewer never consulted)
           Read   /etc/shadow      -> allow      (reviewer never consulted)
           Read   ~/.ssh/id_rsa    -> allow      (reviewer never consulted)
           Write  /etc/passwd      -> deny       (reviewer consulted)

       That is habituation-avoidance working as designed, and it is also a
       larger grant than "allow-list the safe operations" sounds like. The
       reason the docstring says this so loudly is that an operator reading
       only the habituation rationale above could reasonably conclude the
       opposite. Put a tool on this list only when EVERY call it could make is
       one you would approve unread, or narrow it with ``auto_allow_when``.

    ``auto_allow_when(tool_name, arguments) -> bool`` is the opt-in
    argument-aware gate, and it can only SUBTRACT. A prompt is auto-allowed iff
    the tool is on ``auto_allow`` *and* the predicate says yes, so a predicate
    can never approve something the name list did not already approve — the
    default-deny path is untouched by definition, and a caller who wants the
    old behaviour simply leaves it ``None``. It is consulted only for tools
    already on the list, so the common case (no predicate) does not pay for it
    and does not have to think about it::

        ApprovalServer(
            asker=my_asker,
            auto_allow=("Read", "Glob"),
            auto_allow_when=lambda tool, args: str(
                args.get("file_path", args.get("path", ""))
            ).startswith("/workspace/"),
        )

    A predicate that RAISES falls through to the reviewer rather than
    auto-allowing. A broken narrowing rule must not silently widen the grant it
    exists to narrow, and ``_decide``'s never-raises contract means the
    exception cannot be allowed to escape either.

    ``autonomy`` is the run-wide tier from ``ctx.autonomy``, and it is honoured
    through :func:`~agentkit.agents.control.gate.should_gate` — the SAME
    function ``ReActCognition`` calls — rather than a table of tiers repeated
    here. ``HumanGate``'s whole claim is that autonomy is set once per run and
    honoured uniformly; a CLI path that decided independently would break that
    claim in the one place it is hardest to notice, because these prompts are
    answered by a server the operator wired once and then stopped watching.
    Under ``"auto"`` the ``Asker`` is not consulted at all — not asked and
    approved, not consulted. See :meth:`_tier` for the ``key_step`` mapping and
    for what an unrecognised tier does.

    Pass ``autonomy=ctx.autonomy`` to inherit the run's tier::

        async with ApprovalServer(asker=my_asker, autonomy=ctx.autonomy) as approvals:
            ...

    It defaults to ``"gated"`` because that is precisely what this server did
    before it knew about autonomy — ask about everything not auto-allowed. A
    default of ``"auto"`` would have turned every existing caller's approval
    gate into a rubber stamp on upgrade, which is the worst direction for a
    security default to move by accident.

    ``asker`` may be ``None`` only when nothing can reach a human — i.e. under
    ``autonomy="auto"``. Any other tier with no transport raises at
    CONSTRUCTION. Before that check you could build a server that cannot
    possibly answer and discover it only once a run was in flight and a person
    was waiting, where the symptom is a denied tool call in someone else's log
    rather than a stack trace in the wiring code that caused it.
    """

    asker: Asker | None = None
    timeout_s: float | None = None
    auto_allow: tuple[str, ...] = ()
    auto_allow_when: Callable[[str, dict[str, Any]], bool] | None = None
    # Read PER REQUEST, not snapshotted at construction. A supervisor
    # tightening the tier mid-run must affect the NEXT prompt; the alternative
    # is a field that reads as tightened while the server keeps behaving as it
    # did, which is the worst kind of security control. The cost is that a typo
    # can arrive after ``__post_init__`` has had its say, which is why
    # ``_tier`` re-checks and fails closed rather than trusting the constructor.
    autonomy: AutonomyLiteral = "gated"
    run_id: str = ""
    agent: str = ""
    host: str = "127.0.0.1"
    port: int = 0  # 0 = ask the OS for a free one at start()

    _server: Any = field(default=None, init=False, repr=False)
    _task: asyncio.Task[Any] | None = field(default=None, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Refuse, at construction, the configurations that cannot answer.

        Each of these is a server that looks wired and is not, and each fails
        only at the first prompt — by which point a run is in flight, a person
        may be waiting, and the evidence is a denied tool call rather than a
        traceback pointing at the wiring.

        - An unrecognised ``autonomy`` is the sharp one. ``should_gate`` falls
          through anything it does not recognise to its AUTO branch, so
          ``autonomy="ask"`` (a plausible typo, and what the original spec for
          this work actually wrote) would allow-list EVERYTHING, silently. A
          gate must not fail open on a spelling mistake.
        - ``timeout_s=0`` is the socket-API reflex for "no timeout" and means
          the opposite here: ``asyncio.wait_for`` with a non-positive timeout
          expires before the ``Asker`` is ever scheduled, so every prompt is
          denied without anyone being consulted — which is ``dontAsk``, the
          exact failure mode this module exists to remove. ``None`` is how you
          say "wait forever".
        - ``auto_allow="Read"`` — the missing comma — is the other fail-OPEN
          one, and it is quieter than the tier typo because the server keeps
          working. ``auto_allow`` is tested with ``in``, and ``in`` on a string
          is a substring test, so a bare string auto-allows every substring of
          itself: ``"R"``, ``"ea"``, ``"Read"``. Those calls are approved with
          the reviewer never consulted, which is the one outcome this class is
          built to make impossible by accident.
        - No ``asker`` under a gating tier has no answer to give. The condition
          is ``should_gate`` rather than "asker is required", so the one
          coherent transport-less configuration — AUTO, where nothing reaches a
          human — stays legal.
        """
        if self.autonomy not in _TIERS:
            raise ValueError(
                f"ApprovalServer: autonomy={self.autonomy!r} is not a tier — "
                f"use one of {', '.join(_TIERS)}. An unrecognised tier would "
                "gate nothing at all, which is the one failure a permission "
                "gate must not have"
            )
        if isinstance(self.auto_allow, str):
            raise ValueError(
                f"ApprovalServer: auto_allow={self.auto_allow!r} is a string, not a "
                "tuple of tool names. Membership in a string is a SUBSTRING test, so "
                "this would auto-allow every substring of it — approving those tool "
                "calls outright, without the reviewer ever being consulted. "
                f"Pass ({self.auto_allow!r},)"
            )
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(
                f"ApprovalServer: timeout_s={self.timeout_s!r} denies every prompt "
                "before the Asker is even scheduled — pass None to wait "
                "indefinitely, or a positive number of seconds"
            )
        if self.asker is None and should_gate(self.autonomy, key_step=True):
            raise ValueError(
                f"ApprovalServer: autonomy={self.autonomy!r} sends prompts to a human "
                "but no asker is wired, so every permission prompt would be denied. "
                "Pass asker=..., or autonomy='auto' if nothing should reach a person"
            )

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
        self._task = asyncio.create_task(self._serve(self._server))
        # Wait for the bind to actually happen: handing the CLI a URL that is
        # not listening yet turns into a startup timeout 30 seconds later.
        while not self._server.started:
            if self._task.done():  # the bind failed — surface THAT, not a hang
                await self._raise_start_failure()
            await asyncio.sleep(0.01)

    async def _serve(self, server: Any) -> None:
        """Run uvicorn with its ``sys.exit`` contained inside our own task.

        This wrapper is the load-bearing part of the bind-failure fix, not a
        tidy-up. uvicorn calls ``sys.exit(3)`` when it cannot bind, and asyncio
        treats ``SystemExit`` from a Task as a request to stop the LOOP: it
        re-raises into the event loop, which cancels the caller. Measured
        against an occupied port, ``start()`` never reached its own
        ``_task.done()`` branch — it was cancelled at the ``sleep`` first, so
        the caller saw a bare ``CancelledError``, ``except Exception`` caught
        nothing, and the process unwound.

        Converting it to a ``RuntimeError`` *inside* the task keeps it an
        ordinary task failure, which is what the wait loop is written to read.
        """
        try:
            await server.serve()
        except SystemExit as exc:  # uvicorn's failed-bind signal
            raise RuntimeError(f"uvicorn exited with status {exc.code}") from exc

    async def _raise_start_failure(self) -> None:
        """Turn a ``serve()`` that ended before it listened into an error a
        caller can actually catch, and drop the half-built state.

        uvicorn calls ``sys.exit(3)`` when the bind fails, so awaiting the task
        propagates ``SystemExit`` — a ``BaseException``. Measured against a
        port already in use: ``except Exception`` around
        ``async with ApprovalServer(...)`` caught NOTHING, and the wiring the
        module docstring recommends unwound the process instead of failing the
        one run. In the documented FastAPI recipe that is the whole worker.

        Clearing ``_task``/``_server`` is the other half. They used to be left
        in place, so a caller who retried hit ``start``'s idempotence guard and
        got a silent no-op: a server object reporting no error and not
        listening, whose URL then went to the CLI, which failed thirty seconds
        later with a startup timeout — the exact failure this wait loop exists
        to prevent, reached by the path that was supposed to report it.
        """
        task, where = self._task, f"{self.host}:{self.port}"
        self._task = None
        self._server = None
        try:
            if task is not None:
                await task
        except (Exception, SystemExit, asyncio.CancelledError) as exc:
            raise RuntimeError(
                f"ApprovalServer could not listen on {where} "
                f"({type(exc).__name__}: {exc}) — the port may already be in use"
            ) from exc
        raise RuntimeError(
            f"ApprovalServer stopped serving {where} before it began listening"
        )

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
        """Shut the server down and wait for the port to be released.

        Stopping a server that never started used to raise ``SystemExit(3)``
        out of ``__aexit__`` — uvicorn's failed-bind signal, a
        ``BaseException``, so this ``suppress`` did not catch it — during the
        unwinding of whatever had already gone wrong, replacing the real error
        with an exit code. It is not listed here because :meth:`_serve` now
        converts it inside the task, which is the only place it can arise;
        adding it here as well would be a branch no test could reach.
        """
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
        """One permission prompt → at most one ``Asker`` round trip → allow or
        deny. "At most" because the autonomy tier decides whether a human is in
        this loop at all, before anything is asked.

        Never raises. This runs inside an MCP request handler, and an exception
        here reaches the CLI as a malformed result — which it reports as a
        broken permission system rather than a denial, leaving the model to
        retry a call nobody approved. A failure is a DENY with the reason
        attached, so the model sees it and can adapt. That contract is why the
        answer is RENDERED inside a try as well: an ``Asker`` returning
        something that is not a ``Decision`` used to escape here as an
        ``AttributeError``.
        """
        self._seen += 1
        # Snapshot the count for THIS prompt. The CLI can have several tool
        # calls in flight, so several prompts run concurrently on one server,
        # and the id a UI keys its pending cards on must be the number at the
        # time of this prompt rather than whatever the counter has reached by
        # the time the ``Elicitation`` is built. Nothing awaits in between
        # today; a local means that stays true when something does.
        seen = self._seen

        tier = self._tier()
        if not should_gate(tier, requires_approval=False, key_step=True):
            # AUTO: the Asker is not consulted at all. "Asked, and they
            # approved" would produce the identical allow and would still be
            # wrong — the tier's whole meaning is that no person is in this
            # loop, which is why the test for it asserts on the call that did
            # not happen rather than on the answer.
            return _allow(arguments)

        # ``auto_allow`` is inert under MANUAL, and this is a security call
        # rather than an ordering accident. The two knobs are set in different
        # places at different times: ``auto_allow`` at wiring time by whoever
        # built the service, ``autonomy`` per run by whoever launched THIS run.
        # If a static name list could carve exceptions out of MANUAL, then
        # "manual" would silently mean "manual except the forty Reads", and the
        # operator who chose the strictest tier would have nothing in the run to
        # tell them otherwise. The run-level, later, more specific choice wins,
        # and it can only move in the safe direction: more prompts, never fewer.
        # (Under AUTO the list is moot — the branch above already allowed.)
        if (
            tier != Autonomy.MANUAL
            and tool_name in self.auto_allow
            and self._arguments_are_auto_allowed(tool_name, arguments)
        ):
            return _allow(arguments)

        asker = self.asker
        if asker is None:
            # ``__post_init__`` rejects this combination, so getting here means
            # ``autonomy`` was tightened after construction. ``_decide`` must
            # never raise, so it is a deny naming the missing transport: the
            # operator reads the cause in the model's own refusal instead of an
            # AttributeError inside an MCP request handler.
            return _deny(
                f"no reviewer is wired for this run (autonomy={self.autonomy!r}), "
                f"so the {tool_name} call cannot be approved"
            )

        request = Elicitation(
            id=f"cli-approval-{seen}",
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
            decision = await self._ask(asker, request)
        except Exception as exc:  # noqa: BLE001 — see the docstring
            return _deny(f"the approval transport failed ({type(exc).__name__}: {exc})")

        try:
            return self._render(decision, tool_name, arguments)
        except Exception as exc:  # noqa: BLE001 — see the docstring
            # A malformed answer, kept separate from a transport failure
            # because they need different fixes. The HITL API this replaced was
            # ``dict[str, str]`` with ``"approve"`` as a bare string, so an
            # ``Asker`` written from that memory returns a ``str`` and
            # ``decision.kind`` raises. Left to escape, that reaches the CLI as
            # a broken permission system rather than a denial — the model is
            # told the gate is broken and retries a call nobody approved. The
            # type is named so the transport's author can see what they sent.
            return _deny(
                f"the reviewer's answer was not a Decision "
                f"({type(decision).__name__}: {type(exc).__name__}: {exc})"
            )

    def _tier(self) -> Autonomy:
        """The run's autonomy tier, normalised, with an unrecognised one
        treated as MANUAL.

        Two things meet here. ``autonomy`` is read per request from a mutable
        field, so ``__post_init__``'s check is not the last word — a typo can
        be assigned afterwards. And ``should_gate`` falls through anything it
        does not recognise to its AUTO branch, which for a permission gate is
        the one unacceptable answer: a misspelt tier would allow every call.
        Mapping the unknown onto the STRICTEST tier makes the residual failure
        "the reviewer got asked about everything", which someone notices.

        This is not a second copy of the policy — every recognised tier is
        handed straight to ``should_gate``, and MANUAL here is a value, not a
        rule about what MANUAL does.
        """
        if self.autonomy not in _TIERS:
            return Autonomy.MANUAL
        return Autonomy(self.autonomy)

    def _arguments_are_auto_allowed(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> bool:
        """The opt-in argument check, evaluated only for tools already on
        ``auto_allow``. ``True`` when there is no predicate, so the no-predicate
        case is exactly the name-only behaviour it always was.

        Ordering matters and is not incidental: ``tool_name in self.auto_allow``
        is checked FIRST by the caller, so the predicate is a narrowing filter
        over an already-approved set rather than a second, independent way in.
        A predicate that returns ``True`` for a tool nobody allow-listed changes
        nothing.

        A raise is a ``False``, not a crash and not an allow. Three constraints
        meet here: ``_decide`` must never raise (an exception reaches the CLI as
        a broken permission system, which it reports as neither an allow nor a
        deny, leaving the model to retry a call nobody approved); a gate must
        fail closed; and "closed" for a NARROWING predicate means falling
        through to the reviewer, not denying outright — the operator did say
        this tool was routine, so the human is the right place to land, and they
        still see the arguments on the ``Elicitation``.
        """
        if self.auto_allow_when is None:
            return True
        try:
            return bool(self.auto_allow_when(tool_name, arguments))
        except Exception:  # noqa: BLE001 — see the docstring: a broken narrowing
            return False   # rule must never widen the grant it exists to narrow

    async def _ask(self, asker: Asker, request: Elicitation) -> Decision:
        """Await the human, bounded by ``timeout_s``.

        The timeout is enforced HERE rather than trusted to the ``Asker``: the
        protocol allows an implementation to wait forever, and a queue worker
        holding a CLI subprocess open indefinitely is a resource leak with a
        model attached.

        The ``Asker`` is a parameter rather than read off ``self`` so the
        None-check in ``_decide`` narrows it once, at the point where "there is
        no reviewer" is turned into a denial the model can read.

        The non-positive re-check is the same belt-and-braces ``_tier`` applies
        to ``autonomy``, and for the same reason: ``timeout_s`` is a mutable
        public field, so ``__post_init__``'s refusal is not the last word. A
        server whose ``timeout_s`` is set to ``0`` afterwards denies every
        prompt with the ``Asker`` never scheduled — ``dontAsk``, the failure
        mode this module exists to remove — and it did so under the message
        "no answer within 0s", which blames a reviewer who was never asked and
        sends the operator to debug the transport. It still denies, because
        denying is the closed direction; it now says what actually happened.
        """
        timeout = self.timeout_s
        if timeout is None:
            return await asker.ask(request)
        if timeout <= 0:
            return Decision(
                kind="expired",
                note=(
                    f"the approval gate is misconfigured: timeout_s={timeout!r} expires "
                    "before a reviewer can be consulted, so nobody was asked "
                    "(pass None to wait indefinitely, or a positive number of seconds)"
                ),
            )
        try:
            return await asyncio.wait_for(asker.ask(request), timeout=timeout)
        except TimeoutError:
            return Decision(kind="expired", note=f"no answer within {timeout}s")

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

        What counts as an approval is :attr:`Decision.approved` — the SAME
        predicate ``ReActCognition`` gates on — and not a list of kinds
        restated here, for exactly the reason ``autonomy`` is routed through
        ``should_gate``: one run, one ``Asker``, one meaning of yes. This used
        to read ``kind in ("approve", "value")``, which let a ``value``
        decision allow the call while ``Decision.approved`` (and therefore the
        agentkit-native gate, given the identical transport) counted it as a
        refusal. ``value`` is not an answer to an ``approval`` request at all —
        the request goes out with ``choices=("approve", "deny")`` — so an
        ``Asker`` that returns ``Decision(kind="value", value="no")`` was
        answering "no" and being read as "yes".
        """
        if not decision.approved:
            if decision.kind == "expired":
                return _deny(
                    decision.note or f"no approval for {tool_name} arrived before the deadline"
                )
            if decision.kind == "value":
                # Named rather than folded into "a reviewer declined", because
                # the fix is in the transport, not in the run: this request
                # went out as kind="approval" with choices=("approve","deny"),
                # so a ``value`` answer means the Asker handed back whatever
                # the person typed instead of a verdict. The author of that
                # Asker is the only one who can tell yes from no here, and
                # they cannot if the message says the reviewer said no.
                return _deny(
                    decision.note
                    or f"the reviewer's transport answered the {tool_name} approval with "
                    "a value rather than approve/deny, which is not consent"
                )
            return _deny(decision.note or f"a reviewer declined the {tool_name} call")
        if decision.kind == "modify":
            if not isinstance(decision.value, dict):
                return _deny(
                    decision.note
                    or f"a reviewer edited the {tool_name} call but the replacement "
                    f"arguments were {type(decision.value).__name__}, not a dict"
                )
            return _allow(decision.value)
        return _allow(arguments)


__all__ = ["SERVER_NAME", "TOOL_NAME", "ApprovalServer"]
