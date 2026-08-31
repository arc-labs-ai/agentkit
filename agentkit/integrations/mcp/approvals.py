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

Every prompt is RECORDED, as a decision rather than as an increment. A
permission prompt is the point at which a person took responsibility for
something the machine would not do on its own, and a count of them is the one
summary that answers no question worth asking: not what was allowed, not
whether anybody actually looked, not which approval preceded the write you are
staring at. :attr:`ApprovalServer.decisions` hands back an
:class:`ApprovalDecision` per prompt, oldest first, each carrying the call, the
arguments, the verdict, the reason the model was given, WHAT decided
(:data:`ApprovalSource`) and whether a human was reached at all — the same
shape, ordering and ``gate.check`` observation ``HookSettings.decisions`` uses
next door, so a service reading both reads one trail.

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

The server binds ``127.0.0.1`` on an ephemeral port and, by default, requires a
generated bearer token — because "anything able to reach that port can answer
permission prompts on this agent's behalf" was the whole of the old containment
argument, and the CLI whose prompts this answers is the thing running ``Bash``
in that same namespace. A verdict reachable by whatever a build script felt like
doing is a verdict no longer produced only by the reviewer who was supposed to
produce it; it does not have to be attacked to be worthless, it only has to be
reachable. ``cli_kwargs()`` carries the credential, so nothing about the wiring
above changes. ``auth="none"`` restores the old unauthenticated listener for a
host nobody else shares, and it is now something a caller says out loud rather
than what they get by saying nothing. Do not bind a routable address either way.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from agentkit.agents.control.elicitation import Decision, Elicitation
from agentkit.agents.control.gate import Autonomy, should_gate
from agentkit.integrations.mcp._transport import (
    LoopbackMcpTransport,
    McpAuth,
    http_mcp_config,
    qualified_tool_name,
    validated_auth,
)
from agentkit.kernel._frozen import deep_freeze
from agentkit.kernel.protocols import AutonomyLiteral

if TYPE_CHECKING:  # pragma: no cover — typing only
    from agentkit.agents.control.elicitation import Asker
    from agentkit.kernel.ports import ClockPort

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

# WHAT decided a prompt, as a closed set. This is the field the record exists
# for. "Allowed because a person said so" and "allowed because the run's
# autonomy tier does not gate this call" are the two facts an auditor most
# needs kept apart, and a boolean holds neither: both are ``allowed=True``. The
# same collapse in the other direction is worse — a ``timeout`` that degraded
# to a deny reads as a human refusal, so somebody goes looking for the reviewer
# who said no; and an ``error`` reads as a policy decision, so nobody goes
# looking for the broken transport that caused it.
#
# ``error`` covers every way the gate could not function: a raising ``Asker``,
# an answer that was not a ``Decision``, a ``Decision`` of a kind that is not
# an answer to an approval request (``value`` — the transport handed back what
# the person typed rather than a verdict), a tier tightened past a missing
# transport, arguments the CLI's encoder refused. They share the property that
# NOBODY DECIDED — which is what separates them from the other four.
ApprovalSource = Literal["asker", "auto_allow", "autonomy", "timeout", "error"]

# The bound on ONE recorded argument value, in characters of its JSON
# rendering. ``Write`` sends the whole file body and ``Edit`` the whole
# replacement string, so a server holding every prompt for the life of a run
# would otherwise grow with the bytes the agent wrote. The value is truncated,
# never dropped: a key that vanished would read as a call that never carried
# it, which is a worse record than a long one.
_MAX_ARGUMENT_CHARS = 4096

# The bound on ONE ``ctx.emit`` call, in seconds. Recording sits on the
# critical path of the CLI's permission answer — ``_decide`` cannot return
# until ``_record`` has, and ``_record`` awaits the observer — so an observer
# that never returns parks the prompt forever and wedges the run, which is
# ``dontAsk`` by a slower route. Suppressing exceptions does not cover this:
# a coroutine that hangs raises nothing. The reasoning is ``_ask``'s, applied
# to the other injected callable this class awaits — the protocol lets an
# implementation wait forever, so the bound is enforced HERE rather than
# trusted to it. Generous, because a slow log shipper should not truncate a
# real event; short enough that a dead one cannot hold a prompt open.
_EMIT_TIMEOUT_S = 5.0


def _render_value(value: Any) -> str:
    """A string for any argument value, for measurement and for truncation.

    ``default=repr`` because an argument that is not JSON-serialisable still
    has to be recorded — this sits under a never-raises contract, and an audit
    trail that refuses the awkward entries is not an audit trail. The second
    fallback exists because ``repr`` itself is user code and can raise.
    """
    try:
        return json.dumps(value, default=repr)
    except Exception:  # noqa: BLE001 — see the docstring
        try:
            return repr(value)
        except Exception:  # noqa: BLE001 — a __repr__ that raises
            return f"<{type(value).__name__}>"


def _bounded(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """``arguments`` with each value bounded by :data:`_MAX_ARGUMENT_CHARS`.

    The arguments travel because a tool name is not a decision. "We approved
    ``Write``" answers nothing; "we approved ``Write`` of ``/etc/passwd``" is
    the record. A value small enough to keep is kept AS IT ARRIVED rather than
    re-rendered, so ``decision.arguments["file_path"] == "/tmp/out"`` is a
    string comparison and not a JSON one.

    Never raises, for the same reason ``_render_value`` does not: this runs
    after the CLI's answer has already been decided, and losing the record of a
    refusal that happened is the one outcome worth avoiding here.
    """
    out: dict[str, Any] = {}
    try:
        items = list(arguments.items())
    except Exception:  # noqa: BLE001 — a Mapping whose iteration is user code
        return out
    for key, value in items:
        rendered = _render_value(value)
        if len(rendered) <= _MAX_ARGUMENT_CHARS:
            out[str(key)] = value
            continue
        out[str(key)] = (
            f"{rendered[:_MAX_ARGUMENT_CHARS]}… "
            f"<truncated, {len(rendered)} chars of {type(value).__name__}>"
        )
    return out


def _read_back(answer: str) -> tuple[bool, str, Mapping[str, Any] | None]:
    """What the CLI was actually told, parsed back off the wire.

    ``allowed`` / ``reason`` / the executed arguments are read from the answer
    rather than tracked in parallel with it, so the record and the refusal
    cannot disagree: ``reason`` IS the text the model receives, by
    construction. It also catches the one path where the branch that decided
    and the answer that shipped differ — ``_allow`` degrades to a deny when the
    arguments cannot be encoded — which a parallel variable would have filed as
    an approval that never happened.

    ``updatedInput`` is returned rather than the request's own arguments
    because that is what the tool RUNS with. On a ``modify`` the CLI executes
    the reviewer's edit and does not tell the model it changed, so this record
    is the only place in the run those arguments exist at all.
    """
    try:
        body = json.loads(answer)
        if not isinstance(body, dict):  # pragma: no cover — both helpers emit objects
            return False, answer, None
        allowed = body.get("behavior") == "allow"
        updated = body.get("updatedInput")
        message = body.get("message")
        return (
            allowed,
            "" if allowed else str(message or ""),
            updated if isinstance(updated, dict) else None,
        )
    except Exception:  # noqa: BLE001 — _record must not raise; see its docstring
        return False, answer, None


@dataclass(frozen=True)
class ApprovalDecision:
    """One permission prompt, recorded as the decision it was.

    This replaced a counter. A count of permission prompts is the one summary
    that cannot answer any question worth asking about them: not *what did we
    allow*, not *did anybody actually look*, not *which approval preceded the
    write we are now staring at*. A permission prompt is the point at which a
    person took responsibility for something the machine would not do on its
    own, and that is a decision with provenance, not an increment.

    Shaped to match ``HookSettings.decisions`` next door — same ``tool`` /
    ``allowed`` / ``reason`` core, same oldest-first ordering, same
    ``gate.check`` observation — so a service consuming both the CLI's hook
    guard and its permission gate is consuming one shape, not two.

    ``source`` is the field this type exists for; see :data:`ApprovalSource`.
    ``asked`` is deliberately NOT ``source == "asker"``: a prompt can reach a
    person and then time out, and whether somebody was interrupted is a fact
    about what the run cost a human rather than about the verdict. Neither is
    recoverable from the other.

    ``at`` is ISO-8601 UTC from the run's clock (see ``ApprovalServer.clock``),
    so a replayed or fake-clocked run's approval trail lines up with the
    checkpoints beside it. An empty string means the clock could not be read.

    **Nothing here is redacted and nothing here is logged.** ``arguments`` is
    recorded as it arrived, so a ``Bash`` command or a ``Write`` body carrying
    a token is in this record. That is deliberate: what to do about a secret in
    an approved tool call is the caller's policy — they hold the trail and they
    know their retention rules — and a library that half-redacted would hand
    them the worst of both, an audit trail nobody can trust to be complete that
    still leaks whatever the pattern missed. The only bound applied is size
    (:func:`_bounded`), which truncates and never drops.

    Frozen, and ``arguments`` is deep-frozen, for the reason ``HookDecision``
    is frozen: a decision that can be edited after the fact is not an audit
    trail. ``__hash__`` is over the identity subset for the reason
    ``ToolCall.__hash__`` is — a frozen mapping is still a mapping, so still
    unhashable, and the dataclass-derived hash would raise on every record.
    """

    tool: str
    arguments: Mapping[str, Any]
    allowed: bool
    reason: str
    source: ApprovalSource
    at: str
    asked: bool

    def __post_init__(self) -> None:
        # Frozen dataclass — assign through object.__setattr__ to bypass the
        # freeze. Without this, ``frozen`` stops at the field reference and
        # ``record.arguments["file_path"] = "/etc/passwd"`` rewrites the audit
        # trail in place, which is the exact hazard ``_frozen`` exists for.
        object.__setattr__(self, "arguments", deep_freeze(dict(self.arguments)))

    def __hash__(self) -> int:
        """Hash on the identity subset, never on ``arguments``.

        ``arguments`` is a ``FrozenDict`` — immutable, and still a ``dict``
        subclass, so still unhashable. The dataclass-derived ``__hash__`` reads
        every compared field and would therefore raise ``TypeError`` on every
        record. The hash invariant only requires EQUAL objects to hash equally,
        so two decisions differing only in their arguments share a bucket and
        ``__eq__`` (untouched, and still over every field) separates them
        there.
        """
        return hash((self.tool, self.allowed, self.source, self.at, self.asked))


@dataclass
class _Pending:
    """The half of a record that only the branch that decided can know.

    ``_decide``'s outcome is a JSON string, and everything an auditor needs
    about *what happened* is readable from it — except which branch produced
    it. So the branch writes that here on its way past, and the string carries
    the rest. Mutable and per-prompt: the CLI runs several prompts at once on
    one server, so this must never be state on ``self``.
    """

    source: ApprovalSource = "asker"
    asked: bool = False
    # Did the branch INTEND to allow? Only ``_allow`` can turn that into a
    # deny — it degrades when the arguments cannot be encoded — and a denial
    # nobody chose is an ``error``, not the tier or the allow-list deciding.
    allowing: bool = False


@dataclass
class _ReachedAsker:
    """Wraps the ``Asker`` so the record knows whether it was actually invoked.

    ``asked`` cannot be derived from the verdict (a prompt that reached a
    person can still expire) and it cannot be derived from ``source`` either.
    The one path through :meth:`ApprovalServer._ask` that never schedules the
    transport is the non-positive ``timeout_s``, and re-deriving that here
    would be a second copy of ``_ask``'s branch — the drift this module already
    routes ``autonomy`` through ``should_gate`` to avoid. So the fact is
    recorded by the only thing that knows it: the call itself.
    """

    asker: Asker
    reached: bool = False

    async def ask(self, request: Elicitation) -> Decision:
        self.reached = True
        return await self.asker.ask(request)


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
    # The run's context, for ``emit`` only — read through ``getattr`` so a
    # ``NullCtx``, a structural stub predating this field, or nothing at all is
    # simply "no observer wired" rather than a crash inside a permission
    # decision. Same access shape ``resolve_asker`` and ``HookSettings._emit``
    # use, for the same reason.
    ctx: Any | None = None
    # The run's clock, the same ``ClockPort`` seam ``Checkpointer`` takes.
    # ``Ctx`` does not carry one, so it is injected here rather than reached
    # off ``ctx``; see ``_now`` for why ``datetime.now()`` is not an option.
    clock: ClockPort | None = None
    host: str = "127.0.0.1"
    port: int = 0  # 0 = ask the OS for a free one at start()
    # Read once, at ``_listener()``: unlike ``autonomy`` this is not a knob a
    # supervisor turns mid-run. The listener is already bound and the CLI is
    # already holding the config document, so a value changed after ``start()``
    # could only produce a server whose fence disagrees with the credential its
    # own config hands out.
    auth: McpAuth = "bearer"

    # The listener is shared with ``serve_registry`` — see ``_transport``. Both
    # servers had the same bind-and-wait loop and only one of them would have
    # been fixed the next time it was wrong.
    _transport: LoopbackMcpTransport | None = field(default=None, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)
    _decisions: list[ApprovalDecision] = field(default_factory=list, init=False, repr=False)

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
        # Same fail-closed reflex as the tier check above, one layer out: an
        # unrecognised auth mode must not fall through to "no fence". Checked
        # HERE rather than at ``start()`` because a wiring mistake that only
        # surfaces once the listener is up is a wiring mistake discovered by
        # whoever is reading the transcript, not by whoever wrote the line.
        validated_auth(self.auth, where="ApprovalServer")
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

    def _listener(self) -> LoopbackMcpTransport:
        """The listener, built on first reference and kept for this server's
        whole life — including across ``stop()``.

        It used to be built inside ``start()`` and dropped by ``stop()``, which
        was fine while it held nothing but a socket. It holds the bearer token
        now, so a listener rebuilt by a restart would rotate the credential
        while ``mcp_config`` looked unchanged to whoever had already read it —
        and what they would see is the CLI failing to connect, naming nothing.
        Constructing it does not bind; ``reserve()`` inside ``start()`` does.
        """
        if self._transport is None:
            self._transport = LoopbackMcpTransport(
                host=self.host, port=self.port, authenticated=self.auth == "bearer"
            )
        return self._transport

    async def start(self) -> None:
        """Bind the port and serve. Idempotent."""
        transport = self._listener()
        # ``running`` rather than "have I got a transport object": the object
        # now outlives ``stop()``, so its existence stopped meaning "serving".
        if transport.running:
            return
        # The listener is built lazily and CACHED, so it may have been created
        # by an earlier read of ``mcp_config``/``auth_headers``/``cli_kwargs()``
        # — at which point it froze whatever ``host``/``port`` were then. Before
        # the listener outlived ``stop()`` this could not happen: ``start()``
        # built it fresh, so a late ``server.port = ...`` was honoured. Measured
        # after the change: the assignment was ignored and an ephemeral port was
        # bound instead, silently, while the caller believed they had pinned
        # one. Re-syncing restores that without REBUILDING, which would rotate
        # the token out from under a config document the caller already holds —
        # the thing the caching exists to prevent. Safe because binding happens
        # in ``reserve()`` inside ``start()``, not at construction; after a
        # restart these two are already equal.
        transport.host, transport.port = self.host, self.port
        await transport.start(self.build_mcp().streamable_http_app())
        # Copy the OS-chosen port back onto the public field: callers read
        # ``server.port`` (and ``server.url``, which is built from it) to wire
        # the CLI, and leaving it at the 0 they passed would hand them
        # ``http://127.0.0.1:0/mcp``.
        self.port = transport.port

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

        The transport OBJECT is kept — it is the only thing holding this
        server's credential, and dropping it here is what would make a restart
        hand out a token the config the caller is holding does not carry. Its
        own ``stop()`` is idempotent and safe on a listener that never started,
        so keeping it costs nothing.
        """
        if self._transport is not None:
            await self._transport.stop()

    # ---- wiring ----------------------------------------------------------------------------

    @property
    def url(self) -> str:
        """The MCP endpoint. ``/mcp`` is FastMCP's streamable-http mount."""
        return f"http://{self.host}:{self.port}/mcp"

    @property
    def mcp_config(self) -> str:
        """The ``--mcp-config`` value, inline rather than a temp file so there
        is no path to clean up and no window where a stale file points at a
        dead port.

        Inline is also why this one needs no ``stop()``-time cleanup now that
        it carries a credential: the document lives in an argv this process
        built and never touches disk, so there is no 0600 file to get right and
        no stale copy that can outlive the listener it names. That was a
        convenience argument when it was written; it is a containment argument
        now. (``serve_registry``'s document has to be a file — a registry's is
        too big for an argv — and it pays for that with a ``chmod`` and a
        deletion in ``finally``.)
        """
        return json.dumps(http_mcp_config(SERVER_NAME, self.url, token=self._listener().token))

    @property
    def auth_headers(self) -> dict[str, str]:
        """The headers a client must send to be answered, empty under
        ``auth="none"``.

        ``cli_kwargs()`` already carries what the CLI needs. This is the same
        thing for a client that is not the CLI, and it is a mapping rather than
        a raw token deliberately: a bare token invites concatenation into a
        URL, and a credential in a URL is the one placement the fence refuses
        outright, because URLs are logged.
        """
        return self._listener().auth_headers

    @property
    def tool_name(self) -> str:
        """The fully-qualified MCP tool name for ``--permission-prompt-tool``."""
        return qualified_tool_name(SERVER_NAME, TOOL_NAME)

    @property
    def prompts_seen(self) -> int:
        """How many permission prompts this server has been ASKED. A run that
        reports zero either never needed permission or never reached the
        server — worth being able to tell apart.

        Counted on arrival, which is what makes it the ``Elicitation`` id a UI
        keys its pending cards on. :attr:`decisions` is counted on completion,
        so the two are equal except while a prompt is in flight.
        """
        return self._seen

    @property
    def decisions(self) -> tuple[ApprovalDecision, ...]:
        """Every decision this server made, oldest first.

        The same shape and the same ordering as ``HookSettings.decisions``, so
        a service reading both the CLI's hook guard and its permission gate
        reads one trail rather than two.

        "Oldest first" means DECIDED-first, and the distinction is real because
        the CLI runs several tool calls at once: a record exists once there is
        a verdict to record, so a prompt that parked on a slow reviewer lands
        after one that arrived later and was answered immediately. That is the
        order an operator sees the run resolve, and arrival order is not lost —
        it is the ``Elicitation`` id, stamped from :attr:`prompts_seen`.
        """
        return tuple(self._decisions)

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
        deny, recorded either way.

        Never raises. This runs inside an MCP request handler, and an exception
        here reaches the CLI as a malformed result — which it reports as a
        broken permission system rather than a denial, leaving the model to
        retry a call nobody approved. A failure is a DENY with the reason
        attached, so the model sees it and can adapt.

        The verdict is reached FIRST and recorded second, and the recording is
        wrapped, because recording must never fail the decision: this is the
        same ordering ``HookSettings._emit`` keeps next door, and the same
        reason — a refusal that happened matters more than a record of it that
        did not. ``_record``'s own inputs are total functions (see
        :func:`_bounded` and :meth:`_now`); the suppression is the backstop for
        the case they stop being.
        """
        self._seen += 1
        # Snapshot the count for THIS prompt. The CLI can have several tool
        # calls in flight, so several prompts run concurrently on one server,
        # and the id a UI keys its pending cards on must be the number at the
        # time of this prompt rather than whatever the counter has reached by
        # the time the ``Elicitation`` is built. Nothing awaits in between
        # today; a local means that stays true when something does.
        seen = self._seen

        # Per-prompt, never on ``self``, for the same concurrency reason.
        pending = _Pending()
        answer = await self._verdict(tool_name, arguments, seen, pending)
        with contextlib.suppress(Exception):  # see the docstring
            await self._record(tool_name, arguments, answer, pending)
        return answer

    async def _verdict(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        seen: int,
        pending: _Pending,
    ) -> str:
        """The policy: which branch answers this prompt, and what it answers.

        Split out of :meth:`_decide` so recording is one call at one place
        instead of a line before every ``return`` — six branches each
        remembering to append is six chances for the one that mattered to
        forget. ``pending`` carries back the part of the record that is not
        readable from the answer itself; see :class:`_Pending`.

        Never raises, and the contract is why the answer is RENDERED inside a
        try as well: an ``Asker`` returning something that is not a
        ``Decision`` used to escape here as an ``AttributeError``.
        """
        tier = self._tier()
        if not should_gate(tier, requires_approval=False, key_step=True):
            # AUTO: the Asker is not consulted at all. "Asked, and they
            # approved" would produce the identical allow and would still be
            # wrong — the tier's whole meaning is that no person is in this
            # loop, which is why the test for it asserts on the call that did
            # not happen rather than on the answer. The record still exists,
            # and ``source="autonomy"`` is the only thing keeping "the tier
            # does not gate this" from reading as "a person approved it".
            pending.source, pending.allowing = "autonomy", True
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
            pending.source, pending.allowing = "auto_allow", True
            return _allow(arguments)

        asker = self.asker
        if asker is None:
            # ``__post_init__`` rejects this combination, so getting here means
            # ``autonomy`` was tightened after construction. ``_decide`` must
            # never raise, so it is a deny naming the missing transport: the
            # operator reads the cause in the model's own refusal instead of an
            # AttributeError inside an MCP request handler. It is an ``error``
            # and not a policy source, because nobody decided this: the trail
            # is how the operator finds the tightening that broke the wiring.
            pending.source = "error"
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

        # ``_ask`` still takes an ``Asker``; this one records whether it was
        # reached. See ``_ReachedAsker`` for why that is not derivable here.
        reached = _ReachedAsker(asker)
        try:
            decision = await self._ask(reached, request)
        except Exception as exc:  # noqa: BLE001 — see the docstring
            pending.source = "error"
            return _deny(f"the approval transport failed ({type(exc).__name__}: {exc})")
        finally:
            # In the ``finally`` so a transport that raised AFTER interrupting
            # somebody still records that it cost them the interruption.
            pending.asked = reached.reached

        try:
            if decision.kind == "expired":
                # Inside the try because reading ``.kind`` is the first thing
                # that fails on an answer that is not a ``Decision`` at all.
                # An expiry is recorded as ``timeout`` and not as the reviewer
                # declining: nobody refused this, the clock did.
                pending.source = "timeout"
            elif decision.kind == "value":
                # Nobody decided this one either. The request went out as
                # ``kind="approval"`` with ``choices=("approve", "deny")``, so
                # a ``value`` back means the transport handed over whatever the
                # person typed instead of a verdict — possibly the word "yes".
                # ``_render`` already refuses to call that a refusal in the
                # text the model reads ("which is not consent", and the test
                # for it asserts "declined" is absent); the record has to draw
                # the same line or it re-collapses on the way to the trail,
                # filing a broken Asker as a human who said no and sending
                # whoever reads it after a reviewer instead of after the
                # transport whose author is the only one who can fix it.
                pending.source = "error"
            answer = self._render(decision, tool_name, arguments)
            pending.allowing = decision.approved
            return answer
        except Exception as exc:  # noqa: BLE001 — see the docstring
            # A malformed answer, kept separate from a transport failure
            # because they need different fixes. The HITL API this replaced was
            # ``dict[str, str]`` with ``"approve"`` as a bare string, so an
            # ``Asker`` written from that memory returns a ``str`` and
            # ``decision.kind`` raises. Left to escape, that reaches the CLI as
            # a broken permission system rather than a denial — the model is
            # told the gate is broken and retries a call nobody approved. The
            # type is named so the transport's author can see what they sent.
            pending.source = "error"
            return _deny(
                f"the reviewer's answer was not a Decision "
                f"({type(decision).__name__}: {type(exc).__name__}: {exc})"
            )

    # ---- the record ------------------------------------------------------------------------

    async def _record(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        answer: str,
        pending: _Pending,
    ) -> None:
        """Append the decision, then put it on the run's observer.

        The order is the contract, not an implementation detail — the same one
        ``HookSettings._emit`` keeps. The in-memory append comes FIRST and the
        emit is best-effort, so an observer that raises, a bus that is closed,
        or a ``ctx`` with no ``emit`` at all still leaves :attr:`decisions`
        complete and the CLI's answer untouched.

        Everything except ``source`` and ``asked`` is read back off the wire
        (see :func:`_read_back`) so the record cannot describe a run that did
        not happen.
        """
        allowed, reason, updated = _read_back(answer)
        source = pending.source
        if pending.allowing and not allowed:
            # The branch decided to allow and the CLI got a refusal. There is
            # exactly one path that does that: ``_allow`` could not encode the
            # arguments. Nobody chose it, so it is an ``error`` — filing it
            # under ``autonomy`` or ``auto_allow`` would send an auditor
            # looking for a tier or a list that has never denied anything.
            source = "error"
        record = ApprovalDecision(
            tool=tool_name,
            # ``updatedInput`` when there is one, because that is what the tool
            # RUNS with: on a ``modify`` the CLI executes the reviewer's edit
            # and does not tell the model it changed, so recording the request
            # instead would name a call that never happened while the one that
            # did went unrecorded. On a deny there is no ``updatedInput`` and
            # what was refused is what was asked.
            arguments=_bounded(updated if updated is not None else arguments),
            allowed=allowed,
            reason=reason,
            source=source,
            at=self._now(),
            asked=pending.asked,
        )
        self._decisions.append(record)
        await self._emit(record)

    def _now(self) -> str:
        """The decision's timestamp: ISO-8601 UTC, from the RUN's clock.

        ``clock`` is the package's ``ClockPort`` seam — the same field
        ``Checkpointer`` takes, with ``time.time()`` as the same floor when
        nothing is injected. It is a field rather than something read off
        ``ctx`` because ``Ctx`` does not carry a clock; ``datetime.now()`` is
        not an option for the reason ``ClockPort`` exists at all, which is that
        a replayed or fake-clocked run would stamp its approval trail with real
        wall-clock times and stop lining up with the checkpoints beside it.

        UTC with an offset, not a naive local time: these records are compared
        against logs from other hosts, and a naive stamp is a timestamp that
        needs the reader to already know where it was written.

        Never raises. A clock that throws must not take down a decision that
        already happened, so a broken one records ``""`` — an obviously absent
        stamp on a present record, rather than a plausible wrong one.
        """
        try:
            now = self.clock.now() if self.clock is not None else time.time()
            return datetime.fromtimestamp(float(now), tz=UTC).isoformat()
        except Exception:  # noqa: BLE001 — see the docstring
            return ""

    async def _emit(self, record: ApprovalDecision) -> None:
        """Put the decision on the run's observer stream. Best-effort and never
        raises — same contract, and same reason, as ``HookSettings._emit``.

        ``gate.check`` rather than a kind of its own, because it is the kind
        ``ReActCognition`` and ``elicit`` already stamp for this exact event. A
        consumer assembling "every point at which a person took responsibility
        in this run" should not have to know which cognition produced each one.

        The payload carries no ``arguments``, and that is not a redaction of
        the record — :attr:`decisions` keeps them in full. The trail is held by
        the caller who wired this server and knows their retention rules; the
        observer stream fans out to logs and UIs they may not own, and the
        arguments are the half most likely to carry a token. Complete where it
        is held, summary where it is broadcast.
        """
        emit = getattr(self.ctx, "emit", None)
        if emit is None:
            return
        verdict = "allowed" if record.allowed else "refused"
        # ``wait_for`` and not a bare ``await``. ``suppress`` covers an observer
        # that RAISES and covers nothing at all about one that HANGS, and this
        # runs before ``_decide`` can answer the CLI. See ``_EMIT_TIMEOUT_S``.
        # ``TimeoutError`` is an ``Exception``, so the suppression still
        # swallows it, and the record — appended already — survives either way.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                emit(
                    "gate.check",
                    f"claude-cli {record.tool}: {verdict}",
                    payload={
                        "tool": record.tool,
                        "allowed": record.allowed,
                        "reason": record.reason,
                        "source": record.source,
                        "asked": record.asked,
                        "at": record.at,
                    },
                    # The agent this prompt was gated FOR. A run with a
                    # supervisor and three workers puts every gate.check on
                    # one stream, and without this they are indistinguishable.
                    agent=self.agent,
                ),
                timeout=_EMIT_TIMEOUT_S,
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


__all__ = [
    "SERVER_NAME",
    "TOOL_NAME",
    "ApprovalDecision",
    "ApprovalServer",
    "ApprovalSource",
]
