"""Elicitation — pausing a run for a person, as a value request rather than a veto.

The HITL the framework had was ``Agent.resume(run_id, decisions: dict[str, str], ctx)``:
approve/deny per pending tool call, ReAct only. Two independent production systems rejected
it for different reasons and both built their own:

* One needed a human to supply a VALUE (a one-time code) mid-journey, nowhere near a tool
  call. Approve/deny cannot express that.
* One needed to PARK IN PLACE: its coroutine holds live, unserialisable state, so unwinding
  and re-entering through ``resume()`` is impossible.

Both also had to add what the framework omitted — a deadline on a suspended run (one team
called the abandoned-tab case its "silent-stuck finding #1"), and a typed decision, since
``dict[str, str]`` carries no actor and no audit trail.

Four properties, each with a specific mechanism:

**(a) Elicitation, not only approval.** :class:`Elicitation` names what the run needs and
:class:`Decision` carries what the person supplied. ``kind="approval"`` is the old
approve/deny; ``kind="value"`` is "tell me the code". Same primitive, two shapes.

**(b) Parkable in place.** :func:`elicit` awaits an injected :class:`Asker`. A caller
holding live state WAITS inside its own coroutine — nothing unwinds, nothing is
serialised, the stack survives. With no ``Asker`` wired, the classic
return-and-resume path is unchanged, so callers that *can* serialise keep it.

**(c) Deadlined.** ``Elicitation.deadline_s`` bounds the wait. Expiry produces
``Decision(kind="expired")`` — an ordinary recorded outcome that the run degrades through,
not a hang and not an exception.

**(d) Typed.** :class:`Decision` carries ``actor`` and ``at``, so the audit trail can
answer "who approved this, and when".

**Transport is the application's.** The runtime takes an injected ``Asker`` and never
branches on terminal-vs-HTTP-vs-queue. Implementing ``ask`` is the whole integration.

**Secrets never persist.** A one-time code supplied through this path must not reach a log,
a payload, or a checkpoint. :class:`SecretValue` redacts itself in ``repr``/``str``, and a
run that has handled one marks its working context so the tool-loop cognition stops
snapshotting. See :func:`mark_context_tainted`.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover — annotation-only, no runtime import
    from agentkit.runtime.context import RunContext

# What a person can answer. ``modify`` carries replacement tool arguments;
# ``value`` carries the elicited value; ``expired`` is produced by the runtime
# itself when a deadline passes and is never returned by an ``Asker``.
DecisionKind = Literal["approve", "deny", "value", "modify", "expired"]

# What kind of answer a run is asking for.
ElicitationKind = Literal["approval", "value"]

# The key under which a tainted working context is flagged. Read by
# ``ReActCognition._save``; deliberately a scratchpad key rather than a field
# on ``WorkingContext`` so no serialiser is tempted to persist the flag itself.
SECRET_TAINT_KEY = "_agentkit_secret_taint"


class ElicitationExpired(RuntimeError):
    """Raised only by :func:`elicit_or_raise`, for a caller that would rather
    fail loudly than degrade. The default path returns an ``expired``
    :class:`Decision` instead — see the module docstring on (c)."""


class SecretValue:
    """A string that refuses to render itself.

    ``repr`` and ``str`` both return ``'***'``, so the value survives an
    f-string in a log line, an exception message, a ``dataclasses.asdict``
    round-trip that stringifies, and a debugger's variable pane. The real
    content comes out only through :meth:`reveal`, which is deliberately
    ugly to type and easy to grep for in review.

    This is not encryption and does not pretend to be. It closes the ACCIDENT
    — the value ending up somewhere nobody intended — not a determined
    attacker with the process memory.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The actual value. Every call site should be greppable in review."""
        return self._value

    def __repr__(self) -> str:
        return "SecretValue('***')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        # Compares by content so a test can assert on it, but only against
        # another SecretValue — never against a bare str, which would let
        # ``decision.value == "1234"`` quietly work and become the idiom.
        return isinstance(other, SecretValue) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


@dataclass(frozen=True)
class Elicitation:
    """What the run is asking a person for.

    ``prompt`` is shown to the human. ``choices`` narrows an approval to a
    fixed set. ``tool_call`` is set when the request is gating a specific tool
    invocation (the classic approve/deny), and ``None`` when it is a free
    value request from anywhere in a run — which is the case approve/deny
    could not express.

    ``secret=True`` marks the answer as sensitive: :func:`elicit` wraps the
    returned value in :class:`SecretValue` and taints the working context so
    the tool loop stops checkpointing. See :func:`mark_context_tainted`.
    """

    id: str
    prompt: str
    kind: ElicitationKind = "approval"
    choices: tuple[str, ...] = ()
    tool_call: Any = None  # ToolCall | None — typed Any to keep control/ free of kernel gravity
    secret: bool = False
    deadline_s: float | None = None  # None = wait indefinitely (today's behaviour)
    run_id: str = ""
    agent: str = ""

    def redacted(self) -> Elicitation:
        """A copy safe to log or persist. The prompt for a secret request can
        itself be revealing ("enter the code we texted to +44…"), so it is
        replaced rather than trusted."""
        if not self.secret:
            return self
        return replace(self, prompt="<redacted: secret elicitation>")


@dataclass(frozen=True)
class Decision:
    """What a person answered, with provenance.

    Replaces the stringly-typed ``dict[str, str]`` where ``"approve"`` /
    ``"reject"`` / a JSON blob of replacement arguments all shared one slot
    and neither actor nor timestamp existed at all.

    ``value`` holds the elicited value for ``kind="value"`` (a
    :class:`SecretValue` when the request was secret) and the replacement
    arguments dict for ``kind="modify"``.
    """

    kind: DecisionKind
    value: Any = None
    actor: str = ""  # WHO answered — the audit trail's missing half
    at: float = 0.0  # WHEN, epoch seconds; 0.0 means "not stamped"
    note: str = ""

    @property
    def approved(self) -> bool:
        """True for the two kinds that let the gated action proceed."""
        return self.kind in ("approve", "modify")

    def redacted(self) -> Decision:
        """A copy safe to log, persist, or put in an error message.

        A :class:`SecretValue` already refuses to render, but a plain value on
        a request that was *marked* secret must be scrubbed too — otherwise
        the protection depends on every producer remembering to wrap.
        """
        if self.value is None or isinstance(self.value, SecretValue):
            return self
        return replace(self, value="<redacted>")


def stamp(decision: Decision, *, actor: str = "", clock: Any = time.time) -> Decision:
    """Fill in ``actor``/``at`` if the ``Asker`` didn't.

    An ``Asker`` implementation that knows who it is talking to should set
    ``actor`` itself; this is the floor, so an audit trail always has a
    timestamp even from a lazy transport.
    """
    if decision.at and decision.actor:
        return decision
    return replace(
        decision,
        actor=decision.actor or actor,
        at=decision.at or float(clock()),
    )


@runtime_checkable
class Asker(Protocol):
    """The injected human transport. Terminal, HTTP, queue, Slack — the
    runtime never knows and never branches.

    One method. An implementation that AWAITS until a person answers gives
    you the park-in-place behaviour for free: the run's coroutine is simply
    awaiting, holding all of its live state, and nothing had to be
    serialisable.

    .. warning::
       ``ask`` must not block the event loop. Scheduling is cooperative, so a
       synchronous wait — ``input()``, ``requests.get()``, ``time.sleep()`` —
       never yields control, and ``deadline_s`` therefore CANNOT fire: the
       timeout is itself a coroutine that needs the loop to run it. A blocking
       implementation silently converts every deadline into an unbounded hang,
       which is the exact failure the deadline exists to prevent.

       Wrap synchronous work: ``await asyncio.to_thread(input, prompt)``.
    """

    async def ask(self, request: Elicitation) -> Decision: ...


# ── the primitive ────────────────────────────────────────────────────────────


def resolve_asker(ctx: Any, asker: Asker | None = None) -> Asker | None:
    """Find the ``Asker`` for this run: explicit argument, else ``ctx.asker``.

    ``getattr`` rather than attribute access so a ``NullCtx`` or a structural
    test stub without the field is simply "no asker wired" — which routes to
    the classic checkpoint-and-suspend path rather than crashing.
    """
    if asker is not None:
        return asker
    found = getattr(ctx, "asker", None)
    return found if found is not None else None


def mark_context_tainted(context: Any) -> None:
    """Record that a secret has entered this working context.

    The tool-loop cognition reads this flag and STOPS writing checkpoints for
    the rest of the run. That is a real trade — the run loses durability — and
    it is the right one: a one-time code injected as a tool message would
    otherwise be serialised into Postgres and outlive the ten minutes it was
    valid for. Losing resumability is recoverable; a credential in a durable
    store is not.
    """
    scratchpad = getattr(context, "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad[SECRET_TAINT_KEY] = True


def is_context_tainted(context: Any) -> bool:
    """Has a secret passed through this working context?"""
    scratchpad = getattr(context, "scratchpad", None)
    return bool(isinstance(scratchpad, dict) and scratchpad.get(SECRET_TAINT_KEY))


async def elicit(
    ctx: Any,
    request: Elicitation,
    *,
    asker: Asker | None = None,
    context: Any = None,
    clock: Any = time.time,
) -> Decision:
    """Ask a person, in place, with a deadline. The cognition-agnostic primitive.

    Any cognition can call this — it takes a ``Ctx``, not an ``Agent``, and
    knows nothing about tool loops. ``ReActCognition`` routes its approval gate
    through it; a custom cognition can call it directly mid-reasoning; the
    ``ask_human`` tool exposes it to the model.

    Returns ``Decision(kind="expired")`` when ``deadline_s`` passes with no
    answer. Expiry is an ORDINARY OUTCOME, not an exception: the abandoned-tab
    case is common, and a run that hangs forever waiting on a closed browser
    is the failure being fixed. Callers that would rather fail use
    :func:`elicit_or_raise`.

    Returns ``Decision(kind="deny")`` when no ``Asker`` is wired. That is the
    safe default for a direct call — a gate with no human attached must not
    silently pass — and it is NOT the path ``ReActCognition`` takes, which
    checks for an asker first and falls back to checkpoint-and-suspend.

    Emits a ``gate.check`` observation carrying the REDACTED request and
    decision. The unredacted value never reaches the observer.
    """
    resolved = resolve_asker(ctx, asker)
    if resolved is None:
        return stamp(
            Decision(kind="deny", note="no Asker wired on this run"),
            actor="system",
            clock=clock,
        )

    try:
        if request.deadline_s is None:
            answer = await resolved.ask(request)
        else:
            async with asyncio.timeout(request.deadline_s):
                answer = await resolved.ask(request)
    except TimeoutError:
        # ``asyncio.timeout`` raises TimeoutError (3.11+). The run DEGRADES:
        # a typed expired decision the caller treats like a denial, with the
        # distinction preserved so an operator can tell "someone said no" from
        # "nobody was there".
        answer = Decision(kind="expired", note=f"no answer within {request.deadline_s}s")

    answer = stamp(answer, actor="unknown", clock=clock)

    if request.secret and answer.value is not None and not isinstance(answer.value, SecretValue):
        # Wrap centrally so protection doesn't depend on every Asker
        # implementation remembering to. An Asker that already wrapped is
        # left alone.
        answer = replace(answer, value=SecretValue(str(answer.value)))
    if request.secret and answer.kind == "value" and context is not None:
        mark_context_tainted(context)

    # Redacted on BOTH sides. The observer is a fan-out point (Redis, Kafka,
    # a UI socket) and is exactly where an unredacted value would escape.
    await ctx.emit(
        "gate.check",
        render=f"elicitation {request.id}: {answer.kind}",
        payload={
            "elicitation": {
                "id": request.id,
                "kind": request.kind,
                "secret": request.secret,
                "prompt": request.redacted().prompt,
            },
            "decision": {
                "kind": answer.kind,
                "actor": answer.actor,
                "at": answer.at,
                "note": answer.note,
            },
        },
        agent=request.agent,
    )
    return answer


async def elicit_or_raise(ctx: Any, request: Elicitation, **kw: Any) -> Decision:
    """:func:`elicit`, but an expired deadline raises :class:`ElicitationExpired`.

    For a caller that genuinely cannot continue without the answer and would
    rather surface a failure than a degraded result.
    """
    answer = await elicit(ctx, request, **kw)
    if answer.kind == "expired":
        raise ElicitationExpired(f"elicitation {request.id!r} expired: {answer.note}")
    return answer


# ── back-compat with the stringly-typed decision map ─────────────────────────


def coerce_decision(raw: Any, *, actor: str = "", clock: Any = time.time) -> Decision:
    """Accept the legacy ``dict[str, str]`` value shape as a :class:`Decision`.

    ``Agent.resume`` historically took ``{tool_call_id: "approve" | "reject" |
    "deny" | <json args>}``. Rejecting that outright would break every caller,
    so the string forms map onto the typed kinds:

        "approve"          → Decision("approve")
        "reject" / "deny"  → Decision("deny")
        anything else      → Decision("modify", value=<the raw string>)

    The ``modify`` case keeps the raw string rather than parsing it here —
    ``_parse_args`` already owns "JSON, or fall back to the model's args" and
    duplicating that policy would let the two drift.

    A :class:`Decision` passes straight through (stamped), so new callers get
    the typed path and old ones keep working from the same call site.
    """
    if isinstance(raw, Decision):
        return stamp(raw, actor=actor, clock=clock)
    text = str(raw)
    if text == "approve":
        decision = Decision(kind="approve")
    elif text in ("reject", "deny"):
        decision = Decision(kind="deny")
    else:
        decision = Decision(kind="modify", value=text)
    return stamp(decision, actor=actor or "legacy", clock=clock)


# ── exposing elicitation to the model itself ─────────────────────────────────


def ask_human_tool(
    *,
    name: str = "ask_human",
    prompt_prefix: str = "",
    secret: bool = False,
    deadline_s: float | None = 300.0,
) -> Any:
    """A tool the MODEL can call to ask a person for a value mid-run.

    The other two routes into :func:`elicit` are framework-driven — the ReAct
    approval gate fires on a tool call, and a custom cognition calls ``elicit``
    from its own code. This is the third: the model itself decides it needs
    something only a human has ("what was the code we texted you?") and calls
    for it. That is the case approve/deny structurally could not express,
    because there is no tool call to approve — the ASK is the action.

    ``secret=True`` wraps the answer in :class:`SecretValue` and taints the
    working context so the tool loop stops checkpointing (see
    :func:`mark_context_tainted`). Use it for one-time codes and anything else
    that must not outlive the run.

    ``deadline_s`` defaults to five minutes rather than ``None``: a model-
    initiated ask is the most likely to hit an abandoned tab, and an unbounded
    default would make the hang the easy path.

    Returns a ``FunctionTool``; register it like any other::

        ReActCognition(tools=[ask_human_tool(secret=True), ...])
    """
    # Imported inside the function: ``agentkit.tools`` is imported BY the
    # agents layer, so a module-level import here would close the cycle.
    from agentkit.tools import FunctionTool

    # ``ctx`` must be annotated as ``RunContext`` (or left bare) for
    # ``_is_ctx_param`` to recognise it as the injected context rather than a
    # data parameter the model is supposed to fill in. ``ctx: Any`` would be
    # advertised in the tool schema and never injected.
    async def ask_human(question: str, ctx: RunContext) -> str:
        """Ask the human operator a question and wait for their typed answer.

        Use this only when the information genuinely cannot be obtained any
        other way — a one-time code, a confirmation, a preference only they
        know. Returns their answer, or a message saying nobody responded.
        """
        # Stable across processes. ``hash(str)`` is randomised per interpreter
        # (PYTHONHASHSEED), so an id built from it would differ between the
        # process that asked and any process later reading the audit trail.
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
        request = Elicitation(
            id=f"{name}:{digest}",
            prompt=f"{prompt_prefix}{question}" if prompt_prefix else question,
            kind="value",
            secret=secret,
            deadline_s=deadline_s,
        )
        decision = await elicit(ctx, request)
        if decision.kind == "expired":
            return "NO ANSWER: the human did not respond in time. Proceed without it or stop."
        if not decision.value:
            return "NO ANSWER: the human declined to answer."
        # ``reveal()`` here is deliberate and is the ONLY place the value is
        # unwrapped. It has to be: the answer's whole purpose is to re-enter
        # the model's prompt. The protection that matters is downstream — the
        # tainted context is never checkpointed, so the value lives in memory
        # for this run and nowhere else.
        if isinstance(decision.value, SecretValue):
            return decision.value.reveal()
        return str(decision.value)

    return FunctionTool.from_callable(
        ask_human,
        name=name,
        side_effecting=True,  # it interrupts a person; that is an external effect
        requires_approval=False,  # gating an ASK on an approval would be circular
    )


__all__ = [
    "SECRET_TAINT_KEY",
    "ask_human_tool",
    "Asker",
    "Decision",
    "DecisionKind",
    "Elicitation",
    "ElicitationExpired",
    "ElicitationKind",
    "SecretValue",
    "coerce_decision",
    "elicit",
    "elicit_or_raise",
    "is_context_tainted",
    "mark_context_tainted",
    "resolve_asker",
    "stamp",
]
