"""`Handoff` — the typed coordination verb for a coordinator Agent + SelectorPolicy.

Every major agent framework has *some* primitive that says "I'm done; the next turn should be
`target`": OpenAI Swarm has it, the Agents SDK has it, CrewAI has it, LangGraph has it. agentkit
*had* a pattern-matcher (`handoff_selector` reads `HANDOFF:<name>` markers from text), but no
**named verb** — code doing multi-agent routing read as "selector that grovels for a marker"
rather than "Handoff."

This module is the named primitive. A `Handoff(target, reason, message)` is a typed
dataclass that an agent emits, either:

  1. Via a tool call — `handoff_tool(targets=[...])` builds a `FunctionTool` the model can call
     to transfer control. The tool's schema constrains `target` to a known agent name (enum),
     so the model can't hand off to an unknown agent. The tool's execution returns a text form
     of the Handoff that gets fed back into the agent's context AND echoed into the coordinator
     transcript via the agent's final reply.
  2. Via a text marker — `parse_handoff(text)` reads a `HANDOFF:<target> [reason]` marker
     from plain assistant text. The text-honest path for models without tool calling.

The coordinator Agent's SelectorPolicy interprets it. `route_by_handoff(default=...)` is the
selector to use in new code — it reads the last assistant message, finds the Handoff (by
marker), and routes to the named target. The older `handoff_selector(default=...)` is kept
exported for backwards compatibility and is functionally a subset.

This is a **pattern**, not a graph DSL: just a typed return value. The coordinator owns
routing; the agent owns "I want to hand off to X." The seam between them is the Message
transcript and the Handoff marker, exactly the same blackboard discipline the rest of the
coordinator loop uses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentkit.agents.control.selector import Selector
from agentkit.kernel.protocols import Ctx
from agentkit.kernel.types import Message, ToolSchema
from agentkit.tools import FunctionTool

# Re-exported so `Selector` survives ruff's unused-import sweep — under `from __future__
# import annotations` the function return type below becomes a string, and ruff can't
# see the runtime reference.
_ = (Selector,)

# The canonical marker string. Match this in `parse_handoff` and emit this from `handoff_tool`'s
# fn result so both the marker-only and the tool-call paths produce the same transcript shape.
HANDOFF_MARKER = "HANDOFF:"


@dataclass(frozen=True)
class Handoff:
    """A typed transfer of control from one agent to another in a coordinator Agent's roster.

    Returned by an agent (via a tool call result OR an output marker) to say "I'm done; the
    next turn should be `target`, with this context." The coordinator's SelectorPolicy
    interprets it; the receiving agent inherits the transcript and the shared scratchpad.

    `target` — the name of the receiving agent. Must exist in the coordinator's roster;
        ``route_by_handoff`` checks it and falls back to the selector's default (never
        stalls the run), warning once per invented name. Matching is exact, or a unique
        case-insensitive match — so ``HANDOFF:Bob`` still reaches a child named ``bob``.
    `reason` — a short human-readable reason for the transfer. Goes into traces and the
        rendered marker so the receiving agent sees *why* it was handed control.
    `message` — optional text for the receiving agent. It is appended to the rendered
        marker on its own line, so it reaches the next agent inside the rendered transcript
        — NOT as a separate user message. If empty, the receiving agent just inherits the
        transcript with no fresh task framing. Multi-line is fine.
    """

    target: str
    reason: str = ""
    message: str = ""

    def render(self) -> str:
        """Render this Handoff as a marker string that `parse_handoff` can read back.

        Format: `HANDOFF:<target>[ <reason>]`. The optional `message` is appended on its own
        line so multi-line context survives the round-trip without breaking the single-line
        marker contract `handoff_selector` already supports."""
        line = f"{HANDOFF_MARKER}{self.target}"
        if self.reason:
            line = f"{line} {self.reason}"
        if self.message:
            return f"{line}\n{self.message}"
        return line


def parse_handoff(text: str, *, marker: str = HANDOFF_MARKER) -> Handoff | None:
    """Parse a `HANDOFF:<target> [reason]` marker out of plain text.

    Rules (deliberately a simple split, not a regex — fewer surprises, fewer dependencies):
      * Finds the LAST occurrence of `marker` in `text` (matching `handoff_selector`'s rfind
        behaviour — if an agent reasons aloud and then commits to a handoff at the bottom, the
        bottom wins).
      * The token immediately after the marker (whitespace-delimited) is the `target`.
      * Everything on the SAME line after the target, stripped, is the `reason`.
      * Lines AFTER the marker line are concatenated into the `message`.
      * Returns `None` if no marker is found or no target token follows it.
    """
    if not text:
        return None
    idx = text.rfind(marker)
    if idx == -1:
        return None
    tail = text[idx + len(marker) :]
    first_line, _, rest = tail.partition("\n")
    parts = first_line.split(None, 1)  # split on any whitespace, max 2 tokens
    if not parts:
        return None
    # Strip trailing sentence punctuation from the target token. "HANDOFF:bob."
    # is what a model actually writes when it ends a sentence, and a target of
    # ``"bob."`` matches no roster entry — so the handoff was silently lost.
    # Only TRAILING punctuation goes: an agent name is free to contain dots or
    # dashes internally (``team.research-v2``).
    target = parts[0].rstrip(".,;:!?)\"'")
    if not target:
        return None
    reason = parts[1].strip() if len(parts) > 1 else ""
    message = rest.strip()
    return Handoff(target=target, reason=reason, message=message)


def handoff_tool(
    targets: Sequence[str],
    *,
    name: str = "handoff",
    description: str = (
        "Transfer control of the conversation to another agent. Use this when another agent "
        "is better suited to the next step. The receiving agent will see the full transcript."
    ),
) -> FunctionTool:
    """Build a `FunctionTool` the model can call to emit a `Handoff`.

    The schema constrains `target` to the given list (JSON-schema enum) so the model literally
    cannot hand off to an unknown agent — a defense against the "model invents a name" failure
    mode that the marker-only path can't catch upfront.

    The tool's execution returns the Handoff rendered as a `HANDOFF:<target> <reason>` marker
    string. That's a deliberate choice: the rendered text becomes the tool-result message in
    the agent's transcript, the model sees it on its next turn, and when the agent emits its
    final reply (typically echoing the handoff conclusion), the marker survives into the
    coordinator transcript where `route_by_handoff` reads it. Same wire format as the marker-only path,
    one detector handles both."""
    targets_list = list(targets)
    if not targets_list:
        raise ValueError("handoff_tool requires at least one target agent name")

    async def _fn(args: Any, ctx: Ctx) -> str:
        # The payload the chain hands down is normally a ``FrozenDict`` — a
        # ``dict`` SUBCLASS, not the ``MappingProxyType`` this comment used to
        # name — so a narrow ``isinstance(args, dict)`` would pass today where it
        # once silently failed and rejected every handoff. That is true all the
        # way through the ``Invoker``: ``ToolRequest.__post_init__`` deep-freezes
        # ``arguments``, and ``deep_freeze`` now NORMALISES a
        # ``MappingProxyType`` into a ``FrozenDict`` instead of passing it
        # through (measured — ``ToolRequest("t", MappingProxyType({...}), tool)``
        # holds a ``FrozenDict``). So the "a ToolCall stores a proxy verbatim"
        # justification this comment used to give is dead.
        #
        # ``Mapping`` stays because ``args`` is ``Any`` and this ``_fn`` is
        # ``FunctionTool.fn`` — ``run(args, ctx)`` hands it straight down with no
        # freeze of its own, so anything that calls the tool WITHOUT going
        # through a ``ToolRequest`` supplies the mapping unmodified. That is not
        # hypothetical: ``test_handoff_tool_accepts_mappingproxy_arguments``
        # calls ``tool.fn(MappingProxyType({...}), None)`` directly, and measured
        # at that seam ``isinstance(args, dict)`` is False. A ``dict``-only test
        # would silently make that handoff ``target=""``. ``deep_freeze`` also
        # still returns every non-proxy ``Mapping`` by identity — rewriting a
        # caller's own type is the line it refuses to cross — so a ``ChainMap``
        # or a project's own mapping survives even the ``ToolRequest`` path.
        #
        # The FAIL-CLOSED shape below is the part to preserve. A non-mapping
        # yields ``target=""``, which cannot match ``targets_list`` and so falls
        # to the rejection branch — no handoff is routed on an argument shape
        # this function did not understand.
        is_map = isinstance(args, Mapping)
        target = args.get("target", "") if is_map else ""
        reason = args.get("reason", "") if is_map else ""
        message = args.get("message", "") if is_map else ""
        # If the model invented a target despite the enum, fail closed: emit nothing routable.
        # (The agent's final text will still go through; the coordinator's selector falls back to default.)
        if target not in targets_list:
            return f"handoff rejected: {target!r} is not a known target"
        return Handoff(target=target, reason=reason, message=message).render()

    schema = ToolSchema(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": targets_list},
                "reason": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["target"],
        },
    )
    # A handoff is a coordination verb, not an external mutation; it's safe to retry (same
    # target + same reason → same marker), so `idempotent=True`. It's `side_effecting=False`
    # because it doesn't touch the outside world — it just reshapes who-speaks-next.
    return FunctionTool(
        name=name,
        fn=_fn,
        description=description,
        side_effecting=False,
        schema=schema,
        idempotent=True,
    )


def route_by_handoff(default: str = "", *, marker: str = HANDOFF_MARKER) -> Selector:
    """Build a coordinator selector (a `Selector` per `agents/control/selector.py`) that routes by `Handoff`.

    Reads the LAST assistant-or-tool message in the transcript. If it carries a Handoff
    (parsed via `parse_handoff`) naming a child that IS on the roster, routes there.
    Otherwise returns `default` — or, when `default == ""`, returns `None` so the
    coordinator falls back to round-robin (never stalls the run).

    The roster check is the part that was missing. An invented target used to be returned
    verbatim, and ``SelectorPolicy`` — which cannot route a name it does not have —
    discarded it and fell back to round-robin by turn index. So a pinned ``default`` never
    got the turn precisely when it was needed most: the model had just named an agent that
    does not exist. Each distinct invented name now warns once.

    This supersedes `handoff_selector` for new code: same wire format (the `HANDOFF:` marker),
    but it returns a typed `Handoff` internally and degrades gracefully when no default is
    pinned. Keep using `handoff_selector` if you specifically want the always-fallback-to-a-
    pinned-default behaviour the older signature codifies."""

    # Distinct invented targets already reported, so a model looping on the same
    # hallucination warns once rather than once per turn.
    warned: set[str] = set()

    def _select(transcript: Sequence[Message], agents: Sequence[Any]) -> str | None:
        # Read the LAST message in the transcript, regardless of role — matches the existing
        # `handoff_selector` semantics so a Handoff named on the initial user task ALSO routes
        # the first turn. The marker is what matters, not the role that wrote it.
        last_text = transcript[-1].content if transcript else ""
        ho = parse_handoff(last_text or "", marker=marker)
        if ho is None:
            return default or None  # "" → None → coordinator's own round-robin fallback
        resolved = _resolve_target(ho.target, agents)
        if resolved is not None:
            return resolved
        if ho.target not in warned:
            warned.add(ho.target)
            import warnings

            roster = sorted(_roster_names(agents))
            warnings.warn(
                f"handoff target {ho.target!r} is not on the coordinator's roster "
                f"({roster or ['<empty>']}); falling back to "
                + (f"{default!r}" if default else "the coordinator's round-robin")
                + ". The model named an agent that does not exist — constrain it with "
                "handoff_tool(targets=[...]), whose schema enum makes that impossible.",
                UserWarning,
                stacklevel=2,
            )
        return default or None

    return _select


def _roster_names(agents: Sequence[Any]) -> list[str]:
    """The coordinator's child names, as the selector sees them. Children are
    duck-typed (anything with ``.name`` + ``.run``), so read defensively."""
    return [n for n in (getattr(a, "name", "") for a in agents) if n]


def _resolve_target(target: str, agents: Sequence[Any]) -> str | None:
    """Resolve a marker's target against the roster, or ``None`` if it is not there.

    An **empty roster** resolves everything: a caller invoking the selector
    directly (a unit test, a custom loop that passes no agents) must keep the
    pre-validation behaviour rather than have every target rejected.

    An exact name wins. Failing that, a UNIQUE case-insensitive match wins —
    ``HANDOFF:Bob`` for a child named ``bob`` has exactly one possible meaning,
    and canonicalising it is not a guess. Two children differing only by case
    make it ambiguous, and ambiguity is not resolved: the caller's ``default``
    takes over.
    """
    names = _roster_names(agents)
    if not names:
        return target
    if target in names:
        return target
    folded = [n for n in names if n.casefold() == target.casefold()]
    return folded[0] if len(folded) == 1 else None


__all__ = [
    "HANDOFF_MARKER",
    "Handoff",
    "handoff_tool",
    "parse_handoff",
    "route_by_handoff",
]
