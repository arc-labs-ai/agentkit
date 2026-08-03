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

    `target` — the name of the receiving agent. Must exist in the coordinator's roster; an
        unknown target falls back to the selector's default (never stalls the run).
    `reason` — a short human-readable reason for the transfer. Goes into traces and the
        rendered marker so the receiving agent sees *why* it was handed control.
    `message` — optional text the orchestrator wants the receiving agent to see as the next
        user turn. If empty, the receiving agent just inherits the transcript with no fresh
        task framing. Multi-line is fine; it's wrapped inside the marker render.
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
    target = parts[0]
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
        # ``ToolCall.arguments`` is a ``MappingProxyType``, not a
        # ``dict``. Match on ``Mapping`` so the proxy view flows in
        # correctly — otherwise every handoff sees ``target=""`` and
        # falls to the "not a known target" rejection branch.
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
    """Build a coordinator Agent + SelectorPolicy selector (a `Selector` per `agents/control/selector.py`) that routes by `Handoff`.

    Reads the LAST assistant-or-tool message in the transcript. If it carries a Handoff
    (parsed via `parse_handoff`), routes to the named target. Otherwise returns `default` —
    or, when `default == ""`, returns `None` so the coordinator falls back to round-robin
    (never stalls the run).

    This supersedes `handoff_selector` for new code: same wire format (the `HANDOFF:` marker),
    but it returns a typed `Handoff` internally and degrades gracefully when no default is
    pinned. Keep using `handoff_selector` if you specifically want the always-fallback-to-a-
    pinned-default behaviour the older signature codifies."""

    def _select(transcript: Sequence[Message], agents: Sequence[Any]) -> str | None:
        # Read the LAST message in the transcript, regardless of role — matches the existing
        # `handoff_selector` semantics so a Handoff named on the initial user task ALSO routes
        # the first turn. The marker is what matters, not the role that wrote it.
        last_text = transcript[-1].content if transcript else ""
        ho = parse_handoff(last_text or "", marker=marker)
        if ho is not None:
            return ho.target
        return default or None  # "" → None → coordinator's own round-robin fallback

    return _select


__all__ = [
    "HANDOFF_MARKER",
    "Handoff",
    "handoff_tool",
    "parse_handoff",
    "route_by_handoff",
]
