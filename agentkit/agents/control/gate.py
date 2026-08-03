"""HumanGate — the shared human-in-the-loop decision.

Autonomy is set once per run (`RunContext.autonomy`) and honoured uniformly by every pattern. This module
is the single place that decides *whether a given action pauses for a human*. The suspend/resume
**mechanism** is `Suspended` (+ a pattern checkpointing to `ctx.store`); this is the gating **policy** they
share, so tool approval (the loop), node gates (the graph), and step gates all decide the same way.
"""

from __future__ import annotations

from enum import Enum

from agentkit.agents.result import Suspended  # re-exported: the suspend/resume value


class Autonomy(str, Enum):
    """Run-wide autonomy tier. Carried on ``RunContext.autonomy`` and consulted by
    :func:`should_gate` to decide which actions pause for a human.

    - ``AUTO``   — gate only actions a tool/author explicitly required.
    - ``GATED``  — also gate "key" steps/nodes.
    - ``MANUAL`` — gate everything.
    """

    AUTO = "auto"
    GATED = "gated"
    MANUAL = "manual"


def should_gate(
    autonomy: Autonomy | str, *, requires_approval: bool = False, key_step: bool = False
) -> bool:
    """Decide whether an action pauses for a human, given the run autonomy + per-action flags.

    - **MANUAL** → gate everything.
    - **GATED**  → gate explicitly-required actions (`requires_approval`) and `key_step`s.
    - **AUTO**   → gate only explicitly-required actions.

    A tool author's `requires_approval=True` is **always honoured**, even under AUTO — autonomy only
    *adds* gating (MANUAL gates everything; GATED additionally gates key steps).
    """
    if autonomy == Autonomy.MANUAL:
        return True
    if autonomy == Autonomy.GATED:
        return requires_approval or key_step
    return requires_approval  # AUTO


__all__ = ["Autonomy", "should_gate", "Suspended"]
