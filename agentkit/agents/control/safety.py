"""RunPolicy — the "Agents Rule of Two" pre-run check over a tool set's capability tags.

A run that can simultaneously access private data, ingest untrusted content, and exfiltrate via egress
is the lethal trifecta. Tools self-declare `caps`; `RunPolicy.check` flags a set that covers all three
so the caller gates or splits the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TRIFECTA = ("private_data", "untrusted_content", "egress")

PolicyMode = Literal["flag", "deny"]


@dataclass(frozen=True)
class PolicyVerdict:
    """Frozen — the return value of ``RunPolicy.check`` is a decision.
    Callers may stash the verdict and re-consult it, or log it into an
    audit trail alongside the run's other values. In-place mutation of
    ``.allowed`` between inspection and second read would silently flip
    the trifecta gate."""

    allowed: bool
    capabilities: tuple[str, ...]
    reason: str = ""


class RunPolicy:
    """Pre-run trifecta gate. ``mode`` controls what happens when the tool set
    covers all three lethal caps:

    - ``"flag"``  → return a ``PolicyVerdict`` with ``allowed=False`` so the
      caller decides (audit-log it, gate the run behind a human, split the run).
    - ``"deny"``  → raise ``PermissionError`` immediately so the run never starts.
    """

    def __init__(self, *, mode: PolicyMode = "flag") -> None:
        self.mode: PolicyMode = mode  # "flag" → return verdict; "deny" → raise

    def capabilities(self, tools: list[Any]) -> tuple[str, ...]:
        caps: set[str] = set()
        for t in tools:
            caps.update(getattr(t, "caps", ()) or ())
        return tuple(sorted(caps & set(TRIFECTA)))

    def check(self, tools: list[Any]) -> PolicyVerdict:
        caps = self.capabilities(tools)
        if len(caps) < len(TRIFECTA):
            return PolicyVerdict(True, caps)
        reason = (
            "lethal trifecta: this tool set combines private-data access, untrusted-content "
            "ingestion, and egress in one run — require a human gate or split the run"
        )
        if self.mode == "deny":
            raise PermissionError(reason)
        return PolicyVerdict(False, caps, reason)
