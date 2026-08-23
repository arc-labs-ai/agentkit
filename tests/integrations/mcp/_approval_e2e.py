"""Helper process for the real-CLI approval test — NOT collected by pytest.

Run as ``python -m tests.integrations.mcp._approval_e2e <decision> <path>``; it
prints one JSON line describing what happened.

It lives in its own process on purpose. Serving FastMCP's streamable-HTTP app
leaves anyio memory streams for the garbage collector, and this project runs
with warnings-as-errors, so that finalisation surfaces as an unraisable
``ResourceWarning`` collected at pytest's session teardown — where a per-test
filter cannot reach it. Suppressing it globally would blind the whole suite to
a class of real leak. A subprocess boundary contains it exactly: the leak dies
with this interpreter, and the parent test asserts on the verdict.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.agents.control.elicitation import Decision, Elicitation
from agentkit.context import WorkingContext
from agentkit.integrations.mcp import ApprovalServer
from agentkit.testing.fakes.ctx import FakeCtx


class _Reviewer:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.seen: list[str] = []

    async def ask(self, request: Elicitation) -> Decision:
        self.seen.append((request.tool_call or {}).get("name", ""))
        if self.kind == "approve":
            return Decision(kind="approve", actor="reviewer")
        return Decision(kind="deny", actor="reviewer", note="writes need a ticket")


async def main(kind: str, target: Path) -> dict[str, Any]:
    reviewer = _Reviewer(kind)
    async with ApprovalServer(asker=reviewer, timeout_s=90) as approvals:
        cognition = ClaudeCliCognition(
            model="claude-haiku-4-5-20251001",
            tools=("Write",),
            max_turns=2,
            **approvals.cli_kwargs(),
        )
        agent = Agent(name="dev", cognition=cognition)
        result = None
        async for ev in cognition.drive(
            agent,
            f"Write the word hello into {target} using the Write tool.",
            FakeCtx(),
            WorkingContext(),
        ):
            if ev.type == "final":
                result = ev.result
        # The filesystem check happens in ``__main__``, synchronously: reading
        # a path inside a coroutine blocks the loop, and there is no reason to
        # do it while the server is still up.
        return {
            "prompts": approvals.prompts_seen,
            "asked_about": reviewer.seen,
            "stop_reason": result.stop_reason if result else "<no final event>",
            "evals": {k: str(v)[:200] for k, v in (result.evals if result else {}).items()},
        }


if __name__ == "__main__":
    _target = Path(sys.argv[2])
    verdict = asyncio.run(main(sys.argv[1], _target))
    verdict["written"] = _target.exists()
    verdict["content"] = _target.read_text() if _target.exists() else ""
    print(json.dumps(verdict))
