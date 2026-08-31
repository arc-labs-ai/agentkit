"""Helper process for the real-CLI ``serve_registry`` test — NOT collected by pytest.

Run as ``python -m tests.integrations.mcp._serve_registry_cli_e2e <marker-path>``;
prints one JSON line describing what happened.

This is the only test that can say the config file is one the **real binary**
accepts and that a served tool shows up in a real session. Every other test in
this area proves agentkit talks MCP correctly to agentkit, which is necessary
and not sufficient: the CLI has its own opinions about ``--mcp-config`` (a
document shape, a transport name, a startup deadline), and each of them was
wrong at least once while this was being written.

The tool writes to ``marker-path`` so the parent can distinguish "the model
said it called the tool" from "the tool ran in THIS process". The second is the
claim; the first is what a model will happily assert either way.

Its own process for the reason the siblings here are: serving an HTTP MCP app
leaves anyio memory streams for the garbage collector, and warnings-as-errors
turns that into an unraisable ``ResourceWarning`` collected at pytest's session
teardown, where a per-test filter cannot reach it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.context import WorkingContext
from agentkit.integrations.mcp import serve_registry
from agentkit.testing.fakes.ctx import FakeCtx
from agentkit.tools import ToolRegistry, tool

MARKER: Path = Path()


@tool(side_effecting=True, name="run_check")
def run_check(name: str) -> str:
    """Run the named check and report the verdict. Records the call on disk so
    the parent process can prove the tool body really executed here rather than
    trusting the model's account of it."""
    MARKER.write_text(name)
    return f"check {name} passed"


async def main() -> dict[str, Any]:
    registry = ToolRegistry.from_tools([run_check])
    spec = serve_registry(registry, name="engine", ctx=FakeCtx())
    async with spec:
        cognition = ClaudeCliCognition(
            model="claude-haiku-4-5-20251001",
            max_turns=3,
            # ``auto_approve`` is the list the spec computed from
            # ``requires_approval``; splatting it here is the wiring it exists
            # for. ``builtin_tools=False`` leaves the session with the registry
            # and nothing else, which is what makes "the tool appears in the
            # session" a claim about OUR server.
            allowed_tools=spec.auto_approve,
            **spec.cli_kwargs(builtin_tools=False),
        )
        agent = Agent(name="dev", cognition=cognition)
        result = None
        async for ev in cognition.drive(
            agent,
            "Call the run_check tool with name set to lint, then tell me what it returned.",
            FakeCtx(),
            WorkingContext(),
        ):
            if ev.type == "final":
                result = ev.result
        evals = result.evals if result else {}
        # The CLI's own init frame names every MCP server it managed to
        # connect. It is the only place that distinguishes "our server was
        # loaded" from "the CLI ignored the config and answered from the
        # model", which produces an identical-looking transcript.
        init = evals.get("cli_init") or {}
        return {
            "calls_seen": spec.calls_seen,
            "auth": spec.auth,
            # Reported so the parent can assert the binary got past a fence
            # that was really there. The whole bearer-token design rests on the
            # CLI actually sending a ``headers`` entry from ``--mcp-config``,
            # and nothing but the real binary can confirm it: an in-process
            # test would be agentkit sending itself a header it also checks.
            "config_has_credential": "headers" in json.loads(spec.config_path.read_text())[
                "mcpServers"
            ]["engine"],
            "tool_names": list(spec.tool_names),
            "mcp_servers": [
                {"name": s.get("name"), "status": s.get("status")}
                for s in (init.get("mcp_servers") or [])
            ],
            "stop_reason": result.stop_reason if result else "<no final event>",
            "text": (result.output if result else "")[:400],
            "evals": {k: str(v)[:200] for k, v in evals.items()},
        }


if __name__ == "__main__":
    MARKER = Path(sys.argv[1])
    verdict = asyncio.run(main())
    verdict["marker"] = MARKER.read_text() if MARKER.exists() else ""
    print(json.dumps(verdict))
