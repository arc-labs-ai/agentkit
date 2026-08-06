"""04 — ClaudeCliCognition: use your local Claude Code install as the LLM.

Delegates the whole agent loop to a locally-installed ``claude`` CLI (the
one you get from ``pip install claude-code`` / ``brew install`` / Anthropic's
installer). Uses whatever auth the CLI already resolved — no API key on
agentkit's side.

Unlike examples 01-03 (which run against ``FakeLLM``, no external deps),
this one requires the ``claude`` binary on PATH. The script exits cleanly
with a note if it's not installed, so it's still safe in CI.

Run:

    uv run python examples/04_claude_cli.py
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.kernel.types import Scope
from agentkit.runtime import RunContext


async def main() -> None:
    if shutil.which("claude") is None:
        print("Skipping: `claude` CLI not on PATH.")
        print("Install from https://docs.claude.com/en/docs/claude-code and re-run.")
        return

    with tempfile.TemporaryDirectory(prefix="agentkit-claude-cli-") as sandbox_str:
        sandbox = Path(sandbox_str)
        (sandbox / "hello.txt").write_text("The magic number is 137.\n")
        (sandbox / "other.txt").write_text("Nothing to see here.\n")

        agent = Agent(
            name="reader",
            prompt="You are terse. Use tools to answer.",
            cognition=ClaudeCliCognition(
                working_dir=sandbox,
                allowed_tools=("Read", "Glob", "Grep"),
                permission_mode="acceptEdits",
            ),
        )
        ctx = RunContext(correlation_id="example-04", scope=Scope(1, 1))

        print(f"→ sandbox: {sandbox}\n→ streaming events…\n")
        async for ev in agent.stream("What is the magic number in the sandbox files?", ctx):
            if ev.type == "message_delta":
                print(f"  [text] {ev.text!r}")
            elif ev.type == "tool_call":
                tc = ev.tool_call
                print(f"  [tool_call] {tc.name}({dict(tc.arguments)})")
            elif ev.type == "tool_result":
                preview = repr(ev.tool_result)[:120]
                print(f"  [tool_result] {preview}")
            elif ev.type == "final":
                r = ev.result
                print("\n--- final ---")
                print(f"output   : {r.output!r}")
                print(f"tokens   : in={r.usage.input_tokens} out={r.usage.output_tokens}")
                print(f"session  : {r.evals.get('session_id')!r}")
                print(f"duration : {r.evals.get('cli_duration_ms')} ms")


if __name__ == "__main__":
    asyncio.run(main())
