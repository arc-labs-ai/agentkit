"""07 — CodexCliCognition: use your local Codex install as the agent.

Delegates the whole agent loop to a locally-installed ``codex`` CLI (OpenAI's).
Uses whatever auth the CLI already resolved — no API key on agentkit's side.

The sibling of ``04_claude_cli.py``, and the interesting part is what differs.
Codex has no tool allow-list: every session gets the same ``shell`` and
``apply_patch``, and what stops them is an OS sandbox. So the containment here
is ``sandbox=`` rather than a list of tool names, and the second half of this
script shows it doing something — the same task is refused read-only and lands
under ``workspace-write``.

Like example 04, this needs the binary on PATH. The script exits cleanly with a
note if it isn't, so it's still safe in CI.

Run:

    uv run python examples/07_codex_cli.py
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

from agentkit import Agent
from agentkit.agents.cognition import CodexCliCognition
from agentkit.agents.control.safety import RunPolicy
from agentkit.kernel.types import Scope
from agentkit.runtime import RunContext


def _sandbox(root: Path) -> Path:
    """A git repo, because ``codex`` refuses to run outside one by default.

    ``skip_git_repo_check=True`` is the other way, and is what a service with a
    scratch workspace uses. Initialising one here instead keeps the example
    honest about the CLI's own default.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "hello.txt").write_text("The magic number is 137.\n")
    (root / "other.txt").write_text("Nothing to see here.\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


async def read_only_run(sandbox: Path) -> None:
    """Answer a question about the sandbox, without being able to change it."""
    cognition = CodexCliCognition(
        working_dir=sandbox,
        sandbox="read-only",
        ask_for_approval="never",
    )
    agent = Agent(name="reader", prompt="You are terse.", cognition=cognition)
    ctx = RunContext(correlation_id="example-07", scope=Scope(1, 1))

    print(f"→ sandbox: {sandbox}")
    print(f"→ caps:    {cognition.caps}  (RunPolicy allows: {RunPolicy().check([cognition]).allowed})")
    print("→ streaming events…\n")

    async for ev in agent.stream("What is the magic number in the sandbox files?", ctx):
        if ev.type == "message_delta":
            print(f"  [text] {ev.text!r}")
        elif ev.type == "tool_call":
            tc = ev.tool_call
            print(f"  [tool_call] {tc.name}({dict(tc.arguments)})")
        elif ev.type == "tool_result":
            print(f"  [tool_result] {ev.tool_result!r:.120}")
        elif ev.type == "step":
            print(f"  [step] {ev.text}")
        elif ev.type == "final":
            r = ev.result
            print("\n--- final ---")
            print(f"output   : {r.output!r}")
            u = r.usage
            print(f"tokens   : in={u.input_tokens} cached={u.cache_read_tokens} out={u.output_tokens}")
            # An ESTIMATE — the CLI reports no cost at all. See
            # ``evals["cost_source"]`` and the ``pricing=`` field.
            print(f"cost     : ${r.usage.cost_usd:.6f} ({r.evals.get('cost_source')})")
            print(f"thread   : {r.evals.get('session_id')!r}")
            print(f"duration : {r.evals.get('cli_duration_ms')} ms")


async def sandbox_comparison(sandbox: Path) -> None:
    """The same instruction under two sandboxes. Only one of them can obey it."""
    task = "Create a file called written.txt containing exactly the word OK, then stop."
    ctx = RunContext(correlation_id="example-07-sandbox", scope=Scope(1, 1))

    for mode in ("read-only", "workspace-write"):
        target = sandbox / "written.txt"
        target.unlink(missing_ok=True)
        cognition = CodexCliCognition(
            working_dir=sandbox,
            sandbox=mode,  # type: ignore[arg-type]
            ask_for_approval="never",
        )
        await Agent(name="writer", cognition=cognition).run(task, ctx)
        print(f"  sandbox={mode:<16} wrote the file: {target.exists()}")


async def multi_turn(sandbox: Path) -> None:
    """One thread, two turns. The second remembers the first."""
    cognition = CodexCliCognition(working_dir=sandbox, sandbox="read-only", ask_for_approval="never")
    async with cognition.session() as chat:
        async for _ in chat.turn("Remember the number 4271. Reply with just 'ok'."):
            pass
        async for ev in chat.turn("What number did I ask you to remember? Digits only."):
            if ev.type == "final":
                print(f"  turn 2 recalled: {ev.result.output.strip()!r}")
        print(f"  one thread across both turns: {chat.session_id}")


async def main() -> None:
    if shutil.which("codex") is None:
        print("Skipping: `codex` CLI not on PATH.")
        print("Install from https://developers.openai.com/codex/cli and re-run.")
        return

    with tempfile.TemporaryDirectory(prefix="agentkit-codex-cli-") as tmp:
        sandbox = _sandbox(Path(tmp) / "repo")

        await read_only_run(sandbox)

        print("\n--- the sandbox is the containment ---")
        await sandbox_comparison(sandbox)

        print("\n--- one thread, many turns ---")
        await multi_turn(sandbox)


if __name__ == "__main__":
    asyncio.run(main())
