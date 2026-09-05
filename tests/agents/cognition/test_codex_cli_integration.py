"""End to end: the Codex cognition inside the rest of agentkit.

The other files in this set test the cognition against its own contract. This
one tests it as a COMPONENT — wrapped as a tool, dropped into a workflow, driven
through ``agent.stream``, run concurrently, watched by an observer — because
every one of those paths reaches it through code it does not control, and a
cognition that satisfies its own tests while breaking ``Workflow`` has not
shipped.

It also pins the contracts this cognition deliberately does NOT honour.
``ctx.autonomy``, ``agent.memory`` and ``Agent.resume()`` all mean something
everywhere else in the framework and mean nothing here, because the CLI owns
the loop. Each has a test, because "documented as unsupported" and "silently
ignored" look identical from the outside and only one of them is a decision.

The real-binary tests at the bottom are gated on ``codex`` being on PATH and on
``AGENTKIT_SKIP_REAL_CLI``. Cost for the whole file: three small tasks in an
empty sandbox.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CodexCliCognition, ReActCognition
from agentkit.agents.workflow import Workflow
from agentkit.context import WorkingContext
from agentkit.testing.fakes import CliRun, FakeCodexCli, codex_turn
from agentkit.testing.fakes.ctx import FakeCtx
from agentkit.tools.from_agent import as_tool
from tests.agents.cognition.test_codex_cli import drive, final_of

real_codex = pytest.mark.skipif(
    shutil.which("codex") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="codex CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


class RecordingObserverCtx(FakeCtx):
    """FakeCtx that records every ``ctx.emit(...)`` for later assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.emissions: list[dict[str, Any]] = []

    async def emit(
        self,
        kind: str,
        render: str = "",
        *,
        payload: Any = None,
        agent: str = "",
        parent_id: Any = None,
    ) -> None:
        self.emissions.append({"kind": kind, "render": render, "payload": payload, "agent": agent})


def answering(*texts: str) -> FakeCodexCli:
    return FakeCodexCli([CliRun.of(codex_turn(text=t, usage=(10, 0, 5))) for t in texts])


# ─────────────────────────────────────────────────────────────────────────────
# 1. through the Agent
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_run_returns_the_cognitions_result() -> None:
    cog = CodexCliCognition(spawn=answering("nine files"))
    result = await Agent(name="local", cognition=cog).run("count", FakeCtx())
    assert result.output == "nine files"
    assert result.stop_reason == "complete"


@pytest.mark.asyncio
async def test_agent_stream_yields_the_events_in_arrival_order() -> None:
    """A consumer's loop, not a drained list: the events have to arrive as the
    stream produces them, and the terminal one has to be last."""
    cli = FakeCodexCli.script(
        codex_turn(
            reasoning="looking",
            items=[{"type": "command_execution", "command": "ls", "aggregated_output": "a\n", "exit_code": 0}],
            text="one file",
            usage=(10, 0, 5),
        )
    )
    cog = CodexCliCognition(spawn=cli)
    agent = Agent(name="local", cognition=cog)

    kinds: list[str] = []
    async for ev in agent.stream("count", FakeCtx()):
        kinds.append(ev.type)

    assert kinds == ["message_delta", "tool_call", "tool_result", "message_delta", "final"]


@pytest.mark.asyncio
async def test_agent_stream_still_opens_its_own_span() -> None:
    """The Agent boundary's observability is the Agent's, not the cognition's —
    and it reads ``cognition.name``, which is why that attribute is not
    optional."""
    cog = CodexCliCognition(spawn=answering("x"))
    ctx = FakeCtx()
    async for _ in Agent(name="local", cognition=cog).stream("t", ctx):
        pass

    spans = [s for s in ctx.trace.spans if s.name == "invoke_agent"]
    assert spans, "the Agent boundary opened no span"
    assert spans[0].attrs["agentkit.agent.cognition"] == "codex_cli"


@pytest.mark.asyncio
async def test_the_cognition_itself_emits_no_spans_or_observations() -> None:
    """It delegates the loop, so there is nothing of its own to narrate — and a
    cognition that emitted per-payload observations would flood an observer
    with a transcript the CLI already prints."""
    ctx = RecordingObserverCtx()
    cog = CodexCliCognition(spawn=answering("x"))
    await drive(cog, ctx=ctx)

    assert ctx.emissions == []
    assert ctx.trace.spans == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. as a component of something larger
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_codex_agent_can_be_a_tool_for_a_parent_agent() -> None:
    """``as_tool`` turns an agent into a callable another agent's tool loop can
    invoke, and it must not care which cognition is underneath."""
    child = Agent(name="coder", cognition=CodexCliCognition(spawn=answering("patched")))
    tool = as_tool(child, name="delegate", description="hand work to the local codex")

    out = await tool.run({"task": "fix the import"}, FakeCtx())
    assert "patched" in str(out)


@pytest.mark.asyncio
async def test_a_codex_agent_runs_as_a_workflow_node() -> None:
    """A workflow node is just an agent, and the interesting part is that a
    node whose cognition shells out still reports a completed run to the graph
    rather than a partial one."""
    wf = Workflow(name="wf")
    wf.agent("n1", Agent(name="coder", cognition=CodexCliCognition(spawn=answering("done"))))

    result = await wf.run("do it", FakeCtx())
    # ``WorkflowResult`` carries per-node outputs on ``outputs``.
    assert result.outputs["n1"] == "done"


@pytest.mark.asyncio
async def test_a_react_parent_can_delegate_to_a_codex_child() -> None:
    """The composition the two cognitions are for: agentkit owns the outer loop
    and its middleware, the CLI owns one step of it."""
    child = Agent(name="coder", cognition=CodexCliCognition(spawn=answering("child answered")))
    parent = Agent(
        name="parent",
        cognition=ReActCognition(tools=[as_tool(child, name="coder", description="the local codex")]),
    )
    # No LLM is wired, so the parent cannot drive its own loop — the assertion
    # is that the tool registry accepted the child, which is where a cognition
    # mismatch would surface.
    assert "coder" in parent.cognition.tools.names()


# ─────────────────────────────────────────────────────────────────────────────
# 3. what this cognition deliberately ignores
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ctx_autonomy_does_not_influence_the_argv() -> None:
    """agentkit's autonomy tier and Codex's ``--ask-for-approval`` are two
    different vocabularies for two different mechanisms, and translating one
    into the other would silently loosen a sandbox somebody set."""
    argvs = []
    for tier in ("manual", "auto", "dontAsk"):
        cli = answering("x")
        ctx = FakeCtx()
        ctx.autonomy = tier
        await drive(CodexCliCognition(sandbox="read-only", spawn=cli), ctx=ctx)
        argvs.append(tuple(cli.invocations[-1].argv))

    assert len(set(argvs)) == 1, "ctx.autonomy leaked into the argv"


@pytest.mark.asyncio
async def test_agent_memory_is_never_queried() -> None:
    """The CLI manages its own context, so a ``MemorySource`` on the agent is
    not grounded against. Asserted rather than assumed: a memory that is wired
    and never read is a retrieval bug that looks like a quality problem."""

    class _Recording:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def recall(self, query: str, *, k: int = 5, **_: Any) -> list[Any]:
            self.queries.append(query)
            return []

    memory = _Recording()
    cog = CodexCliCognition(spawn=answering("x"))
    agent = Agent(name="local", cognition=cog, memory=memory)
    await agent.run("what did we decide?", FakeCtx())

    assert memory.queries == []


@pytest.mark.asyncio
async def test_resume_is_not_supported_and_says_so() -> None:
    """``Agent.resume()`` is ReAct-only. The CLI-native equivalent is
    ``resume_session_id=``, and pointing a caller at the wrong one costs them a
    conversation."""
    cog = CodexCliCognition(spawn=answering("x"))
    agent = Agent(name="local", cognition=cog)
    with pytest.raises(Exception) as caught:
        await agent.resume("cid", "decision", FakeCtx())
    assert caught.value is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. concurrency
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_drives_are_capped_by_max_concurrent() -> None:
    """The bound exists because the upstream SDK hangs at ~200 live
    subprocesses. Measured through a spawn that records how many were in flight
    at once."""
    peak = 0
    live = 0

    class _Counting(FakeCodexCli):
        async def __call__(self, *argv: str, **kw: Any):  # type: ignore[no-untyped-def]
            nonlocal peak, live
            live += 1
            peak = max(peak, live)
            try:
                await asyncio.sleep(0)
                return await super().__call__(*argv, **kw)
            finally:
                live -= 1

    cli = _Counting([CliRun.of(codex_turn(text="x", usage=(1, 0, 1)))], repeat_last=True)
    cog = CodexCliCognition(max_concurrent=2, spawn=cli)

    async def one() -> Any:
        return final_of(await drive(cog))

    results = await asyncio.gather(*(one() for _ in range(8)))

    assert all(r.output == "x" for r in results)
    assert peak <= 2, f"{peak} spawns were in flight at once with max_concurrent=2"


@pytest.mark.asyncio
async def test_the_semaphore_is_keyed_on_the_binary() -> None:
    """So a ``claude`` session and a ``codex`` session do not compete for one
    bound. Two different programs, two different subprocess tables, no shared
    failure mode."""
    from agentkit.agents.cognition._cli_common import _get_semaphore

    a = _get_semaphore("codex", None, 8)
    b = _get_semaphore("claude", None, 8)
    c = _get_semaphore("codex", None, 8)
    assert a is c
    assert a is not b


@pytest.mark.asyncio
async def test_a_second_config_home_gets_its_own_bound(tmp_path: Path) -> None:
    """Two isolated ``CODEX_HOME``s are two tenants, and one tenant's burst
    should not park the other's turns."""
    from agentkit.agents.cognition._cli_common import _get_semaphore

    one = _get_semaphore("codex", str(tmp_path / "a"), 8)
    two = _get_semaphore("codex", str(tmp_path / "b"), 8)
    assert one is not two


# ─────────────────────────────────────────────────────────────────────────────
# 5. correlation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_correlation_id_reaches_the_result() -> None:
    """No env bridge exists for Codex — see the flags file — so the id has to be
    on the result or there is nothing to join a trace on at all."""
    ctx = FakeCtx()
    ctx.correlation_id = "run-abc-123"
    result = final_of(await drive(CodexCliCognition(spawn=answering("x")), ctx=ctx))
    assert result.evals["external_run_id"] == "run-abc-123"


@pytest.mark.asyncio
async def test_a_ctx_with_no_correlation_id_omits_the_key() -> None:
    """Absent rather than empty: a key whose value is ``""`` looks like a
    correlation id that failed to propagate."""

    class _NoId(FakeCtx):
        def __init__(self) -> None:
            super().__init__()
            # An INSTANCE attribute: ``FakeCtx.__init__`` assigns
            # ``self.correlation_id = "fake-run"``, so a class-level override is
            # shadowed and the test would silently assert nothing.
            self.correlation_id = ""

    result = final_of(await drive(CodexCliCognition(spawn=answering("x")), ctx=_NoId()))
    assert "external_run_id" not in result.evals


# ─────────────────────────────────────────────────────────────────────────────
# 6. both cognitions side by side
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_codex_and_a_claude_agent_can_run_in_one_process() -> None:
    """The ``spawn=`` seam is per instance, which is what makes this possible at
    all: a process-wide ``patch`` of ``create_subprocess_exec`` would have both
    doubles fighting over one name."""
    from agentkit.agents.cognition import ClaudeCliCognition
    from agentkit.testing.fakes import FakeClaudeCli

    claude = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "from claude"}]}},
            {"type": "result", "subtype": "success", "is_error": False, "session_id": "s", "usage": {}},
        ]
    )
    codex = answering("from codex")

    results = await asyncio.gather(
        Agent(name="a", cognition=ClaudeCliCognition(spawn=claude)).run("t", FakeCtx()),
        Agent(name="b", cognition=CodexCliCognition(spawn=codex)).run("t", FakeCtx()),
    )
    assert {r.output for r in results} == {"from claude", "from codex"}


@pytest.mark.asyncio
async def test_both_cognitions_agree_on_the_evals_keys_they_share() -> None:
    """A caller who reads ``evals["session_id"]`` / ``["cli_duration_ms"]`` /
    ``["cli_return_code"]`` should not have to know which CLI ran. The keys
    each cognition adds ALONE are its own business; these three are the
    contract."""
    from agentkit.agents.cognition import ClaudeCliCognition
    from agentkit.testing.fakes import FakeClaudeCli

    claude_cli = FakeClaudeCli.script(
        [
            {"type": "system", "subtype": "init", "session_id": "s"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": "s",
                "duration_ms": 5,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ]
    )
    claude_cog = ClaudeCliCognition(spawn=claude_cli)
    claude_agent = Agent(name="a", cognition=claude_cog)
    claude_events = [
        ev async for ev in claude_cog.drive(claude_agent, "t", FakeCtx(), WorkingContext())
    ]
    codex_result = final_of(await drive(CodexCliCognition(spawn=answering("x"))))

    shared = {"session_id", "cli_duration_ms", "cli_return_code"}
    assert shared <= set(claude_events[-1].result.evals)
    assert shared <= set(codex_result.evals)


# ─────────────────────────────────────────────────────────────────────────────
# 7. against the real binary
# ─────────────────────────────────────────────────────────────────────────────


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_the_real_cli_runs_inside_a_workflow(tmp_path: Path) -> None:
    """The composition, end to end, with a real subprocess: a workflow node
    whose cognition is a CLI still reports a completed run to the graph."""
    (tmp_path / "n.txt").write_text("4271\n")
    cog = CodexCliCognition(
        working_dir=tmp_path,
        sandbox="read-only",
        ask_for_approval="never",
        skip_git_repo_check=True,
    )
    agent = Agent(name="reader", cognition=cog)
    # ``Workflow.agent(name, agent)`` — there is no ``add_node``; the Claude
    # sibling test has always used the real method. This one could not fail on
    # it before, because every real-CLI Codex test died in argv parsing first.
    wf = Workflow(name="wf")
    wf.agent("read", agent)

    result = await wf.run("What number is in n.txt? Digits only.", FakeCtx())
    # WorkflowResult carries per-node outputs on ``outputs``, keyed by node name
    # — same as the Claude sibling test asserts.
    assert "4271" in result.outputs["read"], result.outputs


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_the_real_cli_is_cancelled_and_the_process_dies(tmp_path: Path) -> None:
    """Cancellation against a real subprocess, which is the only place it can be
    proved: the terminal event says ``cancelled`` and the exit code is a signal
    rather than a chosen exit."""
    cog = CodexCliCognition(
        working_dir=tmp_path,
        sandbox="read-only",
        ask_for_approval="never",
        skip_git_repo_check=True,
    )

    class _Deadline(FakeCtx):
        def __init__(self) -> None:
            super().__init__()
            self._at = asyncio.get_event_loop().time() + 1.0

        def check_cancelled(self) -> None:
            if asyncio.get_event_loop().time() >= self._at:
                raise RuntimeError("run cancelled")

    result = final_of(
        await drive(cog, task="Write a very long essay about compilers.", ctx=_Deadline())
    )

    assert result.partial is True
    assert result.evals["stop_reason"] == "cancelled"
    assert result.evals["cli_return_code"] < 0, result.evals


@real_codex
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_the_real_cli_writes_a_file_under_a_workspace_write_sandbox(tmp_path: Path) -> None:
    """The one test that proves containment is real rather than declared: the
    same task is refused under ``read-only`` and lands under
    ``workspace-write``, and the difference is enforced by the OS rather than by
    the model's cooperation.
    """

    def _cog(sandbox: str) -> CodexCliCognition:
        return CodexCliCognition(
            working_dir=tmp_path,
            sandbox=sandbox,  # type: ignore[arg-type]
            ask_for_approval="never",
            skip_git_repo_check=True,
        )

    task = "Create a file called written.txt containing exactly the word OK, then stop."

    await drive(_cog("read-only"), task=task)
    assert not (tmp_path / "written.txt").exists(), "a read-only sandbox wrote a file"

    await drive(_cog("workspace-write"), task=task)
    assert (tmp_path / "written.txt").exists(), "a workspace-write sandbox wrote nothing"
