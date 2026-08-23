"""``ApprovalServer`` — the Claude CLI's permission prompts routed to an ``Asker``.

`ClaudeCliCognition` delegates the whole loop to the CLI, which owns its own
permissions, and that left a service two options: `bypassPermissions` (the
agent may do anything, unattended) or `dontAsk` (anything not pre-approved is
denied outright and the run fails). agentkit already had the missing middle —
`Asker`, the injected human transport behind its own HITL path — but nothing
connected the two.

The CLI's seam is `--permission-prompt-tool`, which names an MCP tool it calls
instead of prompting a terminal. So this server IS an MCP server: it turns each
prompt into an `Elicitation`, awaits the application's `Asker`, and maps the
answer back onto the CLI's allow/deny shape.

Most of these tests drive the decision path directly, which is where the policy
lives. Two go through a real HTTP MCP round trip, because the wire format is
the part with a sharp edge — the CLI accepts exactly one text content block and
reports anything else as a broken permission system rather than a denial.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
from typing import Any

import pytest

from agentkit.agents.control.elicitation import Decision, Elicitation
from agentkit.integrations.mcp.approvals import (
    SERVER_NAME,
    TOOL_NAME,
    ApprovalServer,
)


class _Asker:
    """Records what it was asked and answers with a scripted decision."""

    def __init__(self, *decisions: Decision) -> None:
        self._decisions = list(decisions) or [Decision(kind="approve")]
        self.seen: list[Elicitation] = []

    async def ask(self, request: Elicitation) -> Decision:
        self.seen.append(request)
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


def _server(asker: Any, **kw: Any) -> ApprovalServer:
    return ApprovalServer(asker=asker, **kw)


async def _decide(server: ApprovalServer, tool: str = "Bash", **args: Any) -> dict[str, Any]:
    raw = await server._decide(tool, args or {"command": "ls"})
    parsed: dict[str, Any] = json.loads(raw)
    return parsed


# ── 1. the decision maps onto the CLI's shape ───────────────────────────────


@pytest.mark.asyncio
async def test_an_approval_allows_with_the_original_arguments() -> None:
    """``updatedInput`` is what the tool actually runs with, so passing the
    arguments through unchanged IS the approval."""
    asker = _Asker(Decision(kind="approve", actor="reviewer"))
    out = await _decide(_server(asker), "Bash", command="ls -la")

    assert out == {"behavior": "allow", "updatedInput": {"command": "ls -la"}}


@pytest.mark.asyncio
async def test_a_denial_carries_a_reason_the_model_can_act_on() -> None:
    """The message reaches the MODEL, which may adapt — so the reviewer's note
    is worth more than a bare "no"."""
    asker = _Asker(Decision(kind="deny", note="that path is out of scope"))
    out = await _decide(_server(asker), "Write", file_path="/etc/passwd")

    assert out["behavior"] == "deny"
    assert out["message"] == "that path is out of scope"


@pytest.mark.asyncio
async def test_a_denial_without_a_note_still_says_which_tool() -> None:
    out = await _decide(_server(_Asker(Decision(kind="deny"))), "Bash")
    assert "Bash" in out["message"]


@pytest.mark.asyncio
async def test_a_modification_becomes_approve_with_changes() -> None:
    """The CLI's own ``updatedInput`` semantics: the tool runs with the edited
    arguments and the model is not told they changed. That is how a reviewer
    redirects a write into a sandbox without derailing the run."""
    edited = {"file_path": "/tmp/sandbox/out.txt", "content": "hello"}
    asker = _Asker(Decision(kind="modify", value=edited))

    out = await _decide(_server(asker), "Write", file_path="/etc/out.txt", content="hello")

    assert out == {"behavior": "allow", "updatedInput": edited}


@pytest.mark.asyncio
async def test_an_expired_decision_denies_and_says_so() -> None:
    """"The reviewer said no" and "nobody answered" call for different fixes,
    so the message distinguishes them."""
    asker = _Asker(Decision(kind="expired", note="no answer within 30s"))
    out = await _decide(_server(asker))

    assert out["behavior"] == "deny" and "no answer" in out["message"]


@pytest.mark.parametrize("kind", ["deny", "expired"])
@pytest.mark.asyncio
async def test_anything_that_is_not_an_approval_denies(kind: str) -> None:
    """An approval gate that fails open is not a gate."""
    out = await _decide(_server(_Asker(Decision(kind=kind))))  # type: ignore[arg-type]
    assert out["behavior"] == "deny"


# ── 2. what the human is actually shown ─────────────────────────────────────


@pytest.mark.asyncio
async def test_the_elicitation_carries_the_arguments_not_just_a_sentence() -> None:
    """The arguments ARE the thing being approved — a path, a shell command —
    so they travel structurally rather than flattened into the prompt where a
    UI could not render them."""
    asker = _Asker()
    server = _server(asker, run_id="run-7", agent="dev")

    await _decide(server, "Bash", command="rm -rf /tmp/x")

    (request,) = asker.seen
    assert request.kind == "approval"
    assert request.tool_call == {"name": "Bash", "arguments": {"command": "rm -rf /tmp/x"}}
    assert "Bash" in request.prompt
    assert request.run_id == "run-7" and request.agent == "dev"
    assert request.choices == ("approve", "deny")


# ── 3. the failure modes ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_broken_asker_denies_rather_than_breaking_the_prompt() -> None:
    """An exception here reaches the CLI as a malformed result, which it
    reports as a broken permission system rather than a denial — leaving the
    model to retry a call nobody approved. So a transport failure is a DENY
    with the reason attached."""

    class _Broken:
        async def ask(self, request: Elicitation) -> Decision:
            raise RuntimeError("the approvals queue is down")

    out = await _decide(_server(_Broken()))

    assert out["behavior"] == "deny"
    assert "RuntimeError" in out["message"] and "queue is down" in out["message"]


@pytest.mark.asyncio
async def test_a_slow_asker_is_bounded_by_the_timeout() -> None:
    """The timeout is enforced HERE rather than trusted to the ``Asker``: the
    protocol lets an implementation wait forever, and a queue worker holding a
    CLI subprocess open indefinitely is a resource leak with a model attached."""

    class _Slow:
        async def ask(self, request: Elicitation) -> Decision:
            await asyncio.sleep(10)
            return Decision(kind="approve")

    out = await _decide(_server(_Slow(), timeout_s=0.05))

    assert out["behavior"] == "deny" and "0.05" in out["message"]


@pytest.mark.asyncio
async def test_no_timeout_waits_for_the_person() -> None:
    """``None`` is right for an interactive UI, where the answer arrives when
    the reviewer gets to it."""

    class _Eventually:
        async def ask(self, request: Elicitation) -> Decision:
            await asyncio.sleep(0.01)
            return Decision(kind="approve")

    out = await _decide(_server(_Eventually(), timeout_s=None))
    assert out["behavior"] == "allow"


# ── 4. auto-allow exists to stop habituation ────────────────────────────────


@pytest.mark.asyncio
async def test_auto_allowed_tools_never_reach_the_person() -> None:
    """The CLI prompts for reads too, and a person clicking yes on forty
    ``Read`` calls is not oversight — it is habituation, which is what makes
    the fortieth prompt, the one that mattered, get the same reflexive yes."""
    asker = _Asker(Decision(kind="deny", note="should never be consulted"))
    server = _server(asker, auto_allow=("Read", "Glob"))

    assert (await _decide(server, "Read", file_path="/x"))["behavior"] == "allow"
    assert asker.seen == []

    # ...and a tool NOT on the list still goes to the human.
    assert (await _decide(server, "Bash"))["behavior"] == "deny"
    assert len(asker.seen) == 1


@pytest.mark.asyncio
async def test_prompts_are_counted_including_auto_allowed_ones() -> None:
    """A run reporting zero prompts either never needed permission or never
    reached the server, and those are worth telling apart."""
    server = _server(_Asker(), auto_allow=("Read",))
    await _decide(server, "Read", file_path="/x")
    await _decide(server, "Bash")
    assert server.prompts_seen == 2


# ── 5. the wiring a caller copies ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_cli_kwargs_wire_the_cognition() -> None:
    """These are existing ``ClaudeCliCognition`` fields — the server produces
    values, it does not need the cognition to know about it."""
    server = _server(_Asker(), port=51234)
    kwargs = server.cli_kwargs()

    assert kwargs["permission_prompt_tool"] == f"mcp__{SERVER_NAME}__{TOOL_NAME}"
    config = json.loads(kwargs["mcp_config"][0])
    assert config["mcpServers"][SERVER_NAME] == {
        "type": "http",
        "url": "http://127.0.0.1:51234/mcp",
    }
    # Strict by default: without it the CLI also loads whatever MCP servers the
    # working directory or the user's home configuration happen to define,
    # which is not what a service wiring an approval gate is asking for.
    assert kwargs["strict_mcp_config"] is True


@pytest.mark.asyncio
async def test_the_cognition_accepts_them() -> None:
    """The wiring is only useful if it type-checks against the real fields."""
    from agentkit.agents.cognition import ClaudeCliCognition

    server = _server(_Asker(), port=51235)
    argv = ClaudeCliCognition(**server.cli_kwargs())._build_argv("t", system_prompt="")

    assert argv[argv.index("--permission-prompt-tool") + 1] == server.tool_name
    assert "--strict-mcp-config" in argv
    assert server.url in argv[argv.index("--mcp-config") + 1]


# ── 6. the wire format the CLI demands ─────────────────────────────────────


@pytest.mark.asyncio
async def test_the_result_is_exactly_one_text_block() -> None:
    """THE sharp edge, measured against the real binary before it was fixed:

        Error calling tool (Write): Permission prompt tool returned an invalid
        result. Expected a single text block param with type="text" and a
        string text value.

    FastMCP's default adds a ``structuredContent`` field alongside the text,
    and every prompt then fails with that message while the tool itself looks
    like it ran — the model is told the permission SYSTEM is broken rather than
    that it was denied, so it retries a call nobody approved.

    Driven through FastMCP's own dispatch rather than a socket: the contract is
    the shape of the result, and an HTTP client would only add a transport that
    can fail for its own reasons.
    """
    asker = _Asker(Decision(kind="approve"))
    blocks = await ApprovalServer(asker=asker).build_mcp().call_tool(
        TOOL_NAME, {"tool_name": "Bash", "input": {"command": "ls"}}
    )

    assert isinstance(blocks, list), "a (content, structured) pair fails the CLI's check"
    assert len(blocks) == 1
    assert blocks[0].type == "text"
    assert json.loads(blocks[0].text) == {
        "behavior": "allow",
        "updatedInput": {"command": "ls"},
    }
    assert asker.seen and asker.seen[0].tool_call["name"] == "Bash"


@pytest.mark.asyncio
async def test_a_denial_survives_the_same_dispatch() -> None:
    """The negative half of the contract."""
    asker = _Asker(Decision(kind="deny", note="not on my watch"))
    blocks = await ApprovalServer(asker=asker).build_mcp().call_tool(
        TOOL_NAME, {"tool_name": "Write", "input": {"file_path": "/etc/x"}}
    )

    assert json.loads(blocks[0].text) == {"behavior": "deny", "message": "not on my watch"}


@pytest.mark.asyncio
async def test_the_tool_is_named_what_the_flag_expects() -> None:
    """``--permission-prompt-tool`` takes ``mcp__<server>__<tool>``, so the
    server and tool names are wire contract rather than labels."""
    tools = await ApprovalServer(asker=_Asker()).build_mcp().list_tools()

    assert [t.name for t in tools] == [TOOL_NAME]
    assert ApprovalServer(asker=_Asker()).tool_name == f"mcp__{SERVER_NAME}__{TOOL_NAME}"


# ── 7. the listener ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_server_binds_loopback_on_an_ephemeral_port() -> None:
    """No authentication, so loopback-only IS the containment."""
    async with ApprovalServer(asker=_Asker()) as server:
        assert server.host == "127.0.0.1"
        assert server.port > 0
        assert server.url.startswith("http://127.0.0.1:")


@pytest.mark.asyncio
async def test_stopping_releases_the_port() -> None:
    """A leaked listener would collide with the next agent on the host."""
    server = ApprovalServer(asker=_Asker())
    await server.start()
    port = server.port
    await server.stop()

    with socket.socket() as s:  # binding it again proves the listener is gone
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    server = ApprovalServer(asker=_Asker())
    await server.start()
    port = server.port
    await server.start()
    try:
        assert server.port == port
    finally:
        await server.stop()


# ── 8. against the real binary ─────────────────────────────────────────────
#
# Everything above pins the decision policy and the result shape. Only the CLI
# can confirm that it finds the server, calls the tool, and honours the answer
# — and each of those was wrong at least once while this was being built.
#
# Driven through a helper PROCESS (``_approval_e2e.py``) rather than inline.
# Serving FastMCP's streamable-HTTP app leaves anyio memory streams for the
# garbage collector, and this project runs with warnings-as-errors, so the
# finalisation surfaces as an unraisable ``ResourceWarning`` that pytest
# collects at SESSION teardown — where a per-test filter cannot reach it, and
# where it also fails unrelated tests. Silencing it globally would blind the
# suite to a class of real leak. The subprocess boundary contains it exactly.

real_cli = pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("AGENTKIT_SKIP_REAL_CLI") == "1",
    reason="claude CLI not on PATH or AGENTKIT_SKIP_REAL_CLI=1",
)


def _e2e(decision: str, target: Any) -> dict[str, Any]:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tests.integrations.mcp._approval_e2e", decision, str(target)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=pathlib.Path(__file__).resolve().parents[3],
    )
    assert proc.returncode == 0, f"helper failed:\n{proc.stdout}\n{proc.stderr}"
    verdict: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return verdict


@real_cli
def test_the_real_cli_routes_a_permission_prompt_to_the_asker(tmp_path: Any) -> None:
    """The whole point: the CLI asks, a person answers, the answer is obeyed.

    The reviewer declines, so the file must NOT exist afterwards — the check a
    mocked prompt cannot make.
    """
    target = tmp_path / "out.txt"
    verdict = _e2e("deny", target)

    assert verdict["prompts"] >= 1, "the CLI never reached the approval server"
    assert "Write" in verdict["asked_about"]
    assert not verdict["written"], "the reviewer declined; nothing should be written"
    # The CLI reports the denial to the model rather than failing the run.
    assert verdict["stop_reason"] == "complete", verdict["evals"]


@real_cli
def test_the_real_cli_honours_an_approval(tmp_path: Any) -> None:
    """The positive control. Without it, a bridge that denied EVERYTHING —
    including by accident — would pass the test above."""
    target = tmp_path / "out.txt"
    verdict = _e2e("approve", target)

    assert verdict["prompts"] >= 1
    assert verdict["written"], "the reviewer approved; the write should have happened"
    assert "hello" in verdict["content"]
