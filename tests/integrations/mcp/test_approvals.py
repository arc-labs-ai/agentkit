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
import contextlib
import ipaddress
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.agents.control.elicitation import Decision, Elicitation
from agentkit.agents.control.gate import should_gate
from agentkit.context import WorkingContext
from agentkit.integrations.mcp import approvals
from agentkit.integrations.mcp.approvals import (
    _MAX_ARGUMENT_CHARS,
    SERVER_NAME,
    TOOL_NAME,
    ApprovalServer,
)
from agentkit.testing.fakes.clock import FakeClock
from agentkit.testing.fakes.ctx import FakeCtx
from tests.agents.cognition.test_claude_cli import _FakeProcess, _line


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
    CLI subprocess open indefinitely is a resource leak with a model attached.

    The slow asker waits on an ``Event`` nobody sets rather than sleeping a
    fixed 10s. Those are the same claim — "this asker does not answer" — but
    the sleep encoded the regression's cost into the suite: when the bound
    stopped firing, the test did not fail, it ran the whole ten seconds and
    then failed, once, in whichever full run happened to reach it. A wait that
    never ends cannot be mistaken for a slow one, and the tripwire below turns
    the same regression into a named failure instead of ten wasted seconds.
    """
    never_answered = asyncio.Event()  # deliberately never set

    class _Slow:
        async def ask(self, request: Elicitation) -> Decision:
            await never_answered.wait()
            return Decision(kind="approve")

    # Deliberately generous, and NOT a second assertion about the 0.05s
    # deadline: a tight wall-clock budget on a loaded machine fails a test that
    # is about whether a bound exists at all. If ``timeout_s`` stops being
    # enforced this trips and names the regression; the suite-wide timeout
    # never has to.
    out = await asyncio.wait_for(_decide(_server(_Slow(), timeout_s=0.05)), timeout=10)

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
async def test_auto_allow_matches_the_tool_name_and_ignores_the_arguments() -> None:
    """POSITIVE CONTROL, and the thing the class docstring now says out loud:
    ``auto_allow`` is a grant over a tool NAME, not over the operations that
    tool could perform. Measured against this server with
    ``auto_allow=("Read",)`` and a reviewer that denies everything::

        Read   /etc/passwd      -> allow      (reviewer never consulted)
        Read   /etc/shadow      -> allow      (reviewer never consulted)
        Read   ~/.ssh/id_rsa    -> allow      (reviewer never consulted)
        Write  /etc/passwd      -> deny       (reviewer consulted)

    This passes both before and after the ``auto_allow_when`` change, on
    purpose — the default behaviour is deliberate and must not drift. It is
    here so the next reader finds the limitation pinned by a test rather than
    only asserted by a docstring."""
    asker = _Asker(Decision(kind="deny", note="should never be consulted"))
    server = _server(asker, auto_allow=("Read",))

    for path in ("/etc/passwd", "/etc/shadow", "~/.ssh/id_rsa"):
        assert (await _decide(server, "Read", file_path=path))["behavior"] == "allow"
    assert asker.seen == []

    assert (await _decide(server, "Write", file_path="/etc/passwd"))["behavior"] == "deny"
    assert [r.tool_call["name"] for r in asker.seen] == ["Write"]


@pytest.mark.asyncio
async def test_an_argument_aware_predicate_narrows_an_auto_allowed_tool() -> None:
    """THE FIX. ``auto_allow_when`` lets an operator allow-list the safe
    OPERATIONS they thought they were allow-listing: the same ``Read`` is
    auto-allowed under ``/workspace`` and routed to the reviewer outside it,
    with the arguments attached so the person can see what they are judging."""
    asker = _Asker(Decision(kind="deny", note="out of scope"))
    server = _server(
        asker,
        auto_allow=("Read",),
        auto_allow_when=lambda tool, args: str(args.get("file_path", "")).startswith(
            "/workspace/"
        ),
    )

    assert (await _decide(server, "Read", file_path="/workspace/main.py"))["behavior"] == "allow"
    assert asker.seen == []

    out = await _decide(server, "Read", file_path="/etc/passwd")
    assert out["behavior"] == "deny" and out["message"] == "out of scope"
    assert asker.seen[0].tool_call == {"name": "Read", "arguments": {"file_path": "/etc/passwd"}}


@pytest.mark.asyncio
async def test_the_predicate_can_only_narrow_never_broaden() -> None:
    """The reason this is safe to add to a security-adjacent seam: the name
    list is checked FIRST, so a predicate is a filter over an already-approved
    set and never a second way in. A predicate that says yes to everything
    still cannot auto-allow a tool nobody listed."""
    asker = _Asker(Decision(kind="deny", note="reviewer still decides"))
    server = _server(asker, auto_allow=("Read",), auto_allow_when=lambda tool, args: True)

    out = await _decide(server, "Bash", command="rm -rf /")
    assert out["behavior"] == "deny"
    assert [r.tool_call["name"] for r in asker.seen] == ["Bash"]


@pytest.mark.asyncio
async def test_a_raising_predicate_falls_through_to_the_reviewer() -> None:
    """A broken narrowing rule must not widen the grant it exists to narrow,
    and it must not break the prompt either — ``_decide`` never raises, because
    an exception reaches the CLI as a broken permission system rather than a
    decision. So a raise lands on the human, who was told this tool was routine
    and can now see the call."""
    asker = _Asker(Decision(kind="approve"))

    def _broken(tool: str, args: dict[str, Any]) -> bool:
        raise RuntimeError("the policy service is down")

    server = _server(asker, auto_allow=("Read",), auto_allow_when=_broken)

    out = await _decide(server, "Read", file_path="/workspace/main.py")
    assert out == {"behavior": "allow", "updatedInput": {"file_path": "/workspace/main.py"}}
    assert [r.tool_call["name"] for r in asker.seen] == ["Read"]  # the HUMAN allowed it


@pytest.mark.asyncio
async def test_a_raising_predicate_still_denies_when_the_reviewer_denies() -> None:
    """The other half of the fail-closed claim: falling through to the reviewer
    is not a soft allow. Same broken predicate, a reviewer who says no, and the
    call is denied."""
    def _broken(tool: str, args: dict[str, Any]) -> bool:
        raise RuntimeError("the policy service is down")

    server = _server(
        _Asker(Decision(kind="deny", note="no")), auto_allow=("Read",), auto_allow_when=_broken
    )
    assert (await _decide(server, "Read", file_path="/x"))["behavior"] == "deny"


@pytest.mark.asyncio
async def test_a_narrowed_prompt_is_still_bounded_by_the_timeout() -> None:
    """POSITIVE CONTROL: the timeout stays SERVER-enforced on the path the
    predicate opens. A prompt the predicate pushes to the reviewer is an
    ordinary prompt — it does not bypass ``_ask``'s ``wait_for``, so a queue
    worker cannot be parked forever by narrowing a rule."""
    class _Slow:
        async def ask(self, request: Elicitation) -> Decision:
            await asyncio.sleep(10)
            return Decision(kind="approve")

    server = _server(
        _Slow(),
        timeout_s=0.05,
        auto_allow=("Read",),
        auto_allow_when=lambda tool, args: False,  # narrow everything to the human
    )
    out = await _decide(server, "Read", file_path="/etc/passwd")
    assert out["behavior"] == "deny" and "0.05s" in out["message"]


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
    assert kwargs["permission_mode"] == "default"
    assert kwargs["setting_sources"] == ()
    config = json.loads(kwargs["mcp_config"][0])
    # The credential travels IN the document the caller is handed, which is the
    # whole reason ``cli_kwargs()`` exists: a caller assembling an Authorization
    # header by hand is a caller who will hardcode one.
    entry = config["mcpServers"][SERVER_NAME]
    assert entry["type"] == "http"
    assert entry["url"] == "http://127.0.0.1:51234/mcp"
    assert entry["headers"] == server.auth_headers
    assert entry["headers"]["Authorization"].startswith("Bearer ")
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


# ── 8. autonomy is honoured, and it is the SAME policy ─────────────────────
#
# ``HumanGate``'s claim is that autonomy is set once per run and honoured
# uniformly by every pattern. A CLI path that decided independently would break
# that claim in the one place it is hardest to notice: these prompts are
# answered by a server the operator wired once and then stopped looking at, so
# a divergence surfaces as "the reviewer got asked about things they thought
# were automatic" — or, far worse, the reverse — with nothing in the run
# pointing at the cause.


class _NeverAsk:
    """An ``Asker`` that fails the test if it is consulted at all.

    Asserting on the ANSWER is not enough: under ``autonomy="auto"`` an
    implementation that asks a human and happens to get an approval produces
    exactly the same allow. The claim is that no human was in the loop, so the
    test has to be about the call that did not happen.
    """

    def __init__(self) -> None:
        self.seen: list[Elicitation] = []

    async def ask(self, request: Elicitation) -> Decision:
        self.seen.append(request)
        raise AssertionError("the Asker must not be consulted under autonomy='auto'")


@pytest.mark.asyncio
async def test_autonomy_auto_answers_without_consulting_the_asker() -> None:
    """AUTO means "gate only what a tool author explicitly required", and a CLI
    permission prompt carries no tool author — so under AUTO there is nothing
    to gate and nobody to ask."""
    asker = _NeverAsk()
    out = await _decide(_server(asker, autonomy="auto"), "Bash", command="rm -rf /tmp/x")

    assert out == {"behavior": "allow", "updatedInput": {"command": "rm -rf /tmp/x"}}
    assert asker.seen == [], "the answer was right; the human should not have been in it"


@pytest.mark.parametrize("autonomy", ["gated", "manual"])
@pytest.mark.asyncio
async def test_a_gating_autonomy_reaches_the_asker_and_the_answer_reaches_the_cli(
    autonomy: str,
) -> None:
    """The other half: under a gating tier the prompt goes to the person AND
    what they said is what the CLI is told. A gate that asks and then ignores
    the answer is theatre."""
    approving = _Asker(Decision(kind="approve", actor="reviewer"))
    out = await _decide(_server(approving, autonomy=autonomy), "Bash", command="ls")
    assert out["behavior"] == "allow"
    assert [r.tool_call["name"] for r in approving.seen] == ["Bash"]

    denying = _Asker(Decision(kind="deny", note="not that one"))
    out = await _decide(_server(denying, autonomy=autonomy), "Bash", command="ls")
    assert out == {"behavior": "deny", "message": "not that one"}
    assert len(denying.seen) == 1


@pytest.mark.parametrize("autonomy", ["auto", "gated", "manual"])
@pytest.mark.asyncio
async def test_the_tiers_are_should_gates_and_not_a_second_table(autonomy: str) -> None:
    """The anti-drift test. It asserts the server's behaviour against
    ``should_gate`` itself rather than against three hand-written
    expectations, so a change to the shared policy either moves this server
    with it or fails here — which is the whole reason for not re-deriving the
    tiers locally.

    ``key_step=True`` is the mapping, and it is the load-bearing choice: the
    CLI only prompts for calls IT considers consequential, so every prompt that
    reaches this server is a key step by construction. ``requires_approval=True``
    is the tempting alternative reading and it is wrong — it gates under every
    tier, which makes ``autonomy`` decorative.
    """
    asker = _Asker(Decision(kind="approve"))
    await _decide(_server(asker, autonomy=autonomy), "Bash", command="ls")

    assert bool(asker.seen) is should_gate(autonomy, requires_approval=False, key_step=True)


@pytest.mark.asyncio
async def test_autonomy_defaults_to_gated_which_is_the_behaviour_callers_had() -> None:
    """A default of ``"auto"`` would turn every existing caller's approval gate
    into a rubber stamp on upgrade — the worst possible direction for a
    security default to move by accident. ``"gated"`` is exactly what this
    server did before it knew about autonomy: ask about everything that is not
    auto-allowed."""
    assert ApprovalServer(asker=_Asker()).autonomy == "gated"

    asker = _Asker(Decision(kind="deny", note="no"))
    assert (await _decide(_server(asker)))["behavior"] == "deny"
    assert len(asker.seen) == 1


@pytest.mark.asyncio
async def test_autonomy_is_read_per_request_so_tightening_mid_run_takes_effect() -> None:
    """Read PER REQUEST, not snapshotted at construction — documented, and
    pinned here because the alternative is the worst kind of security control:
    one that READS as tightened while behaving as it did before. A supervisor
    escalating after seeing something must affect the next prompt, not the next
    run."""
    asker = _Asker(Decision(kind="approve"))
    server = _server(asker, autonomy="auto")

    assert (await _decide(server, "Bash"))["behavior"] == "allow"
    assert asker.seen == []

    server.autonomy = "manual"  # the supervisor tightens mid-run

    assert (await _decide(server, "Bash"))["behavior"] == "allow"
    assert len(asker.seen) == 1, "the tightened tier must apply to the next prompt"


def test_an_unknown_tier_is_refused_at_construction() -> None:
    """The spec that asked for this work wrote the tier as ``autonomy="ask"``.
    The real literals are ``auto|gated|manual``, and that typo is not cosmetic:
    ``should_gate`` falls through an unrecognised tier to its AUTO branch, so
    ``autonomy="ask"`` would have allow-listed EVERYTHING, silently. A gate
    must not fail open on a spelling mistake."""
    with pytest.raises(ValueError, match="auto"):
        ApprovalServer(asker=_Asker(), autonomy="ask")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_unknown_tier_assigned_after_construction_fails_closed() -> None:
    """Belt and braces, and the reason the constructor check is not enough on
    its own: ``autonomy`` is a mutable field read per request, so a typo can
    still arrive after the constructor has had its say. At decision time an
    unrecognised tier is treated as MANUAL — the strictest — because the one
    thing it must not do is what ``should_gate`` alone would do with it, which
    is allow the call."""
    asker = _Asker(Decision(kind="deny", note="reviewer says no"))
    server = _server(asker, autonomy="auto")
    server.autonomy = "ask"  # type: ignore[assignment]

    out = await _decide(server, "Bash", command="rm -rf /")

    assert out["behavior"] == "deny"
    assert [r.tool_call["name"] for r in asker.seen] == ["Bash"]


# ── 9. a server that cannot answer says so at construction ─────────────────


@pytest.mark.parametrize("autonomy", ["gated", "manual"])
def test_no_asker_plus_a_gating_autonomy_fails_at_construction(autonomy: str) -> None:
    """Not at the first prompt. Before this you could build a server that
    cannot possibly answer and find out only once a run was in flight and a
    person was waiting — at which point the failure is a denied tool call in
    someone else's log rather than a stack trace in the wiring code that
    caused it."""
    with pytest.raises(ValueError, match="asker"):
        ApprovalServer(autonomy=autonomy)  # type: ignore[arg-type]


def test_no_asker_is_legitimate_under_auto() -> None:
    """The check is derived from the policy rather than bolted on beside it:
    under AUTO nothing reaches a human, so having no human transport is a
    coherent configuration, not an oversight. A blanket "asker is required"
    would have banned it."""
    assert ApprovalServer(autonomy="auto").asker is None


@pytest.mark.asyncio
async def test_no_asker_answers_under_auto_without_touching_the_transport() -> None:
    out = await _decide(ApprovalServer(autonomy="auto"), "Bash", command="ls")
    assert out == {"behavior": "allow", "updatedInput": {"command": "ls"}}


@pytest.mark.asyncio
async def test_tightening_past_the_constructor_check_denies_rather_than_crashing() -> None:
    """The residual case the constructor cannot cover: legal at construction
    (AUTO needs no asker), tightened afterwards. ``_decide`` must never raise,
    so this is a DENY naming the missing transport — the operator reads the
    cause in the model's own refusal instead of an ``AttributeError`` inside an
    MCP request handler."""
    server = ApprovalServer(autonomy="auto")
    server.autonomy = "manual"

    out = await _decide(server, "Bash", command="ls")

    assert out["behavior"] == "deny"
    assert "no reviewer is wired" in out["message"]
    assert "manual" in out["message"], "the message must name the tier that caused it"


@pytest.mark.parametrize("timeout_s", [0, 0.0, -1.0])
def test_a_non_positive_timeout_is_refused_at_construction(timeout_s: float) -> None:
    """``timeout_s=0`` is the socket-API reflex for "no timeout" and here it
    means the exact opposite: ``asyncio.wait_for`` with a non-positive timeout
    expires before the ``Asker`` is ever scheduled, so EVERY prompt is denied
    without anyone being consulted. That is ``dontAsk`` — precisely the failure
    mode this module exists to remove — reached by a plausible typo. ``None``
    is how you say "wait forever"."""
    with pytest.raises(ValueError, match="timeout_s"):
        ApprovalServer(asker=_Asker(), timeout_s=timeout_s)


# ── 10. auto_allow meets autonomy ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_manual_beats_auto_allow() -> None:
    """THE security call, argued in the source: MANUAL wins and ``auto_allow``
    is inert under it.

    The two knobs are set in different places at different times —
    ``auto_allow`` at wiring time by whoever built the service, ``autonomy``
    per run by whoever launched THIS run. If a static name list could carve
    exceptions out of MANUAL, "manual" would silently mean "manual except the
    forty Reads" and the operator who selected it would have nothing in the run
    to tell them otherwise. The override only ever moves in the safe direction:
    more prompts, never fewer."""
    asker = _Asker(Decision(kind="deny", note="I review everything"))
    server = _server(asker, auto_allow=("Read",), autonomy="manual")

    out = await _decide(server, "Read", file_path="/workspace/main.py")

    assert out["behavior"] == "deny"
    assert [r.tool_call["name"] for r in asker.seen] == ["Read"]


@pytest.mark.asyncio
async def test_auto_allow_still_works_under_gated() -> None:
    """POSITIVE CONTROL: MANUAL disabling ``auto_allow`` must not be
    implemented as "``auto_allow`` is off whenever autonomy is set". Under
    GATED — the default, and today's behaviour — the habituation defence is
    untouched."""
    asker = _Asker(Decision(kind="deny", note="should never be consulted"))
    server = _server(asker, auto_allow=("Read",), autonomy="gated")

    assert (await _decide(server, "Read", file_path="/x"))["behavior"] == "allow"
    assert asker.seen == []


@pytest.mark.asyncio
async def test_auto_allow_is_moot_under_auto() -> None:
    """Under AUTO the name list decides nothing, because everything is already
    allowed without a human. Pinned so nobody reads the MANUAL rule above and
    "fixes" the AUTO path into consulting the list."""
    server = ApprovalServer(autonomy="auto", auto_allow=("Read",))
    assert (await _decide(server, "Bash", command="rm -rf /"))["behavior"] == "allow"


@pytest.mark.asyncio
async def test_an_unknown_tier_also_disables_auto_allow() -> None:
    """Fail-closed means closed all the way: an unrecognised tier is MANUAL,
    and MANUAL ignores ``auto_allow``."""
    asker = _Asker(Decision(kind="deny", note="no"))
    server = _server(asker, auto_allow=("Read",), autonomy="gated")
    server.autonomy = "nonsense"  # type: ignore[assignment]

    assert (await _decide(server, "Read", file_path="/x"))["behavior"] == "deny"
    assert len(asker.seen) == 1


# ── 11. answers that are not answers ───────────────────────────────────────


@pytest.mark.asyncio
async def test_an_asker_that_returns_something_that_is_not_a_decision_denies() -> None:
    """The HITL API this replaced was ``dict[str, str]`` with ``"approve"`` as
    a bare string, so an ``Asker`` written from that memory returns a ``str``.
    That escaped ``_decide`` as an ``AttributeError`` — and an exception here is
    NOT a denial to the CLI, it is "the permission system is broken", which
    leaves the model retrying a call nobody approved. Fail closed instead, and
    name the type in the message so the author of the transport can see what
    they returned."""

    class _Stringly:
        async def ask(self, request: Elicitation) -> Any:
            return "approve"

    out = await _decide(_server(_Stringly()))

    assert out["behavior"] == "deny"
    assert "str" in out["message"]


@pytest.mark.asyncio
async def test_an_unknown_decision_kind_denies() -> None:
    """``Decision`` is a plain dataclass with no validation, so ``kind`` can be
    anything at all. Everything that is not an approval denies."""
    out = await _decide(_server(_Asker(Decision(kind="maybe"))))  # type: ignore[arg-type]
    assert out["behavior"] == "deny"


@pytest.mark.asyncio
async def test_a_modify_whose_value_is_not_a_dict_denies() -> None:
    """``modify`` means "run it with THESE arguments". A non-dict cannot be
    arguments, and the fail-open reading — approve with the originals — would
    run the exact call the reviewer was trying to edit."""
    out = await _decide(_server(_Asker(Decision(kind="modify", value="rm -rf /"))))
    assert out["behavior"] == "deny"


# ── 12. concurrency: the CLI asks about several tools at once ──────────────


@pytest.mark.asyncio
async def test_concurrent_prompts_each_get_their_own_identity() -> None:
    """The CLI can have several tool calls in flight, so several prompts land
    on one server at once. ``prompts_seen`` is the audit count and the
    ``Elicitation`` id is what a UI keys its pending cards on — a collision
    there shows up as one reviewer's answer applied to a different tool call,
    which is a security bug wearing a UI bug's clothes."""
    all_parked = asyncio.Event()  # every prompt is inside ask() at once
    release = asyncio.Event()  # ...and none of them answers until we say so

    class _Parked:
        def __init__(self) -> None:
            self.seen: list[Elicitation] = []

        async def ask(self, request: Elicitation) -> Decision:
            self.seen.append(request)
            if len(self.seen) == 24:
                all_parked.set()
            await release.wait()
            return Decision(kind="approve")

    asker = _Parked()
    server = _server(asker)
    pending = [asyncio.create_task(_decide(server, f"Tool{i}")) for i in range(24)]
    await asyncio.wait_for(all_parked.wait(), timeout=10)
    release.set()
    outs = await asyncio.gather(*pending)

    assert all(o["behavior"] == "allow" for o in outs)
    assert server.prompts_seen == 24
    assert len({r.id for r in asker.seen}) == 24


@pytest.mark.asyncio
async def test_the_deadline_still_fires_when_many_prompts_are_parked() -> None:
    """The timeout is per-prompt and server-enforced, so twenty-four parked
    reviewers are twenty-four independent deadlines rather than one queue that
    starves. Verified under load because a bound that only holds for a single
    outstanding prompt is not a bound a queue worker can rely on."""
    never_answered = asyncio.Event()  # deliberately never set

    class _Slow:
        async def ask(self, request: Elicitation) -> Decision:
            await never_answered.wait()
            return Decision(kind="approve")

    server = _server(_Slow(), timeout_s=0.05)
    outs = await asyncio.wait_for(
        asyncio.gather(*(_decide(server, f"Tool{i}") for i in range(24))), timeout=10
    )

    assert all(o["behavior"] == "deny" and "0.05" in o["message"] for o in outs)
    assert server.prompts_seen == 24


# ── 13. the loopback posture, asserted rather than described ───────────────


def test_the_default_bind_address_is_not_routable() -> None:
    """The server has no authentication, so the bind address IS the
    containment: anything that can reach the port can answer permission
    prompts on this agent's behalf. Asserted as a PROPERTY of the address —
    loopback, not global, not unspecified — rather than as the string
    ``"127.0.0.1"``, because the failure that matters is "someone changed the
    default to something reachable", and ``"0.0.0.0"`` is not equal to
    ``"127.0.0.1"`` either."""
    host = ipaddress.ip_address(ApprovalServer(asker=_Asker()).host)

    assert host.is_loopback
    assert not host.is_global
    assert not host.is_unspecified, "0.0.0.0 binds every interface on the box"


# ── 14. what the model actually sees ───────────────────────────────────────
#
# The claim is that a denial is a REFUSAL the model can adapt to, not a dead
# end. These drive the CLI's own stream parser over a canned transcript whose
# tool_result is the exact string ``_decide`` produced, joining the two halves:
# what the server returns is what lands in the run's event stream, and the run
# reaches a normal ``final`` rather than an error.


def _cli_events_for(denial_message: str) -> list[Any]:
    """A CLI transcript in which the denied tool call comes back carrying the
    server's own message, and the model then adapts."""
    lines = [
        _line({"type": "system", "subtype": "init", "session_id": "s", "cwd": "/tmp"}),
        _line(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu-1",
                            "name": "Write",
                            "input": {"file_path": "/etc/passwd"},
                        }
                    ]
                },
            }
        ),
        _line(
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu-1", "content": denial_message}
                    ]
                },
            }
        ),
        _line(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "I could not write that file."}]},
            }
        ),
        _line(
            {
                "type": "result",
                "session_id": "s",
                "duration_ms": 1,
                "total_cost_usd": 0.0,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ),
    ]
    proc = _FakeProcess(stdout_lines=lines)
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        cog = ClaudeCliCognition()
        agent = Agent(name="x", cognition=cog)

        async def _go() -> list[Any]:
            return [ev async for ev in cog.drive(agent, "write it", FakeCtx(), WorkingContext())]

        return asyncio.run(_go())


def test_a_denial_is_visible_in_the_stream_as_a_refusal() -> None:
    """A denied call is not a failed run. The reviewer's reason arrives as the
    tool's result, the model answers around it, and the run ends ``complete`` —
    which is what lets a service keep the turn instead of surfacing an error
    nobody can act on."""
    denial = json.loads(
        asyncio.run(
            _server(_Asker(Decision(kind="deny", note="that path is out of scope")))._decide(
                "Write", {"file_path": "/etc/passwd"}
            )
        )
    )["message"]

    events = _cli_events_for(denial)

    assert [e.tool_result for e in events if e.type == "tool_result"] == [
        "that path is out of scope"
    ]
    final = events[-1]
    assert final.type == "final"
    assert final.result.stop_reason == "complete"
    assert "could not write" in final.result.output


def test_an_expired_deadline_denies_records_and_lets_the_run_continue() -> None:
    """The deadline degrades, it does not abort. The prompt expires, the server
    counts it like any other prompt, the model is told WHY (so "the reviewer
    said no" and "nobody answered in 0.05s" stay distinguishable), and the run
    reaches a normal terminal event."""
    never_answered = asyncio.Event()

    class _Slow:
        async def ask(self, request: Elicitation) -> Decision:
            await never_answered.wait()
            return Decision(kind="approve")

    server = _server(_Slow(), timeout_s=0.05)
    out = json.loads(asyncio.run(server._decide("Write", {"file_path": "/etc/passwd"})))

    assert out["behavior"] == "deny"
    assert "0.05" in out["message"]
    assert server.prompts_seen == 1, "an expired prompt is still a prompt for the audit trail"

    events = _cli_events_for(out["message"])
    assert events[-1].result.stop_reason == "complete"
    assert [e.tool_result for e in events if e.type == "tool_result"] == [out["message"]]


# ── 15. against the real binary ─────────────────────────────────────────────
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


# ── 16. review pass: the ways this gate could still fail open ──────────────
#
# Everything below was found by attacking the shipped implementation, and each
# one is a path where the server ALLOWED a call, or refused to answer at all,
# without the reviewer being consulted. They are grouped here because they
# share a shape: a knob validated once at construction and then read from a
# mutable field, or an answer whose meaning this server decided locally
# instead of asking the shared predicate.


@pytest.mark.asyncio
async def test_a_value_decision_is_not_consent() -> None:
    """The divergence that matters most, because it is the one ``autonomy``
    was routed through ``should_gate`` to avoid: two gates in one codebase
    disagreeing about what a human said.

    ``Decision.approved`` — what ``ReActCognition`` gates on — is
    ``kind in ("approve", "modify")``. This server used to allow on
    ``kind in ("approve", "value")``, so the SAME ``Asker`` returning the SAME
    ``Decision`` was a refusal on agentkit's own tool loop and an approval
    here. And ``value`` is not an answer to this request at all: it goes out
    as ``kind="approval"`` with ``choices=("approve", "deny")``, so a
    transport that returns one is handing back whatever the person typed —
    including the word "no", which was being read as yes.
    """
    said_no = Decision(kind="value", value="no, absolutely not")
    assert not said_no.approved, "the shared predicate does not call this consent"

    out = await _decide(_server(_Asker(said_no)), "Bash", command="rm -rf /")

    assert out["behavior"] == "deny"
    # And it names the transport, not the reviewer: the person did not decline,
    # their Asker handed back a typed answer instead of a verdict, and only its
    # author can fix that — which they cannot if the message blames the human.
    assert "not consent" in out["message"]
    assert "declined" not in out["message"]


@pytest.mark.parametrize(
    "decision",
    [
        Decision(kind="approve"),
        Decision(kind="modify", value={"command": "ls"}),
        Decision(kind="deny"),
        Decision(kind="value", value="approve"),
        Decision(kind="expired"),
    ],
    ids=["approve", "modify", "deny", "value", "expired"],
)
@pytest.mark.asyncio
async def test_allowing_agrees_with_decision_approved_for_every_kind(
    decision: Decision,
) -> None:
    """The anti-drift test for the ANSWER, twin of
    ``test_the_tiers_are_should_gates_and_not_a_second_table`` for the TIER.
    Asserted against ``Decision.approved`` itself rather than a hand-written
    table, so a change to what agentkit counts as consent either moves this
    server with it or fails here."""
    out = await _decide(_server(_Asker(decision)), "Bash", command="ls")

    assert (out["behavior"] == "allow") is decision.approved


@pytest.mark.asyncio
async def test_a_modify_with_non_dict_arguments_says_that_is_what_happened() -> None:
    """``modify`` IS approved by the shared predicate, so the non-dict case is
    a deny this server has to produce itself. The message must not read as "a
    reviewer declined": they did not decline, they tried to EDIT the call and
    their transport sent something that cannot be arguments — a bug in the
    transport rather than a decision the model should adapt around."""
    out = await _decide(_server(_Asker(Decision(kind="modify", value="rm -rf /"))))

    assert out["behavior"] == "deny"
    assert "str" in out["message"] and "dict" in out["message"]


@pytest.mark.parametrize("timeout_s", [0, 0.0, -1.0])
@pytest.mark.asyncio
async def test_a_non_positive_timeout_assigned_after_construction_says_so(
    timeout_s: float,
) -> None:
    """``timeout_s`` gets the same belt-and-braces as ``autonomy``, and needs
    it for the same reason: it is a mutable public field, so the constructor's
    refusal is not the last word.

    ``asyncio.wait_for`` with a non-positive timeout expires before the
    ``Asker`` is scheduled, so every prompt is denied with nobody consulted —
    ``dontAsk``, reached by assignment instead of by argument. It still denies,
    because denying is the closed direction. What changed is the message: it
    used to say "no answer within 0s", which blames a reviewer who was never
    asked and sends whoever is debugging it at the transport."""
    asker = _Asker(Decision(kind="approve"))
    server = _server(asker, timeout_s=30)
    server.timeout_s = timeout_s

    out = await _decide(server, "Bash", command="ls")

    assert out["behavior"] == "deny"
    assert asker.seen == [], "nobody was consulted, so the message must not blame one"
    assert "misconfigured" in out["message"]
    assert "timeout_s" in out["message"]


def test_auto_allow_as_a_bare_string_is_refused_at_construction() -> None:
    """``auto_allow="Read"`` — the missing comma — is the quiet fail-OPEN, and
    it is quiet because the server keeps working.

    ``auto_allow`` is tested with ``in``, and ``in`` on a string is a substring
    test, so ``auto_allow="Read"`` auto-allows ``"R"``, ``"ea"`` and ``"Read"``
    alike: those calls are APPROVED with the reviewer never consulted. Every
    other refusal in ``__post_init__`` catches a server that cannot answer;
    this one catches a server that answers yes to things nobody listed."""
    with pytest.raises(ValueError, match="auto_allow"):
        ApprovalServer(asker=_Asker(), auto_allow="Read")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_unencodable_argument_denies_instead_of_breaking_the_gate() -> None:
    """``_decide`` promises never to raise, and ``json.dumps`` was the one line
    in it that could — on the two branches that were about to ALLOW, and
    outside every ``try``.

    An exception escaping an MCP request handler is not a denial to the CLI: it
    is "the permission system is broken", which leaves the model retrying a
    call nobody approved. So the one path that failed open was the one already
    saying yes. Checked on all three allow paths, since they are three separate
    ``return`` statements."""
    hostile = {"payload": object()}  # nothing json can encode

    auto = await ApprovalServer(autonomy="auto")._decide("Bash", hostile)
    listed = await _server(_Asker(), auto_allow=("Read",))._decide("Read", hostile)
    approved = await _server(_Asker(Decision(kind="approve")))._decide("Bash", hostile)

    for raw in (auto, listed, approved):
        assert json.loads(raw)["behavior"] == "deny"
        assert "could not be encoded" in json.loads(raw)["message"]


@pytest.mark.asyncio
async def test_a_nan_argument_denies_rather_than_emitting_invalid_json() -> None:
    """Python's ``json`` emits a bare ``NaN``, which is not JSON and which the
    CLI's own parser rejects. An allow that arrives as a parse error is the
    same broken gate by a slower route, so it is a deny here instead."""
    out = await _decide(ApprovalServer(autonomy="auto"), "Bash", ratio=float("nan"))

    assert out["behavior"] == "deny"


@pytest.mark.asyncio
async def test_prompts_are_counted_under_auto_where_nothing_else_records_them() -> None:
    """Survived a mutant that skipped the increment under ``autonomy="auto"``.

    Under AUTO there is no ``Asker``, no reviewer and no elicitation — so
    ``prompts_seen`` is the ONLY evidence that this server saw a tool call and
    allowed it. That is exactly the tier where an audit count silently stuck at
    zero is worth catching, and it was the tier no test covered."""
    server = ApprovalServer(autonomy="auto")

    await _decide(server, "Bash", command="ls")
    await _decide(server, "Write", file_path="/etc/passwd")

    assert server.prompts_seen == 2


@pytest.mark.asyncio
async def test_the_deadline_reaches_the_person_being_asked() -> None:
    """Survived a mutant that dropped ``deadline_s`` from the ``Elicitation``.

    ``timeout_s`` is enforced server-side either way, so losing this is
    invisible to the CLI and very visible to the human: their pending card
    shows no deadline, they answer at their own pace, and the answer is thrown
    away by a timeout they were never shown."""
    asker = _Asker(Decision(kind="approve"))
    await _decide(_server(asker, timeout_s=30.0))
    assert asker.seen[0].deadline_s == 30.0

    forever = _Asker(Decision(kind="approve"))
    await _decide(_server(forever))
    assert forever.seen[0].deadline_s is None, "None means wait, and must stay None"


# ── 17. review pass: a listener that never listened ────────────────────────
#
# `start()` had a wait loop whose stated job was "the bind failed — surface
# THAT, not a hang". Measured against an occupied port, it surfaced nothing a
# caller could catch, left the object wedged, and turned the retry into a
# silent success. The cause is that uvicorn signals a failed bind with
# `sys.exit(3)`, and asyncio treats `SystemExit` from a Task as "stop the
# loop" rather than as a task failure.


@pytest.fixture
def occupied_port() -> Any:
    """A port with a listener already on it, so the next bind must fail."""
    hog = socket.socket()
    hog.bind(("127.0.0.1", 0))
    hog.listen(1)
    try:
        yield hog.getsockname()[1]
    finally:
        hog.close()


@pytest.mark.asyncio
async def test_a_failed_bind_is_an_error_the_caller_can_catch(occupied_port: int) -> None:
    """``except Exception`` around the documented wiring used to catch NOTHING.

    uvicorn calls ``sys.exit(3)`` when it cannot bind; asyncio re-raises a
    Task's ``SystemExit`` into the event loop, which cancelled ``start()``
    before it could reach its own failure branch. The caller got a bare
    ``CancelledError`` — a ``BaseException`` — so the recommended
    ``async with ApprovalServer(...)`` unwound the process rather than failing
    the run. In the FastAPI recipe on this seam, that is the whole worker.

    The type is now ``OSError`` rather than a ``RuntimeError`` converted from
    uvicorn's ``SystemExit``: ``LoopbackMcpTransport`` reserves the socket
    before uvicorn is involved, so the real errno reaches the caller. The
    endpoint is re-attached to the message because a bare
    ``OSError: [Errno 48] Address already in use`` does not carry it, and a
    process running several of these servers would get one indistinguishable
    line per collision — which is exactly what the assertion below guards."""
    server = ApprovalServer(asker=_Asker(), port=occupied_port)

    with pytest.raises(OSError) as caught:
        await server.start()

    assert str(occupied_port) in str(caught.value), "say which port, there may be many"


@pytest.mark.asyncio
async def test_a_failed_bind_does_not_wedge_the_server(occupied_port: int) -> None:
    """The silent-success half, and the more dangerous one.

    ``start()`` is idempotent via ``if self._task is not None: return``. A
    failed start used to leave ``_task`` set, so a caller who retried — or who
    caught the first failure and carried on — got a plain return from a server
    that was NOT listening. Its ``url`` then went to the CLI, which failed
    thirty seconds later with a startup timeout: the exact failure the wait
    loop exists to prevent, reached through the path meant to report it.

    The exception type moved from ``RuntimeError`` to ``OSError`` when the
    transport was extracted, and that is an improvement rather than a
    relabelling. ``ApprovalServer`` used to let uvicorn attempt the bind and
    convert its ``sys.exit(3)`` into a ``RuntimeError`` inside the serve task.
    ``LoopbackMcpTransport`` RESERVES the socket before uvicorn ever sees it,
    so the genuine ``OSError: [Errno 48] Address already in use`` arrives in
    the caller's own frame with the errno and the port intact."""
    server = ApprovalServer(asker=_Asker(), port=occupied_port)
    with pytest.raises(OSError):
        await server.start()

    with pytest.raises(OSError):
        await server.start()  # must NOT be a silent no-op

    await server.stop()  # and cleanup must not raise on top of the real error


@pytest.mark.asyncio
async def test_stopping_a_server_that_never_started_does_not_raise(
    occupied_port: int,
) -> None:
    """``stop()`` suppressed ``Exception``, and uvicorn's failed bind is a
    ``SystemExit``. So ``__aexit__`` raised ``SystemExit(3)`` while unwinding
    whatever had already gone wrong, replacing the real error with an exit
    code on the way out of the ``async with``.

    The failure is now an ``OSError`` from the pre-bind rather than a
    ``RuntimeError`` from a converted ``SystemExit`` — see
    ``test_a_failed_bind_does_not_wedge_the_server``. What this test pins is
    unchanged: ``stop()`` on a server that never started must not raise on top
    of the real error."""
    async with contextlib.AsyncExitStack() as stack:
        server = ApprovalServer(asker=_Asker(), port=occupied_port)
        stack.push_async_callback(server.stop)
        with pytest.raises(OSError):
            await server.start()
    # leaving the stack ran stop(); reaching here at all is the assertion

# ── 18. a permission prompt is a decision, not a tick on a counter ─────────


class _RecordingCtx:
    """A ctx whose ``emit`` records. The run's observer is the only place a
    decision made inside this server can reach a UI or a log shipper."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, Any]] = []

    async def emit(
        self,
        kind: str,
        render: str = "",
        *,
        payload: Any = None,
        agent: str = "",
        parent_id: Any = None,
    ) -> None:
        del agent, parent_id
        self.events.append((kind, render, payload))


class _ExplodingCtx:
    """An observer that raises. A refusal that happened matters more than a
    record of it that did not."""

    async def emit(self, *_a: Any, **_kw: Any) -> None:
        raise RuntimeError("the bus is closed")


@pytest.mark.asyncio
async def test_a_human_approval_is_recorded_as_a_decision() -> None:
    """The count that existed could not answer a single question worth asking:
    not what was allowed, not who allowed it, not which approval preceded the
    write we are looking at. The record has to carry the call itself."""
    asker = _Asker(Decision(kind="approve", actor="ops"))
    server = _server(asker)

    out = await _decide(server, "Write", file_path="/tmp/out.txt")

    assert out["behavior"] == "allow"
    (rec,) = server.decisions
    assert rec.tool == "Write"
    assert rec.arguments == {"file_path": "/tmp/out.txt"}
    assert rec.allowed is True
    assert rec.source == "asker"
    assert rec.asked is True
    assert rec.at  # stamped, not blank


@pytest.mark.asyncio
async def test_an_auto_allow_hit_records_its_source_and_asks_nobody() -> None:
    """"Allowed because a person said so" and "allowed because the operator put
    this tool on a list" are the two facts an auditor most needs apart, and a
    boolean cannot hold them. The assertion on ``asker.seen`` is the point:
    ``asked=False`` has to mean nobody was interrupted."""
    asker = _Asker(Decision(kind="deny"))
    server = _server(asker, auto_allow=("Read",))

    out = await _decide(server, "Read", file_path="/etc/hosts")

    assert out["behavior"] == "allow"
    assert asker.seen == []
    (rec,) = server.decisions
    assert (rec.source, rec.asked, rec.allowed) == ("auto_allow", False, True)
    assert rec.arguments == {"file_path": "/etc/hosts"}


@pytest.mark.asyncio
async def test_an_auto_tier_decision_is_still_recorded() -> None:
    """AUTO is the tier with no reviewer, no elicitation and no answer to read,
    so the record is the ONLY evidence the server saw the call at all — and
    ``source`` is what keeps "the tier does not gate this" from reading as "a
    person approved it"."""
    asker = _Asker(Decision(kind="deny"))
    server = _server(asker, autonomy="auto")

    out = await _decide(server, "Bash", command="rm -rf /tmp/x")

    assert out["behavior"] == "allow"
    assert asker.seen == []
    (rec,) = server.decisions
    assert (rec.source, rec.asked, rec.allowed) == ("autonomy", False, True)


@pytest.mark.asyncio
async def test_a_timeout_is_distinguishable_from_a_human_refusal() -> None:
    """This is the whole argument for ``source`` existing. Both rows deny, both
    carry a reason, and nothing else on the record tells them apart — but "a
    reviewer said no" and "nobody was there" call for opposite responses from
    whoever reads the trail."""
    never = asyncio.Event()  # deliberately never set

    class _Slow:
        async def ask(self, request: Elicitation) -> Decision:
            await never.wait()
            return Decision(kind="approve")

    timed_out = _server(_Slow(), timeout_s=0.05)
    refused = _server(_Asker(Decision(kind="deny", note="not that path")))

    await _decide(timed_out, "Write", file_path="/etc/passwd")
    await _decide(refused, "Write", file_path="/etc/passwd")

    (expired,) = timed_out.decisions
    (denied,) = refused.decisions
    assert expired.allowed is denied.allowed is False
    assert expired.source == "timeout"
    assert denied.source == "asker"
    # ...and a prompt that reached a person before expiring says so.
    assert expired.asked is True


@pytest.mark.asyncio
async def test_an_asker_that_raises_records_an_error_and_the_session_survives() -> None:
    """An ``error`` must not read as a policy decision — nobody decided this.
    And the next prompt has to be answered, because a permission gate that
    stops answering is the CLI reporting a broken permission system."""
    calls: list[str] = []

    class _Flaky:
        async def ask(self, request: Elicitation) -> Decision:
            calls.append(request.prompt)
            if len(calls) == 1:
                raise RuntimeError("the queue went away")
            return Decision(kind="approve")

    server = _server(_Flaky())

    first = await _decide(server, "Bash", command="ls")
    second = await _decide(server, "Bash", command="pwd")

    assert first["behavior"] == "deny"
    assert second["behavior"] == "allow"
    broke, ok = server.decisions
    assert (broke.source, broke.allowed, broke.asked) == ("error", False, True)
    assert "RuntimeError" in broke.reason
    assert (ok.source, ok.allowed) == ("asker", True)


@pytest.mark.asyncio
async def test_the_recorded_reason_is_the_text_the_model_receives() -> None:
    """A record whose reason differs from the refusal the model was given
    documents a run that did not happen."""
    server = _server(_Asker(Decision(kind="deny", note="that path is out of scope")))

    out = await _decide(server, "Write", file_path="/etc/passwd")

    (rec,) = server.decisions
    assert rec.reason == out["message"] == "that path is out of scope"


@pytest.mark.asyncio
async def test_decisions_are_oldest_first_and_agree_with_prompts_seen() -> None:
    """``prompts_seen`` is existing public API and stays. The two must not be
    able to disagree, or the count becomes a second, quieter source of truth."""
    server = _server(_Asker(Decision(kind="approve")))

    for tool in ("Read", "Write", "Bash"):
        await _decide(server, tool)

    assert [d.tool for d in server.decisions] == ["Read", "Write", "Bash"]
    assert len(server.decisions) == server.prompts_seen == 3


@pytest.mark.asyncio
async def test_the_decisions_tuple_is_a_snapshot_not_the_live_list() -> None:
    """A caller holding the trail must not be able to edit it, and must not see
    it grow under them while they iterate."""
    server = _server(_Asker(Decision(kind="approve")))
    await _decide(server, "Read")
    snapshot = server.decisions
    await _decide(server, "Write")

    assert len(snapshot) == 1
    assert len(server.decisions) == 2


@pytest.mark.asyncio
async def test_a_recorded_decision_cannot_be_edited_after_the_fact() -> None:
    """Frozen for the same reason ``HookDecision`` is: a decision that can be
    rewritten is not an audit trail."""
    server = _server(_Asker(Decision(kind="approve")))
    await _decide(server, "Write", file_path="/tmp/a")
    (rec,) = server.decisions

    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError is not public
        rec.allowed = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        rec.arguments["file_path"] = "/etc/passwd"  # type: ignore[index]


@pytest.mark.asyncio
async def test_a_ctx_with_no_emit_records_normally_and_raises_nothing() -> None:
    """``ctx`` is optional and a structural stub predating it is legal, so the
    observer is reached through ``getattr``. Recording must not depend on it."""
    for ctx in (None, object()):
        server = _server(_Asker(Decision(kind="approve")), ctx=ctx)
        out = await _decide(server, "Read")
        assert out["behavior"] == "allow"
        assert len(server.decisions) == 1


@pytest.mark.asyncio
async def test_the_decision_reaches_the_run_observer() -> None:
    """``gate.check`` is the kind ``ReActCognition`` and ``elicit`` already
    stamp for this event, so a consumer assembling "every point a person took
    responsibility in this run" does not have to know which path produced it."""
    ctx = _RecordingCtx()
    server = _server(_Asker(Decision(kind="deny", note="no")), ctx=ctx, agent="dev")

    await _decide(server, "Write", file_path="/etc/passwd")

    (kind, render, payload) = ctx.events[0]
    assert kind == "gate.check"
    assert "Write" in render
    assert payload["allowed"] is False
    assert payload["source"] == "asker"
    assert payload["reason"] == "no"


@pytest.mark.asyncio
async def test_the_observer_payload_carries_no_arguments() -> None:
    """The in-memory record is complete and unredacted; the broadcast is
    deliberately not the record. The observer stream fans out to logs and UIs
    the caller may not own, and the arguments are the half most likely to carry
    a token — so the trail keeps them and the event does not."""
    ctx = _RecordingCtx()
    server = _server(_Asker(Decision(kind="approve")), ctx=ctx)

    await _decide(server, "Bash", command="curl -H 'Authorization: Bearer sk-live'")

    (_kind, render, payload) = ctx.events[0]
    assert "sk-live" not in render
    assert "sk-live" not in json.dumps(payload)
    assert "sk-live" in server.decisions[0].arguments["command"]


@pytest.mark.asyncio
async def test_an_observer_that_raises_does_not_lose_the_decision() -> None:
    """The append comes first and the emit is best-effort — the same order
    ``HookSettings._emit`` keeps, for the same reason."""
    server = _server(_Asker(Decision(kind="approve")), ctx=_ExplodingCtx())

    out = await _decide(server, "Read")

    assert out["behavior"] == "allow"
    assert len(server.decisions) == 1


@pytest.mark.asyncio
async def test_a_modify_records_the_arguments_that_actually_ran() -> None:
    """The CLI runs ``updatedInput`` and does not tell the model it changed, so
    the edited arguments exist NOWHERE else in the run. Recording the request
    instead would name a call that never happened while the one that did went
    unrecorded — worse than no trail, because it reads like one."""
    edited = {"file_path": "/tmp/sandbox/out.txt", "content": "hello"}
    server = _server(_Asker(Decision(kind="modify", value=edited)))

    out = await _decide(server, "Write", file_path="/etc/out.txt", content="hello")

    assert out["updatedInput"] == edited
    (rec,) = server.decisions
    assert rec.arguments == edited
    assert (rec.allowed, rec.source, rec.asked) == (True, "asker", True)


@pytest.mark.asyncio
async def test_a_refusal_records_the_arguments_that_were_refused() -> None:
    """A deny has no ``updatedInput``; what was refused is what was asked."""
    server = _server(_Asker(Decision(kind="deny")))

    await _decide(server, "Write", file_path="/etc/passwd")

    (rec,) = server.decisions
    assert rec.arguments == {"file_path": "/etc/passwd"}


@pytest.mark.asyncio
async def test_arguments_that_are_not_json_serialisable_are_still_recorded() -> None:
    """Dropping the field is not an option: a tool name says ``Write``, and a
    decision has to have been about ``Write`` of a particular path."""

    class _Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    server = _server(_Asker(Decision(kind="deny", note="no")))

    await _decide(server, "Bash", command=_Opaque(), cwd="/srv")

    (rec,) = server.decisions
    assert set(rec.arguments) == {"command", "cwd"}
    assert rec.arguments["cwd"] == "/srv"


@pytest.mark.asyncio
async def test_enormous_arguments_are_truncated_rather_than_dropped() -> None:
    """``Write`` carries the whole file body, and a server that held every
    prompt for the life of a run would grow with the bytes the agent wrote."""
    server = _server(_Asker(Decision(kind="approve")))

    await _decide(server, "Write", file_path="/tmp/big", content="x" * 200_000)

    (rec,) = server.decisions
    assert rec.arguments["file_path"] == "/tmp/big"  # small values survive intact
    recorded = rec.arguments["content"]
    assert len(recorded) < _MAX_ARGUMENT_CHARS + 200
    assert "truncated" in recorded
    assert "200002" in recorded  # the true rendered size is named, not hidden


@pytest.mark.asyncio
async def test_an_allow_the_encoder_refuses_is_recorded_as_an_error() -> None:
    """``_allow`` degrades to a deny when the arguments cannot be encoded. The
    branch intended an allow and the CLI got a refusal, and filing that under
    ``autonomy`` would send an auditor looking for a tier that never denied
    anything."""
    server = ApprovalServer(autonomy="auto")

    out = await _decide(server, "Bash", command=float("nan"))

    assert out["behavior"] == "deny"
    (rec,) = server.decisions
    assert (rec.source, rec.allowed) == ("error", False)


@pytest.mark.asyncio
async def test_a_server_with_no_reviewer_records_the_error_it_denies_with() -> None:
    """Reachable only when ``autonomy`` is tightened after construction, which
    is exactly the case where the trail is how anyone finds out."""
    server = ApprovalServer(autonomy="auto")
    server.autonomy = "gated"

    out = await _decide(server, "Bash", command="ls")

    assert out["behavior"] == "deny"
    (rec,) = server.decisions
    assert (rec.source, rec.asked, rec.allowed) == ("error", False, False)
    assert rec.reason == out["message"]


@pytest.mark.asyncio
async def test_a_misconfigured_timeout_records_that_nobody_was_asked() -> None:
    """``asked`` is NOT ``source == "asker"`` and it is not derivable from the
    verdict. ``timeout_s=0`` expires before the transport is scheduled, so the
    prompt cost a person nothing — while an ordinary expiry cost them the
    interruption. Same ``source``, different ``asked``."""
    asker = _Asker(Decision(kind="approve"))
    server = _server(asker)
    server.timeout_s = 0  # refused at construction; assignable afterwards

    out = await _decide(server, "Bash", command="ls")

    assert out["behavior"] == "deny"
    (rec,) = server.decisions
    assert (rec.source, rec.asked) == ("timeout", False)
    assert asker.seen == []


@pytest.mark.asyncio
async def test_an_answer_that_is_not_a_decision_records_an_error() -> None:
    """The HITL API this replaced was ``dict[str, str]`` with ``"approve"`` as
    a bare string, so an ``Asker`` written from memory returns a ``str``. That
    is a transport bug, not a refusal, and the trail has to say which."""

    class _Legacy:
        async def ask(self, request: Elicitation) -> Any:
            return "approve"

    server = _server(_Legacy())

    out = await _decide(server, "Bash", command="ls")

    assert out["behavior"] == "deny"
    (rec,) = server.decisions
    assert (rec.source, rec.allowed, rec.asked) == ("error", False, True)
    assert "str" in rec.reason


@pytest.mark.asyncio
async def test_the_timestamp_comes_from_the_run_clock() -> None:
    """A replayed or fake-clocked run must not stamp its approval trail with
    real wall-clock times, or it stops lining up with the checkpoints beside
    it. ``clock`` is the same ``ClockPort`` seam ``Checkpointer`` takes."""
    server = _server(_Asker(Decision(kind="approve")), clock=FakeClock(start=1_700_000_000.0))

    await _decide(server, "Read")

    (rec,) = server.decisions
    assert rec.at == "2023-11-14T22:13:20+00:00"


@pytest.mark.asyncio
async def test_a_broken_clock_does_not_take_down_the_decision() -> None:
    """A decision that already happened outranks a timestamp for it."""

    class _BrokenClock:
        def now(self) -> float:
            raise RuntimeError("no clock")

        async def sleep(self, seconds: float) -> None:
            return None

    server = _server(_Asker(Decision(kind="approve")), clock=_BrokenClock())

    out = await _decide(server, "Read")

    assert out["behavior"] == "allow"
    (rec,) = server.decisions
    assert rec.at == ""


@pytest.mark.asyncio
async def test_concurrent_prompts_are_ordered_by_when_they_were_decided() -> None:
    """The CLI can ask about several tools at once, so "oldest first" has to
    mean something specific. It means DECIDED-first: a record exists when there
    is a verdict to record, which is also the order an operator reading the
    trail alongside the CLI's own output sees them resolve. Arrival order lives
    on the ``Elicitation`` id, which is stamped from ``prompts_seen``."""
    gates = {name: asyncio.Event() for name in ("Read", "Write", "Bash")}

    class _Gated:
        async def ask(self, request: Elicitation) -> Decision:
            await gates[request.tool_call["name"]].wait()
            return Decision(kind="approve")

    server = _server(_Gated())
    pending = [asyncio.create_task(_decide(server, name)) for name in gates]
    await asyncio.sleep(0.01)  # let all three park inside ask()

    for name in ("Bash", "Read", "Write"):  # answered out of arrival order
        gates[name].set()
        await asyncio.sleep(0.01)
    await asyncio.gather(*pending)

    assert [d.tool for d in server.decisions] == ["Bash", "Read", "Write"]
    assert len(server.decisions) == server.prompts_seen == 3


# ── 19. what the record still could not tell apart, and what it broadcast ──


@pytest.mark.asyncio
async def test_a_value_answer_is_recorded_as_a_broken_transport_not_a_refusal() -> None:
    """``test_a_value_decision_is_not_consent`` already pins that the MESSAGE
    names the transport and never says "declined" — because only the Asker's
    author can tell yes from no here, and they cannot if the text blames the
    human. The RECORD has to draw the same line, and it is the record that
    outlives the run.

    A ``value`` answer means the transport handed back whatever the person
    typed instead of a verdict, so it belongs with the other four ways NOBODY
    DECIDED. Filed as ``asker`` it is byte-for-byte a human refusal, which is
    the collapse ``source`` exists to prevent — and the more expensive
    direction, since the person may well have typed "yes"."""
    typed_an_answer = _server(_Asker(Decision(kind="value", value="yes, go ahead")))
    genuinely_refused = _server(_Asker(Decision(kind="deny", note="not that path")))

    await _decide(typed_an_answer, "Bash", command="rm -rf /")
    await _decide(genuinely_refused, "Bash", command="rm -rf /")

    (broken,) = typed_an_answer.decisions
    (refused,) = genuinely_refused.decisions
    assert broken.allowed is refused.allowed is False  # identical verdicts...
    assert broken.source == "error"  # ...and distinguishable anyway
    assert refused.source == "asker"
    # It still cost a person the interruption, whatever their transport did
    # with the answer.
    assert broken.asked is True


@pytest.mark.asyncio
async def test_an_observer_that_hangs_does_not_park_the_permission_answer() -> None:
    """Recording is on the critical path: ``_decide`` cannot answer the CLI
    until ``_record`` — which awaits ``ctx.emit`` — has returned. Suppressing
    exceptions covers an observer that RAISES and says nothing about one that
    HANGS, and a hung observer holds the permission prompt open forever, which
    is the ``dontAsk`` failure this module exists to remove arriving by a
    slower route. ``timeout_s`` bounds the human; nothing bounded the bus.

    The bound is the same argument ``_ask`` makes about the ``Asker``: the
    protocol lets an implementation wait forever, so it is enforced here.
    """
    started = asyncio.Event()

    class _HangingCtx:
        async def emit(self, *_a: Any, **_kw: Any) -> None:
            started.set()
            await asyncio.Event().wait()  # never set

    server = _server(_Asker(Decision(kind="approve")), ctx=_HangingCtx())

    with patch.object(approvals, "_EMIT_TIMEOUT_S", 0.05):
        out = await asyncio.wait_for(_decide(server, "Read"), timeout=5)

    assert started.is_set(), "the observer really was called"
    assert out["behavior"] == "allow"  # the CLI got its answer
    assert len(server.decisions) == 1  # ...and the trail kept the decision


@pytest.mark.asyncio
async def test_the_record_exists_before_the_observer_is_told() -> None:
    """The stated contract of ``_record`` is that the append comes FIRST and
    the emit is best-effort. An observer that raises cannot show that — the
    suppression is inside ``_emit``, so both orders look identical from
    outside. A UI that renders the trail when it hears the event can: with the
    order reversed, ``gate.check`` announces a decision that is not in
    ``decisions`` yet, and the pane it paints is one row short of the run."""
    seen_at_emit_time: list[int] = []
    server: ApprovalServer

    class _ReentrantCtx:
        async def emit(self, *_a: Any, **_kw: Any) -> None:
            seen_at_emit_time.append(len(server.decisions))

    server = _server(_Asker(Decision(kind="approve")), ctx=_ReentrantCtx())

    await _decide(server, "Read")

    assert seen_at_emit_time == [1], "the event announced a decision the trail did not have"


@pytest.mark.asyncio
async def test_the_observation_says_which_agent_was_gated() -> None:
    """A run with a supervisor and three workers puts every ``gate.check`` on
    one stream. Without the attribution, "somebody's Bash call was refused" is
    all a reader gets, and the whole point of the record is that a decision has
    provenance. ``HookSettings._emit`` does not pass one and should; this does,
    and nothing pinned it."""
    seen: list[str] = []

    class _AttributingCtx:
        async def emit(self, *_a: Any, agent: str = "", **_kw: Any) -> None:
            seen.append(agent)

    server = _server(_Asker(Decision(kind="deny")), ctx=_AttributingCtx(), agent="worker-2")

    await _decide(server, "Bash", command="ls")

    assert seen == ["worker-2"]


@pytest.mark.asyncio
async def test_the_broadcast_carries_the_whole_record_except_the_arguments() -> None:
    """The payload's redaction is exactly one field wide, and the rest of the
    record has to be on the wire or a consumer reading only the stream cannot
    reconstruct the trail. ``asked`` and ``at`` are the two the existing
    coverage missed, and they are the two that carry information no other
    producer of ``gate.check`` emits."""
    ctx = _RecordingCtx()
    server = _server(
        _Asker(Decision(kind="approve")),
        ctx=ctx,
        clock=FakeClock(start=1_700_000_000.0),
    )

    await _decide(server, "Write", file_path="/tmp/out.txt")

    (rec,) = server.decisions
    (_kind, _render, payload) = ctx.events[0]
    assert payload == {
        "tool": rec.tool,
        "allowed": rec.allowed,
        "reason": rec.reason,
        "source": rec.source,
        "asked": rec.asked,
        "at": rec.at,
    }
    assert payload["asked"] is True
    assert payload["at"] == "2023-11-14T22:13:20+00:00"
    assert "arguments" not in payload


@pytest.mark.asyncio
async def test_an_allow_records_the_arguments_the_caller_handed_over() -> None:
    """The allow path reads its record back off the wire, which means it reads
    it back through ``json.loads``. Anything the encoder normalises on the way
    out — a tuple, a non-string key — comes back normalised, while the SAME
    call refused records the originals. Pinned rather than fixed: the CLI's
    arguments arrive as JSON in production, so the normalisation is a no-op
    there, and reading the record off the answer is what stops the trail and
    the refusal disagreeing. But a caller diffing ``decisions`` against what
    they passed should know the two paths do not answer alike."""
    allowed = _server(_Asker(Decision(kind="approve")))
    refused = _server(_Asker(Decision(kind="deny")))
    args: dict[str, Any] = {"tags": ("a", "b")}

    await allowed._decide("Write", dict(args))
    await refused._decide("Write", dict(args))

    assert allowed.decisions[0].arguments == {"tags": ["a", "b"]}  # JSON round trip
    assert refused.decisions[0].arguments == {"tags": ("a", "b")}  # untouched
