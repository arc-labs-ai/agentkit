"""``ApprovalServer`` — the invariants that make it a gate rather than a relay.

Every test here closes a way this module could stop gating while still looking
like it works. The theme is that an approval gate has exactly one safe default:
a prompt it cannot answer, or answers it cannot read, must DENY. A gate that
fails open is not a gate, and every failure path in this module is a place that
was about to allow.

These exercise ``_decide`` directly rather than over a live MCP server: the
policy is the thing under test, and the transport has its own tests next door.
``_decide`` returns the CLI's wire string, so the assertions read it back the
way the CLI would.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agentkit.integrations.mcp.approvals import (
    ApprovalServer,
    Decision,
    Elicitation,
)


def _read(answer: str) -> dict[str, Any]:
    """The CLI's view of an answer: parse it, as the CLI's own parser would."""
    parsed = json.loads(answer)
    assert parsed["behavior"] in ("allow", "deny"), parsed
    return dict(parsed)


class _Asker:
    """Records what it was asked and answers with whatever it was given."""

    def __init__(self, decision: Decision | Exception | Any = None) -> None:
        self._answer = decision if decision is not None else Decision(kind="deny")
        self.requests: list[Elicitation] = []

    async def ask(self, request: Elicitation) -> Any:
        self.requests.append(request)
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


class _NeverAnswers:
    """An ``Asker`` that parks, which is what the protocol permits."""

    def __init__(self) -> None:
        self.calls = 0

    async def ask(self, request: Elicitation) -> Decision:
        self.calls += 1
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


# ── the default is deny ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_transport_that_raises_denies_rather_than_allowing() -> None:
    """The gate's failure mode. An ``Asker`` that raises — a queue that is
    down, a websocket that dropped — must not become an approval. This is the
    single most damaging direction this module could fail in, because the
    exception arrives precisely when nobody is available to notice."""
    asker = _Asker(RuntimeError("the queue is gone"))
    server = ApprovalServer(asker=asker, autonomy="manual")

    answer = _read(await server._decide("Bash", {"command": "rm -rf /"}))

    assert answer["behavior"] == "deny"
    assert "RuntimeError" in answer["message"], "the operator needs the cause"
    assert "the queue is gone" in answer["message"]
    assert server.decisions[0].allowed is False
    assert server.decisions[0].source == "error", (
        "not 'asker' — nobody declined this, the transport broke, and the trail "
        "has to send whoever reads it at the transport rather than at a reviewer"
    )


@pytest.mark.asyncio
async def test_an_expired_prompt_denies_so_a_slow_reviewer_is_not_a_yes() -> None:
    """``expired`` is not consent. A reviewer who stepped away must not become
    an approval by the passage of time."""
    asker = _Asker(Decision(kind="expired", note="no answer within 30s"))
    server = ApprovalServer(asker=asker, autonomy="manual")

    answer = _read(await server._decide("Write", {"file_path": "/etc/passwd"}))

    assert answer["behavior"] == "deny"
    assert "30s" in answer["message"], "the message must say WHICH denial this was"
    assert server.decisions[0].source == "timeout", "the clock refused, not a person"


def test_an_expired_decision_is_not_approved() -> None:
    """``Decision.approved`` is the one predicate the whole gate turns on, and
    it is shared with the agentkit-native HITL path. If ``expired`` ever counts
    as approval here, both paths start allowing timeouts at once."""
    assert Decision(kind="expired").approved is False
    assert Decision(kind="deny").approved is False
    assert Decision(kind="value", value="yes").approved is False
    assert Decision(kind="approve").approved is True
    assert Decision(kind="modify", value={}).approved is True


@pytest.mark.asyncio
async def test_a_value_answer_is_not_consent() -> None:
    """The request goes out as ``kind="approval"`` with
    ``choices=("approve", "deny")``, so a ``value`` back means the transport
    handed over whatever the person typed instead of a verdict — possibly the
    word "no". It used to allow. The message must name the transport, because
    its author is the only one who can tell yes from no here."""
    asker = _Asker(Decision(kind="value", value="yes"))
    server = ApprovalServer(asker=asker, autonomy="manual")

    answer = _read(await server._decide("Bash", {"command": "ls"}))

    assert answer["behavior"] == "deny"
    assert "not consent" in answer["message"]
    assert "declined" not in answer["message"], (
        "a broken transport must not be reported as a reviewer who said no"
    )
    assert server.decisions[0].source == "error"


@pytest.mark.asyncio
async def test_an_answer_that_is_not_a_decision_denies_and_names_the_type() -> None:
    """The HITL API this replaced was ``dict[str, str]`` with ``"approve"`` as
    a bare string, so an ``Asker`` written from that memory returns a ``str``
    and ``decision.kind`` raises. Left to escape, that reaches the CLI as a
    BROKEN permission system rather than a denial — the model is told the gate
    is malfunctioning and retries a call nobody approved."""
    server = ApprovalServer(asker=_Asker("approve"), autonomy="manual")

    answer = _read(await server._decide("Read", {"file_path": "/tmp/x"}))

    assert answer["behavior"] == "deny"
    assert "str" in answer["message"], "the transport's author needs to see what they sent"
    assert server.decisions[0].source == "error"


@pytest.mark.asyncio
async def test_a_denial_carries_the_reviewers_reason_to_the_model() -> None:
    """``message`` reaches the MODEL, which may adapt. A bare "no" sends it
    round the same wall again."""
    asker = _Asker(Decision(kind="deny", note="that path is outside the sandbox"))
    server = ApprovalServer(asker=asker, autonomy="manual")

    answer = _read(await server._decide("Write", {"file_path": "/etc/hosts"}))

    assert answer["message"] == "that path is outside the sandbox"
    assert server.decisions[0].reason == "that path is outside the sandbox"


# ── the arguments are the thing being approved ───────────────────────────────


@pytest.mark.asyncio
async def test_the_arguments_reach_the_human_who_is_approving_them() -> None:
    """A tool name is not a decision. Approving ``Bash`` is meaningless;
    approving ``Bash rm -rf /`` is the decision. If the arguments do not travel
    on the ``Elicitation``, the reviewer approves a call they cannot see."""
    asker = _Asker(Decision(kind="approve"))
    server = ApprovalServer(asker=asker, autonomy="manual", timeout_s=30.0)

    await server._decide("Bash", {"command": "rm -rf /tmp/scratch"})

    request = asker.requests[0]
    assert request.tool_call == {
        "name": "Bash",
        "arguments": {"command": "rm -rf /tmp/scratch"},
    }
    assert request.deadline_s == 30.0, (
        "the deadline must reach the Elicitation or the pending card shows no "
        "deadline, the person answers at their own pace, and the answer is "
        "discarded by a timeout they were never shown"
    )


@pytest.mark.asyncio
async def test_a_reviewers_edits_are_what_the_tool_runs_with() -> None:
    """``modify`` is the CLI's ``updatedInput`` semantics: a reviewer redirects
    a write to a sandbox path without derailing the run. Treating it as a plain
    approval silently discards the edit and runs the ORIGINAL call."""
    asker = _Asker(Decision(kind="modify", value={"file_path": "/tmp/safe"}))
    server = ApprovalServer(asker=asker, autonomy="manual")

    answer = _read(await server._decide("Write", {"file_path": "/etc/passwd"}))

    assert answer["behavior"] == "allow"
    assert answer["updatedInput"] == {"file_path": "/tmp/safe"}
    assert server.decisions[0].arguments == {"file_path": "/tmp/safe"}, (
        "the trail must name the call that actually ran, not the one that was "
        "requested — the arguments that ran exist nowhere else"
    )


@pytest.mark.asyncio
async def test_a_modify_whose_replacement_is_not_a_dict_denies() -> None:
    """``updatedInput`` has to be an argument mapping. A reviewer handing back
    a string cannot be turned into a call, and guessing is worse than refusing."""
    asker = _Asker(Decision(kind="modify", value="/tmp/safe"))
    server = ApprovalServer(asker=asker, autonomy="manual")

    answer = _read(await server._decide("Write", {"file_path": "/etc/passwd"}))

    assert answer["behavior"] == "deny"
    assert "str" in answer["message"]


@pytest.mark.asyncio
async def test_a_small_argument_is_recorded_as_it_arrived_not_as_json() -> None:
    """``decisions[0].arguments["file_path"] == "/tmp/out"`` must be a string
    comparison. Storing the JSON rendering makes it ``'"/tmp/out"'`` with the
    quotes, and every caller comparison against a real path silently fails."""
    server = ApprovalServer(asker=_Asker(Decision(kind="approve")), autonomy="manual")

    await server._decide("Write", {"file_path": "/tmp/out", "count": 3})

    assert server.decisions[0].arguments == {"file_path": "/tmp/out", "count": 3}


@pytest.mark.asyncio
async def test_an_unencodable_argument_denies_instead_of_escaping() -> None:
    """``json.dumps`` raises ``TypeError`` on a value it cannot encode, and the
    one path that could fail open is the one that was about to ALLOW. Under
    ``auto`` the gate allows without asking, so this is that path."""
    server = ApprovalServer(autonomy="auto")

    answer = _read(await server._decide("Write", {"handle": object()}))

    assert answer["behavior"] == "deny"
    assert "could not be encoded" in answer["message"]


# ── the tiers, and auto_allow ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_auto_tier_never_consults_a_person() -> None:
    """The tier's whole meaning is that no person is in this loop, so the
    assertion is on the call that did NOT happen. ``source="autonomy"`` is the
    only thing keeping "the tier does not gate this" from reading in the trail
    as "a person approved it"."""
    asker = _Asker(Decision(kind="deny"))
    server = ApprovalServer(asker=asker, autonomy="auto")

    answer = _read(await server._decide("Bash", {"command": "ls"}))

    assert answer["behavior"] == "allow"
    assert asker.requests == [], "no person is in this loop"
    assert server.decisions[0].source == "autonomy"
    assert server.decisions[0].asked is False
    assert server.prompts_seen == 1, (
        "auto is the one tier with no Asker, no reviewer and no elicitation, so "
        "the count is the ONLY evidence the server saw the call and allowed it"
    )


@pytest.mark.asyncio
async def test_auto_allow_skips_the_reviewer_for_the_named_tool() -> None:
    """The habituation guard: a person clicking yes on forty ``Read`` calls is
    not oversight. If ``auto_allow`` is ignored, every read prompts and the
    fortieth prompt — the one that mattered — gets the same reflexive yes."""
    asker = _Asker(Decision(kind="deny"))
    server = ApprovalServer(asker=asker, autonomy="gated", auto_allow=("Read",))

    answer = _read(await server._decide("Read", {"file_path": "/tmp/x"}))

    assert answer["behavior"] == "allow"
    assert asker.requests == []
    assert server.decisions[0].source == "auto_allow"


@pytest.mark.asyncio
async def test_auto_allow_is_inert_under_manual() -> None:
    """A static name list set at wiring time must not carve exceptions out of
    the strictest tier, chosen per run. Otherwise "manual" silently means
    "manual except the forty Reads" and nothing in the run says so."""
    asker = _Asker(Decision(kind="deny", note="no"))
    server = ApprovalServer(asker=asker, autonomy="manual", auto_allow=("Read",))

    answer = _read(await server._decide("Read", {"file_path": "/tmp/x"}))

    assert answer["behavior"] == "deny"
    assert len(asker.requests) == 1, "the reviewer must still be consulted"


@pytest.mark.asyncio
async def test_the_argument_predicate_only_narrows_and_is_never_a_second_way_in() -> None:
    """``tool_name in auto_allow`` is checked FIRST, so the predicate is a
    narrowing filter over an already-approved set. A predicate returning True
    for a tool nobody allow-listed must change nothing — otherwise
    ``auto_allow_when`` BROADENS the grant it exists to narrow."""
    asker = _Asker(Decision(kind="deny", note="not allowed"))
    server = ApprovalServer(
        asker=asker,
        autonomy="gated",
        auto_allow=("Read",),
        auto_allow_when=lambda name, args: True,
    )

    answer = _read(await server._decide("Write", {"file_path": "/etc/passwd"}))

    assert answer["behavior"] == "deny"
    assert len(asker.requests) == 1, (
        "Write was never allow-listed, so the predicate must not admit it"
    )


@pytest.mark.asyncio
async def test_the_argument_predicate_narrows_to_the_reviewer_not_to_a_denial() -> None:
    """"Closed" for a NARROWING rule means falling through to the human: the
    operator did say this tool was routine, so the reviewer is the right place
    to land, and they still see the arguments."""
    asker = _Asker(Decision(kind="approve"))
    server = ApprovalServer(
        asker=asker,
        autonomy="gated",
        auto_allow=("Read",),
        auto_allow_when=lambda name, args: args.get("file_path", "").startswith("/tmp/"),
    )

    answer = _read(await server._decide("Read", {"file_path": "/etc/shadow"}))

    assert answer["behavior"] == "allow", "the reviewer approved it"
    assert len(asker.requests) == 1, "narrowed out of auto_allow, so a person decides"
    assert server.decisions[0].source == "asker"


@pytest.mark.asyncio
async def test_a_predicate_that_raises_does_not_widen_the_grant() -> None:
    """A broken narrowing rule must never widen what it exists to narrow. It
    falls through to the reviewer rather than allowing or crashing the handler."""
    def boom(name: str, args: dict[str, Any]) -> bool:
        raise ValueError("predicate is broken")

    asker = _Asker(Decision(kind="deny", note="no"))
    server = ApprovalServer(
        asker=asker, autonomy="gated", auto_allow=("Read",), auto_allow_when=boom
    )

    answer = _read(await server._decide("Read", {"file_path": "/tmp/x"}))

    assert answer["behavior"] == "deny"
    assert len(asker.requests) == 1


def test_auto_allow_as_a_bare_string_is_refused_at_construction() -> None:
    """``auto_allow="Read"`` — the missing comma — is a string, and ``in``
    against a string matches every SUBSTRING of it: ``"Re"``, ``"ea"``, ``""``.
    That approves calls with the reviewer never consulted, which is the exact
    opposite of what the operator typed."""
    with pytest.raises(ValueError, match="is a string, not a"):
        ApprovalServer(asker=_Asker(), autonomy="gated", auto_allow="Read")  # type: ignore[arg-type]


def test_a_gating_tier_without_a_reviewer_is_refused_at_construction() -> None:
    """A tier that sends prompts to a human, wired with nobody to send them to,
    would deny every call at runtime — ``dontAsk``, the failure this module
    exists to remove. Better refused where it was written."""
    with pytest.raises(ValueError, match="sends prompts to a human"):
        ApprovalServer(autonomy="manual")


@pytest.mark.asyncio
async def test_a_tier_tightened_after_construction_denies_and_names_the_cause() -> None:
    """``__post_init__`` rejects asker-less gating, so reaching the runtime
    check means ``autonomy`` was tightened afterwards. ``_decide`` must never
    raise, so it denies naming the missing transport — the operator reads the
    cause in the model's own refusal rather than an ``AttributeError`` inside
    an MCP request handler."""
    server = ApprovalServer(autonomy="auto")
    server.autonomy = "manual"

    answer = _read(await server._decide("Bash", {"command": "ls"}))

    assert answer["behavior"] == "deny"
    assert "no reviewer is wired" in answer["message"]
    assert server.decisions[0].source == "error"


# ── the timeout ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_timeout_is_enforced_here_not_trusted_to_the_asker() -> None:
    """The protocol lets an ``Asker`` wait forever, and a queue worker holding
    a CLI subprocess open indefinitely is a resource leak with a model
    attached. The bound has to be applied on this side."""
    asker = _NeverAnswers()
    server = ApprovalServer(asker=asker, autonomy="manual", timeout_s=0.05)

    answer = _read(await server._decide("Bash", {"command": "sleep 1"}))

    assert answer["behavior"] == "deny"
    assert "no answer within 0.05s" in answer["message"]
    assert asker.calls == 1, "the reviewer WAS reached, and then the clock ran out"
    assert server.decisions[0].source == "timeout"
    assert server.decisions[0].asked is True, (
        "whether somebody was interrupted is a fact about what the run cost a "
        "human, and is not recoverable from the verdict"
    )


def test_a_non_positive_timeout_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="denies every prompt"):
        ApprovalServer(asker=_Asker(), autonomy="manual", timeout_s=0)


@pytest.mark.asyncio
async def test_a_timeout_zeroed_after_construction_says_nobody_was_asked() -> None:
    """``timeout_s`` is a mutable public field, so ``__post_init__``'s refusal
    is not the last word. Zeroed afterwards it denied every prompt with the
    ``Asker`` never scheduled — and did so under "no answer within 0s", which
    blames a reviewer who was never asked and sends the operator to debug the
    transport. It still denies; it now says what actually happened."""
    asker = _Asker(Decision(kind="approve"))
    server = ApprovalServer(asker=asker, autonomy="manual", timeout_s=30.0)
    server.timeout_s = 0

    answer = _read(await server._decide("Bash", {"command": "ls"}))

    assert answer["behavior"] == "deny"
    assert "misconfigured" in answer["message"]
    assert asker.requests == [], "nobody was asked, and the message must not imply they were"


@pytest.mark.asyncio
async def test_a_none_timeout_waits_for_the_reviewer() -> None:
    """``None`` waits forever, which is right for an interactive UI."""
    asker = _Asker(Decision(kind="approve"))
    server = ApprovalServer(asker=asker, autonomy="manual", timeout_s=None)

    answer = _read(await server._decide("Read", {"file_path": "/tmp/x"}))

    assert answer["behavior"] == "allow"


# ── the trail ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_prompt_is_recorded_oldest_first_with_its_provenance() -> None:
    """A count of prompts answers no question worth asking: not what was
    allowed, not whether anybody looked, not which approval preceded the write
    you are staring at."""
    asker = _Asker(Decision(kind="approve"))
    server = ApprovalServer(asker=asker, autonomy="gated", auto_allow=("Read",))

    await server._decide("Read", {"file_path": "/tmp/a"})
    await server._decide("Write", {"file_path": "/tmp/b"})

    first, second = server.decisions
    assert (first.tool, first.source, first.asked) == ("Read", "auto_allow", False)
    assert (second.tool, second.source, second.asked) == ("Write", "asker", True)
    assert [d.allowed for d in server.decisions] == [True, True]
    assert server.prompts_seen == 2
    assert all(d.at for d in server.decisions), "the run-clock timestamp must be recorded"


# ── the wiring handed to the CLI ─────────────────────────────────────────────


def test_the_cli_wiring_shuts_out_ambient_config_that_would_bypass_the_reviewer() -> None:
    """Two settings here are load-bearing rather than tidy. Without
    ``strict_mcp_config`` the CLI also loads whatever MCP servers the working
    directory or the user's home configuration define. And with ambient setting
    sources enabled, a user's global ``permissions.allow = ["Write"]`` resolves
    the call BEFORE ``--permission-prompt-tool`` is consulted — silently
    bypassing the reviewer this object exists to ask."""
    server = ApprovalServer(asker=_Asker(), autonomy="manual")

    kwargs = server.cli_kwargs()

    assert kwargs["strict_mcp_config"] is True
    assert kwargs["setting_sources"] == ()
    assert kwargs["permission_prompt_tool"] == server.tool_name
    assert kwargs["permission_mode"] == "default"


@pytest.mark.asyncio
async def test_the_permission_tool_answers_with_one_text_block_and_nothing_else() -> None:
    """``structured_output=False`` is load-bearing. The CLI rejects anything
    but ONE text block — "Expected a single text block param with type='text'
    and a string text value" — and FastMCP's default adds a
    ``structuredContent`` field alongside the text. Every prompt then fails
    with exactly that while the tool itself looks like it ran.

    Goes through ``build_mcp()`` rather than ``_decide`` because the defect is
    in the wire shape, which ``_decide`` never sees."""
    server = ApprovalServer(asker=_Asker(Decision(kind="approve")), autonomy="manual")

    result = await server.build_mcp().call_tool(
        server.tool_name.rsplit("__", 1)[-1],
        {"tool_name": "Read", "input": {"file_path": "/tmp/x"}},
    )

    assert not isinstance(result, dict), (
        "a dict result is FastMCP's structured form; the CLI rejects it outright"
    )
    assert len(result) == 1, f"the CLI accepts exactly one block, got {len(result)}"
    block = result[0]
    assert block.type == "text"
    assert _read(block.text)["behavior"] == "allow"


# ── the listener's bind failure ──────────────────────────────────────────────
#
# These two live here rather than beside the other transport tests because the
# catalogue anchors them to the approval gate: this is the listener the gate
# reaches the CLI through, and a gate that is not actually listening hands the
# CLI a URL that fails thirty seconds later as "MCP server failed to connect",
# with nothing pointing at the port.


def test_a_port_collision_names_the_endpoint_it_could_not_bind() -> None:
    """``str(OSError)`` for a collision is exactly
    ``"[Errno 48] Address already in use"`` — the bare errno does NOT carry the
    endpoint. A process running several of these servers otherwise gets one
    indistinguishable line per collision with nothing saying which port.

    ``errno`` and ``strerror`` must survive, because callers match on those."""
    import errno as errno_mod
    import socket as socket_mod

    from agentkit.integrations.mcp._transport import LoopbackMcpTransport

    holder = socket_mod.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    taken = holder.getsockname()[1]
    try:
        transport = LoopbackMcpTransport(host="127.0.0.1", port=taken)
        with pytest.raises(OSError) as caught:
            transport.reserve()
    finally:
        holder.close()

    assert f"127.0.0.1:{taken}" in str(caught.value), (
        f"the endpoint has to be IN the message: {caught.value}"
    )
    assert caught.value.errno in (errno_mod.EADDRINUSE, errno_mod.EACCES)
    assert caught.value.strerror


def test_a_failed_bind_closes_the_socket_it_opened() -> None:
    """The fd is opened before the bind is attempted, so the failure path owns
    it. Leaking one per collision is invisible until a long-lived host that
    retries has run out of descriptors — at which point the failure surfaces
    somewhere else entirely.

    Asserted on the socket's own ``close``, because an fd count is noisy under
    a test runner that holds its own."""
    import socket as socket_mod

    from agentkit.integrations.mcp import _transport as transport_mod

    closed: list[bool] = []
    real_socket = socket_mod.socket

    class _WatchedSocket(real_socket):  # type: ignore[misc,valid-type]
        def close(self) -> None:
            closed.append(True)
            super().close()

    holder = real_socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    taken = holder.getsockname()[1]
    original = transport_mod.socket.socket
    transport_mod.socket.socket = _WatchedSocket  # type: ignore[assignment]
    try:
        transport = transport_mod.LoopbackMcpTransport(host="127.0.0.1", port=taken)
        with pytest.raises(OSError):
            transport.reserve()
    finally:
        transport_mod.socket.socket = original  # type: ignore[assignment]
        holder.close()

    assert closed, "the socket opened for the failed bind was never closed"
