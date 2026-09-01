"""Containment: the Rule-of-Two tags, and the warning about the chain that is not there.

Two safety mechanisms this cognition participates in, and both of them are the
kind that look wired and do nothing unless something makes them speak:

**``RunPolicy``'s lethal-trifecta gate** reads ``tool.caps`` off everything it
is handed. ``ClaudeCliCognition`` had none at all until someone noticed — a CLI
session holding ``Read`` + ``WebFetch`` + ``Bash`` sailed past the same check
that refuses the equivalent agentkit tool set, because the gate had nothing to
look at. This cognition cannot copy that fix, because Codex has no tool list to
tag: every session gets the same ``shell`` and the SANDBOX decides what it
reaches. So the tags come off ``sandbox`` / ``network_access`` / ``web_search``,
and the configuration that matters is the innocuous-looking one — a read-only
sandbox with web search on is private data, untrusted content and egress, which
is the trifecta exactly.

**The middleware-bypass warning.** The CLI runs its own ``shell`` inside its own
process, so no tool middleware applies to it: no ``egress`` URL check, no
``guard``, no ``audit`` record, no ``memoize`` key. Codex has no ``PreToolUse``
hook, so unlike the Claude cognition there is no escape hatch to steer a caller
toward — which makes saying so out loud the only thing available.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from agentkit.agents.cognition import CODEX_NATIVE_TOOLS, CodexCliCognition
from agentkit.agents.control.safety import TRIFECTA, RunPolicy
from agentkit.testing.fakes import FakeCodexCli, codex_turn
from agentkit.testing.fakes.ctx import FakeCtx
from tests.agents.cognition.test_codex_cli import drive

# ─────────────────────────────────────────────────────────────────────────────
# 1. the tags
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # The default. ``codex exec`` is read-only, and read-only is about
        # WRITES — the agent can still read the whole workspace, which is
        # private data by any reading of the word.
        ({}, ("private_data",)),
        ({"sandbox": "read-only"}, ("private_data",)),
        # Writes do not add a Rule-of-Two tag. The vocabulary has three words
        # and none of them means "mutates local state"; that axis is
        # ``FunctionTool.side_effecting``.
        ({"sandbox": "workspace-write"}, ("private_data",)),
        # Full access is the network too.
        ({"sandbox": "danger-full-access"}, ("egress", "private_data")),
        # The bypass flag IS full access under another name.
        ({"bypass_sandbox": True}, ("egress", "private_data")),
        # The one flag that reopens the workspace sandbox's network.
        ({"sandbox": "workspace-write", "network_access": True}, ("egress", "private_data")),
        # Search is a network call over content nobody in this system authored.
        ({"web_search": True}, ("egress", "private_data", "untrusted_content")),
    ],
)
def test_the_caps_come_from_the_sandbox_and_the_search_flag(
    kwargs: dict[str, Any], expected: tuple[str, ...]
) -> None:
    assert CodexCliCognition(**kwargs).caps == expected


def test_an_unset_sandbox_still_reports_the_clis_own_default() -> None:
    """The case a caps property gets wrong by being literal. Reporting nothing
    for "the flag was not passed" would leave the trifecta gate blind on the
    most common wiring there is — a cognition nobody configured."""
    assert CodexCliCognition().effective_sandbox == "read-only"
    assert CodexCliCognition().caps == ("private_data",)


def test_a_read_only_sandbox_with_web_search_is_refused_as_the_trifecta() -> None:
    """The headline. It reads like the safest configuration Codex has — the
    agent cannot write anything — and it combines all three capabilities."""
    cog = CodexCliCognition(sandbox="read-only", web_search=True)
    assert set(cog.caps) == set(TRIFECTA)

    with pytest.raises(PermissionError, match="lethal trifecta"):
        RunPolicy(mode="deny").check([cog])

    verdict = RunPolicy(mode="flag").check([cog])
    assert verdict.allowed is False
    assert verdict.reason and "lethal trifecta" in verdict.reason


def test_a_read_only_sandbox_without_search_is_allowed() -> None:
    """The gate has to let the ordinary case through, or it is a gate people
    turn off."""
    assert RunPolicy(mode="deny").check([CodexCliCognition()]).allowed is True


def test_an_agentkit_tool_can_complete_a_set_the_cli_started() -> None:
    """The case a per-path check misses entirely. The CLI supplies private data
    and (with full access) egress; one agentkit tool tagged
    ``untrusted_content`` finishes the trifecta, and only a check over BOTH
    lists sees it."""

    class _Scraper:
        name = "scrape"
        caps = ("untrusted_content",)

    cog = CodexCliCognition(sandbox="danger-full-access")
    assert RunPolicy(mode="deny").check([cog]).allowed is True
    with pytest.raises(PermissionError, match="lethal trifecta"):
        RunPolicy(mode="deny").check([cog, _Scraper()])


def test_a_session_reports_the_cognitions_caps() -> None:
    """A session can BE an agent's cognition, so it lands in exactly the place
    the cognition would. Without the delegation ``getattr(session, "caps", ())``
    is the empty tuple and the gate sees a tool-less run."""
    cog = CodexCliCognition(sandbox="read-only", web_search=True)
    assert cog.session().caps == cog.caps
    with pytest.raises(PermissionError, match="lethal trifecta"):
        RunPolicy(mode="deny").check([cog.session()])


def test_every_tag_this_cognition_emits_is_in_run_policys_vocabulary() -> None:
    """``RunPolicy.capabilities`` intersects with ``TRIFECTA`` and silently
    DROPS everything outside it, so an invented tag reads like protection and
    provides none."""
    every: set[str] = set()
    for kwargs in (
        {},
        {"sandbox": "workspace-write", "network_access": True},
        {"sandbox": "danger-full-access"},
        {"web_search": True},
        {"bypass_sandbox": True},
    ):
        every.update(CodexCliCognition(**kwargs).caps)
    assert every <= set(TRIFECTA)


# ─────────────────────────────────────────────────────────────────────────────
# 2. the native tool list
# ─────────────────────────────────────────────────────────────────────────────


def test_the_native_tools_are_the_fixed_set_plus_search_when_enabled() -> None:
    assert CodexCliCognition().native_tools() == tuple(sorted(CODEX_NATIVE_TOOLS))
    assert "web_search" in CodexCliCognition(web_search=True).native_tools()


def test_apply_patch_is_reported_even_under_a_read_only_sandbox() -> None:
    """The tool is present and the model still calls it; the sandbox is what
    makes the write fail. Dropping it here would tell a reader the chain has
    nothing to miss, when what actually happens is an attempted edit the chain
    never saw."""
    assert "apply_patch" in CodexCliCognition(sandbox="read-only").native_tools()


# ─────────────────────────────────────────────────────────────────────────────
# 3. the bypass warning
# ─────────────────────────────────────────────────────────────────────────────


class _Invoker:
    def __init__(self, chain: list[Any]) -> None:
        self.tool_middleware = chain


class _CtxWithChain(FakeCtx):
    def __init__(self, chain: list[Any]) -> None:
        super().__init__()
        self.invoker = _Invoker(chain)


def _egress_like() -> Any:
    class Egress:
        async def on_request(self, call: Any) -> None: ...

    return Egress()


def _memoize_like() -> Any:
    def memoize() -> Any:
        async def mw(call: Any, nxt: Any) -> Any:
            return await nxt(call)

        return mw

    return memoize()


@pytest.mark.asyncio
async def test_a_context_with_tool_middleware_warns_and_names_them() -> None:
    """Naming them is the whole point. The failure is invisible by
    construction, so a generic "middleware may not apply" would leave the
    reader no better off than the silence it replaced."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    cog = CodexCliCognition(spawn=cli)
    ctx = _CtxWithChain([_egress_like(), _memoize_like()])

    with pytest.warns(UserWarning) as caught:
        await drive(cog, ctx=ctx)

    message = str(caught[0].message)
    assert "Egress" in message and "memoize" in message
    # And what IS containing the CLI's tools, since there is no hook to offer.
    assert "shell" in message
    assert "sandbox='read-only'" in message


@pytest.mark.asyncio
async def test_the_warning_says_codex_has_no_hook_escape_hatch() -> None:
    """The Claude warning steers a caller to ``hook_settings`` or to serving
    tools over MCP. Half of that advice does not exist here, and repeating it
    would send someone looking for an API that is not there."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    with pytest.warns(UserWarning) as caught:
        await drive(CodexCliCognition(spawn=cli), ctx=_CtxWithChain([_egress_like()]))
    message = str(caught[0].message)
    assert "no PreToolUse hook" in message
    assert "mcp_servers" in message


@pytest.mark.asyncio
async def test_the_warning_fires_once_per_instance() -> None:
    """Not per drive: a warning that repeats every iteration is noise, and noise
    is what teaches people to add a ``filterwarnings`` line — which is how the
    NEXT silent misconfiguration gets through."""
    cli = FakeCodexCli(
        [
            FakeCodexCli.answering("a")._runs[0],
            FakeCodexCli.answering("b")._runs[0],
        ]
    )
    cog = CodexCliCognition(spawn=cli)
    ctx = _CtxWithChain([_egress_like()])

    with pytest.warns(UserWarning) as caught:
        await drive(cog, ctx=ctx)
    assert len(caught) == 1

    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        await drive(cog, ctx=ctx)
    assert [w for w in second if "CodexCliCognition bypasses" in str(w.message)] == []


@pytest.mark.asyncio
async def test_a_second_cognition_still_warns() -> None:
    """The latch is per instance, not per process. Two cognitions in one service
    are usually two configurations, and the second one is the one nobody has
    audited."""
    ctx = _CtxWithChain([_egress_like()])
    for _ in range(2):
        cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
        with pytest.warns(UserWarning, match="bypasses the Invoker"):
            await drive(CodexCliCognition(spawn=cli), ctx=ctx)


@pytest.mark.asyncio
async def test_no_chain_means_no_warning() -> None:
    """There is nothing being skipped. Warning anyway is how a real signal
    becomes background."""
    cli = FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1)))
    with warnings.catch_warnings():
        # The suite runs with -W error, so an unexpected warning fails here on
        # its own; the explicit filter makes the intent readable rather than
        # incidental.
        warnings.simplefilter("error")
        await drive(CodexCliCognition(spawn=cli), ctx=FakeCtx())


@pytest.mark.asyncio
async def test_a_ctx_with_no_invoker_at_all_does_not_raise() -> None:
    """A missing collaborator must not turn a safety warning into an
    ``AttributeError`` out of the caller's first run. ``make_test_ctx()`` with
    no LLM and a bare structural stub both land here."""

    class _Bare:
        correlation_id = "x"

        def check_cancelled(self) -> None: ...

    cli = FakeCodexCli.script(codex_turn(text="ok", usage=(1, 0, 1)))
    events = await drive(CodexCliCognition(spawn=cli), ctx=_Bare())
    assert events[-1].result.output == "ok"


@pytest.mark.asyncio
async def test_the_warning_latches_even_when_promoted_to_an_error() -> None:
    """``warnings.warn`` can be promoted by the caller's filters. If the latch
    were set after the call, an escaping error would leave it unset and the
    next drive would raise again — turning one wiring complaint into a
    permanent outage."""
    cog = CodexCliCognition(spawn=FakeCodexCli.script(codex_turn(text="x", usage=(1, 0, 1))))
    ctx = _CtxWithChain([_egress_like()])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            await drive(cog, ctx=ctx)

    assert cog._bypass_warned is True
