"""The CLI bypasses the ``Invoker`` — say so, and let ``RunPolicy`` see it anyway.

``_charge_meters`` already carries the finding in prose: "The CLI bypasses the
``Invoker``, so the ``meter()`` middleware never sees this usage and every meter
on the context stays at zero. That is how a documented safety mechanism ends up
doing nothing." Metering was then patched by hand. *Nothing else was.*

Every other tool-chain middleware has the same hole and none of them got the
patch: ``egress()`` (default-deny URL checking — a native ``WebFetch`` reaches
anywhere), ``security()`` (every input guard), ``audit()`` (one record per tool
call — there are none), ``memoize()``/``idempotent()`` (idempotency and
single-flight), and behind ``egress`` the ``Guardrail.check_url`` SSRF and
allowlist checks. A caller who reads agentkit's middleware documentation, wires
the documented tool chain, and then swaps in ``ClaudeCliCognition`` gets a
session where **none of it applies**, with nothing anywhere saying so.

Two things close that, and both are tested here:

1. **A warning that names the middlewares.** Not "middleware may not apply" —
   the failure is invisible by construction, so the message has to be specific
   enough that the reader can check it against their own chain. Fires only when
   the session actually HAS native tools; a session serving everything over MCP
   (``tools=("",)``) routes back through agentkit and is not affected.
2. **Capability tags for the built-in CLI tools**, in ``RunPolicy``'s own
   vocabulary (``TRIFECTA``), so a CLI session holding ``Read`` + ``WebFetch`` +
   ``Bash`` is refusable by the same Rule-of-Two check that refuses the
   equivalent agentkit tool set. Before this, ``grep -c caps claude_cli.py``
   returned 0 and the trifecta was invisible on this path.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentkit import Agent
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.agents.cognition._cli_common import _middleware_name, _tool_middleware_names
from agentkit.agents.cognition.claude_cli import CLI_TOOL_CAPS
from agentkit.agents.control.safety import TRIFECTA, RunPolicy
from agentkit.context import WorkingContext
from agentkit.middlewares import audit, egress, idempotent, meter, security, tracing
from agentkit.testing import FakeLLM, make_test_ctx
from agentkit.tools import FunctionTool
from tests.agents.cognition.test_claude_cli import _FakeProcess, _happy_path_lines
from tests.agents.cognition.test_claude_cli_session import (
    _collect,
    _FakeSessionProcess,
    _patched,
    _turn_lines,
)


class _Guardrail:
    """Minimal ``egress()`` collaborator — it only has to expose ``check_url``."""

    def check_url(self, url: str) -> None:  # pragma: no cover - never reached
        raise AssertionError("the CLI never routes a URL through the egress middleware")


def _ctx(*middleware: Any) -> Any:
    """A real ``RunContext`` whose ``Invoker`` carries ``middleware`` on the TOOL chain."""
    return make_test_ctx(llm=FakeLLM(["ok"]), tool_middleware=list(middleware))


def _drive(cog: ClaudeCliCognition, ctx: Any) -> list[Any]:
    proc = _FakeProcess(stdout_lines=_happy_path_lines())
    with patch(
        "agentkit.agents.cognition.claude_cli.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        agent = Agent(name="x", cognition=cog)

        async def _go() -> list[Any]:
            return [ev async for ev in cog.drive(agent, "t", ctx, WorkingContext())]

        return asyncio.run(_go())


def _warnings_from_drive(cog: ClaudeCliCognition, ctx: Any, *, drives: int = 1) -> list[Any]:
    """Every ``UserWarning`` raised across ``drives`` consecutive ``drive`` calls.

    ``simplefilter("always")`` on purpose: Python's own ``__warningregistry__``
    dedups by (message, category, lineno), which would hide whether the
    *cognition* is latching or the *interpreter* is. The once-per-cognition
    guarantee has to be the code's, not the warning machinery's.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(drives):
            _drive(cog, ctx)
    return [w for w in caught if issubclass(w.category, UserWarning)]


def _bypass_warnings(records: list[Any]) -> list[str]:
    return [str(w.message) for w in records if "bypasses the Invoker" in str(w.message)]


# ── the two the spec demands ────────────────────────────────────────────────


def test_tool_middleware_plus_native_tools_warns_and_names_the_middlewares() -> None:
    """The warning names EVERY middleware on the chain, not "middleware may not apply".

    A generic warning is unactionable: the reader cannot tell whether the thing
    they are relying on is in the list. Naming ``Egress``/``Audit``/``memoize``
    lets them diff the message against the chain they wrote.
    """
    ctx = _ctx(tracing(), meter(), egress(_Guardrail()), idempotent(), audit(), security())
    cog = ClaudeCliCognition(tools=("Read", "WebFetch", "Bash"))

    messages = _bypass_warnings(_warnings_from_drive(cog, ctx))

    assert len(messages) == 1, messages
    for name in ("tracing", "MeterMiddleware", "Egress", "memoize", "Audit", "SecurityMiddleware"):
        assert name in messages[0], f"{name!r} missing from: {messages[0]}"


def test_everything_over_mcp_does_not_warn() -> None:
    """``tools=("",)`` disables every built-in tool, so every call comes back
    through agentkit's own tool path and the chain DOES apply. Warning here
    would be the noise that trains people to filter the real one."""
    ctx = _ctx(egress(_Guardrail()), audit())
    cog = ClaudeCliCognition(tools=("",), mcp_config=('{"mcpServers": {}}',))

    assert _bypass_warnings(_warnings_from_drive(cog, ctx)) == []


# ── the edge cases: what counts as "native tools enabled" ───────────────────


@pytest.mark.parametrize(
    ("tools", "allowed", "disallowed", "expected"),
    [
        # ``tools=None`` leaves the CLI's own default set in place — that is
        # every built-in tool, Bash included.
        (None, (), (), True),
        # ``("",)`` is the CLI's spelling of "disable all tools".
        (("",), (), (), False),
        # ``--allowed-tools`` is an auto-approve list, NOT a grant: it cannot
        # hand back a tool ``--tools ''`` removed. This combination is exactly
        # the one a reader mixes up, so it is pinned.
        (("",), ("Bash", "WebFetch"), (), False),
        (("Read", "Grep"), (), (), True),
        # A restriction naming only MCP tools leaves no native tool behind.
        (("mcp__docs__search",), (), (), False),
        # ...and an unknown name is not assumed to be native either.
        (("NotARealTool",), (), (), False),
        # ``disallowed_tools`` subtracts: covering the whole restricted set is
        # equivalent to enabling none.
        (("Read",), (), ("Read",), False),
        (("Read", "Bash"), (), ("Read",), True),
        # Subtracting one tool from the CLI default set still leaves the rest.
        (None, (), ("Bash",), True),
    ],
)
def test_native_tool_enablement_table(
    tools: tuple[str, ...] | None,
    allowed: tuple[str, ...],
    disallowed: tuple[str, ...],
    expected: bool,
) -> None:
    """``tools`` / ``allowed_tools`` / ``disallowed_tools`` interact; enumerate it."""
    cog = ClaudeCliCognition(tools=tools, allowed_tools=allowed, disallowed_tools=disallowed)
    assert bool(cog.native_tools()) is expected


def test_disallowing_every_builtin_is_equivalent_to_no_native_tools() -> None:
    """``disallowed_tools`` covering the entire built-in table silences the
    warning as surely as ``tools=("",)`` does — otherwise the warning would be
    crying wolf at the most carefully-locked-down configuration there is."""
    cog = ClaudeCliCognition(disallowed_tools=tuple(CLI_TOOL_CAPS))
    assert cog.native_tools() == ()
    assert _bypass_warnings(_warnings_from_drive(cog, _ctx(audit()))) == []


def test_empty_middleware_tuple_and_absent_invoker_do_not_warn() -> None:
    """No middleware, no bypass. An ``Invoker`` with an empty tool chain and a
    ``ctx`` with no invoker at all are the same case, and neither has anything
    to be silently skipped."""
    assert _bypass_warnings(_warnings_from_drive(ClaudeCliCognition(), _ctx())) == []
    assert _bypass_warnings(_warnings_from_drive(ClaudeCliCognition(), make_test_ctx())) == []


def test_chat_only_middleware_does_not_warn() -> None:
    """The chat chain still runs for... nothing, but it is the TOOL chain whose
    absence is the safety hole. Warning about ``chat_middleware`` here would
    name middlewares that were never going to see a tool call anyway."""
    ctx = make_test_ctx(llm=FakeLLM(["ok"]), chat_middleware=[tracing(), meter()])
    assert _bypass_warnings(_warnings_from_drive(ClaudeCliCognition(), ctx)) == []


# ── how often it fires ──────────────────────────────────────────────────────


def test_warns_once_per_cognition_across_many_drives() -> None:
    """Once per cognition, not once per ``drive``. A warning per iteration is
    noise, and noise is what trains people to add a ``filterwarnings`` line —
    which is how the next silent misconfiguration gets through."""
    cog = ClaudeCliCognition(tools=("Read", "Bash"))
    assert len(_bypass_warnings(_warnings_from_drive(cog, _ctx(audit()), drives=4))) == 1


def test_two_cognitions_each_get_their_own_warning() -> None:
    """Per-INSTANCE, not per-process. A process-wide latch would suppress the
    second cognition's warning — and the second cognition is the one nobody has
    looked at yet."""
    ctx = _ctx(audit())
    first = ClaudeCliCognition(tools=("Read", "Bash"))
    second = ClaudeCliCognition(tools=("WebFetch",))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _drive(first, ctx)
        _drive(second, ctx)

    messages = _bypass_warnings(list(caught))
    assert len(messages) == 2, messages
    assert "Bash" in messages[0]
    assert "WebFetch" in messages[1]


# ── capability tags: RunPolicy sees the CLI session now ─────────────────────


def test_declared_caps_use_runpolicy_vocabulary_only() -> None:
    """Tags come from ``TRIFECTA``, not a parallel vocabulary of our own — a tag
    ``RunPolicy.capabilities`` filters out is a tag that protects nothing."""
    assert CLI_TOOL_CAPS
    for name, caps in CLI_TOOL_CAPS.items():
        assert set(caps) <= set(TRIFECTA), f"{name} declares a tag RunPolicy cannot read: {caps}"


def test_read_only_session_is_not_refused() -> None:
    """One leg of the trifecta is not the trifecta. Refusing here would make the
    check useless by making it fire on everything."""
    cog = ClaudeCliCognition(tools=("Read",))
    assert cog.caps == ("private_data",)
    assert RunPolicy().check([cog]).allowed is True


def test_read_webfetch_bash_is_refused_like_the_equivalent_agentkit_tool_set() -> None:
    """The security claim, stated as an equality: the CLI session and the
    agentkit tool set that can do the same three things get the same verdict."""
    cog = ClaudeCliCognition(tools=("Read", "WebFetch", "Bash"))
    equivalent = [
        FunctionTool("read_file", lambda a, c: None, description="d", side_effecting=False, caps=("private_data",)),
        FunctionTool("fetch", lambda a, c: None, description="d", side_effecting=False, caps=("untrusted_content",)),
        FunctionTool("shell", lambda a, c: None, description="d", side_effecting=False, caps=("egress",)),
    ]

    cli_verdict = RunPolicy().check([cog])
    tool_verdict = RunPolicy().check(equivalent)

    assert cli_verdict.allowed is False
    assert cli_verdict.capabilities == tool_verdict.capabilities == tuple(sorted(TRIFECTA))
    with pytest.raises(PermissionError, match="lethal trifecta"):
        RunPolicy(mode="deny").check([cog])


def test_default_tool_set_is_the_trifecta() -> None:
    """``tools=None`` — the default — hands the session every built-in tool.
    That is the trifecta, and it being the DEFAULT is the whole point of making
    it visible."""
    assert RunPolicy().check([ClaudeCliCognition()]).allowed is False


def test_no_native_tools_declares_no_caps() -> None:
    """``tools=("",)`` means the CLI holds nothing; the caps have to follow, or
    the tag would be describing a session that does not exist."""
    cog = ClaudeCliCognition(tools=("",))
    assert cog.caps == ()
    assert RunPolicy().check([cog]).allowed is True


def test_trifecta_reached_by_mcp_tools_plus_native_tools_together() -> None:
    """Neither half is a trifecta alone; together they are. This is the case a
    per-path check misses — the CLI's ``Read`` + ``Bash`` supply private data
    and egress, an MCP-served tool supplies the untrusted content, and only a
    check over BOTH sees it."""
    cog = ClaudeCliCognition(tools=("Read", "Bash"))
    mcp_tool = FunctionTool(
        "ingest_ticket",
        lambda a, c: None,
        description="d",
        side_effecting=False,
        caps=("untrusted_content",),
    )

    assert cog.caps == ("egress", "private_data")
    assert RunPolicy().check([cog]).allowed is True
    assert RunPolicy().check([mcp_tool]).allowed is True
    assert RunPolicy().check([cog, mcp_tool]).allowed is False


def test_a_session_carries_the_cognitions_caps() -> None:
    """``ClaudeCliSession`` can BE an agent's cognition, so it lands in
    ``RunPolicy.check`` wherever the cognition would. Without delegation
    ``getattr(session, "caps", ())`` is empty and the gate sees a tool-less run
    — the same silent hole, one object further out."""
    cog = ClaudeCliCognition(tools=("Read", "WebFetch", "Bash"))
    assert cog.session().caps == cog.caps
    assert RunPolicy().check([cog.session()]).allowed is False


def test_disallowed_tools_narrows_the_caps() -> None:
    """Removing ``WebFetch`` removes the untrusted-content leg it contributed —
    the tags track the session's ACTUAL tool set, not the table."""
    cog = ClaudeCliCognition(tools=("Read", "WebFetch", "Bash"), disallowed_tools=("WebFetch",))
    assert "untrusted_content" not in cog.caps
    assert RunPolicy().check([cog]).allowed is True


# ── review additions: gaps the shipped suite left open ──────────────────────
#
# Six hand-mutants survived the original file. Each test below was written to
# kill one of them, and the mutant is named in the docstring so a future reader
# can re-run it.


def _scope_names(message: str) -> set[str]:
    """The tool names the warning claims the chain is not applied to."""
    scope = message.split("Not applied to ", 1)[1].split(": ", 1)[0]
    return {n.strip() for n in scope.split(",")}


def _session_warnings(cog: ClaudeCliCognition, ctx: Any, *, turns: int = 1) -> list[str]:
    """Every bypass warning raised across ``turns`` turns of ONE session."""
    proc = _FakeSessionProcess([_turn_lines(f"t{i}") for i in range(turns)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _patched(proc):

            async def _go() -> None:
                async with cog.session() as chat:
                    for i in range(turns):
                        await _collect(chat, f"turn {i}", ctx=ctx)

            asyncio.run(_go())
    return _bypass_warnings(list(caught))


def test_a_session_turn_warns_too_not_just_drive() -> None:
    """MUTANT: deleting ``self._cog._warn_if_middleware_bypassed(ctx)`` from
    ``ClaudeCliSession._turn`` survived the original suite entirely.

    A session is the path a chat UI takes, and it is the path where the bypass
    lasts longest — one process, many turns, every native tool call skipping the
    chain. ``drive`` being covered says nothing about ``_turn``: they are two
    separate entry points and only one of them was pinned.
    """
    cog = ClaudeCliCognition(tools=("Read", "Bash"))
    messages = _session_warnings(cog, _ctx(egress(_Guardrail()), audit()))

    assert len(messages) == 1, messages
    assert "Egress" in messages[0]
    assert "Audit" in messages[0]


def test_a_session_warns_on_the_first_turn_and_never_again() -> None:
    """The latch lives on the cognition and a session reuses one, so a long
    conversation pays the warning once. A per-turn warning is the noise the
    design comment argues against, and nothing was checking the session side."""
    cog = ClaudeCliCognition(tools=("Read", "Bash"))
    assert len(_session_warnings(cog, _ctx(audit()), turns=3)) == 1


def test_a_session_serving_everything_over_mcp_does_not_warn() -> None:
    """Same silence guarantee as ``drive``: nothing native, nothing skipped."""
    cog = ClaudeCliCognition(tools=("",), mcp_config=('{"mcpServers": {}}',))
    assert _session_warnings(cog, _ctx(egress(_Guardrail()), audit())) == []


def test_middleware_names_are_the_factory_names_not_closure_qualnames() -> None:
    """MUTANT: ``return qualname`` in place of the ``.<locals>.`` split survived.

    The original test asserted ``"memoize" in message`` — and
    ``"memoize.<locals>.mw"`` contains ``"memoize"``, so that membership check
    passes for the exact output ``_middleware_name`` exists to prevent. Pin the
    tuple instead: it fixes the names, the chain ORDER the docstring promises,
    and the de-duplication, none of which a membership test can see.
    """
    ctx = _ctx(tracing(), meter(), egress(_Guardrail()), idempotent(), audit(), security())

    assert _tool_middleware_names(ctx) == (
        "tracing",
        "MeterMiddleware",
        "Egress",
        "memoize",
        "Audit",
        "SecurityMiddleware",
    )


def test_a_base_middleware_instance_is_named_by_its_class() -> None:
    """The two shapes take different branches of ``_middleware_name`` and only
    one of them was ever reachable through a message-membership assertion."""
    assert _middleware_name(audit()) == "Audit"
    assert _middleware_name(tracing()) == "tracing"


def test_the_emitted_warning_contains_no_closure_qualnames() -> None:
    """The same guarantee stated where the reader actually meets it. A message
    reading ``tracing.<locals>.mw, memoize.<locals>.mw`` is the unreadable output
    the helper was written to avoid, and it passed every original assertion."""
    ctx = _ctx(tracing(), idempotent(), audit())
    message = _bypass_warnings(_warnings_from_drive(ClaudeCliCognition(tools=("Read",)), ctx))[0]

    assert "<locals>" not in message
    assert "mw," not in message


def test_a_repeated_middleware_is_named_once_and_in_chain_order() -> None:
    """MUTANT: ``sorted(set(...))`` for ``dict.fromkeys(...)`` survived.

    The docstring promises "de-duplicated, in chain order" — order so the reader
    can diff the message against the chain they wrote, de-duplication so two
    ``tracing()`` entries read as one name rather than two identical ones."""
    assert _tool_middleware_names(_ctx(tracing(), audit(), tracing())) == ("tracing", "Audit")


@pytest.mark.parametrize("name", ["Task", "SlashCommand"])
def test_an_indirection_tool_is_the_whole_trifecta_on_its_own(name: str) -> None:
    """MUTANT: retagging ``Task``/``SlashCommand`` to ``()`` survived.

    Both are indirections to the full tool set — a subagent, or a user-authored
    command body, can do anything the session can. The table's comment defends
    exactly this and nothing tested it, so laundering the trifecta through one
    innocuous-looking name was free.
    """
    cog = ClaudeCliCognition(tools=(name,))

    assert cog.caps == tuple(sorted(TRIFECTA))
    with pytest.raises(PermissionError, match="lethal trifecta"):
        RunPolicy(mode="deny").check([cog])


def test_a_search_tool_supplies_the_private_data_leg() -> None:
    """MUTANT: stripping ``private_data`` from ``Glob``/``Grep`` survived.

    Only ``Read``, ``WebFetch`` and ``Bash`` were ever exercised, so every other
    row of the table was free to be wrong. ``Grep`` reads the filesystem exactly
    as ``Read`` does; with ``WebFetch`` alongside it that is the full trifecta,
    and no ``Read`` anywhere in the configuration.
    """
    for search in ("Grep", "Glob"):
        cog = ClaudeCliCognition(tools=(search, "WebFetch"))
        assert cog.caps == tuple(sorted(TRIFECTA)), search
        assert RunPolicy().check([cog]).allowed is False, search


def test_the_whole_cap_table_is_pinned() -> None:
    """A ratchet, deliberately brittle: the table IS the security claim, so a new
    CLI tool must not be able to join it — or an existing row to be quietly
    retagged — without someone stating the capability decision out loud here."""
    assert CLI_TOOL_CAPS == {
        "Bash": ("private_data", "egress"),
        "BashOutput": (),
        "Edit": (),
        "ExitPlanMode": (),
        "Glob": ("private_data",),
        "Grep": ("private_data",),
        "KillShell": (),
        "NotebookEdit": (),
        "Read": ("private_data",),
        "SlashCommand": TRIFECTA,
        "Task": TRIFECTA,
        "TodoWrite": (),
        "WebFetch": ("untrusted_content", "egress"),
        "WebSearch": ("untrusted_content", "egress"),
        "Write": (),
    }


def test_native_tools_is_sorted_so_the_warning_text_is_stable() -> None:
    """MUTANT: dropping ``sorted`` from ``native_tools`` survived.

    Not cosmetic: the tool list goes straight into the warning message, so an
    unsorted set makes the text a different string on every interpreter and
    turns any message assertion — here or in a caller's own test — into a
    hash-seed coin flip."""
    cog = ClaudeCliCognition(tools=("WebFetch", "Bash", "Read", "Grep"))

    assert cog.native_tools() == ("Bash", "Grep", "Read", "WebFetch")
    assert "Bash, Grep, Read, WebFetch" in _bypass_warnings(
        _warnings_from_drive(cog, _ctx(audit()))
    )[0]


def test_a_promoted_warning_still_latches() -> None:
    """MUTANT: moving ``self._bypass_warned = True`` to AFTER ``warnings.warn``
    survived.

    Under ``-W error`` the warning becomes an exception that escapes ``drive``.
    If the latch is set afterwards it is never set at all, so every subsequent
    run raises too — a caller who turned warnings into errors to CATCH this one
    would find the cognition permanently unusable instead of warned once.
    """
    cog = ClaudeCliCognition(tools=("Read",))
    ctx = _ctx(audit())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="bypasses the Invoker"):
            _drive(cog, ctx)

    # Latched despite the escape: the second drive is silent, not a second raise.
    assert _bypass_warnings(_warnings_from_drive(cog, ctx)) == []


def test_the_warning_does_not_name_tools_disallowed_tools_removed() -> None:
    """BUG (fixed here): with ``tools=None`` the message said the chain was not
    applied to "every built-in CLI tool" — including the ones the caller had
    explicitly disallowed.

    ``disallowed_tools`` leaves ``tools`` at ``None``, and the scope branch
    tested only ``self.tools is None``, so the most locked-down configuration
    got the most alarming sentence. The reader is being asked to diff this
    message against their own config; naming tools they removed is how they
    learn not to.
    """
    cog = ClaudeCliCognition(disallowed_tools=("Bash", "WebFetch", "Task", "SlashCommand"))
    message = _bypass_warnings(_warnings_from_drive(cog, _ctx(audit())))[0]
    # Compare NAMES, not substrings: "Bash" lives inside "BashOutput", which is
    # a different tool and is legitimately still enabled.
    named = _scope_names(message)

    assert named.isdisjoint({"Bash", "WebFetch", "Task", "SlashCommand"}), named
    assert "Read" in named
    assert "BashOutput" in named
    assert "every built-in CLI tool" not in message


def test_an_untouched_default_set_still_gets_the_prose_shorthand() -> None:
    """The other side of the fix: when nothing was subtracted the session really
    does hold all fifteen, and listing them all would bury the middleware names
    the message exists to deliver."""
    message = _bypass_warnings(_warnings_from_drive(ClaudeCliCognition(), _ctx(audit())))[0]
    assert "every built-in CLI tool" in message


def test_a_scoped_disallow_entry_still_gets_the_prose_shorthand() -> None:
    """``disallowed_tools=("Bash(rm:*)",)`` restricts some Bash invocations, not
    Bash — the session still holds every built-in tool, so the shorthand is
    still true. The fix keys off what was actually subtracted, not off whether
    ``disallowed_tools`` is merely non-empty."""
    cog = ClaudeCliCognition(disallowed_tools=("Bash(rm:*)",))

    assert set(cog.native_tools()) == set(CLI_TOOL_CAPS)
    assert "every built-in CLI tool" in _bypass_warnings(_warnings_from_drive(cog, _ctx(audit())))[0]


def test_an_off_contract_disallowed_tools_does_not_crash_the_first_drive() -> None:
    """REGRESSION: ``native_tools`` ran ``set(self.disallowed_tools)``, so
    ``disallowed_tools=None`` — which every earlier version simply ignored, the
    argv builder testing it only for truthiness — raised ``TypeError: 'NoneType'
    object is not iterable`` out of ``drive``.

    The check is a safety warning bolted to the front of ``drive``; it must not
    be able to take the run down with it.
    """
    cog = ClaudeCliCognition(tools=("Read", "Bash"), disallowed_tools=None)  # type: ignore[arg-type]

    assert cog.native_tools() == ("Bash", "Read")
    assert cog.caps == ("egress", "private_data")
    assert len(_bypass_warnings(_warnings_from_drive(cog, _ctx(audit())))) == 1
