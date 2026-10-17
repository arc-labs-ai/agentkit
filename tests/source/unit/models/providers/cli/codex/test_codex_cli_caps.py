"""What a Codex session reports it can DO, in the Rule-of-Two vocabulary.

WHY THIS FILE EXISTS
--------------------
Codex has no tool allow-list. Every session gets the same ``shell``,
``apply_patch`` and ``update_plan``, and the only containment is the OS sandbox,
so the capability tags a policy gate reads cannot be derived from a tool table
the way the Claude cognition's are — they come from ``--sandbox``,
``network_access`` and ``--search``.

That makes two answers load-bearing in a way neither looks:

* An UNSET ``sandbox`` is not "no capabilities". ``codex exec`` documents
  itself as read-only by default, and a caps property that answered "nothing"
  for the commonest wiring there is — a cognition nobody configured — would
  leave the lethal-trifecta gate blind on precisely the runs nobody audited.
* ``--search`` contributes ``untrusted_content`` as well as ``egress``. Without
  it, ``sandbox="read-only", web_search=True`` — private data, a model reading
  the open web, and a network to send it over, which is the most
  innocuous-looking configuration Codex has — stops being refusable.

The tags are also what makes a session usable where the cognition is, so the
delegation is pinned here too.
"""

from __future__ import annotations

import pytest

from agentkit.models.providers.cli.codex.driver import (
    CODEX_SANDBOX_CAPS,
    CodexCliCognition,
)

# The three tags that together are the lethal trifecta: the agent can read
# private data, it can ingest content an attacker controls, and it has a way
# out. Two of the three is a session worth running; all three is the shape a
# ``RunPolicy(mode="deny")`` exists to refuse.
_TRIFECTA = frozenset({"private_data", "untrusted_content", "egress"})


def test_an_unset_sandbox_reports_the_clis_own_read_only_default_rather_than_nothing() -> None:
    """``CodexCliCognition()`` with nothing configured is the commonest wiring in
    existence, and the CLI still runs it — read-only. Reporting the default as
    "unknown" or as full access are both wrong in the dangerous direction: the
    first makes the trifecta gate see a harmless run, the second makes it refuse
    every unconfigured one until someone turns the gate off."""
    cognition = CodexCliCognition()

    assert cognition.effective_sandbox == "read-only"
    assert cognition.caps == ("private_data",)
    assert "egress" not in cognition.caps


def test_web_search_contributes_untrusted_content_so_read_only_plus_search_is_refusable() -> None:
    """A web search is untrusted content AND a network call, by definition. Drop
    either tag and ``sandbox="read-only", web_search=True`` — which reads as the
    safest thing you can switch on — no longer completes the trifecta, so the
    gate passes the one configuration it was built to catch."""
    cognition = CodexCliCognition(sandbox="read-only", web_search=True)

    assert set(cognition.caps) >= _TRIFECTA
    assert cognition.caps == ("egress", "private_data", "untrusted_content")


def test_a_read_only_sandbox_without_search_is_not_the_trifecta() -> None:
    """The companion to the test above, and the reason it is worth having: a
    gate that refused a plain read-only session would be refusing the
    recommended configuration, and would be switched off within a week."""
    assert set(CodexCliCognition(sandbox="read-only").caps) < _TRIFECTA


def test_bypass_sandbox_reports_itself_as_full_access_because_that_is_what_it_is() -> None:
    """``--dangerously-bypass-approvals-and-sandbox`` turns the sandbox off, so
    reporting the ``sandbox`` field's ``None`` would describe a containment that
    is not in force — the one case where "no flag was passed" must NOT resolve
    to the CLI's default."""
    cognition = CodexCliCognition(bypass_sandbox=True)

    assert cognition.effective_sandbox == "danger-full-access"
    assert "egress" in cognition.caps


def test_network_access_adds_egress_to_the_workspace_write_sandbox_it_reopens() -> None:
    """``sandbox_workspace_write.network_access`` is exactly the flag that turns
    that sandbox's network back on, so the tag has to follow it. Reading the
    sandbox name alone would report a networked session as air-gapped."""
    assert CodexCliCognition(sandbox="workspace-write").caps == ("private_data",)
    assert CodexCliCognition(sandbox="workspace-write", network_access=True).caps == ("egress", "private_data")


def test_every_sandbox_mode_can_read_the_workspace_because_that_is_what_the_agent_is_for() -> None:
    """"read-only" is read-only about WRITES; it is not blind. A table that
    withheld ``private_data`` from it would make the two safest modes look like
    they hold nothing worth exfiltrating."""
    assert set(CODEX_SANDBOX_CAPS) == {"read-only", "workspace-write", "danger-full-access"}
    for mode, tags in CODEX_SANDBOX_CAPS.items():
        assert "private_data" in tags, mode


def test_a_session_delegates_its_caps_so_a_policy_gate_does_not_see_a_tool_less_run() -> None:
    """A session can BE an agent's cognition, so it gets handed to the gate
    wherever the cognition would be. Without the delegation ``getattr(session,
    "caps", ())`` is the empty tuple, and the gate waves through a conversation
    that holds every capability the cognition does."""
    cognition = CodexCliCognition(sandbox="read-only", web_search=True)
    session = cognition.session()

    assert session.caps == cognition.caps
    assert set(session.caps) >= _TRIFECTA


def test_a_deny_native_tool_policy_is_refused_because_codex_cannot_satisfy_it() -> None:
    """Every Codex session has ``shell``, there is no allow-list and no
    PreToolUse hook, so "no ungovernable tool may run" cannot be configured
    here — it can only be declined. Accepting the policy and configuring
    nothing would be a stated guarantee that silently does nothing, which is
    worse than the refusal."""
    with pytest.raises(ValueError, match="native_tool_policy='deny'"):
        CodexCliCognition(native_tool_policy="deny")


def test_network_access_is_refused_under_a_sandbox_where_it_states_nothing_true() -> None:
    """Under ``read-only`` the flag does nothing; under ``danger-full-access``
    the network is already open and the flag reads as a restriction that is not
    there. Either way the caller has written down a containment decision that
    will not be in force, which is the thing construction-time checks exist
    for."""
    with pytest.raises(ValueError, match="network_access"):
        CodexCliCognition(sandbox="read-only", network_access=True)
