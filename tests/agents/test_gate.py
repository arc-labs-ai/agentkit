"""HumanGate policy (ch17 / R10; ADR-0003): `should_gate` decides whether an action pauses for a
human given the run autonomy + per-action flags. Pure, offline, deterministic."""

from agentkit.agents import Autonomy, Suspended
from agentkit.agents.control.gate import should_gate


def test_manual_gates_everything():
    assert should_gate(Autonomy.MANUAL)
    assert should_gate(Autonomy.MANUAL, requires_approval=False, key_step=False)
    assert should_gate(Autonomy.MANUAL, key_step=True)


def test_gated_gates_required_and_key_steps_only():
    assert not should_gate(Autonomy.GATED)  # ordinary action runs
    assert should_gate(Autonomy.GATED, requires_approval=True)  # tool author demanded it
    assert should_gate(Autonomy.GATED, key_step=True)  # a "key" step


def test_auto_gates_only_explicitly_required():
    assert not should_gate(Autonomy.AUTO)
    # autonomy only ADDS gating; AUTO ignores key_step
    assert not should_gate(Autonomy.AUTO, key_step=True)
    # tool author's flag is always honoured
    assert should_gate(Autonomy.AUTO, requires_approval=True)


def test_autonomy_enum_members_and_suspended_reexported():
    """The Autonomy enum carries the three tiers as string-valued members, and Suspended
    (the suspend/resume mechanism) remains re-exported alongside the gate policy."""
    assert tuple(Autonomy) == (Autonomy.AUTO, Autonomy.GATED, Autonomy.MANUAL)
    assert Autonomy.AUTO.value == "auto"
    assert Autonomy.GATED.value == "gated"
    assert Autonomy.MANUAL.value == "manual"
    s = Suspended(run_id="r", pending=[])
    assert s.reason == "awaiting_approval"


def test_should_gate_accepts_string_or_enum():
    """`should_gate` takes either the Autonomy enum OR its string value — the run-wide
    `RunContext.autonomy` is a `str` field, so the gate must honour both call shapes."""
    assert should_gate("manual")
    assert should_gate(Autonomy.MANUAL)
    assert should_gate("gated", requires_approval=True)
    assert should_gate(Autonomy.GATED, requires_approval=True)
    assert not should_gate("auto")
    assert not should_gate(Autonomy.AUTO)
