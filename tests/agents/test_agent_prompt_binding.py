"""An `Agent` refuses a `Prompt` whose declared inputs are not all bound.

Nothing in the framework passes prompt values at call time — `RequestBuilder`
and both CLI cognitions call `render()` with no arguments. So an unbound
`inputs=` prompt is not a maybe-problem: it is a guaranteed `ValueError` on the
first drive. Before this check it surfaced mid-run, after the run had started;
now it surfaces at construction, for the same reason `check_capabilities` does
— the value of the check is catching it before spend.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit.agents import Agent
from agentkit.prompts import Prompt
from agentkit.testing import FakeLLM, make_test_ctx

BRIEFER = Prompt(
    id="briefer",
    version="1.0.0",
    inputs=("tenant", "tone"),
    template="Brief tenant {tenant} in a {tone} tone.",
)


def test_an_unbound_prompt_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="unbound"):
        Agent(name="b", model="m", prompt=BRIEFER)


def test_a_partially_bound_prompt_names_only_the_missing_inputs() -> None:
    """The error has to be actionable — listing every input when one is bound
    sends the reader looking in the wrong place."""
    with pytest.raises(ValueError) as ei:
        Agent(name="b", model="m", prompt=BRIEFER.bind(tenant="acme"))
    assert "['tone']" in str(ei.value)
    assert "tenant" not in str(ei.value).split("has ")[1]


def test_the_error_shows_how_to_fix_it() -> None:
    with pytest.raises(ValueError, match=r"\.bind\(tenant=\.\.\., tone=\.\.\.\)"):
        Agent(name="b", model="m", prompt=BRIEFER)


# ── positive controls: the check must not fire on anything that already worked ──


def test_a_fully_bound_prompt_is_accepted_and_renders_substituted() -> None:
    agent = Agent(name="b", model="m", prompt=BRIEFER.bind(tenant="acme", tone="terse"))
    assert agent.prompt.render() == "Brief tenant acme in a terse tone."  # type: ignore[union-attr]


def test_a_plain_string_prompt_is_unaffected() -> None:
    Agent(name="b", model="m", prompt="just a system prompt")


def test_no_prompt_at_all_is_unaffected() -> None:
    """`Agent("a", "m")` — the ergonomic form — must keep working."""
    Agent(name="b", model="m")


def test_a_prompt_declaring_no_inputs_is_unaffected() -> None:
    Agent(name="b", model="m", prompt=Prompt(id="x", version="1", template="hi there"))


def test_a_bound_agent_runs_end_to_end() -> None:
    agent = Agent(name="b", model="m", prompt=BRIEFER.bind(tenant="acme", tone="terse"))
    result = asyncio.run(agent.run("go", make_test_ctx(llm=FakeLLM("ok"))))
    assert result.output == "ok"
    assert result.prompt_version == "1.0.0", "a bound prompt keeps its version for attribution"


# ── the mutable-dataclass escape hatch ─────────────────────────────────────────


def test_reassigning_prompt_bypasses_the_check_and_check_prompt_reasserts() -> None:
    """`Agent` is a mutable dataclass, so assignment skips `__post_init__` —
    exactly the caveat `check_capabilities` documents. `check_prompt()` is
    public so a caller can re-assert."""
    agent = Agent(name="b", model="m", prompt=BRIEFER.bind(tenant="acme", tone="terse"))
    agent.prompt = BRIEFER  # unbound again; no raise on assignment
    with pytest.raises(ValueError, match="unbound"):
        agent.check_prompt()
