"""``as_cli_agents`` — a ``Skill`` projected into a Claude CLI sub-agent definition.

The tests are organised around the three properties the projection exists to
hold, because each of them is a way the hand-written second copy of a Skill
went wrong:

1. **The tool restriction survives.** A reviewer that is read-only because of
   its tool list must not come out the other side holding the parent session's
   tools. The CLI's sub-agent schema makes ``tools`` OPTIONAL and reads an
   absent key as "inherit everything from the main thread", so the failure here
   is an omission, not a wrong value — nothing in the output looks wrong.
2. **The prompt travels whole.** The sibling defect this guards against is the
   one that truncated a tool description to its docstring's first line; a
   sub-agent whose prompt is paragraph one of five is a different agent.
3. **What cannot be expressed is REFUSED**, by name, at construction — not
   projected into a definition that looks similar and behaves differently.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from agentkit.agents.cognition import (
    ClaudeCliCognition,
    CoordinatorCognition,
    ReActCognition,
    SingleCallCognition,
)
from agentkit.integrations.claude_cli import SkillNotProjectable, as_cli_agents
from agentkit.prompts.prompt import Prompt
from agentkit.skills import Skill
from agentkit.tools import tool


@tool(side_effecting=False)
def fetch_diff(ref: str) -> str:
    """Return the unified diff for a git ref, as a plain string of patch text."""
    return f"diff for {ref}"


@dataclass(slots=True)
class _CustomCognition:
    """A third-party ``Cognition`` implementation the projection has never seen."""

    name: str = "custom"

    def drive(self, agent: Any, task: str, ctx: Any, context: Any) -> AsyncIterator[Any]:
        raise NotImplementedError


def _reviewer(**over: Any) -> Skill:
    """A read-only reviewer, the running example throughout this file."""
    kwargs: dict[str, Any] = {
        "name": "reviewer",
        "description": "Reviews a diff for correctness defects.",
        "prompt": "You are a reviewer. Report defects, never edit.",
        "cognition": ClaudeCliCognition(tools=("Read", "Grep")),
    }
    kwargs.update(over)
    return Skill(**kwargs)


# ── 1. the tool restriction survives ────────────────────────────────────────


def test_a_read_only_skill_keeps_exactly_its_tools() -> None:
    """THE security property. ``Write`` must be absent, and ``tools`` must be
    PRESENT — an omitted key is the CLI's spelling of "inherit the parent's"."""
    agents = as_cli_agents([_reviewer()])
    assert agents["reviewer"]["tools"] == ["Read", "Grep"]
    assert "Write" not in agents["reviewer"]["tools"]


def test_a_tool_free_skill_projects_an_explicit_empty_list() -> None:
    """A ``SingleCallCognition`` has no tool loop at all. The honest projection
    of "no tools" is ``[]``, never a missing key."""
    skill = Skill("summariser", "Summarises.", "Be terse.", SingleCallCognition())
    agents = as_cli_agents([skill])
    assert agents["summariser"]["tools"] == []


def test_a_react_skill_with_no_tools_also_projects_an_explicit_empty_list() -> None:
    """An empty registry is expressible; it is the non-empty one that is not."""
    skill = Skill("thinker", "Thinks.", "Think.", ReActCognition(tools=[]))
    assert as_cli_agents([skill])["thinker"]["tools"] == []


def test_disallowed_tools_are_subtracted_from_the_projected_list() -> None:
    """``--disallowed-tools`` can only ever NARROW an explicit ``--tools`` set,
    and the sub-agent schema has no field for it. Dropping it would widen the
    sub-agent past the skill; subtracting keeps the two descriptions identical."""
    skill = _reviewer(
        cognition=ClaudeCliCognition(tools=("Read", "Grep", "Bash"), disallowed_tools=("Bash",))
    )
    assert as_cli_agents([skill])["reviewer"]["tools"] == ["Read", "Grep"]


def test_the_cli_spelling_of_no_tools_projects_to_an_empty_list() -> None:
    """``tools=("",)`` is the CLI's documented "disable all tools"."""
    skill = _reviewer(cognition=ClaudeCliCognition(tools=("",)))
    assert as_cli_agents([skill])["reviewer"]["tools"] == []


def test_a_skill_that_states_no_tool_restriction_is_refused() -> None:
    """``tools=None`` means "whatever the CLI's default set is". There is no
    finite list to project, and the only shape that expresses it — omitting the
    key — is precisely the inherit-the-parent's-tools widening."""
    skill = _reviewer(cognition=ClaudeCliCognition())
    with pytest.raises(SkillNotProjectable, match="no tool restriction"):
        as_cli_agents([skill])


# ── 2. the prompt travels whole ─────────────────────────────────────────────


def test_the_whole_prompt_survives_not_its_first_line() -> None:
    prompt = (
        "You are a reviewer.\n"
        "\n"
        "Report every defect you find, with a file and a line.\n"
        "\n"
        "Never edit a file. Never run a command that writes.\n"
        "\n"
        "Finish with a one-line verdict: SHIP or BLOCK.\n"
    )
    projected = as_cli_agents([_reviewer(prompt=prompt)])["reviewer"]["prompt"]
    assert projected == prompt
    # Named explicitly: the truncation defect this mirrors kept paragraph one.
    assert "SHIP or BLOCK" in projected
    assert "Never run a command that writes." in projected


def test_a_versioned_prompt_is_rendered_with_its_bindings() -> None:
    """``Prompt`` is the house style. It must arrive SUBSTITUTED — a ``{tenant}``
    that reaches the CLI unrendered is a plausible-looking wrong instruction."""
    prompt = Prompt(
        id="reviewer",
        version="3",
        template="Review for {tenant}.\n\nBlock on any regression.",
        inputs=("tenant",),
    ).bind(tenant="acme")
    projected = as_cli_agents([_reviewer(prompt=prompt)])["reviewer"]["prompt"]
    assert projected == "Review for acme.\n\nBlock on any regression."


def test_an_unbound_prompt_input_fails_loudly_at_projection() -> None:
    prompt = Prompt(id="reviewer", version="3", template="Review for {tenant}.", inputs=("tenant",))
    with pytest.raises(SkillNotProjectable, match="reviewer"):
        as_cli_agents([_reviewer(prompt=prompt)])


def test_an_empty_prompt_is_refused() -> None:
    with pytest.raises(SkillNotProjectable, match="prompt"):
        as_cli_agents([_reviewer(prompt="   ")])


def test_an_empty_description_is_refused() -> None:
    """The CLI routes delegation on ``description``; an empty one is a sub-agent
    the parent can never decide to call."""
    with pytest.raises(SkillNotProjectable, match="description"):
        as_cli_agents([_reviewer(description="")])


# ── 3. what cannot be expressed is refused BY NAME ──────────────────────────


def test_coordinator_cognition_is_refused_by_name() -> None:
    skill = _reviewer(cognition=CoordinatorCognition(children={}, policy=object()))  # type: ignore[arg-type]
    with pytest.raises(SkillNotProjectable, match="CoordinatorCognition"):
        as_cli_agents([skill])


def test_a_react_skill_carrying_agentkit_tools_is_refused_naming_the_tool() -> None:
    """The F1 dependency. A Python callable cannot be reached from inside the
    ``claude`` subprocess; that needs an MCP server, which this does not build."""
    skill = _reviewer(cognition=ReActCognition(tools=[fetch_diff]))
    with pytest.raises(SkillNotProjectable) as exc:
        as_cli_agents([skill])
    assert "fetch_diff" in str(exc.value)
    assert "MCP" in str(exc.value)


def test_an_unknown_cognition_is_refused_by_its_class_name() -> None:
    with pytest.raises(SkillNotProjectable, match="_CustomCognition"):
        as_cli_agents([_reviewer(cognition=_CustomCognition())])


def test_memory_is_refused_rather_than_dropped() -> None:
    """A ``MemorySource`` is an in-process object the RequestBuilder grounds
    against. Dropping it yields a sub-agent that answers the same question
    without the grounding — the silent behavioural difference this refuses."""
    from agentkit.testing import FakeMemory

    with pytest.raises(SkillNotProjectable, match="memory"):
        as_cli_agents([_reviewer(memory=FakeMemory())])


# ── model ───────────────────────────────────────────────────────────────────


def test_a_skill_model_projects() -> None:
    assert as_cli_agents([_reviewer(model="haiku")])["reviewer"]["model"] == "haiku"


def test_no_model_omits_the_key() -> None:
    """Omitted means "inherit the parent's model", which for a model is the
    right default — unlike ``tools``, inheriting here widens nothing."""
    assert "model" not in as_cli_agents([_reviewer()])["reviewer"]


def test_the_cognition_model_wins_over_the_skill_model() -> None:
    """Matches what a RUN does: ``ClaudeCliCognition.model`` is documented as
    winning outright ("the agent's ``model`` field is NOT consulted")."""
    skill = _reviewer(
        model="opus", cognition=ClaudeCliCognition(model="claude-haiku-4-5", tools=("Read",))
    )
    assert as_cli_agents([skill])["reviewer"]["model"] == "claude-haiku-4-5"


def test_a_model_the_cli_cannot_reach_is_refused() -> None:
    """A Skill wired for the agentkit provider chain carries a provider model
    id. Projected verbatim it becomes a sub-agent that dies at CLI startup."""
    with pytest.raises(SkillNotProjectable, match="gpt-4o-mini"):
        as_cli_agents([_reviewer(model="gpt-4o-mini")])


# ── roster-level behaviour ──────────────────────────────────────────────────


def test_two_skills_project_independently() -> None:
    repairer = Skill(
        name="repairer",
        description="Applies a fix.",
        prompt="You fix what the reviewer found.",
        cognition=ClaudeCliCognition(tools=("Read", "Edit", "Write")),
    )
    agents = as_cli_agents([_reviewer(), repairer])
    assert set(agents) == {"reviewer", "repairer"}
    assert agents["reviewer"]["tools"] == ["Read", "Grep"]
    assert agents["repairer"]["tools"] == ["Read", "Edit", "Write"]


def test_an_empty_roster_projects_to_an_empty_dict() -> None:
    assert as_cli_agents([]) == {}


def test_a_duplicate_name_is_refused_rather_than_last_wins() -> None:
    """Last-wins would silently drop a sub-agent the caller wired on purpose —
    the same class of invisible difference every other refusal here guards."""
    with pytest.raises(SkillNotProjectable, match="twice"):
        as_cli_agents([_reviewer(), _reviewer(prompt="A different reviewer.")])


def test_a_name_that_is_not_a_valid_cli_identifier_is_refused() -> None:
    """CLI sub-agent names are lowercase-and-hyphens. Normalising instead would
    key the roster on a name the Skill does not have."""
    with pytest.raises(SkillNotProjectable, match="Code_Reviewer"):
        as_cli_agents([_reviewer(name="Code_Reviewer")])


def test_a_roster_too_large_for_one_argv_entry_fails_loudly() -> None:
    """``--agents <json>`` is ONE argv entry and Linux caps a single argument at
    128 KiB (MAX_ARG_STRLEN). Past that the spawn dies with E2BIG three seconds
    into a run instead of at the wiring line that caused it."""
    with pytest.raises(SkillNotProjectable, match="131072"):
        as_cli_agents([_reviewer(prompt="x" * 200_000)])


def test_the_projection_is_stable_across_calls() -> None:
    skill = _reviewer()
    assert as_cli_agents([skill]) == as_cli_agents([skill])
    assert json.dumps(as_cli_agents([skill])) == json.dumps(as_cli_agents([skill]))


# ── the seam it has to fit ──────────────────────────────────────────────────


def test_the_result_is_accepted_by_claude_cli_cognition() -> None:
    """The whole point of the shape: it goes straight into ``agents=`` and comes
    back out of ``--agents`` as the same JSON."""
    agents = as_cli_agents([_reviewer()])
    cog = ClaudeCliCognition(agents=agents)
    argv = cog._build_argv("review the diff", system_prompt="")
    assert json.loads(argv[argv.index("--agents") + 1]) == {
        "reviewer": {
            "description": "Reviews a diff for correctness defects.",
            "prompt": "You are a reviewer. Report defects, never edit.",
            "tools": ["Read", "Grep"],
        }
    }


# ── review wave: gaps the first suite left open ─────────────────────────────


def test_a_wildcard_disallow_is_refused_not_silently_dropped() -> None:
    """``disallowed_tools`` is NOT limited to bare names — ``ClaudeCliCognition``'s
    own ``__post_init__`` recommends the spelling ``disallowed_tools=('mcp__*',)``.

    Exact-name subtraction cannot honour a wildcard, and dropping it projects a
    sub-agent still holding every ``mcp__`` tool the skill took away. That is a
    WIDENING past the skill, and the projected JSON looks entirely correct.
    """
    skill = _reviewer(
        cognition=ClaudeCliCognition(
            tools=("Read", "mcp__db__query", "mcp__fs__write"), disallowed_tools=("mcp__*",)
        )
    )
    with pytest.raises(SkillNotProjectable) as exc:
        as_cli_agents([skill])
    assert "mcp__*" in str(exc.value)


def test_a_within_tool_specifier_disallow_is_refused() -> None:
    """``Bash(rm:*)`` narrows INSIDE one tool. A sub-agent has ``Bash`` whole or
    not at all, so this is inexpressible — and dropping it projected full Bash."""
    skill = _reviewer(
        cognition=ClaudeCliCognition(tools=("Read", "Bash"), disallowed_tools=("Bash(rm:*)",))
    )
    with pytest.raises(SkillNotProjectable, match=r"Bash\(rm:\*\)"):
        as_cli_agents([skill])


def test_a_pattern_that_bites_nothing_projected_is_not_a_false_alarm() -> None:
    """The refusal is scoped to patterns that actually reach a projected tool.
    A skill that never had Bash loses nothing by disallowing ``Bash(rm:*)``."""
    skill = _reviewer(
        cognition=ClaudeCliCognition(tools=("Read", "Grep"), disallowed_tools=("Bash(rm:*)",))
    )
    assert as_cli_agents([skill])["reviewer"]["tools"] == ["Read", "Grep"]


def test_a_react_registry_of_an_unknown_shape_is_refused_not_read_as_empty() -> None:
    """``ReActCognition.tools`` is annotated ``Any`` and only list/tuple is wrapped
    by ``__post_init__``, so an object with no ``names()`` genuinely arrives here.

    Reading that as "no tools" made the F1-seam refusal below stop guarding and
    silently dropped every tool the caller wired, into a definition that still
    looks well-formed.
    """
    cognition = ReActCognition(tools=[fetch_diff])
    cognition.tools = object()  # an unrecognised registry shape
    with pytest.raises(SkillNotProjectable, match="not a ToolRegistry"):
        as_cli_agents([_reviewer(cognition=cognition)])


def test_a_react_cognition_with_tools_none_is_refused() -> None:
    """``ReActCognition(tools=None)`` constructs (the field is ``Any``). It must
    not project as a tool-free sub-agent on the strength of a missing method."""
    with pytest.raises(SkillNotProjectable, match="not a ToolRegistry"):
        as_cli_agents([_reviewer(cognition=ReActCognition(tools=None))])


def test_the_name_check_is_anchored_at_both_ends() -> None:
    """The suite's only invalid name failed at position 0, so an unanchored
    ``match()`` would have passed every one of these. Trailing garbage is the
    interesting half: ``reviewer_v2`` starts out perfectly valid."""
    for bad in ("reviewer_v2", "reviewer!", "reviewer ", "reviewer-", "-reviewer", ""):
        with pytest.raises(SkillNotProjectable):
            as_cli_agents([_reviewer(name=bad)])


def test_a_whitespace_only_description_is_refused_like_an_empty_one() -> None:
    """The empty-description test used ``""``, which is falsy either way; only a
    whitespace-only one pins the ``.strip()``."""
    with pytest.raises(SkillNotProjectable, match="no description"):
        as_cli_agents([_reviewer(description="   \n\t ")])


def test_the_projected_description_is_stripped() -> None:
    assert as_cli_agents([_reviewer(description="  Reviews a diff.  ")])["reviewer"][
        "description"
    ] == "Reviews a diff."


def test_two_projections_share_no_mutable_state() -> None:
    """The roster is handed to a long-lived cognition. Mutating one projection's
    tool list must not reach through to another's."""
    skill = _reviewer()
    first = as_cli_agents([skill])
    second = as_cli_agents([skill])
    first["reviewer"]["tools"].append("Write")
    assert second["reviewer"]["tools"] == ["Read", "Grep"]
