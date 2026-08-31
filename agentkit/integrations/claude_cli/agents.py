"""Project a ``Skill`` into a Claude CLI sub-agent definition.

``Skill`` is described in its own docstring as "the missing primitive between
``Tool`` and ``Agent`` — prompt + cognition + memory as one wirable unit".
``ClaudeCliCognition.agents`` is a roster of sub-agent definitions the CLI can
delegate to, each of them a prompt plus a tool restriction plus a description.
Those are the same idea at two ends of one wire, and until this module nothing
joined them: an application that had expressed a reviewer as a ``Skill`` had to
restate it as a CLI agent definition by hand — the second description of one
thing that every other part of agentkit is careful to avoid.

    from agentkit.agents.cognition import ClaudeCliCognition
    from agentkit.integrations.claude_cli import as_cli_agents

    cognition = ClaudeCliCognition(
        agents=as_cli_agents([reviewer_skill, repairer_skill]),
    )

The projection is deliberately PARTIAL and loudly so. Most of what a ``Skill``
can hold has no sub-agent equivalent, and the design rule throughout this file
is that anything inexpressible raises :class:`SkillNotProjectable` at the call
site rather than being dropped into a definition that looks similar and behaves
differently. That is not defensiveness; it is the one property the feature is
for. The CLI's sub-agent schema makes ``tools`` OPTIONAL and reads an absent
key as "inherit every tool from the main thread", so the archetypal failure is
an OMISSION — a read-only reviewer that silently acquires ``Write`` produces a
definition in which nothing looks wrong.

Which cognitions project, and why
---------------------------------
``SingleCallCognition``
    Yes. One chat call, no tool loop, so the honest projection of its tool set
    is ``[]`` — written out explicitly, never omitted.

``ClaudeCliCognition``
    Yes, and it is the ONLY cognition that can carry a real tool restriction
    across today. Its ``tools`` tuple is already in the CLI's own vocabulary
    (``"Read"``, ``"Grep"``), so the projection is a rename rather than a
    translation. Nesting one as a sub-agent does not spawn a second CLI: the
    parent session runs the sub-agent itself, so this flattens a would-be
    subprocess-inside-a-subprocess into the thing the CLI already knows how to
    run.

``ReActCognition``
    Only when its registry is EMPTY. Its tools are agentkit ``Tool`` objects —
    Python callables living in this process — and the ``claude`` subprocess has
    no way to call one. See the F1 seam below.

``CoordinatorCognition``
    No. It drives a roster of child ``Agent``s through a ``Policy`` that owns
    who-speaks-next, termination, and aggregation. A sub-agent definition has
    no roster field and no policy field, so the projection would silently
    collapse a multi-agent skill into one flat CLI agent: same name, same
    prompt, different behaviour. Refused by name.

Anything else — a third-party ``Cognition`` implementation — is refused by its
class name, because the projection cannot know what it does.

What does not project
---------------------
``Skill.memory``
    Refused, not dropped. A ``MemorySource`` is an in-process object the
    ``RequestBuilder`` auto-grounds against; a CLI sub-agent runs inside a
    subprocess with no channel back to it. Dropping it yields a sub-agent that
    answers the same question ungrounded, which reads as a quality regression
    rather than as the wiring bug it is.

``ClaudeCliCognition.allowed_tools``
    Not projected, and safe to skip: ``--allowed-tools`` is an auto-approve
    list, not a restriction, so losing it can only make the sub-agent prompt
    for permission MORE often. It never widens what the sub-agent may do.

``ClaudeCliCognition.disallowed_tools``
    Subtracted from the projected list when it is a bare tool name. The
    sub-agent schema has no field for it, and it can only ever narrow an
    already-explicit ``tools`` set, so subtracting is what keeps the two
    descriptions identical. Entries that exact-name subtraction CANNOT honour —
    a wildcard (``mcp__*``) or a within-tool specifier (``Bash(rm:*)``) that
    still bites a projected tool — are refused rather than dropped, because
    dropping one hands the sub-agent a tool the skill took away and leaves
    nothing wrong-looking in the JSON.

Everything else on the cognition (working_dir, permission_mode, budgets,
session identity, …) belongs to the SESSION, which the parent owns. A sub-agent
does not get its own.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from agentkit.agents.cognition import (
    ClaudeCliCognition,
    CoordinatorCognition,
    ReActCognition,
    SingleCallCognition,
)
from agentkit.prompts.prompt import Prompt
from agentkit.skills import Skill

# The CLI documents a sub-agent name as an identifier of lowercase letters,
# digits and hyphens. Anchored with ``fullmatch`` below.
_CLI_AGENT_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# ``--agents <json>`` is a SINGLE argv entry (see ``ClaudeCliCognition._build_argv``:
# ``argv += ["--agents", json.dumps(self.agents)]``). Linux caps one argument at
# MAX_ARG_STRLEN = 32 pages = 131072 bytes, independently of the much larger
# total ARG_MAX; exceeding it fails the spawn with E2BIG several seconds into a
# run, with an error that names neither the flag nor the skill that bloated it.
# Checking the serialised roster here moves that failure to the wiring line.
_MAX_AGENTS_JSON_BYTES = 131072

# The CLI's model aliases for a sub-agent. Anything else is accepted only if it
# names a Claude model, so the Bedrock/Vertex spellings
# (``us.anthropic.claude-…``, ``claude-sonnet-4-5@20250929``) pass too.
_CLI_MODEL_ALIASES = frozenset({"opus", "sonnet", "haiku", "inherit"})


class SkillNotProjectable(ValueError):
    """A ``Skill`` cannot be expressed as a Claude CLI sub-agent definition.

    Raised at projection time — wiring time, not run time — which is the entire
    point: every case this covers would otherwise have produced a syntactically
    valid definition that runs and does the wrong thing. A read-only reviewer
    that inherits ``Write``; a coordinator flattened to a single agent; a
    memory-grounded skill answering ungrounded; a provider model id the CLI
    cannot resolve. None of those are visible in the output dict.

    Subclasses ``ValueError`` for the same reason ``ToolDefinitionError`` does:
    the failure mode genuinely is a bad value, and callers already isolating
    wiring code with ``except ValueError`` keep working.

    The message always names the skill, so a refusal from a ten-skill roster
    points at the one line to change.
    """


def as_cli_agents(skills: Iterable[Skill]) -> dict[str, dict[str, Any]]:
    """Project ``skills`` into the ``agents=`` roster ``ClaudeCliCognition`` takes.

    Returns ``{skill.name: {"description": …, "prompt": …, "tools": [...],
    "model": …}}`` — the shape the CLI's ``--agents`` flag parses, with
    ``model`` present only when the skill pins one. ``tools`` is ALWAYS present,
    which is the module docstring's security property expressed in one line of
    code.

    The roster is keyed on ``skill.name``, the field ``Skill`` documents as its
    "stable identifier" and already uses as the ``Tool`` name at ``as_tool()``
    time. A duplicate is refused rather than resolved last-wins: last-wins
    drops a sub-agent the caller deliberately wired, and the loss is invisible
    in a dict that still looks well-formed. An empty iterable projects to
    ``{}`` — a legitimate degenerate case (a filtered roster), and falsy, so
    ``agents=as_cli_agents(xs) or None`` skips the flag entirely.

    Deterministic and side-effect free: the same skills project to an equal
    dict, in input order, every time. Callers cache the result at wire-time.

    Raises:
        SkillNotProjectable: on anything the sub-agent schema cannot express.
            See the module docstring for the full list and the reasoning.
    """
    roster: dict[str, dict[str, Any]] = {}
    for skill in skills:
        name = skill.name
        if not _CLI_AGENT_NAME.fullmatch(name):
            raise SkillNotProjectable(
                f"skill {name!r} cannot be a CLI sub-agent: a sub-agent name must be "
                "lowercase letters, digits and hyphens (e.g. 'code-reviewer'). Normalising "
                "it here would key the roster on a name the Skill does not have, so the "
                "Skill has to be renamed."
            )
        if name in roster:
            raise SkillNotProjectable(
                f"skill {name!r} appears twice in the roster. CLI sub-agents are keyed by "
                "name, so the second would silently replace the first; rename one."
            )
        roster[name] = _project(skill)

    # Serialised exactly as ``_build_argv`` will serialise it, so the measured
    # size is the one the spawn actually sees.
    size = len(json.dumps(roster).encode("utf-8"))
    if size > _MAX_AGENTS_JSON_BYTES:
        raise SkillNotProjectable(
            f"the projected roster serialises to {size} bytes. The CLI receives --agents as a "
            f"single argv entry and Linux caps one argument at {_MAX_AGENTS_JSON_BYTES} bytes "
            "(MAX_ARG_STRLEN), so this would die with E2BIG mid-spawn. Shorten the prompts, or "
            "split the roster across sessions."
        )
    return roster


def _project(skill: Skill) -> dict[str, Any]:
    """One skill → one sub-agent definition.

    Order of checks is deliberate: the tool restriction is validated FIRST,
    before the cheap string checks, because it is the one whose failure is
    silent. A skill that is both memory-grounded and unprojectably-tooled
    should report the tool problem — that is the one that would have shipped.
    """
    tools = _cli_tools(skill)

    description = skill.description.strip()
    if not description:
        raise SkillNotProjectable(
            f"skill {skill.name!r} has no description. The CLI routes delegation on it — a "
            "sub-agent with an empty description is one the parent can never decide to call."
        )

    prompt = _prompt_text(skill)
    if not prompt.strip():
        raise SkillNotProjectable(
            f"skill {skill.name!r} has no prompt. The prompt IS the sub-agent; without one the "
            "definition adds a name and a description to the CLI's own default behaviour."
        )

    if skill.memory is not None:
        raise SkillNotProjectable(
            f"skill {skill.name!r} carries memory ({type(skill.memory).__name__}), which cannot "
            "cross into a CLI sub-agent: a MemorySource is an in-process object the "
            "RequestBuilder grounds against, and the sub-agent runs inside the `claude` "
            "subprocess with no channel back to it. Refused rather than dropped, because a "
            "sub-agent answering ungrounded reads as a quality regression, not as a wiring bug."
        )

    definition: dict[str, Any] = {
        "description": description,
        # Whole, never truncated. The sibling defect this mirrors cut a tool
        # description down to its docstring's first line; a sub-agent given
        # paragraph one of five is a different agent wearing the same name.
        "prompt": prompt,
        # Always emitted. An absent key means "inherit the parent's tools".
        "tools": tools,
    }
    model = _cli_model(skill)
    if model is not None:
        definition["model"] = model
    return definition


def _prompt_text(skill: Skill) -> str:
    """Render ``skill.prompt`` to the text the sub-agent will actually receive.

    ``Skill.prompt`` is ``Prompt | str``, and the versioned ``Prompt`` is the
    house style, so this has to render rather than ``str()``. Rendering can
    raise — an input that is neither bound nor supplied is a ``ValueError`` —
    and that failure is re-raised as a refusal because the alternative
    (``render`` leaving ``{tenant}`` in place, which it explicitly refuses to
    do) would be a plausible-looking prompt describing the wrong task.
    """
    if isinstance(skill.prompt, Prompt):
        try:
            return skill.prompt.render()
        except ValueError as exc:
            raise SkillNotProjectable(
                f"skill {skill.name!r} has a Prompt that will not render: {exc}. Bind its inputs "
                "before projecting — a sub-agent definition is built once at wiring time and "
                "has no later call site to supply them."
            ) from exc
    return skill.prompt


def _cli_tools(skill: Skill) -> list[str]:
    """The sub-agent's ``tools`` list, or a refusal naming the cognition.

    Never returns ``None`` and is never allowed to be skipped by the caller:
    the whole security property is that this list is always written down.
    """
    cognition = skill.cognition

    if isinstance(cognition, ClaudeCliCognition):
        if cognition.tools is None:
            raise SkillNotProjectable(
                f"skill {skill.name!r} states no tool restriction (ClaudeCliCognition.tools is "
                "None, meaning 'whatever the CLI's default set is'). There is no finite list to "
                "project, and the only shape that would express it — omitting `tools` — is read "
                "by the CLI as 'inherit every tool from the parent session', which is exactly "
                "the widening this projection exists to prevent. Name the tools explicitly, or "
                "pass tools=(\"\",) for none."
            )
        # ``("",)`` is the CLI's documented spelling of "disable all tools", so
        # the empty string is a marker rather than a tool name and must not
        # reach the JSON. ``disallowed_tools`` is subtracted here: the sub-agent
        # schema has no field for it and it can only narrow, so honouring it
        # keeps the projection identical to what the skill would run.
        disallowed = set(cognition.disallowed_tools)
        tools = [t for t in cognition.tools if t and t not in disallowed]
        # Exact-name subtraction is the ONLY narrowing this projection can
        # perform, and ``--disallowed-tools`` is not limited to bare names.
        # A pattern entry that still bites after subtraction has been silently
        # discarded, which widens the sub-agent past the skill — see
        # ``_unsubtractable``.
        lost = _unsubtractable(tools, cognition.disallowed_tools)
        if lost:
            raise SkillNotProjectable(
                f"skill {skill.name!r} disallows {lost}, which a CLI sub-agent definition cannot "
                f"express: its only tool field is the `tools` allow-list {tools}, so a pattern "
                "(`mcp__*`) or a within-tool specifier (`Bash(rm:*)`) has nowhere to go. Dropping "
                "it would hand the sub-agent a tool the skill explicitly took away, and nothing "
                "in the projected JSON would look wrong. Name the surviving tools exactly in "
                "ClaudeCliCognition(tools=...) instead."
            )
        return tools

    if isinstance(cognition, SingleCallCognition):
        # No tool loop exists on this cognition, so its tool set is empty — and
        # writing that down is the point.
        return []

    if isinstance(cognition, ReActCognition):
        # ``ReActCognition.tools`` is a ``ToolRegistry`` after ``__post_init__``
        # (a plain list is wrapped there); ``names()`` is its lookup surface.
        registry: Any = cognition.tools
        # ``ReActCognition.tools`` is annotated ``Any`` and only list/tuple is
        # wrapped by its ``__post_init__``, so an unrecognised shape genuinely
        # reaches here. Refused rather than treated as empty: reading "no
        # ``names()``" as "no tools" makes the refusal below stop guarding the
        # moment the registry changes shape, and it silently drops every tool
        # the caller wired into a definition that still looks well-formed.
        names_of = getattr(registry, "names", None)
        if not callable(names_of):
            raise SkillNotProjectable(
                f"skill {skill.name!r} has a ReActCognition whose tools are a "
                f"{type(registry).__name__}, not a ToolRegistry, so this projection cannot "
                "enumerate them. It will not assume that means 'no tools': that would project an "
                "empty allow-list over a skill that may carry several, and the definition would "
                "look correct. Pass a ToolRegistry (or a plain list, which ReActCognition wraps)."
            )
        names: list[str] = sorted(names_of())
        if not names:
            return []
        # ── F1 seam ──────────────────────────────────────────────────────────
        # An agentkit ``Tool`` is a Python callable in THIS process. The
        # ``claude`` subprocess can only invoke its own built-ins and tools
        # served over MCP, so there is no projection of a FunctionTool that is
        # not a lie. When the MCP SERVER side lands (``agentkit.integrations.mcp``
        # exposing agentkit Tools to an external client), this refusal becomes a
        # projection: stand the server up, add its config to the cognition's
        # ``mcp_config``, and emit the tool names here as
        # ``mcp__<server>__<tool>`` — the CLI's naming convention for MCP tools —
        # alongside any built-ins. Until then, refusing is the only honest move.
        raise SkillNotProjectable(
            f"skill {skill.name!r} carries agentkit tools {names} on its ReActCognition. Those "
            "are Python callables in this process; a CLI sub-agent runs inside the `claude` "
            "subprocess and can only call the CLI's own built-in tools or tools served over "
            "MCP. Expose them from an MCP server and name them in a "
            "ClaudeCliCognition(tools=...) instead."
        )

    if isinstance(cognition, CoordinatorCognition):
        raise SkillNotProjectable(
            f"skill {skill.name!r} uses CoordinatorCognition, which drives a roster of child "
            "agents through a Policy that owns who-speaks-next, termination and aggregation. A "
            "CLI sub-agent definition has no roster field and no policy field, so projecting it "
            "would collapse a multi-agent skill into one flat agent with the same name and the "
            "same prompt — and a different behaviour. Project the children individually, and "
            "let the CLI's own delegation stand in for the policy."
        )

    raise SkillNotProjectable(
        f"skill {skill.name!r} uses {type(cognition).__name__}, which this projection does not "
        "know how to express as a CLI sub-agent. Projectable today: SingleCallCognition, "
        "ClaudeCliCognition, and a tool-free ReActCognition."
    )


def _unsubtractable(tools: list[str], disallowed: Iterable[str]) -> list[str]:
    """The ``disallowed_tools`` entries that exact-name subtraction did not honour.

    ``--disallowed-tools`` is not a list of bare tool names. The CLI accepts
    wildcards over names (``mcp__*`` — the spelling ``ClaudeCliCognition``'s own
    ``__post_init__`` recommends) and specifiers that narrow WITHIN one tool
    (``Bash(rm:*)``). A sub-agent definition has exactly one tool field, an
    allow-list of whole tool names, so neither survives the crossing:

    * ``Bash(rm:*)`` cannot be expressed at all — the sub-agent either has
      ``Bash`` entirely or does not have it.
    * ``mcp__*`` removes a set the exact-match subtraction never touched.

    Both were previously dropped, which handed the sub-agent a tool the skill
    had explicitly taken away while the projected JSON still read as correct.
    Reported only when the entry actually bites one of the tools being
    projected — a pattern aimed at something this skill does not have is
    genuinely a no-op, and refusing it would be a false alarm.
    """
    lost: list[str] = []
    for entry in disallowed:
        if "(" in entry:
            # ``Bash(rm:*)`` → the whole tool is still projected, unrestricted.
            if entry.split("(", 1)[0] in tools:
                lost.append(entry)
        elif "*" in entry:
            prefix = entry.split("*", 1)[0]
            if any(t.startswith(prefix) for t in tools):
                lost.append(entry)
    return lost


def _cli_model(skill: Skill) -> str | None:
    """The sub-agent's ``model``, or ``None`` to inherit the parent's.

    A ``ClaudeCliCognition``'s own ``model`` WINS over ``Skill.model``, because
    that is what a run does: the cognition documents that "the agent's ``model``
    field is NOT consulted". Projecting ``Skill.model`` there would produce a
    sub-agent running on a different model than the same skill run directly —
    the second-description problem, reintroduced inside the fix for it.

    Omitting ``model`` is safe in a way omitting ``tools`` is not: inheriting
    the parent's model changes cost and quality, never authority.
    """
    cognition = skill.cognition
    model = cognition.model if isinstance(cognition, ClaudeCliCognition) else None
    model = model or skill.model
    if model is None:
        return None
    if model.lower() not in _CLI_MODEL_ALIASES and "claude" not in model.lower():
        raise SkillNotProjectable(
            f"skill {skill.name!r} pins model {model!r}, which the `claude` CLI cannot resolve. "
            "A Skill wired for agentkit's own provider chain carries a provider model id; "
            "projected verbatim it becomes a sub-agent that dies at CLI startup. Use an alias "
            f"({', '.join(sorted(_CLI_MODEL_ALIASES))}) or a Claude model id."
        )
    return model


__all__ = ["SkillNotProjectable", "as_cli_agents"]
