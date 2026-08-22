"""Who speaks next must be who the model actually named.

Both routing primitives picked the wrong agent, silently, on inputs a real
model produces constantly.

``llm_selector`` scanned the roster in ROSTER order for a substring:

    next((n for n in names if n in out), None)

So ``"Not alice — bob should go next"`` selected **alice** (first roster entry
that appears anywhere in the reply), a reply of
``"alice should proceed"`` selected **ed** off an ``["ed", "alice"]`` roster
("ed" is inside "proceed"), and a ``["bob", "bobby"]`` roster resolved a reply
of ``"bobby"`` to **bob**.

``route_by_handoff`` never checked the marker's target against the roster
despite documenting that it did, so an invented name was returned verbatim.
``SelectorPolicy`` cannot route a name it does not have, so it discarded the
choice and fell back to round-robin by turn index — meaning the operator's
pinned ``default`` was ignored at exactly the moment it mattered: the model had
just named an agent that does not exist.

Nothing here needs a model: the selectors are pure functions of (transcript,
roster), and the coordinator tests use recording stubs.
"""

from __future__ import annotations

import asyncio

import pytest

from agentkit import Agent
from agentkit.agents.cognition import CoordinatorCognition
from agentkit.agents.control.handoff import (
    Handoff,
    parse_handoff,
    route_by_handoff,
)
from agentkit.agents.control.termination import MaxMessages
from agentkit.agents.policies.selector_policy import (
    SelectorPolicy,
    _resolve_roster_name,
    llm_selector,
)
from agentkit.agents.result import AgentResult
from agentkit.kernel.types import Message, Usage
from agentkit.testing import FakeLLM, make_test_ctx


class _Rec:
    def __init__(self, name: str, out: str = "ok") -> None:
        self.name, self.output, self.calls = name, out, []

    async def run(self, task, ctx, *, context=None):  # noqa: ANN001, ANN202
        self.calls.append(task)
        return AgentResult(output=self.output, usage=Usage())


class _Chooser:
    """An ``Agent``-like whose reply is fixed — stands in for the model that
    ``llm_selector`` asks "who should speak next?"."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def run(self, prompt, ctx):  # noqa: ANN001, ANN202
        return AgentResult(output=self.reply, usage=Usage())


def _pick(reply: str, names: list[str]) -> str | None:
    sel = llm_selector(_Chooser(reply))
    agents = [_Rec(n) for n in names]
    return asyncio.run(sel([Message("user", "hi")], agents, make_test_ctx(llm=FakeLLM("x"))))


# ── 1. llm_selector: the reply decides, not the roster order ────────────────


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("bob", "bob"),  # the instructed happy path
        ("  bob  ", "bob"),  # whitespace
        ("bob.", "bob"),  # a model that punctuates
        ("Not alice — bob should go next", "bob"),  # THE regression: was "alice"
        ("definitely NOT alice, pick bob", "bob"),  # was "alice"
        ("alice", "alice"),  # the positive control
        ("alice then bob then alice", "alice"),  # last mention commits
        ("nobody is available", None),  # was "bob", inside "nobody"
        ("", None),
        ("carol", None),  # not on the roster at all
    ],
)
def test_the_reply_decides_who_speaks(reply: str, expected: str | None) -> None:
    assert _pick(reply, ["alice", "bob"]) == expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("bobby", "bobby"),  # was "bob" — a prefix roster entry stole it
        ("bob", "bob"),
        ("bobby should go next", "bobby"),
        ("planner is best", "planner"),
    ],
)
def test_a_name_that_is_a_prefix_of_another_does_not_steal_it(reply: str, expected: str) -> None:
    assert _pick(reply, ["bob", "bobby", "plan", "planner"]) == expected


@pytest.mark.parametrize(
    ("reply", "names", "expected"),
    [
        # "ed" lives inside "proceed", at a LATER offset than "alice" — so
        # last-mention-wins picks it unless matching is whole-word.
        ("alice should proceed", ["ed", "alice"], "alice"),
        # "plan" lives inside "planning".
        ("critic should review the planning notes", ["plan", "critic"], "critic"),
        # and the name really being spelled still wins.
        ("hand it to plan", ["plan", "critic"], "plan"),
    ],
)
def test_a_name_inside_an_unrelated_word_is_not_a_mention(
    reply: str, names: list[str], expected: str
) -> None:
    """Substring matching read any fragment as a choice. A roster is full of
    short names that live inside ordinary English words."""
    assert _resolve_roster_name(reply, names) == expected


def test_an_exact_reply_beats_a_later_mention() -> None:
    """Rule 1 is what handles names carrying INTERNAL punctuation. ``"v2"`` is a
    whole-word match inside ``"team.research-v2"`` — at a later offset than the
    full name — so last-mention-wins would answer ``"v2"`` for a reply that is
    exactly ``"team.research-v2"``. An exact reply is unambiguous; the prompt
    asked for only a name."""
    assert _resolve_roster_name("team.research-v2", ["team.research-v2", "v2"]) == "team.research-v2"
    # Without the internal punctuation there is nothing for rule 1 to save, and
    # rules 2-3 already agree:
    assert _resolve_roster_name("bob", ["bobby", "bob"]) == "bob"


def test_ties_at_one_offset_go_to_the_longest_name() -> None:
    """``"bobby"`` contains ``"bob"`` at the same offset. The longer name is the
    one the text actually spells."""
    assert _resolve_roster_name("please ask bobby to continue", ["bob", "bobby"]) == "bobby"


# ── 2. route_by_handoff: the roster is checked ──────────────────────────────


def test_an_invented_target_falls_back_to_the_pinned_default() -> None:
    """THE regression. The target is not on the roster, so the selector must
    return the operator's default rather than a name the coordinator cannot
    route."""
    sel = route_by_handoff(default="critic")
    roster = [_Rec("researcher"), _Rec("critic")]
    with pytest.warns(UserWarning, match="not on the coordinator's roster"):
        choice = sel([Message("assistant", "done. HANDOFF:Bobbery please continue")], roster)
    assert choice == "critic"


def test_an_invented_target_warns_once_per_name() -> None:
    """A model that loops on the same hallucination must not produce one warning
    per turn — but a NEW invented name is new information."""
    sel = route_by_handoff(default="critic")
    roster = [_Rec("researcher"), _Rec("critic")]
    msgs = [Message("assistant", "HANDOFF:Ghost go")]

    with pytest.warns(UserWarning, match="'Ghost'"):
        sel(msgs, roster)
    with pytest.warns(UserWarning, match="'Phantom'"):
        sel([Message("assistant", "HANDOFF:Phantom go")], roster)

    import warnings as _w

    with _w.catch_warnings(record=True) as again:
        _w.simplefilter("always")
        sel(msgs, roster)
    assert again == []


def test_a_real_target_still_routes() -> None:
    """The positive control — validation must not break the happy path."""
    sel = route_by_handoff(default="critic")
    roster = [_Rec("researcher"), _Rec("critic")]
    assert sel([Message("assistant", "HANDOFF:researcher go")], roster) == "researcher"


def test_a_case_differing_target_is_canonicalised_not_rejected() -> None:
    """``HANDOFF:Bob`` for a child named ``bob`` has exactly one possible
    meaning, so resolving it is canonicalisation rather than guessing."""
    sel = route_by_handoff(default="critic")
    assert sel([Message("assistant", "HANDOFF:Researcher go")], [_Rec("researcher")]) == "researcher"


def test_an_ambiguous_case_fold_is_not_guessed() -> None:
    """Two children differing only by case make the fold ambiguous. Ambiguity
    falls back to the default — the framework does not pick one."""
    sel = route_by_handoff(default="critic")
    roster = [_Rec("bob"), _Rec("BOB"), _Rec("critic")]
    with pytest.warns(UserWarning):
        assert sel([Message("assistant", "HANDOFF:Bob go")], roster) == "critic"


def test_an_empty_roster_keeps_the_pre_validation_behaviour() -> None:
    """A caller invoking the selector directly — a unit test, a custom loop —
    passes no agents. Validating against nothing would reject every target."""
    sel = route_by_handoff(default="critic")
    assert sel([Message("assistant", "HANDOFF:anyone go")], []) == "anyone"


# ── 3. end to end through the coordinator ───────────────────────────────────


def test_the_default_gets_the_turn_after_an_invented_handoff() -> None:
    """Three children so the distinction is sharp: after the invented handoff on
    turn 0, round-robin by turn index would pick ``critic`` (index 1), while the
    pinned default is ``editor``."""
    kids = {
        "researcher": _Rec("researcher", "done. HANDOFF:Bobbery take over"),
        "critic": _Rec("critic", "critique"),
        "editor": _Rec("editor", "edited"),
    }
    coord = Agent(
        name="c",
        cognition=CoordinatorCognition(
            children=kids,
            policy=SelectorPolicy(selector=route_by_handoff(default="editor")),
            termination=MaxMessages(3),
        ),
    )
    with pytest.warns(UserWarning, match="'Bobbery'"):
        res = asyncio.run(
            coord.run(
                "HANDOFF:researcher please start",
                make_test_ctx(llm=FakeLLM("x"), correlation_id="e2e1"),
            )
        )

    spoke = [m.name for m in res.evals["messages"] if m.role == "assistant"]
    assert spoke[0] == "researcher"  # routed by the marker on the task
    assert spoke[1] == "editor", "the pinned default must get the turn, not the turn index"
    assert kids["critic"].calls == []


# ── 4. parse_handoff: a model that ends a sentence ──────────────────────────


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("HANDOFF:bob.", "bob"),  # was "bob.", matching no roster entry
        ("HANDOFF:bob!", "bob"),
        ('HANDOFF:bob"', "bob"),
        ("HANDOFF:bob)", "bob"),
        ("HANDOFF:bob", "bob"),
        ("HANDOFF:team.research-v2", "team.research-v2"),  # INTERNAL dots survive
    ],
)
def test_trailing_punctuation_is_not_part_of_the_target(text: str, target: str) -> None:
    ho = parse_handoff(text)
    assert ho is not None and ho.target == target


def test_a_marker_with_only_punctuation_is_not_a_handoff() -> None:
    """Stripping must not manufacture an empty target."""
    assert parse_handoff("HANDOFF:...") is None


def test_render_round_trips_through_parse() -> None:
    """The tool path renders a marker the marker path reads back — one wire
    format, one detector, as the module docstring promises."""
    ho = Handoff(target="bob", reason="better suited", message="focus on the tables")
    back = parse_handoff(ho.render())
    assert back == ho
