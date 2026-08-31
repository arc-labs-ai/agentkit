"""What of a tool's docstring actually reaches the model.

The framework used to send the FIRST LINE and nothing else. Everything below it was
read by people editing the source and by nobody else, while still sitting in the
docstring looking as though it had been delivered. That cost a real defect: the
sentence a model most needed -- *use this only when the information genuinely cannot
be obtained any other way* -- was in the second paragraph of ``ask_human`` and never
shipped.

These tests pin the replacement rule, which is four claims at once:

  1. the WHOLE docstring is the model-facing spec;
  2. a ``---`` line ends it, so an author with a human-only tail says so out loud;
  3. a developer note (``TODO``/``FIXME``/``XXX``/``HACK`` at line start) above that
     line is a decoration-time error, because it is now an instruction to a model;
  4. the >=30-char floor and explicit ``description=`` are unchanged, and the floor
     measures whatever text was actually chosen.

All offline & deterministic; no LLM, no real I/O."""

from __future__ import annotations

import pytest

from agentkit.tools import FunctionTool, ToolDefinitionError, tool

# ---- 1. the whole docstring ships --------------------------------------------------------


def test_one_line_docstring_is_unchanged() -> None:
    # The overwhelmingly common shape (131 of the 139 tools in this repo). Whatever
    # else changes, this must not: no trailing newline, no reflow, byte-identical.
    @tool(side_effecting=False)
    def search(query: str) -> str:
        """Search the indexed corpus for `query` and return the top hit."""
        return query

    assert search.description == "Search the indexed corpus for `query` and return the top hit."
    assert search.schema.description == search.description


def test_multi_paragraph_docstring_ships_whole_including_the_second_paragraph() -> None:
    # The regression this whole change exists for. The second paragraph is the part
    # the model most needs and is exactly the part that used to be dropped.
    @tool(side_effecting=False)
    def ask_human(question: str) -> str:
        """Ask the human operator a question and wait for their typed answer.

        Use this only when the information genuinely cannot be obtained any
        other way -- a one-time code, a confirmation, a preference only they know.
        """
        return question

    assert "Use this only when the information genuinely cannot" in ask_human.description
    assert ask_human.description.startswith("Ask the human operator a question")
    # cleandoc's dedent applies, so the tail is not indented on the wire
    assert "\n        Use this" not in ask_human.description


def test_single_paragraph_wrapped_over_several_source_lines_is_not_cut_mid_sentence() -> None:
    # The case most likely to be misclassified as "multi-paragraph", and the one the
    # old rule got silently and visibly wrong: the model was shown a fragment ending
    # in "and a reason to". Two tools in this repo's own suite were in this state.
    @tool(side_effecting=False)
    def echo(text: str) -> str:
        """A tool, so the ReAct loop has something to run and a reason to
        checkpoint between iterations."""
        return text

    assert echo.description.endswith("checkpoint between iterations.")
    assert "\n" in echo.description  # the wrap is preserved, not reflowed away


def test_leading_blank_lines_do_not_become_leading_whitespace_on_the_wire() -> None:
    def spaced(x: str) -> str:
        return x

    spaced.__doc__ = "\n\n    Do the documented thing and report what happened.\n    "
    t = FunctionTool.from_callable(spaced, side_effecting=False)
    assert t.description == "Do the documented thing and report what happened."


def test_windows_line_endings_are_normalised_out_of_the_description() -> None:
    # A docstring authored on Windows carries \r\n. Under the first-line rule the \r
    # was chopped off by the split; now that the whole text ships, an un-normalised
    # docstring would put raw \r bytes in the JSON we send the provider.
    def crlf(x: str) -> str:
        return x

    # The second paragraph is INDENTED, which is the load-bearing half of the case:
    # ``inspect.getdoc`` dedents by finding common leading whitespace and a \r defeats
    # that search entirely, so a normalise-only fix would leave the indent behind and
    # send it to the provider.
    crlf.__doc__ = (
        "Do the documented thing properly.\r\n\r\n    And here is the second paragraph.\r\n    "
    )
    t = FunctionTool.from_callable(crlf, side_effecting=False)
    assert "\r" not in t.description
    assert t.description == "Do the documented thing properly.\n\nAnd here is the second paragraph."


def test_the_real_ask_human_tool_now_advertises_its_second_paragraph() -> None:
    """The defect that motivated all of the above, asserted against the shipped tool.

    ``ask_human`` is the one tool inside agentkit with a multi-paragraph docstring, and
    the sentence in its second paragraph -- do not use this when the information is
    available another way -- is the single most consequential thing it can tell a model,
    because the failure it prevents is interrupting a person who did not need to be
    interrupted. It was written, reviewed, and never sent.
    """
    from agentkit.agents.control.elicitation import ask_human_tool

    t = ask_human_tool()
    assert t.description.startswith("Ask the human operator a question")
    assert "genuinely cannot be obtained any" in t.description
    assert t.schema is not None
    assert t.schema.description == t.description


# ---- 2. the explicit cut ------------------------------------------------------------------


def test_a_bare_rule_line_ends_the_model_facing_part() -> None:
    @tool(side_effecting=False)
    def quote(ticker: str) -> str:
        """Return the last traded price for `ticker`, in USD.

        Use this rather than reasoning about prices you remember.

        ---

        Implementation note: hits the vendor's v2 endpoint, see ticket #442.
        """
        return ticker

    assert "Use this rather than reasoning" in quote.description
    assert "ticket #442" not in quote.description
    assert "---" not in quote.description
    assert quote.description.endswith("prices you remember.")


def test_a_line_that_merely_contains_a_rule_is_not_the_marker() -> None:
    # The marker is a line that is EXACTLY ``---``, not a line containing it. Prose
    # uses ``---`` as an em dash all the time, and a substring test would silently
    # truncate at the first one — the exact failure mode this change exists to end.
    @tool(side_effecting=False)
    def priced(ticker: str) -> str:
        """Return the last traded price for `ticker`.

        Quotes are delayed --- by fifteen minutes on the free tier, and by
        none at all on the paid one.
        """
        return ticker

    assert "by fifteen minutes" in priced.description
    assert "none at all on the paid one." in priced.description


def test_a_docstring_that_is_only_a_human_tail_fails_at_decoration() -> None:
    with pytest.raises(ToolDefinitionError) as exc:

        @tool(side_effecting=False)
        def hidden(x: str) -> str:
            """---

            Everything about this tool is addressed to a human being, at length.
            """
            return x

    msg = str(exc.value)
    assert "'hidden'" in msg
    assert "at least 30 characters" in msg


def test_rule_line_inside_an_explicit_description_is_left_alone() -> None:
    # ``description=`` is text the author handed us directly, not a docstring we are
    # interpreting. We do not go looking for markers in it.
    def f(x: str) -> str:
        return x

    t = FunctionTool.from_callable(
        f, description="Do the thing.\n---\nAnd this part too, at length.", side_effecting=False
    )
    assert "---" in t.description


# ---- 3. developer notes are now instructions, so they are refused -------------------------


@pytest.mark.parametrize("marker", ["TODO", "FIXME", "XXX", "HACK"])
def test_a_developer_note_above_the_rule_fails_at_decoration(marker: str) -> None:
    def noted(x: str) -> str:
        return x

    noted.__doc__ = (
        f"Look up an order by its id and return its status line.\n\n{marker}: this is O(n^2).\n"
    )
    with pytest.raises(ToolDefinitionError) as exc:
        FunctionTool.from_callable(noted, side_effecting=False)
    msg = str(exc.value)
    assert "'noted'" in msg
    assert marker in msg
    assert "---" in msg  # the message tells the author the remedy


def test_a_developer_note_below_the_rule_is_fine() -> None:
    def noted(x: str) -> str:
        return x

    noted.__doc__ = (
        "Look up an order by its id and return its status line.\n\n---\n\nTODO: this is O(n^2).\n"
    )
    t = FunctionTool.from_callable(noted, side_effecting=False)
    assert t.description == "Look up an order by its id and return its status line."


def test_the_word_todo_mid_sentence_is_not_a_developer_note() -> None:
    # A tool that manages a to-do list must still be able to say so. The marker is
    # recognised at LINE START only, which is where a developer note is written.
    @tool(side_effecting=True)
    def add_item(text: str) -> str:
        """Append a TODO: item to the shared checklist and return its new id."""
        return text

    assert add_item.description.startswith("Append a TODO: item")


# ---- 4. unchanged contracts ---------------------------------------------------------------


def test_explicit_description_still_wins_over_a_multi_paragraph_docstring() -> None:
    def documented(x: str) -> str:
        """This docstring is long enough on its own to clear the floor.

        And it has a second paragraph that must not appear either.
        """
        return x

    t = FunctionTool.from_callable(
        documented, description="The explicit spec the author chose to send.", side_effecting=False
    )
    assert t.description == "The explicit spec the author chose to send."


def test_floor_still_fires_on_a_thin_one_line_docstring() -> None:
    with pytest.raises(ToolDefinitionError) as exc:

        @tool(side_effecting=False)
        def thin(x: str) -> str:
            """too short"""
            return x

    assert "at least 30 characters" in str(exc.value) and "'thin'" in str(exc.value)


def test_the_floor_measures_the_whole_chosen_text_not_the_first_line() -> None:
    # Answers the "does the floor apply to the line or the total?" question out loud:
    # the total, because the total is what the model is shown. This tool would have
    # been REJECTED under the first-line rule (first line is 12 chars).
    @tool(side_effecting=False)
    def brief(x: str) -> str:
        """Cancel it.

        There is no undo; the order is gone from the vendor's side immediately.
        """
        return x

    assert brief.description.startswith("Cancel it.")
    assert "no undo" in brief.description


# ---- 5. gaps found in review ---------------------------------------------------------------


def test_a_developer_note_indented_under_a_heading_is_still_refused() -> None:
    """A note nested one level in is still a note, and this is the ordinary way to write one.

    ``inspect.getdoc`` dedents by the COMMON leading whitespace, so a note written under a
    heading or inside a list keeps its own extra indent all the way through the dedent. A
    column-0-only check never sees it, and the note ships to the model as an instruction --
    which is the exact failure this check exists to stop, arriving by the most ordinary
    route there is.
    """

    def lookup_order(order_id: str) -> str:
        return order_id

    lookup_order.__doc__ = (
        "Look up an order by its id and return its current status line.\n"
        "\n"
        "Notes on behaviour:\n"
        "  TODO: this is O(n^2), rewrite before the Q3 launch\n"
    )
    with pytest.raises(ToolDefinitionError) as exc:
        FunctionTool.from_callable(lookup_order, side_effecting=False)
    assert "TODO" in str(exc.value)


@pytest.mark.parametrize("bullet", ["-", "*", "+"])
def test_a_developer_note_as_a_list_item_is_still_refused(bullet: str) -> None:
    # The single most common way a note is actually written in a docstring: as one
    # bullet among real ones. Both the indent and the bullet stand between the marker
    # and column 0.
    def lookup_order(order_id: str) -> str:
        return order_id

    lookup_order.__doc__ = (
        "Look up an order by its id and return its current status line.\n"
        "\n"
        "Notes on behaviour:\n"
        "  - returns the vendor's status verbatim\n"
        f"  {bullet} FIXME: the retry loop double-counts cancellations\n"
    )
    with pytest.raises(ToolDefinitionError) as exc:
        FunctionTool.from_callable(lookup_order, side_effecting=False)
    assert "FIXME" in str(exc.value)


@pytest.mark.parametrize("word", ["XXXL", "HACKS", "TODOS", "FIXMEs"])
def test_a_word_that_merely_begins_with_a_marker_is_not_a_developer_note(word: str) -> None:
    # The other half of the marker rule, and the half no test pinned: ``\b``. Without it
    # "XXXL is a size, not a note." at the start of a line becomes a decoration-time
    # error in somebody else's application. The mid-sentence test does not cover this,
    # because there the marker is saved by not being at line start at all.
    def sized(x: str) -> str:
        return x

    sized.__doc__ = f"Look up an order by its id and return its size.\n\n{word} is a word, not a note.\n"
    t = FunctionTool.from_callable(sized, side_effecting=False)
    assert word in t.description


def test_lone_carriage_returns_are_normalised_out_too() -> None:
    # The CRLF test covers ``.replace("\r\n", "\n")``. The second replace, for a bare
    # \r, was unpinned -- and a bare \r is what a \r\r paragraph break degrades to. An
    # un-normalised \r would go into the JSON we hand the provider.
    def cr_only(x: str) -> str:
        return x

    cr_only.__doc__ = "Do the documented thing properly.\r\rAnd here is the second paragraph.\r"
    t = FunctionTool.from_callable(cr_only, side_effecting=False)
    assert "\r" not in t.description
    assert t.description == "Do the documented thing properly.\n\nAnd here is the second paragraph."


def test_the_floor_error_says_when_the_human_tail_is_what_emptied_it() -> None:
    # The whole point of the conditional clause on the floor message. Without it the
    # author is told "got 0 chars: ''" about a docstring they can plainly see is not
    # empty, and nothing points at the ``---`` line that is the actual cause.
    def hidden(x: str) -> str:
        return x

    hidden.__doc__ = "---\n\nEverything about this tool is addressed to a human being, at length.\n"
    with pytest.raises(ToolDefinitionError) as exc:
        FunctionTool.from_callable(hidden, side_effecting=False)
    msg = str(exc.value)
    assert "---" in msg
    assert "for humans and was not counted" in msg


def test_the_floor_error_does_not_blame_a_tail_that_was_never_there() -> None:
    # ... and the clause must stay off when there is no marker, or it sends the author
    # hunting for a ``---`` line their docstring does not contain.
    def thin(x: str) -> str:
        return x

    thin.__doc__ = "too short"
    with pytest.raises(ToolDefinitionError) as exc:
        FunctionTool.from_callable(thin, side_effecting=False)
    assert "was not counted" not in str(exc.value)


def test_trailing_whitespace_on_inner_lines_does_not_reach_the_provider() -> None:
    # ``inspect.cleandoc`` dedents and trims blank lines at the ends, but it does NOT
    # rstrip interior lines (verified). Now that whole docstrings ship, whatever an
    # editor left at the end of a line would ride to the provider verbatim.
    def spaced(x: str) -> str:
        return x

    spaced.__doc__ = (
        "Look up an order by its id and return its status.   \n\nUse it for order numbers.   \n"
    )
    t = FunctionTool.from_callable(spaced, side_effecting=False)
    assert " \n" not in t.description
    assert t.description == (
        "Look up an order by its id and return its status.\n\nUse it for order numbers."
    )
