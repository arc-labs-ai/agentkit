"""``Prompt.render()`` substitutes the inputs the prompt declares.

It used to be ``return self.template.strip()`` — taking no arguments at all —
while ``docs/concepts/prompts.md`` called it "the pure function from the
pre-declared inputs to a rendered string". So ``inputs=`` was decorative and a
templated prompt shipped its placeholders to the model verbatim:

    Prompt(template="Hello {name}", inputs=("name",)).render()
    -> 'Hello {name}'

That is worse than a crash. It is a plausible-looking prompt that quietly
describes the wrong task, and the model answers it.
"""

from __future__ import annotations

import pytest

from agentkit.prompts import Prompt


def _p(template: str, *inputs: str) -> Prompt:
    return Prompt(id="p", version="1", template=template, inputs=tuple(inputs))


# ── 1. it substitutes ───────────────────────────────────────────────────────


def test_declared_inputs_are_substituted() -> None:
    assert _p("Hello {name}, tone={tone}.", "name", "tone").render(
        name="Ada", tone="terse"
    ) == "Hello Ada, tone=terse."


def test_a_placeholder_used_twice_is_substituted_twice() -> None:
    assert _p("{x} and {x}", "x").render(x="a") == "a and a"


def test_non_string_values_are_coerced() -> None:
    """A caller passing an int or a list should not have to remember to str()
    it — the result is a prompt either way."""
    assert _p("n={n}", "n").render(n=42) == "n=42"


# ── 2. braces that are not placeholders survive ─────────────────────────────


def test_json_in_a_template_is_left_alone() -> None:
    """THE reason this is a literal replacement and not ``str.format``. System
    prompts are full of braces that are not placeholders — a JSON Schema, an
    example payload, a code fence. ``format`` would raise ``KeyError`` on them,
    or silently eat a user's doubled-brace escaping."""
    rendered = _p('Return {"type": "object"} for {kind}.', "kind").render(kind="invoices")
    assert rendered == 'Return {"type": "object"} for invoices.'


def test_an_undeclared_placeholder_is_not_touched() -> None:
    """Only DECLARED names are replaced. A ``{foo}`` the prompt never declared
    is literal text as far as this is concerned — silently blanking it would be
    the same class of bug in the other direction."""
    assert _p("{a} then {b}", "a").render(a="1") == "1 then {b}"


def test_format_would_have_broken_this() -> None:
    """Pinning the distinction explicitly, so a future 'simplification' to
    ``str.format`` fails here rather than in someone's production prompt."""
    template = 'Schema: {"required": ["id"]}'
    with pytest.raises((KeyError, IndexError, ValueError)):
        template.format(kind="x")
    assert _p(template + " for {kind}", "kind").render(kind="x") == (
        'Schema: {"required": ["id"]} for x'
    )


# ── 3. it refuses rather than half-rendering ────────────────────────────────


def test_a_missing_input_is_refused() -> None:
    """A half-filled prompt reaches the model looking plausible. Refusing is
    the only outcome that surfaces the mistake."""
    with pytest.raises(ValueError, match=r"missing \['tone'\]"):
        _p("{name}/{tone}", "name", "tone").render(name="Ada")


def test_an_unexpected_input_is_refused() -> None:
    """Almost always a renamed placeholder or a typo, and silently ignoring it
    renders the OLD template with none of the new values."""
    with pytest.raises(ValueError, match=r"unexpected \['toen'\]"):
        _p("{tone}", "tone").render(tone="a", toen="b")


def test_the_error_names_the_prompt_and_version() -> None:
    """A run has many prompts; "missing input" alone does not say which."""
    with pytest.raises(ValueError, match=r"'p' v1 declares inputs \['name'\]"):
        _p("{name}", "name").render()


def test_values_passed_to_a_prompt_that_declares_none_are_refused() -> None:
    """Passing values to a prompt with no ``inputs`` is a call-site mistake, not
    a no-op — most likely the ``inputs=`` was forgotten."""
    with pytest.raises(ValueError, match="declares no inputs"):
        _p("static").render(name="Ada")


# ── 4. the existing surface is unchanged ────────────────────────────────────


def test_a_prompt_with_no_inputs_renders_as_before() -> None:
    """Every caller in the framework passes no arguments and declares no
    inputs. That path must be byte-identical, including the strip."""
    assert _p("  hello  ").render() == "hello"


def test_the_builtins_still_render() -> None:
    """``COMPACTION_SUMMARY`` / ``COMPACTION_IMPORTANCE`` are called with no
    arguments from the compaction capability."""
    from agentkit.prompts.builtin import COMPACTION_IMPORTANCE, COMPACTION_SUMMARY

    for prompt in (COMPACTION_SUMMARY, COMPACTION_IMPORTANCE):
        assert prompt.render()
        assert "{" not in prompt.render() or prompt.inputs
