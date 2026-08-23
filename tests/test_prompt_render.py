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


# ── 5. bind(): the missing half of the contract ─────────────────────────────
#
# ``render(**values)`` alone made ``inputs=`` unusable end to end: every
# consumer in the framework (RequestBuilder ×2, ClaudeCLICognition ×1) calls
# ``render()`` with ZERO arguments, so declaring a single input turned an
# ``Agent(prompt=...)`` run into a ValueError. ``bind()`` puts the values on the
# value — where ``docs/concepts/prompts.md`` always said they lived — so the
# no-argument ``render()`` at those call sites keeps working.


def test_bind_then_render_takes_no_arguments() -> None:
    """THE point of bind(): the framework's call sites pass nothing."""
    assert _p("Hello {name}", "name").bind(name="Ada").render() == "Hello Ada"


def test_bind_returns_a_new_prompt_and_leaves_the_original_unbound() -> None:
    """A Prompt is a value; binding is not a mutation. The original must still
    refuse to render, or a shared module-level Prompt would silently pick up
    another call site's tenant."""
    base = _p("Hello {name}", "name")
    bound = base.bind(name="Ada")
    assert bound is not base
    assert bound.render() == "Hello Ada"
    assert base.bound == {}
    with pytest.raises(ValueError, match=r"missing \['name'\]"):
        base.render()


def test_bind_preserves_identity_and_version() -> None:
    """A bound prompt is still the SAME prompt for attribution — traces stamp
    id+version, and a bound copy that renamed itself would break that."""
    bound = _p("{a}", "a").bind(a="1")
    assert (bound.id, bound.version, bound.inputs) == ("p", "1", ("a",))


def test_binding_is_partial_and_accumulates() -> None:
    """Edge case: bind what you know at construction, the rest at the call
    site. A partially bound prompt still refuses to render."""
    half = _p("{tenant}/{tone}", "tenant", "tone").bind(tenant="acme")
    with pytest.raises(ValueError, match=r"missing \['tone'\]"):
        half.render()
    assert half.bind(tone="terse").render() == "acme/terse"


def test_binding_the_same_name_twice_is_last_write_wins() -> None:
    """Edge case: rebinding is an override, not an error — that is what makes
    a shared base prompt reusable per call site."""
    assert _p("{x}", "x").bind(x="1").bind(x="2").render() == "2"


def test_render_kwargs_override_a_bound_value() -> None:
    """A bound value is a default; a one-off call may still override it."""
    assert _p("{x}", "x").bind(x="bound").render(x="call") == "call"


def test_binding_an_undeclared_name_is_refused_at_bind_time() -> None:
    """Edge case: fail at bind(), not at render() — the earlier a renamed
    placeholder surfaces, the cheaper it is."""
    with pytest.raises(ValueError, match=r"unexpected \['toen'\]"):
        _p("{tone}", "tone").bind(toen="a")


def test_binding_onto_a_prompt_that_declares_no_inputs_is_refused() -> None:
    """Same call-site mistake as passing values to render() on such a prompt."""
    with pytest.raises(ValueError, match="declares no inputs"):
        _p("static").bind(name="Ada")


def test_a_bound_prompt_leaves_non_placeholder_braces_alone() -> None:
    """Edge case: the JSON-schema template must survive the bound path too, not
    just the render(**values) path."""
    p = _p('Return {"type": "object"} for {kind}.', "kind").bind(kind="invoices")
    assert p.render() == 'Return {"type": "object"} for invoices.'


def test_the_bound_mapping_is_read_only() -> None:
    """``bound`` is part of a frozen value; handing out a mutable dict would let
    a caller edit a prompt in place from the outside."""
    p = _p("{x}", "x").bind(x="1")
    with pytest.raises(TypeError):
        p.bound["x"] = "2"  # type: ignore[index]


def test_bound_does_not_alias_the_callers_dict() -> None:
    values = {"x": "1"}
    p = Prompt(id="p", version="1", template="{x}", inputs=("x",), bound=values)
    values["x"] = "2"
    assert p.render() == "1"


def test_a_prompt_with_no_inputs_still_renders_with_no_bind() -> None:
    """The zero-input path — every current caller — is untouched by bind()."""
    p = _p("  hello  ")
    assert p.render() == "hello"
    assert p.bound == {}
    assert p.bind().render() == "hello"


# ── 6. end to end: the substituted text must actually reach the model ───────


def test_a_bound_prompt_reaches_the_model_substituted() -> None:
    """The regression this whole file exists for, at the only altitude that
    proves it: Agent -> cognition -> RequestBuilder -> render() -> LLM. The
    RequestBuilder call sites pass no arguments, so an unbound-but-declaring
    prompt used to blow up here and a pre-``render(**values)`` one used to ship
    ``{tenant}`` verbatim."""
    import asyncio

    from agentkit.agents import Agent
    from agentkit.testing import FakeLLM, make_test_ctx

    seen: list[str] = []

    def record(*, system: str, user: str, model: str) -> str:
        seen.append(system)
        return "ok"

    prompt = Prompt(
        id="briefer",
        version="1.0.0",
        template='Brief tenant {tenant}. Reply as {"type": "object"}.',
        inputs=("tenant",),
    ).bind(tenant="acme")

    ctx = make_test_ctx(llm=FakeLLM(record))
    result = asyncio.run(Agent(name="briefer", prompt=prompt).run("summarise Q3", ctx))

    assert result.output == "ok"
    assert seen, "the LLM was never called"
    assert "Brief tenant acme." in seen[0]
    assert "{tenant}" not in seen[0], "an unsubstituted placeholder reached the model"
    assert '{"type": "object"}' in seen[0], "non-placeholder braces were mangled"


def test_an_unbound_prompt_that_declares_inputs_never_reaches_a_run() -> None:
    """POSITIVE CONTROL. If someone 'fixes' render() by making it permissive
    again, an unbound prompt ships ``{tenant}`` to the model — so this test, not
    a production prompt, is where that regression stops.

    The refusal now lands at CONSTRUCTION rather than on the drive: nothing in
    the framework passes prompt values at call time, so an unbound prompt is a
    guaranteed failure and the only question was whether the caller learns
    before or after the run starts. `Agent.check_prompt` is covered directly in
    ``tests/agents/test_agent_prompt_binding.py``; what this asserts is the
    end-to-end property — no run can ever begin with one.
    """
    import asyncio

    from agentkit.agents import Agent
    from agentkit.testing import FakeLLM, make_test_ctx

    prompt = Prompt(id="briefer", version="1.0.0", template="Brief {tenant}.", inputs=("tenant",))
    ctx = make_test_ctx(llm=FakeLLM("ok"))

    with pytest.raises(ValueError, match=r"'briefer' v1.0.0"):
        asyncio.run(Agent(name="briefer", prompt=prompt).run("summarise Q3", ctx))

    # And the underlying render() refusal is intact independent of the Agent —
    # the construction gate is a second line of defence, not a replacement.
    with pytest.raises(ValueError, match=r"declares inputs \['tenant'\]"):
        prompt.render()


def test_the_zero_input_agent_path_is_unchanged() -> None:
    """The form every current caller uses — a plain string prompt, no inputs —
    must be byte-identical to today."""
    import asyncio

    from agentkit.agents import Agent
    from agentkit.testing import FakeLLM, make_test_ctx

    seen: list[str] = []

    def record(*, system: str, user: str, model: str) -> str:
        seen.append(system)
        return "ok"

    ctx = make_test_ctx(llm=FakeLLM(record))
    asyncio.run(Agent(name="a", prompt="  You are terse.  ").run("hi", ctx))
    assert seen[0] == "You are terse."


# ── a value type must stay a value type ────────────────────────────────────
#
# Adding `bound` as a Mapping on a frozen dataclass broke three things that had
# always worked, silently and only at the call site that happened to use them:
#
#   hash(prompt)        TypeError: unhashable type: 'dict'
#   deepcopy(prompt)    TypeError: cannot pickle 'mappingproxy' object
#   pickle(prompt)      same
#
# The deepcopy one is not hypothetical here: `Checkpointer.snapshot` deep-copies
# state at the durable seam and the ReAct cognition deep-copies per drive, so a
# Prompt reachable from either would have taken the whole run down.


def test_a_prompt_is_still_hashable() -> None:
    """It was, before `bound` existed. Anyone keeping prompts in a dict, a set,
    or behind an `lru_cache` depends on this."""
    p = _p("hi {a}", "a")
    assert hash(p) is not None
    assert len({p, _p("hi {a}", "a")}) == 1  # value semantics preserved


def test_a_bound_prompt_hashes_into_the_same_bucket_as_its_unbound_original() -> None:
    """`__hash__` covers IDENTITY (id/version/template/inputs), not the binding.

    Sound rather than a shortcut: `__eq__` still compares `bound`, and the hash
    invariant only requires that EQUAL objects hash equally. Two prompts
    differing only in bindings collide, which is what a bucket is for — and it
    is the useful behaviour for a cache keyed on prompt identity.
    """
    p = _p("hi {a}", "a")
    b = p.bind(a="x")
    assert hash(p) == hash(b)
    assert p != b, "identical hashes must not mean identical values"


def test_a_bound_prompt_survives_deepcopy() -> None:
    """`Checkpointer.snapshot` deep-copies state; the ReAct cognition
    deep-copies per drive. A prompt that cannot be deep-copied breaks the run,
    not just the copy."""
    import copy

    b = _p("hi {a}", "a").bind(a="x")
    clone = copy.deepcopy(b)
    assert clone.render() == "hi x"
    assert clone == b


def test_deepcopy_copies_the_bound_values_too() -> None:
    """A caller binding a mutable value expects the copy to be independent —
    otherwise the 'copy' aliases state the original still owns."""
    import copy

    original = _p("{v}", "v").bind(v=[1, 2])
    clone = copy.deepcopy(original)
    assert clone.bound["v"] == [1, 2]
    assert clone.bound["v"] is not original.bound["v"]


def test_a_bound_prompt_survives_pickle_and_arrives_still_frozen() -> None:
    """A durable store or a process boundary must not choke on the proxy —
    and the prompt it hands back must still be a VALUE.

    The frozen-ness half is the part that needs saying. `Prompt` used to carry
    a `__reduce__` naming a module-level `_rebuild_prompt` factory, so unpickle
    came back through the constructor and `__post_init__` re-froze `bound` on
    the way in. Both are deleted: pickle now takes the default protocol, which
    restores `__dict__` directly and runs `__post_init__` ZERO times (measured
    by counting calls — one on construction, one on `dataclasses.replace`, none
    on unpickle or deepcopy). So the freeze rides entirely on the payload
    container being a `FrozenDict` that pickles itself.

    Equality cannot see that slipping, which is why the render/`==` assertions
    alone were not enough: a `FrozenDict` compares equal to the plain `dict` it
    would decay into, so a revived prompt could be silently mutable and every
    other assertion here would still pass."""
    import pickle

    b = _p("hi {a}", "a").bind(a="x")
    assert pickle.loads(pickle.dumps(b)).render() == "hi x"

    deep = _p("{v}", "v").bind(v={"tags": ["a"]})
    clone = pickle.loads(pickle.dumps(deep))
    assert clone == deep
    with pytest.raises(TypeError):
        clone.bound["v"] = "evil"
    with pytest.raises(TypeError):
        clone.bound["v"]["tags"].append("evil")  # deep, not just at the top


def test_a_copied_prompt_is_still_immutable() -> None:
    """POSITIVE CONTROL. The obvious way to fix deepcopy is to store a plain
    dict, which would make `prompt.bound["a"] = ...` succeed and quietly
    un-freeze a frozen value. The copy must still refuse."""
    import copy

    clone = copy.deepcopy(_p("hi {a}", "a").bind(a="x"))
    with pytest.raises(TypeError):
        clone.bound["a"] = "evil"  # type: ignore[index]
