"""The top-level ``agentkit`` package does not re-export test doubles.

If ``from agentkit import FakeLLM`` worked and ``__all__`` advertised
Fakes alongside production exports, a typo or stale import could let a
test double slip into a real run. Test doubles live behind
``from agentkit.testing import …`` and are NOT reachable from the
top-level production surface.
"""

from __future__ import annotations

import importlib

import pytest

_TEST_DOUBLES = [
    "FakeClock",
    "FakeCompactor",
    "FakeCtx",
    "FakeFetch",
    "FakeGrounder",
    "FakeLLM",
    "FakeMemory",
    "FakeSearch",
    "FakeTool",
    "Turn",
    "make_test_ctx",
]


@pytest.mark.parametrize("name", _TEST_DOUBLES)
def test_test_doubles_not_reachable_from_top_level_agentkit(name: str) -> None:
    """Each test-double name resolves only via the explicit
    ``agentkit.testing`` boundary; the top-level package refuses on
    attribute access AND on ``from agentkit import``."""
    import agentkit

    importlib.reload(agentkit)  # defensive — drop any cached attribute
    assert not hasattr(agentkit, name), (
        f"agentkit re-exports test double {name!r} into the production "
        "surface. Move it back behind agentkit.testing."
    )
    assert name not in (agentkit.__all__ or ()), (
        f"agentkit.__all__ advertises test double {name!r}. Production imports should not see it."
    )


@pytest.mark.parametrize("name", _TEST_DOUBLES)
def test_test_doubles_still_reachable_via_agentkit_testing(name: str) -> None:
    """Test doubles remain reachable through the proper
    ``agentkit.testing`` path — legitimate test code must not be
    broken by hiding them from the top-level surface."""
    testing = importlib.import_module("agentkit.testing")
    assert hasattr(testing, name), (
        f"agentkit.testing lost {name!r}. "
        "Restore it — only the production re-export was meant to go."
    )


def test_top_level_imports_production_classes_still_work() -> None:
    """Production exports remain reachable from the top-level package.
    A representative sample is named so any regression that nukes too
    much surfaces a clear failure."""
    import agentkit

    for name in ("Agent", "Workflow", "Skill", "ToolRegistry", "FunctionTool", "FileTool"):
        assert hasattr(agentkit, name), (
            f"Production export {name!r} disappeared from top-level agentkit."
        )


# ── dead prompts/validator + evaluator skeletons are absent ──────────────────
#
# ``Validator``, ``PromptEvaluator``, ``ValidationResult``, and
# ``EvaluationScore`` are not part of the framework surface. The Agent's
# parse-and-repair uses its own ``SchemaAdapter`` path; carrying dead
# validator types would mislead callers into thinking the framework
# would invoke them.


@pytest.mark.parametrize(
    "name",
    ["Validator", "ValidationResult", "PromptEvaluator", "EvaluationScore"],
)
def test_dead_validator_evaluator_skeletons_are_gone(name: str) -> None:
    """The names must not be reachable through ``agentkit.prompts``.
    Apps that want validation contracts ship their own — keeping
    dead types in the framework misled callers into thinking the
    framework would invoke them."""
    import agentkit.prompts as prompts

    assert not hasattr(prompts, name), (
        f"agentkit.prompts exposes {name!r}. It should not be part of the framework surface."
    )


def test_prompts_validator_and_evaluator_modules_were_removed() -> None:
    """The underlying modules are absent — not just hidden from
    ``__all__``. Otherwise ``from agentkit.prompts.validator import
    Validator`` would still work and the dead surface would survive."""
    with pytest.raises(ImportError):
        importlib.import_module("agentkit.prompts.validator")
    with pytest.raises(ImportError):
        importlib.import_module("agentkit.prompts.evaluator")


def test_prompts_prompt_still_works() -> None:
    """``Prompt`` itself remains — the load-bearing surface in
    ``agentkit.prompts``."""
    from agentkit.prompts import Prompt

    p = Prompt(id="t", version="1", template="hello {name}")
    assert p.id == "t" and p.version == "1"
