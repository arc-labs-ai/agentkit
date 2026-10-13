"""The top-level ``agentkit`` package does not re-export test doubles.

If ``from agentkit import FakeLLM`` worked and ``__all__`` advertised
Fakes alongside production exports, a typo or stale import could let a
test double slip into a real run. Test doubles live behind
``from agentkit.core.testing import …`` and are NOT reachable from the
top-level production surface.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

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
    ``agentkit.core.testing`` boundary; the top-level package refuses on
    attribute access AND on ``from agentkit import``."""
    import agentkit

    importlib.reload(agentkit)  # defensive — drop any cached attribute
    assert not hasattr(agentkit, name), (
        f"agentkit re-exports test double {name!r} into the production "
        "surface. Move it back behind agentkit.core.testing."
    )
    assert name not in (agentkit.__all__ or ()), (
        f"agentkit.__all__ advertises test double {name!r}. Production imports should not see it."
    )


@pytest.mark.parametrize("name", _TEST_DOUBLES)
def test_test_doubles_still_reachable_via_agentkit_testing(name: str) -> None:
    """Test doubles remain reachable through the proper
    ``agentkit.core.testing`` path — legitimate test code must not be
    broken by hiding them from the top-level surface."""
    testing = importlib.import_module("agentkit.core.testing")
    assert hasattr(testing, name), (
        f"agentkit.core.testing lost {name!r}. Restore it — only the production re-export was meant to go."
    )


def test_production_classes_are_exported_by_their_owners() -> None:
    for module_name, names in {
        "agentkit.agents": ("Agent", "Workflow"),
        "agentkit.agents.skills": ("Skill",),
        "agentkit.tools": ("ToolRegistry", "FunctionTool", "FileTool"),
    }.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert name in module.__all__ and hasattr(module, name)


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
    """The names must not be reachable through ``agentkit.agents.prompts``.
    Apps that want validation contracts ship their own — keeping
    dead types in the framework misled callers into thinking the
    framework would invoke them."""
    import agentkit.agents.prompts as prompts

    assert not hasattr(prompts, name), (
        f"agentkit.agents.prompts exposes {name!r}. It should not be part of the framework surface."
    )


def test_prompts_validator_and_evaluator_modules_were_removed() -> None:
    """The underlying modules are absent — not just hidden from
    ``__all__``. Otherwise ``from agentkit.agents.prompts.validator import
    Validator`` would still work and the dead surface would survive."""
    with pytest.raises(ImportError):
        importlib.import_module("agentkit.agents.prompts.validator")
    with pytest.raises(ImportError):
        importlib.import_module("agentkit.agents.prompts.evaluator")


def test_prompts_prompt_still_works() -> None:
    """``Prompt`` itself remains — the load-bearing surface in
    ``agentkit.agents.prompts``."""
    from agentkit.agents.prompts import Prompt

    p = Prompt(id="t", version="1", template="hello {name}")
    assert p.id == "t" and p.version == "1"


_REMOVED_NAMESPACES = (
    "kernel",
    "runtime",
    "events",
    "middlewares",
    "adapters",
    "capabilities",
    "context",
    "memory",
    "calls",
    "minds",
    "prompts",
    "skills",
    "testing",
)


@pytest.mark.parametrize("name", _REMOVED_NAMESPACES)
def test_removed_aggregate_namespaces_have_no_import_aliases(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"agentkit.{name}")


def test_only_six_domain_packages_live_at_the_root() -> None:
    import pkgutil

    import agentkit

    packages = {item.name for item in pkgutil.iter_modules(agentkit.__path__) if item.ispkg}
    assert packages == {"core", "models", "agents", "state", "tools", "integrations"}


def test_new_domain_owners_provide_harness_contracts() -> None:
    from agentkit.agents.execution.runner import AgentRunner
    from agentkit.agents.minds import Mind, ReActMind
    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.integrations.mcp.gateway import McpGateway
    from agentkit.models.contracts.types import Answer, Binding, SessionHandle
    from agentkit.state.context import WorkingContext
    from agentkit.tools.execution.policy import ToolSet

    assert CallRuntime.__module__ == "agentkit.core.runtime.calls"
    assert Binding.__module__ == Answer.__module__ == SessionHandle.__module__ == "agentkit.models.contracts.types"
    assert AgentRunner.__module__ == "agentkit.agents.execution.runner"
    assert isinstance(ReActMind(), Mind)
    assert WorkingContext.__module__.startswith("agentkit.state.")
    assert ToolSet.__module__ == "agentkit.tools.execution.policy"
    assert McpGateway.__module__ == "agentkit.integrations.mcp.gateway"


def test_package_initializers_only_assemble_exports() -> None:
    """Package assembly cannot hide business logic or eager client creation."""
    package_root = Path(__file__).resolve().parents[3] / "agentkit"
    forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Call, ast.For, ast.While)
    violations = [
        f"{path.relative_to(package_root)}:{node.lineno}:{type(node).__name__}"
        for path in sorted(package_root.rglob("__init__.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, forbidden)
    ]
    assert not violations, "Move implementation to its owner module: " + ", ".join(violations)


def test_provider_families_are_explicit_physical_packages() -> None:
    import pkgutil

    import agentkit.models.providers as providers

    assert {item.name for item in pkgutil.iter_modules(providers.__path__) if item.ispkg} == {
        "frontiers", "cli"
    }
    for removed in ("agentkit.models.cli", "agentkit.models.transports"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(removed)


@pytest.mark.parametrize("owner", ("core", "models", "agents", "state", "tools", "integrations"))
def test_domain_roots_only_contain_package_assembly(owner: str) -> None:
    root = Path(__file__).resolve().parents[3] / "agentkit" / owner
    assert {path.name for path in root.iterdir() if path.is_file()} == {"__init__.py"}
    assert any(path.is_dir() and not path.name.startswith("_") for path in root.iterdir())


def test_root_api_is_exactly_namespace_assembly_and_version() -> None:
    import agentkit

    expected = {"core", "models", "agents", "state", "tools", "integrations", "__version__"}
    assert set(agentkit.__all__) == expected
    assert {name for name in vars(agentkit) if not name.startswith("_")} == expected - {"__version__"}
