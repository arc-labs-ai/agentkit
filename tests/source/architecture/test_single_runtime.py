"""Guard the public architecture: one runtime, one harness answer and one accounting Usage."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import agentkit
from agentkit.agents import AgentRunner
from agentkit.core.runtime import CallRuntime
from agentkit.models.contracts.types import Answer, Usage

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "agentkit"
EXAMPLES = ROOT / "tests" / "examples"
RETIRED_MODULES = (
    "agentkit.agents.cognition",
    "agentkit.agents.control",
    "agentkit.agents.evaluation",
    "agentkit.agents.policies",
    "agentkit.agents.requests",
    "agentkit.agents.workflows",
    "agentkit.core.policies",
    "agentkit.core.runtime.context",
    "agentkit.core.runtime.invoker",
    "agentkit.core.runtime.meter",
    "agentkit.core.runtime.null_ctx",
    "agentkit.core.contracts.middleware",
    "agentkit.core.contracts.replay",
    "agentkit.core.contracts.leases",
    "agentkit.core.telemetry.observation",
    "agentkit.core.events.bus",
    "agentkit.integrations.observers",
    "agentkit.models.client",
    "agentkit.state.checkpoints",
    "agentkit.state.compaction",
    "agentkit.state.leases",
    "agentkit.state.memory",
    "agentkit.state.replay",
    "agentkit.state.stores",
    "agentkit.state.vectors",
    "agentkit.tools.adapters.agent",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_public_surface_has_one_runtime_and_agent_harness() -> None:
    assert CallRuntime.__module__ == "agentkit.core.runtime.calls"
    assert AgentRunner.__module__ == "agentkit.agents.execution.runner"
    assert "AgentRunner" in agentkit.agents.__all__
    assert "CallRuntime" in importlib.import_module("agentkit.core.runtime").__all__
    for retired in ("Agent", "AgentResult", "Workflow", "WorkflowResult", "Invoker", "RunContext", "Services"):
        assert not hasattr(agentkit.agents, retired)
        assert not hasattr(importlib.import_module("agentkit.core.runtime"), retired)


def test_runtime_and_harness_share_one_public_answer_and_accounting_usage() -> None:
    assert Answer.__module__ == "agentkit.models.contracts.types"
    assert Usage.__module__ == "agentkit.models.contracts.types"
    assert "Answer" not in importlib.import_module("agentkit.core.contracts.types").__dict__
    assert "Usage" not in importlib.import_module("agentkit.core.contracts.types").__dict__
    assert CallRuntime.run.__annotations__["return"].endswith("Answer[Any]")
    assert AgentRunner.run_agent.__annotations__["return"].endswith("Answer[Any]")


def test_retired_modules_are_absent_and_nobody_imports_them() -> None:
    all_imports: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        all_imports.update(_imported_modules(path))
    for retired in RETIRED_MODULES:
        assert not any(path == retired or path.startswith(retired + ".") for path in all_imports), retired
        try:
            importlib.import_module(retired)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"retired module still imports: {retired}")


def test_runnable_examples_use_the_current_runtime_contracts() -> None:
    """Source examples must not teach the removed Agent/Invoker runtime path."""
    retired_names = {"Agent", "AgentResult", "Workflow", "WorkflowResult", "Invoker", "RunContext", "Services"}
    for path in EXAMPLES.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(
                    node.module == module or node.module.startswith(module + ".") for module in RETIRED_MODULES
                ), f"{path.relative_to(ROOT)} imports retired module {node.module}"
                if node.module in {"agentkit.agents", "agentkit.core.runtime"}:
                    imported = {alias.name for alias in node.names}
                    assert not imported.intersection(retired_names), (
                        f"{path.relative_to(ROOT)} imports retired runtime names "
                        f"{sorted(imported.intersection(retired_names))}"
                    )


def test_provider_usage_is_converted_at_the_pricing_boundary() -> None:
    from agentkit.core.contracts.types import Mode, ProviderUsage
    from agentkit.models.catalog.pricing import normalize_usage

    provider_usage = ProviderUsage(input_tokens=3, output_tokens=2)
    accounting_usage = normalize_usage(Mode.OPENAI_API, {
        "input_tokens": provider_usage.input_tokens,
        "output_tokens": provider_usage.output_tokens,
    })
    assert isinstance(accounting_usage, Usage)
    assert (accounting_usage.input_uncached, accounting_usage.output) == (3, 2)
