"""Packaging & public-API hygiene: the type marker ships, the curated `__all__` has no drift, and a
star-import resolves. These guard the surface a downstream user actually imports."""

import os

import agentkit


def test_py_typed_marker_ships_with_the_package():
    marker = os.path.join(os.path.dirname(agentkit.__file__), "py.typed")
    assert os.path.exists(marker)  # PEP 561 — agentkit is type-checked by consumers


def test_every_name_in_all_resolves():
    missing = [name for name in agentkit.__all__ if not hasattr(agentkit, name)]
    assert not missing, f"names in __all__ with no attribute (export drift): {missing}"


def test_all_has_no_duplicates():
    assert len(agentkit.__all__) == len(set(agentkit.__all__))


def test_star_import_exposes_all():
    ns: dict = {}
    exec("from agentkit import *", ns)
    for name in agentkit.__all__:
        assert name in ns


def test_key_surface_is_exported():
    # Retain every headline contract while checking its actual public owner.
    import importlib

    owners = {
        "agentkit.agents": ("Agent", "Workflow", "AgentResult", "WorkflowResult", "Suspended", "Handoff", "RunPolicy", "Autonomy"),
        "agentkit.agents.prompts": ("Prompt",),
        "agentkit.agents.requests": ("RequestBuilder",),
        "agentkit.tools": ("tool", "ToolRegistry"),
        "agentkit.state.context": ("WorkingContext", "PrefixContext", "ContextScope"),
        "agentkit.state.memory": ("VectorMemory",),
        "agentkit.core.guardrails": ("Guardrail",),
        "agentkit.state.checkpoints": ("Checkpointer",),
        "agentkit.core.runtime": ("RunContext", "Budget"),
        "agentkit.core": ("collect",),
        "agentkit.models.providers.frontiers": ("claude",),
    }
    for module_name, names in owners.items():
        owner = importlib.import_module(module_name)
        for name in names:
            assert name in owner.__all__ and hasattr(owner, name), f"{module_name}.{name}"
    from agentkit.models.client.chat import Chat

    assert Chat.__module__ == "agentkit.models.client.chat"


def test_legacy_pre_refactor_symbols_are_not_exported():
    # The team/orchestrator/planner vocabulary was retired in favour of
    # ``Agent(children=..., policy=...)`` + the four explicit policies in
    # ``agentkit.agents.policies``.
    # The autonomy module-level constants collapsed into the ``Autonomy`` enum. Guard against
    # accidental reintroduction so callers can't silently depend on a dead surface.
    legacy = {
        # team/orchestrator vocabulary (replaced by Agent + Policy)
        "RoundRobinTeam",
        "SelectorTeam",
        "Orchestrator",
        "LedgerOrchestrator",
        "TeamResult",
        "OrchestratorResult",
        "LedgerResult",
        # request-shaping rename
        "Composer",
        # planner/step/task vocabulary (subsumed by Workflow + PlanPolicy)
        "Step",
        "Planner",
        "StaticPlanner",
        "Task",
        "TaskLedger",
        "ProgressLedger",
        # autonomy constants (replaced by Autonomy enum)
        "AUTO",
        "GATED",
        "MANUAL",
        "LEVELS",
    }
    leaked = legacy & set(agentkit.__all__)
    assert not leaked, f"legacy pre-refactor symbols still in __all__: {sorted(leaked)}"


def test_kernel_subpackage_exports_its_own_primitives():
    # importing from the documented L0 layer must reach the kernel's own contract + types
    import agentkit.core as k

    for name in (
        "Delta",
        "assemble_deltas",
        "collect",
        "collect_one",
        "Cancelled",
        "CancellationToken",
        "Failure",
        "compose_failures",
        "Observation",
        "ObserverPort",
        "streams",
    ):
        assert name in k.__all__ and hasattr(k, name)
