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
    # the headline primitives a user reaches for — guards against an accidental drop from __all__.
    # Tracks the post-refactor surface: Agent/Workflow runtimes + their result types, the
    # capability seams (RequestBuilder/Tool/ToolRegistry), the in-flight context shapes, the
    # coordination contracts (Handoff/RunPolicy/Autonomy), the kernel/runtime plumbing every
    # host needs, and the batteries-included Chat client.
    for name in (
        # high-level runtimes + result types
        "Agent",
        "Workflow",
        "AgentResult",
        "WorkflowResult",
        "Suspended",
        # prompt + request-shaping capabilities
        "Prompt",
        "RequestBuilder",
        "tool",
        "ToolRegistry",
        # in-flight context shapes
        "WorkingContext",
        "PrefixContext",
        "ContextScope",
        # capability seams
        "VectorMemory",
        "Guardrail",
        "Checkpointer",
        # coordination contracts
        "Handoff",
        "RunPolicy",
        "Autonomy",
        # kernel/runtime plumbing every host touches
        "RunContext",
        "Budget",
        "collect",
        # batteries-included client
        "Chat",
        "claude",
    ):
        assert name in agentkit.__all__ and hasattr(agentkit, name)


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
    import agentkit.kernel as k

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
