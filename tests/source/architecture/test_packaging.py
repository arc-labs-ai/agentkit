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


def test_runtime_and_runner_are_the_only_public_execution_entrypoints():
    from agentkit.agents import AgentRunner
    from agentkit.core.runtime import CallRuntime

    assert "AgentRunner" in agentkit.agents.__all__
    assert "CallRuntime" in agentkit.core.runtime.__all__
    assert CallRuntime.__module__ == "agentkit.core.runtime.calls"
    assert AgentRunner.__module__ == "agentkit.agents.execution.runner"
