"""Every name a package exports must actually be the thing it claims to be.

This exists because the same bug happened three times in this codebase, each
time in a different package: a MODULE and a FUNCTION sharing one name.

    agentkit/adapters/llm/registry.py      vs  registry()
    agentkit/agents/control/elicit.py      vs  elicit()
    agentkit/middlewares/security.py       vs  security()

Importing a submodule binds its name onto the parent package. When that import
runs AFTER the ``from .other import <same_name>`` line — which import sorting
decides, not the author — the submodule wins and the exported callable is
silently replaced by a module object. The third instance shipped:

    >>> from agentkit.middlewares import security
    >>> security()
    TypeError: 'module' object is not callable

It was in ``__all__``, and in the canonical middleware chain in two docs pages,
so it broke on first contact for anyone following them. No test caught it
because every test imported the concrete class instead.

The check is deliberately blunt: walk the public packages, and for each name in
``__all__`` assert it is not a module unless the package meant to export a
module. A lowercase name bound to a module is the signature of this bug.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

# Packages whose ``__all__`` is a curated public surface. Sub-packages that
# legitimately re-export modules are listed in _MODULE_EXPORTS below.
_PACKAGES = [
    "agentkit",
    "agentkit.middlewares",
    "agentkit.agents",
    "agentkit.agents.cognition",
    "agentkit.agents.control",
    "agentkit.agents.policies",
    "agentkit.adapters.llm",
    "agentkit.adapters.store",
    "agentkit.capabilities",
    "agentkit.kernel",
    "agentkit.runtime",
    "agentkit.tools",
    "agentkit.context",
    "agentkit.prompts",
    "agentkit.memory",
    "agentkit.testing",
]

# Names a package exports ON PURPOSE as modules. An entry here is a deliberate
# decision, which is the point — the bug this file exists for is the ACCIDENTAL
# kind, where nobody chose anything.
_MODULE_EXPORTS: dict[str, set[str]] = {
    "agentkit": {"middlewares", "streams"},
    "agentkit.kernel": {"streams"},
}


@pytest.mark.parametrize("package", _PACKAGES)
def test_no_exported_name_is_secretly_a_module(package: str) -> None:
    """THE regression, generalised. A name in ``__all__`` that resolves to a
    module is almost always a submodule shadowing a same-named callable."""
    mod = importlib.import_module(package)
    allowed = _MODULE_EXPORTS.get(package, set())

    shadowed = []
    for name in getattr(mod, "__all__", ()):
        obj = getattr(mod, name, None)
        if inspect.ismodule(obj) and name not in allowed:
            shadowed.append(f"{package}.{name} -> {obj!r}")

    assert not shadowed, (
        "these exported names resolve to MODULES, not the callables/classes they "
        "advertise — a submodule of the same name has shadowed them. Rename the "
        f"submodule (see this file's docstring): {shadowed}"
    )


@pytest.mark.parametrize("package", _PACKAGES)
def test_every_exported_name_actually_resolves(package: str) -> None:
    """A name in ``__all__`` that does not exist breaks ``from pkg import *``
    and every IDE completion, and is invisible until someone tries it."""
    mod = importlib.import_module(package)
    missing = [n for n in getattr(mod, "__all__", ()) if not hasattr(mod, n)]
    assert not missing, f"{package}.__all__ names things that do not exist: {missing}"


def test_no_export_is_shadowed_by_a_submodule_from_elsewhere() -> None:
    """The structural form, catching the collision at the SOURCE.

    Writing the blunt version of this check taught me the precise rule, because
    it flagged five names that are collisions and are nonetheless safe.

    ``middlewares/tracing.py`` defines ``tracing()``, and ``__init__`` says
    ``from agentkit.middlewares.tracing import tracing``. That statement imports
    the submodule (binding ``middlewares.tracing`` to the module) and THEN binds
    the local name to the function — in that order, within the one statement. So
    the function always wins. Same for ``memoize``, ``meter``, ``compaction``,
    ``output_coerce``.

    ``security()`` lived in ``guard.py`` while ``security.py`` was a different
    module. The submodule binding therefore happened on a LATER line than the
    factory import, and overwrote it. That is the whole difference, and it is
    decided by import ORDER — which the formatter owns, not the author.

    So the dangerous shape is: an exported name that collides with a submodule
    AND is defined somewhere other than that submodule. That is what this
    checks, and it catches all three historical instances while leaving the five
    safe ones alone.
    """
    collisions = []
    for package in _PACKAGES:
        mod = importlib.import_module(package)
        path = getattr(mod, "__path__", None)
        if path is None:
            continue
        submodules = {m.name for m in pkgutil.iter_modules(path)}
        for name in getattr(mod, "__all__", ()):
            if name not in submodules or name in _MODULE_EXPORTS.get(package, set()):
                continue
            obj = getattr(mod, name, None)
            defined_in = getattr(obj, "__module__", None)
            if defined_in is not None and defined_in != f"{package}.{name}":
                collisions.append(
                    f"{package}.{name} is defined in {defined_in} but {package} also "
                    f"contains a submodule named {name!r}"
                )

    assert not collisions, (
        "an exported name collides with a submodule AND is defined elsewhere, so "
        "whichever import runs last wins — and import order is the formatter's "
        f"decision, not yours. Rename the submodule: {collisions}"
    )
