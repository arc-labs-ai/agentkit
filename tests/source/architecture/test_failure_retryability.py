"""`Failure.retryable` must agree with its own classification, package-wide.

WHY THIS IS AN ARCHITECTURE TEST
--------------------------------
`retryable` is a separate constructor argument that DEFAULTS TO FALSE, while the
runtime gates every retry on exactly that flag::

    retry_allowed = failure.retryable and ...

So `Failure(FailureClass.TRANSIENT_NETWORK)` — which reads perfectly natural, and
which names the most retryable condition there is — produces a failure the
runtime will never retry. Nothing raises. The classification in the logs says
"transient" and the behaviour says "give up", and the two are written three
lines apart.

That shipped at two sites in the API stream parsers: a response body truncated
mid-stream made ONE request, while the same physical event one layer earlier (a
socket error) made three. Both sites named the right class and got the wrong
behaviour.

A per-site test cannot prevent the next one, because the next one is a site
nobody has written yet. This walks the package instead, so a new construction
with the default flag fails here rather than in production. The scan is
deliberately syntactic — it reads the AST, not the runtime — because the point
is to catch the SPELLING that is easy to get wrong.
"""

from __future__ import annotations

import ast
import pathlib

from agentkit.models.contracts.constants import FailureClass

# Exactly the set `contracts.failures.classify` derives `retryable` from, and the
# set the runtime's retry gate honours. Kept as literals rather than imported so
# that widening the production set cannot silently widen this test.
RETRYABLE_CLASSES = {"RATE_LIMITED", "OVERLOADED", "TIMEOUT", "TRANSIENT_NETWORK"}

PACKAGE = pathlib.Path(__file__).resolve().parents[3] / "agentkit"


def _failure_constructions() -> list[tuple[str, int, str, bool]]:
    """Every `Failure(FailureClass.X, ...)` in the package, and whether it says retryable."""
    found: list[tuple[str, int, str, bool]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Failure"):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if not (
                isinstance(first, ast.Attribute)
                and isinstance(first.value, ast.Name)
                and first.value.id == "FailureClass"
            ):
                continue
            states_retryable = len(node.args) > 1 or any(kw.arg == "retryable" for kw in node.keywords)
            found.append((str(path.relative_to(PACKAGE.parent)), node.lineno, first.attr, states_retryable))
    return found


def test_the_scan_finds_the_constructions_it_is_meant_to_police() -> None:
    """A guard that silently matches nothing passes forever. If the construction
    spelling changes, this fails and tells the next person to re-aim the scan
    rather than letting it rot into a no-op."""
    found = _failure_constructions()
    assert len(found) > 20, f"the scan found only {len(found)} Failure constructions; has the spelling changed?"
    assert {name for _, _, name, _ in found} <= {member.name for member in FailureClass}, (
        "the scan read something that is not a FailureClass member"
    )


def test_no_retryable_classification_is_constructed_as_non_retryable() -> None:
    """The invariant. A failure whose class is one the runtime retries must say so,
    because `retryable` defaults to False and the gate reads the flag, not the class.

    `retryable` is not derived automatically on purpose — the dataclass is a plain
    value and several non-transient classes legitimately want the default — so the
    rule is enforced here instead of being enforced by construction."""
    offenders = [
        f"{path}:{line} builds Failure(FailureClass.{name}) without retryable="
        for path, line, name, states in _failure_constructions()
        if name in RETRYABLE_CLASSES and not states
    ]
    assert not offenders, (
        "a retryable classification constructed with the default retryable=False will never be "
        "retried, while its class says it should be:\n  " + "\n  ".join(offenders)
    )
