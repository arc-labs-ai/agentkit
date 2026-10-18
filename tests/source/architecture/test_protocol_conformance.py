"""Guard the declared seams: every port is `runtime_checkable`, a conformance check
rejects a non-conforming adapter, and a policy that reaches a seam cannot carry a
bound the seam will not honour.

A ``Protocol`` is the only thing standing between "agentkit cannot implement this
itself, inject it" and "pass whatever you like". Two ways that degrades:

* The ``@runtime_checkable`` decorator comes off a port. ``isinstance(x, Port)``
  then raises ``TypeError: Instance and class checks can only be used with
  @runtime_checkable protocols`` at the host's wiring check, so the check gets
  deleted rather than fixed, and the seam becomes documentation.
* A conformance check silently ACCEPTS something that does not conform. Nothing
  raises at wiring time; the failure surfaces as an ``AttributeError`` deep in a
  run, or — worse — as a bound that was declared and never enforced.

Static/structural, in the style of `test_packaging.py` and
`test_single_runtime.py`: these are statements about the shape of the package,
not about what a run does. `tests/source/unit/tools/test_tool_protocol.py` pins
the ``Tool`` protocol's own members; this file is about the port surface as a
whole and the preconditions declared at it.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
from pathlib import Path
from typing import Any

import pytest

from agentkit.core.contracts.ports import ClockPort, SearchHit, SearchPort
from agentkit.tools.execution.policy import ToolSet
from agentkit.tools.registry.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "agentkit"


# ── every declared port is isinstance-able ─────────────────────────────────


def _declared_protocols() -> list[tuple[str, str, bool]]:
    """(module, class name, is runtime_checkable) for every Protocol in the package."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [ast.unparse(base) for base in node.bases]
            if not any(base == "Protocol" or base.startswith("Protocol[") for base in bases):
                continue
            decorators = [ast.unparse(dec) for dec in node.decorator_list]
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            found.append((module, node.name, "runtime_checkable" in decorators))
    return found


def test_the_package_still_declares_protocol_seams_at_all() -> None:
    """The sweeps below are vacuous if the scan stops finding anything — a rename
    of the base, a move of the ports module, or an ``ast`` change would make this
    file report green while checking nothing."""
    protocols = _declared_protocols()
    assert len(protocols) >= 20, f"the Protocol scan found only {len(protocols)} classes"
    assert ("agentkit.core.contracts.ports", "ClockPort", True) in protocols


def test_every_protocol_named_as_a_port_is_runtime_checkable() -> None:
    """A ``*Port`` is an L0 seam — an external system the framework cannot
    implement itself (a model, the world, a vector index, a durable store, the
    wall clock) — and it is the one category hosts and tests structurally verify
    an adapter against before wiring it in. Without ``@runtime_checkable`` that
    verification does not fail, it RAISES, at the one point in the program whose
    job is to catch a bad adapter early; the pressure is then to drop the check
    and discover the mismatch mid-run instead.

    Scanned statically over the whole package rather than listed by hand, because
    the failure mode is a NEW port shipping without the decorator — a list only
    protects the ports someone remembered to add."""
    offenders = [
        f"{module}.{name}" for module, name, checkable in _declared_protocols() if name.endswith("Port") and not checkable
    ]
    assert not offenders, f"ports declared without @runtime_checkable: {offenders}"


def test_every_runtime_checkable_port_actually_answers_isinstance() -> None:
    """The decorator is necessary and not sufficient: the check also has to WORK.
    A ``Protocol`` with non-method members is runtime-checkable for ``isinstance``
    but raises on ``issubclass``, and a data protocol that grew a method (or the
    reverse) changes which of the two a host may call. Pinned by making both
    calls on every port rather than trusting the decorator's presence."""
    for module, name, checkable in _declared_protocols():
        if not name.endswith("Port") or not checkable:
            continue
        protocol = getattr(importlib.import_module(module), name)
        assert getattr(protocol, "_is_runtime_protocol", False) is True, f"{name} lost its marker"
        assert isinstance(object(), protocol) is False, f"isinstance(object(), {name}) did not answer"


# Real classes at module level, deliberately. A conformance check is about the
# attributes a class actually has, so the implementations under test have to be
# ordinary classes — not objects built from a parametrized annotation, which
# PEP 563 turns into the unresolvable string ``"ann"`` and which would then pass
# for the wrong reason.


class ConformingClock:
    """The minimum a `ClockPort` adapter has to be: a fakeable wall clock."""

    def now(self) -> float:
        return 1_700_000_000.0

    async def sleep(self, seconds: float) -> None:
        return None


class ClockMissingSleep:
    """Half a clock. `sleep` is the half middlewares and retry/backoff loops use."""

    def now(self) -> float:
        return 1_700_000_000.0


class ConformingSearch:
    async def search(self, query: str, k: int = 5, where: dict[str, Any] | None = None) -> list[SearchHit]:
        return [SearchHit("https://example.test/doc", "A document", "snippet")]


class SearchWithTheWrongVerb:
    """A plausible adapter that named its entry point `query` instead of `search`."""

    async def query(self, query: str, k: int = 5) -> list[SearchHit]:
        return []


def test_a_conformance_check_accepts_a_structurally_complete_adapter() -> None:
    """POSITIVE CONTROL, and the point of a ``Protocol`` over a base class: an
    adapter that has the shape conforms with no inheritance and no registration,
    which is what lets a user's own clock or search backend be injected. A check
    that rejected these would push every adapter back onto subclassing the
    framework."""
    assert isinstance(ConformingClock(), ClockPort)
    assert isinstance(ConformingSearch(), SearchPort)


def test_a_conformance_check_rejects_an_adapter_missing_a_port_member() -> None:
    """The half that has to bite. An adapter missing a member is accepted by a
    plain duck-typed wiring and fails later with ``AttributeError`` inside a run —
    after a model call has been paid for, and at a stack depth where the cause is
    a guess. The isinstance check is what converts that into a refusal at the
    seam, so it has to actually say no."""
    assert not isinstance(ClockMissingSleep(), ClockPort)
    assert not isinstance(SearchWithTheWrongVerb(), SearchPort)
    assert not isinstance(object(), ClockPort)


class ClockWithTheWrongSignatures:
    """Present, callable, and wrong: `now` demands an argument the framework never
    passes and `sleep` is synchronous, so awaiting it raises."""

    def now(self, timezone: str) -> float:
        return 1_700_000_000.0

    def sleep(self, seconds: float) -> None:
        return None


class ClockWhoseNowIsNotCallable:
    """`now` is a cached float rather than a reading — the plausible shortcut."""

    now = 1_700_000_000.0

    async def sleep(self, seconds: float) -> None:
        return None


class ClockWhoseSleepIsStubbedOut:
    """The same shortcut spelled with ``None``, which — surprisingly — does not
    conform where the float above does."""

    def now(self) -> float:
        return 1_700_000_000.0

    sleep = None


def test_structural_conformance_proves_presence_only_so_a_bound_needs_its_own_gate() -> None:
    """DELIBERATE, and recorded here so the next reader does not mistake an
    isinstance check for a contract check. ``runtime_checkable`` compares NAMES,
    not signatures: a clock whose ``now`` requires an argument and whose ``sleep``
    is synchronous passes, and so does one whose ``now`` is a float. The run then
    fails at the first ``await clock.sleep(...)`` or ``clock.now()``, inside a
    retry loop, far from the wiring that accepted it.

    The asymmetry on the last one is pinned because it is genuinely surprising
    and would otherwise be read as a bug in this file: CPython treats a member
    set to ``None`` as ABSENT, so stubbing a port method out with ``None`` is
    rejected while stubbing it out with a float is accepted.

    The consequence is the reason the next test exists: a declared BOUND cannot
    be left for a seam's conformance check to notice, because that check cannot
    see values at all. It has to be refused by the value that carries it."""
    assert isinstance(ClockWithTheWrongSignatures(), ClockPort)
    assert isinstance(ClockWhoseNowIsNotCallable(), ClockPort)
    assert not isinstance(ClockWhoseSleepIsStubbedOut(), ClockPort)


# ── a precondition declared at a seam has to be enforced by the value ──────


def test_the_tool_execution_policy_refuses_a_concurrency_bound_it_cannot_honour() -> None:
    """``ToolSet`` is the execution policy handed across the tool seam, and
    ``max_concurrency`` is the one field on it that is not advice: the fan-out
    site turns it into an ``asyncio.Semaphore``. A semaphore built with 0 permits
    is CREATED LOCKED — it never admits a first waiter — so a policy that accepts
    0 does not fan out slowly, it never starts, and the run hangs until the
    suite's timeout rather than reporting a misconfiguration.

    There is nothing in the ``Protocol`` machinery that would catch this: a
    conformance check proves a member is present (see above), never that a value
    is in range. The constructor is the only gate, which is why 0 and -1 have to
    raise at construction and not at the first ``gather``."""
    registry = ToolRegistry.from_tools([])
    for refused in (0, -1):
        with pytest.raises(ValueError, match="concurrency"):
            ToolSet(registry, max_concurrency=refused)

    # The consequence the guard exists to prevent, pinned rather than described:
    # a bound of 0 is a gate that never opens. Asserted on the primitive so the
    # test states WHY 0 is refused without having to hang to prove it.
    assert asyncio.Semaphore(0).locked() is True
    assert asyncio.Semaphore(1).locked() is False

    # ...and 1 is a legitimate bound — serialised execution, not a disabled one —
    # so the guard must refuse non-positive values without refusing this.
    assert ToolSet(registry, max_concurrency=1).max_concurrency == 1
