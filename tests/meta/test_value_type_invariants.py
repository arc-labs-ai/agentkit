"""Every public value type must be hashable, deep-copyable, and picklable.

WHY THIS FILE EXISTS
--------------------
One bug shape recurred five separate times in this codebase, each found by
accident rather than by a test:

    Prompt        a frozen dataclass gained a Mapping field and silently became
                  unhashable AND unpicklable (so `deepcopy` broke too)
    signals       six frozen signals carried mutable list/dict payloads: a
                  receiver could rewrite an audited `used_cost` after the fact
    ToolCall      `arguments` is a MappingProxyType — immutable, but proxies are
                  unhashable, so every tool-call Message was unhashable, and
                  `WorkingContext.merge(mode="union")` raised on the coordinator
                  fan-in path
    FrozenContext its docstring promised "safe to use as a memoization-cache
                  key" while any dict in the scratchpad made it unhashable
    LLMResult     a hand-written field-by-field rebuild dropped `parsed`

The common root: a type is declared `frozen=True`, everyone reasons about it as
a value, and nothing checks that it actually behaves like one. `deepcopy` and
`pickle` are not academic here — `Checkpointer.snapshot` deep-copies state at
the durable seam, and these types cross process and storage boundaries.

THE RATCHET
-----------
The sweep is parametrised over EVERY public frozen dataclass minus the
allowlist, so a newly exported value type is covered automatically — it either
satisfies the invariant or fails on the spot. Nobody has to remember to add it.

An earlier draft of this file also carried a
`test_every_public_value_type_is_accounted_for` that diffed the type list
against itself. It could never fail — `HEALTHY` is DERIVED from the same set it
was compared to — so it asserted nothing while looking like the file's
centrepiece. It was deleted rather than propped up with a hand-maintained list
the sweep does not need. Recording it here because writing a vacuous test into
the very file built to prevent vacuous tests is the easiest mistake here.

`KNOWN_BROKEN` is a shrink-only allowlist, in the same spirit as
`KNOWN_UNDOCUMENTED` in `test_docs_match_code.py`. Entries carry the reason.
Deleting one is always progress; adding one needs a justification in review.

THE TRAP THIS FILE IS BUILT TO AVOID
------------------------------------
A ratchet that constructs types with MINIMAL arguments passes vacuously. A
`Message` is perfectly hashable until it carries `tool_calls` — the exact bug.
So every instance built here is REPRESENTATIVE: containers are non-empty, and
nested agentkit types are built recursively rather than left as None. Twice
while developing this, a shortcut annotation match ("str" matching inside
`tuple[tuple[str, object], ...]`) produced a confident wrong answer, which is
why synthesis refuses rather than guesses when it is unsure.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle
from enum import Enum
from typing import Any

import pytest

import agentkit

# ── types that do NOT yet satisfy the invariant ────────────────────────────
#
# Every one is the same shape: a `frozen=True` dataclass carrying a mutable
# dict/list payload, so it is frozen in name only and unhashable in fact.
# Surfaced by this file rather than fixed by it — each needs its own decision
# about what belongs in the hash, which is a change to public semantics.
#
# SHRINK ONLY. Deleting an entry is progress; adding one needs a reason.
KNOWN_BROKEN: dict[str, str] = {
    # EMPTY, and that is the point of recording it.
    #
    # This started at twelve entries, every one the same shape: a `frozen=True`
    # dataclass carrying a mutable dict/list payload, unhashable in fact while
    # being reasoned about as a value. All twelve were fixed the same way — hash
    # an identity SUBSET, leave `__eq__` untouched — after checking case by case
    # that nothing in the framework mutated the payload in place.
    #
    # The payloads were deliberately NOT frozen. `Checkpoint.state` is
    # `json.dumps`'d into a JSONB column, `AgentResult.evals` and memory
    # metadata are serialised and written into after construction, and a
    # `MappingProxyType` is neither JSON-serialisable nor compatible with
    # `dataclasses.asdict`. Freezing them would have traded an inconvenience for
    # a data-path break. That mutability is a separate, still-open question —
    # `cp.state["turn"] = 99` does still rewrite a durable record through a
    # "frozen" field — and `tests/kernel/test_value_type_hashability.py`
    # guards the plain-dict shape so a future attempt to freeze them trips a
    # test rather than the Postgres adapter in production.
    #
    # Adding an entry here needs a reason in review; the entry must say which
    # field forces it. Deleting one is always progress.
}

# Types whose representative instance cannot be synthesized from annotations
# alone — Protocols and abstract seams have no fields to read. Register a
# concrete stand-in rather than letting synthesis guess.
FACTORIES: dict[str, Any] = {
    # `ContextScope` is a Protocol; `LastNTurns` is the simplest real impl.
    "ContextScope": lambda: agentkit.LastNTurns(2),
    # These carry a CROSS-FIELD constraint that annotations cannot express, so
    # synthesis would build something the constructor rightly rejects.
    # `Prompt` validates that every bound name was declared in `inputs`.
    "Prompt": lambda: agentkit.Prompt(
        id="p", version="1", template="hi {who}", inputs=("who",)
    ).bind(who="world"),
    # `Decision` / `Observation` are keyed by Literal aliases that are not
    # exported as classes, so there is nothing to introspect.
    "Decision": lambda: agentkit.Decision(kind="approve"),
    "Observation": lambda: agentkit.Observation(kind="tool_result", payload={"k": "v"}),
    # `WorkflowResult.stop_reason` is a Literal alias too, and it has no
    # default, so synthesis cannot reach the type at all.
    "WorkflowResult": lambda: agentkit.WorkflowResult(
        outputs={"node": "value"}, usage=agentkit.Usage(), steps=1, stop_reason="complete"
    ),
    # `VersionedEvent` is generic over the application's event type.
    "VersionedEvent": lambda: agentkit.VersionedEvent(version=1, event="e"),
}


def _public_frozen_dataclasses() -> dict[str, type]:
    out: dict[str, type] = {}
    for name in agentkit.__all__:
        obj = getattr(agentkit, name, None)
        if not inspect.isclass(obj) or not dataclasses.is_dataclass(obj):
            continue
        params = getattr(obj, "__dataclass_params__", None)
        if params is not None and params.frozen:
            out[name] = obj
    return out


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are NOT inside brackets.

    A naive ``text.split(",")`` turns ``tuple[str, object], ...`` into
    ``["tuple[str", " object]", " ..."]`` — the first fragment has an opening
    bracket and no closing one, and the next `rindex("]")` raises
    ``ValueError: substring not found``. That was a real bug in this file
    before it was a comment.
    """
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


class _Unsynthesizable(Exception):
    """Raised instead of guessing. A guess that happens to construct something
    is worse than a refusal: it produces a confident wrong verdict."""


def _synth(ann: Any, depth: int = 0) -> Any:
    """Build a REPRESENTATIVE value for a field annotation.

    Container checks come FIRST and containers are built NON-EMPTY. Both
    choices are load-bearing: matching `str` before `tuple` misreads
    `tuple[tuple[str, object], ...]` as a string, and an empty container hides
    exactly the payload bugs this file exists to catch.
    """
    if depth > 3:
        raise _Unsynthesizable("nested too deeply")
    # A module using `from __future__ import annotations` gives string
    # annotations; a class built at runtime gives the type object itself.
    # Handle both so the ratchet behaves the same either way.
    if isinstance(ann, type) and not dataclasses.is_dataclass(ann):
        if ann in (str, int, float, bool):
            return {str: "x", int: 1, float: 1.0, bool: False}[ann]
        if ann in (dict, list, set, tuple):
            raise _Unsynthesizable(f"bare {ann.__name__} annotation")
    text = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    low = text.lower()

    # Optional / unions: take the first member that we can build.
    if "|" in text:
        for part in text.split("|"):
            part = part.strip()
            if part == "...":
                continue
            if part.lower() in {"none", "nonetype"}:
                continue
            try:
                return _synth(part, depth + 1)
            except _Unsynthesizable:
                continue
        return None

    inner = text[text.index("[") + 1 : text.rindex("]")] if "[" in text else ""
    if low.startswith(("tuple", "sequence", "frozenset")):
        parts = _split_top_level(inner) if inner else ["str"]
        if parts and parts[-1] == "...":
            # Variadic `tuple[X, ...]` — one representative element is enough.
            return (_synth(parts[0], depth + 1),)
        # FIXED-LENGTH `tuple[str, object]` needs one value PER slot. Building
        # a 1-tuple here produced `ValueError: not enough values to unpack`
        # against `FrozenContext.scratchpad`, whose entries are (key, value)
        # pairs — a ratchet that cannot build the type cannot check it.
        return tuple(_synth(x, depth + 1) for x in parts)
    if low.startswith("list"):
        first = _split_top_level(inner)[0] if inner else "str"
        return [_synth(first, depth + 1)]
    if low.startswith(("dict", "mapping")):
        return {"k": "v"}
    if low.startswith("str"):
        return "x"
    if low.startswith("int"):
        return 1
    if low.startswith("float"):
        return 1.0
    if low.startswith("bool"):
        return False
    if low.startswith(("any", "object")):
        return "x"

    if low in {"...", "ellipsis"}:
        return "x"
    name = text.strip()
    if name in FACTORIES:
        return FACTORIES[name]()
    named = getattr(agentkit, name, None)
    if inspect.isclass(named):
        if issubclass(named, Enum):
            # An enum's own members are the only valid values; picking the
            # first is representative and never a guess.
            return next(iter(named))
        if dataclasses.is_dataclass(named):
            return _build(named, depth + 1)
    raise _Unsynthesizable(text)


def _build(cls: type, depth: int = 0) -> Any:
    if cls.__name__ in FACTORIES:
        return FACTORIES[cls.__name__]()
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        has_default = (
            f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        )
        try:
            value = _synth(f.type, depth)
        except _Unsynthesizable:
            if has_default:
                continue
            raise
        kwargs[f.name] = value
    return cls(**kwargs)


ALL_TYPES = _public_frozen_dataclasses()
HEALTHY = sorted(n for n in ALL_TYPES if n not in KNOWN_BROKEN)


@pytest.mark.parametrize("name", HEALTHY)
def test_a_public_value_type_behaves_like_a_value(name: str) -> None:
    """Hashable, deep-copyable, picklable — with a REPRESENTATIVE payload.

    `deepcopy` and `pickle` are not theoretical: `Checkpointer.snapshot`
    deep-copies state at the durable seam, and a `MappingProxyType` used to
    freeze a payload is unpicklable, which breaks BOTH. That combination is
    what bit `Prompt`, the signals, and `ToolCall`.
    """
    try:
        inst = _build(ALL_TYPES[name])
    except _Unsynthesizable as e:
        pytest.fail(
            f"cannot build a representative {name} (field type {e}). Register a "
            f"factory in FACTORIES so this type is really checked — do NOT relax "
            f"the synthesizer into guessing."
        )
    hash(inst)
    copy.deepcopy(inst)
    pickle.loads(pickle.dumps(inst))


@pytest.mark.parametrize("name", sorted(KNOWN_BROKEN))
def test_a_known_broken_type_is_still_broken(name: str) -> None:
    """The allowlist must not rot. When someone fixes one of these, this test
    fails and tells them to delete the entry — that is what makes the ratchet
    tighten instead of quietly accumulating stale excuses."""
    cls = ALL_TYPES.get(name)
    if cls is None:
        pytest.fail(f"{name} is in KNOWN_BROKEN but is no longer a public frozen dataclass")
    try:
        inst = _build(cls)
    except _Unsynthesizable as e:
        # NOT "still broken" — "never actually checked". Swallowing this is how
        # `WorkflowResult` sat on the allowlist looking verified while its hash
        # was never once called: `stop_reason` is a Literal alias with no
        # default, so `_build` raised and this test read the raise as proof of
        # brokenness. An allowlist entry that cannot be evaluated is worse than
        # no entry, because it looks like coverage.
        pytest.fail(
            f"{name} is on KNOWN_BROKEN but cannot be built (field type {e}), so "
            f"its entry has never been verified. Register a FACTORIES entry."
        )
    try:
        hash(inst)
    except TypeError:
        return
    pytest.fail(
        f"{name} is hashable now — delete it from KNOWN_BROKEN "
        f"(reason recorded there: {KNOWN_BROKEN[name]})"
    )


def test_the_allowlist_has_no_stale_entries() -> None:
    """A name that is no longer exported must be removed, or the list becomes a
    graveyard nobody trusts."""
    stale = sorted(set(KNOWN_BROKEN) - set(ALL_TYPES))
    assert not stale, f"KNOWN_BROKEN names that are no longer public: {stale}"


def test_the_synthesizer_builds_representative_payloads_not_empty_ones() -> None:
    """POSITIVE CONTROL for the ratchet itself.

    A `Message` is hashable until it carries `tool_calls` — which is exactly the
    bug that shipped. If the synthesizer built empty containers, every test
    above would pass while checking nothing. Assert the payloads are real.
    """
    msg = _build(agentkit.Message)
    assert msg.tool_calls, "Message built with an EMPTY tool_calls — the ratchet is vacuous"
    assert isinstance(msg.tool_calls[0], agentkit.ToolCall)


def test_the_synthesizer_refuses_rather_than_guesses() -> None:
    """POSITIVE CONTROL. Twice while writing this, a loose annotation match
    produced a confident wrong answer. Unknown types must raise, not fall back
    to None — a None-filled instance hashes fine and proves nothing."""
    with pytest.raises(_Unsynthesizable):
        _synth("SomeTypeThatDoesNotExist")
