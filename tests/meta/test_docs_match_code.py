"""The docs must not describe an API that does not exist.

Two doc-lies shipped in a single review round of this codebase:

* a docstring promised "a late decision can be refused" while ``resume()``
  never looked at the deadline — it wasn't even persisted;
* a recipe recommended a middleware wiring that aborted the run on the first
  malformed response.

Neither was catchable by reading, because both read perfectly. Prose is the
one part of a project with no compiler, and for a framework the docs ARE the
API — a reader who copies a snippet that names a symbol we deleted has hit a
bug we shipped.

So: every ``python`` block in ``docs/`` must PARSE, and every ``agentkit``
symbol it imports must actually exist. This is deliberately not a doctest
runner — the snippets legitimately need API keys, network, and a real
provider. Import-level truth is the part that can be checked cheaply and is
also the part that rots fastest.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re

import pytest

DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"

# ```python ... ``` and ```py ... ``` fences, with optional mkdocs attributes
# (`{ .python title="x" }`) after the language tag.
_FENCE = re.compile(r"^```(?:python|py)\b[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)

# ``mental-models/`` is a design-narrative directory whose snippets use
# deliberate placeholders like ``llm=<FakeLLM>``. Requiring those to parse
# would mean rewriting narrative prose into runnable code for no reader's
# benefit. The rest of the docs are held to the strict standard, because those
# are the ones someone copies from.
#
# This exclusion used to justify itself with "it is absent from the mkdocs
# nav". That stopped being true the moment those four scenarios were added to
# the nav — they had been orphaned and invisible, which was its own bug. The
# exclusion is still right; only its reason had to change. Recorded because a
# stale justification is how a correct exclusion gets deleted later by someone
# who checks the premise and finds it false.
_UNPUBLISHED = {"mental-models", "PUBLISHING.md"}

_MARKDOWN_FILES = sorted(
    p
    for p in DOCS.rglob("*.md")
    if not (set(p.relative_to(DOCS).parts) & _UNPUBLISHED)
)

# The full tree, for the "is this symbol mentioned anywhere" check — a name
# discussed only in a design note still counts as documented.
_ALL_MARKDOWN = sorted(DOCS.rglob("*.md"))


def _snippets(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every python fence in a file, with the line it starts on."""
    text = path.read_text()
    out = []
    for match in _FENCE.finditer(text):
        line = text[: match.start()].count("\n") + 2
        out.append((line, match.group(1)))
    return out


ALL_SNIPPETS = [
    pytest.param(path, line, code, id=f"{path.relative_to(DOCS)}:{line}")
    for path in _MARKDOWN_FILES
    for line, code in _snippets(path)
]


def test_the_docs_contain_python_snippets_to_check() -> None:
    """Guard against the checker silently covering nothing — if the fence
    regex stops matching (a docs tooling change, a different fence style),
    every test below would vacuously pass."""
    assert len(ALL_SNIPPETS) > 20, f"only found {len(ALL_SNIPPETS)} python snippets; regex broken?"


@pytest.mark.parametrize(("path", "line", "code"), ALL_SNIPPETS)
def test_every_documented_snippet_parses(path: pathlib.Path, line: int, code: str) -> None:
    """A snippet with a syntax error is one a reader cannot run at all."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        pytest.fail(
            f"{path.relative_to(DOCS)}:{line} does not parse — {exc.msg} "
            f"(snippet line {exc.lineno})"
        )


def _imported_agentkit_symbols(code: str) -> list[tuple[str, str]]:
    """``(module, name)`` for every ``from agentkit... import name``.

    Only ``from``-imports are checked. A bare ``import agentkit.x`` is covered
    by the module-exists half of the same assertion.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # reported by the parse test; don't double-fail
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("agentkit"):
            for alias in node.names:
                if alias.name != "*":
                    found.append((node.module, alias.name))
    return found


SYMBOL_CASES = [
    pytest.param(path, line, module, name, id=f"{path.relative_to(DOCS)}:{line}:{module}.{name}")
    for path, line, code in [(p.values[0], p.values[1], p.values[2]) for p in ALL_SNIPPETS]
    for module, name in _imported_agentkit_symbols(code)
]


def test_the_docs_import_agentkit_symbols_to_check() -> None:
    """Same vacuity guard as above, for the symbol half."""
    assert len(SYMBOL_CASES) > 20, f"only found {len(SYMBOL_CASES)} imports; extractor broken?"


@pytest.mark.parametrize(("path", "line", "module", "name"), SYMBOL_CASES)
def test_every_symbol_the_docs_import_actually_exists(
    path: pathlib.Path, line: int, module: str, name: str
) -> None:
    """The load-bearing check.

    A rename that updates the code and the tests but not the docs leaves a
    snippet that fails on line one. That is the single most common way a
    framework's documentation goes stale, and it is trivially detectable.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - only on a genuinely broken doc
        pytest.fail(f"{path.relative_to(DOCS)}:{line} imports {module!r}, which does not exist: {exc}")

    if hasattr(mod, name):
        return
    # ``from agentkit.adapters.llm import providers`` imports a SUBMODULE, which
    # is not an attribute of the parent package until something imports it.
    # Checking `hasattr` alone would flag every such line as missing.
    try:
        importlib.import_module(f"{module}.{name}")
        return
    except ImportError:
        pass
    assert hasattr(mod, name), (
        f"{path.relative_to(DOCS)}:{line} imports {name!r} from {module!r}, "
        f"but that module has no such attribute. The docs describe an API that "
        f"does not exist — rename the doc or restore the export."
    )


# ── the reverse direction: the public surface is documented somewhere ────────


# A RATCHET, not a target. These public exports predate this check and are not
# mentioned anywhere in ``docs/``. The test below fails on any NEW undocumented
# export, and also fails if one of these gets documented without being removed
# from the list — so the number can only go down.
#
# Do not add to this list to make a build green. Write the doc.
KNOWN_UNDOCUMENTED: frozenset[str] = frozenset({
    "AgentkitError",
    "ControlSignal",
    "DataSignal", "ErrorClass",
    # "PrefixContext" / "FrozenContext" / "ContextDiff" / "ContextScope" /
    # "LastNTurns" / "RoleFilter" / "Tagged" / "Since" / "AllOf" / "AnyOf" /
    # "ApproxTokenCounter" / "TiktokenCounter" ratcheted off:
    # docs/concepts/context.md documents the four axes, the cache-stable
    # prefix, the slice predicates and the token-counter seam.
    # "SearchHit" / "FetchResponse" / "ProviderAuthError" ratcheted off:
    # docs/concepts/adapters.md documents the port table, the normalised
    # search/fetch result shapes and the provider error taxonomy.
    # "NoopObserver" / "NoopMetrics" / "NoopReplayStore" / "ReplayRecord" /
    # "AlwaysOnSampler" / "TraceIdRatioSampler" ratcheted off:
    # docs/concepts/observability.md documents the no-op defaults on
    # ``Services``, the replay side channel, and the sampler seam.
    # "ReplayStore" ratcheted off: the replay-store section in
    # docs/api-reference/adapters.md now names the protocol and its put()
    # contract while documenting AGENTKIT_REPLAY_DIR.
    # "MutationJournal" ratcheted off: docs/concepts/memory.md names it as
    # JournalMemory's backing store.
    # "OutputCoercionError" / "ToolShapeError" / "render_result" ratcheted
    # off: docs/concepts/tools.md documents the tool error taxonomy and the
    # as_tool result renderer.
    # "Reranker" / "score_sort_rerank" ratcheted off: docs/concepts/memory.md
    # documents the CompositeMemory rerank seam and its default.
    "StoreUnavailable",
    "VersionedEvent", "collect_one", "compose_failures",
})


def test_every_public_export_is_mentioned_in_the_docs() -> None:
    """A symbol nobody documented is a symbol nobody can find.

    Checked against the whole ``docs/`` tree as plain text rather than against
    a curated list, so mentioning a name anywhere — a recipe, a concept page,
    the cheatsheet — counts. The point is not thoroughness of prose; it is
    catching a public export that was added and never written about.

    Exemptions are listed explicitly with a reason, so growing the list is a
    visible decision rather than a silent one.
    """
    import agentkit

    corpus = "\n".join(p.read_text() for p in _ALL_MARKDOWN)

    # Names that are legitimately undocumented in prose.
    exempt = {
        "__version__",  # metadata, not API
        "middlewares",  # re-exported submodule; its members are documented
        "streams",  # ditto
    }

    missing = sorted(
        name
        for name in (agentkit.__all__ or ())
        if name not in exempt and name not in corpus
    )
    new = sorted(set(missing) - KNOWN_UNDOCUMENTED)
    assert not new, (
        "these public exports are never mentioned anywhere in docs/:\n  "
        + "\n  ".join(new)
        + "\n\nDocument them, or drop them from __all__ if they are not public."
    )

    # The ratchet only tightens. If a name here has since been documented,
    # remove it from the baseline in the same commit.
    fixed = sorted(KNOWN_UNDOCUMENTED - set(missing))
    assert not fixed, (
        "these names are now documented — delete them from KNOWN_UNDOCUMENTED so "
        "the ratchet keeps tightening:\n  " + "\n  ".join(fixed)
    )
