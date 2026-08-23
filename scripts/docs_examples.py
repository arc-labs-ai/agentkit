#!/usr/bin/env python
"""Do the documented examples still work?

Two checks, both aimed at one failure mode this project keeps hitting: prose
written from intent while the code moved underneath it. `mkdocs build --strict`
proves the links resolve; `tests/meta/test_docs_match_code.py` proves the
symbols are mentioned. Neither runs a single line of the code a reader copies.

    imports   every `from agentkit… import X` written in docs/ resolves.
              Fast (one import per module). This is the half worth running on
              every gate, and it covers the fragments too — a snippet that
              never executes can still name something that does not exist.

    fences    every ```python block is executed in a subprocess. Slow: one
              process per block. Run it before a release or after touching
              docs, not on every commit.

Blocks that legitimately cannot run offline are reported by CATEGORY rather
than counted as failures, because "needs ANTHROPIC_API_KEY" and "this snippet
is wrong" are different problems and lumping them together buries the second:

    key       needs a provider credential
    extra     needs an optional third-party package (langchain, openai, …)
    fragment  a deliberate continuation — `await` at module level, or a name
              defined in an earlier block

Anything else is a real failure and exits non-zero.

`mental-models/` is skipped: it is a design-narrative directory whose snippets
use deliberate placeholders like ``llm=<FakeLLM>``, the same exclusion
``tests/meta/test_docs_match_code.py`` makes and for the same reason.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap

DOCS = pathlib.Path("docs")
SKIP_DIRS = {"mental-models"}
# Fences inside a `!!!` admonition are indented four spaces by Material's
# syntax; without dedenting, every one of them looks like an IndentationError.
FENCE = re.compile(r"```python\n(.*?)```", re.S)
IMPORT = re.compile(r"^\s*from (agentkit[\w\.]*) import ([^\n(]+)$", re.M)


def _pages() -> list[pathlib.Path]:
    return sorted(p for p in DOCS.rglob("*.md") if not (SKIP_DIRS & set(p.parts)))


def check_imports() -> int:
    """Every documented `from agentkit… import X` must resolve."""
    bad: list[str] = []
    checked = 0
    for page in _pages():
        for module, names in IMPORT.findall(page.read_text()):
            try:
                mod = importlib.import_module(module)
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                bad.append(f"{page}: cannot import {module} ({type(exc).__name__})")
                continue
            for raw in names.split(","):
                name = raw.strip().split(" as ")[0].strip()
                if not name or not name.isidentifier():
                    continue
                checked += 1
                # `from pkg import submodule` is valid even though the
                # attribute does not exist until the submodule is imported, so
                # fall back to importing it rather than reporting a false miss.
                if hasattr(mod, name):
                    continue
                try:
                    importlib.import_module(f"{module}.{name}")
                except Exception:  # noqa: BLE001
                    bad.append(f"{page}: {module}.{name} does not exist")
    print(f"  documented symbols checked : {checked}")
    print(f"  unresolved                 : {len(bad)}")
    for line in bad:
        print(f"    {line}")
    return 1 if bad else 0


def _classify(stderr: str) -> str | None:
    if "API_KEY" in stderr or "ProviderNotConfigured" in stderr:
        return "key"
    if "ModuleNotFoundError" in stderr:
        return "extra"
    if "outside function" in stderr or "outside async function" in stderr:
        return "fragment"
    if "NameError" in stderr:
        return "fragment"
    return None


def check_fences() -> int:
    ran = total = 0
    buckets = {"key": 0, "extra": 0, "fragment": 0}
    failures: list[str] = []
    for page in _pages():
        blocks = [textwrap.dedent(b) for b in FENCE.findall(page.read_text())]
        for i, block in enumerate(blocks, 1):
            total += 1
            if len(block.strip()) < 20:
                ran += 1
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                fh.write(block)
                path = fh.name
            try:
                proc = subprocess.run(  # noqa: S603
                    [sys.executable, path], capture_output=True, text=True, timeout=240
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{page} #{i}: TIMEOUT")
                continue
            if proc.returncode == 0:
                ran += 1
                continue
            kind = _classify(proc.stderr)
            if kind:
                buckets[kind] += 1
            else:
                last = (proc.stderr.strip().splitlines() or ["?"])[-1]
                failures.append(f"{page} #{i}: {last[:120]}")
    print(f"  fences executed            : {ran}/{total}")
    print(f"    needs a credential       : {buckets['key']}")
    print(f"    needs an optional extra  : {buckets['extra']}")
    print(f"    deliberate fragment      : {buckets['fragment']}")
    print(f"  real failures              : {len(failures)}")
    for line in failures:
        print(f"    {line}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--imports-only",
        action="store_true",
        help="run only the fast import check (suitable for every gate)",
    )
    args = ap.parse_args()
    rc = check_imports()
    if not args.imports_only:
        rc |= check_fences()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
