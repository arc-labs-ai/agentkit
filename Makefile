# agentkit — single command surface.
# CI and contributor tooling call these targets; the CI config stays stable
# even if the underlying tool changes. Python via uv; builds via hatchling.

.DEFAULT_GOAL := help
.PHONY: help setup test test-matrix lint typecheck check cov mutants docs-check docs-examples build publish-dry docs-serve clean

# ── Help ────────────────────────────────────────────────────────────────────

help: ## Show this help
	@echo ""
	@echo "agentkit — common targets"
	@echo ""
	@echo "  Boot for local dev:"
	@echo "    make setup           # install runtime + dev deps into .venv"
	@echo "    make test            # run the test suite"
	@echo "    make cov             # tests + branch-coverage threshold"
	@echo "    make mutants         # do the tests actually catch a broken invariant?"
	@echo "    make check           # the gate that must be green before 'done'"
	@echo "    make docs-examples   # slow: execute every code block in docs/"
	@echo ""
	@echo "  All targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS=":.*?## "}; {printf "    \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────────────

setup: ## Install all dependencies (runtime + dev) via uv
	uv sync --all-extras

# ── Test ────────────────────────────────────────────────────────────────────

test: ## Run pytest
	uv run pytest

test-matrix: ## Run the suite on every supported interpreter (3.12 and 3.13)
	@# `check` runs ONE interpreter — whichever .venv happens to hold — and CI
	@# runs the matrix, so a version-specific break passes locally and fails on
	@# push. That is not hypothetical: a test pinned a CPython error message,
	@# then pinned a CPython bug that upstream fixed in a 3.13 PATCH release,
	@# and both landed green here and red in CI. `requires-python = ">=3.12"`
	@# and the classifiers promise 3.13, so the gate should check it.
	uv run --python 3.12 pytest -q
	uv run --python 3.13 pytest -q

# ── Lint ────────────────────────────────────────────────────────────────────

lint: ## Lint (ruff check)
	uv run ruff check agentkit tests

# ── Typecheck ───────────────────────────────────────────────────────────────

typecheck: ## Type-check the public surface (mypy --strict)
	uv run mypy --strict agentkit

# ── Coverage ────────────────────────────────────────────────────────────────

cov: ## Run the suite under branch coverage and enforce the threshold
	uv run --with coverage coverage run -m pytest
	uv run --with coverage coverage report

# ── Mutation testing ────────────────────────────────────────────────────────
# Coverage proves a line RAN; this proves a test would NOTICE if it were wrong.
# Not part of `check` — it rewrites source in place, so it wants a clean tree.

mutants: ## Run the curated mutant catalogue (see scripts/mutants.py)
	uv run python scripts/mutants.py

mutants-verify: ## Fast: check every mutant anchor still resolves
	uv run python scripts/mutants.py --verify

# ── Docs examples ───────────────────────────────────────────────────────────
# `mkdocs build --strict` proves the links resolve and the meta tests prove the
# symbols are mentioned; neither runs a line of the code a reader copies.

docs-check: ## Fast: every documented `from agentkit… import X` resolves
	uv run python scripts/docs_examples.py --imports-only

docs-examples: ## Slow: execute every ```python block in docs/
	uv run python scripts/docs_examples.py

# ── The gate ────────────────────────────────────────────────────────────────

check: lint typecheck mutants-verify docs-check cov test-matrix ## The gate: lint + types + docs + coverage-gated tests on every supported Python
	@echo "✓ all checks passed"

# ── Build / publish ─────────────────────────────────────────────────────────

build: ## Build wheel + sdist into dist/
	rm -rf dist
	uv build

publish-dry: build ## Build + validate distributions with twine (no upload)
	uv run --with twine twine check dist/*

# ── Docs ────────────────────────────────────────────────────────────────────

docs-serve: ## Serve the mkdocs site locally with live reload
	uv sync --group docs
	uv run mkdocs serve

# ── Housekeeping ────────────────────────────────────────────────────────────

clean: ## Remove build artefacts + caches
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .hypothesis site .mutants-backup
	rm -f .coverage .coverage.*
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
