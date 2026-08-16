# agentkit — single command surface.
# CI and contributor tooling call these targets; the CI config stays stable
# even if the underlying tool changes. Python via uv; builds via hatchling.

.DEFAULT_GOAL := help
.PHONY: help setup test lint typecheck check cov mutants build publish-dry docs-serve clean

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

# ── The gate ────────────────────────────────────────────────────────────────

check: lint typecheck mutants-verify cov ## The gate: lint + types + coverage-gated tests
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
