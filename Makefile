# agentkit — single command surface.
# CI and contributor tooling call these targets; the CI config stays stable
# even if the underlying tool changes. Python via uv; builds via hatchling.

.DEFAULT_GOAL := help
.PHONY: help setup test lint typecheck check build publish-dry docs-serve clean

# ── Help ────────────────────────────────────────────────────────────────────

help: ## Show this help
	@echo ""
	@echo "agentkit — common targets"
	@echo ""
	@echo "  Boot for local dev:"
	@echo "    make setup           # install runtime + dev deps into .venv"
	@echo "    make test            # run the test suite"
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

# ── The gate ────────────────────────────────────────────────────────────────

check: lint typecheck test ## The gate: lint + typecheck + tests must all pass
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
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .hypothesis site
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
