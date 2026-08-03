# Contributing to agentkit

Thanks for taking the time to contribute. agentkit is a small, async-first
framework for building agentic-AI features, and it aims to stay a **low-level**
library: composable primitives, no framework-flavoured magic, and a public
surface that is stable enough to build on.

This document is a shortcut through the parts of that goal that touch you as a
contributor: how to get set up, how PRs are reviewed, and what "green" looks like.

## Getting set up

agentkit uses [`uv`](https://docs.astral.sh/uv/) for Python dependency
management and virtualenvs. The only prerequisite is a recent uv (>= 0.5)
and Python 3.12 or 3.13 available on your machine.

```bash
git clone https://github.com/arc-labs-ai/agentkit.git
cd agentkit
make setup       # uv sync — installs runtime + dev deps into .venv
make test        # pytest — should be green on a fresh clone
```

If `make test` is not green on `main`, that is a bug — please open an issue.

Editor tips:

- `uv run <cmd>` runs a command inside the project venv without activating it.
- `uv run pytest tests/path/to/test_file.py -k my_case` is the fastest inner loop.
- The `agentkit` package is `src`-less: import paths mirror the directory layout
  under `agentkit/`.

## Branches, commits, PRs

- Branch off `main`; keep branches short-lived. Rebase on top of `main` before
  requesting review rather than merging `main` back in.
- One concern per PR. If a change is easier to review split in two, split it.
- Commit messages are **imperative present tense**: "add retry to LLM adapter",
  not "added retry" or "adds retry". First line ≤ 72 chars, body wraps at 72.
  Reference issues with `Fixes #123` in the body when applicable.
- Draft PRs are welcome — mark them "Draft" until you want review.

## Adding tests

Every behavioural change needs a test. The test suite is `pytest` +
`pytest-asyncio` and lives under `tests/`, mirroring the package layout
(`tests/kernel/`, `tests/runtime/`, etc.).

- Prefer unit tests that exercise a single primitive. Integration tests exist
  but should be reserved for cases where composition is the thing under test.
- Async tests use `@pytest.mark.asyncio`. Do **not** call `asyncio.run` inside a
  test — let pytest-asyncio own the loop.
- No sleeps for synchronisation. If a test needs to wait for a signal, use an
  `asyncio.Event` or `asyncio.wait_for` with a tight timeout.
- Property tests use `hypothesis`. Keep strategies deterministic (seed inside
  the strategy, not at module scope) so failures are reproducible.
- Fixtures shared by more than one file go in the nearest `conftest.py`, not
  inside a test module.

If your change touches the public API surface (`agentkit/__init__.py`,
`agentkit/client.py`, top-level module re-exports), add a test that imports
and exercises the exact symbol as an external user would.

## Code review checklist

Before requesting review, walk through this list yourself:

- [ ] `make check` is green locally (lint + typecheck + tests).
- [ ] Public-surface symbols have docstrings and type hints; internal helpers
      at least have type hints.
- [ ] No new sync I/O on hot paths — every I/O boundary is `async def` +
      `await`. No `time.sleep`, no blocking HTTP clients.
- [ ] No new dependencies without a note in the PR body justifying them.
      Optional deps go under `[project.optional-dependencies]` in
      `pyproject.toml`, not the base install.
- [ ] Nothing under `agentkit/` imports from `tests/`, and nothing under
      `agentkit/` carries product-specific concepts from any consumer. This is
      a framework; it should have no knowledge of any particular product.
- [ ] Changelog entry added if the change is user-visible.
- [ ] Docs under `docs/` updated if you added/renamed/removed a public symbol.

Reviewers will look for:

1. **Does the public API stay minimal?** A new export is a maintenance promise.
2. **Are error paths modelled explicitly?** No swallowed exceptions, no
   `except Exception: pass` — surface failures to the caller.
3. **Is cancellation handled?** Long-running coroutines must be cancel-safe:
   respond to `CancelledError`, release resources in `finally`.
4. **Is the change documented where a first-time reader would look for it?**

## CI gates

Every PR runs the following via `.github/workflows/ci.yml`:

- `make lint` — `ruff check` on the whole tree.
- `make typecheck` — `mypy --strict` on `agentkit/` (the public surface).
- `make test` — `pytest` on Python 3.12 and 3.13.

The `check` target is exactly what CI runs, so if `make check` is green locally
it will be green on CI. If it fails on CI but passes locally, that is a bug —
please flag it in your PR rather than papering over with a retry.

Release publishing (`.github/workflows/release.yml`) is triggered by tags of
the form `v*.*.*` and uses PyPI **trusted publishing** (OIDC) — there are no
API tokens in repo secrets. Only maintainers cut releases.

## Reporting security issues

Please do not open a public issue for a security concern. See
[`SECURITY.md`](./SECURITY.md) for the private disclosure process.

## Code of conduct

By participating you agree to abide by our
[Code of Conduct](./CODE_OF_CONDUCT.md).
