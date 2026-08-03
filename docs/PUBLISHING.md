# Publishing agentkit

How releases work. Read this before your first release.

## PyPI distribution name

**Distribution name:** `arc-agentkit` (the string you `pip install`).
**Import name:** `agentkit` (unchanged — every existing user of `from agentkit import ...` keeps working).

PyPI's namespace for `agentkit` is owned by a different unrelated project
(`vik@japanvik.net`, "distributed llm agent swarms", v0.0.4 · Mar 2026).
Reserving a brand-attributed name keeps discovery clean and preserves the
`import agentkit` ergonomics.

The distinction is standard in Python packaging — e.g. `pip install Django` →
`import django`, `pip install python-dotenv` → `import dotenv`.

## Release flow (once per version)

We publish via **PyPI Trusted Publisher (OIDC)** — no long-lived PyPI API
tokens live in the repo or in secrets. GitHub Actions signs a request to
PyPI, PyPI verifies against the trusted-publisher configuration on the
project page, and the release is uploaded. If you leaked a token, PyPI would
issue a warning and revoke; with OIDC there's no token to leak.

### One-time setup

1. **Register the distribution on TestPyPI first** (dry-run environment):
   - Go to https://test.pypi.org/manage/account/publishing/
   - "Add a new publisher"
   - Owner: `arc-labs-ai`
   - Repository: `agentkit`
   - Workflow name: `release.yml`
   - Environment: `pypi` (matches the `environment:` block in `release.yml`)

2. **Same again for real PyPI**:
   - https://pypi.org/manage/account/publishing/
   - Same shape.

3. **First-time publish will fail on PyPI** until PyPI has a version of the
   package registered. Options: register the name via an initial dummy release
   from the maintainer's account (bootstrap), OR publish to TestPyPI first and
   then request PyPI reserve the name.

### Every release

1. **Update `CHANGELOG.md`** — new entry at the top under the new version
   number (semver). Format: `## [0.2.0] - 2026-XX-XX` + bullet list of
   changes.
2. **Bump version** in `pyproject.toml`. Semver: patch for bug fix, minor for
   backward-compatible feature, major for breaking change.
3. **Merge to main** with the changelog + version bump.
4. **Tag the commit** with `v<version>` (matching pyproject exactly):

   ```bash
   git tag v0.2.0 && git push origin v0.2.0
   ```

5. GitHub Actions `release.yml` fires on the tag:
   - Builds wheel + sdist via `hatchling`.
   - Runs `twine check dist/*` to catch metadata errors before upload.
   - Requests OIDC token, publishes to PyPI.
   - Creates a GitHub Release with generated notes.

6. **Verify** on PyPI (~2 minutes after publish):

   ```bash
   pip install --upgrade arc-agentkit
   python -c "from importlib.metadata import version; print(version('arc-agentkit'))"
   ```

## Local dry-run before tagging

Always run this before creating the tag:

```bash
make build          # hatchling → dist/*.whl + dist/*.tar.gz
make publish-dry    # twine check dist/*
```

If `publish-dry` complains about long-description rendering, README markdown,
or invalid classifiers — fix those in the same commit as the changelog and
version bump. Failing publishes leave dead space on the version list.

## SemVer policy

We're pre-1.0 (currently 0.x). Rules we hold:

- **Patch** (`0.x.PATCH`) — bug fixes, docs, internal refactors that don't
  change the public API's behavior.
- **Minor** (`0.MINOR.0`) — new public API surface OR intentional
  behavior changes to existing API surface. Log in CHANGELOG under
  "Changed" / "Deprecated" / "Removed" sections.
- **Major** (`MAJOR.0.0`) — reserved for the 1.0 stabilization pass and
  any subsequent hard-breaking rewrites.

Public API = anything exported from `agentkit/__init__.py` or a top-level
subpackage's `__init__.py`. Anything under `_private.py` or prefixed with
`_` is fair game to change in a patch release.

## Yanking a bad release

If a release ships broken:

```bash
pip install --upgrade twine
twine yank arc-agentkit 0.2.0 --reason "critical bug in X — use 0.2.1"
```

Yanking hides the version from `pip install arc-agentkit` (unbounded)
but leaves `pip install arc-agentkit==0.2.0` working for anyone who
pinned it. Then ship the fixed version.

Never delete a released version. PyPI enforces this — deletions break
downstream lockfiles for anyone who pinned to the yanked version.

## Common gotchas

- **Trusted publisher misconfigured** → 403 on upload. Check that the
  workflow file name in the publisher config matches `release.yml`
  exactly.
- **Wheel builds but sdist doesn't** → hatchling's `[tool.hatch.build]`
  didn't include a source file. Add it to `packages` or
  `include`.
- **`twine check` warns about README** → PyPI parses the README as
  reStructuredText by default. We set `readme = "README.md"` and PyPI
  handles the `text/markdown` content-type via `content_type` inference
  from the file extension. Make sure the `content_type` metadata field
  is set to `text/markdown` in the built wheel's METADATA.
- **Version mismatch between tag and `pyproject.toml`** → CI fails
  early. Fix the mismatch before re-tagging.
