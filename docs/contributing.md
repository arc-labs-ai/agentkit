---
title: Contributing
---

# Contributing

The authoritative guide lives at the repository root:

- **[`CONTRIBUTING.md`](https://github.com/arc-labs/agentkit/blob/main/CONTRIBUTING.md)** — full contributor guide.
- **[`CODE_OF_CONDUCT.md`](https://github.com/arc-labs/agentkit/blob/main/CODE_OF_CONDUCT.md)** — the community standards we hold each other to.
- **[`CHANGELOG.md`](https://github.com/arc-labs/agentkit/blob/main/CHANGELOG.md)** — release notes.

## Quick pointers

- **Bug reports and feature requests** — open an issue on
  [GitHub](https://github.com/arc-labs/agentkit/issues).
- **Design changes** — start a discussion before opening a large PR,
  especially for anything touching the kernel, the middleware
  contract, or an agent Protocol.
- **Docs improvements** — edit files under `docs/` and open a PR; the
  "Edit this page" link at the top of every page takes you there
  directly.

## Local docs preview

```bash
uv pip install mkdocs-material 'mkdocstrings[python]' mkdocs-git-revision-date-localized-plugin
uv run mkdocs serve
```

Then open <http://127.0.0.1:8000>.
