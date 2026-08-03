# Security Policy

The agentkit maintainers take security bugs seriously. Thank you for improving
the security of agentkit responsibly.

## Supported versions

Security fixes are backported to the current minor release only. Pre-1.0
releases carry no long-term support guarantee — please stay close to `main`.

| Version   | Supported          |
|-----------|--------------------|
| `0.x`     | Latest minor only  |
| `< 0.1`   | No                 |

Once agentkit reaches `1.0`, this table will expand to cover the latest two
minor versions.

## Reporting a vulnerability

**Please do not report security issues through public GitHub issues, discussions,
or pull requests.**

Report vulnerabilities via one of the following channels:

1. **Preferred**: GitHub's private vulnerability reporting on this repository —
   the "Report a vulnerability" button under the *Security* tab.
2. **Email**: `security@arc-labs.ai` with the subject line
   `[agentkit] vulnerability report`. Encrypt sensitive details with our PGP
   key on request.

Include as much of the following as you can:

- A clear description of the issue and its impact.
- A minimal reproduction (code snippet, sequence of calls, or PoC).
- The affected version(s) or commit SHA.
- Any suggested mitigation or fix.

## Disclosure timeline

We aim to follow a **coordinated disclosure** model:

| Day    | What happens                                                    |
|--------|-----------------------------------------------------------------|
| 0      | Report received. Acknowledgement sent within **72 hours**.      |
| 0–7    | Triage: reproduce, assess severity (CVSS), scope affected code. |
| 7–30   | Fix developed and tested privately in a security branch.        |
| 30–90  | Release published; advisory + CVE (if applicable) coordinated.  |
| +90    | Full public disclosure, credit to reporter (if desired).        |

For actively exploited or trivially exploitable issues we will move faster and
coordinate a shorter embargo with the reporter.

## Scope

In scope:

- The `arc-agentkit` Python package published to PyPI (imported as `agentkit`).
- Source in this repository, including build tooling (`pyproject.toml`,
  workflows under `.github/workflows/`).

Out of scope:

- Vulnerabilities in third-party dependencies (report those upstream — we will
  update our pins once a fix is released).
- Downstream applications that consume agentkit as a dependency — report to
  that application's own security channel.
- Social engineering of maintainers, physical attacks, or issues in test-only
  code paths.

## Credit

With your permission we will acknowledge reporters in the release notes and the
GitHub Security Advisory for the fix. If you prefer to remain anonymous, tell
us in the report.
