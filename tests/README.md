# Validation and usage

The tree separates source-code verification, real-model evaluation, and runnable examples.

```text
tests/
  source/
    unit/
      core/ models/ agents/ state/ tools/ integrations/
    conformance/   shared runtime/provider contracts
    integration/   external backend acceptance
    architecture/ package ownership, exports, docs and value invariants
    evaluation/   tests of the eval engine, graders and sandbox
    fixtures/     recorded source-test inputs
  evals/
    engine/       typed contracts, configuration, runner and reporting
    suites/       application workflows and provider conformance
    graders/      independent acceptance oracles
    fixtures/     attributed datasets, business records and source regressions
    sandbox/      bounded workspace and process tools
    reports/      retained redacted run artifacts
  examples/       executable public API composition
```

Pytest defaults to `tests/source`. It does not collect deliberately failing coding fixtures, call real model APIs, or execute examples simply because they live under `tests`.

```bash
AGENTKIT_SKIP_REAL_CLI=1 uv run pytest
uv run mypy --strict agentkit
uv run ruff check agentkit tests
python -m tests.evals --list
python -m tests.examples.support_harness --help
python -m tests.examples.plugin_execution.main --provider openai_api
```

Every runnable example owns a directory under `tests/examples`. See its
[index](examples/README.md) for the renamed module entry points and the plugin
example's separate scripted, real-CLI and paid-provider test instructions.

Run live evaluations explicitly through `python -m tests.evals`; see [the eval guide](evals/README.md) for credentials, model selection, bounds and reports. Framework source tests should mirror the owning package and assert behavior rather than import compatibility. Evaluation-engine tests verify graders and failure handling; they are not substitutes for live application evidence.

Keep original failed reports. Count paired success/failure coverage only when both workflows pass on the same supported backend. Unsupported modes, missing credentials and budget exhaustion are distinct from observed model failures.
