# Runnable examples

Run from the repository root after `uv sync --all-extras`. Each example owns its
own package, entry point and supporting files; there are no shared application
globals or credentials between examples.

| Directory | Run command | Execution |
| --- | --- | --- |
| `call_runtime/` | `uv run python -m tests.examples.call_runtime` | Scripted one-shot call |
| `single_agent/` | `uv run python -m tests.examples.single_agent` | Scripted AgentRunner |
| `agent_runner_tools/` | `uv run python -m tests.examples.agent_runner_tools` | Scripted host-tool loop |
| `react_tools_and_events/` | `uv run python -m tests.examples.react_tools_and_events` | Scripted ReAct and events |
| `support_harness/` | `uv run python -m tests.examples.support_harness --help` | Host-owned support harness |
| `plugin_execution/` | `uv run python -m tests.examples.plugin_execution.main --provider openai_api` | Credential-free assessment by default |

The four numbered scripts formerly directly in this directory moved into the
first four packages above. Use the module commands instead of their old paths.
The support harness retains its existing module entry point. The plugin example
includes its own [setup, tests, limitations and explicit paid-call opt-in](plugin_execution/README.md).

```sh
AGENTKIT_SKIP_REAL_CLI=1 AGENTKIT_LIVE=0 uv run pytest -q tests/examples/plugin_execution/tests
uv run pytest -q tests/source/conformance/test_example_layout.py
```

Directory isolation organizes code and call-owned state; it does not sandbox
Python code or CLI native tools. Examples are source-tree development material,
excluded from the library wheel.
