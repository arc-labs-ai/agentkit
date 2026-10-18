# Independent tools, skills and hooks through one runtime

This example needs only this repository. The host declares `lookup_record`,
captures an in-memory collection for each call, loads versioned skill guidance
and composes resource permissions and lifecycle observers into `ToolSet`.
`CallRuntime.run` owns all model turns, attempts, usage and terminal outcomes.
The example implements no MCP server, permission dispatcher or model loop.

## Layout

```text
plugin_execution/
  main.py
  configuration/providers.py
  plugins/tools/{catalog,handlers}.py
  plugins/skills/instructions.md
  plugins/hooks/{permissions,lifecycle}.py
  wiring/runtime.py
  tests/conftest.py
  tests/{test_provider_matrix,test_permissions,test_scope_isolation,test_cli_containment}.py
  README.md
```

The four providers use the same declarations, implementations, skill text,
permission callback and lifecycle hooks. Only their `Binding` changes. The skill
text grants nothing. A tool must be in the explicit registry, its input must
validate, and the permission hook must allow the requested record in this call's
scope. An ALLOW from one contribution cannot override another's DENY. ASK without
an explicit approval handler refuses. The collection lives in the host callback,
not in a provider workspace. Observers receive IDs and safe status codes; they
do not receive arguments or returned record data.

## Setup and credential-free commands

From the repository root:

```sh
uv sync --all-extras
uv run python -m tests.examples.plugin_execution.main --provider openai_api
uv run python -m tests.examples.plugin_execution.main --provider anthropic_api
uv run python -m tests.examples.plugin_execution.main --provider claude_code
uv run python -m tests.examples.plugin_execution.main --provider codex
AGENTKIT_SKIP_REAL_CLI=1 AGENTKIT_LIVE=0 uv run pytest -q tests/examples/plugin_execution/tests
```

Without `--execute`, the program only assesses requirements. CLI assessment may
launch `--version` with an empty temporary HOME/environment. It resolves no
provider credential and starts no model execution. Native CLI binaries are
optional for the scripted tests; missing binaries remain incompatibilities.

The scripted API tests use real `ApiAdapter` instances with `httpx.MockTransport`
and the actual runtime/ToolSet. They prove the granted schema reaches dispatch,
the model receives the host result, the permission hook runs before the handler,
IDs survive, and terminal usage and call events settle. Negative twins cover
denial, unattended approval, undeclared tools, concurrent record scopes and
cancellation while awaiting authorization.

The CLI rows reach the same host tool over the wire a CLI actually uses. The
`scripted_cli` fixture stands in for the binary at the spawn seam: it is handed
the argv and environment a real `claude`/`codex` process would have received,
reads the host-tool MCP endpoint `CliAdapter` published there, and calls the
tool over that authenticated loopback transport. `ToolSet.serve()` wraps each
tool so its `run()` delegates to `ToolSet.execute`, so a CLI call lands on the
same permission contributions, observers and invocation identities the API
adapters use — not on a second copy of the decision. These rows therefore prove
the **transport and the policy path**. They prove nothing about a real CLI
process: there is no process, native tool, sandbox or credential in them.
Real-process characterization lives in `tests/source/integration/execution`,
and the refusal the example's strict profile produces for both CLI modes is
pinned in `tests/test_cli_containment.py`.

| Demonstrated case | OpenAI API | Anthropic API | Claude Code | Codex |
| --- | --- | --- | --- | --- |
| Granted call reaches the handler, model reads the host result | Scripted HTTP | Scripted HTTP | Scripted CLI | Scripted CLI |
| Denied call: handler never runs, refusal recorded | Scripted HTTP | Scripted HTTP | Scripted CLI | Scripted CLI |
| Undeclared tool refused before any handler | Scripted HTTP | Scripted HTTP | Scripted CLI | Scripted CLI |
| Unattended ASK refuses instead of defaulting to allow | Scripted HTTP | Scripted HTTP | Scripted CLI | Scripted CLI |
| Cancellation during a blocked permission: cleanup, one terminal outcome | Scripted HTTP | Scripted HTTP | Scripted CLI | Scripted CLI |
| Concurrent tenant scopes cannot read each other's collection | Scripted HTTP | Scripted HTTP | Scripted CLI | Scripted CLI |
| Strict profile refuses before credential or dispatch | n/a | n/a | Local refusal | Local refusal |
| Contained native execution of a real binary | Unsupported | Unsupported | Unsupported | Unsupported |

The CLI rows above request `HOST_TOOL_PATH_REQUIREMENTS` rather than the
example's `REQUIREMENTS`, and build their runtime with
`native_tool_policy="allow"`. That is not incidental: the strict profile's
`NATIVE_TOOLS_UNAVAILABLE` requirement, which `native_tool_policy="deny"` also
implies, is exactly what refuses both CLI modes before dispatch. A
demonstration of the permission path on a CLI mode therefore has to drop the
containment claim and say so, which is why the two profiles have two names in
`configuration/providers.py`. Nothing else about those calls changes.

## Explicit provider execution (potentially paid)

Configure the relevant `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` securely in your
shell. Do not place its value on the command line. Then explicitly authorize a
call with:

```sh
uv run python -m tests.examples.plugin_execution.main --provider openai_api --execute
uv run python -m tests.examples.plugin_execution.main --provider anthropic_api --execute
```

Use `--model` to select a model available to your account. The expected answer
is `{"record_id":"record-1","status":"ready"}`. The runtime validates its
type; no live provider response is guaranteed by scripted test success.

Native saved-session files can be supplied explicitly with `--auth-file PATH`
for either CLI; the program never reads unrelated native authentication files
implicitly. The credential resolver reads the file only after compatibility
passes. CLI strict execution currently refuses before that read. This example
does not persist refreshed subscription credentials; production hosts must use
the existing `Credential.on_changed` callback to persist rotations safely.

The example requests strict native-tool absence, host-tool transport, structured
output and usage reporting. It does not request an exact hard spending cap.
Metering is separate from spending enforcement: unsupported hard caps must be
requested with `ExecutionRequirement.HARD_SPENDING_CAP` and will refuse before
credentials in all four modes. Unpriced usage stays unpriced. No caller limit is
removed or relaxed by the example.

## Real CLI evidence and unsupported combinations

```sh
AGENTKIT_SKIP_REAL_CLI=0 AGENTKIT_LIVE=0 uv run pytest -q tests/source/integration/execution -o junit_family=xunit1
```

These separate tests use actual CLI processes, synthetic credentials and local
scripted providers. Claude Code 2.1.300 is characterized across fresh, resume,
fork and correction paths, including native auth-file parsing. Granted calls
work, denied/unattended calls have no host effect, and planted native Read/Bash,
hooks and discovered MCP commands cause no effects in those fixtures. Complete
plugin/managed configuration/memory exclusion remains unverified, as does the
classification of its native `StructuredOutput` schema helper. Both version
numbers record what was measured, not what is installed: `claude` now reports
2.1.302 and `codex` upgraded itself to 0.163.0 mid-session, which is why
`MINIMUM_TESTED_CLI_VERSION` is a floor rather than an exact pin. Codex 0.162.1
executes the planted native `view_image` read outside its job directory despite
feature restrictions. That test demonstrates an unsupported boundary, not
successful confinement. The probes fail if their observed protocol changes.

| Requested profile | OpenAI API | Anthropic API | Claude Code | Codex |
| --- | --- | --- | --- | --- |
| Example's strict host-tool provider wire | Supported with trusted callbacks | Supported with trusted callbacks | Unverified; refused | Unsupported; refused |
| Complete MCP discovery/implementation isolation | Unverified; refused | Unverified; refused | Unverified; refused | Unverified; refused |
| Contained native execution | Unsupported | Unsupported | Unsupported | Unsupported |
| Exact hard dollar cap | Unsupported | Unsupported | Unsupported | Unsupported |

API adapters declare only host functions, never provider-side MCP connections.
The broader `MCP_ALLOWLIST` requirement also covers host-side MCP implementations;
AgentKit cannot independently verify arbitrary host callbacks. An API request
hook that can rewrite the wire makes native absence, the granted catalog and
schema support unverified. Ordinary subprocess backends do not establish file,
network, process or credential isolation. Same-user native tools could read a
staged authentication file or environment; a private HOME is not containment.
No real provider authentication or paid call was used to produce local evidence.

The strict CLI sentinel test proves refusal before model dispatch and no effect
on the sentinel. It does not demonstrate an allowed CLI call under the strict
profile, and there is no shipped protected CLI profile to demonstrate that
clean twin. The scripted CLI rows above are not that twin either: they
demonstrate the permission path under the relaxed host-tool profile, with a
scripted binary, which is a claim about the bridge and not about containment.
Required upstream/backend work is described in
[the design report](../../../docs/design/host-tool-execution.md).

## Consumer migration

Compose tools from an explicit authorized registry, attach resource policies via
`ToolSet(permissions=(...), approval=..., observers=(...))`, and call
`CallRuntime.run(..., execution=...)`. A plugin returns declarations/callbacks;
it never selects a provider or acquires execution authority through skill text.
Use `ToolInvocationRequest.scope` from the trusted host context, not a model
argument, to enforce tenant/resource grants. Use the three invocation identities
honestly: provider IDs can be absent and are correlation data only.

Keep complete `SessionHandle` objects, including execution requirements and
native-tool policy, when resuming/forking. Rebuilding a session from a CLI ID
alone drops host restrictions. Reassess after adapter changes; configuration
receipts reject stale assessments but are not isolation attestations. Retain the
existing runtime for calls and accounting and close it with `async with`.
For intentional model-visible recovery instructions, raise public
`ToolExecutionError(safe_message, code=...)`; unexpected exception details are
redacted. Host code is responsible for reviewing the explicit safe message.
