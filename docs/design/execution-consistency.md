# Execution consistency: implemented scope and open acceptance

This work preserves `CallRuntime` as the model loop and attempt/accounting
owner. It **does not make four protected execution modes available**. Both API
modes support host-function execution without configuring provider-native tools.
Claude Code strict native-tool absence remains unverified, Codex strict absence
remains unsupported, and no shipped backend proves contained native execution.

## Ownership and flow

`AgentRunner` forwards `execution=` through minds, verification slices and
delegation. `CallRuntime.ask` delegates to `run`. Before credential resolution,
the runtime compiles the schema and uses the adapter's credential-free
execution evaluator. It records effective policy and implementation identities
in an `execution_policy` transport event before the first attempt. Unmet
requirements return `execution_incompatible`, zero attempts, incomplete usage
and unpriced cost. Schema rejection retains its separate local failure code.

The runtime owns attempt start/finish, correction, retry, cancellation monitoring,
usage normalization, pricing and settlement. Adapters own protocol mapping.
API tool calls return to the common runtime loop. CLI tools call the same
`ToolSet` through the existing authenticated MCP bridge while the provider CLI
owns its internal conversation. There is no second host agent loop or spending
service. The backend owns process launch, ordered pipes and process cleanup.
An attempt counts a runtime transport invocation. A CLI invocation may make
multiple internal provider requests; their individual retries/attempt identities
are not exposed by this public counter. Exact internal request accounting remains
a vendor transport visibility limitation, not a measured per-request guarantee.

`SessionHandle.execution` and `native_tool_policy` preserve the host ceiling.
Omitting requirements on resume/fork inherits them. Additional requests union
requirements and intersect explicit roots/destinations; a broader request cannot
expand the original authority. This pure intersection does not resolve symlinks
or establish confinement. No shipped profile can satisfy those filesystem
requirements. In-memory continuation, correction and schema fallback use the
same invocation requirements. Hosts serializing sessions must retain these new
fields; reconstructing only a native CLI session ID loses the host's ceiling.

## Capability matrix

S = supported in the stated scope, V = unverified, U = unsupported. A required
V or U refuses before credentials. A compatible report is a protocol/policy
preflight, not proof of provider authentication, model availability or acceptance
of a particular schema. Compilation/settings checks still run before dispatch.

| Requirement | OpenAI API | Anthropic API | Claude Code + subprocess | Codex + subprocess |
| --- | --- | --- | --- | --- |
| Native tools unavailable | S | S | V | U |
| Host ToolSet protocol | S | S | S | S |
| MCP allowlist over complete discovery surface | U | U | V | V |
| Filesystem read roots | U | U | U | U |
| Filesystem write roots | U | U | U | U |
| Network destinations | U | U | U | U |
| Job process authority / peer observation | U | U | U | U |
| Credentials inaccessible to tool executors | U | U | U | U |
| Complete configuration exclusion | U | U | U | U |
| Schema protocol + canonical validation | S | S | S | S |
| Complete descendant cancellation | U | U | U | U |
| Usage metering / missing-measurement distinction | S | S | S | S |
| Observed spending threshold between turns | S | S | U | U |
| Exact hard spending cap | U | U | U | U |

API scope is explicit: the adapter emits only the granted host function catalog.
Host callbacks are trusted code, not sandboxed processes. Their filesystem,
network, child processes and credential access are not constrained by Agentkit.
An unrestricted API `request_hook` can rewrite the wire catalog/schema, so native
absence, host-tool catalog and schema support become V when it is configured.
This is `request_hook_authority_unverified`; strict requests refuse. No hook is
run merely to assess capabilities.

Adapters without an execution evaluator cannot satisfy implicit DENY either.
Their fallback report includes the requested native absence instead of returning
an empty compatible report. Legacy mock tests now require zero dispatch under
DENY; a separate WARN clean twin retains forwarding/pricing/slice coverage.

The CLI S entries describe implemented protocol support, not verified
confinement. The installed binaries tested here are Claude Code 2.1.300 and
Codex 0.162.1. These are observations, not repository pins or integrity evidence.
Backend declarations, versions, executable names and accepted flags cannot
promote V/U to S. In particular a backend declaration cannot turn Codex's
unsupported native absence into an unverified capability awaiting promotion.
Custom adapters/backends are trusted host implementations; no caller boolean
attests isolation. Unsupported backend inspection is a local refusal.

## Host tool authority and identities

The host supplies a granted registry. `AgentRunner` filters its declarations
through `ToolGrant`. `ToolSet` checks current approval before effects and now
validates every custom tool against its declared JSON Schema, independently of
Python signature coercion or MCP client validation. Arguments are snapshotted
and frozen. Unknown tools, invalid inputs and unapproved tools cause no effect.
Deferred discovery does not replace current registry/approval checks. Tools
continue to enforce their own resource grants; ToolSet is not a resource sandbox.
Results retain bounds, cancellation and the existing host result policy.

`agentkit.tools.ToolInvocation` and `current_tool_invocation()` expose three
distinct identities. Handlers receive the same value at `ctx.tool_invocation`:

- `host_invocation_id`: generated before execution, unique to this host call;
- `provider_tool_call_id`: actual provider ID when supplied, otherwise `None`;
- `transport_request_id`: MCP JSON-RPC request ID, otherwise `None`.

Actual processes establish Claude's `_meta["claudecode/toolUseId"]` and Codex's
`_meta["callId"]` forwarding on the tested versions. Neither is authorization.
Missing, malformed or conflicting provider identities stay unknown. Client
metadata never overrides the host scope/cancellation context. SDK handlers get
the provider call ID directly. Task-local bindings reset after completion and
isolate parallel callbacks. CLI portable tool messages retain all three IDs in
metadata. Their transcript pairing ID uses the provider ID if present; an honest
host pairing ID remains labeled as such when the provider association is absent.
CLI streaming item IDs remain transport identifiers and must not be assumed to
equal a provider call ID; use the portable message association for reconciliation.
Resumed CLI callbacks now correlate to the current call, not the previous run ID.

## Backend and credential boundary

`CliInvocation` carries `mode`, `call_id`, `attempt` (zero means unspecified),
`execution` and `native_tool_policy` alongside its immutable argv/options. The
runtime attempt ordinal is supplied unchanged. Its repr excludes argv,
environment and requirement roots. The ordinary subprocess backend applies the
same compatibility evaluator and independently rejects protected invocations
made directly through this port. This additional check does not confer evidence.

The ordinary backend supplies explicit child environment/configuration, ordered
pipes, exit status, bounded process-group termination and cleanup. It cannot
restrict same-user filesystem access, symlink escapes, network/private services,
peer process observation, or descendants detaching into another process group.
It advertises none of those guarantees. The real-process tests deliberately
demonstrate some prohibited operations succeeding under this unprotected backend
to prevent accidental promotion of its guarantees.

No verified remote backend is shipped. `ExecutionBackend` is still a process
port, not a remote staging/attestation protocol. A supervisor needs explicit
artifact/configuration inputs, a job/policy-bound lease and independent evidence
verification before any secret delivery. Agentkit must check launch/lease
correspondence, implementation identity, policy binding and liveness. Its trusted
worker must establish root mounts/symlink handling, namespace/process visibility,
descendant reaping, immutable discovery configuration and destination filtering
including redirects, DNS rebinding and private services. Self-reported
`BackendReport.guarantees` does not implement those checks. Protected remote
profiles remain unavailable until this exists and provider-specific planted
violations pass.

Existing credential handling retains private temporary HOME/XDG directories,
allowlisted environment variables and native auth files, refreshed-byte
`Credential.on_changed` callbacks, and removal of staged files even if persistence
fails. Credentials remain excluded from session bundles and public diagnostics.
Neither API host callbacks nor same-user CLI native tools are thereby prevented
from reading host credentials. In particular CLI control-plane authentication
and model-controlled native execution need separate authority (a credential-free
executor or authenticated transport broker). A shared container/private HOME is
insufficient. Tenant allocation/reuse is the host's responsibility. File removal
is not secure erasure; a compromised host/binary/backend remains outside the
claimed protections.

## Accounting and outcomes

Both CLI adapters now use captured `raw_usage` rather than inventing zero
counters from `ProviderUsage` defaults. API completion also requires measured
input/output counters to label token usage complete. Captured cache-read,
5-minute cache-write and 1-hour cache-write counts remain disjoint. Reported cost
can remain known when token completeness is false; missing cost is unpriced.

API stream interruption/EOF without confirmed completion is now
`execution_uncertain`, non-retryable and usage-incomplete, preserving captured
partial counts. This matches CLI worker/disconnect semantics. Known provider
HTTP/in-band rejections retain their own classifications; local incompatibility
is not a provider refusal. Runtime cancellation closes API streams, cancels host
callbacks and terminates CLI process groups. It does not promise to stop arbitrary
host-created or detached descendant processes. Cleanup failure is observable.

Usage metering, stopping after observed cost and exact spending caps remain
separate. `HARD_SPENDING_CAP` refuses in all four modes. Existing call limits
are never stripped; the existing Codex `max_cost_usd` refusal remains independent
of its native-tool incompatibility. No cancellation mechanism promises an exact
dollar cap. The host still owns external ledger settlement and reconciliation.
Codex's existing cost-limit refusal now also reports local/not-started metadata
and incomplete, unpriced accounting instead of a measured zero for an unrun call.

## Minimal host integration for all four modes

Keep credentials/model choices in host resolvers. This example deliberately
requests strict host-tool execution for every binding; the two CLI reports
currently refuse. It never chooses a different provider or relaxes requirements.

```python
from pydantic import BaseModel
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models import Binding, ExecutionRequirement, ExecutionRequirements
from agentkit.tools import ToolRegistry, tool
from agentkit.tools.execution.policy import ToolSet

class Status(BaseModel):
    status: str

@tool(side_effecting=False)
def lookup(key: str, ctx) -> str:
    """Read a record under the host's granted resource policy."""
    # Validate the real resource grant using ctx.scope in the host implementation.
    return key

async def execute_bindings(bindings: list[Binding]):
    # Supply one binding each for openai_api, anthropic_api, claude_code, codex.
    # Binding(mode, model, CredentialRef(host_resolver)) uses the host's saved auth.
    required = ExecutionRequirements(required=frozenset({
        ExecutionRequirement.NATIVE_TOOLS_UNAVAILABLE,
        ExecutionRequirement.HOST_TOOLS,
        ExecutionRequirement.SCHEMA,
        ExecutionRequirement.USAGE,
    }))
    tools = ToolSet(ToolRegistry.from_tools([lookup]))
    results = {}
    async with CallRuntime(native_tool_policy="deny") as runtime:
        for binding in bindings:
            report = await runtime.evaluate_execution(binding, execution=required)
            if not report.compatible:
                results[binding.mode] = report
                continue
            results[binding.mode] = await runtime.run(
                binding, "Look up the granted record and return its status.", tools,
                output=Status, execution=required, call_id=f"status/{binding.mode}",
            )
    return results
```

Mirror continues to call through cognition. Thread requirements into its existing
runtime/runner invocation and retain complete session handles. Render stable
incompatibility codes/explanations when checking a connection for a phase.
Successful authentication does not establish phase compatibility. Preserve
uncertain attempts for reconciliation. Tenancy, bindings, phase workflows and
ledger accounting stay in Mirror. No migration to another model loop is needed.

## Evidence, limits and vendor requests

Focused regressions were observed red before implementation for API assessment,
schema validation before custom effects, session retention, API stream uncertainty,
identity propagation, runner forwarding, missing CLI usage, API hook expansion,
policy event recording and backend request forwarding/refusal. Eight planted
source mutants additionally protect the principal invariants.

Real local-provider probes exercise the actual Claude and Codex adapters, CLI
processes and authenticated MCP ToolSet bridge, structured output and actual
provider call identities. Claude tests valid planted Read/Bash requests and a
project hook under bare API-key and non-bare fake-staged-home branches. The fake
staged file does **not** prove real saved-session authentication or credential
refresh with the vendor. The older bare-mode probe now streams planted tool
arguments correctly rather than placing them only in the initial block.

Codex's real WARN invocation advertises MCP resource discovery, request-user-input,
image viewing, a multi-agent namespace and goal tools after the requested feature
restrictions. Its host tools use a provider namespace on the actual wire. The
read-only sandbox separately blocks a write while allowing an outside fake auth
file read. None of these is native-tool absence or credential isolation.

Remaining acceptance work is explicit: real saved-subscription authentication,
complete user/project/config/plugin/skill/hook/memory discovery exclusion across
resume/fork, peer process/file/credential isolation, prohibited egress including
private services, and detached descendant termination. No passing protection is
claimed for these. No real provider credentials or live paid calls were used.

Concrete upstream/backend requests:

1. Codex: a documented, complete, immutable native-tools-unavailable profile
   retaining only explicitly supplied host MCP tools, including on resume/fork.
   Feature switches, read-only sandboxes and denial-returning tools are insufficient.
2. Claude: verify a complete customization-disabled invocation that retains saved
   authentication and explicit granted MCP, covering filesystem/memory features,
   structured-output helpers and inherited session configuration. The complete
   adapter surface is not proven by the local probe.
3. Both: separate authenticated control-plane access from credential-free native
   execution before contained execution can be supported, with independently
   verified filesystem/network/process/configuration/cleanup evidence.

Official references were checked for mechanisms, not used to attest binaries:
[Claude CLI reference](https://code.claude.com/docs/en/cli-reference),
[Claude authentication](https://code.claude.com/docs/en/authentication),
[Codex provider configuration](https://learn.chatgpt.com/docs/config-file/config-reference).
In particular current Claude documentation separates bare mode from tool
availability and describes customization/authentication differences. Current
Codex documentation describes custom provider transport configuration, not
native-tool absence. Neither substitutes for tests of installed versions.

## Final verification

### Follow-up: installed CLI validation

The installed Claude Code 2.1.300 and Codex 0.162.1 were exercised again in
temporary directories, through the real adapters and MCP bridge against local
scripted providers. The execution suite passed **24 tests, zero skipped and zero
failed**. Only synthetic credentials were supplied; no vendor authentication or
paid model request was made. These tests characterize the existing WARN path;
they do not weaken or bypass the production DENY preflight.

Claude's twelve cases cover fresh calls, resume, fork and canonical-validation
correction for API-key, staged-home API-key and synthetic saved-session profiles.
The local provider checks the expected authentication header without retaining
or printing it. This proves native authentication-file parsing and selection,
not subscription validity, vendor token refresh or vendor acceptance. Granted
host calls and typed answers work in each case. Planted Read/Bash requests do not
read the outside fake credential or create the outside marker. Planted project
and user hooks and discovered MCP commands do not run. Project instructions and
skill canaries are absent from the captured requests. The observed catalog
contains only the granted host tool and the `StructuredOutput` schema helper.
This helper must be accounted for explicitly when defining native-tool absence;
it is not a granted host function. These cases do not cover every plugin,
managed configuration or memory mechanism, or attest a general execution profile.

Codex's two cases are a clean host-tool/schema exchange and a planted native
`view_image` call. Both return the same typed answer. In the planted case, the
CLI reads a valid synthetic PNG outside its job directory and sends its image
data to the local provider, despite the current feature restrictions and
read-only policy. The clean twin sends no image. This establishes actual native
execution, beyond catalog advertisement: the tested profile cannot satisfy
`NATIVE_TOOLS_UNAVAILABLE`. It also demonstrates why a read-only sandbox is not a
filesystem read boundary. No credential material was used for this canary.

The labels therefore retain different meanings: Claude is **unverified** because
Agentkit has not established a complete, binding admission profile across the
adapter surface; Codex is **unsupported** because the tested restrictions permit
native execution. Identical runtime semantics require the same requirement to
be enforced, not identical capability claims for different mechanisms. No
production capability status changed in this follow-up.

Reproduce the credential-free process probes with the installed binaries:

```sh
AGENTKIT_SKIP_REAL_CLI=0 AGENTKIT_LIVE=0 uv run pytest -q tests/source/integration/execution -o junit_family=xunit1
```

The verification table below records the earlier implementation gate, before
these additional CLI cases. The follow-up gate is recorded separately so its
expanded skip count is not confused with that historical result.

| Follow-up check | Result |
| --- | --- |
| `AGENTKIT_SKIP_REAL_CLI=1 AGENTKIT_LIVE=0 make check` | Passed |
| Coverage run and Python 3.12/3.13 matrix, each | 1,795 passed, 47 skipped, 0 failed |
| Coverage | 81%; existing 80% threshold unchanged |
| Lint, strict types, mutation anchors, documented imports | Passed; 148 typed files, 21 documented imports |
| Formatting of the two changed CLI test files | Passed |
| Separate real-process execution suite | 24 passed, 0 skipped, 0 failed |

The 47 skips comprise 24 argument-parser cases, 16 offline CLI cases exercised
separately, six opt-in live-provider cases and one optional msgspec case. The
full source mutation catalogue was not rerun for this test/documentation-only
follow-up; the 110/110 result below belongs to the earlier implementation gate.

### Earlier implementation gate

The final source was held fixed during the gate. No limits, coverage thresholds
or native-tool guards were relaxed. Repository Git metadata and publication were
untouched; no new release artifact/version was created.

| Check | Final result |
| --- | --- |
| `AGENTKIT_SKIP_REAL_CLI=1 AGENTKIT_LIVE=0 make check` | Passed |
| Coverage run, Python 3.13.14 | 1,795 passed, 36 skipped, 0 failed |
| Python 3.12.13 matrix | 1,795 passed, 36 skipped, 0 failed |
| Python 3.13.14 matrix | 1,795 passed, 36 skipped, 0 failed |
| Branch coverage | 81% reported; existing 80% threshold unchanged |
| Ruff lint | Passed |
| Formatting check for seven new/formatted files | Passed |
| Strict mypy | Passed, 148 source files |
| Mutation anchor verification | Passed |
| Full mutation catalogue, isolated source and virtual environment | 110/110 killed; 0 survivors/unrunnable |
| Mutation restoration | SHA-256 hashes matched for all 148 source files |
| Documented imports | 21 checked, 0 unresolved |
| Focused execution consistency regressions | 46 passed, 0 skipped, 0 failed |
| Separate actual process/CLI acceptance probes | 13 passed, 0 skipped, 0 failed |

The 36 main-suite skips are 24 real CLI argument-parser cases, six opt-in live
provider cases, five offline CLI probes run separately above, and one optional
msgspec case. The real CLI probes use fake credentials and local scripted
providers, not a paid model or vendor authentication endpoint. Argument-parser
checks and real saved-subscription/provider-network validation were not run.
Skipped checks are not enforcement evidence.

An intermediate matrix run reported 1 failed, 1,794 passed and 36 skipped on
the newly added Codex cost-refusal accounting regression. The final source fixes
that regression and the complete gate above passes. Earlier intended red tests
included four custom-schema violation cases, four API capability/dispatch
cases, three SDK/MCP identity cases, six runner/missing-usage cases, two unsafe
API-hook cases, four missing policy-record cases, four backend contract cases,
and three unknown-adapter DENY failures (the Codex twin already refused). The
actual Codex metadata probe separately failed with `None` instead of the supplied
`call_local`, then passed after preserving `_meta.callId`. Test-fixture errors
were corrected before recording these behavioral failures.

The full mutation catalogue was first run in a separate source copy, then
repeated with its own virtual environment so the Python matrix could not change
its interpreter/dependencies. Both runs killed 110/110 mutants. No source
mutations ran in the working repository. All eight newly added execution mutants
were killed by the focused regressions.

The success of these protocol and lifecycle checks does not complete protected
execution acceptance: the unsupported/unverified matrix and vendor/backend
requests above remain binding.
