# Execution isolation: design and verification record

## Baseline inspection (2026-10-18)

`CallRuntime` owns policy ceilings, attempts, events, pricing and retries.
`CliAdapter` stages allowlisted credentials, serves host tools through MCP and
constructs the provider cognitions. Both one-shot drivers use `_run_cli_process`;
Claude's separate persistent session transport is not used by `CallRuntime`.
`IsolatedHome` provides private temporary directories and credential cleanup.
It is **not an execution sandbox**. A same-user native tool can read those files.

No CLI version is pinned by this repository or its CI. The binaries inspected
for this record were Claude Code **2.1.300** and Codex **0.162.1**; a later
`--version` probe reports Claude Code **2.1.302** and Codex **0.163.0** — the
installed `codex` upgraded ITSELF mid-session, which is the reason
`MINIMUM_TESTED_CLI_VERSION` is a floor and not an exact match. Neither
measured binary is still installed at the version it was measured at, so every
version below is a record of what was observed and not a claim about what is
present. Older comments describe Claude 2.1.236 and Codex 0.152.1; these are
observations, not version pins.
Argument acceptance tests do not establish authority boundaries.

## Capability matrix as implemented (2026-10-18)

Every status below is a reading of `agentkit/models/providers/cli/capabilities.py`
(the two CLI modes) and `agentkit/models/providers/frontiers/capabilities.py`
(the two API modes), confirmed by calling `CallRuntime.evaluate_execution` once
per mode under a `DENY` ceiling. The matrix is answered in full for every mode
so a host can use it for discovery; the requirements that can refuse a call are
the ones the request declares, plus `native_tools_unavailable` whenever the
policy ceiling is `DENY`.

| Requirement | `claude_code` | `codex` | `anthropic_api` / `openai_api` |
| --- | --- | --- | --- |
| `native_tools_unavailable` | unverified | unsupported | **supported** |
| `host_tools` | supported | supported | supported |
| `mcp_allowlist` | unverified | unverified | unverified |
| `filesystem_read` | unsupported | unsupported | unsupported |
| `filesystem_write` | unsupported | unsupported | unsupported |
| `network` | unsupported | unsupported | unsupported |
| `processes` | unsupported | unsupported | unsupported |
| `credentials` | unsupported | unsupported | unsupported |
| `configuration` | unsupported | unsupported | unsupported |
| `schema` | supported | supported | supported |
| `cancellation` | unsupported | unsupported | unverified |
| `usage` | supported | supported | supported |
| `observed_spending_threshold` | supported | supported | supported |
| `hard_spending_cap` | unsupported | unsupported | unsupported |

Identity travels with the matrix and never moves a status. The CLI modes report
`backend="subprocess"` with the interpreter version and the installed binary's
`--version` reading. The API modes report `backend="https-sse"` with the HTTP
client version, and the binary-version slot carries a per-vendor wire contract —
`anthropic-messages@2023-06-01` or `openai-responses@v1` — because an API
invocation has no binary to version.

The SDK column differs from the CLI one for a structural reason, not a weaker
standard: a CLI invocation is a provider-owned process with provider-native
tools inside it, while an API invocation has no process and no native tools, so
every tool the model can reach is a host function the adapter declared and the
runtime invokes.

- `native_tools_unavailable` is supported because the adapter can enable no
  provider-native or provider-hosted tool. It degrades to **unverified**
  whenever a `request_hook` is installed, because a hook replaces the request
  body after the adapter built it and could reintroduce a provider-hosted tool,
  so the dispatched payload is no longer the payload that was assessed.
- `cancellation` is unverified rather than unsupported: cancelling the awaiting
  task closes the stream locally and reports partial usage, but the cooperative
  token does not interrupt a response in flight and the provider is not proven
  to stop, or to stop billing for, work already started.
- `mcp_allowlist` is unverified because no MCP configuration reaches the
  provider, yet host tools may themselves be served by MCP clients inside this
  process, which agentkit does not verify.
- Every containment boundary, `credentials` included, stays unsupported on the
  API modes for a reason of its own: host tool callbacks run in this process,
  with this process's filesystem, network, descendants, environment and the
  provider credential in reach, and agentkit does not sandbox them. A host
  callback is host authority, not a sandbox.

`observed_spending_threshold` is supported on all four modes, and the evidence
is the runtime's own accounting rather than anything a provider accepts. The runtime hands each attempt a decremented ceiling,
accumulates what came back, and ends the invocation once `max_cost_usd` is
reached, failing closed to `limit_reached` when the spend cannot be known. A
CLI budget flag is not the justification and must not become one: argv
acceptance proves argv threading, and this matrix's first rule is that flags are
not proofs.

Two things about its scope, because conflating them with `hard_spending_cap` —
unsupported on all four modes — is the whole hazard:

- The unit of enforcement differs by mode and in neither case is it mid-flight.
  On the API modes it is a turn; on the CLI modes it is a whole autonomous run.
  Either way the unit that crosses the threshold completes and is billed, and
  only the next dispatch is withheld. No argument either binary accepts turns
  that into a cap.
- The ledger is fed unevenly. `claude_code` reports its own `total_cost_usd` per
  run, while `codex` reports token counts only, so its threshold rides entirely
  on the price table and closes the invocation as soon as a token kind is
  unpriced. Honest, and weaker in practice. Note also that the runtime still
  refuses `max_cost_usd` outright for `codex` (see below), so the supported
  verdict is not reachable through that limit on that mode today.

The CLI reasons behind the columns above, unchanged:

- Claude's empty `--tools` list omits native tools, but no verified denial
  profile exists for this binary and environment, so the verdict is unverified.
  `--bare` excludes discovery and disables saved OAuth; empty setting sources
  alone are not evidence. Argument acceptance does not establish an authority
  boundary.
- Codex keeps its file, patch, plan and other native capabilities, so strict
  denial is unsupported outright. Its read-only sandbox denies writes but not
  reads, which is why it satisfies neither DENY nor credential isolation.
- `mcp_allowlist` is unverified on both, for the mechanism each supplies and
  not for a missing one: Claude passes `--strict-mcp-config` so no other MCP
  configuration is read, and Codex is given an owned `CODEX_HOME` so it ignores
  the user's. Neither is evidence that discovery was actually excluded.
- Process-group cancellation reaches in-group descendants only; a child that
  detaches into another session escapes it, which is why complete descendant
  termination is refused rather than advertised.

Three structural facts hold across the whole matrix:

- There is still **no contained execution backend**. The only shipped backend
  reports `subprocess` and declares no guarantee at all, so every containment
  boundary is unsupported on every mode.
- A report's self-declared `guarantees` — `BackendReport` on the CLI side,
  `TransportReport` on the API side — can only downgrade a verdict to
  unverified. It can never raise one to supported, because that would let
  whoever constructs the report decide which boundaries exist.
- A required unverified requirement refuses, and the refusal is local: a
  pre-dispatch incompatibility makes zero provider attempts, leaves usage
  incomplete and cost unpriced.

## Contract and backend design

Extend the existing runtime, never replace it. Keep NativeToolPolicy.DENY
strict: a containment request cannot override it. Add explicit execution
requirements, typed supported/unsupported/unverified assessments and stable
incompatibility codes. Capability evaluation has no credential resolver or paid
call dependency. Missing evidence refuses before credential-bearing startup.

Use a typed execution backend at the existing spawn seam. Provider adapters
retain argv/configuration construction and event parsing. Backends receive an
explicit invocation (argv, stdin/configuration through the existing pipes,
allowlisted environment, cwd) and own process termination and cleanup. A
remote backend must expose the same ordered byte streams and process outcome.
No scheduler, service, tenant store or secret vault is added to agentkit.

Backends are trusted host code, like credential resolvers and adapters, not
caller assertions. A backend inspection report must identify implementation,
version and binary version and distinguish its observations from guarantees.
The ordinary subprocess implementation cannot attest containment. No generic
`isolated=True` or user-supplied capability override enables a protected profile.
A future supervisor needs a job-bound lease, policy digest, independently
checked identity/version and evidence that restrictions precede secret delivery.
Until such a backend and provider-specific planted-violation tests exist,
contained execution remains visibly unsupported.

Saved subscription authentication with native tools requires separation of the
CLI control plane from its tool executor: credential-free tool processes and
file operations, brokered authenticated provider transport, and immutable
policy-controlled configuration. A container containing both CLI credentials
and tools does not provide that separation. Neither inspected adapter currently
implements it. This needs an adapter/upstream change, not a policy downgrade.

Unknown post-start outcomes must remain non-retryable and usage-incomplete.
A confirmed local rejection is distinct from provider refusal/authentication.
Process-group cancellation is best effort for detached descendants and must
never be advertised as complete job termination. Cleanup removes files, not
secure erasure, and provides no protection against a compromised host.

## Provider references

These current documents explain mechanisms, not proof for an older binary:
[Claude CLI reference](https://code.claude.com/docs/en/cli-reference),
[Codex security](https://developers.openai.com/codex/security).

## Implemented public contract

Import requirements and assessments from `agentkit.models` (also available in
`agentkit.models.contracts`). Pass `execution=` to the existing `ask`/`run`
methods. Call `CallRuntime.evaluate_execution` before choosing a binding. Its
result separates the full matrix from the unmet requirements for this request.
Evaluation never resolves credentials; the default backend reads `--version`
in a temporary home with a minimal environment. A version is metadata, not a
binary integrity attestation. Custom backends currently have unknown binary
versions. No version string or `BackendReport.guarantees` enables isolation.

```python
from pathlib import Path
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models import Binding, ExecutionRequirements, NativeExecution

async def assess_phase(runtime: CallRuntime, binding: Binding, root: Path):
    requirements = ExecutionRequirements(
        native_execution=NativeExecution.CONTAINED,
        read_roots=(root,),
        write_roots=(root,),
        network_destinations=(),
    )
    report = await runtime.evaluate_execution(
        binding, execution=requirements,
    )
    # Both shipped CLI profiles currently refuse these requirements.
    return report.compatible, report.incompatibilities
```

This request does not widen a runtime or adapter policy ceiling. A DENY ceiling
continues to refuse native execution. `CONTAINED` is a separate opt-in request,
not an implementation of DENY. Unspecified authority in a contained request
defaults to empty roots/destinations. Every security boundary is mandatory.
Ordinary calls with no execution requirements preserve WARN/ALLOW semantics;
they are not secure multi-tenant execution profiles.

Both CLI adapters expose schema, host tool protocol and usage metering support.
Neither CLI adapter/backend combination currently establishes filesystem roots,
egress, job process separation, credential secrecy, complete descendant
termination or a hard dollar cap. MCP discovery exclusion is unverified.
Claude native denial is unverified for the complete adapter invocation surface;
Codex native denial is unsupported. Required unverified capabilities fail closed.
The local minimal Claude test below is deliberately narrower than that surface.

`execution_incompatible` is a **local**, pre-dispatch failure. Its safe
`mode_detail` includes `dispatch_state=not_started`, backend/binary versions and
structured incompatibilities (`requirement`, `status`, stable `code`,
`explanation`). Attempts stay zero; usage is incomplete and cost unpriced.
Missing measurements are not reported as measured zeros. Consuming applications
must not treat this as a provider refusal or authentication rejection.

Other new public failure classes are `execution_uncertain` (potentially executed,
non-retryable) and `transport`. A supervisor must raise `ExecutionOutcomeUnknown`
if it loses contact during start after execution might have begun. Stream loss,
malformed completion and worker crashes preserve uncertain outcomes. First-event
timeouts no longer automatically retry. Authentication retry still requires the
existing resolver to return a rotated credential. Existing limit/cancellation
outcomes and ordered event/attempt ownership remain in the single runtime.
CLI diagnostics are no longer copied into public failure details.

`ExecutionRequirement.HARD_SPENDING_CAP` is unsupported independently of usage
metering and of the supported `OBSERVED_SPENDING_THRESHOLD`. CLI budget flags and threshold cancellation are not exact hard dollar
caps. The existing Codex `max_cost_usd` refusal remains; removing it does not
remove the independent DENY incompatibility.

### Execution backend port

Import `ExecutionBackend`, `ExecutionProcess`, `CliInvocation`, `BackendReport`,
`ExecutionOutcomeUnknown` and `SubprocessExecutionBackend` from
`agentkit.models.providers.cli`. Supply `CliAdapter(execution_backend=backend)`;
legacy `spawn=` remains a test/transport seam and cannot be combined with it.
Invocations hide argv/environment in repr, freeze their configuration and reject
unknown process options. The backend returns ordered stdout/stderr readers,
stdin, exit status, termination and cleanup methods. Provider command construction
and parsing remain in the drivers. `terminate_tree` is invoked on cancellation;
`aclose` is invoked on every process exit path. Cleanup failures are observable.
The ordinary backend kills its process group, including in-group descendants,
but cannot kill children that detach into another session or isolate peers.

The port enables replacing local process execution; it does **not yet provide a
remote artifact/credential staging protocol or verified isolation leases**. A
remote supervisor must not infer authorization from an argv path or a temporary
home. Before enabling any protected remote profile, add an explicit input and
configuration manifest, job/policy-bound lease, evidence verifier and provider
acceptance tests. Its evidence must establish namespaces/root mounts, immutable
configuration, egress filtering (including DNS/rebinding and private services),
process visibility, descendant reaping, and authentication separation **before**
secret delivery. Agentkit must verify lease identity, policy binding, versions,
liveness and launch correspondence; the host must trust the attestation issuer
and execution worker. Backend self-declarations alone stay unverified. This
remaining work is intentionally not represented as an implemented guarantee.

### Credential handling and residual risk

Only `.credentials.json` (Claude) or `auth.json` (Codex) is staged at mode 0600.
Environment credentials use the existing provider allowlists. HOME and XDG paths
now point beneath the call-owned home; unrelated host environment, proxy and TLS
configuration are excluded unless explicitly supplied through supported options.
Claude skills are explicitly disabled separately from bare mode. Saved OAuth
continues to require non-bare mode; its full configuration surface is unverified.

The existing `on_changed` callback receives refreshed bytes once, then staged
credentials are removed even if persistence fails. Credentials are excluded from
session bundles. A trusted host can intentionally resolve the same credential
again; agentkit does not claim to enforce tenant credential allocation. Same-user
native tools can read staged auth or environment credentials in an unrestricted
call. Temporary file deletion is not secure erasure. There is no protection
against a compromised host, binary or trusted backend. Detached descendants can
outlive process-group cancellation and are why complete cancellation is refused.

### Evidence and pending acceptance

Focused tests first failed on missing public contracts, generic local failure
classification, inherited HOME/XDG, missing backend cleanup/termination calls,
missing skill exclusion, disconnect classification and deny-by-default roots.
The corresponding tests passed after implementation. Legacy rejection tests now
assert the new failure class **and** retain no-dispatch checks plus zero attempts.
The former first-event retry test now asserts one attempt, no retry, incomplete
usage and unpriced outcome.

Real, credential-free tests in `tests/source/integration/execution` establish:

- Local subprocess cleanup after success/crash, in-group descendant cancellation,
  and credential refresh/removal for both native file names.
- Ordinary subprocess limitations: outside fake credentials can be read/modified
  and private localhost services can be reached. No containment is advertised.
- Claude 2.1.300 against a local scripted provider: empty tool list omits native
  tools and refuses planted Bash/Read operations; planted user/project hooks and
  project MCP do not execute for that bare, skill-disabled invocation.
- Codex 0.162.1 real read-only sandbox: writes fail but outside fake credentials
  remain readable. This independently demonstrates why it cannot satisfy DENY
  or a credential-isolation requirement.

These are not paid-model or provider-network tests. Existing real argument-parser
tests are separately skipped when `AGENTKIT_SKIP_REAL_CLI=1`. No real provider
credentials are needed or used for the new tests. There is no passing protected
execution profile yet. Full provider-specific containment, saved-subscription
configuration exclusion, peer process/credential isolation, denied private
service access and complete detached-descendant termination remain acceptance
work, not implied successes.

### Host integration (including Mirror)

Keep all calls through cognition and the existing `CallRuntime`. Thread the
phase's `ExecutionRequirements` through cognition, and call the credential-free
assessment when checking whether a connection is executable. Render its stable
codes and safe explanations. Preserve DENY on phases that require tools to be
unavailable; a contained phase must deliberately request a different contract
under an appropriate policy ceiling. Keep unsupported profiles unavailable.
Handle `execution_uncertain` as reconciliation work; never retry it as a confirmed
pre-dispatch refusal. Mirror retains tenancy, bindings, workflows and ledger
accounting. Successful login alone says nothing about phase compatibility.

## Installed Codex feature inventory

A credential-free `codex features list` under an empty HOME/CODEX_HOME on
0.162.1 reports shell/unified execution, apps, plugins, remote plugins, hooks,
MCP dependency installation, skill search, image viewing/generation, browser
and computer use, code-mode hosting, multi-agent, sleep and goals as enabled
stable features. Actual tool advertisement also depends on model and session
configuration. This inventory is a lower-bound warning, not a complete native
tool denylist. The legacy `CODEX_NATIVE_TOOLS` tuple (`shell`, `apply_patch`,
`update_plan`) is not an exhaustive inventory for this binary.

The existing WARN path requests shell/unified exec/apps/plugins/skill-MCP-install
and web-search restrictions. It does not prove that all other native features
are unavailable. Its degradation now says restrictions were requested and native
tools remain available. File reads, patching, planning, MCP discovery and the
other features must each be independently verified before any protected profile
can be enabled. Read-only write denial does not establish credential secrecy,
network isolation or native tool absence.

## Final verification (2026-10-18)

No source mutations overlapped the final verification run.

| Command/check | Result |
| --- | --- |
| `AGENTKIT_SKIP_REAL_CLI=1 make check` | Passed |
| Coverage test run, Python 3.13.14 | 1,480 passed, 33 skipped, 0 failed |
| Python 3.12.13 matrix run | 1,480 passed, 33 skipped, 0 failed |
| Python 3.13.14 matrix run | 1,480 passed, 33 skipped, 0 failed |
| Branch coverage gate | 77% reported; threshold unchanged |
| Ruff lint | Passed |
| Strict mypy | Passed, 146 source files |
| Mutation anchor verification | Passed; no source left mutated |
| Full mutation catalogue, run sequentially | 84/84 mutants killed |
| Documented imports | 11 checked, 0 unresolved |
| Separate real-process/CLI execution tests | 10 passed, 0 skipped, 0 failed |
| Formatting of eight new production/test files | Passed |

The 33 main-suite skips comprise 24 existing real CLI argument-parser tests,
six opt-in live/provider tests, two new offline CLI probes (run separately above)
and one optional msgspec test. Real parser tests were deliberately separated
from the credential-free enforcement probes. No paid model or provider-network
validation was run. Successful authentication was not used as execution evidence.

The original no-cost-cap Codex DENY scenario was reproduced through the existing
public runtime: its test initially failed because the failure was generic
`invalid_request`. It now passes with `execution_incompatible`, confirmed
`not_started`, no credential resolution/dispatch, zero attempts, incomplete
usage and unpriced cost. Other focused regressions demonstrated red/green for
HOME/XDG inheritance, skill exclusion, backend cleanup/termination, unknown
startup outcomes, unexplained worker exits, missing binaries and stalled cleanup.

Full execution-isolation acceptance remains **unmet**. In particular, a verified
all-native-tools-off interface is needed for Codex strict DENY; a filesystem
sandbox cannot supply it. Both providers need verified authentication/tool
separation before contained native execution can be enabled. Claude's saved
subscription/configuration profile still needs independent planted-violation
verification. The remote staging/lease/evidence contract is design work, not an
implemented protected backend. No shipped profile proves peer process/credential
isolation, private-service egress denial or complete detached-descendant cleanup.
These combinations remain visibly refused rather than receiving a weaker policy.
