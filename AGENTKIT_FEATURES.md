# Features agentkit still needs to support a CLI-driven coding agent

**Two are open. The eleven that were here before have landed and their specifications are gone from this
file**, because a delivered request kept beside an outstanding one makes the outstanding one harder to find,
and the reasoning that survives delivery belongs next to the code that uses it — which is
`CODING_LOOP_PLAN.md` §7, where each landed surface is written up against the build pipeline that consumes
it. For the record, and once: F1 `serve_registry` · F2 `ApprovalServer` (all but the bullet below) · F3
`hook_settings` · F4 `as_cli_agents` · F5 `FakeClaudeCli` + the `spawn=` seam · M1 `MemoryItem.id` and
`CompositeMemory` dedupe · M2 `ReadOnlyMemory` · M3 `GroundingSource` · S1 `compare_and_set` / `increment` /
`scan` on the port and all four backends · C1 `Workflow.map` · C2 `attempt_until_stuck`. Pinned at
`9f370b8`, taken as `arc-agentkit[mcp,fast]`.

Both of the two below are about the same thing from opposite ends: **an agentkit MCP server is the callee,
and being the callee is where its guarantees are thinnest.** One cannot say who called it. The other cannot
say who was allowed to.

Each states the failure it prevents, the surface, what it has to get right, and the tests that would make it
real — the shape the eleven were written in, because it is the shape that got them built.

---

## A1 · Record an approval as a decision, not as a count

**Priority: blocking for any build that has to be audited. Not blocking Phase 1.**

This is the one bullet of F2 that did not land. The other three did: the wait is bounded by `timeout_s`, a
denial reaches the model as a reason it can act on, and — new in this version — `autonomy` is read from the
run and honoured through the same `should_gate` every other pattern uses, with a gating tier and no `asker`
refused at construction rather than denying every prompt at run time.

### The gap

`ApprovalServer` exposes `prompts_seen`, which is an integer. Internally that is one `self._seen += 1` per
prompt. There is no record of **which** call was approved, **who** approved it, **when**, or **why** — and
nothing is emitted to the run's observer either, so a run cannot reconstruct its own approvals afterwards
from any source.

The asymmetry is what makes this worth fixing rather than working around, because the sibling in the same
package got it right. `HookSettings.decisions` hands back every decision in order, each carrying the tool,
the verdict and the reason, and emits `gate.check` on `ctx.emit` as it goes. Two servers, both answering
"may this run?", and only one of them can say afterwards what it answered.

**What it costs in practice.** A permission prompt is the single most consequential event in a run — it is
the point at which a person took responsibility for something the machine would not do on its own. A count
of them is the one summary that cannot answer any question worth asking: not *what did we allow*, not *did
anybody actually look*, not *which approval preceded the write we are now looking at*. A build that has to
answer those keeps a parallel record beside the `Asker` it supplies, which is a second implementation of
something the server already has in hand and drops.

### The surface

```python
@dataclass(frozen=True)
class ApprovalDecision:
    tool: str                    # the qualified name the CLI asked about
    arguments: Mapping[str, Any] # what it wanted to do — the half a tool name cannot carry
    allowed: bool
    reason: str                  # why, in a sentence; the same text the model receives on a deny
    source: Literal["asker", "auto_allow", "autonomy", "timeout", "error"]
    at: str                      # ISO-8601, from the run's clock and never from a local one
    asked: bool                  # whether a human was actually reached
```

```python
async with ApprovalServer(asker=my_asker, autonomy=ctx.autonomy) as approvals:
    ...
approvals.decisions   # tuple[ApprovalDecision, ...], oldest first
```

Mirroring `HookSettings.decisions` deliberately: same name, same ordering, same "oldest first", so a caller
consuming both is consuming one shape. And emitted as it happens, best-effort, on the same `gate.check`
channel — a decision that reaches the observer is a decision a live surface can render, which is the
difference between an audit trail and a post-mortem.

### What it has to get right

- **`source` is a closed set, and it is the whole point.** *Allowed because a person said so* and *allowed
  because the autonomy tier does not gate* are the two facts an auditor most needs apart, and they are
  indistinguishable in a boolean. A `timeout` that degraded to deny must not read as a human refusal, and an
  `error` must not read as a policy decision.
- **`asked` is not the same as `source == "asker"`.** A prompt can reach a human and time out. Whether
  somebody was interrupted is a fact about the run's cost to a person, and it is not recoverable from the
  verdict.
- **`arguments` travels.** A tool name says `Write`; a decision needs to have been about `Write` of a
  particular path. Truncation is fine and a bound is expected — dropping the field is not, because an
  approval that cannot name its subject cannot be checked against what the run then did.
- **Recording never fails the decision.** Same contract `HookSettings._emit` keeps: a refusal that happened
  matters more than a record of it that did not, so the emit is best-effort and the in-memory append comes
  first.
- **Nothing is redacted here, and nothing is logged here.** The server holds the record; what a caller does
  with arguments that may carry a secret is the caller's policy, and a library that half-redacted would
  give an audit trail nobody can trust to be complete.

### Tests that make it real

- a prompt a human allows records `source="asker"`, `asked=True`, and the arguments it was about
- an `auto_allow` hit records `source="auto_allow"` and `asked=False` — and no `Asker` is invoked at all
- under `autonomy="auto"` a decision is still recorded, with `source="autonomy"`
- a prompt that expires records `source="timeout"`, `allowed=False`, and is distinguishable from a human
  denial by nothing other than `source` — which is why `source` has to exist
- an `Asker` that raises records `source="error"` and denies; the session survives and the next prompt is
  answered
- `decisions` is ordered oldest-first and its length equals `prompts_seen`, so the count that exists today
  cannot silently disagree with the record that replaces it
- a `ctx` with no `emit` records normally and raises nothing

---

## A2 · An MCP server anything on the host can call is not a fence

**Priority: blocking for any worker that runs untrusted code, which for a cloning engine is every worker.**

### The gap

`_transport.py` carries this warning, and it is accurate:

> Everything here binds `127.0.0.1` with **no authentication**: anything able to reach the port can call the
> tools behind it. Loopback-only is the containment.

Both MCP servers in the package sit on it — `ApprovalServer` and `serve_registry`. There is no token, no
header check, no peer credential, and no Unix-socket option. The `host` field exists and no caller
overrides it, which is the right default and is not a control.

**Loopback is a containment argument that holds exactly as long as nothing untrusted shares the host.** That
is a reasonable assumption for a developer running an agent on a laptop. It is the wrong assumption for the
case this integration was built for, and the reason is specific rather than general: the whole point of
`ClaudeCliCognition` is that the CLI runs `Bash`. A session with `Bash` executes whatever the task requires
— a build, a package install, a test suite, a script out of a repository the agent was pointed at — inside
the same network namespace as the server that holds the agent's own tools. The trust boundary the loopback
argument assumes is precisely the boundary the tool set is designed to cross.

**Why this is worse for a served registry than for approvals.** An approval server answers prompts; the
worst a caller can do is approve things. A served `ToolRegistry` is whatever the application put in it. In
this repository that includes `close_slice` — the call that records a requirement as landed. A verdict that
can be reached by anything sharing the namespace is a verdict no longer produced only by the deterministic
runner that was supposed to produce it, and *that* is the property the whole build gate rests on. It does
not have to be attacked to be worthless; it only has to be reachable.

**The alternative transport does not help.** `transport="stdio"` exists, and the CLI *spawns* a stdio
server — so the tools are rebuilt in a fresh process, and the registry's closures and the live `ctx` do not
survive the boundary. `serve_registry` says so at the point of refusal, correctly. Any application whose
tools close over live run state — which is every application that has a reason to serve its own tools rather
than ship them — is therefore forced onto the unauthenticated path. There is no configuration that avoids
it.

**agentkit has already written the better answer once.** `hook_settings` puts an `AF_UNIX` socket in a
`0700` directory and makes the filesystem the authorisation check, and its docstring names the comparison
itself: *"`ApprovalServer` next door binds loopback TCP with no authentication and documents loopback as the
containment; this is the same shape with a better fence."* The fence exists in the package. It has not been
applied to the two servers that need it most.

### The surface

Preferred, because it needs nothing invented and matches the mechanism already in the package:

```python
spec = serve_registry(registry, name="engine", ctx=ctx, transport="unix")
# binds <0700 dir>/engine.sock; the config document is
# {"mcpServers": {"engine": {"type": "http", "url": "http://localhost/mcp",
#                            "unixSocket": "/…/engine.sock"}}}
```

`ApprovalServer(transport="unix")` for the same reason and by the same route.

If the CLI's `--mcp-config` cannot address a Unix socket — which is the one thing that decides between these
two, and is a question about the CLI rather than about agentkit — then the fallback is a bearer token on the
existing loopback listener:

```python
spec = serve_registry(registry, name="engine", ctx=ctx)   # generates a token
# {"mcpServers": {"engine": {"type": "http", "url": …,
#                            "headers": {"Authorization": "Bearer <token>"}}}}
```

A token is strictly weaker: it is readable by anything that can read the config file, which on a shared host
is a smaller set than "anything that can open a socket" but not a small one. It is worth having anyway,
because it turns *anything on the host* into *anything that can read this process's temp directory*, and the
config file is already `0600` for exactly that reason.

### What it has to get right

- **Unauthenticated must stop being the default, or stop being silent.** Either is defensible; the current
  state is neither. A warning in a module docstring is read by the person maintaining the module, not by the
  person wiring a worker.
- **The containment has to be the OS, not a check the server performs.** A `0700` directory is enforced by
  the kernel on every access; a token comparison is enforced by code that has to be right on every path,
  including the error paths. The hook module chose the first and explained why.
- **The failure mode is refusal, never fallback.** A server asked for an authenticated transport that cannot
  provide one must raise at wiring time. Silently degrading to unauthenticated loopback would be the exact
  shape of bug the package refuses everywhere else — a guard that looks wired and enforces nothing.
- **The token, if it exists, is generated and never accepted from a caller.** A caller-supplied token is a
  token that will be a constant in somebody's config file.
- **`cli_kwargs()` keeps carrying everything needed.** Whatever the transport, a caller should not be
  assembling headers or socket paths by hand — that is the same argument that put the config document
  behind a function in the first place.

### Tests that make it real

- a served registry on the authenticated transport answers a properly-addressed call
- the same server refuses a call that does not carry the credential, and the refusal is a transport-level
  rejection rather than a tool error — a bad *caller* is not a bad *call*, and reflecting it to the model
  would teach it to work around a security boundary
- the socket path is inside a `0700` directory, asserted on the mode, not on the path shape
- a second process running as another user cannot open it
- requesting the authenticated transport where it cannot be provided raises at wiring time, and the error
  names what to do instead
- `stop()` removes the socket and the config file together, so a stale config can never outlive the listener
  it names
- the existing loopback behaviour still works and is now something a caller has to ask for by name

---

## What Mirror does about these two in the meantime

Neither blocks the build pipeline, and it is worth writing down why so that nobody treats them as gating.

**A1 does not block because Phase 1 asks nobody anything.** The fence refuses on the path alone and never
prompts, and the refusals it produces are already recorded — `HookSettings.decisions` is the source the
review surface's refusal panel consumes. Human approval arrives in Phase 3, and the first build that needs
an approval trail is the first build that needs A1.

**A2 is mitigated rather than solved, and the mitigation should be named as one.** A worker runs one build,
in its own container, and the served registry is reachable only from inside it. That reduces the exposure to
*the agent's own `Bash` and whatever the clone's build steps execute* — which is a smaller set than "the
host" and is still the untrusted set. So the honest statement is that the tool port is inside the blast
radius of the code the agent runs, and the compensating control is that a slice's worktree is thrown away
and its verdict re-derived on the integration tree, where a forged `close_slice` does not survive. That is a
real control and it is not the same as the port being closed.
