# Changelog

All notable changes to `arc-agentkit` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Two batches of work in this cycle: five gaps reported from production use, and
a follow-up sweep for other major issues. Everything is additive except the
concurrency-bound change called out below.

### Fixed — `security()` was uncallable, and `Prompt.render()` ignored its inputs

Two documented entry points that failed on first contact.

- **`from agentkit.middlewares import security; security()` raised
  `TypeError: 'module' object is not callable`.** `middlewares/__init__.py`
  imported the factory from `guard.py`, then a later line's
  `from ...security import Audit, Egress` bound the SUBMODULE onto the package
  over the top of it. It was in `__all__` and in the canonical middleware chain
  in two docs pages. `security.py` is renamed to `egress_audit.py` — which is
  also what it contains — rather than patching the binding, because reordering
  imports is undone by the formatter and an explicit re-bind is one more thing
  to forget.

  This was the THIRD instance of a module shadowing a same-named callable in
  this codebase (`registry.py`/`registry()`, `elicit.py`/`elicit()` were the
  first two). `tests/meta/test_public_surface.py` now walks every public
  package and fails on any recurrence. Writing it taught the precise rule:
  `tracing`, `memoize`, `meter`, `compaction` and `output_coerce` are the same
  collision and are all SAFE, because their factory lives in the same-named
  file so the function binds last within one statement. The dangerous shape is
  a name that collides with a submodule *and* is defined somewhere else.

- **`Prompt.render()` took no arguments and returned `self.template.strip()`,**
  so `inputs=` was decorative and placeholders shipped to the model verbatim:
  `Prompt(template="Hello {name}", inputs=("name",)).render()` → `'Hello {name}'`.
  That is worse than a crash — a plausible-looking prompt that quietly
  describes the wrong task. It now substitutes, and raises `ValueError` on a
  missing or unexpected input.

  Substitution is a literal replacement of the declared names, not
  `str.format`: system prompts are full of braces that are not placeholders,
  and `format` would raise on a JSON Schema or eat a user's escaping. A prompt
  declaring no `inputs` renders exactly as before.

### Added — `ClaudeCliSession.interrupt()`: stop a turn, keep the conversation

The session could send turns but not stop one mid-flight, which left a chat UI
with only cancellation — and cancelling terminates the process, because no
protocol message retracts a half-finished turn, so the conversation ends with
it. The CLI has its own verb for the intent, over the control protocol on
stdin, and it keeps the session alive: verified against the binary, an
interrupted turn is followed by a normal one in the same process, which then
exits 0.

The interrupted turn still yields exactly one terminal `final`, with
`stop_reason="interrupted"` — `terminated` in the closed taxonomy, since
somebody stopped it deliberately — and `partial=True`, because whatever text
arrived is a fragment. The CLI ends an interrupted turn with
`error_during_execution`, the same subtype it uses for a genuine execution
failure, so the payload alone cannot tell them apart; the session stamps the
reason from state it holds rather than inferring it from an ambiguous field.

`InterruptReceipt.still_queued` names messages the CLI will run anyway — the
difference between "the agent stopped" and "the agent stopped and has three
more things to do". `interrupt(cancel_queued=True)` drops those too, and is
refused up front when the CLI does not advertise `interrupt_cancel_queued_v1`
rather than being silently ignored.

Also added `session.capabilities` / `session.supports(...)`, folded live from
`system/init` — `interrupt()` feature-detects against them mid-turn, so
capabilities that only landed at the terminal event were too late to act on.

**Call it from a separate task.** The CLI acknowledges on the same stdout the
`turn()` loop is draining, so awaiting `interrupt()` inline in that loop
suspends the only reader. That degrades rather than deadlocks: after
`ack_timeout_s` the receipt returns `delivered=True` with an `error` explaining
it, and the interrupt still takes effect because the write already happened.
Getting it wrong costs a field on a receipt, not a hung application.

### Added — `ApprovalServer`: the CLI's permission prompts reach an `Asker`

`ClaudeCliCognition` delegates the whole loop to the CLI, which owns its own
permissions. That left a service two options, both bad: `bypassPermissions`
(the agent may do anything, unattended) or `dontAsk` (anything not pre-approved
is denied outright and the run fails). agentkit already had the missing middle
— `Asker`, the injected human transport behind its own HITL path — but nothing
connected the two.

The CLI's seam is `--permission-prompt-tool`, which names an MCP tool it calls
instead of prompting a terminal. `ApprovalServer` (in
`agentkit.integrations.mcp`, needs the `mcp` extra) is that tool: each prompt
becomes an `Elicitation` carrying the tool name and its **arguments**, the
application's `Asker` answers, and the `Decision` maps back onto the CLI's
allow/deny shape. `modify` becomes an approve-with-changes, so a reviewer can
redirect a write into a sandbox without derailing the run.

```python
async with ApprovalServer(asker=my_asker, auto_allow=("Read",)) as approvals:
    cognition = ClaudeCliCognition(model="claude-sonnet-4-6", **approvals.cli_kwargs())
```

- **It fails closed.** A broken transport, a timeout, an unrecognised decision
  — all deny with the reason attached. An approval gate that fails open is not
  a gate.
- **`timeout_s` is enforced by the server**, not trusted to the `Asker`: the
  protocol lets an implementation wait forever.
- **`auto_allow` exists to prevent habituation** — the CLI prompts for reads
  too, and forty reflexive yeses are not oversight.
- The server binds loopback on an ephemeral port with **no authentication**;
  loopback-only is the containment.

No new dependency: `uvicorn` and `starlette` arrive with the existing `mcp`
extra.

### Added — `ClaudeCliCognition.session()`: one process, many turns

`drive()` spawns a subprocess per turn. That costs two to five seconds of CLI
warm-up every time, and — the part that actually matters — the turns share no
context. Measured on the same two-turn conversation against the real binary:

| | wall | turn 2 answers |
|---|---|---|
| `session()` | 9.7s | `8137` |
| `drive()` ×2 | 16.1s | *"I don't have a record of you asking me to remember a number"* |

A session holds the process open and feeds turns over stdin as
newline-delimited JSON (`--input-format stream-json`), so the CLI keeps its own
conversation context in memory. It is also `Cognition`-shaped, so it can *be* an
agent's cognition and consecutive `agent.run(...)` calls share one process.

Every per-turn contract holds unchanged, because both paths run the same
`_TurnState` fold and the same `_finalise`: exactly one terminal `final` event,
the same stop-reason taxonomy, the same structured-output handling, the same
metering. What differs is only what a shared process implies, each a deliberate
trade:

- **Turns are serialised** — one stdin and one transcript, so a second
  concurrent `turn()` waits rather than interleaving two conversations.
- **A dead process stays dead.** A CLI that exits mid-session ends the session;
  that turn reports the new `session_closed` stop reason and later turns refuse
  rather than silently starting a fresh conversation with no history.
- **Cancelling a turn ends the session**, since no protocol message retracts a
  half-finished turn.
- **`output=` is not per-turn**: `--json-schema` is fixed at spawn, and asking
  for it per turn is refused with that explanation rather than silently
  returning prose.

### Added — token streaming and startup diagnostics for the CLI cognition

- **`partial_messages=True`** (`--include-partial-messages`) streams the
  provider's own token deltas. Without it the CLI emits one `assistant` message
  per *completed* block, so `message_delta` arrived per paragraph — the class
  docstring called these "token chunks", which they were not.

  Turning it on creates a duplication hazard, because the same text arrives
  twice: once as `stream_event` deltas and again in the completed `assistant`
  message. The rule is that deltas are for EMITTING and the completed message
  is for ACCUMULATING, so a consumer sees each token once and
  `AgentResult.output` is written once. Non-text deltas (`signature_delta`,
  `input_json_delta`) are ignored rather than rendered as assistant text.

- **`evals["cli_init"]`** carries the `system/init` startup facts: the model
  that actually ran, which MCP servers connected, and — only when non-empty —
  `mcp_server_errors` / `plugin_errors`. The CLI validates each `--mcp-config`
  entry, skips the invalid ones and runs anyway, exiting cleanly, so their
  presence is the signal and the CI gate the CLI docs recommend. All of it was
  previously discarded.

- **`evals["api_retries"]`** records each `system/api_retry`, and each one is
  also emitted live as a `step` event — the only thing in the payload that
  explains a run which went quiet for forty seconds.

### Added — the CLI flags a service actually ships with

Reachable before only through `extra_args` — hand-written CLI syntax inside
application code, with no validation and no way for the cognition to know what
was set.

`add_dirs`, `mcp_config` + `strict_mcp_config`, `settings`, `agents`,
`fallback_model`, `effort`, `no_session_persistence`, plus two that matter
specifically at scale:

- **`bare=True`** (`--bare`) skips auto-discovery of hooks, skills, commands,
  subagents, plugins, MCP servers, auto memory and `CLAUDE.md` — the CLI docs
  call it "the recommended mode for scripted and SDK calls". Without it a `-p`
  session runs the hooks in the working directory's `.claude/settings.json` and
  connects the servers in its `.mcp.json`, including from a repository you just
  cloned. Bare mode also never reads OAuth credentials or the keychain, so the
  cognition warns when no API credential is in the environment and no `settings`
  is configured: the CLI's own error ("Not logged in · Please run /login")
  points at exactly the wrong fix on a machine where `claude` works
  interactively.
- **`stable_prompt_prefix=True`**
  (`--exclude-dynamic-system-prompt-sections`) moves per-machine sections out
  of the system prompt so the cache-stable prefix is identical across users and
  machines. Refused alongside `system_prompt_mode="replace"`, where it is a
  silent no-op.

Impossible combinations are refused at construction: `strict_mcp_config`
without `mcp_config`, and an `add_dirs` entry that is not a directory.

### Added — CLI spend is on the framework's books, and its ceiling on the CLI

`ClaudeCliCognition` bypasses the `Invoker`, so the `meter()` middleware never
saw its usage and every meter on the context stayed at zero no matter what the
CLI spent. A $50 CLI run against a $1 `Budget` completed happily with a ledger
reading `$0.00`. The class docstring admitted it ("callers who need a hard
ceiling on CLI spend must impose it externally") — the same shape as the
`ActorBudget` that was wired to nothing.

- **Before the spawn**: the run's *remaining* headroom goes out as
  `--max-budget-usd`, so the CLI stops itself mid-flight rather than being
  audited after the money is gone. An already-exhausted budget refuses to spawn
  at all — two to five seconds of CLI warm-up to be told what we already know —
  and reports the resumable `budget_exhausted`.
- **After the run**: the reported cost and tokens are charged to every meter on
  the context and to `ctx.actor_budget`. A `Quota` is charged through a minimal
  call shim, since it partitions on `call.ctx.scope.key()` and a bare `None`
  would have crashed it.
- A ceiling crossed by *this* run lands in `evals["meter_error"]` rather than
  raising: the money is spent, the answer exists, and converting the charge
  into an exception would lose a result the caller already paid for — and break
  the terminal-event guarantee on the way.
- `meter_spend=False` opts a run out of both ends.

### Added — `output=` types a CLI-delegated run

`ClaudeCliCognition` ignored `agent.output` entirely: the schema was never
sent, `AgentResult.parsed` was always `None`, and a caller who declared
`output=Invoice` got prose back with no sign that the typing they asked for had
been dropped.

The CLI has first-class support for this. The agent's schema now goes out as
`--json-schema` (through the same `SchemaAdapter` that types the rest of the
framework), the CLI validates its own final answer against it and re-prompts
itself on a mismatch, and the validated value comes back through that adapter
as a real Python object on `AgentResult.parsed`. The raw dict stays on
`evals["structured_output"]`.

Three outcomes, and only the first is a success — the other two previously
looked like one:

| Outcome | `evals["stop_reason"]` | typed |
|---|---|---|
| A conforming value | *(clean)* | `complete` |
| Retries exhausted | `error_max_structured_output_retries` | `invalid_output` |
| Exit 0 with no value | `structured_output_missing` | `invalid_output` |
| A value the Python type rejects | `structured_output_mismatch` | `invalid_output` |

`json_schema=` on the cognition overrides `output=` for a shape that is not the
agent's Python type. Verified end to end against the real binary: the adapter's
schema dialect is one the CLI accepts, and the round trip yields the declared
type.

### Fixed — `ClaudeCliCognition` flags meant something other than they said

Three mappings were wrong against the CLI reference, each producing a run that
looks fine.

- **`agent.prompt` went to `--system-prompt`, which REPLACES the entire Claude
  Code system prompt** — tool guidance, environment info, all of it. An agent
  given a one-line persona silently became a bare chat model still holding
  tools it no longer knew how to drive. It now goes to
  `--append-system-prompt`, which the CLI docs recommend for exactly this;
  `system_prompt_mode="replace"` opts into the old behaviour. **This changes
  the default behaviour of an existing field.**

- **`session_id` was documented as "resume an existing session"** and passed
  `--session-id`, which *names a new session* and must be a valid UUID.
  Following that docstring produced a fresh conversation with no history and no
  error. Added `resume_session_id` (`--resume`), `continue_session`
  (`--continue`) and `fork_session` (`--fork-session`); impossible combinations
  and a non-UUID `session_id` are now refused at construction, with the error
  pointing at `resume_session_id`.

- **`allowed_tools` reads like a sandbox and is not one.** `--allowed-tools` is
  an auto-approve list: every unnamed tool stays available and merely prompts.
  Added `tools` for `--tools`, which is the flag that restricts what the
  session has, plus `permission_prompt_tool` for `--permission-prompt-tool`.

A real-CLI test now spawns the binary with the full flag set and asserts it
accepts the argv — the one thing a mocked subprocess can never tell us.

### Hardened — the memory tool's path confinement

- `FileTool._confine` now refuses a **backslash** and a **NUL byte** outright.
  `posixpath` reads `\..\etc` as one ordinary filename, so it passed the
  traversal check — and then meant traversal to a backend running on Windows.
  The guarantee is now platform-independent rather than true only where it was
  tested. (The lexical `..`/absolute-escape checks were already correct; they
  are now pinned by tests and mutants, including the sibling-prefix case
  `/memoriesX` vs root `/memories`, and `rename` confining both ends.)

- The docstring now states plainly that confinement is **lexical**: a symlink
  inside the root that points outside it still escapes, because this check
  cannot see the filesystem. A filesystem-backed implementation must re-check
  after `os.path.realpath`. The previous wording promised that an injected
  backend "can't be turned into an arbitrary read/write/delete primitive",
  which was more than the code could deliver.

### Fixed — a tool call is checked against the schema the model was shown

- **Unknown argument names were dropped silently.** A parameter with a DEFAULT
  then ran with that default, so a model calling
  `notify(message="page the on-call, prod is down")` against
  `def notify(msg: str = "default message")` got back `"sent: default message"` —
  a side-effecting tool reporting success for something it was never asked to
  do, with nothing downstream able to tell. Unknown arguments now raise the new
  `ToolArgumentError`, which names the tool, the offending keys and the accepted
  set so the retry middleware can hand the model something it can act on.

- A missing required argument surfaced as a raw
  `TypeError: search() missing 1 required positional argument`, naming the
  Python function rather than the tool. Same typed error now, and a typo reports
  both halves (unexpected `querry`, missing `query`) since that is what the model
  needs to correct it.

- **`**kwargs` now receives the extras.** A tool declaring it was previously
  neither strict nor permissive — the extras were dropped and it never saw one.
  Declaring `**kwargs` is the opt-out from the check.

- **Structured parameters advertised as `{"type": "string"}`.** A Pydantic /
  dataclass / attrs parameter now gets its real object schema through the same
  `adapt()` dispatcher `output_schema` uses. The old fragment was instructing the
  model to send a string to a parameter whose annotation promised an object.

- **`Enum` parameters advertised as a bare string** with no member list, leaving
  the model free to invent one. They now emit `enum` (and a type when the member
  values are homogeneous), matching how `Literal` was already handled.

### Fixed — the signal channel's audit tap could wedge the run it audits

- **`SignalChannel.emit` awaited a queue nothing reads.** The documented
  concurrency model has the owning agent reading `inbox` and `merge_inbox`;
  `outbox` is the audit/replay tap. `emit` awaited both, so a run whose parent
  was draining perfectly still stopped dead once the tap filled — measured, the
  9th emit on a `buffer_size=8` channel blocked forever, and at the default size
  an agent went silent after its 256th signal. That is a deadlock wearing
  backpressure's clothes.

  The tap now drops its oldest entry instead (a rolling window is what a
  diagnostic tap is for; dropping the newest would make it go blind exactly when
  a run gets busy), counts the loss on `channel.dropped`, and warns once. The
  delivery path to the parent still awaits — that is the one place backpressure
  belongs, and a mutant pins it.

### Fixed — a termination condition is run-local state

- **Concurrent coordinator runs shared one condition.** A `TerminationCondition`
  is stateful and lives on the cognition, which lives on a long-lived `Agent` a
  server reuses. `ReActCognition` already deep-copied per drive for exactly this
  reason; the coordinator policies did not, so the documented "termination is
  per-drive" invariant held for leaf agents and quietly failed for teams. Two
  concurrent runs with `MaxTurns(4)` got **3 turns and 2 turns**. Both policies
  now clone, and the instance the caller passed in is never advanced.

- **Cloning had broken `ExternalTermination`.** The caller holds a handle and
  the run holds a copy, so `set()` could never reach a running loop — an
  "externally triggered stop" that only worked if triggered before the run
  started. It now opts out of the copy via `__deepcopy__`: an external switch is
  not per-run state. One switch shared by two concurrent runs stops both, which
  is the point of it.

- **`judge_termination` matched `YES` as a substring.** "Not yet — yesterday's
  draft is still open." stopped the run; so did "There is no simple yes/no
  answer here." Negation and hedging are precisely what a judge produces, and
  the docstring promised "stops only on an explicit affirmative". The affirmative
  must now LEAD the reply (leading punctuation skipped, case-insensitive, word
  boundary after). The bias is deliberate: "Answer: YES" reads as a non-stop and
  costs one more turn, which the hard ceiling bounds — while a false stop
  truncates the work and reports it complete, and nothing catches that.

- **`Stop` is frozen.** A condition latches its `Stop` and hands the same
  instance to every caller on every later turn, so a consumer assigning to
  `.reason` was rewriting the condition's own record of why it stopped.

### Fixed — routing picks the agent the model actually named

- **`llm_selector` scanned the roster in its own order for a substring.** The
  whole selector was `next((n for n in names if n in out), None)`, so a reply of
  `"Not alice — bob should go next"` routed to **alice** — the first roster entry
  appearing anywhere in the reply, regardless of where or why. A roster name
  living inside an ordinary word counted as a choice (`ed` inside `proceed`), and
  a `["bob", "bobby"]` roster resolved a reply of `"bobby"` to **bob**.

  Resolution is now: an exact reply wins; otherwise the LAST whole-word mention
  wins, longest name breaking a tie at the same offset. Reading the last mention
  follows the precedent `parse_handoff` already sets with its `rfind` — a model
  that reasons aloud commits at the end.

- **`route_by_handoff` never checked its target against the roster,** though the
  docstring said it did. An invented name was returned verbatim; `SelectorPolicy`
  cannot route a name it does not have, so it discarded the choice and fell back
  to round-robin by turn index — meaning the operator's pinned `default` was
  ignored at exactly the moment it mattered most. Targets are now resolved
  against the roster (exact, or a unique case-insensitive match, so `HANDOFF:Bob`
  still reaches `bob`), with a one-shot warning per invented name. An ambiguous
  case-fold is not guessed, and an empty roster keeps the old behaviour so direct
  callers are unaffected.

- **`parse_handoff` kept trailing punctuation on the target.** `HANDOFF:bob.`
  yielded `target="bob."`, which matches no roster entry — so the handoff a model
  wrote at the end of a sentence was silently lost. Internal punctuation
  survives, since `team.research-v2` is a legal agent name.

### Fixed — the plan human-gate is durable on a real store

- **The gate checkpoint held live dataclasses.** `PlanPolicy` put `Step`,
  `Usage` and `AgentResult` instances straight into `ctx.store.set`.
  `InMemoryStore` holds objects, so the whole feature tested green while a
  `FileStore` raised `TypeError: Object of type Step is not JSON serializable`
  — the human gate did not work on the persistence anyone deploys. There is now
  an explicit JSON-safe wire format, applied **unconditionally** so an
  in-memory test cannot pass over a broken encoding.

- **`resume` required `ctx.store`,** so a `Services(checkpointer=...)` wiring —
  the documented durable seam — could suspend but never resume. The gate now
  goes through the same `resolve_checkpointer` order as every other producer, at
  its own namespaced slot `{run_id}:plan`, with `CheckpointStatus.SUSPENDED` so
  an auto-resume supervisor can tell "waiting on a human" from "engine in
  motion". Records under the old raw `plan_policy:<run_id>` key are still read
  (and cleared) on resume, so a plan suspended across the upgrade finishes.

- **A gate reached with no seam wired suspended in silence,** handing back a run
  id whose every `resume` raises. It now warns, exactly as `Workflow` does.

- A child result whose `evals` / `parsed` cannot be serialized no longer takes
  the run down at the gate: those two fields are dropped with a warning, since
  losing the whole run is strictly worse than losing them.

### Fixed — a plan is checked against the roster before it dispatches

- **A step naming an unknown child raised `KeyError('reseacher')` mid-flight.**
  It surfaced from `children[s.agent]` inside the dispatch loop — after the
  earlier groups had run and spent money, and with their results unreachable,
  because the accumulator is a local of `_run_groups`. Under `best_effort=True`
  that also broke the mode's one promise: partial progress survives. Unknown
  children now either refuse up front (fail-fast) or land in `evals["errors"]`
  as a `PERMANENT` `Failure` while the rest of the plan runs (best-effort).
  `resume` re-validates too, since the roster is re-supplied by the caller.

- **A gate sharing a group with dispatch steps silently dropped them.** The
  group suspends before any of its steps run, and resume continues at the group
  *after* the gate — so those steps were announced in the trace and then never
  executed, on approve and on reject alike. Refused as `PlanShapeError`: whether
  the work belongs before or after the human's decision is exactly what the plan
  failed to say, so the framework states the problem instead of picking one.

- A `Step` with neither an `agent` nor a `gate_name` reached `children[None]`.
  Also refused, by name.

### Fixed — every producer stamps the typed stop reason

- **A plan parked on a human gate reported itself `complete`.**
  `AgentResult.stop_reason` is the closed taxonomy a caller branches on, and
  `is_suspended` / `is_resumable` derive from it — but only the tool loop ever
  set it. Every coordinator policy (round-robin, selector, ledger, plan) wrote
  its real reason into `evals["stop_reason"]` and left the typed field at its
  `"complete"` default. So a suspended plan read back as
  `is_suspended is False` with its checkpoint sitting in the store: an
  application branching on the typed field never prompted its human and never
  called `resume`. A coordinator that ran out of turns was likewise
  indistinguishable from one that finished its work.

  The free-form → closed mapping now lives once, beside the taxonomy, as
  `agents.result.stop_reason_for`; the tool loop's private copy is gone. It is
  total — an unrecognised reason becomes `terminated`, never a guess — so a
  producer can stamp the field unconditionally.

- **`stop_reason` was dropped on the durable round trip.** `result_to_dict` /
  `dict_to_result` back coordinator resume, and neither carried the field, so
  every rehydrated result read back as `complete`. Records written before this
  upgrade in place, deriving the category from the free-form reason they do
  carry.

- **Added `AgentStopReason.failed`.** `ClaudeCliCognition` guarantees a terminal
  event even when the subprocess never starts, so it reports failures as data
  rather than raising — the one producer that legitimately can. Mapping
  `spawn_failed` / `cli_exit_2` onto `terminated` would have read as "something
  stopped this on purpose", which is the opposite of what happened.

  A source-level test now refuses a new framework reason string that is
  categorised nowhere, so the next producer has to make the decision rather
  than inherit a silent fallback.

### Fixed — one checkpointer resolution order, not three

- **A `Services(store=...)` wiring gave silently non-durable coordinator runs.**
  The coordinator policies had their own resolution order that stopped at
  `ctx.checkpointer` and deliberately excluded the store bridge, on the stated
  grounds that "coordinator runs require a real `Checkpointer` for durability".
  That does not survive inspection: the bridge is exactly as durable as the
  store behind it, and its only documented limitation — a single slot per run,
  no version history — costs nothing here, because no policy reads history.
  Meanwhile a completed coordinator run left **zero** keys in the store, with
  no warning, while ReAct runs and Workflow gates on the same wiring persisted
  fine. All three producers now share `resolve_checkpointer`.

  Viable only because of the slot namespacing below: a coordinator writing at
  the run id and its children at `{run_id}:agent:{name}` no longer collide.

### Fixed — a child agent deleting its coordinator's checkpoint

- **Every producer keyed durable state on `ctx.correlation_id`**, and
  `ctx.child()` propagates that unchanged — so a coordinator and each of its
  child agents wrote to the SAME checkpoint slot. Not merely an overwrite: a
  child finishing normally calls `_clear`, which is
  `Checkpointer.delete(run_id)`, removing ALL versions for the id. Measured: a
  coordinator wrote its in-progress turn state, one child completed
  successfully, and the coordinator's checkpoint was **gone** — so a crash then
  lost a run that had checkpointed. Version numbering restarted too, breaking
  the monotonic-version guarantee the Checkpointer documents. Parallel siblings
  clobbered each other by the same mechanism.

  The tool loop now owns `"{run_id}:agent:{name}"`, exposed as the public
  `ReActCognition.checkpoint_slot` because an operator tool listing or clearing
  durable state needs the same derivation. `Suspended.run_id` still carries the
  plain id a caller passes back to `Agent.resume`, so nothing public changed.
  `resume` reads the legacy bare-id slot as a fallback, so an in-flight suspend
  survives the upgrade — **guarded by a payload shape check**, because the bare
  id is exactly the slot other producers still use, and reading it
  unconditionally re-introduced the collision from the other direction (a child
  loop picked up its coordinator's state and died in `rehydrate` with
  `KeyError: 'messages'`).

  Known limit, stated rather than hidden: two children sharing one agent *name*
  in a single run still share a slot. That is ambiguous by identity; name them
  distinctly.

### Fixed — FileStore durability, and a StorePort contract nobody checked

`InMemoryStore`'s docstring calls itself "the offline reference `StorePort` and
the contract every durable backend matches". Nothing checked that claim, and
they had drifted.

- **A crash during a write left the checkpoint permanently unreadable.**
  `FileStore.set` used `path.write_text`, which is not atomic — so a process
  that died mid-write produced a truncated file and every later `get` raised
  `JSONDecodeError` forever. The adapter exists to "survive a process restart,
  so a human-gate suspend or a crashed run resumes from disk"; the failure mode
  it is *for* was the one that broke it. Now writes to a temp file in the same
  directory and `os.replace`s it, so a reader sees the whole old file or the
  whole new one. (Without `fsync` this survives a process crash, not a power
  loss — which is the claim the docstring makes.)
- **A corrupt entry now names its file.** Writes are atomic, so an unparseable
  file means external corruption. It raises `StoreUnavailable` rather than
  returning `None`, because reporting "no checkpoint" would restart a run that
  has durable state — and the message includes the path, where a bare
  `JSONDecodeError` from inside a `to_thread` frame gave an operator nothing.
- **One torn log line destroyed the whole audit trail.** `list()` raised on the
  first unparseable line, so a crash during an append made every *earlier*
  record unreadable too. Bad lines are now skipped (with a one-shot warning)
  and the surviving records returned.
- **`get_or_set` broke single-flight on a falsy value.** It tested
  `existing is not None`, conflating "nothing stored" with "`None` stored", so
  a producer returning `None` re-ran on every call: 3 invocations against
  `InMemoryStore`'s 1, on identical input. Now keyed on presence.
- **A silently-ignored `ttl` warns once.** FileStore has no expiry sweeper, so
  a ttl is permanent — which matters most for idempotency, where a key that
  never expires dedupes a legitimate retry of the same operation forever. It
  was a docstring note; it is now visible at the call site.
- **Added a `StorePort` conformance contract** (11 properties × memory/file) to
  the protocol suite, so the next backend either matches the reference or fails.

### Fixed — a provider failure that read as a complete answer

- **An in-band SSE error frame was swallowed by both providers.** Anthropic and
  OpenAI can deliver a failure INSIDE a 200 response, part-way through a
  stream, once the headers are long gone
  (`{"type":"error","error":{"type":"overloaded_error"}}` /
  `{"error":{"type":"server_error"}}`). Neither translator had a branch for it,
  so the frame fell through every `elif`, the loop ended normally, and the
  caller received a **truncated answer presented as a complete one** — partial
  text, `finish_reason=None`, no exception anywhere. An agent takes that
  half-sentence as the model's final word. And because nothing raised,
  `retry()` never fired: the most retryable provider failure there is, an
  overload, was the one the resilience layer never saw. Both paths now raise a
  classified `ProviderError` via a shared `raise_if_error_frame`.
- **The error classifier missed the underscore forms providers actually use.**
  `_TRANSIENT` had `rate limit` (with a space) while the wire carries
  `rate_limit_error`, and had no `server_error`, so the errors above landed in
  `UNKNOWN`. They were still retried — only `PERMANENT` fails fast — but
  classified on nothing. Added `rate_limit`, `server_error`, `529`. Bare `500`
  is deliberately still absent: it is a substring of `5000`, which appears in
  ordinary text like `max_tokens 5000`, and a false `TRANSIENT` there retries a
  request that can never succeed.

### Fixed — tenant isolation, and two fail-open controls

- **`memoize` leaked cached answers across tenants.** `Scope`'s own docstring
  calls itself the key "threaded through every memory recall / cache key /
  meter / callback", but `memoize` took an arbitrary `key` callable and added
  nothing to it — so isolation depended on every caller remembering. They did
  not, because the key the cheatsheet and the LangChain migration guide TAUGHT
  was `lambda c: c.request.messages[-1].content`, which ignores the model, the
  tools, the temperature and the tenant. Measured: two tenants asking the same
  question, one provider call, and tenant 999 receiving tenant 1's answer.
  Every key is now namespaced by `ctx.scope.key()` **inside** the middleware —
  a boundary that relies on a caller-supplied key is not a boundary.
- **`memoize()` now works with no arguments** and defaults to an exact-match
  key over the fields that change the answer (model, messages, tool names,
  response_format, temperature, max_tokens). `key` was required, which pushed
  the most dangerous decision in a cache — what counts as "the same call" —
  onto every caller. Two docs pages already showed a bare `memoize()`, which
  raised `TypeError`. Tool schemas reduce to their names, so editing a
  description does not invalidate the cache.
- **`egress(None)` built a security control that checked nothing.** It sat in
  the chain with every SSRF and allowlist check silently off, which is how
  `egress(config.guardrail)` behaves when the config is unset. Now raises at
  wiring time, along with a `TypeError` for an object that has no `check_url`.
- **A late success closed an OPEN circuit breaker**, adding an `OPEN → CLOSED`
  edge the class docstring says does not exist (`CLOSED → OPEN → cooldown →
  HALF_OPEN → CLOSED`). Reachable at any concurrency above one — the normal
  case for the documented pattern of one breaker shared per dependency: enough
  in-flight calls fail to trip it, then a straggler that started before the
  trip reports success and reopens the gate. Measured through the real
  `retry()` middleware: a 300-second cooldown skipped by one late success,
  sending the herd back at a failing provider. A call admitted under the old
  state is evidence about the past; only the post-cooldown probe speaks to
  recovery.

### Checked and found sound

- The SSRF host blocker was audited against the classic bypasses and blocks
  all of them: decimal (`2130706433`), hex, octal, short-form (`127.1`), IPv6
  loopback, IPv4-mapped IPv6 (`[::ffff:127.0.0.1]`), userinfo
  (`user@127.0.0.1`), cloud metadata (`169.254.169.254`), RFC1918, and
  unspecified. A hostname that RESOLVES to a private IP is allowed, which is
  documented behaviour — name resolution is the injected `url_check`'s job.
- `SlidingWindowCompactor` preserves the system prompt when trimming.

### Fixed — the fan-out reservation path

- **A starved fan-out silently produced no-op children on two of three axes.**
  Only `steps` failed fast when a slice would round to nothing; `tokens` and
  `cost` floored to zero, so a reservation of zero "succeeded" and each child
  was handed an already-exhausted envelope. A fan-out of 3 against 2 tokens ran
  three children that each stopped immediately and looked like a completed
  wave. All three axes now fail fast with `BudgetExhausted`, naming the axis,
  and a fan-out from an already-exhausted parent refuses outright rather than
  carving zero-sized slices.
- **A child was over-granted a step it was never reserved.** The step axis was
  handed `max(slice_steps, 1)`, so with a one-step slice a child could take a
  second turn the parent had not committed — and `settle_child` caps usage at
  the reservation, making that spend invisible on the parent's books. Children
  now get exactly their slice.
- **Slices are carved in `Decimal`**, off `remaining_cost()` rather than the
  float mirror. Equal shares, so reservation order cannot skew fairness.
- **`_tightest_axis` crashed on the path it exists to explain.** It mixed the
  float mirror with the request amount, so once `run_agents` began slicing in
  `Decimal` the diagnostic raised `TypeError: unsupported operand type(s) for
  -: 'float' and 'decimal.Decimal'` instead of naming the blocking axis. Every
  money-bearing `ActorBudget` parameter now accepts what `to_money` accepts.
- `kernel/concurrency.py` coverage **57% → 91%**, including the reservation /
  settlement path and `run_sync`'s nested-loop branch (a sync host calling in
  from inside an async caller — the branch that quietly regresses into a
  deadlock). Coverage ratchet raised 85 → 87.

### Fixed — an inert ActorBudget, and one durable seam

- **`ActorBudget` did nothing.** Nothing in the framework ever charged it: the
  `meter()` middleware charges `ctx.run.all_meters` (the run `Budget` plus any
  `Quota`), and `ActorBudget` is not a `Meter` — four axes, a sync `charge`, no
  guard/charge protocol — so it was never in that list. And no loop consulted
  `exhausted()`, even though `charge` is documented as soft-exceed-then-stop
  *because* "the loop checks `exhausted()` and stops cleanly". The only thing
  that ever touched the envelope was `run_agents` reserving slices and
  releasing them with zero usage. Measured: **$3.00 of real spend against a
  $1.00 cap left `used_cost` at zero and `exhausted()` False**, while the
  run-scoped `Budget` correctly recorded $3.00. A documented safety mechanism
  that never ran. Both ends are now wired, and the terminal reason names the
  exhausted axis.
- **`ActorBudget`'s cost axis is an exact `Decimal` ledger**, replacing the
  epsilon threshold added earlier this cycle. `max_cost_usd` / `used_cost_usd`
  / `reserved_cost_usd` remain float MIRRORS so `run_agents` and existing
  readers are untouched; `max_cost()` / `used_cost()` / `reserved_cost()` /
  `remaining_cost()` are the exact accessors.
- **A ceiling crossed by the CLOSING call no longer discards the answer.** The
  post-call check fired after every chat call, so a run whose final call
  happened to exhaust the budget reported `budget_exhausted` with
  `partial=True` — for work already paid for, with a good result in hand. The
  check now runs only where the loop is about to spend *more*: before a tool
  dispatch, or before a repair retry.
- **`Workflow` now persists through the same seam as everything else.** It
  wrote only to `ctx.store`, while the ReAct cognition prefers
  `ctx.checkpointer` — so wiring the documented durable seam left workflow
  human-gates silently unpersisted. Both producers now share one
  `resolve_checkpointer`, gates are marked `SUSPENDED` (a status a bare KV
  write could not express), and `resume` falls back to reading the legacy
  `workflow:<run_id>` key so in-flight suspends survive the upgrade.

### Changed — CI

- Action majors bumped together (`actions/checkout@v7`,
  `actions/upload-artifact@v7`, `actions/download-artifact@v8`,
  `astral-sh/setup-uv@v10`). GitHub had forced every Node-20 action onto Node
  24, annotating every run. Our usage is limited to long-stable inputs, so the
  majors carry no interface change for us. The PyPI publish action stays
  SHA-pinned — it holds signing authority.

### Fixed — concurrency, budgets and workflow suspend

- **Nested fan-out deadlocked.** `ctx.semaphore()` returned ONE semaphore for
  the whole agent tree, and a parent's fan-out holds its permits for the entire
  duration of each child run — so a nested fan-out drew from a pool its own
  ancestors had already drained. Reproduced through the public API: an Agent
  dispatching two `as_tool` sub-agents that each dispatch their own tools hangs
  forever at `max_concurrency=2`. The pool is now keyed on `ctx.depth`, which
  breaks the cycle structurally since every nesting boundary goes through
  `ctx.child()`. **Behaviour change:** the bound is now `max_concurrency` per
  LEVEL, so worst-case in-flight work is `max_concurrency * (max_depth + 1)`.
  A single tree-wide cap cannot be both deadlock-free and respected by nested
  acquisition.
- **`gather_best_effort` swallowed cooperative cancellation.** It correctly
  re-raised `asyncio.CancelledError` but caught agentkit's own `Cancelled`
  under `except Exception`, turning it into a `Failure` slot. The token is
  shared across the run tree, so a tripped token gave the caller N independent
  "failures" with no way to tell an aborted run from a batch where everything
  broke at once. `Cancelled` is an abort and now aborts.
- **`ActorBudget` under-reported exhaustion.** Its float axes kept an
  `== 0.0` check, which does not survive float arithmetic: a `$1.00` cap
  charged ten times at `$0.10` left `remaining == 1.11e-16` and
  `exhausted() == False`, so the agent loop ran past its cap. `remaining_*`
  clamps with `max(0.0, …)`, catching an overshoot but not an undershoot. It
  was also inconsistent — `0.1 + 0.2` against a `0.3` cap lands exactly on
  zero — so the bug depended on which numbers a caller picked. Now thresholded
  at one unit of `MONEY_SCALE`. (The run-scoped `Budget` already had an exact
  Decimal ledger; converting ActorBudget's float reservation API is a larger
  change worth doing on its own.)
- **`Quota` never evicted expired tenants.** `_prune` only touched the key
  being guarded, so a tenant that went quiet leaked its dict entry forever —
  5000 distinct scopes left 5000 retained keys long after every window had
  expired. That is a slow leak in exactly the multi-tenant deployment the class
  exists for, and worse than one-entry-per-customer when a scope carries a
  per-user id. Added a sweep, at most once per window.
- **A workflow gate suspended with no store persisted nothing, silently.**
  `run()` returned `stop_reason="suspended"` with a `Suspended` object while
  writing no checkpoint; the truth surfaced later, usually in another process,
  as "no suspended workflow <id> to resume" with nothing pointing at the cause.
  Now warns at suspend time. The warning also surfaces an asymmetry: Workflow
  persists via `ctx.store`, while the ReAct cognition prefers
  `ctx.checkpointer` — so wiring only a `Checkpointer` leaves a workflow gate
  unpersisted too.

#### Notes on the above

- Abandoning `Agent.stream` before the `final` event releases the provider's
  HTTP stream at generator finalization, not at your `break`. On CPython that
  is prompt (200 abandoned streams measured to zero un-released after one event
  loop turn) but not deterministic; use `contextlib.aclosing` for a hard
  guarantee. Making every framework layer cascade the close would mean
  re-indenting ~22 `async for` sites across the middleware chain, which is not
  worth the risk for a leak CPython already collects.

### The five production-feedback briefs

Five gaps reported from production use. Everything here is **additive** —
no existing wiring changes behaviour, and the full pre-change test suite
passes untouched.

#### Added

- **`StreamEvent.partial_output`** — the in-progress typed object, forwarded
  from `Delta.partial` by both chat cognitions. An application can now stream
  a typed object through `Agent.stream` alone; previously the framework could
  parse a partial structured output and had nowhere to deliver it, forcing
  callers to reach into `_resolve_request_builder()` / `_output_adapter` and
  drive `ctx.invoker.stream` by hand. Named `partial_output`, not `partial`,
  to stay distinguishable from `AgentResult.partial` (a `bool` meaning "this
  run terminated incompletely"). Consumers must gate on `model_fields_set` —
  required fields may be unset. `assemble_deltas` still drops `partial` by
  design; see its docstring.
- **`Agent` warns once** when an output schema is declared but `output_coerce()`
  is missing from the chat chain — previously `partial_output` was silently
  `None` forever while `AgentResult.parsed` kept working, so nothing looked
  broken. `Invoker` now exposes `chat_middleware` / `tool_middleware` so the
  chain is introspectable.
- **`agentkit.adapters.llm.model_registry`** — one table mapping a model name
  to the provider that serves it *and* the capabilities it declares.
  `resolve_llm(model)` / `client.from_env(model)` read credentials from the
  environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
  `OPENROUTER_API_KEY`), replacing the bespoke bootstrap every application was
  writing. Provider factories load lazily by dotted path, so the registry
  stays importable on a zero-dependency install. Fallback is **opt-in**
  (`fallback="fake"`, warns once); a missing optional extra raises
  `MissingProviderExtra` naming the pip extra and is never absorbed into the
  fallback. Applications extend it with `register_model` / `register_provider`
  / `register_rule`. Credentials never appear in an error, warning, or repr.
- **Model capability declaration + bind-time refusal.** `Capability` is
  tri-state (`YES` / `NO` / `UNKNOWN`); an unregistered model reports
  `UNKNOWN`, never a guess in either direction. `Agent(requires=(...),
  min_context_window=..., on_unknown_capability=...)` refuses a mismatch in
  `__post_init__` — before any spend — raising `CapabilityMismatch` naming the
  capability and the model. A tool-holding cognition implies `tools`
  automatically; derived requirements raise on a declared `NO` but stay silent
  on `UNKNOWN`, so existing wiring with custom model names is unaffected.
  `Agent.check_capabilities()` re-asserts after a post-construction mutation.
- **`Charge` verdict + recoverable budget exhaustion.** `Meter.guard` /
  `Meter.charge` now return a `Charge` (`ok`, `reason`, exact `spent` /
  `remaining` as `Decimal`, cumulative `usage`). With `Budget(on_exceeded="stop")` the
  meter records the spend and returns the verdict instead of raising, and the
  tool-loop cognition writes a **current `suspended` checkpoint** before ending
  the run with `stop_reason="budget_exhausted"`. Previously `MeterExceeded`
  unwound out of `invoker.stream` past every checkpoint write, leaving a stale
  `running` snapshot that recorded only part of the money actually spent.
- **`Budget.usage`** — the full `Usage` accumulates (input / output /
  cache-read / cache-write tokens), shared by reference across `ctx.child()`,
  so a multi-agent run's token totals survive to the end. Applications no
  longer re-aggregate what the framework already summed.
- **`AgentResult.stop_reason`** — a closed `Literal` (`complete`, `suspended`,
  `expired`, `budget_exhausted`, `max_iterations`, `invalid_output`,
  `terminated`) plus `is_suspended` / `is_resumable`. A suspended run is now
  distinguishable from a failed one without parsing the `evals` bag; a failed
  run still raises and produces no `AgentResult` at all. The free-form detail
  string stays in `evals["stop_reason"]`.
- **`agentkit.agents.control.elicit`** — human-in-the-loop as value
  elicitation. `Elicitation` names what the run needs and `Decision` carries
  what a person supplied, with `actor` and `at` for the audit trail. An
  injected `Asker` (on `Services.asker`) makes a gated decision **park in
  place** — the cognition awaits inside its own coroutine, so live
  unserialisable state survives — while the return-and-resume path stays
  available and unchanged for callers that can serialise. `elicit(ctx, ...)`
  takes a `Ctx`, so it works from any cognition, not just ReAct;
  `ask_human_tool()` exposes it to the model itself.
- **Deadlines on a suspended run.** `Elicitation.deadline_s` and
  `ReActCognition(approval_deadline_s=...)` bound the wait; expiry produces
  `Decision(kind="expired")` and the run **degrades and continues** rather
  than hanging. `Suspended.deadline_at` carries absolute wall-clock expiry for
  an operator UI. `elicit_or_raise` opts into `ElicitationExpired` instead.
- **`SecretValue`** — redacts itself in `repr`/`str`; `reveal()` is the one
  explicit way out. A working context that has handled a secret elicitation is
  **never checkpointed again** for the rest of that run, so a one-time code
  cannot outlive its validity inside a durable store.

#### Changed

- **Money is `Decimal`.** `Budget` / `Quota` keep an exact ledger at six
  decimal places; a hundred charges of `0.01` now sum to exactly `1.00`.
  `budget.spent()` and `budget.spent_cents()` are the exact reads, and
  quantization happens at read time so sub-cent calls are not rounded away.
  `budget.spent_usd` remains a `float` **mirror**, re-derived after every
  charge — every existing reader and the documented
  `Budget(spent_usd=saved.state[...])` resume path keep working. An
  over-precise *ceiling* raises `MoneyPrecisionError` at construction; an
  over-precise *charge* is quantized rather than aborting a run mid-flight.
- **`_infer_response_format` reads the registry**, not `model.startswith("gpt-")`.
  Provider-native `json_schema` mode is wired from the declared
  `native_json_schema` capability; the `gpt-` prefix survives only as a
  last-resort guess for an unregistered name.
- **`Agent.resume` / `ReActCognition.resume` accept a typed `Decision`** as
  well as the legacy `str` form, which is coerced. Signatures widened to
  `Mapping[str, str | Decision]` (`dict` is invariant, so a caller holding a
  `dict[str, str]` would otherwise fail type-checking). A missing entry still
  defaults to a denial. Denial messages now name the actor.
- **`Quota` returns verdicts too**, so both `Meter` implementations behave
  uniformly under the middleware's `all_meters` iteration.

#### Fixed

- **`output_coerce()` no longer defeats parse-and-repair.** The middleware
  strict-parses at end-of-stream and re-raises; that exception escaped past
  the cognitions' reflect-and-retry branch, so adding the middleware — the
  very wiring that enables streamed partials — aborted the run on the first
  malformed response. Both cognitions now catch `OutputCoercionError` and let
  `agent.parse` re-raise it inside the repair loop. (`output_coerce` itself is
  unchanged.)

#### Fixed (found in review of the above)

- **`Agent.resume` bypassed the `RunPolicy` lethal-trifecta gate.** The gate
  lived inline in `stream()`, so an agent whose tool set combines
  private-data access, untrusted-content ingestion, and egress was denied on
  `run()` and then executed that exact tool on `resume()`. Now shared by both
  entry points via `Agent._run_policy_gate`. Pre-existing; security-relevant,
  because resume is the path a human has just approved something on — and
  approving one tool *call* is not approval of the capability *combination*.
- **Retrying an exhausted budget spent another call each time.** Under
  `on_exceeded="stop"`, `guard()`'s not-ok verdict is ignored by the meter
  middleware (a middleware cannot write a checkpoint), and the cognitions only
  checked *after* the call — so every retry of an already-stopped run bought
  one more turn before noticing. Both cognitions now pre-flight the budget.
- **`Budget.max_cost_usd` was ignored after construction.** The normalised
  ceiling was cached in `__post_init__`, so `budget.max_cost_usd = 10.0` — the
  documented way to raise a ceiling and resume — silently did nothing. Now
  re-derives on assignment (non-strictly, because that read path is reached
  from inside `charge()` and raising there would abort a run mid-flight). Same
  for `Quota.max_usd`.
- **A suspend deadline was decorative.** `Suspended.deadline_at` was stamped
  but never persisted and never checked, so an operator answering an hour late
  still got the tool executed. It is now written into the checkpoint and
  honoured on `resume()`: pending calls become `expired`, the run degrades and
  continues, and the tool does not run.
- **The secret-taint containment covered one of seven checkpoint writers.** It
  sat in `ReActCognition._save`; the coordinator policies persist a blackboard
  scratchpad through their own `snapshot` calls. Moved to `Checkpointer.snapshot`
  — the single seam every producer passes through.
- **The park path emitted no `tool_call` event**, so a consumer counting them
  to render "running X…" saw nothing on approved gates.
- **`ask_human_tool` elicitation ids were unstable across processes**
  (`hash(str)` is randomised by `PYTHONHASHSEED`), breaking audit-trail
  correlation. Now a SHA-256 prefix.
- **`Charge.spent_usd` collided with `Budget.spent_usd`** — same name, `Decimal`
  on one class and `float` on the other. Renamed to `Charge.spent` /
  `Charge.remaining`; `*_usd` now means float everywhere and `spent`/`remaining`
  mean Decimal everywhere.

#### Notes

- An `Asker` whose `ask` blocks the event loop (`input()`, `requests.get()`,
  `time.sleep()`) **cannot be deadlined** — scheduling is cooperative, so the
  timeout coroutine never runs and `deadline_s` becomes an unbounded hang. The
  `Asker` Protocol carries an explicit warning and a test pins the behaviour.
  Wrap synchronous work in `asyncio.to_thread`.
- `Quota` never evicts per-tenant keys from its internal window dicts, so a
  long-running process with unbounded tenant churn grows slowly. Pre-existing,
  not addressed here.
- `Budget(on_exceeded=...)` defaults to `"raise"`. Flipping it would silently
  change control flow in every existing wiring — a run that used to abort
  would continue past its ceiling in any caller ignoring the return value.
  Callers opt into recoverability.
- `output_coerce` re-parses the whole accumulated buffer per text delta, which
  is O(n²) in response length. Left as-is deliberately; sample partials if it
  shows up in a profile.

## [0.1.0] — 2026-08-04

Initial public release. Distributed on PyPI as `arc-agentkit`; imported as
`agentkit`.

### Added
- Complete framework: `kernel/` (opinion-free value types, ports, middleware
  contract, resilience, concurrency, observation), `runtime/` (`RunContext`,
  `Invoker`, `Budget`, `Quota`, `EventBus`, `NullCtx`), `middlewares/` (the
  chat + tool chain: `tracing`, `meter`, `retry`, `fallback`, `memoize`,
  `output_coerce`, `compaction`, `security`, `egress`, `audit`), `context/`
  (`WorkingContext` + `TokenCounter`), `memory/` (unified `MemorySource`
  Protocol + `Composite`/`Sequential`/`Vector`/`File`/`Journal`/`Scratchpad`/
  `Tool` sources + `Scoped`/`Compacted`/`Cached` decorators), `tools/`
  (Tool Protocol + `FunctionTool` + `ToolRegistry` + `@tool` +
  `as_tool`), `prompts/` (versioned `Prompt`), `skills/`
  (`Skill` Facade), `agents/` (`Agent` + `Workflow` + Cognition Protocol
  with `SingleCallCognition`/`ReActCognition`/`CoordinatorCognition` +
  signals + policies), `capabilities/` (`RequestBuilder`, `Grounder`,
  `Compactor` (4 strategies), `Guardrail`, `Evaluator`, `Checkpointer`,
  `SchemaAdapter`), `adapters/` (concrete Port impls behind opt-in extras),
  `testing/` (`FakeLLM`, `FakeFetch`, `FakeSearch`, `FakeMemory`, `FakeTool`,
  `FakeCompactor`, `FakeGrounder`, `make_test_ctx`).
- Batteries-included provider client: `Chat` + `claude` / `openai` /
  `deepseek` / `openrouter` presets, pre-wired with `tracing → meter →
  retry` on a `RunContext`.
- Typed error taxonomy raised at adapter/port boundaries: `AgentkitError`
  (base), `CheckpointerError`, `StoreUnavailable`, `ProviderAuthError`.
  Adapters wrap backend exceptions (`asyncpg`, `httpx`, `redis`, …) with
  `raise …Error(…) from exc` so callers pattern-match on the framework
  taxonomy without importing backend types.
- `RunPolicy` auto-invoked from `Agent.run` when set: `mode="deny"` raises
  `PermissionError` before the first cognition drive; `mode="flag"` stamps
  a `policy.flagged` observation and lets the run continue. One
  `policy.check` span per run carries mode, capabilities, and verdict.
- `Checkpointer` acquires a per-`run_id` `asyncio.Lock` around the
  read-compute-write cycle in `snapshot`, and deep-copies `state` +
  `metadata` on save so subsequent producer-side mutation cannot corrupt
  the stored version. Concurrent snapshots see monotonic distinct
  versions instead of racing on `next_version = max + 1`.
- Apache-2.0 `LICENSE` + `NOTICE` at the package root; `py.typed` marker
  ships in the wheel; `agentkit.__version__` populated from installed
  distribution metadata.
- mkdocs-material documentation site deployed to
  [`arc-labs-ai.github.io/agentkit`](https://arc-labs-ai.github.io/agentkit/).
- `docs/mental-models/` covering four canonical use cases (multi-tenant
  chat, autonomous DevOps investigator, long-running enrichment,
  coordinated research), each stressing a distinct set of framework
  invariants.
- Runnable examples under `examples/` — every example uses `FakeLLM` and
  needs no API key: `01_single_agent.py`, `02_streaming_and_tools.py`,
  `03_composed_middlewares.py`.
- Optional-extras: `http`, `postgres`, `redis`, `observability`, `fast`.

### Changed
- `Suspended.pending` typed as
  `tuple[ToolCall, ...] | tuple[str, ...]` (previously an unconstrained
  tuple). The tool-approval surface emits `ToolCall`s; `Workflow`'s
  `human_gate` node emits gate-name `str`s. The narrowed union catches
  drift from a third caller passing arbitrary objects, and pins the
  suspend/resume handshake at both ends of the wire.
- `ProviderAuthError` multi-inherits from
  `agentkit.kernel.errors.ProviderAuthError` (kernel taxonomy) and the
  transport-level `ProviderError`, so raised instances satisfy
  `except AgentkitError`, `except ProviderAuthError`, and legacy
  `except ProviderError` blocks simultaneously.

### Fixed
- Roughly 200 `mypy --strict` findings across `kernel/`, `runtime/`,
  `agents/`, `capabilities/`, and `adapters/` — missing return types,
  `Any`-typed public seams, `Optional` mis-annotations, and a handful of
  variance mistakes on Protocol generics. `uv run mypy --strict agentkit`
  is now clean on the full tree.
