# Building a harness on agentkit

A harness owns application state, durable scheduling, credentials, authorization and deployment. Agentkit provides model calls, canonical context, tool execution, versioned agent definitions, replaceable minds, and event ports. These contracts do not depend on any one application repository.

## Minds and agents

`agentkit.agents.minds` separates reasoning behavior from model transport. `SingleCallMind` performs one call, `ReActMind` uses the shared tool loop, and `ChainOfThoughtMind` adds deliberate internal reasoning guidance while returning only the requested artifact. Chain of thought is a prompt-level strategy; it does not expose private model reasoning. Single-call and chain-of-thought minds refuse nonempty toolsets rather than silently ignoring tools. Implement the `Mind` protocol to add planning or other strategies; inject the implementation through `AgentRunner(minds=...)`.

`AgentDefinition` specifies version, binding or `ModelSelection.INHERIT`, persona skills, discoverable skills, structural `ToolGrant`, settings, limits and `DelegationPolicy`. Its digest excludes credentials. `MemoryAgentStore` prevents replacing the contents of an existing version; production hosts implement `AgentStore` on persistent storage.

`InstructionSet` carries the goal, definition of done, rules, stop conditions, escalation guidance, orientation and optional verifier. Without a verifier these fields guide the model; they are not executable host policies. A false verifier resumes the conversation with correction guidance, consuming the same turn, spend and wall-clock envelope. Repeated identical answers stop according to `no_progress_slices`; without a configured turn ceiling verifier continuation is additionally limited to ten slices. Tool permissions and limits remain enforced by the runtime, independently of prompt text.

Delegated agents receive fresh context and their own permitted hosted tools. CLI native capabilities are governed separately by `NativeToolPolicy`; structural grants alone do not disable every Codex built-in capability. The parent reserves a configured fraction of its spend ceiling for delegation and keeps the remaining envelope for its own calls. Each child reserves its explicit cost ceiling, capped by the remaining shared delegation envelope. Known unused allocation is released for subsequent children; unknown spend retains its whole reservation and makes the aggregate cost unknown. This reservation is conservative: unused child funds are not lent back to a parent already executing. An ancestor depth ceiling cannot be expanded by a child; concurrency is bounded per invoking agent. Child call events carry the parent call ID. Distributed global concurrency, checkpointing, approval workflows and task queues belong to the harness.

## Skills

`SkillSource`, `SkillRef` and `SkillDocument` describe approved, versioned resources. `FileSkillSource` validates YAML frontmatter, file sizes, containment and file digests. `MemorySkillSource` supports host-backed documents and resources. Persona bodies are composed as untrusted content. Discoverable skills expose an index plus resource-loading tools; activation revalidates the approved manifest and required tool availability.

`TriggeredSkill` binds an approved `SkillRef` to event-kind patterns on `AgentDefinition.triggered_skills`. The host supplies event facts explicitly through `AgentRunner.run_agent(triggering_events=...)`; matching skills are pinned into that invocation and revalidated. The runner does not subscribe itself to arbitrary event streams. Event-triggered scheduling belongs to an explicit host subscription: resolve approved references and invoke the runner. There is no implicit permission to execute scripts because an event or skill body requests it. Skills do not grant tools or expand delegation policy.

## Events and internal reactions

```python
from agentkit.models.contracts.types import CallEvent
from agentkit.models.contracts.constants import EventKind
from agentkit.core.events import EventHub

async def host_reaction(event):
    # Record evidence, schedule follow-up work, or run an approved agent.
    # Keep the handler bounded; a durable scheduler can accept long jobs.
    if event.kind == EventKind.TOOL_RESULT:
        await hub.publish_reaction(event, "harness.evidence_ready")

hub = EventHub(handler_timeout_s=5, queue_size=128)
unsubscribe = hub.subscribe("evidence", host_reaction, kinds=(EventKind.TOOL_RESULT,))
# Pass events=hub to a CallRuntime invocation, then await hub.drain()/aclose().
```

`CallEvent` carries a schema version, call ID, sequence, timestamp, optional parent ID and frozen fields. `EventKind` exports common lifecycle kinds and custom host kinds remain possible. Use `publish_reaction(parent_event, kind, fields)` for custom reactions: it creates a distinct child stream and preserves `parent_call_id`. An optional stable `reaction_id` supports host idempotency. Automatic sequences are rejected inside an explicitly sequenced runtime stream, so a reaction cannot consume the runtime's next event ID. Runtime calls assign ordered IDs; `EventHub` assigns IDs to custom publications with sequence zero, rejects decreasing/duplicate sequences, and fills missing timestamps. Call IDs must be unique across distinct invocations.

`EventHub` dispatches subscribed handlers in publication order with a bounded queue and handler timeout. A handler's follow-up publication, including publications from descendant adapter/tool tasks, is queued, avoiding recursive dispatch. Handlers can await child runtime calls using distinct call IDs and the same hub; queued child events are delivered after the current handler finishes. Ordinary publication awaits handler acknowledgement; nested publication failures surface through `drain`. A full queue raises `EventDeliveryError`. One failed subscriber does not prevent other subscribers from receiving the event.

An optional `EventJournal` persists before delivery; an append failure prevents delivery. `SQLiteEventJournal` is a concrete local durable adapter with ordered per-call reads and exact-event idempotency keyed by `event_id`. Conflicting content under an existing ID is rejected. A host replays pending work through its own scheduler and makes subscriber effects idempotent. The in-memory hub does not restore sequence allocation or subscriptions after restart. `EventPublisher` is the broker integration port. This is not distributed exactly-once delivery.

`BusEventSink` bridges call events into the existing `EventBus` with bounded fan-out and ring replay; the host explicitly closes streams unless `close_on_finish=True`. Slow-consumer behavior follows the existing bus contract. `TraceEventSink` bridges events into `TracePort` spans and records normalized usage and cost without requiring an OpenTelemetry SDK.

## Durable event records and worker admission

```python
from agentkit.core.events import EventHub
from agentkit.state.events import SQLiteEventJournal

async def publish_recorded_event(database_path, handler, event):
    journal = SQLiteEventJournal(database_path)
    async with EventHub(journal=journal) as hub:
        hub.subscribe("host-consumer", handler)
        await hub.publish(event)

async def replay_call(journal, call_id, cursor, handler, save_cursor):
    for event in await journal.read(call_id=call_id, after_sequence=cursor):
        await handler(event)
        await save_cursor(call_id, event.sequence)
```

Persist subscriber cursors separately and advance them after the effect is acknowledged. A crash between the effect and cursor update can redeliver an event; the effect must deduplicate by event ID or use a host transaction. Journal replay does not restore hub subscriptions or sequence allocation. Resume an explicitly sequenced stream from its stored sequence, or start a fresh call ID. `SQLiteEventJournal` uses SQLite WAL and FULL synchronization on a local filesystem; the acceptance tests cover real subprocess restart and journal-before-delivery ordering, not hardware power-loss or multi-host storage behavior. Journal retention is host-owned.

`SQLiteLeaseStore` implements the host `LeaseStore` admission port on a local SQLite database shared by processes. It does not replace `AgentRunner`'s local semaphore automatically.

```python
from agentkit.state.leases import SQLiteLeaseStore

async def admitted_work(database_path, owner, work):
    leases = SQLiteLeaseStore(database_path)
    lease = await leases.acquire("host-workers", owner, ttl_s=30, capacity=2)
    if lease is None:
        raise RuntimeError("host worker capacity exhausted")
    try:
        # work must enforce lease.fencing_token at protected side-effect commits.
        return await work(lease)
    finally:
        await leases.release(lease)
```

For work lasting longer than its TTL, the host calls `renew(lease, ttl_s=...)` and stops protected work when renewal returns `None`. Renewal preserves the fencing token; replacement admission obtains a higher token. Downstream transactions must reject stale tokens. A lease cannot stop an expired worker by itself. A crashed owner occupies its slot until expiry; cancellation while a thread-backed acquisition commits can also leave a slot until expiry. Capacity is fixed per resource. The adapter is local multi-process admission, not a multi-host lease service or exactly-once execution; clock behavior and fencing enforcement remain host responsibilities.

External MCP credentials follow the same ownership boundary. `ServerConfig.token` is a host callable resolved for each request; an async resolver can obtain rotated credentials. Controlled HTTP acceptance tests exercise refresh-token exchange, bearer rotation, expired-token failure and approval revocation. The gateway does not implement OAuth login, expiry scheduling or refresh-on-401. Public HTTPS/address validation remains the default; private endpoints require an explicit narrowly approved host client factory.

## Architecture boundaries still to track

The `Agent`/`Cognition` stack remains available alongside the model-neutral call/mind contracts. Both now share dated pricing, canonical schema validation/compilation and accounting provenance; result shapes and orchestration interfaces remain distinct. Registry routing and accepted parameter profiles have complementary roles. There is no automatic durable workflow engine, global delegation ledger, persistent agent store, OAuth login service, or sandbox manager. A production harness should supply these facilities and validate its chosen provider/CLI versions against its own acceptance cases.

## Open-source architecture comparison

The reviewed projects provide useful boundaries without defining agentkit's application behavior:

| Primary source | Boundary observed | Agentkit design choice |
| --- | --- | --- |
| [OpenAI Agents SDK model interface](https://openai.github.io/openai-agents-python/ref/models/interface/) and [context management](https://openai.github.io/openai-agents-python/context/) | Model implementations have a common interface; local application dependencies are distinct from information sent to models. | Keep transport behind `ModeAdapter`; attach canonical conversation content explicitly and retain host dependencies in the runner/services. |
| [Pydantic AI models and providers](https://pydantic.dev/docs/ai/models/overview/) | Model protocol, provider authentication/connection and capability profiles are separate concerns. | Keep host credential resolution, transport binding and capability validation separate; report unsupported settings before dispatch. |
| [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | Thread checkpoints and cross-thread stores have different scopes; in-memory persistence does not survive a restart. | Distinguish call session/context snapshots from durable host agent stores and event journals; leave scheduling and restart recovery to the harness. |

These are design comparisons, not a claim that agentkit implements all of those frameworks' features or has been benchmarked at their deployment scale. A future durable harness can build on these ports without making the low-level library depend on a particular database, broker or graph engine.
