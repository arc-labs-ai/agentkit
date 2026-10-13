# One path for building a harness

Use `AgentDefinition` and `AgentRunner` for a harness that needs interchangeable API and CLI models. A definition describes permissions and limits; the runner executes it. `Agent` and `Cognition` remain available for the separate middleware/coordinator interface. Their result/accounting contracts differ, so do not combine their usage records or assume their settings are interchangeable.

The [support example](https://github.com/arc-labs-ai/agentkit/blob/main/tests/examples/support_harness/application.py) demonstrates the complete public composition path. Its contract tests cover evidence retrieval, actual tool execution, denied cross-organization order access, missing evidence, explicit note writes and independent context copies. This is a usage example; live application acceptance belongs to the evaluation suites.

## Ownership and public imports

Each owner root contains only `__init__.py`. These initializers collect public exports; named implementation modules live inside responsibility subpackages. Import supported symbols from the owner shown below. Names prefixed with `_` are internal implementation details.

| Need | Public entry point | Responsibility |
| --- | --- | --- |
| Define/run an agent | `agentkit.agents` | `AgentDefinition`, `InstructionSet`, `AgentRunner`, `ToolGrant` |
| Choose reasoning behavior | `agentkit.agents.minds` | Single-call, ReAct, deliberate reasoning; independent of transport |
| Bind a model | `agentkit.models` | `Binding`, `CredentialRef`, `ModelSettings`, `CallLimits` |
| Run/close model resources | `agentkit.core.runtime` | `CallRuntime`, retry and bounded execution |
| Direct model APIs | `agentkit.models.providers.frontiers` | HTTP clients and API transport |
| Installed coding clients | `agentkit.models.providers.cli` | Claude/Codex drivers and isolated homes |
| Declare/grant tools | `agentkit.tools` | `FunctionTool`, `tool`, `ToolRegistry`; grants on the agent definition |
| Conversation state | `agentkit.state.context` | `WorkingContext`, `PrefixContext`, slicing, forks, snapshots |
| External recall | `agentkit.state.memory` | `MemorySource`, `MemoryItem`, source/decorator implementations |
| Instructions/templates | `agentkit.agents.prompts` | Versioned `Prompt`; composition stays separate from model payloads |
| Observe/react | `agentkit.core.events` | `CallEvent`, `EventHub`, `EventBus`, journal and bridge ports |
| Persist/recover | `agentkit.state.checkpoints` | Checkpoint contracts; storage adapters in its `adapters` package |
| Persistent file memory | `agentkit.tools.filesystem` | `StoreFiles` over a host store, usable by `FileMemory` and `FileTool` |
| Durable event records | `agentkit.state.events` | `SQLiteEventJournal`; subscriber cursors and restart scheduling remain host-owned |
| Host worker admission | `agentkit.state.leases` | `SQLiteLeaseStore`, `Lease`, `LeaseStore`; explicit renewal and downstream fencing |

`CallRuntime` owns transport resources and enforces model-call limits. `RunContext` carries host identity/services/middleware meters for memory and the coordinator interface. `WorkingContext` contains information used during reasoning. These are different responsibilities; passing a `RunContext` where a `WorkingContext` is required is an error.

## The execution sequence

1. The host authenticates the caller and selects its organization-scoped data and credentials.
2. Query a `MemorySource`, inspect/admit its records, and attach selected evidence to `WorkingContext.prefix.grounding`. User-role evidence blocks keep recalled content distinct from system instructions.
3. Register typed tools and grant only the permitted tool names through `ToolGrant`. Tool implementations enforce record-level authorization too.
4. Execute `AgentRunner.run_agent` with a definition, instructions, binding, working context, unique call ID, optional cancellation token and event sink.
5. Inspect `Answer.outcome`, failures, degradations and accounting. Retain canonical context and the returned session for continuation.
6. Save memory or business results only through an explicit host decision or an authorized tool. A model-authored summary is not automatically trusted evidence.

Run the example after installing the `calls` extra and configuring a host credential:

```bash
python -m tests.examples.support_harness --mode openai_api --model gpt-5.4-mini
```

The sample backend is in-memory. Durable storage, tenant enforcement, approval persistence and retention policies must be supplied by the deployed harness.

## Context operations

```python
from agentkit.core.contracts.types import Message, MessageRole
from agentkit.state.context import LastNTurns, WorkingContext

context = WorkingContext()
context.append(Message(MessageRole.USER, "Observed duplicate payment; no refund issued."))
context.note("review_state", "pending")
branch = context.fork()
branch.note("review_state", "approved")
changes = context.diff(branch)
recent = context.slice(LastNTurns(4))
snapshot = context.freeze()
```

`append` and `extend` update the transcript. `note` stores host scratchpad state; scratchpad notes are not automatically sent to the model. `fork` creates an independent branch. `slice` selects a transcript view. `merge` uses `concat` by default, including last-write-wins scratchpad updates; `union` deduplicates messages. Review conflicting branches before merging them. `freeze` produces an immutable snapshot. Use a separate context per conversation and its lock when coordinating concurrent mutations.

Changing the context, model or provider can require canonical replay instead of opaque native continuation. The runtime records that degradation. A session handle is portable conversation state, not a durable job scheduler.

## Memory operations

`MemorySource.query(query, k=..., ctx=..., where=...)` retrieves records. `write(items, ctx=...)` stores explicit records. `ReadOnlyMemory` refuses writes; `ScopedMemory` supplies scope filtering to compatible backends. Backend enforcement is still necessary: a filter is not a universal access-control service.

The common protocol has no generic delete/update/transaction operation. Use a backend's documented operation or `StorePort` where its semantics fit. Do not invent universal retention guarantees. The support example keeps approved policies and model notes under separate paths and recalls only approved policy paths.

## Persistent file memory

`StoreFiles` supplies a bounded virtual file tree over a `StorePort`. A `FileStore` preserves its JSON tree across process restarts. The namespace is selected by the authenticated host, not by model tool arguments. Virtual paths remain beneath `/memories`; they are not arbitrary operating-system paths.

```python
from agentkit.core.contracts.types import Scope
from agentkit.core.runtime import RunContext
from agentkit.state.memory import FileMemory
from agentkit.state.stores.file import FileStore
from agentkit.tools.filesystem import StoreFiles

async def file_memory_example(data_dir):
    store = FileStore(data_dir)
    files = StoreFiles(store, namespace="approved-host-tenant")
    memory = FileMemory(files)
    ctx = RunContext("memory-example", Scope())
    await files.create("/memories/policy.txt", "Duplicate charge: request review.")
    await files.str_replace("/memories/policy.txt", "request review", "request supervisor review")
    await files.rename("/memories/policy.txt", "/memories/review.txt")
    found = await memory.query("duplicate charge", k=3, ctx=ctx,
                               where={"path_prefix": "/memories"})
    await files.delete("/memories/review.txt")
    return found
```

`FileMemory.query` ranks files by case-insensitive substring occurrence count; zero matches are excluded and ties use path order. It is lexical recall, not semantic/vector retrieval. Empty queries return no records. `FileMemory.write` creates files using each item's `metadata["path"]` or a content-derived path; existing files can be refused. Use backend `create(..., overwrite=True)`, `str_replace`, `insert`, `rename` and `delete` for explicit file CRUD. These operations are not universal `MemorySource` methods or automatic retention policies.

`StoreFiles` uses bounded compare-and-set retries for each tree edit and inherits its store's durability and atomicity. Share one `FileStore` instance between concurrent users in a process. `FileStore` does not provide cross-process CAS locking; use a store with the required transaction guarantees for that deployment. Selecting a namespace does not authenticate a caller or make recalled text trusted.

## Prompts and tools

`Prompt(id, version, template, inputs=...)` declares its placeholders. Use `bind` to prepare reusable values and `render` to validate required inputs. Prompt composition owns ordering, trusted rules and manifests. Prompt text does not grant tool permissions or enforce business approval.

Tool methods exposed to models need bounded typed parameters, useful descriptions and explicit side-effect declarations. Internal helpers stay out of the registry. Changing a tool's implementation should not require copying an agent loop or provider adapter.

## Deployment acceptance

Run the source suite and the paired live evaluations for your configured models and CLI versions. Check cancellation/cleanup, denied writes, authentication rejection, safe resume, missing evidence and recovery—not just successful answers. Unknown costs remain unpriced; native CLI restrictions and distributed durability have explicit limitations in the [implementation status](../design/implementation-status.md).
