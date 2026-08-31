# Cheatsheet

Every primitive, tight. Skim in 90 seconds; grab the invocation you
need; go. This page has no prose. For the *why*, follow the links to
[Concepts](concepts/kernel.md) or [Recipes](recipes/index.md).

## Boot

```python
from agentkit import Agent, Scope, RunContext, Services
from agentkit.agents.cognition import SingleCallCognition, ReActCognition

# The composition: identity + prompt + cognition + tools.
agent = Agent(
    name="briefer",
    model="claude-sonnet-4-6",
    prompt="Answer concisely.",
    cognition=SingleCallCognition(),  # default; ReActCognition(tools=[...]) for a tool loop
)

# The per-run universe. `services=` wires the Invoker + Store + Vector + Trace + ...
ctx = RunContext(
    correlation_id="run-42",
    scope=Scope(org_id="acme"),
    services=Services(invoker=my_invoker),  # see the "Invoker" section
    autonomy="auto",  # "auto" | "gated" | "manual"
)

result = await agent.run("hello", ctx)         # collect the stream into one AgentResult
async for ev in agent.stream("hello", ctx): ...  # or handle StreamEvents live

# `Services(store=...)` is the one KV seam. Cache half: get / set / get_or_set /
# delete / append / list. Coordination half below.
from agentkit.adapters.store import InMemoryStore  # FileStore / RedisStore / PostgresStore

store = InMemoryStore()

# Read-modify-write. Returns whether it APPLIED — losing is the expected half of an
# optimistic loop, not a fault. Absent key equals `expected=None`; compared by equality.
current = await store.get("run-42:ordinal")
applied = await store.compare_and_set("run-42:ordinal", current, (current or 0) + 1)

# One atomic counter + window. `by` is a non-bool int (StoreValueError otherwise) and
# may be negative. `ttl` OPENS a window; it never slides an open one, or a rate limiter
# under sustained traffic never resets. Honoured by InMemoryStore + RedisStore, ignored
# with a one-time warning by FileStore, NotImplementedError on PostgresStore.
used = await store.increment("run-42:calls", 1, ttl=60)

# KEYS under a prefix — an AsyncIterator, no promised order, never the append-log
# namespace. `limit` is a cap on what you will receive, not a page cursor.
async for key in store.scan("run-42:", limit=100): ...
```

## Cognition

```python
from agentkit.agents.cognition import (
    SingleCallCognition,
    ReActCognition,
    CoordinatorCognition,
    ClaudeCliCognition,
)

# One chat call + optional parse-and-repair.
SingleCallCognition()

# Chat ↔ tool loop, HITL suspend/resume, durable resume via Checkpointer.
ReActCognition(tools=[my_tool, another_tool], max_iterations=8)

# Multi-agent: one coordinator + children driven by a Policy.
from agentkit.agents.policies import RoundRobinPolicy, SelectorPolicy
CoordinatorCognition(
    children={"planner": planner_agent, "researcher": researcher_agent},
    policy=RoundRobinPolicy(max_turns=6),
)

# Delegate the loop to a locally-installed `claude` CLI. No API key.
ClaudeCliCognition(
    model="claude-sonnet-4-6",
    permission_mode="acceptEdits",
    tools=("Read", "Grep", "Bash"),          # what the session HAS
    allowed_tools=("Read", "Grep"),          # ...and what runs unprompted
    max_concurrent=8,
)
# agent.prompt is APPENDED to the CLI's own system prompt (--append-system-prompt);
# system_prompt_mode="replace" opts into overriding it. Resume a prior run with
# resume_session_id=<evals["session_id"]> — session_id= NAMES a new session.
```

## Tools

```python
from agentkit import (
    tool, FunctionTool, ToolRegistry, FileTool, InMemoryFiles, ToolArgumentError,
)

# The 90% case. `side_effecting=` is REQUIRED — decoration-time
# ToolDefinitionError (a ValueError) otherwise.
@tool(side_effecting=False)
async def search(query: str) -> str:
    """Search the web for `query`. Returns two bulleted hits."""
    return "..."

@tool(side_effecting=True, idempotent=False, requires_approval=True)
async def publish(title: str, body: str) -> str:
    """Publish `title` to the wiki. Not idempotent; always ask a human first."""
    return f"published {title}"

# Composite: many tools behind one name lookup.
registry = ToolRegistry.from_tools([search, publish])

# Filesystem-backed memory tool (read/write/list under a confined root).
files_tool = FileTool(backend=InMemoryFiles(), root="/memories")

# A call is CHECKED against the signature the model was shown: an unknown
# argument name raises ToolArgumentError (naming the tool, the bad key and the
# accepted set) instead of being dropped — a dropped key let a defaulted
# parameter run with its default and report success. Declare **kwargs if the
# tool genuinely accepts arbitrary keys; then the extras are passed through.
@tool(side_effecting=False)
async def flexible(a: str, **extra: object) -> str:
    """Accept arbitrary extra keyword arguments alongside the declared one."""
    return "ok"
```

Parameter schemas are derived from the annotations: primitives, `list`/`dict`,
`Literal` and `Enum` (both become an `enum` list) and typed structs (Pydantic /
dataclass / attrs, which become a real `object` schema via the same `adapt()`
the `output=` path uses). Anything else degrades to `string`.

## Prompts

```python
from agentkit import Prompt

p = Prompt(
    id="briefer.system",
    version="1.2.0",
    template="You are a terse briefer. Cite every claim.",
    inputs=(),  # declared placeholders. With none declared, render() returns the
                # stripped template and REFUSES any kwargs; declare a name here
                # and render(name=...) substitutes `{name}` (missing or
                # unexpected keys raise ValueError, never a half-filled prompt).
)

# Wire on the Agent — the version travels on every trace + AgentResult.prompt_version.
Agent(name="briefer", prompt=p, cognition=SingleCallCognition())
```

## Memory

```python
from agentkit import (
    CompositeMemory,
    SequentialMemory,
    VectorMemory,
    ScopedMemory,
)
from agentkit.memory import CachedMemory, JournalMemory, FileMemory, ScratchpadMemory
from agentkit.memory import MemoryItem, MemoryWriteRefused, ReadOnlyMemory

# Vector store adapter (any VectorPort — pgvector, qdrant, memory-backed, ...).
memory = VectorMemory(vector=my_vector_port)

# Filesystem-backed reads.
memory = FileMemory(files=InMemoryFiles())

# `id` is the backend's own record key, and KEYWORD-ONLY — declaring it third would
# have rebound every positional MemoryItem("c", "vector", 0.91, meta) silently.
MemoryItem(content="Refunds take 5 days.", source="handbook", id="chunk-7", score=0.9)

# Fan-out across many sources, merge + rerank the top-k. Two sources returning one
# fact is the NORMAL case (the journal is what the index was built from), so the same
# fact is one item: matching `id` OR matching stripped content, higher score survives,
# stamped `dedupe_sources` + `dedupe_count`. `id=""` and blank content are not identity.
memory = CompositeMemory(sources=[VectorMemory(...), JournalMemory(...)])  # dedupe="id"
CompositeMemory(sources=[...], dedupe="content")   # backends that have no ids
CompositeMemory(sources=[...], dedupe=None)        # concatenate, chosen not inherited

# Fan-in in order (first non-empty wins).
memory = SequentialMemory(sources=[fast_cache_source, slow_vector_source])

# Decorators (compose freely).
memory = ScopedMemory(inner=memory)             # enforces ctx.scope at the boundary
memory = CachedMemory(inner=memory, ttl_seconds=60, max_entries=256)

# Read-only by POLICY, not by backend. `query` is a total pass-through — same k, same
# `where`, same items, same `source` stamp, so dedupe still merges it with the bare path.
kb = ReadOnlyMemory(inner=curated_kb)                     # write() -> MemoryWriteRefused
kb = ReadOnlyMemory(inner=curated_kb, on_write="ignore")  # dropped; kb.refused_writes counts
kb.accepts_writes                                         # False — CompositeMemory reads this

Agent(name="researcher", cognition=ReActCognition(tools=[search]), memory=memory)
```

## Middleware chain

```python
from agentkit import chain, BaseMiddleware, Call, Handler, MiddlewareContext
from agentkit.middlewares import (
    tracing, retry, meter, fallback, memoize, semantic_memoize,
    output_coerce, compaction, security, egress, audit,
)

# The canonical chat chain. Order is deterministic — outermost first.
from agentkit import SlidingWindowCompactor

chat_middleware = [
    tracing(),                                          # outermost: one span across everything below
    compaction(SlidingWindowCompactor(keep_recent=10)), # shrink transcripts BEFORE meter counts tokens
    meter(),                                            # guard/charge Budget + Quota on every call
    fallback(models=["gpt-4o", "gpt-4o-mini"]),         # rewrite request.model on hard failures
    retry(),                                            # re-invoke on transient failures + optional CircuitBreaker
    memoize(),                     # exact-match cache; every key is scope-partitioned
]

# Custom transform/guard/observe middleware.
class Redact(BaseMiddleware):
    async def on_request(self, ctx: MiddlewareContext) -> None:
        ...  # mutate ctx.request

# Custom raw resilience/caching middleware (must re-invoke, skip, or wrap `next`).
async def stopwatch(call: Call, nxt: Handler):
    async for item in nxt(call):
        yield item
```

## Capabilities

```python
from agentkit import (
    RequestBuilder,
    Grounder,
    GroundingSource,
    render_grounding,
    Checkpointer,
    Guardrail,
    Evaluator,
    SlidingWindowCompactor,
    TruncationCompactor,
    SummarizationCompactor,
    ImportanceFilteringCompactor,
)
from agentkit.adapters.checkpoint import InMemoryCheckpointStore

# Compactors (fold the transcript before it hits the model window).
compactor = SlidingWindowCompactor(keep_recent=10)       # dep-free
compactor = TruncationCompactor(max_tokens=12_000)       # dep-free
compactor = SummarizationCompactor(summarizer=my_llm)    # needs an LLMPort
compactor = ImportanceFilteringCompactor(filterer=my_llm)

# Durable snapshot/resume — powers HITL suspend AND crash-resume.
checkpointer = Checkpointer(port=InMemoryCheckpointStore())
# Production: PostgresCheckpointStore from arc-agentkit[postgres].

# The RequestBuilder assembles a ChatRequest from prompt + memory + tools.
# The Agent builds one automatically from `prompt=` — override to plug grounding.

# `grounder=` returns TEXT, so by the time it reaches the prefix the item is gone —
# and a memory a MODEL wrote is no longer distinguishable from a recorded fact.
# `grounding=` returns the MemoryItems, so provenance survives to the prompt.
from agentkit.memory import as_grounding_source

source: GroundingSource = as_grounding_source(memory, k=5)  # (ctx, task) -> Sequence[MemoryItem]
builder = RequestBuilder(
    prompt=my_prompt,
    grounding=source,
    admit=lambda i: i.metadata.get("tier") != "inferred",  # veto, BEFORE render
    render=render_grounding,        # the default: one `[source] content` line per item
    record_grounding=True,          # admitted items -> wc.scratchpad["grounding"], as
)                                   # JSON-safe dicts so the record crosses a checkpoint
# `grounder=` and `grounding=` together is a ValueError, and so is admit / render /
# record_grounding without `grounding=` — a veto that silently applies to nothing is
# the exact failure `admit` exists to prevent.
```

## Concurrency

```python
from agentkit import (
    CancellationToken, Cancelled,
    gather_bounded, gather_best_effort,
    run_agents, run_sync,
    attempt_until_stuck, Stuck,
)
from agentkit.agents.workflow import MapExpansionChanged, Workflow
import asyncio

# Fan-out N children, bounded by a semaphore. One failure cancels the rest.
sem = asyncio.Semaphore(4)
results = await gather_bounded([coro1(), coro2(), coro3()], sem=sem)

# Same, but isolate failures into Failure objects (no sibling cancel).
results = await gather_best_effort([coro1(), coro2()], sem=sem)

# Multi-agent: each pair runs under ctx.child(), sharing budget + cancel.
results = await run_agents([(agent_a, "task-a"), (agent_b, "task-b")], ctx)

# A DAG node whose fan-out width is decided at RUNTIME. Elements run one level deeper
# under gather_bounded, so they cannot deadlock on the wave's own permits; `bounded_by`
# narrows on top of `max_concurrency`, it never widens. One map = one `max_steps`.
wf = Workflow("release")
wf.fn("plan", make_plan)
wf.map("implement", over=lambda done: done["plan"].requirements,  # decided at runtime
       each=lambda item: agent_for(item), after="plan", bounded_by=4)
result = await wf.run("ship it", ctx)
result.outputs["implement"]             # element outputs, in expansion order
result.outputs["implement#expansion"]   # the identity list, written BEFORE any element
# A resume whose `over=` returns a different collection raises MapExpansionChanged
# rather than threading element 2's output into element 0's slot. `key=` when `str(item)`
# embeds an address. A best_effort map's Failures cannot cross a downstream human_gate.

# Retry bounded by RECURRENCE, not by a count: three different failures is progress,
# two identical ones is not. Every signature seen is compared, not just the previous
# one, so an A,B,A,B oscillation is caught. `fingerprint` returning None IS success.
answer = await attempt_until_stuck(
    lambda: run_one(),
    fingerprint=lambda outcome: outcome.failure_signature,
    on_repeat="escalate",   # raise Stuck(failure); "stop" RETURNS the identical Failure
    max_attempts=4,         # a BACKSTOP — exhaustion returns a Failure in both modes
)
# Exceptions are never fingerprinted — that is `run_with_resilience`'s job. Nest them:
# attempt_until_stuck(lambda: run_with_resilience(call_model, max_attempts=3), ...)

# Cooperative cancel across the whole subtree.
token = CancellationToken()
ctx.cancel = token
token.cancel()  # every check_cancelled() from now on raises Cancelled

# Sync host driving async agentkit — one bridge.
result = run_sync(agent.run("hi", ctx))
```

## Signals + control (multi-agent)

```python
from agentkit import (
    Suspended, Handoff, DoneSignal, ProgressSignal, EscalateSignal,
    CancelSignal, BudgetReducedSignal, RedirectSignal, ContextUpdateSignal,
    MergeWithPeerSignal, BlockedSignal, SignalChannel, SignalEnvelope,
    ActorBudget, BudgetExhausted,
)

# Frozen result value returned when a run pauses for approval.
assert isinstance(result.evals.get("suspended"), Suspended)
decisions = {tc.id: "approve" for tc in result.evals["suspended"].pending}
final = await agent.resume(result.evals["suspended"].run_id, decisions, ctx)

# Handoff: route to a peer. Consumed by SelectorPolicy / handoff-routing.
signal = Handoff(target="specialist", reason="user asked about billing")

# ActorBudget: per-child slice of the parent's envelope.
parent_budget = ActorBudget(max_tokens=10_000, max_cost_usd=1.0,
                             max_steps=20, max_wall_seconds=60.0)
```

## Budget + Quota + Meter

```python
from agentkit import Budget, Quota, MeterExceeded
from agentkit.middlewares import meter

# Per-run ceiling. Enforced by meter() under an async lock.
budget = Budget(
    max_cost_usd=0.50,
    max_calls=50,
    max_depth=4,
    max_concurrency=8,
)

# Per-tenant rolling window, keyed by Scope.
quota = Quota(max_rpm=60, max_tpm=100_000, max_usd=1.0, window=60.0)

# Wire both onto RunContext.meters (Budget goes on RunContext.budget directly).
ctx = RunContext(
    correlation_id="run-42",
    scope=Scope(org_id="acme"),
    budget=budget,
    meters=[quota],
    services=Services(invoker=Invoker(llm=llm, chat_middleware=[meter()])),
)

try:
    await agent.run(task, ctx)
except MeterExceeded as exc:
    ...  # the run halted cleanly at the ceiling

# Money is Decimal. `spent_usd` is a float MIRROR, fine for display.
budget.spent()         # Decimal("0.4831") — exact, reconciles to the cent
budget.spent_cents()   # Decimal("0.48")   — quantize once, at read time
budget.usage           # cumulative Usage: input/output/cache tokens, whole tree

# Recoverable exhaustion: return a verdict instead of raising, so the
# cognition can write a checkpoint BEFORE it stops.
budget = Budget(max_cost_usd="1.00", on_exceeded="stop")
result = await agent.run(task, ctx)
if result.stop_reason == "budget_exhausted":
    assert result.is_resumable      # a current checkpoint exists
```

## Provider from the environment + model capabilities

```python
from agentkit.adapters.llm import (
    Capability, ModelCapabilities, ModelEntry,
    model_capabilities, register_model, register_rule, resolve_llm,
)
from agentkit.client import from_env

# Provider picked from the model name; credential read from the env.
# Raises ProviderNotConfigured if unkeyed — pass fallback="fake" to degrade.
llm = resolve_llm("claude-sonnet-4-6")          # -> LLMPort
async with from_env("gpt-4o-mini") as chat:      # -> Chat
    ...

# Capabilities are declared per model, never guessed from the name.
model_capabilities("claude-sonnet-4-6").vision   # Capability.YES
model_capabilities("who-knows").vision           # Capability.UNKNOWN (never True)

# Refused at CONSTRUCTION — before any spend.
agent = Agent("ocr", "claude-sonnet-4-6", requires=("vision",),
              min_context_window=100_000,
              on_unknown_capability="warn")       # "refuse" | "allow"

# Declare your own model / routing rule.
register_model(ModelEntry("acme-v3", provider="openai",
                          capabilities=ModelCapabilities(tools=Capability.YES)))
register_rule(lambda name: "openai" if name.startswith("acme-") else None)
```

## Batteries-included LLM presets

```python
from agentkit import claude, openai, deepseek, openrouter

# Each returns a Chat wired with `tracing → meter → retry`.
async with claude(api_key="sk-...", model="claude-sonnet-4-6") as chat:
    result = await chat("hi", system="Answer briefly.")

async with openai(api_key="sk-...", model="gpt-4o-mini") as chat:
    result = await chat("hi", system="Answer briefly.")

async with deepseek(api_key="sk-...", model="deepseek-chat") as chat:
    ...

async with openrouter(api_key="sk-...", model="anthropic/claude-3.5-sonnet") as chat:
    ...
```

Any HTTP LLM is one `LLMPort` impl away — bring your own vLLM,
Ollama, or in-house wrapper the same way the presets do.

## MCP (Model Context Protocol)

```python
# pip install "arc-agentkit[mcp]"
from agentkit.integrations.mcp import (
    MCPClient, StdioServer, StreamableHttpServer,
    mcp_tools, mcp_resources, mcp_prompts,
)

server = StdioServer(command="uvx", args=("mcp-server-time",))
# Or: StreamableHttpServer(url="https://mcp.example.com/sse")

async with MCPClient(server) as mcp:
    tools = await mcp_tools(mcp, prefix="time_")   # list[Tool] — drop into ReActCognition
    memory = mcp_resources(mcp, name="time_docs")  # MemorySource
    prompts = await mcp_prompts(mcp)               # dict[str, Prompt]

    agent = Agent(
        name="clock",
        prompt=prompts.get("system") or "Answer with the current time.",
        cognition=ReActCognition(tools=tools),
        memory=memory,
    )

# The other direction — agentkit AS the MCP server, which is how anything you know
# reaches the CLI's own loop.
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.integrations.mcp import (
    ApprovalServer, McpServerSpec, serve_registry, stdio_command,
)

# Serve a ToolRegistry you already have. Sync: it reserves the port and writes the
# config, so the spec is complete before anything listens. `async with` starts it.
spec: McpServerSpec = serve_registry(registry, name="engine", ctx=ctx, timeout_s=120.0)
async with spec:
    ClaudeCliCognition(
        model="claude-sonnet-4-6",
        allowed_tools=spec.auto_approve,          # requires_approval tools still prompt
        **spec.cli_kwargs(builtin_tools=False),   # mcp_config + strict_mcp_config=True
    )
spec.tool_names         # ('mcp__engine__deploy', ...) — the CLI's spelling
spec.caps               # feed RunPolicy().check(...) BEFORE you start the CLI
spec.calls_seen         # 0 means never called a tool OR never reached the server

# `transport="stdio"` needs `command=` — the CLI spawns a fresh process, so this
# process's registry, closures and ctx do not cross the boundary.
serve_registry(registry, name="engine", ctx=ctx,
               transport="stdio", command=stdio_command("myapp.mcp_server"))

# HTTP binds 127.0.0.1 AND requires a generated bearer token by default: loopback
# alone was the containment, and the point of the CLI is that it runs Bash in that
# same namespace. No caller-supplied tokens — one you can set becomes a constant.
spec.auth_headers       # {"Authorization": "Bearer ..."} for a non-CLI client
serve_registry(registry, name="engine", ctx=ctx, auth="none")   # opt out BY NAME
# transport="stdio" + auth="bearer" RAISES: there is no request to carry a header.

# CLI permission prompts -> your Asker. The turn parks while a human thinks.
async with ApprovalServer(asker=my_asker, timeout_s=120.0, auto_allow=("Read",)) as ap:
    ClaudeCliCognition(model="claude-sonnet-4-6", **ap.cli_kwargs())
for d in ap.decisions:  # one ApprovalDecision per prompt, oldest first
    d.tool, d.allowed, d.reason, d.at
    d.source   # asker | auto_allow | autonomy | timeout | error — a bool collapses
               # "a person said yes" into "the tier did not gate this call"
    d.asked    # separate from source=="asker": a prompt can reach a human and expire
```

## Claude CLI seams (the loop is not yours)

```python
from agentkit.agents import RunPolicy
from agentkit.agents.cognition import ClaudeCliCognition
from agentkit.integrations.claude_cli import (
    HookSettings, SkillNotProjectable, as_cli_agents, hook_settings,
)

# The CLI's OWN tools (Write, Edit, Bash, WebFetch) bypass the Invoker, so egress,
# guard, audit, memoize and Guardrail.check_url do not run. The cognition warns once
# per instance, NAMING the middlewares that will not apply. Three ways to close it:
# serve everything over MCP, generate hooks (below), or accept it knowingly.

# A CLI session declares capability tags, so RunPolicy can see its trifecta.
ClaudeCliCognition(tools=("Read",)).caps             # ('private_data',)
ClaudeCliCognition(tools=("Read", "WebFetch")).caps  # full trifecta — WebFetch is two legs
RunPolicy().check(entries)                           # reads `.caps`; a session is ONE entry
# tools=None (the DEFAULT) is every built-in tool, i.e. the trifecta. Task and
# SlashCommand each count as all three: they are indirections to the whole tool set.

# Make the chain reach native tools — a PreToolUse hook answered by THIS process.
# Call it from inside a running loop; it returns already listening.
settings: HookSettings = hook_settings(
    middleware=tool_chain,          # the same list you hand Invoker(tool_middleware=...)
    ctx=ctx,
    tools=("Write", "Edit", "Bash"),  # matcher is ANCHORED: ^(Write|Edit|Bash)$
    timeout_s=5.0,                  # inner deadlines must fire first — the CLI's own
)                                   # hook timeout is NON-blocking, i.e. fail-open
async with settings:
    ClaudeCliCognition(settings=settings.path)
settings.decisions   # what the hook allowed/denied, for the audit
# Unparseable payload = deny, never pass. A middleware needing a live ctx (Invoker,
# store handle, cancel token) is refused at GENERATION time, not silently skipped.

# Project a Skill into a CLI sub-agent. The tool restriction survives — a reviewer
# that is read-only because of its tool list must not arrive holding the parent's.
ClaudeCliCognition(agents=as_cli_agents([reviewer_skill, repairer_skill]))
# A skill carrying custom agentkit tools raises SkillNotProjectable, by name, at
# construction: serve them with serve_registry and the sub-agent reaches them as
# mcp__<server>__<tool>.
```

## Structured output

```python
from pydantic import BaseModel
from agentkit import adapt

class Answer(BaseModel):
    summary: str
    confidence: float

# Output schema is built once; its JSON Schema goes into the prompt prefix
# and output_coerce enforces the shape on the way out. Repairs on parse-fail.
agent = Agent(
    name="briefer",
    model="gpt-4o-mini",
    prompt="Give a summary and a confidence in [0, 1].",
    output=Answer,             # Pydantic / attrs / dataclass / raw JSON Schema dict
    max_repairs=1,
)
result = await agent.run("brief on octopus cognition", ctx)
result.parsed  # -> Answer(summary=..., confidence=...)

# Stream the object AS IT IS WRITTEN. Needs output_coerce() in the chat
# chain (the Agent warns once if it's missing).
async for ev in agent.stream(task, ctx):
    p = ev.partial_output          # None for unstructured runs
    if p is not None and "summary" in p.model_fields_set:
        render(p.summary)          # required fields MAY BE UNSET — always gate

# Or on a Tool:
@tool(side_effecting=False, output_schema=Answer)
async def brief(topic: str) -> Answer:
    """Return a structured brief for the topic."""
    return Answer(summary="...", confidence=0.7)
```

## OpenTelemetry

```python
# pip install "arc-agentkit[observability]"
from agentkit.adapters.observability import (
    otel_tracer, otel_meter, otel_sampler,
    otel_exporter_otlp_http, otel_metrics_exporter_otlp_http,
)

# At process startup — reads OTEL_EXPORTER_OTLP_ENDPOINT etc. from env.
otel_exporter_otlp_http()
otel_metrics_exporter_otlp_http(interval_ms=15_000)

services = Services(
    invoker=my_invoker,
    trace=otel_tracer(),      # pulls the global TracerProvider
    metrics=otel_meter(),     # pulls the global MeterProvider
    sampler=otel_sampler(0.1),
)
```

## Testing kit — for your tests, not for production

```python
# The doubles the framework's own suite uses. NEVER wire these into a
# real run: they're for your unit tests and any mock adapters you write.
from agentkit.testing import (
    FakeLLM, FakeFetch, FakeSearch, FakeMemory, FakeTool, FakeClock,
    FakeGrounder, FakeCompactor, FakeCtx, RecordingTracer, Turn,
    FakeClaudeCli, make_test_ctx,
)
from agentkit import ToolCall
from agentkit.agents.cognition import ClaudeCliCognition

# One reply.
llm = FakeLLM("42")

# Scripted multi-turn (tool_call → tool_result → final).
llm = FakeLLM.script([
    Turn(tool_calls=(ToolCall("c1", "search", {"query": "octopus"}),)),
    Turn(content="Octopuses use tools."),
])

# Real RunContext wired around any fake seams (only the LLM is fake).
ctx = make_test_ctx(
    llm=llm,
    autonomy="gated",
    correlation_id="test-42",
)

# The `claude` CLI path, offline and free. Sits at the SPAWN seam, not above it, so
# the real stream-json parsing, budget charging and event mapping still run — which
# is where every bug on this path has lived. Raw `bytes` payloads express a line
# that is not valid UTF-8; that shape used to bill a completed run at $0.00.
cli = FakeClaudeCli.script([{"type": "result", "subtype": "success", "result": "done"}])
cognition = ClaudeCliCognition(spawn=cli)   # .replay(path) replays a recorded session
cli.spawns          # spawn count. One past the recording raises ScriptExhausted —
                    # pass repeat_last=True when the unbounded loop IS the test.
cli.invocations     # argv / cwd / env / stdin per spawn, no subprocess patch needed
# Reading a line never awaits: wait_for() cannot fire mid-stream and gather() over
# two drives runs them in series. Cancel through ctx.check_cancelled().
```

Test doubles live under `agentkit.testing.*` on purpose — a
`from agentkit import FakeLLM` shape would let production code
accidentally pin a fake. The import boundary is the guardrail.

## Human-in-the-loop (elicit a value, park, deadline)

```python
from agentkit.agents.control.elicitation import (
    Decision, Elicitation, ask_human_tool, elicit,
)

class MyAsker:                       # terminal / HTTP / queue — runtime doesn't care
    async def ask(self, request: Elicitation) -> Decision:
        return Decision(kind="value", value=await prompt_user(request.prompt),
                        actor="alice@corp")

# Wired on Services -> a gated tool call PARKS in place (live state survives)
# instead of checkpointing and unwinding. Unset -> classic suspend/resume.
services = Services(invoker=invoker, asker=MyAsker())

# Ask from ANY cognition — no tool call needed.
d = await elicit(ctx, Elicitation(id="otp", prompt="code?", kind="value",
                                  secret=True, deadline_s=120))
d.kind      # "value" | "approve" | "deny" | "modify" | "expired"
d.actor, d.at                        # who answered, and when
d.value                              # SecretValue when secret=True; .reveal() to read

# Let the MODEL ask.
ReActCognition(tools=[ask_human_tool(secret=True)], approval_deadline_s=120)

# Terminal state is a closed Literal, so suspended != failed.
result.stop_reason   # complete | suspended | expired | budget_exhausted
                     # | max_iterations | invalid_output | terminated | failed
                     # `failed` only from ClaudeCliCognition, which reports the
                     # error as data so its terminal event still fires.
result.is_suspended, result.is_resumable
```
