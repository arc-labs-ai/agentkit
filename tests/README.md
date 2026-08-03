# agentkit tests

The test tree mirrors the production package layout under `agentkit/`.
When you're about to touch code, look at the same-named test folder —
that's where you'll find the invariants for the module you're editing,
and where any new test you write belongs.

## Layout

```
tests/
├── kernel/         primitives — value types, resilience, streams, ports, middleware contract
├── runtime/        RunContext · Invoker · Budget/Quota · EventBus · cancellation
├── middlewares/    tracing · output_coerce · hooks — the chat/tool chain middlewares
├── capabilities/   Compactor · Checkpointer · RequestBuilder · output_schema adapters
├── context/        WorkingContext — the in-flight reasoning-state seam
├── tools/          Tool Protocol · FunctionTool · ToolRegistry · tool conventions
├── agents/         Agent · Cognition (single-call / ReAct / Coordinator) · Policy · signals · handoff · Workflow
├── memory/         MemorySource · Cached · Scoped · Compacted · Vector · File · Journal · Scratchpad · Tool
├── skills/         Skill Facade (prompt + cognition + memory + model)
├── adapters/       concrete Port impls — LLM providers · stores · replay
├── observability/  tracing spans · metrics · rollups · trace linkage · policy-dispatch events
├── meta/           packaging · public-API surface · non-functional / cross-cutting
└── testing/        the framework's own test doubles (FakeLLM, FakeCtx, …)
```

## Test-file distribution (at a glance)

Rough counts — run `pytest tests/<dir> --collect-only -q` for exact numbers.

| Folder | Files | Tests |
|---|---:|---:|
| kernel | 11 | ~178 |
| agents | 14 | ~157 |
| capabilities | 12 | ~124 |
| memory | 10 | ~65 |
| observability | 15 | ~64 |
| runtime | 7 | ~58 |
| adapters | 8 | ~56 |
| tools | 5 | ~47 |
| context | 5 | ~44 |
| meta | 4 | ~38 |
| middlewares | 4 | ~16 |
| testing | 1 | ~12 |
| skills | 1 | ~9 |

The distribution reflects where the framework's complexity lives —
kernel + agents + capabilities carry most of the invariants.

## Adding a new test

1. Find the production module you're testing (e.g., `agentkit/agents/cognition/react.py`).
2. Open the mirror folder (`tests/agents/`).
3. Extend the existing file that covers that module (`tests/agents/test_agent_loop.py`
   for the ReAct loop) — or create a new file if the concern doesn't fit.
4. Prefer property + adversarial coverage where the invariant benefits (see
   `tests/kernel/test_kernel.py` for Hypothesis-based examples).

## Related docs

- `docs/mental-models/` — narrative use cases with invariant tables. Each
  invariant names the test file that locks it in. Read these first
  when auditing a subsystem for correctness.
- `ARCHITECTURE.md` — the framework's shape at 10,000 ft.
