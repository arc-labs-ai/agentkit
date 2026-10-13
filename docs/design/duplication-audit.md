# Duplication audit and architecture ownership

The implementation now has six public namespaces: **core, models, agents, state, tools, integrations**. Folder reduction is a physical ownership change; it does not justify combining behaviors with different delivery, validation or accounting contracts. The user authorized a breaking migration, so old aggregate namespaces were removed and have no import wrappers.

## Dependency direction and owners

| Namespace | Owns | Allowed dependency direction |
| --- | --- | --- |
| `core` | Foundational messages/results, ports/codecs, concurrency/guardrails, runtime enforcement and event delivery | Primitive modules remain independent of providers. `core.runtime` sits above model/state/tool ports and lazily loads concrete transports; the entire core namespace is not a foundation-only layer. |
| `models` | Model bindings/settings; `providers.frontiers` HTTP and `providers.cli` native transports; wire schemas, authentication, token decoding, versioned pricing and capability data | Shared model contracts depend on `core`; provider I/O dependencies load only where required. Native CLI composition also uses agent/cognition lifecycle contracts. |
| `agents` | Agent definitions/stores, instructions, prompts/skills, reasoning minds, bounded delegation and orchestration | `core`, `models`, `state`, `tools`, host integration ports |
| `state` | Working/frozen context, checkpoints, memory and retention boundaries | `core`; stores implement ports. Compaction currently imports agent prompt templates, and checkpoint rehydration lazily imports typed agent results; these specific couplings remain visible migration boundaries. |
| `tools` | Registry, execution policy, result/image/overflow handling and host-approved tools | `core` and model schema compilation for function tools; integrations adapt its contract |
| `integrations` | MCP clients/servers and external-service/observability adapters | `core` and the specific model/tool/state ports being implemented; no application-specific behavior |

Generic message/port enums belong to core vocabulary; model-specific enums belong to models. Vendor credential selection belongs to model transport policy. Authentication helpers must not import agent execution solely to access an enum. A provider factory must remain usable independently of an agent harness.

## Concrete consolidation

| Repeated responsibility | Canonical owner | Consolidation and evidence |
| --- | --- | --- |
| Anthropic API-key versus bearer selection | `models/providers/auth.py` | One `AnthropicCredential`/`anthropic_credential` policy now supplies legacy chat, legacy streaming and unified API/CLI paths. The legacy provider previously always emitted `x-api-key`, which failed for bearer credentials. Tests compare real mock-HTTP requests across all three transport paths, explicit bearer precedence, OAuth recognition, custom version/header preservation, redacted representation and dependency boundaries. |
| API endpoint defaults | `models/providers/endpoints.py` | Provider factories and unified HTTP adapters reference `ApiEndpoint`/`ApiPath`, preserving custom endpoint support. |
| Canonical message wire encoding | `core/serialization/messages.py` | Checkpoint and memoization paths reuse one message serializer/deserializer rather than independently dropping metadata, multimodal blocks or nested tool arguments. JSON wire payloads are thawed; canonical values remain immutable. |
| Versioned provider prices | `models/catalog/pricing.py` | Compact provider accounting and model-neutral calls share the same dated table; compact usage conversion refuses unknown prices and unresolved cache TTL instead of inventing a rate. |
| Native CLI terminal classification | `models/providers/cli/_common.py` | Shared process-exit/terminal-stop functions and schema-stderr markers replace repeated predicates. Shared terminal classification is used by the separate lifecycle-specific drivers. |
| Hosted tool execution/MCP serving | Tool registry and result policy, adapted by integrations | The unified tool path wraps the existing registry and MCP server. Error/image/overflow policy is shared rather than maintained separately for CLI and API modes. |
| Call events into host bus/trace | Event sink bridges | Handler dispatch publishes through explicit bus/trace ports. It does not implement a second incompatible broadcast broker. |

The authentication regression tests are behavioral parity checks: they verify headers actually emitted by chat, streaming and unified calls, not merely that one helper delegates to another. Optional HTTP extras remain required by HTTP provider classes; the credential helper itself imports without HTTP or provider SDK dependencies.

## Similar names with different semantics

These are explicit migration boundaries, not candidates for blind aliasing:

- **Usage and pricing:** compact cognition results retain floating-point estimates and two aggregate cache buckets, while model-neutral results retain Decimal accounting and disjoint five-minute/one-hour cache writes. Both now use the dated `models.catalog.pricing.PriceTable`; the compact provider adapter owns no rates and does not strip model prefixes or dates. Unknown rates or ambiguous cache-write TTLs raise `PriceUnavailableError` instead of becoming a free estimate. Both usage contracts represent unknown costs as `None` with the same public `CostSource` enum. Known-plus-unknown aggregation remains unknown; Budget/Quota expose `cost_complete` while retaining their exact known Decimal lower bound and refusing money-bounded continuation. Sharing rates does not make the result value types interchangeable; conversion must preserve token conventions and explicit uncertainty.
- **Structured output:** the legacy Anthropic provider translates `response_format` into best-effort prompt guidance with a warning. Both agent execution paths use the shared canonical JSON Schema validator and compiler, including optional-null cleanup and original-raw diagnostics. Unified calls compile supported wire dialects and validate against canonical JSON Schema. Combining these behind the same name would imply provider enforcement that the legacy method does not supply.
- **Failure classification:** legacy resilience is exception/message based, with configurable classification. Unified calls classify structured HTTP/provider facts and distinguish billing exhaustion, authentication, refusal and transient failures. A clean migration can share structured facts, then make the new taxonomy authoritative; retaining both predicates without a documented boundary is remaining policy duplication.
- **Capability data:** the existing model registry resolves factories and broad model capabilities; unified setting profiles decide accepted parameter values and explicit degradations. These are complementary roles. Shared provider/model identity data is a future consolidation point; converting UNKNOWN into unsupported/supported would change policy.
- **OpenAI protocols:** Chat Completions and Responses have different request/output items, tool continuation and reasoning preservation. Share HTTP/SSE transport and canonical content primitives; keep protocol translators separate.
- **Claude/Codex lifecycles:** persistent Claude control sessions and Codex thread/respawn behavior differ. Both reuse CLI process infrastructure and message-boundary tracking; one combined state machine would hide lifecycle requirements.
- **MCP clients:** the older text-oriented wrapper emits omission markers; the newer gateway preserves rich results and validates endpoint identity, approval digests and public-network boundaries. Migration requires an explicit result-contract choice, not a silent return-type change.
- **Event delivery:** the existing event bus supplies bounded fan-out/ring replay and slow-subscriber behavior. The event hub supplies acknowledged ordered handlers, journals and queued reactions. A bridge preserves both guarantees; merging them would change acknowledgement and loss behavior.

## Coherent harness entry points

Use `core.runtime.calls.CallRuntime` with `models.contracts.types.Binding` and `models.contracts.types.Answer` for model-neutral execution. Compose behavior through `agents.execution.runner.AgentRunner` and `agents.minds`; use `state.context.WorkingContext` for portable context, `tools.execution.policy.ToolSet` for execution policy, and `integrations.mcp.gateway.McpGateway` for external tools. Each import names a physical owner. The root API exposes only these six namespaces and version metadata. Each owner root contains only its export-assembly initializer; contracts, serialization, execution, catalogues and provider implementations live in responsibility subpackages.

The existing `Agent`/`Cognition` compositions remain a different execution interface. They reuse transports, canonical schema validation/compilation, explicit capability declarations and shared nullable accounting provenance, while their orchestration/result types remain distinct. New harnesses should select the model-neutral contracts consistently instead of mixing both result conventions within one budget ledger. This is an explicit remaining semantic consolidation boundary, not a claim that physical relocation eliminated every duplicate policy.

## Readiness limits

The six-group structure is an ownership contract, not proof of production readiness. Open items include deployment-specific sandbox verification, host wiring and multi-host acceptance of durable journals/leases, credential refresh, and live acceptance of the supported model/CLI versions. The real-workflow matrix and its failures are recorded separately from offline architectural regression checks.

The evaluation coverage map distinguishes contract witnesses, application-workload witnesses and unsupported backend combinations. A paired failure scenario counts only when its independent state/evidence checks pass alongside the related positive scenario on the same backend. Neither test count nor folder count establishes reliability for every application.

## Test and evaluation ownership

Offline unit tests follow the six owners under `tests/source/unit/{core,models,agents,state,tools,integrations}`. Cross-owner contracts stay in `tests/source/conformance`, real boundary combinations in `tests/source/integration`, namespace/documentation guards in `tests/source/architecture`, and evaluation infrastructure/oracle tests in `tests/source/evaluation`. The executable eval harness lives under `tests/evals`, with runner/contracts in `engine`, domain workflows in `suites`, independent oracles in `graders`, and workspace policy in `sandbox`. Package initializers assemble exports; implementation and metadata resolution live in owner modules. Architecture guards require exactly six root packages, explicit `models.providers.frontiers` and `models.providers.cli` families, and absence of removed aggregate imports.

The workload catalogue is a separate executable contract: domain workflows, expected outcomes, supported backend combinations and feature witnesses are declared independently of the runner. A successful grader must inspect tool receipts, state transitions or independent data/source oracles; narrative claims alone cannot establish success. Evaluation registration does not require copying runtime logic into domain folders.

Public owner exports include `ModelRegistry` for configured provider/model resolution; `CAPABILITY_NAMES` lists registry capability keys. `RegistryError` is its failure base and `UnknownModel` identifies unresolved models. `MindKind` names single-call, ReAct and chain-of-thought strategies without coupling their implementation to a provider. Core execution exposes `OnRepeat` for recurrence delivery (`escalate` or `stop`) and `backoff_delay` for capped exponential full-jitter retry delay. These utilities keep provider policy separate from orchestration.

Additional public contracts follow these owner boundaries. `AttrsAdapter`, `DataclassAdapter`, `PydanticAdapter` and `JsonSchemaAdapter` implement model output-schema adaptation. `HttpLLM` supplies shared HTTP provider behavior; vendor request translators remain separate. `BudgetCheck` is a request-builder callback for checking an estimated token count. `RetryPolicy` bounds call attempts, delay and elapsed time; `MeterMiddleware` accounts through the host ledger, while `SecurityMiddleware` evaluates tool access policy. `to_money` converts monetary inputs to finite fixed-scale Decimal values and can refuse excess precision.

The public `models.validate_schema` helper checks a declared JSON Schema before dispatch. `models.validate_output` validates a returned value against that schema. Both execution interfaces use these shared rules, so invalid declarations fail before a paid call and invalid responses cannot silently bypass validation.

Memory composition uses `DedupeMode` (`id` or `content`); `DEDUPE_SOURCES_KEY` and `DEDUPE_COUNT_KEY` preserve duplicate provenance and collapsed counts. `OnWritePolicy` controls whether a read-only memory refuses or ignores writes. `GROUNDING_RECORD_KEY` names the request builder's admitted-evidence scratchpad record.

`StoreBackedCheckpointStore` adapts a generic store to the latest-checkpoint contract; it does not provide historical versions. `resolve_checkpointer` selects an explicit checkpoint facade or derives one from host services, and `ckpt_key` defines its storage key. `coord_state_to_dict`/`coord_state_from_dict`, `prefix_to_dict`/`dict_to_prefix`, `result_to_dict`/`dict_to_result` and `usage_to_dict` encode typed checkpoint state. `msg_to_dict`/`dict_to_msg` reuse the canonical message codec so multimodal content, metadata and nested tool arguments survive persistence.

Durable composition now includes `StoreFiles`, a bounded CAS file-tree over a host StorePort for FileTool and FileMemory. `ToolSet.with_registry` preserves overflow, timeout and deferred-discovery policy when AgentRunner filters authorized tools; each invocation starts with fresh discovery state. `SQLiteEventJournal` supplies durable ordered event replay and exact-event idempotency. `LeaseStore`, `Lease` and `SQLiteLeaseStore` provide host admission, expiry, renewal and fencing for multiple processes; hosts wire admission and enforce fencing at downstream commits.
