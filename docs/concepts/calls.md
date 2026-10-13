# Model-neutral calls

Install `arc-agentkit[calls]` for HTTP, schema validation, YAML skills and MCP support. The core package still has no required third-party dependencies. The existing `Agent`/`Cognition` interfaces remain available; the unified contract is owned by `agentkit.core.runtime.calls` for execution and `agentkit.models.contracts.types` for call values. Agent definitions and minds live in `agentkit.agents`; context/checkpoints live in `agentkit.state`. The removed aggregate namespaces have no compatibility wrappers.

```python
from decimal import Decimal
from agentkit.core.contracts.types import Mode, MessageRole
from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef
from agentkit.core.runtime.calls import CallRuntime
from agentkit.models.contracts.schema import OutputSpec

# The host owns credentials, including rotation; resolve them at dispatch.
binding = Binding(Mode.OPENAI_API, "gpt-5.4-mini", CredentialRef(lambda: {
    "OPENAI_API_KEY": "replace-with-host-resolved-key",
}))
spec = OutputSpec({
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
})

async def calculate():
    async with CallRuntime() as runtime:
        return await runtime.ask(
            binding, "What is 6 times 7?", output=spec,
            limits=CallLimits(max_cost_usd=Decimal("0.02"), wall_clock_s=30),
            call_id="calculate-1",
        )
```

`Mode` selects `CLAUDE_CODE`, `CODEX`, `ANTHROPIC_API`, or `OPENAI_API`. `Binding` specifies transport, model and a host-owned `CredentialRef`. `ApiEndpoint` and `ApiPath` expose default vendor endpoints; custom HTTP endpoints remain configurable through the existing HTTP adapters. `MessageRole` identifies canonical system, developer, user, assistant and tool messages.

`Answer` contains the final value/text, transcript, `Outcome`, disjoint `Usage`, raw provider usage, optional cost, effective settings, session and context snapshot. `CostSource` distinguishes reported costs, versioned `PriceTable` estimates and unpriced models. Unknown cost is `None`; it is never silently zero. `FailureClass` distinguishes structured authentication, quota, transient, refusal and other failures. `CallLimits` bounds execution; estimates cannot stop a remote provider from spending tokens before a final usage report arrives.

`OutputSpec` uses a canonical JSON Schema. The compiler translates supported constraints into each transport's schema dialect, then validates the returned value against the canonical schema. Unsupported constructs fail before dispatch. Provider strict-schema rewrites do not remove canonical validation. One corrective response is allowed after invalid output. `Answer.prompt_manifest` records the initially composed request; corrective and verifier continuation text belongs to the canonical conversation and does not rewrite that initial provenance.

## Settings and authentication

`ModelSettings` are checked against a host-overridable, versioned `CapabilityTable`. Sampling controls without verified support are refused. Effort is mapped to a supported level with an explicit `Degradation`; presentation hints may be dropped with an explanation. `Effort` exports the standard effort vocabulary. Mode-wide CLI profiles are conservative defaults, not guarantees for every installed CLI version/model.

`AnthropicCredential` and `AuthKind` distinguish API keys from bearer credentials. `anthropic_credential` prefers `ANTHROPIC_AUTH_TOKEN`; OAuth tokens in the legacy `ANTHROPIC_API_KEY` slot are recognized and routed as bearer credentials. Claude Code receives `ANTHROPIC_AUTH_TOKEN` for bearer tokens and `ANTHROPIC_API_KEY` for API keys. A host must obtain and refresh OAuth credentials; agentkit does not launch a login flow or store refresh tokens. Credentials are excluded from credential representations and session bundles.

## Context and sessions

Pass a `WorkingContext` to attach host metadata and canonical messages to the call; `Answer.context` is a frozen snapshot. Canonical `Message` supports metadata and structured content blocks, including images and tool calls/results. `SessionHandle` carries canonical messages plus transport-specific state. Resume with `resume=answer.session`.

Same-model continuation preserves opaque provider history. A model or vendor switch replays canonical conversation content and reports that nonportable opaque reasoning state was dropped. Private reasoning signatures cannot be moved between providers. Native CLI session files are bounded, isolated from credentials, and bundled for continuation in a fresh runtime. Treat session handles as sensitive conversation data and put them in host storage with an explicit retention policy.

## Tools, skills and adapters

`ToolSet` wraps the existing `ToolRegistry`, applying common result/error/image/overflow policy. `NativeToolPolicy` separately controls CLI built-in capabilities. Claude native tools are disabled. Codex `WARN` disables verified shell/execution features and emits a degradation because other native tools can bypass registry grants; `DENY` refuses Codex before spawning when full native-tool denial cannot be guaranteed, and `ALLOW` is an explicit host choice. Registry grants alone are not a Codex sandbox. `OverflowPolicy` spills large results to bounded file references; the host owns retention of those files. `DeferredToolName` exposes the deferred search/call tool identifiers. Deferred tools are discovered through `search_tools`; callers must load a tool before executing it. `serve` exposes the same result policy through local MCP.

`McpGateway`, `ServerConfig` and `McpPolicy` identify external servers by endpoint/command identity and pin raw tool definitions to approval digests. Allow/deny rules are structural, and deny wins. `McpServerStatus` exposes connected, unavailable and closed connection states. `McpProtocolVersion` names supported protocol versions; `McpProtocolError` reports malformed protocol exchanges. The gateway supports stateless per-request metadata/header mirroring, JSON/SSE responses, and initialized Streamable HTTP for supported 2025 revisions. Public HTTPS addresses are resolved and pinned. Deprecated two-endpoint HTTP+SSE, background subscriptions, MRTR and OAuth browser discovery are not provided by the built-in client; a host client factory can implement additional transports. Stdio launchers must be supplied by the host; MCP cannot turn an arbitrary process into a sandbox. Hosts supply credential resolvers and stronger network/process isolation appropriate to their deployment.

Transport implementations live in `agentkit.models.providers.frontiers.calls` and `agentkit.models.providers.cli.calls`; wire schemas, settings and pricing live in `agentkit.models`; execution enforcement lives in `agentkit.core.runtime.calls`; agent orchestration and minds live in `agentkit.agents`. `ModeAdapter` and `Turn` are the injection seam for deterministic tests or additional transports. Existing HTTP streaming and CLI drivers are reused. The legacy provider/cognition APIs still have distinct result types and are not automatically converted into the new call contract.

## Validation and support boundaries

The conformance tests exercise synthetic faults and real transport translations. Opt-in live evaluations exercise installed CLIs and provider APIs; see `tests/evals/README.md`. Passing those checks is evidence for the tested versions and cases, not a guarantee of durable execution, distributed delivery, sandbox security, or every provider/model capability.

Provider behavior is grounded in the official [OpenAI Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses), [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create), [Anthropic authentication](https://platform.claude.com/docs/en/manage-claude/authentication), and [MCP specification](https://modelcontextprotocol.io/specification/2026-07-28). Host adapters must track changes to their pinned versions.
