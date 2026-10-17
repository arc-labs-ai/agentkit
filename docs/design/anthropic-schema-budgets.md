# Anthropic structured-output reference accounting

This fix is on `fix/anthropic-schema-budgets`, based on `main`. It contains no
changes from the separate CLI execution/isolation branch. No consuming
application or live provider credentials are needed to reproduce or test it.

## Root cause and wire semantics

The Anthropic compiler inlined each local reference at its response location,
then retained and traversed `$defs`. It counted a reachable field again in its
definition, including descendants expanded there. The complete Pydantic
reproduction in `tests/source/conformance/test_schema_reference_budgets.py`
failed locally with these counts:

| Count | Before | Correct emitted schema |
| --- | ---: | ---: |
| Optional fields | 31 | 13 |
| Union-bearing locations | 0 | 0 |
| Properties | 35 | 15 |

The reachable shape contains one Answer property, eight Angle properties and
three properties in each of ProductRoute and RouteFamily. Two Angle properties
are required. Thus optional fields total `1 + 6 + 3 + 3 = 13`.

The compiler now expands references, removes the redundant `$defs` and legacy
`definitions` at supported schema locations, and measures the completed wire
schema. It refuses any remaining reference/definition in an unsupported location
before returning a schema, so pruning cannot leave dangling references. Unused
definitions contribute nothing. A shared definition emitted under two properties
contributes its fields twice: those are distinct wire locations. Definition names
are not a deduplication key after expansion.

Budget traversal visits schema locations, not literal data in default, enum or
example annotations. A location containing both `anyOf` and a type array counts
as one union-bearing location. Canonical reference validation follows the same
schema-location distinction and never fetches external references.

## Provider restrictions versus library policy

Verified against Anthropic's official
[structured-output documentation](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
on 2026-10-18:

- The documented request limits are 24 optional parameters, 16 union-typed
  parameters and 20 strict tools. Optional means absent from `required`;
  union-typed means `anyOf` or a type array.
- These limits combine JSON output and strict tool schemas in one request.
  Agentkit's current API adapter sends hosted tools without `strict: true`, so
  this compiler checks the output schema. A host adding strict tools or changing
  the payload must also respect the combined request limits.
- Local references and definitions are supported; external references and
  recursive schemas are unsupported. The documentation does not specify a
  deduplication algorithm for retained definitions. This compiler emits no
  references or definitions for Anthropic, making its local counts explicit in
  the wire tree rather than guessing about provider graph accounting.
- Passing explicit limits does not guarantee provider acceptance: additional
  grammar complexity limits and a compilation timeout also exist.

The 24/16 thresholds were **not** raised or removed. Agentkit remains more
conservative than the documented dialect in other ways: it rejects `allOf`,
strips several bounds/patterns/formats into descriptions, and validates the
returned value against the original canonical constraints. That existing policy
was not widened by this fix. Conflicting validation keywords alongside a
reference are now refused rather than silently overwriting target fields.

Claude Code remains a separate draft-07 compilation path with rewritten local
references and definitions; Anthropic API's 24/16 check is not imposed on it.
OpenAI API and Codex retain their strict schemas, references/definitions,
required-nullable optional encoding and existing size/depth limits. Native-tool
and spending policy checks are unchanged.

## Public lifecycle and migration

`compile_schema` still returns a schema dictionary and raises `SchemaError` for
unsupported schemas when called directly. `SchemaError` now exposes stable
`code` and numeric `details`. Invalid canonical JSON Schema is normalized to
that exception rather than leaking a third-party schema exception.

`CallRuntime.ask` and `run` now convert compilation errors into the existing
`Answer`/`Failure` lifecycle:

- `outcome=failed`, `failure.classification=invalid_request`, `retryable=False`.
- `mode_detail.reason=schema_compilation_failed`, `origin=local`, and
  `dispatch_state=not_started` distinguish local compilation from provider
  refusal or provider request rejection.
- `mode_detail.code` identifies the condition. Budget failures include
  `optional_count`, `optional_limit`, `union_count`, `union_limit` and
  `property_count`. Explanations are fixed actionable text, never raw exceptions,
  schema field names, prompts, credentials or provider response bodies.
- Zero attempts, zero credential resolutions and zero transport requests;
  ordered `call_started`/`call_finished` events and no `attempt_started` event.
  Missing usage/cost remain incomplete/unpriced.

Codes are `schema_budget_exceeded`, `invalid_schema`, `recursive_schema`,
`unresolved_reference`, `unsupported_ref_siblings`,
`unsupported_reference_location`, and the general `unsupported_schema`.
No new failure-class enum is introduced. Consumers that caught uncaught
`SchemaError` around runtime calls should instead inspect `answer.failure`.
Direct compiler users can continue catching `SchemaError`.

The original schema is never mutated, answer fields are not dropped, and no
provider, model, dialect or unstructured-output fallback is substituted.
Canonical response validation still rejects invalid enum values and bounds,
including constraints weakened on the wire. Valid bounded siblings that cannot
be combined safely must be explicitly normalized by the caller, not silently
relaxed by the compiler.

## Integration example

Continue using the host's existing runtime through its cognition integration.
`AnswerShape` below is the application's Pydantic output type, not Agentkit's
result wrapper. The binding and credential resolver remain host-owned.

```python
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CallLimits

async def request_structured(runtime: CallRuntime, binding: Binding, output_type: type):
    result = await runtime.ask(
        binding,
        "Produce the requested structured answer.",
        output=output_type,
        limits=CallLimits(max_output_tokens=4096),
        call_id="structured-answer-001",
    )
    if result.failure and result.failure.mode_detail.get("reason") == "schema_compilation_failed":
        return {"local_schema_error": dict(result.failure.mode_detail)}
    return result
```

No application-specific compatibility shim or field renaming is needed. A
passing local check is not evidence of a successful live provider connection.

## Test evidence

Before changing the compiler, the exact reproduction failed with the observed
31/0/35 counts. The initial reference-budget suite had **10 failures and 18
passes**. All four scripted lifecycle cases initially leaked exceptions; the
supported case could not dispatch. Two annotation-data tests independently
caught literal `$ref` values being mistaken for schema references.

Regression tests cover the reproduction, unused definitions, repeated and nested
references, exact 24/16 boundaries, real overflows, recursive/unresolved refs,
immutability, reference sibling constraints, safe pruning, canonical validation,
all four mode dialects, and supported versus refused scripted HTTP dispatch.
The mutation catalogue also plants redundant definitions, double union counting
and widened limits; those regressions must be killed by these tests.

Provider-network and paid-call probes are intentionally excluded. No credentials
or authorization for such probes were supplied.
