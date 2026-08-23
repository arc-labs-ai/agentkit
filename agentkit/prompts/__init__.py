"""agentkit.prompts — the prompt-management skeleton.

The framework defines WHAT a prompt is. Applications own storage,
discovery, resolution, and any validation/evaluation contracts they
choose to add — those don't belong in the framework's public surface
unless the framework actually dispatches them.

The skeleton:
- `Prompt` (data type) — id + version + template + bind() + render()
- `builtin` — the framework's own seed prompts (Compactor)

Why no validator/evaluator skeleton: exporting ``Validator`` /
``PromptEvaluator`` type aliases the framework never dispatches would
imply a runtime validation surface that doesn't exist. The Agent's
parse-and-repair uses its own ``SchemaAdapter`` path. Apps that need
that pattern ship their own shapes.

Why no store/registry/resolver: prompts are code (defined as
constants, version-controlled in git). Apps that need runtime
mutation, dynamic loading, or a UI to inspect prompt history ship
their own infrastructure on top of this skeleton.
"""

from agentkit.prompts.prompt import Prompt

__all__ = ["Prompt"]
