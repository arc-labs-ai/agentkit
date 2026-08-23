"""Framework-owned seed prompts.

These are the prompts agentkit's own machinery (compaction, future
guardrail-style rephrasers, etc.) sends to an LLM on behalf of the
running agent. They live here for three reasons:

1. **Single source of truth.** A free-floating string inside
   `compaction.py` was easy to drift from — when an operator tuned
   compaction wording for production, they'd patch the file rather than
   bump a version. Promoting these to `Prompt` objects gives them the
   same id+version discipline every feature-owned prompt already has.

2. **Attribution.** A compaction summary now lands in traces with a
   `prompt_version` stamp; if compaction quality regresses, we can map
   the bad output back to the exact template that produced it.

3. **Separation of concerns.** A compaction *strategy* is policy
   (when/how to fold the transcript). A compaction *prompt* is wording.
   Mixing the two inside one file was the single-responsibility
   violation we set out to fix in this pass.

These are seeds: the version label says so. Framework callers just
`.render()` the seed; apps that want to override these per-tenant own
that machinery on top of the `Prompt` skeleton.
"""

from __future__ import annotations

from agentkit.prompts.prompt import Prompt

COMPACTION_SUMMARY = Prompt(
    id="agentkit.compaction.summary",
    version="seed-2026-06-17",
    template=(
        "Summarize the conversation so far, preserving facts, decisions, "
        "tool results, and open questions. Be concise and lossless on "
        "anything an agent would need to continue."
    ),
)
"""System prompt for `SummarizationCompactor`. Bump the version label
whenever the template wording changes so traces stay attributable."""

COMPACTION_IMPORTANCE = Prompt(
    id="agentkit.compaction.importance",
    version="seed-2026-06-17",
    template=(
        "Analyze the following conversation turns and identify the most "
        "important information that must be preserved for future reasoning. "
        "Return only the essential turns or a concise synthesis of key points."
    ),
)
"""System prompt for `ImportanceFilteringCompactor`. Same versioning
contract as the summary seed."""


__all__ = ["COMPACTION_IMPORTANCE", "COMPACTION_SUMMARY"]
