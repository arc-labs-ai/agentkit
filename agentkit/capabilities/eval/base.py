"""Eval capability — cheap synchronous code checks + an optional LLM judge.

Two tiers (the reference pattern): `code_evals` is deterministic and fast (gate obvious
failures); `judge` is an LLM-as-judge meant to run **off the hot path** (the caller fires it
async / in a worker). Both are failure-isolated — a broken check or judge never raises into
the agent run; it yields a falsy/empty result that monitoring records.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from agentkit.kernel.types import ChatRequest, Message


class Evaluator:
    def __init__(
        self,
        code_checks: dict[str, Callable[[Any], bool]] | None = None,
        *,
        judge_model: str = "",
        rubric: str = "",
    ):
        self._checks = code_checks or {}
        self._judge_model = judge_model
        self._rubric = (
            rubric or 'You are a strict evaluator. Return JSON {"score": 0..1, "reason": str}.'
        )

    def code_evals(self, output: Any) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name, check in self._checks.items():
            try:
                results[name] = bool(check(output))
            except Exception:
                results[name] = False  # a broken check is a failed check, not a crash
        return results

    async def judge(self, ctx: Any, *, goal: str, output: Any) -> dict[str, Any]:
        """Run the judge model through the invoker chain — tracing +
        retry + meter all fire. Direct ``self._judge_llm.complete()`` would
        bypass middleware: a flaky judge couldn't be retried and its cost
        wouldn't land on the run's budget. Routing through
        ``ctx.invoker.chat()`` keeps the judge on the same rails as
        every other model call.

        Off-hot-path: caller is expected to fire-and-forget on a worker
        / background task. Returns ``{}`` on any failure (broken judge
        result, JSON parse failure, missing invoker) so monitoring
        records the empty result without crashing the run.
        """
        invoker = getattr(ctx, "invoker", None)
        if invoker is None or not self._judge_model:
            return {}
        try:
            request = ChatRequest(
                messages=[
                    Message(role="system", content=self._rubric),
                    Message(role="user", content=f"GOAL: {goal}\nOUTPUT: {output}"),
                ],
                model=self._judge_model,
                response_format={"type": "json_object"},
            )
            result = await invoker.chat(request, ctx)
            parsed = json.loads(result.content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


__all__ = ["Evaluator"]
