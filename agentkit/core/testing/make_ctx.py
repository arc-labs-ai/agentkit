"""Small context for tool and MCP contract tests; it does not emulate a runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentkit.core.contracts.types import Scope
from agentkit.core.execution.concurrency import CancellationToken


@dataclass
class TestToolContext:
    scope: Scope
    cancel: CancellationToken
    correlation_id: str = "test-run"
    invoker: Any = None
    autonomy: str = "auto"

    def check_cancelled(self) -> None:
        self.cancel.raise_if_cancelled()


def make_test_ctx(*, scope: Scope | None = None, cancel: CancellationToken | None = None, **_: Any) -> TestToolContext:
    return TestToolContext(scope or Scope(), cancel or CancellationToken())


__all__ = ["TestToolContext", "make_test_ctx"]
