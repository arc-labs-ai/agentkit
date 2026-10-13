"""Transport event emission without execution policy."""

import inspect
from typing import Any

from agentkit.models.contracts.types import CallEvent


async def _emit(emit: Any, kind: str, **fields: Any) -> None:
    result = emit(CallEvent(kind, "", fields))
    if inspect.isawaitable(result):
        await result
