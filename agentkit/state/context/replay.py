"""Provider-neutral context policy and a hard UTF-8 bound for portable replay."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass

from agentkit.core.serialization.messages import message_to_dict
from agentkit.state.context.context import WorkingContext
from agentkit.state.context.policy import ContextPolicy


@dataclass(frozen=True)
class BoundedReplay:
    context: WorkingContext
    byte_size: int
    changed: bool
    over_limit: bool


async def prepare_bounded_replay(
    context: WorkingContext, *, context_policy: ContextPolicy | None = None, max_bytes: int = 1024 * 1024
) -> BoundedReplay:
    """Apply the host policy once and measure the actual portable transcript.

    Prefix/grounding are included in the bound. A policy cannot bypass the
    bound by retaining oversized protected evidence. The caller must report
    ``limit_reached`` and avoid dispatch when ``over_limit`` is true.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("portable replay byte bound must be a positive integer")
    original = (context.prefix, tuple(context.messages))
    if context_policy is not None:
        revised = context_policy(context)
        if inspect.isawaitable(revised):
            revised = await revised
        if not isinstance(revised, WorkingContext):
            raise TypeError("context_policy must return WorkingContext")
        context = revised
    encoded = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False).iterencode(
        [message_to_dict(message) for message in (*context.prefix.as_messages(), *context.messages)]
    )
    byte_size = sum(len(piece.encode("utf-8")) for piece in encoded)
    return BoundedReplay(
        context, byte_size, (context.prefix, tuple(context.messages)) != original, byte_size > max_bytes
    )


__all__ = ["BoundedReplay", "prepare_bounded_replay"]
