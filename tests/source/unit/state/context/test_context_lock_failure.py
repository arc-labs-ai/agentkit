"""Context locks serialize work and release on interruption without claiming rollback."""

import asyncio

import pytest

from agentkit.core.contracts.types import Message, MessageRole
from agentkit.state.context import WorkingContext


@pytest.mark.asyncio
async def test_cancelled_context_mutation_releases_lock_and_preserves_already_applied_evidence():
    context, entered = WorkingContext(), asyncio.Event()

    async def mutate(ctx):
        ctx.append(Message(MessageRole.USER, "sampling started"))
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(context.apply_locked(mutate))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(context.apply_locked(lambda ctx: ctx.append(Message(MessageRole.USER, "cleanup observed"))), 1)
    assert [message.content for message in context.messages] == ["sampling started", "cleanup observed"]


@pytest.mark.asyncio
async def test_failing_context_mutation_releases_lock_for_subsequent_host_work():
    context = WorkingContext()

    def mutate(ctx):
        ctx.note("stage", "started")
        raise ValueError("host mutation rejected")

    with pytest.raises(ValueError, match="rejected"):
        await context.apply_locked(mutate)
    await asyncio.wait_for(context.apply_locked(lambda ctx: ctx.note("stage", "reviewed")), 1)
    assert context.get("stage") == "reviewed"
