"""Portable replay is transformed and bounded before any provider dispatch."""

import pytest

from agentkit.core.contracts.types import Message, MessageRole
from agentkit.state.context import WorkingContext
from agentkit.state.context.policy import SlidingWindowContextPolicy
from agentkit.state.context.replay import prepare_bounded_replay


def long_context():
    return WorkingContext(
        messages=[
            Message(MessageRole.USER, "x" * (17 * 1024 * 1024)),
            Message(MessageRole.ASSISTANT, "old work complete"),
            Message(MessageRole.USER, "Current task: continue from the approved brief."),
        ]
    )


@pytest.mark.asyncio
async def test_seventeen_mib_portable_transcript_passes_through_context_policy_and_becomes_bounded():
    result = await prepare_bounded_replay(long_context(), context_policy=SlidingWindowContextPolicy(1), max_bytes=4096)
    assert result.changed and not result.over_limit and result.byte_size <= 4096
    assert len(result.context.messages) == 1 and result.context.messages[0].content.startswith("Current task:")


@pytest.mark.asyncio
async def test_context_too_large_after_policy_is_still_refused_and_clean_twin_is_preserved():
    unchanged = long_context()
    result = await prepare_bounded_replay(unchanged, context_policy=lambda context: context, max_bytes=4096)
    assert result.over_limit and not result.changed and result.byte_size > 4096
    small = WorkingContext(messages=[Message(MessageRole.USER, "small current task")])
    result = await prepare_bounded_replay(small, max_bytes=4096)
    assert result.context is small and not result.changed and not result.over_limit


@pytest.mark.asyncio
async def test_async_policy_is_applied_once_and_noncontext_result_is_not_silently_used():
    called = []

    # Use the public deterministic policy inside an async host callback.
    async def proper(context):
        called.append(True)
        return SlidingWindowContextPolicy(1)(context)

    result = await prepare_bounded_replay(long_context(), context_policy=proper, max_bytes=4096)
    assert called == [True] and not result.over_limit
    with pytest.raises(TypeError, match="WorkingContext"):
        await prepare_bounded_replay(long_context(), context_policy=lambda _: None, max_bytes=4096)


@pytest.mark.asyncio
async def test_protected_prefix_and_unicode_are_included_in_the_byte_bound():
    from agentkit.state.context import PrefixContext

    context = WorkingContext(
        prefix=PrefixContext(system_prompt="é" * 3000), messages=[Message(MessageRole.USER, "small task")]
    )
    result = await prepare_bounded_replay(context, context_policy=SlidingWindowContextPolicy(1), max_bytes=4096)
    assert result.over_limit and result.byte_size >= 6000


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
async def test_replay_bound_must_be_positive_integer(bound):
    with pytest.raises(ValueError, match="positive integer"):
        await prepare_bounded_replay(WorkingContext(), max_bytes=bound)
