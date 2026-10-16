import asyncio

from agentkit.core.contracts.types import Message, MessageRole
from agentkit.core.testing import FakeLLM, FakeTurn
from agentkit.models.contracts.types import Turn as AdapterTurn


def test_fake_turn_is_distinct_from_model_adapter_turn():
    assert FakeTurn is not AdapterTurn

    fake = FakeLLM.script([FakeTurn(content="scripted reply")])
    result = asyncio.run(fake.chat(messages=[Message(MessageRole.USER, "question")], model="test-model"))

    assert result.content == "scripted reply"
