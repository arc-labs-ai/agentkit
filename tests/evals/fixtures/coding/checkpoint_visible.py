import json

from agentkit.core.contracts.types import Message
from agentkit.core.serialization.messages import message_from_dict, message_to_dict


def test_checkpoint_preserves_multimodal_tool_evidence_and_attribution() -> None:
    message = Message(
        "tool",
        "chart evidence",
        name="read_chart",
        tool_call_id="chart-1",
        content_blocks=({"type": "image", "mimeType": "image/png", "data": "eA=="},),
        metadata={"agent_id": "analyst", "is_error": False, "source": {"document": "quarterly"}},
    )
    restored = message_from_dict(json.loads(json.dumps(message_to_dict(message))))
    assert restored.content_blocks == message.content_blocks
    assert restored.metadata == message.metadata
    assert restored.name == "read_chart" and restored.tool_call_id == "chart-1"
