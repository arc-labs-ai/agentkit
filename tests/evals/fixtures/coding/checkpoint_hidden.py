import json

from agentkit.core.contracts.types import Message
from agentkit.core.serialization.messages import message_from_dict, message_to_dict


def test_message_content_metadata_and_blocks_survive_json_roundtrip() -> None:
    original = Message(
        "assistant",
        "chart summary",
        metadata={"agent_id": "analyst", "nested": {"facts": ["a", "b"]}},
        content_blocks=({"type": "image", "mimeType": "image/png", "data": "eA=="},),
    )
    restored = message_from_dict(json.loads(json.dumps(message_to_dict(original))))
    assert restored == original


def test_optional_fields_and_serialized_payload_mutability() -> None:
    plain = message_from_dict({"role": "user", "content": "plain text"})
    assert plain.content_blocks == () and plain.metadata == {}
    message = Message(
        "user", metadata={"source": {"pages": [1, 4]}}, content_blocks=({"type": "text", "text": "payload"},)
    )
    wire = message_to_dict(message)
    wire["metadata"]["source"]["pages"].append(8)
    wire["content_blocks"][0]["text"] = "edited wire"
    assert message.metadata["source"]["pages"] == [1, 4]
    assert message.content_blocks[0]["text"] == "payload"
