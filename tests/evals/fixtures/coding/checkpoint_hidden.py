import json

from agentkit.core.contracts.types import Message, ToolCall
from agentkit.core.serialization.messages import message_from_dict, message_to_dict
from agentkit.state.checkpoints import dict_to_prefix, prefix_to_dict
from agentkit.state.context import PrefixContext


def test_grounding_metadata_and_tool_calls_survive_json_checkpoint() -> None:
    original = PrefixContext(
        "Retain evidence",
        (
            Message(
                "assistant",
                tool_calls=(ToolCall("c", "extract", {"pages": [2, 7]}),),
                metadata={"agent_id": "researcher", "nested": {"facts": ["a", "b"]}},
            ),
            Message(
                "tool",
                tool_call_id="c",
                content_blocks=(
                    {"type": "text", "text": "details"},
                    {"type": "image", "mimeType": "image/png", "data": "eA=="},
                ),
                metadata={"is_error": True, "overflowed": True},
            ),
        ),
        "Required output format",
    )
    restored = dict_to_prefix(json.loads(json.dumps(prefix_to_dict(original))))
    assert restored == original
    assert restored.grounding[0].tool_calls[0].arguments["pages"] == [2, 7]


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
