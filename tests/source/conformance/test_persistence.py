import json

from agentkit.core.contracts.types import Message, MessageRole, ToolCall
from agentkit.state.checkpoints.persistence import dict_to_msg, msg_to_dict


def test_checkpoint_preserves_multimodal_content_context_metadata_and_tool_calls():
    original = Message(
        MessageRole.ASSISTANT,
        "image evidence",
        tool_calls=(ToolCall("c", "inspect", {"path": "a"}),),
        content_blocks=({"type": "image", "data": "encoded", "mimeType": "image/png"},),
        metadata={"agent": "analyst", "source": {"ids": ["evidence-1"]}},
    )
    wire = msg_to_dict(original)
    restored = dict_to_msg(json.loads(json.dumps(wire)))
    assert restored == original
    wire["metadata"]["source"]["ids"].append("changed")
    assert original.metadata["source"]["ids"] == ["evidence-1"]


def test_existing_text_checkpoint_remains_readable_without_new_optional_fields():
    assert dict_to_msg({"role": "user", "content": "hello"}) == Message("user", "hello")


def test_checkpoint_tool_arguments_are_deeply_mutable_without_aliasing_records():
    original = Message("assistant", tool_calls=(ToolCall("c", "inspect", {"paths": [{"parts": ["a"]}]}),))
    wire = msg_to_dict(original)
    wire["tool_calls"][0]["arguments"]["paths"][0]["parts"].append("b")
    assert original.tool_calls[0].arguments["paths"][0]["parts"] == ["a"]
    restored = dict_to_msg(json.loads(json.dumps(wire)))
    assert restored.tool_calls[0].arguments["paths"][0]["parts"] == ["a", "b"]


def test_checkpoint_helpers_share_kernel_codec():
    from agentkit.core.serialization.messages import message_from_dict, message_to_dict

    assert msg_to_dict is message_to_dict
    assert dict_to_msg is message_from_dict
