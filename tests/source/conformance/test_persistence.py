"""The message wire codec — what a durable record is allowed to lose.

WHY THIS FILE EXISTS
--------------------
``agentkit/core/serialization/messages.py`` is the one definition of a
``Message`` on the wire: checkpoints write through it, transcript consumers and
resumed runs read back through it. It is a pair of hand-written projections,
not a generated one, so a field is carried because somebody listed it — and a
field that stops being listed produces no error anywhere. The encode still
returns a well-formed JSON object, the decode still returns a well-formed
``Message``, the suite stays green, and the loss only shows up on the far side
of a restart as context the agent no longer has.

``metadata`` is the field most exposed to that, and the least likely to be
noticed. It is where first-class context rides — provenance, a correlation id,
a tool's own annotation, the scope tags a router branches on — rather than
anything a transcript renders, so a checkpoint that silently drops it produces
a resumed run that reads correctly and behaves differently. Nothing in the
codec's own signature or types would change.

The round trip is asserted THROUGH ``json`` rather than straight from encode to
decode. The encoded container's whole purpose is to be a durable record, and
``json.dumps`` is the step that discovers a value no store can hold — which is
also why the encode has to hand back plain mutable containers rather than the
frozen ones the ``Message`` holds.
"""

from __future__ import annotations

import json

import pytest

from agentkit.core.contracts.types import Message, ToolCall
from agentkit.core.serialization._immutable import FrozenDict, FrozenList
from agentkit.core.serialization.messages import (
    message_from_dict,
    message_to_dict,
    tool_call_from_dict,
    tool_call_to_dict,
)


def _round_trip(message: Message) -> Message:
    """Encode, store, load, decode — the four steps a checkpoint actually takes."""
    return message_from_dict(json.loads(json.dumps(message_to_dict(message))))


def test_message_metadata_survives_the_round_trip_a_resumed_run_depends_on() -> None:
    """``metadata`` is first-class context, not decoration.

    It carries the facts a run was given about its own messages — where a turn
    came from, which correlation it belongs to, what scope a router should treat
    it as. None of that is visible in the rendered transcript, so a checkpoint
    that dropped it would restore a conversation that LOOKS identical and routes
    differently: the same prompts, the same text, and a lost provenance nobody
    can reconstruct because the only copy was the one the codec threw away.

    Asserted nested, because freezing and thawing are both recursive and a
    one-level projection is the failure that hides best — the key is present,
    the shape is right, and the value two levels down is gone.
    """
    message = Message(
        role="assistant",
        content="the diff is clean",
        metadata={
            "source": "checkpoint",
            "trace": {"correlation_id": "c-1", "span": {"id": "s-1", "depth": 2}},
            "tags": ["tenant:acme", "scope:review"],
        },
    )

    restored = _round_trip(message)

    assert restored.metadata == message.metadata
    assert restored.metadata["trace"]["span"]["id"] == "s-1", "a nested value two levels down"
    assert restored.metadata["tags"] == ["tenant:acme", "scope:review"]


def test_the_encoded_container_is_plain_mutable_json_a_store_can_add_its_own_keys_to() -> None:
    """A ``Message``'s payloads are deep-frozen, because the message is a value;
    the ENCODED form is a record on its way to a store, and a store adds keys —
    a row id, a version, a digest — before it writes. Handing back the frozen
    containers would make the obvious next line (`record["metadata"]["v"] = 1`)
    raise from inside a persistence adapter, which is the mirror of the bug
    ``thaw`` exists for: a resumed context that refused its first write.

    ``FrozenDict`` IS a ``dict``, so ``isinstance`` cannot tell them apart and
    an equality check passes either way. The type has to be compared exactly.
    """
    message = Message(role="user", content="go", metadata={"nested": {"a": 1}}, content_blocks=({"type": "text"},))

    assert isinstance(message.metadata, FrozenDict), "the value itself is frozen, as intended"

    encoded = message_to_dict(message)

    assert type(encoded["metadata"]) is dict, "a frozen record cannot be annotated by its store"
    assert type(encoded["metadata"]["nested"]) is dict, "thawing is recursive or it is pointless"
    encoded["metadata"]["written_at"] = "2026-01-01T00:00:00Z"
    assert encoded["metadata"]["written_at"], "a store must be able to stamp its own record"


def test_a_tool_call_keeps_its_arguments_through_the_same_record() -> None:
    """A tool call's arguments are the one field of a transcript that is
    re-executed rather than re-read. A checkpoint that restored the call with
    empty arguments would replay a DIFFERENT call under the same id, and the
    result message beside it would still correlate — so the replay would look
    consistent while doing something nobody asked for."""
    call = ToolCall("call-1", "write_file", {"path": "/tmp/x", "opts": {"mode": "append"}})

    restored = tool_call_from_dict(json.loads(json.dumps(tool_call_to_dict(call))))

    assert (restored.id, restored.name) == ("call-1", "write_file")
    assert restored.arguments == {"path": "/tmp/x", "opts": {"mode": "append"}}
    assert type(tool_call_to_dict(call)["arguments"]) is dict, "the record is writable"


def test_every_field_a_tool_result_turn_needs_to_be_reconstructable_survives() -> None:
    """The whole projection, driven once on a message that uses all of it. A
    ``role="tool"`` turn is the strictest case: without ``tool_call_id`` and
    ``name`` the provider rejects the next request outright, so these two are
    the fields whose loss fails loudly — which is exactly why they are asserted
    beside ``metadata``, whose loss does not."""
    message = Message(
        role="tool",
        content='{"ok": true}',
        name="write_file",
        tool_call_id="call-1",
        content_blocks=({"type": "text", "text": "ok"},),
        metadata={"duration_ms": 12},
    )

    restored = _round_trip(message)

    assert restored.role == "tool"
    assert restored.content == '{"ok": true}'
    assert restored.name == "write_file"
    assert restored.tool_call_id == "call-1"
    assert restored.content_blocks == ({"type": "text", "text": "ok"},)
    assert restored.metadata == {"duration_ms": 12}


def test_an_assistant_turn_that_requested_tools_round_trips_its_calls_as_a_tuple() -> None:
    """``tool_calls`` is a ``tuple`` on the value and a ``list`` on the wire, and
    the decode has to put the tuple back: ``Message`` is hashable only because
    every field is, and a list would make a restored message unhashable while
    the one it was built from was fine. That is how ``merge(mode="union")``
    passed its tests and failed in production once before, on the same type."""
    message = Message(
        role="assistant",
        tool_calls=(ToolCall("call-1", "read_file", {"path": "/tmp/a"}),),
        metadata={"attempt": 1},
    )

    restored = _round_trip(message)

    assert isinstance(restored.tool_calls, tuple)
    assert [c.name for c in restored.tool_calls] == ["read_file"]
    assert restored.tool_calls[0].arguments == {"path": "/tmp/a"}
    assert hash(restored) == hash(message), "a restored message must stay usable as a key"


def test_a_decoded_message_is_frozen_again_so_a_resumed_run_cannot_rewrite_history() -> None:
    """Decoding ends where encoding began: the record thaws for the store, and
    the ``Message`` rebuilt from it is a value again. Without the re-freeze a
    resumed run could mutate a turn that has already been persisted and
    observed, and the next checkpoint would write a transcript that disagrees
    with the one already on disk."""
    restored = _round_trip(Message(role="user", content="go", metadata={"nested": {"a": 1}}))

    assert isinstance(restored.metadata, FrozenDict)
    assert isinstance(restored.metadata["nested"], FrozenDict)
    with pytest.raises(TypeError, match="frozen value"):
        restored.metadata["nested"]["a"] = 2


def test_an_absent_optional_field_decodes_to_its_default_rather_than_raising() -> None:
    """Records written by an earlier version of the codec are still on disk, and
    a restart has to read them. Every optional key is read with a default for
    that reason: a ``KeyError`` here is not a bad record, it is a run that
    cannot resume at all."""
    restored = message_from_dict({"role": "user"})

    assert (restored.content, restored.name, restored.tool_call_id) == ("", None, None)
    assert restored.tool_calls == ()
    assert restored.content_blocks == ()
    assert restored.metadata == {}


def test_a_list_payload_is_thawed_as_a_list_and_not_left_frozen() -> None:
    """``content_blocks`` and any list inside ``metadata`` go through the same
    recursive thaw as the dicts, and ``FrozenList`` is a ``list`` subclass that
    an ``isinstance`` check cannot distinguish. Checked exactly, because a store
    that appends to the record it is about to write would otherwise raise from
    inside the adapter."""
    encoded = message_to_dict(
        Message(role="user", content="go", metadata={"tags": ["a", {"b": 1}]})
    )

    assert type(encoded["metadata"]["tags"]) is list
    assert type(encoded["metadata"]["tags"][1]) is dict
    assert not isinstance(encoded["metadata"]["tags"], FrozenList)
