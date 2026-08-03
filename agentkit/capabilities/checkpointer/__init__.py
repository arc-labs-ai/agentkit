"""Checkpointer capability — the durable-resume facade + the shape (de)serializers.

`base.py` houses the `Checkpointer` class (the ergonomic facade over `CheckpointPort`).
`persistence.py` houses the shape-conversion plumbing (Message/Usage/ToolCall ↔ dict, the
coordinator-state shape, and the legacy `StoreBackedCheckpointStore` bridge) that the
ReAct loop and the coordinator policies use to round-trip mid-run state through a
checkpoint.

Everything here is re-exported so callers say
``from agentkit.capabilities.checkpointer import …`` — they shouldn't have to know which
submodule a given name lives in.
"""

from agentkit.capabilities.checkpointer.base import Checkpointer
from agentkit.capabilities.checkpointer.persistence import (
    StoreBackedCheckpointStore,
    ckpt_key,
    coord_state_from_dict,
    coord_state_to_dict,
    dict_to_msg,
    dict_to_prefix,
    dict_to_result,
    dict_to_tc,
    msg_to_dict,
    prefix_to_dict,
    rehydrate,
    result_to_dict,
    tc_to_dict,
    usage_to_dict,
)

__all__ = [
    "Checkpointer",
    "StoreBackedCheckpointStore",
    "ckpt_key",
    "coord_state_from_dict",
    "coord_state_to_dict",
    "dict_to_msg",
    "dict_to_prefix",
    "dict_to_result",
    "dict_to_tc",
    "msg_to_dict",
    "prefix_to_dict",
    "rehydrate",
    "result_to_dict",
    "tc_to_dict",
    "usage_to_dict",
]
