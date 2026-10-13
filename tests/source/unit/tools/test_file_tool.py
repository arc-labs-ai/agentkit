"""`FileTool`'s model-facing contract — the four things a rename would break.

`FileTool().schema.name` is `'memory'`, which reads like a collision with the
`agentkit.state.memory` package. It is not a slip: this tool is agentkit's
implementation of Anthropic's client-side memory tool
(`{"type": "memory_20250818", "name": "memory"}`), which fixes the name, the
command set, the single-call `command=` dispatch, and the `/memories` root. The
match was verified against the vendor contract and is exact on all four.

That makes the name a WIRE CONTRACT rather than a label, and wire contracts need
a test, because the failure mode of getting this wrong is silent: a renamed tool
still works, it just works against a name the model has never seen and every
stored transcript disagrees with. `tests/source/unit/tools/test_memory_tool.py` covers the
tool's BEHAVIOUR (path confinement, command semantics, registry wiring); this
file covers only what is visible to the model and why it may not be tidied.

These tests pass before and after the documentation change that accompanies
them — nothing about the tool's behaviour was altered, and that is the finding.
They are a ratchet against a future "fix", not a fix. Proven to bite, by
mutation rather than by revert since there is no behaviour change to revert:
renaming `TOOL_NAME` to `"files"` fails two of them
(`..._matches_the_vendor_memory_tool`, `..._deliberately_different`), and adding
a seventh command fails `..._command_set_matches_the_vendor_memory_tool_exactly`.
"""

from __future__ import annotations

from agentkit.tools.filesystem.files import TOOL_NAME, FileTool

# Anthropic's memory tool, as published: tool name, the six commands, and the
# directory it operates on. Spelled out as literals rather than imported from
# the module under test, so this file fails when the module drifts instead of
# agreeing with it.
VENDOR_TOOL_NAME = "memory"
VENDOR_COMMANDS = ("view", "create", "str_replace", "insert", "delete", "rename")
VENDOR_ROOT = "/memories"


def test_the_model_facing_name_matches_the_vendor_memory_tool() -> None:
    """The name a model sees. Both `FileTool.name` (what `ToolRegistry` keys on)
    and `schema.name` (what reaches the provider) must be it — they are two
    fields that can drift, which is why `TOOL_NAME` exists and why both are
    asserted."""
    tool = FileTool()
    assert TOOL_NAME == VENDOR_TOOL_NAME
    assert tool.name == VENDOR_TOOL_NAME
    assert tool.schema.name == VENDOR_TOOL_NAME


def test_the_command_set_matches_the_vendor_memory_tool_exactly() -> None:
    """Not a superset and not a subset. A model that has seen the vendor tool
    will emit exactly these six; an extra one it never emits is dead weight in
    the schema, and a missing one is a call that fails at runtime."""
    enum = tuple(FileTool().schema.parameters["properties"]["command"]["enum"])
    assert enum == VENDOR_COMMANDS


def test_the_call_shape_is_one_tool_with_a_command_argument() -> None:
    """The vendor's dispatch shape: ONE tool taking `command=`, not six tools.
    Splitting it into `memory_view` / `memory_create` / … would match no prompt
    the model was trained on and would multiply the tool menu by six."""
    params = FileTool().schema.parameters
    assert params["required"] == ["command", "path"]
    assert params["properties"]["command"]["type"] == "string"


def test_the_default_root_is_the_vendor_convention() -> None:
    """`/memories` is where a model that knows this tool will address paths
    without being told. Overridable for callers who are deliberately off the
    vendor contract; the DEFAULT is the contract."""
    assert FileTool().root == VENDOR_ROOT


def test_the_class_name_and_the_tool_name_are_deliberately_different() -> None:
    """The asymmetry the module docstring argues for, pinned so it reads as a
    decision rather than an oversight: the class is named for the READER (it is
    a file-tree tool, sitting beside `agentkit.state.memory`, which is the retrieval
    surface), the tool is named for the MODEL."""
    assert FileTool.__name__ == "FileTool"
    assert FileTool.name == "memory"
