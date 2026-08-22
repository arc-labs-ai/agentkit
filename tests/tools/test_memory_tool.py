"""Memory as a Tool (ch13): the agent-managed `memory(command=…)` file tree —
view/create/str_replace/insert/delete/rename — registerable like any tool. Offline & deterministic."""

import asyncio

import pytest

from agentkit.agents import Agent
from agentkit.agents.cognition import ReActCognition
from agentkit.kernel.types import Scope, ToolCall
from agentkit.runtime import Budget, Invoker, RunContext
from agentkit.runtime.context import Services
from agentkit.testing import FakeLLM, Turn
from agentkit.tools import FileTool, ToolRegistry
from agentkit.tools.file_tool import InMemoryFiles


def _run(coro):
    return asyncio.run(coro)


def _mem():
    return FileTool()


# ---- the command set --------------------------------------------------------------------------


def test_create_then_view_file_and_directory():
    m = _mem()
    assert (
        _run(
            m.run(
                {"command": "create", "path": "/memories/bugs/race.md", "file_text": "shared state"}
            )
        )
        == "created /memories/bugs/race.md"
    )
    assert _run(m.run({"command": "view", "path": "/memories/bugs/race.md"})) == "shared state"
    assert (
        _run(m.run({"command": "view", "path": "/memories/bugs"})) == "/memories/bugs/race.md"
    )  # dir listing


def test_str_replace_and_insert_and_rename_and_delete():
    m = _mem()
    _run(m.run({"command": "create", "path": "/memories/n.md", "file_text": "a\nb\nc"}))
    assert (
        _run(
            m.run(
                {"command": "str_replace", "path": "/memories/n.md", "old_str": "b", "new_str": "B"}
            )
        )
        == "edited /memories/n.md"
    )
    assert _run(m.run({"command": "view", "path": "/memories/n.md"})) == "a\nB\nc"
    _run(
        m.run({"command": "insert", "path": "/memories/n.md", "insert_line": 1, "insert_text": "X"})
    )
    assert _run(m.run({"command": "view", "path": "/memories/n.md"})) == "a\nX\nB\nc"
    _run(
        m.run({"command": "rename", "path": "/memories/n.md", "new_path": "/memories/archive/n.md"})
    )
    assert _run(m.run({"command": "view", "path": "/memories/archive/n.md"})) == "a\nX\nB\nc"
    _run(m.run({"command": "delete", "path": "/memories/archive/n.md"}))
    with pytest.raises(FileNotFoundError):
        _run(m.run({"command": "view", "path": "/memories/archive/n.md"}))


def test_unknown_command_and_missing_path_raise():
    m = _mem()
    with pytest.raises(ValueError):
        _run(m.run({"command": "nope", "path": "/x"}))
    with pytest.raises(ValueError):
        _run(m.run({"command": "view"}))


# ---- registers + drives through an Agent --------------------------------------------------


def test_agentloop_uses_memory_tool_via_tool_calls():
    mem = FileTool()
    llm = FakeLLM.script(
        [
            Turn(
                tool_calls=(
                    ToolCall(
                        "c1",
                        "memory",
                        {
                            "command": "create",
                            "path": "/memories/p.md",
                            "file_text": "singleton notes",
                        },
                    ),
                )
            ),
            Turn(
                tool_calls=(
                    ToolCall("c2", "memory", {"command": "view", "path": "/memories/p.md"}),
                )
            ),
            Turn(content="done"),
        ]
    )
    ctx = RunContext(
        "r", Scope(1, 2), Budget(max_cost_usd=10.0), Services(invoker=Invoker(llm=llm))
    )
    res = _run(
        Agent(name="reviewer", model="m", cognition=ReActCognition(tools=[mem])).run("review", ctx)
    )
    assert res.output == "done"
    # the agent's writes persisted in the tool's backend across the two calls
    assert _run(mem.run({"command": "view", "path": "/memories/p.md"})) == "singleton notes"


def test_memory_tool_is_registerable_and_side_effecting():
    reg = ToolRegistry.from_tools([FileTool()])
    assert reg.names() == ["memory"]
    assert reg.get("memory").side_effecting and reg.get("memory").schema is not None


# ---- root confinement (path-traversal guard) --------------------------------------------------


def test_memory_tool_confines_paths_to_root():
    """Regression: create/rename/delete/view must stay under `root` — absolute escapes and `..`
    traversal are rejected before reaching the backend (critical for an injected filesystem backend)."""
    m = FileTool()
    for bad in (
        "/etc/passwd",
        "../../etc/passwd",
        "/memories/../../etc/passwd",
        "/memories/../secret",
    ):
        with pytest.raises(PermissionError):
            _run(m.run({"command": "create", "path": bad, "file_text": "x"}))
    # a relative path is anchored under root, not the cwd
    assert (
        _run(m.run({"command": "create", "path": "notes.md", "file_text": "ok"}))
        == "created /memories/notes.md"
    )
    # rename cannot smuggle the destination out of root either
    with pytest.raises(PermissionError):
        _run(m.run({"command": "rename", "path": "/memories/notes.md", "new_path": "/etc/evil"}))


def test_insert_preserves_trailing_newline_and_validates_range():
    """Regression: insert must not drop a trailing newline or normalize line endings, and an
    out-of-range insert_line must error rather than silently append."""
    fs = InMemoryFiles()
    _run(fs.create("/memories/n.md", "a\nb\n"))  # note the trailing newline
    _run(fs.insert("/memories/n.md", 1, "X"))
    assert _run(fs.view("/memories/n.md")) == "a\nX\nb\n"  # trailing newline preserved
    with pytest.raises(ValueError):
        _run(fs.insert("/memories/n.md", 99, "Z"))  # out of range → error, not silent append


# ---- custom root, friendly errors, non-clobbering rename --------------------------------------


def test_custom_root_is_listable_when_empty():
    m = FileTool(root="/notes")
    assert (
        _run(m.run({"command": "view", "path": "/notes"})) == ""
    )  # empty custom root lists, not 404
    _run(m.run({"command": "create", "path": "/notes/a.md", "file_text": "x"}))
    assert _run(m.run({"command": "view", "path": "/notes"})) == "/notes/a.md"


def test_missing_required_args_raise_friendly_valueerror():
    m = FileTool()
    for args in (
        {"command": "create", "path": "/memories/x"},  # no file_text
        {"command": "str_replace", "path": "/memories/x", "old_str": "a"},  # no new_str
        {"command": "rename", "path": "/memories/x"},
    ):  # no new_path
        with pytest.raises(ValueError):
            _run(m.run(args))


def test_rename_refuses_to_clobber_existing():
    m = FileTool()
    _run(m.run({"command": "create", "path": "/memories/a.md", "file_text": "A"}))
    _run(m.run({"command": "create", "path": "/memories/b.md", "file_text": "B"}))
    with pytest.raises(ValueError):
        _run(m.run({"command": "rename", "path": "/memories/a.md", "new_path": "/memories/b.md"}))


# ── refuse-clobber on create + delete-root guard + Tool Protocol ─────────────
#
# Three safety invariants:
#   (a) ``create`` on an existing path refuses unless ``overwrite=True``
#       instead of silently overwriting.
#   (b) ``delete("/memories")`` refuses the root path explicitly so a
#       prefix match cannot wipe the whole tree.
#   (c) FileTool satisfies its own ``Tool`` Protocol (``description`` +
#       ``output_schema`` declared).


def test_create_refuses_to_clobber_existing_file_by_default():
    """Twice-created same path → second call raises ``FileExistsError``
    and the original content survives."""
    m = _mem()
    _run(m.run({"command": "create", "path": "/memories/note.md", "file_text": "original"}))
    with pytest.raises(FileExistsError):
        _run(m.run({"command": "create", "path": "/memories/note.md", "file_text": "replacement"}))
    # Original untouched.
    assert _run(m.run({"command": "view", "path": "/memories/note.md"})) == "original"


def test_create_with_overwrite_true_replaces_explicitly():
    """The opt-in path: ``overwrite=True`` lets the model deliberately
    replace. The return message says "replaced" instead of "created" so
    the operation record reflects what really happened."""
    m = _mem()
    _run(m.run({"command": "create", "path": "/memories/note.md", "file_text": "original"}))
    msg = _run(
        m.run(
            {
                "command": "create",
                "path": "/memories/note.md",
                "file_text": "replacement",
                "overwrite": True,
            }
        )
    )
    assert msg == "replaced /memories/note.md"
    assert _run(m.run({"command": "view", "path": "/memories/note.md"})) == "replacement"


def test_delete_refuses_root_path():
    """``delete("/memories")`` must refuse the root path with a clear
    error rather than prefix-matching every file under it."""
    m = _mem()
    _run(m.run({"command": "create", "path": "/memories/a.md", "file_text": "A"}))
    _run(m.run({"command": "create", "path": "/memories/b.md", "file_text": "B"}))
    with pytest.raises(PermissionError):
        _run(m.run({"command": "delete", "path": "/memories"}))
    # Both files survive.
    assert _run(m.run({"command": "view", "path": "/memories/a.md"})) == "A"
    assert _run(m.run({"command": "view", "path": "/memories/b.md"})) == "B"


def test_delete_subpath_still_works():
    """Regression — the root-guard must not break legitimate subpath
    deletions, including directory prefix deletions."""
    m = _mem()
    _run(m.run({"command": "create", "path": "/memories/bugs/a.md", "file_text": "A"}))
    _run(m.run({"command": "create", "path": "/memories/bugs/b.md", "file_text": "B"}))
    _run(m.run({"command": "create", "path": "/memories/notes/c.md", "file_text": "C"}))
    # Directory-level delete under root is allowed.
    _run(m.run({"command": "delete", "path": "/memories/bugs"}))
    with pytest.raises(FileNotFoundError):
        _run(m.run({"command": "view", "path": "/memories/bugs/a.md"}))
    # Unrelated subtree untouched.
    assert _run(m.run({"command": "view", "path": "/memories/notes/c.md"})) == "C"


def test_file_tool_satisfies_tool_protocol():
    """``isinstance(FileTool(), Tool)`` must return True under
    ``@runtime_checkable``. The previous version was missing
    ``description`` and ``output_schema`` and quietly failed its own
    contract — any caller doing ``isinstance(t, Tool)`` rejected it."""
    from agentkit.tools.base import Tool

    assert isinstance(FileTool(), Tool)


# ── Path confinement is the tool's security boundary ────────────────────────
#
# ``FileTool`` is designed to take an INJECTED backend, so ``_confine`` is what
# stops a filesystem-backed one from becoming an arbitrary read/write/delete
# primitive. The lexical traversal checks were already right; these pin the two
# characters that slipped through and the honesty of the guarantee.


@pytest.mark.parametrize(
    "path",
    [
        "../etc/passwd",
        "/etc/passwd",
        "/memories/../etc/passwd",
        "/memories/./../../etc",
        "..",
        "/",
        "/memoriesX/secret",  # a sibling that merely shares the prefix
        "/memories/a/../../../etc/shadow",
        "/memories/..",
    ],
)
def test_traversal_and_absolute_escapes_are_refused(path: str) -> None:
    with pytest.raises(PermissionError):
        FileTool()._confine(path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("subdir/note.md", "/memories/subdir/note.md"),  # relative → under root
        ("/memories//deep///note.md", "/memories/deep/note.md"),  # collapsed
        ("/memories", "/memories"),  # the root itself is listable
        ("/memories/ok.md", "/memories/ok.md"),
    ],
)
def test_legitimate_paths_survive_normalisation(path: str, expected: str) -> None:
    assert FileTool()._confine(path) == expected


def test_a_backslash_is_refused() -> None:
    r"""``posixpath`` reads ``\..\etc`` as one ordinary filename, so it passed
    the traversal check — and then meant traversal to a backend running on
    Windows. Refusing the character makes the guarantee platform-independent
    rather than true only where it was tested."""
    with pytest.raises(PermissionError, match="backslash"):
        FileTool()._confine("\\..\\etc")
    with pytest.raises(PermissionError, match="backslash"):
        FileTool()._confine("/memories/win\\path")


def test_a_nul_byte_is_refused() -> None:
    """The classic C-level truncation trick. Python's own ``open`` rejects it,
    but the backend is injected and need not be Python's."""
    with pytest.raises(PermissionError, match="NUL"):
        FileTool()._confine("/memories/a\x00b")


def test_rename_confines_both_ends() -> None:
    """The destination is a path too — confining only the source would make
    ``rename`` the escape hatch for everything the other commands refuse."""
    tool = FileTool()
    asyncio.run(tool.run({"command": "create", "path": "/memories/a.md", "file_text": "x"}, None))
    with pytest.raises(PermissionError):
        asyncio.run(
            tool.run({"command": "rename", "path": "/memories/a.md", "new_path": "/etc/evil"}, None)
        )


def test_delete_refuses_to_wipe_the_root() -> None:
    """The backend deletes by prefix, so one ``delete("/memories")`` would erase
    every note. The trailing-slash spelling must be refused too — it normalises
    to the same path."""
    tool = FileTool()
    asyncio.run(tool.run({"command": "create", "path": "/memories/a.md", "file_text": "x"}, None))
    for spelling in ("/memories", "/memories/"):
        with pytest.raises(PermissionError):
            asyncio.run(tool.run({"command": "delete", "path": spelling}, None))
    assert asyncio.run(tool.run({"command": "view", "path": "/memories"}, None)) == "/memories/a.md"
