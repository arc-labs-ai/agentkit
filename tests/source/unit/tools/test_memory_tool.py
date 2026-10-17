"""``FileTool`` / ``InMemoryFiles`` — the model-driven ``memory`` tool.

`test_file_tool.py` next door pins the vendor CONTRACT (the name, the command
set, the call shape, the root). This file pins the BEHAVIOUR, which had no
tests: the backend operations and, more importantly, the confinement in
`FileTool._confine`.

Confinement is the part that matters. This tool takes paths chosen by a model
and hands them to an injected backend, so a hole here turns the backend into
an arbitrary read/write/delete primitive. Five catalogue entries were parked
pointing at this file, and every one of them is a way out of the root.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentkit.tools.filesystem.files import FileTool, InMemoryFiles

ROOT = "/memories"


async def _run(tool: FileTool, **args: Any) -> str:
    return await tool.run(args, None)


# ── confinement: the five ways out ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_backslash_never_reaches_the_backend() -> None:
    r"""`posixpath` treats ``\..\etc`` as ONE ordinary filename, so it
    normalises to something that looks perfectly confined — and then means
    traversal on a backend running on Windows. Refusing the character makes
    the guarantee platform-independent instead of true only where it was
    tested."""
    with pytest.raises(PermissionError, match="backslash"):
        await _run(FileTool(), command="view", path=r"\..\etc\passwd")


@pytest.mark.asyncio
async def test_a_nul_byte_never_reaches_the_backend() -> None:
    """The classic C-level truncation trick. Python's own ``open()`` raises on
    it, but the backend is INJECTED and need not be Python's."""
    with pytest.raises(PermissionError, match="NUL"):
        await _run(FileTool(), command="view", path=f"{ROOT}/a\x00.md")


@pytest.mark.asyncio
async def test_a_sibling_sharing_the_roots_prefix_is_not_inside_it() -> None:
    """The separator is the whole check. ``startswith(self.root)`` without the
    trailing "/" admits ``/memoriesevil``, and that directory is one `mkdir`
    away on any filesystem-backed implementation."""
    with pytest.raises(PermissionError, match="escapes root"):
        await _run(FileTool(), command="view", path="/memoriesevil/a.md")


@pytest.mark.asyncio
async def test_rename_confines_its_destination_as_well_as_its_source() -> None:
    """Confining only the source makes rename the escape hatch: read a note
    inside the root, write it anywhere. Both ends go through `_confine`."""
    tool = FileTool()
    await _run(tool, command="create", path=f"{ROOT}/a.md", file_text="secret")
    with pytest.raises(PermissionError, match="escapes root"):
        await _run(tool, command="rename", path=f"{ROOT}/a.md", new_path="/etc/exfiltrated.md")
    assert await _run(tool, command="view", path=f"{ROOT}/a.md") == "secret"


@pytest.mark.asyncio
async def test_delete_refuses_to_wipe_the_root() -> None:
    """The backend deletes by PREFIX, so ``delete("/memories")`` would erase
    every note in one call. A specific subpath or file is still deletable."""
    tool = FileTool()
    await _run(tool, command="create", path=f"{ROOT}/keep.md", file_text="keep")
    with pytest.raises(PermissionError, match="refuses to wipe root"):
        await _run(tool, command="delete", path=ROOT)
    assert await _run(tool, command="view", path=f"{ROOT}/keep.md") == "keep"


# ── confinement: the rest of the grammar ─────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [f"{ROOT}/../etc/passwd", "/etc/passwd", f"{ROOT}/a/../../b"])
async def test_traversal_and_absolute_escapes_are_refused(path: str) -> None:
    with pytest.raises(PermissionError, match="escapes root"):
        await _run(FileTool(), command="view", path=path)


@pytest.mark.asyncio
async def test_a_relative_path_is_resolved_against_the_root() -> None:
    """Convenience for the model, not a second grammar: it lands inside the
    root like everything else."""
    tool = FileTool()
    await _run(tool, command="create", path="notes/a.md", file_text="body")
    assert await _run(tool, command="view", path=f"{ROOT}/notes/a.md") == "body"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [None, "", 0])
async def test_a_missing_path_is_refused_before_dispatch(path: Any) -> None:
    with pytest.raises(ValueError, match="'path' is required"):
        await _run(FileTool(), command="view", path=path)


@pytest.mark.asyncio
async def test_the_root_itself_is_addressable() -> None:
    assert await _run(FileTool(), command="view", path=ROOT) == ""


# ── dispatch ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_unknown_command_is_refused_and_lists_the_real_ones() -> None:
    """The command set is a wire contract. A model that invents one gets a
    message naming the six, which is what lets it correct itself."""
    with pytest.raises(ValueError, match="unknown memory command"):
        await _run(FileTool(), command="chmod", path=ROOT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "args", "missing"),
    [
        ("create", {}, "file_text"),
        ("str_replace", {"old_str": "a"}, "new_str"),
        ("str_replace", {"new_str": "b"}, "old_str"),
        ("insert", {"insert_line": 0}, "insert_text"),
        ("insert", {"insert_text": "x"}, "insert_line"),
        ("rename", {}, "new_path"),
    ],
)
async def test_a_missing_required_argument_names_what_is_missing(
    command: str, args: dict[str, Any], missing: str
) -> None:
    with pytest.raises(ValueError, match=missing):
        await _run(FileTool(), command=command, path=f"{ROOT}/a.md", **args)


@pytest.mark.asyncio
async def test_the_whole_command_set_round_trips() -> None:
    tool = FileTool()
    assert await _run(tool, command="create", path=f"{ROOT}/a.md", file_text="one\ntwo")
    assert await _run(tool, command="str_replace", path=f"{ROOT}/a.md", old_str="one", new_str="1")
    assert await _run(
        tool, command="insert", path=f"{ROOT}/a.md", insert_line=1, insert_text="mid"
    )
    assert await _run(tool, command="view", path=f"{ROOT}/a.md") == "1\nmid\ntwo"
    assert await _run(tool, command="rename", path=f"{ROOT}/a.md", new_path=f"{ROOT}/b.md")
    assert await _run(tool, command="delete", path=f"{ROOT}/b.md")
    assert await _run(tool, command="view", path=ROOT) == ""


@pytest.mark.asyncio
async def test_insert_line_is_coerced_so_a_stringified_number_still_works() -> None:
    """Models emit JSON, and a number sometimes arrives as a string. `run`
    coerces with int() rather than refusing a well-formed intent."""
    tool = FileTool()
    await _run(tool, command="create", path=f"{ROOT}/a.md", file_text="a")
    await _run(tool, command="insert", path=f"{ROOT}/a.md", insert_line="1", insert_text="b")
    assert await _run(tool, command="view", path=f"{ROOT}/a.md") == "a\nb"


@pytest.mark.asyncio
async def test_create_refuses_to_clobber_unless_overwrite_is_requested() -> None:
    """Default refuse-clobber, because an unconditional write actively lies
    about what happened when the path already held a note from a prior run."""
    tool = FileTool()
    await _run(tool, command="create", path=f"{ROOT}/a.md", file_text="first")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        await _run(tool, command="create", path=f"{ROOT}/a.md", file_text="second")
    out = await _run(
        tool, command="create", path=f"{ROOT}/a.md", file_text="second", overwrite=True
    )
    assert out.startswith("replaced")
    assert await _run(tool, command="view", path=f"{ROOT}/a.md") == "second"


@pytest.mark.asyncio
async def test_a_custom_root_is_confined_to_itself() -> None:
    tool = FileTool(root="notes")
    await _run(tool, command="create", path="/notes/a.md", file_text="x")
    with pytest.raises(PermissionError, match="escapes root"):
        await _run(tool, command="view", path=f"{ROOT}/a.md")


# ── the in-memory backend ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_backend_lists_a_directory_and_reads_a_file() -> None:
    fs = InMemoryFiles()
    await fs.create(f"{ROOT}/b.md", "b")
    await fs.create(f"{ROOT}/a.md", "a")
    assert await fs.view(f"{ROOT}/a.md") == "a"
    assert await fs.view(ROOT) == f"{ROOT}/a.md\n{ROOT}/b.md"


@pytest.mark.asyncio
async def test_an_empty_root_lists_as_empty_but_a_missing_path_raises() -> None:
    fs = InMemoryFiles()
    assert await fs.view(ROOT) == ""
    with pytest.raises(FileNotFoundError):
        await fs.view(f"{ROOT}/missing.md")


@pytest.mark.asyncio
async def test_str_replace_touches_only_the_first_occurrence() -> None:
    fs = InMemoryFiles()
    await fs.create(f"{ROOT}/a.md", "x y x")
    await fs.str_replace(f"{ROOT}/a.md", "x", "z")
    assert await fs.view(f"{ROOT}/a.md") == "z y x"


@pytest.mark.asyncio
async def test_str_replace_reports_an_absent_needle() -> None:
    fs = InMemoryFiles()
    await fs.create(f"{ROOT}/a.md", "body")
    with pytest.raises(ValueError, match="old_str not found"):
        await fs.str_replace(f"{ROOT}/a.md", "absent", "z")


@pytest.mark.asyncio
async def test_insert_splits_on_newline_so_a_trailing_blank_line_survives() -> None:
    """`split("\\n")` rather than `splitlines()`: the latter discards the final
    empty element, so a file ending in a newline would quietly lose it on every
    edit."""
    fs = InMemoryFiles()
    await fs.create(f"{ROOT}/a.md", "a\n")
    await fs.insert(f"{ROOT}/a.md", 1, "b")
    assert await fs.view(f"{ROOT}/a.md") == "a\nb\n"


@pytest.mark.asyncio
async def test_insert_out_of_range_names_the_valid_range() -> None:
    fs = InMemoryFiles()
    await fs.create(f"{ROOT}/a.md", "a\nb")
    with pytest.raises(ValueError, match="out of range"):
        await fs.insert(f"{ROOT}/a.md", 99, "x")


@pytest.mark.asyncio
async def test_delete_removes_a_subtree_by_prefix() -> None:
    fs = InMemoryFiles()
    await fs.create(f"{ROOT}/dir/a.md", "a")
    await fs.create(f"{ROOT}/dir/b.md", "b")
    await fs.create(f"{ROOT}/keep.md", "keep")
    await fs.delete(f"{ROOT}/dir")
    assert await fs.view(ROOT) == f"{ROOT}/keep.md"


@pytest.mark.asyncio
async def test_deleting_nothing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        await InMemoryFiles().delete(f"{ROOT}/missing")


@pytest.mark.asyncio
async def test_rename_will_not_clobber_an_existing_note() -> None:
    fs = InMemoryFiles()
    await fs.create(f"{ROOT}/a.md", "a")
    await fs.create(f"{ROOT}/b.md", "b")
    with pytest.raises(ValueError, match="already exists"):
        await fs.rename(f"{ROOT}/a.md", f"{ROOT}/b.md")
    assert await fs.view(f"{ROOT}/b.md") == "b"


@pytest.mark.asyncio
async def test_renaming_a_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        await InMemoryFiles().rename(f"{ROOT}/missing.md", f"{ROOT}/b.md")


@pytest.mark.asyncio
async def test_an_injected_backend_is_used_instead_of_the_default() -> None:
    """The point of the seam: a durable backend replaces the in-memory one and
    still goes through the same confinement."""
    backend = InMemoryFiles()
    tool = FileTool(backend=backend)
    await _run(tool, command="create", path=f"{ROOT}/a.md", file_text="x")
    assert await backend.view(f"{ROOT}/a.md") == "x"
