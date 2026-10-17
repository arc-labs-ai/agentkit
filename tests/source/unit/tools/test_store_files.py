"""``StoreFiles`` — durable virtual file trees over a host ``StorePort``.

11% covered before this file. The reason is structural rather than neglect:
no concrete ``StorePort`` ships in the package any more, so there was nothing
to construct a ``StoreFiles`` with. ``StorePort`` is a Protocol, so the right
answer is a faithful double — faithful being the load-bearing word, since a
permissive store would let every test below pass without proving anything.

``_Store`` therefore implements the documented contract rather than a
convenient approximation: comparison by EQUALITY (three of the four real
backends round-trip through JSON, so the caller never holds the object that
was written), an absent key comparing equal to ``expected=None``, and a
deep copy on write so a caller mutating the dict it passed cannot reach
inside the store afterwards.

The invariants worth having here are the confinement ones. This backend takes
paths chosen by a MODEL, and every one of them is a chance to leave the
namespace it was given.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from agentkit.tools.filesystem.store import StoreFiles

ROOT = "/memories"


class _Store:
    """An in-memory StorePort faithful to the documented CAS contract."""

    def __init__(self, *, lose: int = 0) -> None:
        self._data: dict[str, Any] = {}
        self.lose = lose  # fail this many compare_and_set calls first
        self.attempts = 0

    async def get(self, key: str) -> Any | None:
        return copy.deepcopy(self._data.get(key))

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        self._data[key] = copy.deepcopy(value)

    async def compare_and_set(
        self, key: str, expected: Any, value: Any, *, ttl: int | None = None
    ) -> bool:
        self.attempts += 1
        if self.lose > 0:
            self.lose -= 1
            return False
        if self._data.get(key) != expected:  # equality, never identity
            return False
        self._data[key] = copy.deepcopy(value)
        return True


def _files(store: Any = None, **kwargs: Any) -> StoreFiles:
    return StoreFiles(store or _Store(), namespace="tenant-a", **kwargs)


# ── construction ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("namespace", ["", "   ", "has\x00nul"])
def test_a_usable_namespace_is_mandatory(namespace: str) -> None:
    """The namespace is the tenant boundary and the host supplies it. An empty
    or NUL-bearing one would collapse tenants into a shared key."""
    with pytest.raises(ValueError, match="namespace"):
        StoreFiles(_Store(), namespace=namespace)


@pytest.mark.parametrize("bounds", [{"max_bytes": 0}, {"max_files": 0}, {"max_bytes": -1}])
def test_bounds_must_be_positive(bounds: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="bounds must be positive"):
        StoreFiles(_Store(), namespace="t", **bounds)


def test_the_memory_root_cannot_be_the_whole_tree() -> None:
    """A root of "/" would make the confinement check in `_path` vacuous —
    every path starts with "/"."""
    with pytest.raises(ValueError, match="non-root"):
        StoreFiles(_Store(), namespace="t", root="/")


def test_two_namespaces_do_not_share_a_tree() -> None:
    store = _Store()
    a, b = (
        StoreFiles(store, namespace="tenant-a"),
        StoreFiles(store, namespace="tenant-b"),
    )
    assert a._key != b._key


# ── path safety ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "why"),
    [
        ("memories/a.txt", "relative"),
        ("C:\\memories\\a.txt", "backslash — a Windows path is not a virtual one"),
        ("/memories/a\x00.txt", "NUL truncates in C APIs"),
        ("/memories/../etc/passwd", "parent traversal"),
        ("/memories/./a.txt", "same-directory alias"),
        ("/memories//a.txt", "empty component"),
    ],
)
def test_the_path_grammar_refuses_aliases_and_traversal(path: str, why: str) -> None:
    """These paths come from a MODEL. Each spelling below is a way to name
    something outside the tree while looking like it is inside."""
    with pytest.raises(ValueError, match="paths must be|aliases and traversal"):
        _files()._path(path)


@pytest.mark.parametrize("path", ["/elsewhere/a.txt", "/memoriesevil/a.txt", "/"])
def test_a_path_outside_the_root_is_refused(path: str) -> None:
    """`/memoriesevil` is the one worth having: a prefix check without the
    separator admits any directory whose name merely starts with the root."""
    with pytest.raises(PermissionError, match="outside the memory root"):
        _files()._path(path)


def test_the_root_itself_and_its_descendants_are_inside() -> None:
    files = _files()
    assert files._path(ROOT) == ROOT
    assert files._path(f"{ROOT}/nested/a.txt") == f"{ROOT}/nested/a.txt"


def test_root_is_addressable_only_for_a_listing() -> None:
    assert _files()._path("/", listing=True) == "/"


# ── create ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_then_view_round_trips() -> None:
    files = _files()
    assert await files.create(f"{ROOT}/a.txt", "hello") == f"created {ROOT}/a.txt"
    assert await files.view(f"{ROOT}/a.txt") == "hello"


@pytest.mark.asyncio
async def test_create_refuses_to_clobber_unless_asked() -> None:
    files = _files()
    await files.create(f"{ROOT}/a.txt", "first")
    with pytest.raises(FileExistsError):
        await files.create(f"{ROOT}/a.txt", "second")
    assert await files.view(f"{ROOT}/a.txt") == "first"

    assert await files.create(f"{ROOT}/a.txt", "second", overwrite=True) == (
        f"replaced {ROOT}/a.txt"
    )
    assert await files.view(f"{ROOT}/a.txt") == "second"


@pytest.mark.asyncio
async def test_the_root_is_a_directory_not_a_file() -> None:
    with pytest.raises(ValueError, match="leaf file"):
        await _files().create(ROOT, "content")


@pytest.mark.asyncio
async def test_non_string_content_is_refused() -> None:
    with pytest.raises(ValueError, match="string content"):
        await _files().create(f"{ROOT}/a.txt", 42)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_file_cannot_shadow_a_directory_or_vice_versa() -> None:
    """The tree is a flat path->content map, so nothing structural stops
    `/memories/a.txt` and `/memories/a.txt/b.txt` from both existing. One of
    them would then be unreachable through `view`."""
    files = _files()
    await files.create(f"{ROOT}/dir/b.txt", "inner")
    with pytest.raises(ValueError, match="collision"):
        await files.create(f"{ROOT}/dir", "now a file")

    await files.create(f"{ROOT}/leaf.txt", "leaf")
    with pytest.raises(ValueError, match="collision"):
        await files.create(f"{ROOT}/leaf.txt/child.txt", "child of a file")


# ── view ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_viewing_a_directory_lists_its_descendants_sorted() -> None:
    files = _files()
    for name in ("c.txt", "a.txt", "b.txt"):
        await files.create(f"{ROOT}/{name}", name)
    assert await files.view(ROOT) == "\n".join(
        [f"{ROOT}/a.txt", f"{ROOT}/b.txt", f"{ROOT}/c.txt"]
    )


@pytest.mark.asyncio
async def test_an_empty_root_lists_as_empty_rather_than_missing() -> None:
    """The root always exists conceptually, even with nothing in it. Raising
    FileNotFoundError for an empty tree would make a fresh tenant look broken."""
    assert await _files().view(ROOT) == ""
    assert await _files().view("/") == ""


@pytest.mark.asyncio
async def test_viewing_something_that_does_not_exist_raises() -> None:
    with pytest.raises(FileNotFoundError):
        await _files().view(f"{ROOT}/missing.txt")


# ── str_replace ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_str_replace_changes_the_first_occurrence_only() -> None:
    files = _files()
    await files.create(f"{ROOT}/a.txt", "x y x")
    assert await files.str_replace(f"{ROOT}/a.txt", "x", "z") == f"edited {ROOT}/a.txt"
    assert await files.view(f"{ROOT}/a.txt") == "z y x"


@pytest.mark.asyncio
async def test_an_empty_old_str_is_refused() -> None:
    """`"".replace` matches at every position, so an empty needle would insert
    rather than replace — a silent corruption of the whole file."""
    files = _files()
    await files.create(f"{ROOT}/a.txt", "body")
    with pytest.raises(ValueError, match="must not be empty"):
        await files.str_replace(f"{ROOT}/a.txt", "", "z")


@pytest.mark.asyncio
async def test_str_replace_reports_a_missing_file_and_a_missing_needle_differently() -> None:
    files = _files()
    with pytest.raises(FileNotFoundError):
        await files.str_replace(f"{ROOT}/missing.txt", "a", "b")
    await files.create(f"{ROOT}/a.txt", "body")
    with pytest.raises(ValueError, match="old_str not found"):
        await files.str_replace(f"{ROOT}/a.txt", "absent", "b")


# ── insert ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "expected"),
    [(0, "new\na\nb"), (1, "a\nnew\nb"), (2, "a\nb\nnew")],
)
async def test_insert_places_a_line_at_each_valid_boundary(line: int, expected: str) -> None:
    files = _files()
    await files.create(f"{ROOT}/a.txt", "a\nb")
    await files.insert(f"{ROOT}/a.txt", line, "new")
    assert await files.view(f"{ROOT}/a.txt") == expected


@pytest.mark.asyncio
async def test_insert_strips_a_trailing_newline_so_one_call_adds_one_line() -> None:
    files = _files()
    await files.create(f"{ROOT}/a.txt", "a")
    await files.insert(f"{ROOT}/a.txt", 1, "new\n")
    assert await files.view(f"{ROOT}/a.txt") == "a\nnew"


@pytest.mark.asyncio
@pytest.mark.parametrize("line", [-1, 3, True, 1.0, "1"])
async def test_an_out_of_range_or_non_integer_line_is_refused(line: Any) -> None:
    """`True` is the interesting one: `isinstance(True, int)` is True in Python,
    so a bool would otherwise be accepted as line 1."""
    files = _files()
    await files.create(f"{ROOT}/a.txt", "a\nb")
    with pytest.raises(ValueError, match="insert_line"):
        await files.insert(f"{ROOT}/a.txt", line, "new")


@pytest.mark.asyncio
async def test_insert_into_a_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        await _files().insert(f"{ROOT}/missing.txt", 0, "new")


# ── delete ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_removes_a_file() -> None:
    files = _files()
    await files.create(f"{ROOT}/a.txt", "x")
    assert await files.delete(f"{ROOT}/a.txt") == f"deleted {ROOT}/a.txt"
    with pytest.raises(FileNotFoundError):
        await files.view(f"{ROOT}/a.txt")


@pytest.mark.asyncio
async def test_deleting_a_directory_removes_its_whole_subtree() -> None:
    files = _files()
    await files.create(f"{ROOT}/dir/a.txt", "a")
    await files.create(f"{ROOT}/dir/nested/b.txt", "b")
    await files.create(f"{ROOT}/keep.txt", "keep")

    await files.delete(f"{ROOT}/dir")

    assert await files.view(ROOT) == f"{ROOT}/keep.txt"


@pytest.mark.asyncio
async def test_deleting_nothing_raises_rather_than_succeeding_silently() -> None:
    with pytest.raises(FileNotFoundError):
        await _files().delete(f"{ROOT}/missing")


# ── rename ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rename_moves_content_and_leaves_nothing_behind() -> None:
    files = _files()
    await files.create(f"{ROOT}/a.txt", "body")
    assert await files.rename(f"{ROOT}/a.txt", f"{ROOT}/b.txt") == (
        f"renamed {ROOT}/a.txt -> {ROOT}/b.txt"
    )
    assert await files.view(f"{ROOT}/b.txt") == "body"
    with pytest.raises(FileNotFoundError):
        await files.view(f"{ROOT}/a.txt")


@pytest.mark.asyncio
async def test_rename_will_not_overwrite_or_collide() -> None:
    files = _files()
    await files.create(f"{ROOT}/a.txt", "a")
    await files.create(f"{ROOT}/b.txt", "b")
    with pytest.raises(ValueError, match="exists or collides"):
        await files.rename(f"{ROOT}/a.txt", f"{ROOT}/b.txt")

    await files.create(f"{ROOT}/dir/c.txt", "c")
    with pytest.raises(ValueError, match="exists or collides"):
        await files.rename(f"{ROOT}/a.txt", f"{ROOT}/dir")
    with pytest.raises(ValueError, match="exists or collides"):
        await files.rename(f"{ROOT}/a.txt", ROOT)


@pytest.mark.asyncio
async def test_renaming_a_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        await _files().rename(f"{ROOT}/missing.txt", f"{ROOT}/b.txt")


@pytest.mark.asyncio
async def test_rename_cannot_escape_the_root() -> None:
    files = _files()
    await files.create(f"{ROOT}/a.txt", "a")
    with pytest.raises(PermissionError, match="outside the memory root"):
        await files.rename(f"{ROOT}/a.txt", "/elsewhere/a.txt")


# ── bounds ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_file_count_bound_is_enforced() -> None:
    files = _files(max_files=2)
    await files.create(f"{ROOT}/a.txt", "a")
    await files.create(f"{ROOT}/b.txt", "b")
    with pytest.raises(ValueError, match="exceeds configured bounds"):
        await files.create(f"{ROOT}/c.txt", "c")


@pytest.mark.asyncio
async def test_the_byte_bound_is_enforced() -> None:
    files = _files(max_bytes=64)
    with pytest.raises(ValueError, match="exceeds configured bounds"):
        await files.create(f"{ROOT}/a.txt", "x" * 500)


# ── the optimistic loop ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_lost_compare_and_set_is_retried_not_raised() -> None:
    """Losing a CAS is the expected half of an optimistic loop, not a fault.
    The caller asked for one write and must get one write."""
    store = _Store(lose=3)
    files = StoreFiles(store, namespace="t")
    assert await files.create(f"{ROOT}/a.txt", "x") == f"created {ROOT}/a.txt"
    assert store.attempts == 4


@pytest.mark.asyncio
async def test_unbounded_contention_eventually_gives_up_loudly() -> None:
    """A retry loop with no bound is a hang. Sixty-four losses is not
    contention any more, it is a store that will never accept the write."""
    store = _Store(lose=10_000)
    files = StoreFiles(store, namespace="t")
    with pytest.raises(RuntimeError, match="contention exceeded"):
        await files.create(f"{ROOT}/a.txt", "x")


# ── the stored shape ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored",
    [
        ["not", "a", "map"],
        {"/memories/a.txt": 42},
        {7: "non-string key"},
        "a bare string",
    ],
)
async def test_a_corrupt_stored_tree_is_refused_rather_than_coerced(stored: Any) -> None:
    """The tree is shared durable state. Coercing a malformed document would
    silently discard whatever a previous writer put there."""
    store = _Store()
    files = StoreFiles(store, namespace="t")
    await store.set(files._key, stored)
    with pytest.raises(ValueError, match="not a string map"):
        await files.view(f"{ROOT}/a.txt")


@pytest.mark.asyncio
async def test_a_stored_tree_holding_the_root_as_a_file_is_refused() -> None:
    store = _Store()
    files = StoreFiles(store, namespace="t")
    await store.set(files._key, {ROOT: "the root as a file"})
    with pytest.raises(ValueError, match="directory as a file"):
        await files.view(f"{ROOT}/a.txt")
