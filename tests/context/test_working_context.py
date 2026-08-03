"""WorkingContext — pure data operations on an agent's transcript + scratchpad.

Orchestration-flavored methods (`summarise`, `ground`) belong to the
callers that wire in a `Compactor` / `Retriever` — they are policies
mixing with state and are not part of `WorkingContext`. The tests
below cover what `WorkingContext` *is* responsible for — append,
note/get, fork-is-independent, and merge — nothing more.
"""

import asyncio

from agentkit.context import WorkingContext
from agentkit.kernel.types import Message


def run(coro):
    return asyncio.run(coro)


def msg(role="user", content=""):
    return Message(role, content)


def test_append_note_get():
    wc = WorkingContext().append(msg(content="hi")).note("topic", "redis")
    assert len(wc.messages) == 1 and wc.get("topic") == "redis" and wc.get("missing", 0) == 0


def test_fork_is_independent():
    base = WorkingContext().append(msg(content="a")).note("k", 1)
    fork = base.fork()
    fork.append(msg(content="b")).note("k", 2)
    assert len(base.messages) == 1 and base.get("k") == 1  # original untouched
    assert len(fork.messages) == 2 and fork.get("k") == 2


def test_merge_aggregates_child_into_parent():
    parent = WorkingContext().append(msg(content="p")).note("a", 1)
    child = WorkingContext().append(msg(content="c")).note("b", 2)
    parent.merge(child)
    assert [m.content for m in parent.messages] == ["p", "c"]
    assert parent.scratchpad == {"a": 1, "b": 2}
