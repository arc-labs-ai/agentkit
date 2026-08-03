"""PrefixContext — the cache-stable head of a WorkingContext.

The prefix is frozen by construction so it stays bit-identical
across turns (KV-cache discipline). These tests pin:

  1. ``frozen=True`` is honoured — attempting to mutate any field
     raises ``FrozenInstanceError``.
  2. ``as_messages()`` orders the prefix as
     ``[system_prompt] + grounding + [schema_block]`` (any field
     may be absent).
  3. Equality is structural — two prefixes built from the same
     inputs are equal AND hash equal, so a prefix can key a recall
     cache without identity pitfalls.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agentkit.context import PrefixContext
from agentkit.kernel.types import Message


def test_prefix_is_frozen_assignment_raises():
    p = PrefixContext(system_prompt="seed")
    with pytest.raises(FrozenInstanceError):
        p.grounding = ()  # type: ignore[misc]


def test_as_messages_orders_system_prompt_grounding_schema():
    p = PrefixContext(
        system_prompt="seed",
        grounding=(Message("system", "ground-1"), Message("system", "ground-2")),
        schema_block="schema-json",
    )
    msgs = p.as_messages()
    assert [m.role for m in msgs] == ["system", "system", "system", "system"]
    assert [m.content for m in msgs] == ["seed", "ground-1", "ground-2", "schema-json"]


def test_as_messages_skips_empty_system_prompt():
    p = PrefixContext(grounding=(Message("system", "only-ground"),))
    msgs = p.as_messages()
    assert [m.content for m in msgs] == ["only-ground"]


def test_as_messages_skips_none_schema_block():
    p = PrefixContext(system_prompt="seed")
    msgs = p.as_messages()
    assert len(msgs) == 1
    assert msgs[0].content == "seed"


def test_as_messages_returns_fresh_list():
    p = PrefixContext(system_prompt="seed")
    a = p.as_messages()
    b = p.as_messages()
    assert a == b and a is not b  # mutation safety: caller-owned list


def test_prefix_equality_is_structural():
    p1 = PrefixContext(system_prompt="seed", grounding=(Message("system", "x"),))
    p2 = PrefixContext(system_prompt="seed", grounding=(Message("system", "x"),))
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_prefix_inequality_when_inputs_differ():
    p1 = PrefixContext(system_prompt="seed-A")
    p2 = PrefixContext(system_prompt="seed-B")
    assert p1 != p2
