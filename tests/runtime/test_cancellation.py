"""CancellationToken (ch18 / R11): cooperative abort, shared down the RunContext tree."""

import pytest

from agentkit.kernel.concurrency import CancellationToken, Cancelled
from agentkit.kernel.types import Scope
from agentkit.runtime.context import RunContext


def test_token_cancel_then_raises():
    t = CancellationToken()
    assert t.cancelled is False
    t.raise_if_cancelled()                 # no-op when not cancelled
    t.cancel()
    assert t.cancelled is True
    with pytest.raises(Cancelled):
        t.raise_if_cancelled()


def test_runcontext_check_cancelled_noop_without_token():
    RunContext("cid", Scope(1, 2)).check_cancelled()   # no token attached → no-op


def test_cancel_is_shared_to_children():
    t = CancellationToken()
    ctx = RunContext("cid", Scope(1, 2), cancel=t)
    child = ctx.child()
    assert child.cancel is t               # shared across the tree
    t.cancel()                             # cancelling the parent…
    with pytest.raises(Cancelled):
        child.check_cancelled()            # …cancels the child
    with pytest.raises(Cancelled):
        ctx.check_cancelled()
