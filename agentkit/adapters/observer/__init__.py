"""Concrete observer adapters for the observation channel.

- `CollectingObserver` / `QueueObserver` — terminal sinks (list capture; bounded async-consumable queue).
- `PolicyObserver` / `RollupObserver` — emission-*cadence* wrappers around an inner observer.
- `Hooks` — lifecycle subscription (`on(stage, handler)`), sugar over the stream.

In-process transports. A cross-process transport (e.g. Redis pub/sub → SSE) is a future adapter on the
same `ObserverPort`. The public surface is `agentkit.adapters.observer`.
"""

from agentkit.adapters.observer.cadence import PolicyObserver, RollupObserver
from agentkit.adapters.observer.hooks import Hooks
from agentkit.adapters.observer.sinks import CollectingObserver, QueueObserver

__all__ = ["CollectingObserver", "QueueObserver", "PolicyObserver", "RollupObserver", "Hooks"]
