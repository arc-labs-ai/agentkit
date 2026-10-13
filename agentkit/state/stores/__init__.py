"""StorePort adapters — ONE durable-KV seam backing checkpoints, idempotency, audit, and cache.

`InMemoryStore` is the offline reference + contract (single-flight `get_or_set`, never-negative on a
raised producer, append-only logs). `FileStore` is the zero-dependency durable adapter (survives a
restart). `RedisStore`/`PostgresStore` (behind extras) are the same shape over real infra. One KV seam
serves checkpoints, idempotency, the audit log, and the result cache.

One backend per module (single responsibility); the public surface is `agentkit.state.stores`.
"""

from agentkit.state.stores.file import FileStore
from agentkit.state.stores.memory import InMemoryStore
from agentkit.state.stores.postgres import PostgresStore
from agentkit.state.stores.redis import RedisStore

__all__ = ["InMemoryStore", "FileStore", "RedisStore", "PostgresStore"]
