"""StorePort adapters — ONE durable-KV seam backing checkpoints, idempotency, audit, and cache.

`InMemoryStore` is the offline reference + contract (single-flight `get_or_set`, never-negative on a
raised producer, append-only logs). `FileStore` is the zero-dependency durable adapter (survives a
restart). `RedisStore`/`PostgresStore` (behind extras) are the same shape over real infra. One KV seam
serves checkpoints, idempotency, the audit log, and the result cache.

One backend per module (single responsibility); the public surface is `agentkit.adapters.store`.
"""

from agentkit.adapters.store.file import FileStore
from agentkit.adapters.store.memory import InMemoryStore
from agentkit.adapters.store.postgres import PostgresStore
from agentkit.adapters.store.redis import RedisStore

__all__ = ["InMemoryStore", "FileStore", "RedisStore", "PostgresStore"]
