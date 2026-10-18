"""Host implementations; no provider or MCP dependencies."""

from collections.abc import Mapping

from agentkit.core import Scope


def lookup_handler(records: Mapping[str, str], scope: Scope, effects: list[str]):
    collection = dict(records)

    async def lookup_record(ctx, record_id: str) -> dict:
        """Read a status from this call's authorized record collection."""
        if ctx.scope != scope:
            raise PermissionError("record scope mismatch")
        effects.append(record_id)
        return {"record_id": record_id, "status": collection[record_id]}

    return lookup_record
