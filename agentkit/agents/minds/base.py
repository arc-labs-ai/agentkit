"""Mind is a reasoning strategy above the model-neutral call runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentkit.core.runtime.calls import CallRuntime
    from agentkit.models.contracts.types import Answer, Binding


@runtime_checkable
class Mind(Protocol):
    """Choose how to reason; leave transport and enforcement to CallRuntime.

    Minds never import vendor SDKs or subprocess adapters. Harnesses can inject
    a different implementation without changing bindings or tool execution.
    """

    @property
    def name(self) -> str: ...

    async def run(
        self, runtime: CallRuntime, binding: Binding, prompt: Any, tools: Any = None, **kwargs: Any
    ) -> Answer[Any]: ...
