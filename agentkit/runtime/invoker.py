"""Invoker — the one runner. Sends a unit of work (ChatRequest | ToolRequest) through its composed
middleware chain to a terminal seam call.

The app builds the two chains ONCE (an explicit, ordered list of middlewares) and hands them here.
Patterns then call `ctx.invoker.chat(req)` / `ctx.invoker.invoke_tool(req)` and get all the
cross-cutting behaviour (trace, retry, meter, memoize, fallback, egress, audit) for free — without
re-implementing any of it. Swapping or reordering a concern is editing the list passed to `Invoker`,
not patching a method.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from agentkit.kernel.middleware import Call, chain, collect, collect_one
from agentkit.kernel.types import ChatRequest, Delta, LLMResult, ToolRequest


class Invoker:
    """Runs a unit of work through its composed (stream-shaped) middleware chain to a terminal seam stream.

    `stream()` is the one primitive; `chat()` ≡ `collect(stream())`; `invoke_tool()` collects the one-item
    tool stream. The app builds the two chains once; patterns pick the consumption mode.
    """

    def __init__(
        self, *, llm: Any, chat_middleware: Sequence[Any] = (), tool_middleware: Sequence[Any] = ()
    ):
        self._llm = llm
        # Keep the composed lists addressable. ``chain()`` closes over them
        # and returns an opaque callable, so without this the wiring is
        # write-only — nothing downstream can answer "is output_coerce in
        # this chain?", and a missing middleware becomes a silent behavioural
        # hole instead of a diagnosable one (see ``Agent._warn_if_no_coerce``).
        # Tuples, not lists: introspection must not be able to mutate the
        # chain that was already composed above.
        self.chat_middleware: tuple[Any, ...] = tuple(chat_middleware)
        self.tool_middleware: tuple[Any, ...] = tuple(tool_middleware)
        self._chat = chain(list(chat_middleware), self._terminal_chat)
        self._tool = chain(list(tool_middleware), self._terminal_tool)

    def stream(
        self, request: ChatRequest, ctx: Any, *, meta: dict[str, Any] | None = None
    ) -> AsyncIterator[Delta]:
        # ``meta`` is the typed per-call carrier for middlewares (e.g.
        # output_coerce reads ``call.meta["output_adapter"]``). Smuggling
        # per-call data via ``ctx._output_adapter`` would be
        # concurrency-unsafe: two agents sharing a ctx would stomp it.
        return self._chat(Call("chat", request, ctx, meta=dict(meta) if meta else {}))

    async def chat(
        self, request: ChatRequest, ctx: Any, *, meta: dict[str, Any] | None = None
    ) -> LLMResult:
        result: LLMResult = await collect(
            self._chat(Call("chat", request, ctx, meta=dict(meta) if meta else {})),
            kind="chat",
        )  # ≡ collect(stream)
        return result

    async def invoke_tool(
        self, request: ToolRequest, ctx: Any, *, meta: dict[str, Any] | None = None
    ) -> Any:
        return await collect_one(
            self._tool(Call("tool", request, ctx, meta=dict(meta) if meta else {}))
        )

    # ---- terminals: the actual seam streams -----------------------------------------------

    async def _terminal_chat(self, call: Call) -> AsyncIterator[Delta]:
        r: ChatRequest = call.request
        async for d in self._llm.stream(
            messages=r.messages,
            model=r.model,
            tools=r.tools,
            response_format=r.response_format,
            temperature=r.temperature,
            max_tokens=r.max_tokens,
            cache_hint=r.cache_hint,
        ):
            yield d

    async def _terminal_tool(self, call: Call) -> AsyncIterator[Any]:
        r: ToolRequest = call.request
        yield await r.tool.run(
            r.arguments, call.ctx
        )  # one-item stream (ToolPort.run bridges sync fns)
