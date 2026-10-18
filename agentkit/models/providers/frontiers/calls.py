"""Transport adapters for the four call modes.

No repair, retry, pricing or schema policy lives here. API history preserves
vendor output blocks verbatim so reasoning signatures survive continuation.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from agentkit.core.contracts.types import Mode
from agentkit.models.adapters.messages import anthropic_history
from agentkit.models.contracts.constants import EventKind, FailureClass
from agentkit.models.contracts.failures import classify
from agentkit.models.contracts.types import Binding, Failure, SessionHandle, Turn, resolve_credential
from agentkit.models.providers._events import _emit
from agentkit.models.providers.auth import anthropic_credential
from agentkit.models.providers.endpoints import ApiEndpoint, ApiPath


class ApiAdapter:
    """HTTPS/SSE transport; HTTP client's retries are zero by construction."""

    def __init__(self, *, client: Any = None, request_hook: Any = None, base_url: str | None = None) -> None:
        self.client = client
        self.request_hook = request_hook
        self.base_url = base_url

    async def turn(
        self,
        binding: Binding,
        prompt: Any,
        *,
        settings: Any,
        limits: Any,
        schema: Any,
        tools: Any,
        session: SessionHandle | None,
        cancel: Any,
        emit: Any,
        call_id: str = "",
        native_agents: Any = None,
        workspace: Any = None,
    ) -> Turn:
        if native_agents or workspace:
            return Turn(
                failure=Failure(
                    FailureClass.INVALID_REQUEST,
                    mode_detail={"reason": "CLI workspace features are unavailable on API modes"},
                )
            )
        from agentkit.models.providers.frontiers.base import HttpLLM

        anthropic = binding.mode == Mode.ANTHROPIC_API
        credential = await resolve_credential(binding.credential)
        if credential.home:
            raise ValueError("frontier API credentials must be supplied as environment values")
        credentials = credential.env
        auth = anthropic_credential(credentials) if anthropic else None
        key = auth.secret if auth else credentials.get("OPENAI_API_KEY") if not anthropic else None
        if not key:
            return Turn(failure=Failure(FailureClass.AUTH))

        class Transport(HttpLLM):
            def _headers(self) -> dict[str, str]:
                if anthropic:
                    return {
                        **(auth.headers() if auth else {}),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    }
                return {"Authorization": f"Bearer {key}", "content-type": "application/json"}

        transport = Transport(
            api_key=key,
            client=self.client,
            request_hook=self.request_hook,
            base_url=self.base_url or (ApiEndpoint.ANTHROPIC if anthropic else ApiEndpoint.OPENAI),
            timeout=limits.wall_clock_s or 180,
        )
        history = [dict(x) for x in session.history] if session else []
        user_content: Any = prompt.user
        if prompt.images:
            if anthropic:
                user_content = [{"type": "text", "text": prompt.user}]
                user_content.extend(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.mime_type,
                            "data": image.data_url().split(",", 1)[1],
                        },
                    }
                    for image in prompt.images
                )
            else:
                user_content = [{"type": "input_text", "text": prompt.user}]
                user_content.extend(
                    {"type": "input_image", "image_url": image.data_url()} for image in prompt.images
                )
        if not history:
            history.append({"role": "user", "content": user_content})
        elif prompt.images and history[-1].get("role") == "user" and history[-1].get("content") == prompt.user:
            history[-1] = {**history[-1], "content": user_content}
        opts = {k: v for k, v in asdict(settings).items() if v is not None}
        payload: dict[str, Any] = {"model": binding.model, "stream": True}
        cap = settings.max_output_tokens or limits.max_output_tokens or 4096
        if anthropic:
            history = list(anthropic_history(history))
            payload.update(messages=history, max_tokens=cap)
            if prompt.system:
                cache_control = {"type": "ephemeral"}
                if settings.cache_ttl is not None:
                    cache_control["ttl"] = settings.cache_ttl.value
                payload["system"] = [{"type": "text", "text": prompt.system, "cache_control": cache_control}]
            for name in ("temperature", "top_p", "top_k"):
                if name in opts:
                    payload[name] = opts[name]
            config: dict[str, Any] = {}
            if settings.effort:
                config["effort"] = settings.effort
                payload["thinking"] = {"type": "adaptive"}
            if schema is not None:
                config["format"] = {"type": "json_schema", "schema": schema}
            if config:
                payload["output_config"] = config
            if tools:
                payload["tools"] = [
                    {"name": t.name, "description": t.description, "input_schema": t.parameters}
                    for t in tools.schemas()
                ]
        else:
            payload.update(
                input=history,
                store=False,
                include=["reasoning.encrypted_content"],
                max_output_tokens=cap,
                prompt_cache_key=prompt.prefix_digest,
            )
            if settings.cache_ttl is not None:
                payload["prompt_cache_options"] = {"ttl": settings.cache_ttl.value}
            if prompt.system:
                payload["instructions"] = prompt.system
            for name in ("temperature", "top_p"):
                if name in opts:
                    payload[name] = opts[name]
            reasoning = {}
            if settings.effort:
                reasoning["effort"] = settings.effort
            if settings.reasoning_summary:
                reasoning["summary"] = settings.reasoning_summary
            if reasoning:
                payload[EventKind.REASONING] = reasoning
            text = {}
            if schema is not None:
                text["format"] = {"type": "json_schema", "name": "answer", "schema": schema, "strict": True}
            if settings.verbosity:
                text["verbosity"] = settings.verbosity
            if text:
                payload["text"] = text
            if settings.tier:
                payload["service_tier"] = settings.tier
            if tools:
                payload["tools"] = [
                    {
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "strict": False,
                    }
                    for t in tools.schemas()
                ]
        try:
            if anthropic:
                result = await self._anthropic(transport, payload, emit)
            else:
                result = await self._openai(transport, payload, emit)
        except Exception as exc:
            result = Turn(failure=classify(exc))
        finally:
            if self.client is None:
                await transport.aclose()
        result.history = tuple(history) + result.history
        return result

    async def _openai(self, transport: Any, payload: Any, emit: Any) -> Turn:
        final: dict[str, Any] | None = None
        try:
            async for event in transport._stream_events(ApiPath.RESPONSES, payload):
                kind = event.get("type", FailureClass.UNKNOWN)
                await _emit(emit, EventKind.TRANSPORT, mode_event=kind)
                if kind == "response.output_text.delta":
                    await _emit(emit, EventKind.MESSAGE_DELTA, text=event.get("delta", ""))
                elif kind == "response.reasoning_summary_text.delta":
                    await _emit(emit, EventKind.REASONING, summary=event.get("delta", ""))
                elif kind in ("response.completed", "response.incomplete", "response.failed"):
                    final = event.get("response") or {}
                    if kind == "response.failed":
                        return Turn(raw_usage=final.get(EventKind.USAGE) or {}, failure=classify(final))
                elif kind == "error":
                    # `classify` returns None for a frame whose `error` is not a
                    # mapping, and `failure=None` is a SUCCESSFUL turn. An error
                    # frame must never read as success, so an unclassifiable one
                    # degrades to UNKNOWN rather than vanishing.
                    return Turn(failure=classify(event) or Failure(FailureClass.UNKNOWN))
        except asyncio.CancelledError:
            return Turn(
                raw_usage=(final or {}).get("usage") or {},
                usage_complete=False,
                failure=Failure(FailureClass.CANCELLED),
            )
        except Exception as exc:
            return Turn(
                raw_usage=(final or {}).get("usage") or {}, usage_complete=final is not None, failure=classify(exc)
            )
        if final is None:
            # A truncated response body is the textbook retryable transport fault, and
        # `retryable` is NOT derived from the class — it defaults to False, so
        # naming the class alone produced a TRANSIENT_NETWORK that the runtime
        # refused to retry (`retry_allowed = failure.retryable and ...`). Measured,
        # before: a stream ending with no terminal frame made 1 request, while the
        # SAME physical event one layer earlier — a socket error — made 3. It was
        # the only transport failure on this path that was never retried.
            return Turn(failure=Failure(FailureClass.TRANSIENT_NETWORK, True))
        messages = []
        calls = []
        refused = False
        items = final.get("output") or []
        for item in items:
            if item.get("type") == "message":
                blocks = item.get("content") or []
                refused |= any(b.get("type") == "refusal" for b in blocks)
                messages.append("".join(b.get("text", "") for b in blocks if b.get("type") == "output_text"))
            elif item.get("type") == "function_call":
                calls.append(
                    {
                        "id": item["call_id"],
                        "name": item["name"],
                        "arguments": json.loads(item.get("arguments") or "{}"),
                    }
                )
        return Turn(
            tuple(messages),
            raw_usage=final.get(EventKind.USAGE) or {},
            refused=refused,
            truncated=final.get("status") == "incomplete",
            tool_calls=tuple(calls),
            history=tuple(items),
        )

    async def _anthropic(self, transport: Any, payload: Any, emit: Any) -> Turn:
        blocks: dict[int, dict[str, Any]] = {}
        args: dict[int, str] = {}
        usage: dict[str, Any] = {}
        stop = None
        terminal = False
        try:
            async for event in transport._stream_events(ApiPath.ANTHROPIC_MESSAGES, payload):
                kind = event.get("type", FailureClass.UNKNOWN)
                await _emit(emit, EventKind.TRANSPORT, mode_event=kind)
                if kind == "error":
                    # See the sibling parser: an unclassifiable error frame
                    # degrades to UNKNOWN instead of reading as success.
                    return Turn(raw_usage=usage, failure=classify(event) or Failure(FailureClass.UNKNOWN))
                if kind == "message_start":
                    usage.update((event.get("message") or {}).get(EventKind.USAGE) or {})
                elif kind == "content_block_start":
                    index = event["index"]
                    blocks[index] = dict(event["content_block"])
                elif kind == "content_block_delta":
                    index, delta = event["index"], event["delta"]
                    block = blocks[index]
                    if delta["type"] == "text_delta":
                        block["text"] = block.get("text", "") + delta["text"]
                        await _emit(emit, EventKind.MESSAGE_DELTA, text=delta["text"])
                    elif delta["type"] == "input_json_delta":
                        args[index] = args.get(index, "") + delta["partial_json"]
                    elif delta["type"] == "thinking_delta":
                        block["thinking"] = block.get("thinking", "") + delta["thinking"]
                    elif delta["type"] == "signature_delta":
                        block["signature"] = block.get("signature", "") + delta["signature"]
                elif kind == "content_block_stop":
                    index = event["index"]
                    if index in args:
                        blocks[index]["input"] = json.loads(args[index] or "{}")
                elif kind == EventKind.MESSAGE_DELTA:
                    usage.update(event.get(EventKind.USAGE) or {})
                    stop = (event.get("delta") or {}).get("stop_reason")
                elif kind == "message_stop":
                    terminal = True
        except asyncio.CancelledError:
            return Turn(raw_usage=usage, usage_complete=False, failure=Failure(FailureClass.CANCELLED))
        except Exception as exc:
            return Turn(raw_usage=usage, usage_complete=terminal, failure=classify(exc))
        if not terminal:
            # A truncated response body is the textbook retryable transport fault, and
        # `retryable` is NOT derived from the class — it defaults to False, so
        # naming the class alone produced a TRANSIENT_NETWORK that the runtime
        # refused to retry (`retry_allowed = failure.retryable and ...`). Measured,
        # before: a stream ending with no terminal frame made 1 request, while the
        # SAME physical event one layer earlier — a socket error — made 3. It was
        # the only transport failure on this path that was never retried.
            return Turn(
                raw_usage=usage, usage_complete=False, failure=Failure(FailureClass.TRANSIENT_NETWORK, True)
            )
        content = [blocks[i] for i in sorted(blocks)]
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = tuple(
            {"id": b["id"], "name": b["name"], "arguments": b.get("input") or {}}
            for b in content
            if b.get("type") == "tool_use"
        )
        # pause_turn resumes with the untouched assistant content, even without
        # a tool call initiated by this client.
        if stop == "pause_turn":
            calls = ({"id": "", "name": "__pause_turn__", "arguments": {}},)
        return Turn(
            (text,),
            raw_usage=usage,
            refused=stop == "refusal",
            truncated=stop == "max_tokens",
            tool_calls=calls,
            history=({"role": "assistant", "content": content},),
        )
