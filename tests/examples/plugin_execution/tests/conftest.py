"""Provider SSE scripts only. No tool dispatch or application model loop."""

import json

import httpx
import pytest


@pytest.fixture
def scripted_client():
    clients = []

    def build(mode, *, record_id="record-1", tool_name="lookup_record", status="ready"):
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            answer = json.dumps({"record_id": record_id, "status": status})
            arguments = json.dumps({"record_id": record_id})
            first = len(requests) == 1
            if mode == "anthropic_api":
                block = (
                    {"type": "tool_use", "id": "provider-call-1", "name": tool_name, "input": {}}
                    if first
                    else {"type": "text", "text": ""}
                )
                delta = (
                    {"type": "input_json_delta", "partial_json": arguments}
                    if first
                    else {"type": "text_delta", "text": answer}
                )
                events = [
                    {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
                    {"type": "content_block_start", "index": 0, "content_block": block},
                    {"type": "content_block_delta", "index": 0, "delta": delta},
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use" if first else "end_turn"},
                        "usage": {"output_tokens": 1},
                    },
                    {"type": "message_stop"},
                ]
            else:
                output = (
                    [{"type": "function_call", "call_id": "provider-call-1", "name": tool_name, "arguments": arguments}]
                    if first
                    else [
                        {"type": "message", "content": [{"type": "output_text", "text": answer}]},
                    ]
                )
                events = [
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "output": output,
                            "usage": {"input_tokens": 2, "output_tokens": 1},
                        },
                    }
                ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
        clients.append(client)
        return client, requests

    yield build
    # MockTransport owns no sockets; each async test closes its client.
    assert all(client.is_closed for client in clients)
