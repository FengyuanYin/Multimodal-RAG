import socket
import json

import pytest
from starlette.requests import Request

from web import proxy


def test_url_syntax_rejects_credentials_and_non_http():
    assert proxy._is_safe_url("https://example.com/page")
    assert not proxy._is_safe_url("file:///etc/passwd")
    assert not proxy._is_safe_url("https://user:pass@example.com")


@pytest.mark.asyncio
async def test_target_validation_blocks_private_dns(monkeypatch):
    async def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(proxy.asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="本地、内网或保留地址"):
        await proxy._validate_target("http://attacker.example/resource")


@pytest.mark.asyncio
async def test_target_validation_accepts_global_dns(monkeypatch):
    async def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(proxy.asyncio.get_running_loop(), "getaddrinfo", fake_getaddrinfo)
    assert await proxy._validate_target("https://example.com") == "https://example.com"


@pytest.mark.asyncio
async def test_forward_relays_stream_without_buffering(monkeypatch):
    closed = {"client": False, "upstream": False}

    async def allow_target(url):
        return url

    class FakeUpstream:
        status_code = 200

        async def aiter_bytes(self):
            yield b"data: one\n\n"
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            closed["upstream"] = True

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, request, stream=False):
            assert stream is True
            return FakeUpstream()

        async def aclose(self):
            closed["client"] = True

    monkeypatch.setattr(proxy, "_validate_target", allow_target)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    body = json.dumps({"base_url": "https://example.com/v1", "stream": True}).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/proxy/chat/completions", "headers": []}, receive)
    response = await proxy._forward(request, "chat/completions")
    chunks = [chunk async for chunk in response.body_iterator]

    assert b"".join(chunks) == b"data: one\n\ndata: [DONE]\n\n"
    assert response.media_type == "text/event-stream"
    assert closed == {"client": True, "upstream": True}
