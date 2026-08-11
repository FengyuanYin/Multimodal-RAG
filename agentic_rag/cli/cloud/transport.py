"""Bounded sync HTTP transport with error classification and retries."""

from __future__ import annotations

from contextlib import contextmanager
import json
import time
from typing import Any, Iterator

import httpx

from ..cancellation import CancellationToken
from ..errors import CancelledError, UpstreamError, classify_http_error
from ..security import redact


class HttpTransport:
    def __init__(self, service: str, timeout: float = 60.0, retries: int = 2, *, client: httpx.Client | None = None, secrets: tuple[str, ...] = ()) -> None:
        self.service, self.retries, self.secrets = service, max(0, min(retries, 5)), secrets
        self._owned = client is None
        self.client = client or httpx.Client(timeout=httpx.Timeout(connect=min(timeout, 20.0), read=timeout, write=timeout, pool=20.0), follow_redirects=False)

    def request_json(self, method: str, url: str, *, headers: dict[str, str] | None = None, json_body: Any = None, content: bytes | None = None, cancel: CancellationToken | None = None, idempotent: bool = False, max_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
        attempts = self.retries + 1 if idempotent else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            if cancel:
                cancel.checkpoint()
            try:
                response = self.client.request(method, url, headers=headers, json=json_body, content=content)
                if response.status_code >= 400:
                    if idempotent and response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                        self._backoff(response, attempt, cancel)
                        continue
                    raise classify_http_error(response.status_code, self.service)
                if len(response.content) > max_bytes:
                    raise UpstreamError(f"{self.service} response exceeded the allowed size")
                try:
                    value = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise UpstreamError(f"{self.service} returned invalid JSON") from exc
                if not isinstance(value, dict):
                    raise UpstreamError(f"{self.service} returned an unsupported response shape")
                return value
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    self._sleep(min(0.5 * (2 ** attempt), 4.0), cancel)
                    continue
                error = UpstreamError(f"{self.service} network request failed ({type(exc).__name__})")
                error.code = "UPSTREAM_NETWORK"
                raise error from exc
        raise UpstreamError(f"{self.service} request failed ({type(last_error).__name__})")

    @contextmanager
    def stream(self, method: str, url: str, *, headers: dict[str, str] | None = None, json_body: Any = None, content: bytes | None = None, cancel: CancellationToken | None = None) -> Iterator[httpx.Response]:
        if cancel:
            cancel.checkpoint()
        try:
            with self.client.stream(method, url, headers=headers, json=json_body, content=content) as response:
                if response.status_code >= 400:
                    response.read()
                    raise classify_http_error(response.status_code, self.service)
                yield response
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            error = UpstreamError(f"{self.service} stream failed ({type(exc).__name__})")
            error.code = "UPSTREAM_NETWORK"
            raise error from exc

    def probe_status(self, method: str, url: str, *, headers: dict[str, str] | None = None, cancel: CancellationToken | None = None) -> int:
        """Return an HTTP status without downloading or parsing the response body."""
        if cancel:
            cancel.checkpoint()
        try:
            with self.client.stream(method, url, headers=headers) as response:
                status = response.status_code
                if status in {401, 402, 403, 429} or status >= 500:
                    raise classify_http_error(status, self.service)
                return status
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            error = UpstreamError(f"{self.service} connectivity probe failed ({type(exc).__name__})")
            error.code = "UPSTREAM_NETWORK"
            raise error from exc

    @staticmethod
    def _sleep(seconds: float, cancel: CancellationToken | None) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if cancel:
                cancel.checkpoint()
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _backoff(self, response: httpx.Response, attempt: int, cancel: CancellationToken | None) -> None:
        try:
            seconds = min(float(response.headers.get("retry-after", "0") or 0), 10.0)
        except ValueError:
            seconds = 0.0
        self._sleep(seconds or min(0.5 * (2 ** attempt), 4.0), cancel)

    def close(self) -> None:
        if self._owned:
            self.client.close()
