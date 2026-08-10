"""Cloud-only OpenAI-compatible LLM, embedding, and VLM client."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from typing import Any, Iterator

from ..cancellation import CancellationToken
from ..errors import ConfigurationError, UpstreamError
from ..models import ServiceProfile
from .transport import HttpTransport


class OpenAICompatibleClient:
    def __init__(self, profile: ServiceProfile, api_key: str, *, service: str = "OpenAI-compatible", transport: HttpTransport | None = None) -> None:
        self.profile, self.api_key, self.service = profile, api_key, service
        self.transport = transport or HttpTransport(service, profile.timeout_seconds, profile.retries, secrets=(api_key,))

    @property
    def profile_fingerprint(self) -> str:
        value = f"{self.profile.base_url.rstrip('/')}\0{self.profile.model}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ConfigurationError(f"{self.service} API key is not configured")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def stream_chat(self, messages: list[dict[str, Any]], cancel: CancellationToken, *, temperature: float = 0.2) -> Iterator[str]:
        body = {"model": self.profile.model, "messages": messages, "temperature": temperature, "stream": True}
        url = f"{self.profile.base_url.rstrip('/')}/chat/completions"
        with self.transport.stream("POST", url, headers=self._headers(), json_body=body, cancel=cancel) as response:
            for line in response.iter_lines():
                cancel.checkpoint()
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    choices = payload.get("choices") or []
                    delta = (choices[0].get("delta") or {}).get("content") if choices else None
                except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                    raise UpstreamError(f"{self.service} returned a malformed stream event") from exc
                if isinstance(delta, str) and delta:
                    yield delta

    def embeddings(self, texts: list[str], cancel: CancellationToken) -> list[list[float]]:
        if not texts:
            return []
        output: list[list[float]] = []
        batch_size = max(1, min(self.profile.batch_size, 128))
        for start in range(0, len(texts), batch_size):
            cancel.checkpoint()
            batch = texts[start:start + batch_size]
            payload = self.transport.request_json(
                "POST", f"{self.profile.base_url.rstrip('/')}/embeddings", headers=self._headers(),
                json_body={"model": self.profile.model, "input": batch}, cancel=cancel,
            )
            data = payload.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise UpstreamError(f"{self.service} embedding response count does not match input")
            ordered = sorted(data, key=lambda item: int(item.get("index", -1)))
            expected_dimensions = None
            for index, item in enumerate(ordered):
                if int(item.get("index", -1)) != index:
                    raise UpstreamError(f"{self.service} embedding indexes are invalid")
                vector = item.get("embedding")
                if not isinstance(vector, list) or not vector or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector):
                    raise UpstreamError(f"{self.service} returned an invalid embedding vector")
                expected_dimensions = expected_dimensions or len(vector)
                if len(vector) != expected_dimensions:
                    raise UpstreamError(f"{self.service} embedding dimensions changed within one response")
                output.append([float(value) for value in vector])
        return output

    def describe_image(self, content: bytes, mime_type: str, prompt: str, cancel: CancellationToken) -> str:
        if len(content) > 10 * 1024 * 1024:
            raise ConfigurationError("Image exceeds the 10MB VLM request limit")
        data_url = f"data:{mime_type};base64,{base64.b64encode(content).decode('ascii')}"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}]
        return "".join(self.stream_chat(messages, cancel, temperature=0.0)).strip()

    def close(self) -> None:
        self.transport.close()
