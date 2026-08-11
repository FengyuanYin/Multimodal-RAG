"""Real, bounded cloud connectivity probes with stable result classes."""

from __future__ import annotations

import time

from ..errors import AutoMemoryError, CancelledError, ConfigurationError
from ..models import EventKind, OutputEvent, ProbeResult
from ..security import redact


SERVICES = ("llm", "embedding", "vlm", "reranker", "mineru", "web")


class ConnectionTester:
    def __init__(self, context) -> None:
        self.context = context

    def test_services(self, services: set[str], output, cancel) -> list[ProbeResult]:
        unknown = services - set(SERVICES)
        if unknown:
            raise ConfigurationError(f"Unknown service for connectivity test: {sorted(unknown)[0]}")
        results = []
        for service in SERVICES:
            if service not in services:
                continue
            cancel.checkpoint()
            started = time.perf_counter()
            try:
                verified = self._probe(service, cancel)
                status = "success" if verified else "reachable_unverified"
                code = "PROBE_OK" if verified else "PROBE_REACHABLE_UNVERIFIED"
                message = "connection verified" if verified else "endpoint reachable; capability not fully verified"
            except CancelledError:
                raise
            except AutoMemoryError as exc:
                status, code = self._classify(exc)
                message = redact(exc.message, self.context.credentials.redaction_values())
            except Exception as exc:
                status, code = "response_error", "PROBE_INTERNAL"
                message = f"unexpected probe failure ({type(exc).__name__})"
            latency = round((time.perf_counter() - started) * 1000, 2)
            result = ProbeResult(service, status, code, message, latency)
            results.append(result)
            marker = "OK" if status == "success" else "INFO" if status in {"reachable_unverified", "skipped"} else "ERROR"
            output.emit(OutputEvent(EventKind.RESULT, text=f"[{marker}] {service}: {message} ({code}, {latency:.0f} ms)"))
        return results

    def _probe(self, service: str, cancel) -> bool:
        ctx = self.context
        if service == "llm":
            if not ctx.llm_client:
                raise ConfigurationError("LLM credential is not configured")
            ctx.llm_client.probe_chat(cancel)
            return True
        if service == "embedding":
            if not ctx.embedding_client:
                raise ConfigurationError("Embedding credential is not configured")
            ctx.embedding_client.probe_embedding(cancel)
            return True
        if service == "vlm":
            if not ctx.vlm_client:
                raise ConfigurationError("VLM credential is not configured")
            ctx.vlm_client.probe_chat(cancel)
            return True
        if service == "reranker":
            if not ctx.reranker_client:
                raise ConfigurationError("Reranker credential is not configured")
            ctx.reranker_client.probe(cancel)
            return True
        if service == "mineru":
            client = ctx.mineru_client()
            try:
                return client.probe(cancel, official=ctx.config.mineru_mode == "official")
            finally:
                client.close()
        if service == "web":
            key = ctx.credentials.get("tavily_api_key")
            ctx.web_client.search("AutoMemory connectivity", ctx.config.web_provider, key, cancel, limit=1)
            return True
        raise ConfigurationError(f"Unknown service: {service}")

    @staticmethod
    def _classify(error: AutoMemoryError) -> tuple[str, str]:
        code = getattr(error, "code", "PROBE_ERROR")
        if code == "UPSTREAM_AUTH":
            return "auth_error", "PROBE_AUTH"
        if code in {"UPSTREAM_RATE_LIMIT", "UPSTREAM_QUOTA"}:
            return "rate_limited", "PROBE_RATE_LIMIT"
        if code == "UPSTREAM_NETWORK":
            return "network_error", "PROBE_NETWORK"
        if code == "UPSTREAM_MODEL":
            return "model_error", "PROBE_MODEL"
        if isinstance(error, ConfigurationError) and any(marker in error.message.lower() for marker in ("credential", "api key", "not configured")):
            return "skipped", "PROBE_NOT_CONFIGURED"
        return "response_error", "PROBE_RESPONSE"
