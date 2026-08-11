"""Cohere-compatible cloud reranking."""

from __future__ import annotations

from ..cancellation import CancellationToken
from ..errors import ConfigurationError, UpstreamError
from ..models import RetrievalHit, ServiceProfile
from .transport import HttpTransport


class CohereRerankClient:
    def __init__(self, profile: ServiceProfile, api_key: str, *, transport: HttpTransport | None = None) -> None:
        self.profile, self.api_key = profile, api_key
        self.transport = transport or HttpTransport("Reranker", profile.timeout_seconds, profile.retries, secrets=(api_key,))

    def rerank(self, query: str, candidates: list[RetrievalHit], top_k: int, cancel: CancellationToken) -> list[RetrievalHit]:
        if not candidates:
            return []
        if not self.api_key:
            raise ConfigurationError("Reranker API key is not configured")
        payload = self.transport.request_json(
            "POST", f"{self.profile.base_url.rstrip('/')}/v2/rerank",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json_body={"model": self.profile.model, "query": query, "documents": [item.text[:4000] for item in candidates], "top_n": min(top_k, len(candidates))},
            cancel=cancel,
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise UpstreamError("Reranker returned an unsupported response")
        seen, output = set(), []
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or index < 0 or index >= len(candidates) or index in seen or not isinstance(score, (int, float)):
                raise UpstreamError("Reranker returned invalid candidate indexes or scores")
            seen.add(index)
            candidate = candidates[index]
            candidate.channel_scores["reranker"] = float(score)
            candidate.score = float(score)
            output.append(candidate)
        return output

    def close(self) -> None:
        self.transport.close()

    def probe(self, cancel: CancellationToken) -> None:
        candidate = RetrievalHit("probe", "probe", "probe", "AutoMemory connectivity probe", 1, "text", 0.0)
        result = self.rerank("AutoMemory", [candidate], 1, cancel)
        if not result:
            raise UpstreamError("Reranker connectivity response contained no result")
