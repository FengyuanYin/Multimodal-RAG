from __future__ import annotations

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.cloud.cohere_compatible import CohereRerankClient, resolve_rerank_url
from agentic_rag.cli.models import RetrievalHit, ServiceProfile


class Transport:
    def __init__(self):
        self.calls = []

    def request_json(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return {"results": [{"index": 0, "relevance_score": 0.95}]}

    def close(self):
        pass


def test_rerank_endpoint_resolution_is_provider_compatible() -> None:
    assert resolve_rerank_url("https://api.siliconflow.cn/v1") == "https://api.siliconflow.cn/v1/rerank"
    assert resolve_rerank_url("https://api.cohere.com") == "https://api.cohere.com/v2/rerank"
    assert resolve_rerank_url("https://gateway.example/v2/") == "https://gateway.example/v2/rerank"


def test_siliconflow_rerank_uses_official_payload_and_response_shape() -> None:
    transport = Transport()
    profile = ServiceProfile("https://api.siliconflow.cn/v1", "BAAI/bge-reranker-v2-m3", "reranker_api_key")
    client = CohereRerankClient(profile, "secret", transport=transport)
    candidate = RetrievalHit("chunk", "doc", "Document", "candidate text", 1, "text", 0.1)

    result = client.rerank("query", [candidate], 1, CancellationToken())

    method, url, kwargs = transport.calls[0]
    assert (method, url) == ("POST", "https://api.siliconflow.cn/v1/rerank")
    assert kwargs["json_body"] == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "query",
        "documents": ["candidate text"],
        "top_n": 1,
    }
    assert result[0].score == 0.95
