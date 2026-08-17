from __future__ import annotations

import base64
from io import StringIO
from types import SimpleNamespace

import httpx
import pytest

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.cloud.openai_compatible import OpenAICompatibleClient
from agentic_rag.cli.cloud.transport import HttpTransport
from agentic_rag.cli.config import AutoMemoryConfig
from agentic_rag.cli.errors import CancelledError, UpstreamError
from agentic_rag.cli.models import ParsedDocument, ServiceProfile
from agentic_rag.cli.paths import AutoMemoryPaths
from agentic_rag.cli.services.ingestion import IngestionService
from agentic_rag.cli.storage import KnowledgeRepository


class BatchTransport:
    def __init__(self) -> None:
        self.calls = []
        self.waits = []

    def request_json(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        texts = kwargs["json_body"]["input"]
        return {"data": [{"index": index, "embedding": [1.0, float(index + 1)]} for index, _ in enumerate(texts)]}

    def wait(self, seconds, cancel):
        cancel.checkpoint()
        self.waits.append(seconds)

    def close(self):
        pass


def test_embeddings_are_throttled_retryable_and_report_progress() -> None:
    transport = BatchTransport()
    profile = ServiceProfile("https://example.test/v1", "embed", "embedding_api_key", batch_size=32)
    client = OpenAICompatibleClient(profile, "secret", service="Embedding", transport=transport)
    progress = []

    vectors = client.embeddings(
        [f"chunk-{index}" for index in range(141)],
        CancellationToken(),
        on_progress=lambda completed, total: progress.append((completed, total)),
        batch_delay_seconds=1.0,
    )

    assert len(vectors) == 141
    assert progress == [(32, 141), (64, 141), (96, 141), (128, 141), (141, 141)]
    assert transport.waits == [1.0, 1.0, 1.0, 1.0]
    assert all(call[2]["idempotent"] is True for call in transport.calls)


class FlakyHttpClient:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def request(self, method, url, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        pass


def test_idempotent_request_retries_connect_error(monkeypatch) -> None:
    request = httpx.Request("POST", "https://example.test/embeddings")
    client = FlakyHttpClient([
        httpx.ConnectError("temporary", request=request),
        httpx.Response(200, request=request, json={"data": []}),
    ])
    transport = HttpTransport("Embedding", retries=2, client=client)
    monkeypatch.setattr(transport, "_sleep", lambda seconds, cancel: None)

    result = transport.request_json("POST", str(request.url), idempotent=True)

    assert result == {"data": []}
    assert client.calls == 2


def test_non_retryable_auth_error_is_not_replayed() -> None:
    request = httpx.Request("POST", "https://example.test/embeddings")
    client = FlakyHttpClient([httpx.Response(401, request=request, json={"error": "no"})])
    transport = HttpTransport("Embedding", retries=2, client=client)

    with pytest.raises(UpstreamError) as captured:
        transport.request_json("POST", str(request.url), idempotent=True)

    assert captured.value.code == "UPSTREAM_AUTH"
    assert client.calls == 1


def test_rate_limit_honors_retry_after(monkeypatch) -> None:
    request = httpx.Request("POST", "https://example.test/embeddings")
    client = FlakyHttpClient([
        httpx.Response(429, request=request, headers={"Retry-After": "2"}, json={"error": "slow down"}),
        httpx.Response(200, request=request, json={"data": []}),
    ])
    transport = HttpTransport("Embedding", retries=2, client=client)
    waits = []
    monkeypatch.setattr(transport, "_sleep", lambda seconds, cancel: waits.append(seconds))

    result = transport.request_json("POST", str(request.url), idempotent=True)

    assert result == {"data": []}
    assert waits == [2.0]
    assert client.calls == 2


def test_throttle_wait_is_cancellable() -> None:
    transport = HttpTransport("Embedding", client=FlakyHttpClient([]))
    cancel = CancellationToken()
    cancel.cancel()

    with pytest.raises(CancelledError):
        transport.wait(1.0, cancel)


class RecordingOutput:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class IngestionEmbeddingClient:
    profile_fingerprint = "profile"

    def embeddings(self, texts, cancel, *, on_progress, batch_delay_seconds):
        assert batch_delay_seconds == 1.0
        on_progress(len(texts), len(texts))
        return [[1.0, 0.0] for _ in texts]


class RecordingVectorStore:
    def __init__(self):
        self.records = []

    def existing_ids(self, record_ids, scope):
        return set()

    def add(self, records):
        self.records.extend(records)
        return len(records)


def test_ingestion_connects_embedding_progress_to_output(tmp_path) -> None:
    paths = AutoMemoryPaths.resolve(tmp_path)
    knowledge = KnowledgeRepository(paths.knowledge_db, paths.backups_dir)
    config = AutoMemoryConfig(retrieval_mode="hybrid", chunk_size=64, chunk_overlap=8)
    output = RecordingOutput()
    vectors = RecordingVectorStore()
    service = IngestionService(knowledge, paths, config, vector_store=vectors, embedding_client=IngestionEmbeddingClient())

    result = service.ingest_parsed(
        ParsedDocument("demo", [{"page": 1, "text": "evidence " * 20}], [], "test"),
        "demo.pdf",
        "default",
        output,
        CancellationToken(),
    )

    progress = [event for event in output.events if event.phase == "embedding"]
    assert progress[0].completed == 0
    assert progress[-1].completed == progress[-1].total
    assert knowledge.get_document(result["document_id"])["status"] == "ready"
    assert vectors.records
    knowledge.close()


class FailingEmbeddingClient:
    profile_fingerprint = "profile"

    def embeddings(self, texts, cancel, *, on_progress, batch_delay_seconds):
        on_progress(1, len(texts))
        raise UpstreamError("temporary failure")


def test_failed_embedding_preserves_parsed_knowledge_for_retry(tmp_path) -> None:
    paths = AutoMemoryPaths.resolve(tmp_path)
    knowledge = KnowledgeRepository(paths.knowledge_db, paths.backups_dir)
    config = AutoMemoryConfig(retrieval_mode="hybrid", chunk_size=64, chunk_overlap=8)
    service = IngestionService(knowledge, paths, config, vector_store=RecordingVectorStore(), embedding_client=FailingEmbeddingClient())
    media = [{
        "id": "figure1", "page": 1, "type": "image", "label": "figure1", "caption": "",
        "data": base64.b64encode(b"image-bytes").decode("ascii"), "mime_type": "image/png", "quality": "exact",
    }]

    result = service.ingest_parsed(
        ParsedDocument("demo", [{"page": 1, "text": "evidence " * 20}], media, "test"),
        "demo.pdf", "default", RecordingOutput(), CancellationToken(),
    )

    assert knowledge.get_document(result["document_id"])["status"] == "ready"
    assert result["indexes"]["degraded"][0]["index"] == "embedding"
    assert list(paths.media_dir.rglob("*.*"))
    knowledge.close()
