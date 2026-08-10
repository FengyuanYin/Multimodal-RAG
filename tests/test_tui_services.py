import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from agentic_rag.tui.events import CancelToken, JobCancelled
from agentic_rag.tui.models import AutoMemoryConfig, ChatRequest
from agentic_rag.tui.services.chat import ChatService
from agentic_rag.tui.services.evaluation import evaluate_ranking
from agentic_rag.tui.services.mineru import MinerUService
from agentic_rag.tui.storage import StateRepository


def test_cancel_token_stops_work():
    token = CancelToken()
    token.cancel()
    with pytest.raises(JobCancelled):
        token.checkpoint()


def test_evaluation_metrics_are_deterministic():
    results = [
        SimpleNamespace(doc_id="chunk-a", metadata={"document_id": "doc-a", "media_refs": [{"media_id": "fig-1"}]}),
        SimpleNamespace(doc_id="chunk-b", metadata={"document_id": "doc-b"}),
    ]
    metrics = evaluate_ranking(results, ["doc-a"], ["fig-1"], 2)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_k"] == 1.0
    assert metrics["media_recall_at_k"] == 1.0


def test_mineru_archive_normalization_and_zip_slip_guard():
    content = [{"type": "text", "page_idx": 0, "text": "Grounded text"}]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("result_content_list.json", json.dumps(content))
    parsed = MinerUService().normalize_archive(buffer.getvalue(), "paper.pdf", "test")
    assert parsed.pages == [{"page": 1, "text": "Grounded text"}]
    unsafe = io.BytesIO()
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.md", "bad")
    with pytest.raises(ValueError, match="unsafe path"):
        MinerUService().normalize_archive(unsafe.getvalue(), "paper.pdf", "test")


class _Completions:
    def create(self, **kwargs):
        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]) for text in ("hello", " world")]


class _Retriever:
    last_trace = None

    def __init__(self):
        self.calls = 0

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        return [SimpleNamespace(doc_id="doc-a_chunk_0000", content="Evidence", score=0.9, metadata={"document_id": "doc-a", "page": 2}, modality="text", media_refs=[])]


def _chat_runtime():
    retriever = _Retriever()
    repository = SimpleNamespace(list_documents=lambda include_unsearchable=True: [{"id": "doc-a", "name": "Paper"}])
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    runtime = SimpleNamespace(
        config=AutoMemoryConfig(memory_enabled=False),
        orchestrator=SimpleNamespace(llm_client=client),
        retriever=retriever,
        repository=repository,
    )
    return runtime, retriever


def test_chat_direct_does_not_retrieve_and_rag_does(tmp_path):
    state = StateRepository(tmp_path / "state.db")
    runtime, retriever = _chat_runtime()
    service = ChatService(runtime, state)
    direct = state.create_conversation("Direct")
    assert service.stream(ChatRequest(direct.id, "Hi", "direct")).answer == "hello world"
    assert retriever.calls == 0
    rag = state.create_conversation("RAG")
    result = service.stream(ChatRequest(rag.id, "What?", "rag"))
    assert retriever.calls == 1
    assert result.sources[0]["document"] == "Paper"
    state.close()


def test_chat_cancellation_persists_interrupted_status(tmp_path):
    state = StateRepository(tmp_path / "state.db")
    runtime, _ = _chat_runtime()
    service = ChatService(runtime, state)
    conversation = state.create_conversation("Cancel")
    token = CancelToken()

    def cancel_after_first(event):
        token.cancel()

    with pytest.raises(JobCancelled):
        service.stream(ChatRequest(conversation.id, "Hi", "direct"), emit=cancel_after_first, cancel=token)
    assert state.list_messages(conversation.id)[-1].status == "interrupted"
    state.close()
