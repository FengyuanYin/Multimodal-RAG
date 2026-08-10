from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.cli.errors import UsageError
from agentic_rag.cli.models import ChunkRecord, DocumentRecord
from agentic_rag.cli.storage import KnowledgeRepository, StateRepository


def test_state_survives_restart_and_recovers_streams(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    backups = tmp_path / "backups"
    state = StateRepository(database, backups)
    conversation = state.create_conversation("Persistent")
    state.append_message(conversation["id"], "assistant", "partial", "direct", "streaming")
    state.add_memory("Prefer concise answers")
    state.create_evaluation("eval_1", "dataset.json", {"mode": "keyword"})
    state.update_evaluation("eval_1", "success", {"mrr": 1.0}, "result.json")
    state.close()

    reopened = StateRepository(database, backups)
    assert reopened.ensure_active_conversation() == conversation["id"]
    assert reopened.list_messages(conversation["id"])[0]["status"] == "interrupted"
    assert reopened.list_memories()[0]["content"] == "Prefer concise answers"
    assert reopened.list_evaluations()[0]["summary"]["mrr"] == 1.0
    assert reopened.integrity_check() == "ok"
    reopened.close()


def test_memory_rejects_credentials(tmp_path: Path) -> None:
    state = StateRepository(tmp_path / "state.db", tmp_path / "backups")
    with pytest.raises(UsageError):
        state.add_memory("api_key=sk-abcdefghijklmnopqrstuvwxyz")
    state.close()


def test_knowledge_document_and_vector_roundtrip(tmp_path: Path) -> None:
    knowledge = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    document = DocumentRecord("doc_1", "fingerprint", "Paper", "paper.txt", "text", "default", "text", 1, "ready")
    chunk = ChunkRecord("chunk_1", "doc_1", 1, 0, "industrial retrieval evidence")
    knowledge.commit_document(document, [chunk], [], [("chunk_1", "chunk", "profile", [3.0, 4.0])])
    item = knowledge.get_document("doc_1")
    assert item and item["chunks"][0]["text"] == chunk.text
    vector = knowledge.list_embeddings("profile")[0]["vector"]
    assert vector == pytest.approx([0.6, 0.8])
    knowledge.close()
