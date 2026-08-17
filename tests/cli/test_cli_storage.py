from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.cli.errors import UsageError
from agentic_rag.cli.models import ChunkRecord, DocumentRecord, MediaRecord
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


def test_knowledge_document_roundtrip_does_not_store_vectors(tmp_path: Path) -> None:
    knowledge = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    document = DocumentRecord("doc_1", "fingerprint", "Paper", "paper.txt", "text", "default", "text", 1, "ready")
    chunk = ChunkRecord("chunk_1", "doc_1", 1, 0, "industrial retrieval evidence")
    knowledge.commit_document(document, [chunk], [])
    item = knowledge.get_document("doc_1")
    assert item and item["chunks"][0]["text"] == chunk.text
    tables = {row[0] for row in knowledge._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "embeddings" not in tables
    knowledge.close()


def test_media_vlm_analysis_cache_is_scoped_and_cascades(tmp_path: Path) -> None:
    knowledge = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    document = DocumentRecord("doc_vlm", "fp_vlm", "Visual", "visual.pdf", "pdf", "default", "text", 1, "ready")
    chunk = ChunkRecord("chunk_vlm", "doc_vlm", 1, 0, "see figure")
    media = MediaRecord("image_vlm", "doc_vlm", 1, "image", "Figure 1", checksum="checksum-a", storage_path="image.png")
    knowledge.commit_document(document, [chunk], [media])
    analysis = {"media_id": "image_vlm", "image_type": "other", "content": "visible facts"}

    knowledge.upsert_media_vlm_analysis("image_vlm", "checksum-a", "profile-a", "prompt-v1", analysis)

    assert knowledge.get_media_vlm_analysis("image_vlm", "checksum-a", "profile-a", "prompt-v1") == analysis
    assert knowledge.get_media_vlm_analysis("image_vlm", "checksum-b", "profile-a", "prompt-v1") is None
    assert knowledge.get_media_vlm_analysis("image_vlm", "checksum-a", "profile-b", "prompt-v1") is None
    assert knowledge.get_media_vlm_analysis("image_vlm", "checksum-a", "profile-a", "prompt-v2") is None
    assert knowledge._conn.execute("PRAGMA user_version").fetchone()[0] == 5

    knowledge.delete_document("doc_vlm")
    assert knowledge._conn.execute("SELECT count(*) FROM media_vlm_analyses").fetchone()[0] == 0
    knowledge.close()
