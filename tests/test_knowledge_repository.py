import sqlite3

import pytest

from agentic_rag.memory.knowledge_repository import KnowledgeRepository


def _fixture():
    document = {"id": "doc_a", "fingerprint": "sha256:a", "name": "年度报告"}
    chunks = [{"id": "chunk_a", "text": "如图1所示，收入增长。", "metadata": {"page": 2}}]
    media = [{"id": "img_a", "type": "image", "page": 2, "label": "图1", "search_text": "收入趋势图"}]
    refs = [{
        "id": "ref_a", "chunk_id": "chunk_a", "media_id": "img_a", "page": 2,
        "label": "图1", "media_type": "image", "confidence": 1.0, "resolution": "exact",
    }]
    return document, chunks, media, refs


def test_transactional_upsert_and_cascade_delete(tmp_path):
    repo = KnowledgeRepository(str(tmp_path / "knowledge.db"))
    document, chunks, media, refs = _fixture()
    assert repo.upsert_document(document, chunks, media, refs) == "doc_a"
    assert repo.integrity_check() == "ok"
    assert repo.list_chunks("doc_a")[0]["page"] == 2
    assert repo.list_references("doc_a")[0]["media_id"] == "img_a"
    assert repo.delete_document("doc_a")
    assert repo.list_chunks("doc_a") == []
    assert repo.list_media("doc_a") == []
    assert repo.list_references("doc_a") == []
    repo.close()

def test_failed_upsert_rolls_back_everything(tmp_path):
    repo = KnowledgeRepository(str(tmp_path / "knowledge.db"))
    document, chunks, media, refs = _fixture()
    refs[0]["chunk_id"] = "missing"
    with pytest.raises(ValueError):
        repo.upsert_document(document, chunks, media, refs)
    assert repo.get_document("doc_a") is None
    assert repo.list_chunks() == []
    assert repo.list_media() == []
    repo.close()


def test_fingerprint_upsert_is_idempotent(tmp_path):
    repo = KnowledgeRepository(str(tmp_path / "knowledge.db"))
    document, chunks, media, refs = _fixture()
    repo.upsert_document(document, chunks, media, refs)
    document["id"] = "different_generated_id"
    chunks[0]["text"] = "更新后的内容"
    assert repo.upsert_document(document, chunks, media, refs) == "doc_a"
    assert len(repo.list_documents()) == 1
    assert repo.list_chunks("doc_a")[0]["text"] == "更新后的内容"
    repo.close()
