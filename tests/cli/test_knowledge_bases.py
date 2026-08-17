from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.cli.errors import UsageError
from agentic_rag.cli.models import ChunkRecord, DocumentRecord
from agentic_rag.cli.storage import KnowledgeRepository


def test_knowledge_base_lifecycle_and_document_scope(tmp_path: Path) -> None:
    repo = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    item = repo.create_knowledge_base("Research")
    assert repo.resolve_knowledge_base("research") == item["id"]
    repo.commit_document(DocumentRecord("doc_a","fp_a","A","a.txt","text",item["id"],"text",1,"ready"), [ChunkRecord("chunk_a","doc_a",1,0,"alpha")], [])
    assert repo.get_document_in_base("doc_a", item["id"])
    assert repo.get_document_in_base("doc_a", "default") is None
    with pytest.raises(UsageError):
        repo.delete_knowledge_base(item["id"])
    repo.delete_knowledge_base(item["id"], force=True)
    assert repo.get_document("doc_a") is None
    with pytest.raises(UsageError):
        repo.delete_knowledge_base("default", force=True)
    repo.close()


def test_sentence_window_never_crosses_documents(tmp_path: Path) -> None:
    repo = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    for doc_id in ("a", "b"):
        repo.commit_document(DocumentRecord(doc_id,"fp_"+doc_id,doc_id,doc_id,"text","default","text",1,"ready"), [ChunkRecord(f"{doc_id}{n}",doc_id,1,n,f"{doc_id}-{n}") for n in range(3)], [])
    before, after = repo.get_chunk_window("a1", 1, 1)
    assert [item["id"] for item in before] == ["a0"]
    assert [item["id"] for item in after] == ["a2"]
    repo.close()
