from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.cli.models import ChunkRecord, DocumentRecord, GraphEdgeRecord, GraphNodeRecord
from agentic_rag.cli.services.graph_export import GraphExportService
from agentic_rag.cli.storage import KnowledgeRepository


def test_graph_export_writes_png_and_traceable_json(tmp_path: Path) -> None:
    repo = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    repo.commit_document(DocumentRecord("doc","fp","Doc","doc.txt","text","default","text",1,"ready"), [ChunkRecord("chunk","doc",1,0,"evidence")], [])
    nodes = [
        GraphNodeRecord("entity:a","default","entity","concept","Alpha","doc",1,"chunk"),
        GraphNodeRecord("entity:b","default","entity","concept","Beta","doc",1,"chunk"),
    ]
    edges = [GraphEdgeRecord("edge","default","entity","entity:a","entity:b","related_to","doc","chunk")]
    repo.replace_document_graph("doc","entity",nodes,edges)
    result = GraphExportService(repo, tmp_path / "exports").export("default","entity","map.png")
    assert result.png_path.read_bytes().startswith(b"\x89PNG")
    metadata = json.loads(result.metadata_path.read_text("utf-8"))
    assert metadata["edges"][0]["evidence_chunk_id"] == "chunk"
    repo.close()
