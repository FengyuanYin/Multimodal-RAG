from __future__ import annotations

from pathlib import Path

from agentic_rag.cli.cancellation import CancellationToken
from agentic_rag.cli.config import AutoMemoryConfig
from agentic_rag.cli.models import ChunkRecord, DocumentRecord, RetrievalHit
from agentic_rag.cli.services.query_rewrite import QueryRewriteResult
from agentic_rag.cli.services.retrieval import RetrievalService
from agentic_rag.cli.storage import KnowledgeRepository
from agentic_rag.memory.vector_store import SearchResult


class Embedder:
    profile_fingerprint = "profile"
    def embeddings(self, texts, cancel):
        return [[1.0, 0.0] for _ in texts]


class Vectors:
    def __init__(self):
        self.filters = []

    def search(self, vector, top_k, filter):
        self.filters.append(filter)
        return [SearchResult("c1", 0.9, {
            "document_id": "doc", "document": "Doc", "text": "alpha relation",
            "page": 1, "modality": "text", "media_refs": [],
        })]


class Rewriter:
    def rewrite(self, question, limit, cancel):
        return QueryRewriteResult([question, "alpha relation"])


class Graphs:
    def search(self, query, scope, kind, depth, limit, cancel):
        return [RetrievalHit("c1","doc","Doc","alpha relation",1,"text",1.0,{kind+"_graph":1.0},[],graph_paths=[{"source":"Alpha","relation":"related_to","target":"Beta"}])]


class Reranker:
    def __init__(self, events): self.called = False; self.events = events
    def rerank(self, query, candidates, top_k, cancel):
        self.called = True
        self.events.append("rerank")
        assert all(not item.window_before and not item.window_after for item in candidates)
        return candidates[:top_k]


class VisualRouter:
    def __init__(self, reranker, events):
        self.reranker, self.events = reranker, events

    def enrich(self, hits, cancel):
        assert self.reranker.called
        assert all(not item.window_before and not item.window_after for item in hits)
        self.events.append("visual")
        return {"eligible_images": 0, "unique_images": 0, "cache_hits": 0, "primary_calls": 0, "fallback_calls": 0, "analyzed": 0, "degraded": []}


def test_advanced_pipeline_rewrites_fuses_reranks_then_windows(tmp_path: Path) -> None:
    repo = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    chunks = [ChunkRecord(f"c{i}","doc",1,i,text) for i,text in enumerate(["before context","alpha relation","after context"])]
    repo.commit_document(DocumentRecord("doc","fp","Doc","doc.txt","text","default","text",1,"ready"),chunks,[])
    events = []
    reranker = Reranker(events)
    vectors = Vectors()
    visuals = VisualRouter(reranker, events)
    service = RetrievalService(repo,AutoMemoryConfig(rag_mode="advanced"),vector_store=vectors,embedding_client=Embedder(),reranker_client=reranker,query_rewriter=Rewriter(),graph_retriever=Graphs(),visual_router=visuals)
    service.rebuild()
    result = service.search("alpha", "default", "advanced", 1, CancellationToken())
    assert reranker.called
    assert result.trace["queries"] == ["alpha", "alpha relation"]
    assert any("entity_graph" in name for name in result.trace["channels"])
    assert result.hits[0].window_before[0]["id"] == "c0"
    assert result.hits[0].window_after[0]["id"] == "c2"
    assert result.hits[0].graph_paths
    assert events == ["rerank", "visual"]
    assert result.trace["advanced_vlm"]["primary_calls"] == 0
    assert all(item.namespace == "cli" and item.knowledge_base_id == "default" for item in vectors.filters)
    repo.close()


def test_visual_router_only_receives_post_rerank_top_k_hits(tmp_path: Path) -> None:
    repo = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")
    seen = []

    class KeepOnlyReranker:
        def rerank(self, query, candidates, top_k, cancel):
            return [next(item for item in candidates if item.target_id == "keep")]

    class CaptureVisuals:
        def enrich(self, hits, cancel):
            seen.extend(item.target_id for item in hits)
            return {"eligible_images": 0, "unique_images": 0, "cache_hits": 0, "primary_calls": 0, "fallback_calls": 0, "analyzed": 0, "degraded": []}

    service = RetrievalService(repo, AutoMemoryConfig(rag_mode="advanced"), reranker_client=KeepOnlyReranker(), visual_router=CaptureVisuals())
    service._keyword = lambda *_args: [
        RetrievalHit("keep", "doc", "Doc", "keep", 1, "text", 1.0),
        RetrievalHit("drop", "doc", "Doc", "drop", 1, "text", 0.9, media_refs=[{"media_id": "dropped-image", "media_type": "image"}]),
    ]

    result = service.search("query", "default", "advanced", 1, CancellationToken())

    assert [item.target_id for item in result.hits] == ["keep"]
    assert seen == ["keep"]
    repo.close()


def test_non_advanced_mode_never_calls_visual_router(tmp_path: Path) -> None:
    repo = KnowledgeRepository(tmp_path / "knowledge.db", tmp_path / "backups")

    class FailIfCalled:
        def enrich(self, hits, cancel):
            raise AssertionError("visual router must be Advanced-only")

    service = RetrievalService(repo, AutoMemoryConfig(rag_mode="balanced"), visual_router=FailIfCalled())
    service._keyword = lambda *_args: [RetrievalHit("hit", "doc", "Doc", "evidence", 1, "text", 1.0, media_refs=[{"media_id": "image", "media_type": "image"}])]

    result = service.search("query", "default", "balanced", 1, CancellationToken())

    assert result.hits[0].target_id == "hit"
    assert "advanced_vlm" not in result.trace
    repo.close()
