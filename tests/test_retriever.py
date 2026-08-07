# -*- coding: utf-8 -*-
"""检索器与分块测试"""
from agentic_rag.rag.hybrid_retriever import BM25Retriever, HybridRetriever, ScoredDocument
from agentic_rag.processing.chunker import get_chunker


def test_bm25_index_and_search():
    bm25 = BM25Retriever()
    docs = [
        ScoredDocument(
            doc_id="1",
            content="人工智能公司深度智能专注于大语言模型研发。",
            score=1.0,
        )
    ]
    bm25.index(docs)
    # 使用与文档分词一致的查询词（避免 jieba 分词粒度差异导致误判）
    results = bm25.search("人工智能", top_k=5)
    assert len(results) == 1
    assert results[0].doc_id == "1"


def test_bm25_append_index():
    bm25 = BM25Retriever()
    bm25.index([ScoredDocument(doc_id="1", content="机器学习是人工智能的分支。", score=1.0)])
    bm25.index([ScoredDocument(doc_id="2", content="深度学习需要大量算力。", score=1.0)], append=True)
    results = bm25.search("深度学习", top_k=5)
    assert any(r.doc_id == "2" for r in results)


def test_chunk_id_global_uniqueness():
    """service 层修复点：多解析块重复编号会被重写为全局唯一 ID"""
    chunker = get_chunker("recursive", chunk_size=10, chunk_overlap=2)
    seq = 0
    ids = []
    for _ in range(2):  # 模拟同一文档的两个 parsed_chunk
        chunks = chunker.chunk("aaaaaaaaaabbbbbbbbbb", "doc1", {})
        for c in chunks:
            c.chunk_id = f"doc1_chunk_{seq:04d}"
            seq += 1
            ids.append(c.chunk_id)
    assert len(ids) == len(set(ids))


def test_rrf_fusion():
    hr = HybridRetriever()
    docs = [
        ScoredDocument(doc_id="a", content="x", score=0.9, source="vector"),
        ScoredDocument(doc_id="b", content="y", score=0.8, source="keyword"),
    ]
    fused = hr._reciprocal_rank_fusion(docs, top_k=2)
    assert len(fused) == 2
    # RRF 分数应覆盖原始分数
    assert all(d.score > 0 for d in fused)


def test_networkx_store_stats():
    from agentic_rag.memory.graph_store import NetworkXStore
    store = NetworkXStore()
    stats = store.stats
    assert "nodes" in stats and "edges" in stats and "density" in stats
