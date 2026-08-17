"""Keyword, cloud-vector, hybrid, multimodal, and cloud-reranked retrieval."""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any

from ..cancellation import CancellationToken
from ..models import RetrievalHit, RetrievalResult
from ..rag_presets import get_preset, migrate_retrieval_mode
from ...memory.vector_store import VectorFilter


class RetrievalService:
    def __init__(self, knowledge, config, *, vector_store=None, embedding_client=None, reranker_client=None, query_rewriter=None, graph_retriever=None, visual_router=None) -> None:
        self.knowledge, self.config = knowledge, config
        self.vector_store = vector_store
        self.embedding_client, self.reranker_client = embedding_client, reranker_client
        self.query_rewriter, self.graph_retriever = query_rewriter, graph_retriever
        self.visual_router = visual_router
        self._lock = RLock()
        self._bm25 = None
        self._corpus: list[dict[str, Any]] = []
        self._corpus_signature: tuple[int, str] | None = None

    def rebuild(self, cancel: CancellationToken | None = None) -> int:
        if cancel:
            cancel.checkpoint()
        chunks = self.knowledge.list_chunks()
        with self._lock:
            self._corpus = chunks
            self._bm25 = None
            if chunks:
                from rank_bm25 import BM25Okapi
                import jieba
                jieba.setLogLevel(logging.WARNING)
                tokenized = [[token.lower() for token in jieba.cut(item["text"])] for item in chunks]
                self._bm25 = BM25Okapi(tokenized)
            self._corpus_signature = (len(chunks), chunks[-1]["id"] if chunks else "")
        return len(chunks)

    def search(self, query: str, scope: str, mode: str, top_k: int, cancel: CancellationToken) -> RetrievalResult:
        cancel.checkpoint()
        preset = get_preset(migrate_retrieval_mode(mode))
        requested_top_k = top_k or preset.top_k
        candidate_k = max(requested_top_k, preset.candidate_k)
        rewrite = self.query_rewriter.rewrite(query, preset.rewrite_limit, cancel) if preset.rewrite_queries and self.query_rewriter else None
        queries = rewrite.queries if rewrite else [query]
        trace = {"requested_mode": preset.name, "queries": queries, "channels": {}, "degraded": list(rewrite.degraded if rewrite else [])}
        channel_results: dict[str, list[RetrievalHit]] = {}
        for query_index, variant in enumerate(queries):
            prefix = f"q{query_index}:"
            if "keyword" in preset.channels:
                channel_results[prefix + "keyword"] = self._keyword(variant, scope, candidate_k, cancel)
            if "vector" in preset.channels:
                if not self.embedding_client or not self.vector_store:
                    trace["degraded"].append({"channel": "vector", "reason": "embedding or Milvus not configured"})
                else:
                    try:
                        channel_results[prefix + "vector"] = self._vector(variant, scope, candidate_k, cancel)
                    except Exception as exc:
                        trace["degraded"].append({"channel": "vector", "reason": type(exc).__name__})
            if self.graph_retriever and "entity_graph" in preset.channels:
                channel_results[prefix + "entity_graph"] = self.graph_retriever.search(variant, scope, "entity", preset.graph_depth, candidate_k, cancel)
            if self.graph_retriever and "reference_graph" in preset.channels:
                channel_results[prefix + "reference_graph"] = self.graph_retriever.search(variant, scope, "reference", max(1,preset.graph_depth), candidate_k, cancel)
        trace["channels"] = {name: len(items) for name, items in channel_results.items()}
        fused = self._fuse(channel_results, candidate_k)
        if "multimodal" in preset.channels:
            for hit in fused:
                if hit.media_refs:
                    hit.channel_scores["multimodal"] = 1.0
                    hit.score += 0.01
            fused.sort(key=lambda item: (-item.score, item.target_id))
            trace["channels"]["multimodal"] = sum(bool(item.media_refs) for item in fused)
        trace["fusion"] = [{"target_id": item.target_id, "score": item.score} for item in fused[:candidate_k]]
        if preset.rerank and self.reranker_client and fused:
            try:
                fused = self.reranker_client.rerank(query, fused, requested_top_k, cancel)
                trace["channels"]["reranker"] = len(fused)
            except Exception as exc:
                trace["degraded"].append({"channel": "reranker", "reason": type(exc).__name__})
        elif preset.rerank:
            trace["degraded"].append({"channel": "reranker", "reason": "cloud reranker not configured"})
        hits = fused[:requested_top_k]
        if preset.name == "advanced" and self.visual_router is not None:
            visual_report = self.visual_router.enrich(hits, cancel)
            trace["advanced_vlm"] = visual_report
            if visual_report.get("degraded"):
                trace["degraded"].append({
                    "channel": "advanced_vlm",
                    "reason": "partial_failure",
                    "count": len(visual_report["degraded"]),
                })
        if preset.window_before or preset.window_after:
            for hit in hits:
                hit.window_before, hit.window_after = self.knowledge.get_chunk_window(hit.target_id, preset.window_before, preset.window_after)
            trace["sentence_window"] = [{"target_id": item.target_id,"before":[x["id"] for x in item.window_before],"after":[x["id"] for x in item.window_after]} for item in hits]
        trace["result_count"] = len(hits)
        return RetrievalResult(hits, trace)

    def _keyword(self, query: str, scope: str, top_k: int, cancel: CancellationToken) -> list[RetrievalHit]:
        chunks = self.knowledge.list_chunks(category_id=scope)
        signature = (len(chunks), chunks[-1]["id"] if chunks else "")
        if self._corpus_signature != signature or scope != "all":
            from rank_bm25 import BM25Okapi
            import jieba
            jieba.setLogLevel(logging.WARNING)
            corpus = chunks
            tokenized = [[token.lower() for token in jieba.cut(item["text"])] for item in corpus]
            bm25 = BM25Okapi(tokenized) if tokenized else None
        else:
            corpus, bm25 = self._corpus, self._bm25
        if not bm25:
            return []
        import jieba
        jieba.setLogLevel(logging.WARNING)
        cancel.checkpoint()
        scores = bm25.get_scores([token.lower() for token in jieba.cut(query)])
        ranked = sorted(enumerate(scores), key=lambda pair: (-float(pair[1]), corpus[pair[0]]["id"]))[:top_k]
        return [self._hit(corpus[index], float(score), "keyword") for index, score in ranked if float(score) > 0]

    def _vector(self, query: str, scope: str, top_k: int, cancel: CancellationToken) -> list[RetrievalHit]:
        vectors = self.embedding_client.embeddings([query], cancel)
        if not vectors:
            return []
        cancel.checkpoint()
        vector_filter = VectorFilter(
            namespace="cli",
            knowledge_base_id=None if scope == "all" else scope,
            profile_fingerprint=self.embedding_client.profile_fingerprint,
        )
        rows = self.vector_store.search(vectors[0], top_k=top_k, filter=vector_filter)
        output = []
        for row in rows:
            item = row.payload
            output.append(RetrievalHit(
                row.id, str(item.get("document_id", "")), str(item.get("document", "")),
                str(item.get("text") or item.get("content") or ""), int(item.get("page", 1)),
                str(item.get("modality", "text")), float(row.score), {"vector": float(row.score)},
                list(item.get("media_refs") or []),
            ))
        return output

    @staticmethod
    def _hit(item: dict[str, Any], score: float, channel: str) -> RetrievalHit:
        return RetrievalHit(item["id"], item["document_id"], item["document"], item["text"], int(item["page"]), item["modality"], score, {channel: score}, item.get("media_refs") or [])

    @staticmethod
    def _fuse(channels: dict[str, list[RetrievalHit]], limit: int) -> list[RetrievalHit]:
        fused: dict[str, RetrievalHit] = {}
        for channel, hits in channels.items():
            for rank, hit in enumerate(hits, 1):
                score = 1.0 / (60 + rank)
                if hit.target_id not in fused:
                    fused[hit.target_id] = RetrievalHit(hit.target_id, hit.document_id, hit.document, hit.text, hit.page, hit.modality, 0.0, {}, list(hit.media_refs), graph_paths=list(hit.graph_paths))
                fused[hit.target_id].score += score
                fused[hit.target_id].channel_scores[channel] = hit.score
                for path in hit.graph_paths:
                    if path not in fused[hit.target_id].graph_paths:
                        fused[hit.target_id].graph_paths.append(path)
        return sorted(fused.values(), key=lambda item: (-item.score, item.target_id))[:limit]
