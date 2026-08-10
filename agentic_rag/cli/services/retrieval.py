"""Keyword, cloud-vector, hybrid, multimodal, and cloud-reranked retrieval."""

from __future__ import annotations

import math
from threading import RLock
from typing import Any

from ..cancellation import CancellationToken
from ..models import RetrievalHit, RetrievalResult


class RetrievalService:
    def __init__(self, knowledge, config, *, embedding_client=None, reranker_client=None) -> None:
        self.knowledge, self.config = knowledge, config
        self.embedding_client, self.reranker_client = embedding_client, reranker_client
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
                tokenized = [[token.lower() for token in jieba.cut(item["text"])] for item in chunks]
                self._bm25 = BM25Okapi(tokenized)
            self._corpus_signature = (len(chunks), chunks[-1]["id"] if chunks else "")
        return len(chunks)

    def search(self, query: str, scope: str, mode: str, top_k: int, cancel: CancellationToken) -> RetrievalResult:
        cancel.checkpoint()
        candidate_k = max(top_k, int(self.config.candidate_k))
        trace = {"requested_mode": mode, "channels": {}, "degraded": []}
        channel_results: dict[str, list[RetrievalHit]] = {}
        if mode in {"keyword", "hybrid", "multimodal"}:
            channel_results["keyword"] = self._keyword(query, scope, candidate_k, cancel)
            trace["channels"]["keyword"] = len(channel_results["keyword"])
        if mode in {"vector", "hybrid", "multimodal"}:
            if not self.embedding_client:
                trace["degraded"].append({"channel": "vector", "reason": "cloud embedding not configured"})
            else:
                channel_results["vector"] = self._vector(query, scope, candidate_k, cancel)
                trace["channels"]["vector"] = len(channel_results["vector"])
        fused = self._fuse(channel_results, candidate_k)
        if mode == "multimodal":
            for hit in fused:
                if hit.media_refs:
                    hit.channel_scores["multimodal"] = 1.0
                    hit.score += 0.01
            fused.sort(key=lambda item: (-item.score, item.target_id))
            trace["channels"]["multimodal"] = sum(bool(item.media_refs) for item in fused)
        if self.reranker_client and fused:
            try:
                fused = self.reranker_client.rerank(query, fused, top_k, cancel)
                trace["channels"]["reranker"] = len(fused)
            except Exception as exc:
                trace["degraded"].append({"channel": "reranker", "reason": type(exc).__name__})
        else:
            trace["degraded"].append({"channel": "reranker", "reason": "cloud reranker not configured"})
        hits = fused[:top_k]
        trace["result_count"] = len(hits)
        return RetrievalResult(hits, trace)

    def _keyword(self, query: str, scope: str, top_k: int, cancel: CancellationToken) -> list[RetrievalHit]:
        chunks = self.knowledge.list_chunks(category_id=scope)
        signature = (len(chunks), chunks[-1]["id"] if chunks else "")
        if self._corpus_signature != signature or scope != "all":
            from rank_bm25 import BM25Okapi
            import jieba
            corpus = chunks
            tokenized = [[token.lower() for token in jieba.cut(item["text"])] for item in corpus]
            bm25 = BM25Okapi(tokenized) if tokenized else None
        else:
            corpus, bm25 = self._corpus, self._bm25
        if not bm25:
            return []
        import jieba
        cancel.checkpoint()
        scores = bm25.get_scores([token.lower() for token in jieba.cut(query)])
        ranked = sorted(enumerate(scores), key=lambda pair: (-float(pair[1]), corpus[pair[0]]["id"]))[:top_k]
        return [self._hit(corpus[index], float(score), "keyword") for index, score in ranked if float(score) > 0]

    def _vector(self, query: str, scope: str, top_k: int, cancel: CancellationToken) -> list[RetrievalHit]:
        rows = self.knowledge.list_embeddings(self.embedding_client.profile_fingerprint, scope)
        if len(rows) > int(self.config.max_vector_items):
            rows = rows[: int(self.config.max_vector_items)]
        vectors = self.embedding_client.embeddings([query], cancel)
        if not vectors:
            return []
        query_vector = vectors[0]
        norm = math.sqrt(sum(value * value for value in query_vector))
        if not norm:
            return []
        normalized = [value / norm for value in query_vector]
        ranked = []
        for row in rows:
            cancel.checkpoint()
            if len(row["vector"]) != len(normalized):
                continue
            score = sum(left * right for left, right in zip(normalized, row["vector"]))
            ranked.append((score, row))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["target_id"]))
        return [RetrievalHit(item["target_id"], item["document_id"], item["document"], item["text"], int(item["page"]), item["modality"], float(score), {"vector": float(score)}, item["media_refs"]) for score, item in ranked[:top_k]]

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
                    fused[hit.target_id] = RetrievalHit(hit.target_id, hit.document_id, hit.document, hit.text, hit.page, hit.modality, 0.0, {}, list(hit.media_refs))
                fused[hit.target_id].score += score
                fused[hit.target_id].channel_scores[channel] = hit.score
        return sorted(fused.values(), key=lambda item: (-item.score, item.target_id))[:limit]
