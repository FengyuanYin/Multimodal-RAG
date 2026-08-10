"""Retrieval evaluation with progress, cancellation, and atomic export."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
import tempfile
from time import perf_counter
import math

from ..events import CancelToken, EventCallback, JobProgress
from ..security import ensure_within


def evaluate_ranking(results: list, expected: list, expected_media: list, top_k: int) -> dict:
    """Compute deterministic binary-relevance retrieval metrics."""
    relevant = {str(item) for item in expected}
    media_relevant = {str(item) for item in expected_media}
    ranked = results[:top_k]
    identifiers = []
    media_found = set()
    for item in ranked:
        metadata = getattr(item, "metadata", {}) or {}
        identifiers.append({str(getattr(item, "doc_id", "")), str(metadata.get("document_id", "")), str(metadata.get("doc_id", ""))})
        for ref in getattr(item, "media_refs", []) or metadata.get("media_refs", []):
            media_found.add(str(ref.get("media_id", "") if isinstance(ref, dict) else getattr(ref, "media_id", "")))
    hits = [bool(ids & relevant) for ids in identifiers]
    hit_count = sum(hits)
    precision = hit_count / max(1, top_k)
    recall = hit_count / len(relevant) if relevant else None
    first = next((index for index, hit in enumerate(hits, 1) if hit), None)
    mrr = 1 / first if first else 0.0
    dcg = sum(1 / math.log2(index + 1) for index, hit in enumerate(hits, 1) if hit)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(relevant), top_k) + 1))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "ndcg_at_k": dcg / ideal if ideal else None,
        "media_recall_at_k": len(media_found & media_relevant) / len(media_relevant) if media_relevant else None,
    }


class EvaluationService:
    def __init__(self, runtime, state) -> None:
        self.runtime = runtime
        self.state = state

    @staticmethod
    def load_dataset(path: Path) -> list[dict]:
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation dataset: {exc}") from exc
        if isinstance(payload, dict):
            payload = payload.get("cases") or payload.get("items")
        if not isinstance(payload, list) or not payload:
            raise ValueError("evaluation dataset must contain a non-empty list")
        for index, item in enumerate(payload):
            if not isinstance(item, dict) or not str(item.get("query") or "").strip():
                raise ValueError(f"evaluation case {index + 1} is missing query")
        return payload

    def run(self, path: Path, top_k: int = 5, mode: str = "keyword", emit: EventCallback | None = None, cancel: CancelToken | None = None, job_id: str = "evaluation") -> dict:
        cancel = cancel or CancelToken()
        dataset = self.load_dataset(path)
        run_id = self.state.create_evaluation_run(str(path), {"top_k": top_k, "mode": mode})
        cases = []
        try:
            for index, item in enumerate(dataset, 1):
                cancel.checkpoint()
                if emit:
                    emit(JobProgress(job_id, "evaluation", "retrieve", str(item["query"]), index - 1, len(dataset)))
                started = perf_counter()
                use_vector = mode in {"vector", "hybrid", "multimodal"}
                use_keyword = mode in {"keyword", "hybrid", "multimodal"}
                results = self.runtime.retriever.retrieve(str(item["query"]), top_k=top_k, use_vector=use_vector, use_keyword=use_keyword)
                reranker = getattr(self.runtime.orchestrator, "reranker", None)
                if reranker and results:
                    results = reranker.rerank(str(item["query"]), results, top_k=top_k)
                metrics = evaluate_ranking(results, item.get("expected", []), item.get("expected_media", []), top_k)
                cases.append({"id": item.get("id", f"case_{index}"), "query": item["query"], **metrics, "latency_ms": round((perf_counter() - started) * 1000, 3)})
            fields = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "media_recall_at_k", "latency_ms"]
            summary = {name: round(mean(values), 6) if (values := [case[name] for case in cases if case.get(name) is not None]) else None for name in fields}
            result = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "top_k": top_k, "mode": mode, "count": len(cases), "summary": summary, "cases": cases}
            self.state.finish_evaluation_run(run_id, "success")
            return result
        except Exception:
            self.state.finish_evaluation_run(run_id, "cancelled" if cancel.cancelled else "error")
            raise

    def export(self, result: dict, destination: Path, exports_root: Path) -> Path:
        destination = ensure_within(exports_root, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".automemory-eval-", suffix=".json", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.state.finish_evaluation_run(str(result.get("run_id", "")), "success", str(destination))
        return destination
