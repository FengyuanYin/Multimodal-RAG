from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable

from .metrics import evaluate_ranking


class EvaluationRunner:
    def __init__(self, retrieve: Callable, k: int = 5):
        self.retrieve = retrieve
        self.k = k

    def run(self, dataset: Iterable[dict]) -> dict:
        cases = []
        for index, item in enumerate(dataset):
            started = perf_counter()
            results = self.retrieve(item["query"], top_k=self.k)
            metrics = evaluate_ranking(results, item.get("expected", []), item.get("expected_media", []), self.k)
            cases.append({
                "id": item.get("id", f"case_{index + 1}"), "query": item["query"],
                **metrics, "latency_ms": round((perf_counter() - started) * 1000, 3),
            })
        names = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "media_recall_at_k", "latency_ms"]
        summary = {}
        for name in names:
            values = [case[name] for case in cases if case[name] is not None]
            summary[name] = round(mean(values), 6) if values else None
        return {
            "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "k": self.k, "count": len(cases), "summary": summary, "cases": cases,
        }
