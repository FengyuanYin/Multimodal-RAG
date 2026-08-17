from __future__ import annotations

from math import log2
from typing import Iterable, Mapping, Sequence


def _id(item) -> str:
    if isinstance(item, Mapping):
        return str(item.get("id", item.get("doc_id", item.get("document_id", ""))))
    return str(getattr(item, "doc_id", getattr(item, "id", "")))


def precision_at_k(results: Sequence, expected: Iterable[str], k: int) -> float:
    wanted = set(expected)
    return sum(_id(item) in wanted for item in results[:k]) / max(1, k)


def recall_at_k(results: Sequence, expected: Iterable[str], k: int):
    wanted = set(expected)
    return sum(_id(item) in wanted for item in results[:k]) / len(wanted) if wanted else None


def reciprocal_rank(results: Sequence, expected: Iterable[str]) -> float:
    wanted = set(expected)
    return next((1.0 / rank for rank, item in enumerate(results, 1) if _id(item) in wanted), 0.0)


def ndcg_at_k(results: Sequence, expected: Iterable[str], k: int):
    wanted = set(expected)
    if not wanted:
        return None
    dcg = sum((1.0 if _id(item) in wanted else 0.0) / log2(rank + 1) for rank, item in enumerate(results[:k], 1))
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, min(k, len(wanted)) + 1))
    return dcg / ideal if ideal else 0.0


def media_recall_at_k(results: Sequence, expected_media: Iterable[str], k: int):
    wanted = set(expected_media)
    if not wanted:
        return None
    actual = set()
    for item in results[:k]:
        refs = item.get("media_refs", []) if isinstance(item, Mapping) else getattr(item, "media_refs", [])
        for ref in refs:
            actual.add(ref.get("media_id", "") if isinstance(ref, Mapping) else getattr(ref, "media_id", ""))
        metadata = item.get("metadata", {}) if isinstance(item, Mapping) else getattr(item, "metadata", {})
        actual.add(metadata.get("media_id", ""))
    return len(wanted & actual) / len(wanted)


def evaluate_ranking(results: Sequence, expected: Iterable[str], expected_media: Iterable[str] = (), k: int = 5) -> dict:
    return {
        "precision_at_k": precision_at_k(results, expected, k),
        "recall_at_k": recall_at_k(results, expected, k),
        "mrr": reciprocal_rank(results, expected),
        "ndcg_at_k": ndcg_at_k(results, expected, k),
        "media_recall_at_k": media_recall_at_k(results, expected_media, k),
    }
