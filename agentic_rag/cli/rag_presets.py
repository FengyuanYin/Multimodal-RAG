"""Product-level RAG presets for the AutoMemory terminal."""

from __future__ import annotations

from types import MappingProxyType

from .errors import ConfigurationError
from .models import RagPreset


_PRESETS = MappingProxyType({
    "fast": RagPreset("fast", "BM25 keyword search; fastest and no embedding cost", frozenset({"keyword"}), top_k=5, candidate_k=20),
    "balanced": RagPreset("balanced", "BM25 + cloud embedding; recommended", frozenset({"keyword", "vector"}), top_k=5, candidate_k=30),
    "multimodal": RagPreset("multimodal", "Balanced retrieval plus figures, tables, and reference evidence", frozenset({"keyword", "vector", "multimodal", "reference_graph"}), top_k=6, candidate_k=36),
    "advanced": RagPreset(
        "advanced",
        "Query rewrite + BM25 + embedding + multimodal + dual graphs + rerank + context window",
        frozenset({"keyword", "vector", "multimodal", "entity_graph", "reference_graph"}),
        rewrite_queries=True, rerank=True, top_k=6, candidate_k=48,
        window_before=1, window_after=1, graph_depth=2, rewrite_limit=3,
    ),
})

_LEGACY = {"keyword": "fast", "vector": "balanced", "hybrid": "balanced", "multimodal": "multimodal"}


def get_preset(name: str) -> RagPreset:
    try:
        return _PRESETS[name.strip().lower()]
    except KeyError as exc:
        raise ConfigurationError("RAG mode must be fast, balanced, multimodal, or advanced") from exc


def list_presets() -> list[RagPreset]:
    return list(_PRESETS.values())


def migrate_retrieval_mode(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in _PRESETS:
        return lowered
    return _LEGACY.get(lowered, "balanced")
