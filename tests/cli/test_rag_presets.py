from __future__ import annotations

from agentic_rag.cli.config import AutoMemoryConfig
from agentic_rag.cli.rag_presets import get_preset, list_presets


def test_fixed_mode_matrix_and_legacy_migration() -> None:
    assert [item.name for item in list_presets()] == ["fast", "balanced", "multimodal", "advanced"]
    assert get_preset("fast").channels == {"keyword"}
    assert get_preset("balanced").channels == {"keyword", "vector"}
    assert "reference_graph" in get_preset("multimodal").channels
    advanced = get_preset("advanced")
    assert advanced.rewrite_queries and advanced.rerank
    assert advanced.window_before == advanced.window_after == 1
    assert AutoMemoryConfig.from_dict({"schema_version": 1, "retrieval_mode": "keyword", "active_category": "all"}).rag_mode == "fast"
    assert AutoMemoryConfig.from_dict({"schema_version": 1, "retrieval_mode": "hybrid"}).rag_mode == "balanced"


def test_new_config_uses_balanced_and_default_knowledge_base() -> None:
    config = AutoMemoryConfig()
    assert config.rag_mode == "balanced"
    assert config.active_category == "default"
