from __future__ import annotations

from agentic_rag.cli.provider_presets import match_preset, presets_for


def test_required_provider_catalog() -> None:
    assert {item.id for item in presets_for("llm")} == {"openai", "deepseek", "siliconflow", "custom"}
    assert {item.id for item in presets_for("embedding")} == {"openai", "deepseek", "siliconflow", "custom"}
    assert {item.id for item in presets_for("vlm")} == {"openai", "deepseek", "siliconflow", "custom"}
    assert {item.id for item in presets_for("reranker")} == {"cohere", "custom"}
    assert {item.id for item in presets_for("mineru")} == {"official", "selfhost"}
    assert {item.id for item in presets_for("web")} == {"duckduckgo", "tavily"}


def test_url_matching_ignores_slash_case_and_query() -> None:
    assert match_preset("llm", "https://API.OPENAI.com/v1/?ignored=1").id == "openai"
    assert match_preset("llm", "https://example.com/v1") is None
