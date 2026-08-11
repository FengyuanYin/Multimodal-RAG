"""Auditable cloud-provider presets for the setup wizard."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from .models import ProviderPreset


_OPENAI = "https://api.openai.com/v1"
_DEEPSEEK = "https://api.deepseek.com"
_SILICONFLOW = "https://api.siliconflow.cn/v1"


PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset("openai", "llm", "OpenAI", "openai-compatible", _OPENAI, "gpt-4.1-mini", "llm_api_key"),
    ProviderPreset("deepseek", "llm", "DeepSeek", "openai-compatible", _DEEPSEEK, "deepseek-v4-flash", "llm_api_key"),
    ProviderPreset("siliconflow", "llm", "SiliconFlow", "openai-compatible", _SILICONFLOW, "deepseek-ai/DeepSeek-V3.2", "llm_api_key"),
    ProviderPreset("custom", "llm", "Custom OpenAI-compatible", "openai-compatible", "", "", "llm_api_key"),
    ProviderPreset("openai", "embedding", "OpenAI", "openai-compatible", _OPENAI, "text-embedding-3-small", "embedding_api_key"),
    ProviderPreset("deepseek", "embedding", "DeepSeek-compatible (model required)", "openai-compatible", _DEEPSEEK, "", "embedding_api_key"),
    ProviderPreset("siliconflow", "embedding", "SiliconFlow", "openai-compatible", _SILICONFLOW, "BAAI/bge-m3", "embedding_api_key"),
    ProviderPreset("custom", "embedding", "Custom OpenAI-compatible", "openai-compatible", "", "", "embedding_api_key"),
    ProviderPreset("openai", "vlm", "OpenAI", "openai-compatible", _OPENAI, "gpt-4.1-mini", "vlm_api_key"),
    ProviderPreset("deepseek", "vlm", "DeepSeek-compatible (vision model required)", "openai-compatible", _DEEPSEEK, "", "vlm_api_key"),
    ProviderPreset("siliconflow", "vlm", "SiliconFlow", "openai-compatible", _SILICONFLOW, "Qwen/Qwen2.5-VL-72B-Instruct", "vlm_api_key"),
    ProviderPreset("custom", "vlm", "Custom OpenAI-compatible", "openai-compatible", "", "", "vlm_api_key"),
    ProviderPreset("siliconflow", "reranker", "SiliconFlow", "siliconflow-rerank", _SILICONFLOW, "BAAI/bge-reranker-v2-m3", "reranker_api_key"),
    ProviderPreset("cohere", "reranker", "Cohere", "cohere-compatible", "https://api.cohere.com", "rerank-v3.5", "reranker_api_key"),
    ProviderPreset("custom", "reranker", "Custom Cohere-compatible", "cohere-compatible", "", "", "reranker_api_key"),
    ProviderPreset("official", "mineru", "MinerU Official", "mineru", "https://mineru.net/api/v4", "", "mineru_api_key"),
    ProviderPreset("selfhost", "mineru", "Self-hosted MinerU", "mineru", "", "", "mineru_api_key", False),
    ProviderPreset("duckduckgo", "web", "DuckDuckGo (free)", "search", "https://html.duckduckgo.com", "", "", False),
    ProviderPreset("tavily", "web", "Tavily", "search", "https://api.tavily.com", "", "tavily_api_key"),
)


def presets_for(service: str) -> tuple[ProviderPreset, ...]:
    return tuple(item for item in PRESETS if item.service == service)


def normalize_base_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if not value:
        return ""
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def match_preset(service: str, base_url: str) -> ProviderPreset | None:
    normalized = normalize_base_url(base_url)
    return next(
        (item for item in presets_for(service) if item.id != "custom" and normalize_base_url(item.base_url) == normalized),
        None,
    )


def validate_catalog() -> None:
    seen: set[tuple[str, str]] = set()
    for item in PRESETS:
        key = (item.service, item.id)
        if key in seen:
            raise ValueError(f"Duplicate provider preset: {item.service}/{item.id}")
        seen.add(key)
        if item.base_url and (urlsplit(item.base_url).scheme not in {"http", "https"} or urlsplit(item.base_url).query):
            raise ValueError(f"Invalid provider URL: {item.service}/{item.id}")


validate_catalog()
