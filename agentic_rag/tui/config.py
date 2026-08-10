"""AutoMemory safe configuration and process-local secret handling."""

from __future__ import annotations

import os
from threading import RLock
from typing import Mapping
from urllib.parse import urlsplit

from .models import AutoMemoryConfig, RetrievalMode
from .paths import AutoMemoryPaths
from .security import looks_secret_key


class SecretStore:
    ENV_NAMES = {
        "llm_api_key": ("AUTOMEMORY_LLM_API_KEY", "AGR_LLM_API_KEY", "OPENAI_API_KEY"),
        "vlm_api_key": ("AUTOMEMORY_VLM_API_KEY", "AGR_VLM_API_KEY"),
        "mineru_api_key": ("AUTOMEMORY_MINERU_API_KEY", "MINERU_API_KEY"),
        "tavily_api_key": ("AUTOMEMORY_TAVILY_API_KEY", "TAVILY_API_KEY"),
    }

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ
        self._runtime: dict[str, str] = {}
        self._lock = RLock()

    def get(self, name: str) -> str:
        with self._lock:
            if name in self._runtime:
                return self._runtime[name]
            for env_name in self.ENV_NAMES.get(name, ()):
                value = self._environ.get(env_name, "").strip()
                if value:
                    return value
        return ""

    def set(self, name: str, value: str) -> None:
        if name not in self.ENV_NAMES:
            raise KeyError(f"unsupported secret: {name}")
        with self._lock:
            if value.strip():
                self._runtime[name] = value.strip()
            else:
                self._runtime.pop(name, None)

    def configured(self, name: str) -> bool:
        return bool(self.get(name))

    def values_for_redaction(self) -> tuple[str, ...]:
        return tuple(value for name in self.ENV_NAMES if (value := self.get(name)))

    def __repr__(self) -> str:
        flags = {name: self.configured(name) for name in self.ENV_NAMES}
        return f"SecretStore(configured={flags})"


def validate_config(config: AutoMemoryConfig) -> AutoMemoryConfig:
    if config.retrieval_mode not in {item.value for item in RetrievalMode}:
        raise ValueError("invalid retrieval mode")
    if config.web_provider not in {"duckduckgo", "tavily"}:
        raise ValueError("invalid Web search provider")
    if config.mineru_mode not in {"official", "selfhost"}:
        raise ValueError("invalid MinerU mode")
    if not 1 <= int(config.top_k) <= 50:
        raise ValueError("top_k must be between 1 and 50")
    if not 64 <= int(config.chunk_size) <= 4096:
        raise ValueError("chunk_size must be between 64 and 4096")
    if not 0 <= int(config.chunk_overlap) < int(config.chunk_size):
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    for label, value in (("LLM Base URL", config.llm_base_url), ("VLM Base URL", config.vlm_base_url), ("MinerU URL", config.mineru_url)):
        if value:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError(f"{label} must be a credential-free HTTP(S) URL")
    return config


def safe_config_payload(config: AutoMemoryConfig) -> dict:
    payload = config.to_safe_dict()
    forbidden = [key for key in payload if looks_secret_key(key)]
    if forbidden:
        raise ValueError(f"secret fields cannot be persisted: {', '.join(forbidden)}")
    return payload


def to_project_settings(config: AutoMemoryConfig, secrets: SecretStore, paths: AutoMemoryPaths):
    from agentic_rag.config import settings

    return settings.model_copy(update={
        "app_name": "AutoMemory",
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "llm_base_url": config.llm_base_url or None,
        "llm_api_key": secrets.get("llm_api_key") or None,
        # Empty means keyword-only startup. Loading a local embedding model is
        # always an explicit opt-in in the TUI so startup remains lightweight.
        "embedding_model": config.embedding_model or "",
        "reranker_model": config.reranker_model or "",
        "vlm_model": config.vlm_model or "gpt-4o-mini",
        "vlm_base_url": config.vlm_base_url or None,
        "vlm_api_key": secrets.get("vlm_api_key") or None,
        "enable_multimodal_retrieval": config.retrieval_mode == RetrievalMode.MULTIMODAL.value,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "knowledge_db_path": str(paths.knowledge_db),
        "vector_db_path": str(paths.vector_dir),
        "media_store_path": str(paths.media_file),
        "log_file": str(paths.logs_dir / "automemory.log"),
    })
