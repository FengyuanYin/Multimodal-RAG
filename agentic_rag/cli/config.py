"""Versioned non-secret configuration with atomic Windows-safe writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .errors import ConfigurationError
from .models import ServiceProfile
from .security import looks_secret_name, validate_http_url


SCHEMA_VERSION = 1


def _default_llm() -> ServiceProfile:
    return ServiceProfile("https://api.openai.com/v1", "gpt-4o-mini", "llm_api_key", 120.0, 2, 1)


def _default_embedding() -> ServiceProfile:
    return ServiceProfile("https://api.openai.com/v1", "text-embedding-3-small", "embedding_api_key", 60.0, 2, 32)


def _default_vlm() -> ServiceProfile:
    return ServiceProfile("https://api.openai.com/v1", "gpt-4o-mini", "vlm_api_key", 120.0, 2, 1)


def _default_reranker() -> ServiceProfile:
    return ServiceProfile("https://api.cohere.com", "rerank-v3.5", "reranker_api_key", 60.0, 2, 64)


@dataclass(slots=True)
class AutoMemoryConfig:
    schema_version: int = SCHEMA_VERSION
    llm: ServiceProfile = field(default_factory=_default_llm)
    embedding: ServiceProfile = field(default_factory=_default_embedding)
    vlm: ServiceProfile = field(default_factory=_default_vlm)
    reranker: ServiceProfile = field(default_factory=_default_reranker)
    mineru_mode: str = "official"
    mineru_url: str = "https://mineru.net/api/v4"
    web_provider: str = "duckduckgo"
    retrieval_mode: str = "keyword"
    top_k: int = 5
    candidate_k: int = 25
    chunk_size: int = 800
    chunk_overlap: int = 120
    memory_enabled: bool = True
    active_category: str = "all"
    max_vector_items: int = 50_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "AutoMemoryConfig":
        value = dict(value or {})
        version = int(value.get("schema_version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            raise ConfigurationError(f"Config schema {version} is newer than this AutoMemory version supports")
        base = cls()
        scalar_fields = {
            "mineru_mode", "mineru_url", "web_provider", "retrieval_mode", "top_k",
            "candidate_k", "chunk_size", "chunk_overlap", "memory_enabled",
            "active_category", "max_vector_items",
        }
        for name in scalar_fields:
            if name in value:
                setattr(base, name, value[name])
        base.llm = ServiceProfile.from_dict(value.get("llm", {}), _default_llm())
        base.embedding = ServiceProfile.from_dict(value.get("embedding", {}), _default_embedding())
        base.vlm = ServiceProfile.from_dict(value.get("vlm", {}), _default_vlm())
        base.reranker = ServiceProfile.from_dict(value.get("reranker", {}), _default_reranker())
        return validate_config(base)


def validate_config(config: AutoMemoryConfig) -> AutoMemoryConfig:
    if config.retrieval_mode not in {"keyword", "vector", "hybrid", "multimodal"}:
        raise ConfigurationError("retrieval_mode must be keyword, vector, hybrid, or multimodal")
    if config.web_provider not in {"duckduckgo", "tavily"}:
        raise ConfigurationError("web_provider must be duckduckgo or tavily")
    if config.mineru_mode not in {"official", "selfhost"}:
        raise ConfigurationError("mineru_mode must be official or selfhost")
    if not 1 <= int(config.top_k) <= 50:
        raise ConfigurationError("top_k must be between 1 and 50")
    if not int(config.top_k) <= int(config.candidate_k) <= 250:
        raise ConfigurationError("candidate_k must be between top_k and 250")
    if not 64 <= int(config.chunk_size) <= 8192:
        raise ConfigurationError("chunk_size must be between 64 and 8192")
    if not 0 <= int(config.chunk_overlap) < int(config.chunk_size):
        raise ConfigurationError("chunk_overlap must be smaller than chunk_size")
    for name in ("llm", "embedding", "vlm", "reranker"):
        profile = getattr(config, name)
        validate_http_url(profile.base_url, allow_private=True, resolve=False)
        if not profile.model.strip() or looks_secret_name(profile.model):
            raise ConfigurationError(f"{name}.model is invalid")
        if not profile.credential_name or looks_secret_name(profile.credential_name) is False:
            raise ConfigurationError(f"{name}.credential_name must identify a credential")
        if not 1 <= int(profile.retries) <= 5 or not 1 <= float(profile.timeout_seconds) <= 600:
            raise ConfigurationError(f"{name} timeout/retries are outside allowed limits")
    validate_http_url(config.mineru_url, allow_private=True, resolve=False)
    return config


class ConfigStore:
    def __init__(self, path: Path, backups_dir: Path) -> None:
        self.path, self.backups_dir = path, backups_dir

    def load(self) -> AutoMemoryConfig:
        if not self.path.exists():
            return AutoMemoryConfig()
        try:
            value = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Config file is invalid: {exc}", hint=f"Move or repair {self.path}") from exc
        if not isinstance(value, dict):
            raise ConfigurationError("Config root must be a JSON object")
        return AutoMemoryConfig.from_dict(value)

    def save(self, config: AutoMemoryConfig) -> None:
        config = validate_config(config)
        payload = config.to_dict()
        self._reject_secrets(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            backup = self.backups_dir / "config.previous.json"
            shutil.copy2(self.path, backup)
        fd, temporary = tempfile.mkstemp(prefix=".automemory-config-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def _reject_secrets(cls, value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                full = f"{prefix}.{key}" if prefix else str(key)
                if looks_secret_name(str(key)) and str(key) != "credential_name":
                    raise ConfigurationError(f"Secret field cannot be persisted: {full}")
                cls._reject_secrets(item, full)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                cls._reject_secrets(item, f"{prefix}[{index}]")
