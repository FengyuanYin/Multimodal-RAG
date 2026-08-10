"""Typed records shared by AutoMemory services and Textual widgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class Workspace(str, Enum):
    CHAT = "chat"
    KNOWLEDGE = "knowledge"
    EVALUATION = "evaluation"
    SETTINGS = "settings"
    HELP = "help"


class ChatMode(str, Enum):
    DIRECT = "direct"
    RAG = "rag"


class RetrievalMode(str, Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    CANCELLED = "cancelled"
    ERROR = "error"


class MessageStatus(str, Enum):
    STREAMING = "streaming"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class AutoMemoryConfig:
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = ""
    reranker_model: str = ""
    vlm_model: str = ""
    vlm_base_url: str = ""
    retrieval_mode: str = RetrievalMode.KEYWORD.value
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    build_graph: bool = True
    memory_enabled: bool = True
    memory_auto_extract: bool = False
    mineru_mode: str = "official"
    mineru_url: str = ""
    web_provider: str = "duckduckgo"
    theme: str = "dark"

    def to_safe_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "AutoMemoryConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: item for key, item in (value or {}).items() if key in allowed})


@dataclass
class Conversation:
    id: str
    title: str
    created_at: float
    updated_at: float


@dataclass
class MessageRecord:
    id: str
    conversation_id: str
    role: str
    content: str
    chat_mode: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class MemoryRecord:
    id: str
    content: str
    enabled: bool
    created_at: float
    updated_at: float


@dataclass
class CategoryRecord:
    id: str
    name: str
    created_at: float
    updated_at: float


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class CapturedPage:
    title: str
    url: str
    text: str
    content_type: str = "text/html"


@dataclass
class ParsedExternalDocument:
    title: str
    pages: list[dict[str, Any]]
    media: list[dict[str, Any]] = field(default_factory=list)
    parser: str = "external"


@dataclass
class IngestionSource:
    name: str
    source_type: str
    source: str
    collection_id: str
    parser: str
    content: str | bytes | Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatRequest:
    conversation_id: str
    question: str
    mode: str = ChatMode.DIRECT.value
    collection_id: str = "all"


@dataclass
class ChatResult:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticItem:
    name: str
    status: Literal["ok", "degraded", "error"]
    detail: str
