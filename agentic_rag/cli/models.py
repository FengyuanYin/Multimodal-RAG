"""Typed records shared by the AutoMemory CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal


class InputKind(str, Enum):
    EMPTY = "empty"
    COMMAND = "command"
    DIRECT_CHAT = "direct_chat"
    RAG_CHAT = "rag_chat"


class EventKind(str, Enum):
    DELTA = "delta"
    PROGRESS = "progress"
    RESULT = "result"
    WARNING = "warning"
    ERROR = "error"
    SOURCES = "sources"


@dataclass(slots=True)
class CLIOptions:
    prompt: str | None = None
    home: Path | None = None
    no_color: bool = False
    debug: bool = False
    version: bool = False
    force_plain: bool = False


@dataclass(slots=True)
class ParsedInput:
    kind: InputKind
    raw_text: str
    name: str = ""
    arguments: list[str] = field(default_factory=list)
    question: str = ""


@dataclass(slots=True)
class CommandSpec:
    name: str
    summary: str
    usage: str
    handler: Callable[..., "CommandResult"]
    aliases: tuple[str, ...] = ()
    group: str = "General"
    primary: bool = False


@dataclass(frozen=True, slots=True)
class RagPreset:
    name: str
    description: str
    channels: frozenset[str]
    rewrite_queries: bool = False
    rerank: bool = False
    top_k: int = 5
    candidate_k: int = 25
    chunk_size: int = 800
    chunk_overlap: int = 120
    window_before: int = 0
    window_after: int = 0
    graph_depth: int = 0
    rewrite_limit: int = 1


@dataclass(slots=True)
class OutputEvent:
    kind: EventKind
    text: str = ""
    task_id: str = ""
    phase: str = ""
    completed: int = 0
    total: int = 0
    code: str = ""
    data: Any = None


@dataclass(slots=True)
class CommandResult:
    ok: bool = True
    text: str = ""
    data: Any = None
    exit_requested: bool = False


@dataclass(slots=True)
class ServiceProfile:
    base_url: str
    model: str
    credential_name: str
    timeout_seconds: float = 60.0
    retries: int = 2
    batch_size: int = 32

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: "ServiceProfile") -> "ServiceProfile":
        allowed = cls.__dataclass_fields__.keys()
        merged = default.to_dict()
        merged.update({key: item for key, item in (value or {}).items() if key in allowed})
        return cls(**merged)


@dataclass(slots=True)
class DocumentRecord:
    id: str
    fingerprint: str
    title: str
    source: str
    source_type: str
    category_id: str
    parser: str
    page_count: int
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(slots=True)
class ChunkRecord:
    id: str
    document_id: str
    page: int
    sequence: int
    text: str
    modality: str = "text"
    media_refs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MediaRecord:
    id: str
    document_id: str
    page: int
    media_type: str
    label: str
    caption: str = ""
    mime_type: str = "application/octet-stream"
    checksum: str = ""
    storage_path: str = ""
    quality: str = "derived"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    target_id: str
    document_id: str
    document: str
    text: str
    page: int
    modality: str
    score: float
    channel_scores: dict[str, float] = field(default_factory=dict)
    media_refs: list[dict[str, Any]] = field(default_factory=list)
    window_before: list[dict[str, Any]] = field(default_factory=list)
    window_after: list[dict[str, Any]] = field(default_factory=list)
    graph_paths: list[dict[str, Any]] = field(default_factory=list)
    visual_evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    trace: dict[str, Any]


@dataclass(slots=True)
class GraphNodeRecord:
    id: str
    knowledge_base_id: str
    graph_kind: str
    node_type: str
    label: str
    document_id: str | None = None
    page: int | None = None
    evidence_chunk_id: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdgeRecord:
    id: str
    knowledge_base_id: str
    graph_kind: str
    source_id: str
    target_id: str
    relation_type: str
    document_id: str | None = None
    evidence_chunk_id: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphExportResult:
    png_path: Path
    metadata_path: Path
    graph_kind: str
    original_nodes: int
    original_edges: int
    exported_nodes: int
    exported_edges: int
    truncated: bool = False


@dataclass(slots=True)
class ParsedDocument:
    title: str
    pages: list[dict[str, Any]]
    media: list[dict[str, Any]] = field(default_factory=list)
    parser: str = "local"
    markdown: str = ""
    markdown_source: Literal["mineru_original", "generated", "none"] = "none"
    markdown_media_refs: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DocumentArtifactRecord:
    id: str
    document_id: str
    artifact_type: str
    relative_path: str
    mime_type: str
    source: str
    checksum: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class ContextBudget:
    context_limit: int
    hard_input_limit: int
    fixed_prefix_tokens: int
    summary_tokens: int = 0
    recent_history_tokens: int = 0
    tool_tokens: int = 0
    question_tokens: int = 0
    output_reserve_tokens: int = 0
    safety_reserve_tokens: int = 0
    estimator: str = "conservative-v1"
    requires_compaction: bool = False


@dataclass(slots=True)
class ModelStreamEvent:
    kind: Literal["text_delta", "tool_call", "usage"]
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass(slots=True)
class CapturedPage:
    title: str
    url: str
    text: str
    content_type: str = "text/html"


@dataclass(slots=True)
class DiagnosticItem:
    name: str
    status: Literal["ok", "degraded", "error"]
    detail: str


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    id: str
    service: str
    label: str
    protocol: str
    base_url: str = ""
    default_model: str = ""
    credential_name: str = ""
    requires_secret: bool = True


@dataclass(slots=True)
class SetupDraft:
    config: Any
    secrets: dict[str, str] = field(default_factory=dict)
    changed_services: set[str] = field(default_factory=set)
    test_after_save: bool = True


@dataclass(frozen=True, slots=True)
class ProbeResult:
    service: str
    status: Literal[
        "success", "auth_error", "rate_limited", "network_error",
        "model_error", "response_error", "reachable_unverified", "skipped",
    ]
    code: str
    message: str
    latency_ms: float | None = None
