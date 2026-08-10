"""Local and cloud configuration diagnostics with redacted error history."""

from __future__ import annotations

from collections import deque
import importlib.util

from ..models import DiagnosticItem
from ..security import redact


class DiagnosticsService:
    def __init__(self, paths, state, knowledge, credentials, config) -> None:
        self.paths, self.state, self.knowledge = paths, state, knowledge
        self.credentials, self.config = credentials, config
        self._errors: deque[str] = deque(maxlen=50)

    def record_error(self, error: object) -> None:
        self._errors.append(redact(error, self.credentials.redaction_values()))

    def report(self) -> list[DiagnosticItem]:
        items = [
            DiagnosticItem("state database", "ok" if self.state.integrity_check() == "ok" else "error", str(self.paths.state_db)),
            DiagnosticItem("knowledge database", "ok" if self.knowledge.integrity_check() == "ok" else "error", str(self.paths.knowledge_db)),
            DiagnosticItem("data directory", "ok" if self.paths.root.is_dir() else "error", str(self.paths.root)),
        ]
        for service, name in (("LLM", "llm_api_key"), ("Embedding", "embedding_api_key"), ("VLM", "vlm_api_key"), ("Reranker", "reranker_api_key"), ("MinerU", "mineru_api_key"), ("Tavily", "tavily_api_key")):
            source = self.credentials.source(name)
            items.append(DiagnosticItem(service, "ok" if source != "not-configured" else "degraded", source))
        for package in ("prompt_toolkit", "httpx", "fitz", "openpyxl", "bs4"):
            present = importlib.util.find_spec(package) is not None
            items.append(DiagnosticItem(f"dependency:{package}", "ok" if present else "error", "available" if present else "missing"))
        return items

    def recent_errors(self) -> list[str]:
        return list(self._errors)
