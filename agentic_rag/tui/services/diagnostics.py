"""Sanitized AutoMemory health and dependency diagnostics."""

from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path

from ..models import DiagnosticItem
from ..security import redact


class DiagnosticsService:
    OPTIONAL_PACKAGES = ("textual", "fitz", "bs4", "chromadb", "sentence_transformers", "torch", "pytesseract", "pandas")

    def __init__(self, paths, state, runtime, secrets) -> None:
        self.paths, self.state, self.runtime, self.secrets = paths, state, runtime, secrets
        self._errors: deque[str] = deque(maxlen=50)

    def record_error(self, error: object) -> None:
        self._errors.append(redact(error, self.secrets.values_for_redaction()))

    def report(self) -> list[DiagnosticItem]:
        items = list(self.runtime.health().items)
        items.append(DiagnosticItem("state database", "ok" if self.state.integrity_check() == "ok" else "error", str(self.paths.state_db)))
        for name in self.OPTIONAL_PACKAGES:
            available = importlib.util.find_spec(name) is not None
            items.append(DiagnosticItem(f"dependency:{name}", "ok" if available else "degraded", "available" if available else "not installed"))
        for label, path in (("data", self.paths.root), ("exports", self.paths.exports_dir), ("logs", self.paths.logs_dir)):
            writable = path.is_dir() and path.exists()
            items.append(DiagnosticItem(f"path:{label}", "ok" if writable else "error", str(path)))
        return items

    def recent_errors(self) -> list[str]:
        return list(self._errors)
