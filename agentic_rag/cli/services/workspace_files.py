"""ID-only file access inside managed document workspaces."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import uuid

from ..errors import ConfigurationError, UsageError
from ..security import ensure_within, safe_filename


class WorkspaceFileService:
    MAX_WRITE = 2 * 1024 * 1024

    def __init__(self, repository, root: Path, exports_dir: Path, estimator) -> None:
        self.repository, self.root, self.exports_dir, self.estimator = repository, root.resolve(), exports_dir, estimator

    def create_markdown(self, workspace_id: str, kind: str, display_name: str, content: str, purpose: str) -> dict:
        raw = content.encode("utf-8")
        if len(raw) > self.MAX_WRITE:
            raise ConfigurationError("Workspace Markdown exceeds the 2 MiB limit")
        directory = self.root / workspace_id
        directory.mkdir(parents=True, exist_ok=True)
        file_id = f"wsf_{uuid.uuid4().hex}"
        name = safe_filename(display_name, f"{kind}.md")
        if not name.lower().endswith(".md"):
            name += ".md"
        destination = directory / f"{file_id}-{name}"
        fd, temporary = tempfile.mkstemp(prefix=".workspace-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, destination)
            record = {"id": file_id, "workspace_id": workspace_id, "file_kind": kind, "relative_path": destination.relative_to(self.root).as_posix(), "display_name": name, "checksum": hashlib.sha256(raw).hexdigest(), "byte_size": len(raw), "token_estimate": self.estimator.estimate_text(content), "purpose": purpose, "status": "ready", "metadata": {}}
            self.repository.add_file(record)
            return record
        except Exception:
            Path(temporary).unlink(missing_ok=True); destination.unlink(missing_ok=True)
            raise

    def read_text(self, workspace_id: str, file_id: str, start: int = 0, max_chars: int = 48_000) -> dict:
        record = self.repository.get_file(workspace_id, file_id)
        if not record:
            raise UsageError("Workspace file not found or unavailable")
        path = self._path(record)
        text = path.read_text("utf-8")
        start, max_chars = max(0, int(start)), max(1, min(int(max_chars), 48_000))
        return {"file_id": file_id, "start": start, "end": min(len(text), start + max_chars), "total_chars": len(text), "complete": start == 0 and len(text) <= max_chars, "text": text[start:start + max_chars]}

    def _path(self, record: dict) -> Path:
        path = (self.root / record["relative_path"]).resolve()
        try: path.relative_to(self.root)
        except ValueError as exc: raise ConfigurationError("Workspace file path escapes its managed root") from exc
        if not path.is_file() or path.is_symlink(): raise ConfigurationError("Workspace file is missing or unsafe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["checksum"]: raise ConfigurationError("Workspace file integrity check failed")
        return path

    def export_file(self, workspace_id: str, file_id: str, filename: str | None = None) -> Path:
        import shutil
        record = self.repository.get_file(workspace_id, file_id)
        if not record: raise UsageError("Workspace file not found")
        source = self._path(record)
        destination = ensure_within(self.exports_dir, self.exports_dir / safe_filename(filename or record["display_name"]))
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, destination)
        return destination

    def delete_generated_file(self, workspace_id: str, file_id: str) -> None:
        record = self.repository.get_file(workspace_id, file_id)
        if not record: raise UsageError("Generated workspace file not found")
        path = self._path(record)
        self.repository.mark_file_deleted(workspace_id, file_id)
        path.unlink(missing_ok=True)
