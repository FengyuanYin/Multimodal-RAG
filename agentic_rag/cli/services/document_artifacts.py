"""Atomic persistence and verification of full document Markdown."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from ..errors import ConfigurationError
from ..models import DocumentArtifactRecord, ParsedDocument


class DocumentArtifactService:
    def __init__(self, knowledge, root: Path) -> None:
        self.knowledge, self.root = knowledge, root.resolve()

    def save_markdown(self, document_id: str, parsed: ParsedDocument) -> DocumentArtifactRecord | None:
        if not parsed.markdown.strip() or parsed.markdown_source == "none":
            return None
        directory = self.root / document_id
        directory.mkdir(parents=True, exist_ok=True)
        content = parsed.markdown.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        checksum = hashlib.sha256(content).hexdigest()
        relative = Path(document_id) / f"source-{checksum[:16]}.md"
        destination = self.root / relative
        fd, temporary = tempfile.mkstemp(prefix=".markdown-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            manifest = destination.with_suffix(".json")
            manifest.write_text(json.dumps({"media_refs": parsed.markdown_media_refs}, ensure_ascii=False, indent=2), encoding="utf-8")
            record = DocumentArtifactRecord(f"artifact_{uuid.uuid4().hex}", document_id, "source_markdown", relative.as_posix(), "text/markdown", parsed.markdown_source, checksum, len(content))
            self.knowledge.upsert_document_artifact(record)
            return record
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            destination.with_suffix(".json").unlink(missing_ok=True)
            raise

    def verify(self, artifact: dict) -> Path:
        path = (self.root / str(artifact["relative_path"])).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ConfigurationError("Document Markdown path escapes the managed asset directory") from exc
        if not path.is_file() or path.is_symlink():
            raise ConfigurationError("Document Markdown file is missing or unsafe")
        content = path.read_bytes()
        if len(content) != int(artifact["byte_size"]) or hashlib.sha256(content).hexdigest() != artifact["checksum"]:
            raise ConfigurationError("Document Markdown integrity check failed")
        return path

    def remove_artifact_files(self, artifact: dict) -> None:
        path = self.verify(artifact)
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
