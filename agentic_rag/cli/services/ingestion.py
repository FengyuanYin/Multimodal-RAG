"""Atomic local, Web, VLM, and MinerU knowledge ingestion."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import re
import tempfile
import uuid

from ..cancellation import CancellationToken
from ..errors import CancelledError, ConfigurationError
from ..models import ChunkRecord, DocumentRecord, EventKind, MediaRecord, OutputEvent, ParsedDocument
from ..parsers import parse_local
from ..security import safe_filename


class IngestionService:
    CHUNKER_VERSION = "recursive-v1"

    def __init__(self, knowledge, paths, config, *, embedding_client=None, vlm_client=None, state=None) -> None:
        self.knowledge, self.paths, self.config = knowledge, paths, config
        self.embedding_client, self.vlm_client, self.state = embedding_client, vlm_client, state

    @staticmethod
    def _emit(output, task_id: str, phase: str, text: str, completed: int = 0, total: int = 0) -> None:
        if output:
            output.emit(OutputEvent(EventKind.PROGRESS, text=text, task_id=task_id, phase=phase, completed=completed, total=total))

    def ingest_local(self, paths: list[Path], category: str, output, cancel: CancellationToken, *, use_vlm: bool = False) -> list[dict]:
        results = []
        for index, path in enumerate(paths, 1):
            cancel.checkpoint()
            path = path.expanduser().resolve()
            if not path.is_file():
                raise ConfigurationError(f"File does not exist: {path}")
            task_id = self.state.create_task("ingest", {"source": str(path)}) if self.state else f"task_{uuid.uuid4().hex}"
            self._emit(output, task_id, "parse", f"Parsing {path.name}", index - 1, len(paths))
            try:
                parsed = parse_local(path, cancel)
                if use_vlm and parsed.media:
                    self._describe_media(parsed, cancel)
                result = self._commit(parsed, str(path), path.suffix.lower().lstrip(".") or "file", category, task_id, output, cancel, fingerprint_bytes=path.read_bytes())
                if self.state:
                    self.state.update_task(task_id, "success", "ready", result)
                results.append(result)
            except Exception as exc:
                if self.state:
                    self.state.update_task(task_id, "cancelled" if isinstance(exc, CancelledError) else "error", "failed", {"error": type(exc).__name__})
                raise
        return results

    def ingest_web(self, page, category: str, output, cancel: CancellationToken) -> dict:
        task_id = self.state.create_task("web-ingest", {"source": page.url}) if self.state else f"task_{uuid.uuid4().hex}"
        parsed = ParsedDocument(page.title, [{"page": 1, "text": page.text}], [], "web")
        try:
            result = self._commit(parsed, page.url, "web", category, task_id, output, cancel, fingerprint_bytes=page.text.encode("utf-8"))
            if self.state:
                self.state.update_task(task_id, "success", "ready", result)
            return result
        except Exception as exc:
            if self.state:
                self.state.update_task(task_id, "cancelled" if isinstance(exc, CancelledError) else "error", "failed", {"error": type(exc).__name__})
            raise

    def ingest_parsed(self, parsed: ParsedDocument, source: str, category: str, output, cancel: CancellationToken, source_type: str = "pdf") -> dict:
        task_id = self.state.create_task("cloud-ingest", {"source": source}) if self.state else f"task_{uuid.uuid4().hex}"
        try:
            result = self._commit(parsed, source, source_type, category, task_id, output, cancel, fingerprint_bytes=(source + "\0" + "\n".join(str(page.get("text", "")) for page in parsed.pages)).encode("utf-8"))
            if self.state:
                self.state.update_task(task_id, "success", "ready", result)
            return result
        except Exception as exc:
            if self.state:
                self.state.update_task(task_id, "cancelled" if isinstance(exc, CancelledError) else "error", "failed", {"error": type(exc).__name__})
            raise

    def _describe_media(self, parsed: ParsedDocument, cancel: CancellationToken) -> None:
        if not self.vlm_client:
            raise ConfigurationError("VLM is not configured; omit --vlm or configure the cloud VLM profile")
        for item in parsed.media:
            cancel.checkpoint()
            raw = base64.b64decode(item.get("data") or "", validate=True)
            caption = self.vlm_client.describe_image(raw, str(item.get("mime_type") or "image/png"), "Describe this image for document retrieval. State visible facts only.", cancel)
            item["caption"] = caption
            page = int(item.get("page") or 1)
            for page_item in parsed.pages:
                if int(page_item.get("page") or 1) == page:
                    page_item["text"] = f"{page_item.get('text','')}\n\n[{item.get('label','figure')}: {caption}]".strip()
                    break

    def _commit(self, parsed: ParsedDocument, source: str, source_type: str, category: str, task_id: str, output, cancel: CancellationToken, *, fingerprint_bytes: bytes) -> dict:
        fingerprint = hashlib.sha256(source.encode("utf-8") + b"\0" + fingerprint_bytes).hexdigest()
        existing = self.knowledge.get_by_fingerprint(fingerprint)
        if existing:
            return {"status": "duplicate", "document_id": existing["id"], "message": f"Already imported: {existing['title']}"}
        if not any(item["id"] == category for item in self.knowledge.list_categories()):
            raise ConfigurationError(f"Knowledge category not found: {category}")
        document_id = f"doc_{fingerprint[:20]}"
        self._emit(output, task_id, "chunk", "Creating deterministic chunks")
        chunks = self._chunk(parsed, document_id, source, category, cancel)
        stored_paths: list[Path] = []
        try:
            media = self._store_media(parsed, document_id, stored_paths, cancel)
            embeddings = []
            if self.config.retrieval_mode in {"vector", "hybrid", "multimodal"}:
                if not self.embedding_client:
                    raise ConfigurationError("Cloud embedding is required for the selected retrieval mode")
                self._emit(output, task_id, "embedding", f"Embedding {len(chunks)} chunks in the cloud")
                vectors = self.embedding_client.embeddings([item.text for item in chunks], cancel)
                if len(vectors) != len(chunks):
                    raise ConfigurationError("Embedding result count does not match chunks")
                embeddings = [(item.id, "chunk", self.embedding_client.profile_fingerprint, vector) for item, vector in zip(chunks, vectors)]
            cancel.checkpoint()
            self._emit(output, task_id, "commit", "Committing knowledge transaction")
            document = DocumentRecord(document_id, fingerprint, parsed.title, source, source_type, category, parsed.parser, len(parsed.pages), "ready", {"chunker_version": self.CHUNKER_VERSION})
            self.knowledge.commit_document(document, chunks, media, embeddings)
            return {"status": "success", "document_id": document_id, "chunk_count": len(chunks), "media_count": len(media), "message": f"Imported {parsed.title}"}
        except Exception:
            for path in stored_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def _chunk(self, parsed: ParsedDocument, document_id: str, source: str, category: str, cancel: CancellationToken) -> list[ChunkRecord]:
        chunks, sequence = [], 0
        size, overlap = int(self.config.chunk_size), int(self.config.chunk_overlap)
        for page in parsed.pages:
            cancel.checkpoint()
            text = re.sub(r"\r\n?", "\n", str(page.get("text") or "")).strip()
            cursor = 0
            while cursor < len(text):
                cancel.checkpoint()
                end = min(len(text), cursor + size)
                if end < len(text):
                    boundary = max(text.rfind("\n\n", cursor, end), text.rfind("。", cursor, end), text.rfind(". ", cursor, end))
                    if boundary > cursor + size // 2:
                        end = boundary + 1
                content = text[cursor:end].strip()
                if content:
                    chunks.append(ChunkRecord(f"{document_id}_chunk_{sequence:05d}", document_id, int(page.get("page") or 1), sequence, content, "text", [], {"source": source, "category_id": category, "chunker_version": self.CHUNKER_VERSION}))
                    sequence += 1
                if end >= len(text):
                    break
                cursor = max(cursor + 1, end - overlap)
        if not chunks:
            raise ConfigurationError("Parser produced no readable text chunks")
        return chunks

    def _store_media(self, parsed: ParsedDocument, document_id: str, stored_paths: list[Path], cancel: CancellationToken) -> list[MediaRecord]:
        output = []
        document_dir = self.paths.media_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(parsed.media, 1):
            cancel.checkpoint()
            try:
                raw = base64.b64decode(item.get("data") or "", validate=True)
            except Exception as exc:
                raise ConfigurationError("Parsed media contains invalid base64") from exc
            checksum = str(item.get("checksum") or hashlib.sha256(raw).hexdigest())
            extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(str(item.get("mime_type") or ""), ".bin")
            filename = safe_filename(f"{item.get('id') or f'media_{index}'}{extension}")
            destination = document_dir / filename
            fd, temporary = tempfile.mkstemp(prefix=".automemory-media-", dir=document_dir)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            stored_paths.append(destination)
            media_id = f"{document_id}_{item.get('id') or f'media_{index}'}"
            output.append(MediaRecord(media_id, document_id, int(item.get("page") or 1), str(item.get("type") or "image"), str(item.get("label") or f"media{index}"), str(item.get("caption") or ""), str(item.get("mime_type") or "application/octet-stream"), checksum, str(destination), str(item.get("quality") or "derived"), {}))
        return output
