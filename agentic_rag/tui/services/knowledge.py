"""Knowledge-base operations backed by the existing project repositories."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterable

from ..events import CancelToken, EventCallback, JobProgress
from ..models import CapturedPage, IngestionSource, ParsedExternalDocument
from ..security import ensure_within, safe_filename


class KnowledgeService:
    def __init__(self, runtime, state) -> None:
        self.runtime = runtime
        self.state = state

    def list_documents(self, collection_id: str | None = None) -> list[dict]:
        documents = self.runtime.repository.list_documents(include_unsearchable=True) if self.runtime.repository else []
        if collection_id and collection_id != "all":
            documents = [item for item in documents if item.get("category_id") == collection_id]
        for item in documents:
            item["chunk_count"] = len(self.runtime.repository.list_chunks(item["id"]))
            item["media_count"] = len(self.runtime.repository.list_media(item["id"], include_content=False))
        return documents

    def document_detail(self, document_id: str) -> dict | None:
        document = self.runtime.repository.get_document(document_id)
        if not document:
            return None
        return {**document, "chunks": self.runtime.repository.list_chunks(document_id), "media": self.runtime.repository.list_media(document_id, include_content=False), "references": self.runtime.repository.list_references(document_id)}

    def ingest_local(self, paths: Iterable[Path], collection_id: str = "default", emit: EventCallback | None = None, cancel: CancelToken | None = None, job_id: str = "ingest") -> dict:
        from agentic_rag.service import ingest_documents

        cancel = cancel or CancelToken()
        sources = []
        paths = list(paths)
        for index, path in enumerate(paths, 1):
            cancel.checkpoint()
            path = path.expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"file does not exist: {path}")
            if emit:
                emit(JobProgress(job_id, "ingest", "prepare", path.name, index - 1, len(paths)))
            suffix = path.suffix.lower()
            modality = "pdf" if suffix == ".pdf" else "image" if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"} else "table" if suffix in {".csv", ".tsv", ".xlsx", ".xls"} else "text"
            if modality == "pdf":
                content = str(path)
            elif modality == "image":
                content = path.read_bytes()
            elif suffix in {".csv", ".tsv"}:
                content = path.read_text("utf-8-sig", errors="replace")
            elif modality == "table":
                try:
                    import pandas as pd
                except ImportError as exc:
                    raise RuntimeError("Excel ingestion requires the table extra: pip install -e '.[table]'") from exc
                content = pd.read_excel(path).to_csv(index=False)
            else:
                content = path.read_text("utf-8", errors="replace")
            sources.append({"content": content, "modality": modality, "collection": collection_id, "metadata": {"source": str(path), "title": path.name, "parser": "local", "category_id": collection_id}})
        cancel.checkpoint()
        if emit:
            emit(JobProgress(job_id, "ingest", "parse-index", "Parsing and indexing local documents", 0, len(paths)))
        result = ingest_documents(self.runtime.orchestrator, sources, chunk_size=self.runtime.config.chunk_size, chunk_overlap=self.runtime.config.chunk_overlap, build_graph=self.runtime.config.build_graph)
        return result

    def ingest_page(self, page: CapturedPage, collection_id: str = "default", cancel: CancelToken | None = None) -> dict:
        external = ParsedExternalDocument(page.title, [{"page": 1, "text": page.text}], [], "web")
        return self.ingest_external(external, page.url, collection_id, cancel=cancel)

    def ingest_external(self, document: ParsedExternalDocument, source: str, collection_id: str = "default", emit: EventCallback | None = None, cancel: CancelToken | None = None, job_id: str = "ingest") -> dict:
        from agentic_rag.memory.multi_modal_parser import DocumentChunk, MediaAsset, detect_media_refs
        from agentic_rag.processing.chunker import get_chunker

        cancel = cancel or CancelToken()
        fingerprint = hashlib.sha256((source + "\0" + "\n".join(str(page.get("text", "")) for page in document.pages)).encode("utf-8")).hexdigest()
        doc_id = f"doc_{fingerprint[:20]}"
        existing = next((item for item in self.runtime.repository.list_documents(include_unsearchable=True) if item.get("fingerprint") == fingerprint), None)
        if existing:
            return {"status": "duplicate", "document_id": existing["id"], "message": "Document already exists"}
        chunker = get_chunker("recursive", chunk_size=self.runtime.config.chunk_size, chunk_overlap=self.runtime.config.chunk_overlap)
        media_assets = []
        for index, item in enumerate(document.media, 1):
            data = item.get("data") or ""
            media_assets.append(MediaAsset(
                id=f"{doc_id}_{item.get('id') or f'media_{index}'}", doc_id=doc_id,
                type=str(item.get("type") or "image"), page=int(item.get("page") or 1),
                label=str(item.get("label") or f"media{index}"), caption=str(item.get("caption") or ""),
                data=data, search_text=str(item.get("search_text") or item.get("caption") or ""),
                mime_type=str(item.get("mime_type") or ""), checksum=str(item.get("checksum") or (hashlib.sha256(data.encode()).hexdigest() if data else "")),
                extraction_method=str(item.get("extraction_method") or document.parser), quality=str(item.get("quality") or "exact"), metadata={"source": source},
            ))
        media_index = {item.label: item for item in media_assets}
        chunks, references, sequence = [], [], 0
        for page in document.pages:
            cancel.checkpoint()
            page_no = int(page.get("page") or 1)
            for chunk in chunker.chunk(str(page.get("text") or ""), doc_id=doc_id, metadata={"page": page_no, "source": source, "category_id": collection_id, "parser": document.parser}):
                chunk.chunk_id = f"{doc_id}_chunk_{sequence:04d}"
                sequence += 1
                refs = detect_media_refs(chunk.content, doc_id, page_no, media_index)
                chunk.media_refs = refs
                chunk.metadata["media_refs"] = [vars(ref) for ref in refs]
                chunks.append(chunk)
                references.extend({**vars(ref), "chunk_id": chunk.chunk_id} for ref in refs)
        cancel.checkpoint()
        if emit:
            emit(JobProgress(job_id, "ingest", "commit", "Writing knowledge transaction"))
        self.runtime.repository.upsert_document({
            "id": doc_id, "fingerprint": fingerprint, "name": document.title,
            "source_type": "web" if document.parser == "web" else "pdf", "source": source,
            "category_id": collection_id, "parser": document.parser,
            "page_count": len(document.pages), "status": "ready", "metadata": {"source": source, "parser": document.parser},
        }, chunks, media_assets, references)
        if media_assets and self.runtime.orchestrator.media_store:
            self.runtime.orchestrator.media_store.add_many(media_assets)
        indexed = self.runtime.retriever.rebuild_from_repository()
        return {"status": "success", "document_id": doc_id, "chunk_count": len(chunks), "media_count": len(media_assets), "indexed_items": indexed, "message": f"Ingested {document.title}"}

    def delete_document(self, document_id: str) -> dict:
        from agentic_rag.service import delete_document
        return delete_document(self.runtime.orchestrator, document_id)

    def rebuild_indexes(self) -> dict:
        from agentic_rag.service import rebuild_indexes
        return rebuild_indexes(self.runtime.orchestrator)

    def export_media(self, media_id: str, exports_root: Path, filename: str = "") -> Path:
        media = next((item for item in self.runtime.repository.list_media(include_content=True) if item["id"] == media_id), None)
        if not media:
            raise KeyError(media_id)
        suffix = mimetype_suffix(media.get("mime_type", ""), media.get("type", ""))
        destination = ensure_within(exports_root, exports_root / safe_filename(filename or f"{media_id}{suffix}"))
        content = media.get("content") or b""
        if isinstance(content, memoryview):
            content = content.tobytes()
        if isinstance(content, str):
            try:
                content = base64.b64decode(content)
            except ValueError:
                content = content.encode("utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".automemory-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination


def mimetype_suffix(mime: str, kind: str) -> str:
    return {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "text/csv": ".csv"}.get(mime, ".txt" if kind == "table" else ".bin")
