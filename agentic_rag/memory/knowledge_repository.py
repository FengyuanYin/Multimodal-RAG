"""SQLite 主知识库：文档、分块、媒体与引用的事务化事实来源。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional
import json
import sqlite3
import time


SCHEMA_VERSION = 1


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _loads(value: Optional[str], fallback: Any) -> Any:
    try:
        return json.loads(value) if value else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


class KnowledgeRepository:
    """单实例 SQLite 仓库，派生索引可从这里重建。"""

    def __init__(self, path: str = "./data/knowledge/knowledge.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'text',
                    source TEXT NOT NULL DEFAULT '',
                    category_id TEXT NOT NULL DEFAULT '',
                    parser TEXT NOT NULL DEFAULT 'local',
                    page_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'ready',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL DEFAULT 1,
                    text TEXT NOT NULL,
                    modality TEXT NOT NULL DEFAULT 'text',
                    tokens_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS media (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    page INTEGER NOT NULL DEFAULT 1,
                    label TEXT NOT NULL DEFAULT '',
                    caption TEXT NOT NULL DEFAULT '',
                    search_text TEXT NOT NULL DEFAULT '',
                    mime_type TEXT NOT NULL DEFAULT '',
                    checksum TEXT NOT NULL DEFAULT '',
                    content BLOB,
                    extraction_method TEXT NOT NULL DEFAULT '',
                    quality TEXT NOT NULL DEFAULT 'derived',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_media_document ON media(document_id);
                CREATE INDEX IF NOT EXISTS idx_media_label ON media(document_id, page, type, label);
                CREATE TABLE IF NOT EXISTS media_references (
                    id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    media_id TEXT REFERENCES media(id) ON DELETE SET NULL,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL DEFAULT 1,
                    label TEXT NOT NULL DEFAULT '',
                    media_type TEXT NOT NULL DEFAULT 'image',
                    offset INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0,
                    resolution TEXT NOT NULL DEFAULT 'unresolved',
                    reason TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_refs_document ON media_references(document_id);
                CREATE INDEX IF NOT EXISTS idx_refs_chunk ON media_references(chunk_id);
                CREATE TABLE IF NOT EXISTS migrations (
                    name TEXT PRIMARY KEY,
                    applied_at REAL NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def upsert_document(
        self,
        document: Any,
        chunks: Iterable[Any],
        media: Iterable[Any],
        references: Iterable[Any],
    ) -> str:
        fingerprint = str(_get(document, "fingerprint", "")).strip()
        if not fingerprint:
            raise ValueError("document fingerprint is required")
        requested_id = str(_get(document, "id", _get(document, "doc_id", ""))).strip()
        if not requested_id:
            raise ValueError("document id is required")
        now = time.time()
        chunk_items, media_items, ref_items = list(chunks), list(media), list(references)

        with self.transaction() as conn:
            existing = conn.execute("SELECT id, created_at FROM documents WHERE fingerprint = ?", (fingerprint,)).fetchone()
            doc_id = existing["id"] if existing else requested_id
            created_at = existing["created_at"] if existing else float(_get(document, "created_at", now) or now)
            conn.execute(
                """INSERT INTO documents
                (id,fingerprint,name,source_type,source,category_id,parser,page_count,status,created_at,updated_at,schema_version,metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  fingerprint=excluded.fingerprint,name=excluded.name,source_type=excluded.source_type,
                  source=excluded.source,category_id=excluded.category_id,parser=excluded.parser,
                  page_count=excluded.page_count,status=excluded.status,updated_at=excluded.updated_at,
                  schema_version=excluded.schema_version,metadata_json=excluded.metadata_json""",
                (
                    doc_id, fingerprint, str(_get(document, "name", _get(document, "title", "未命名文档"))),
                    str(_get(document, "source_type", _get(document, "modality", "text"))),
                    str(_get(document, "source", "")), str(_get(document, "category_id", "")),
                    str(_get(document, "parser", "local")), int(_get(document, "page_count", 1) or 1),
                    str(_get(document, "status", "ready")), created_at, now,
                    int(_get(document, "schema_version", SCHEMA_VERSION) or SCHEMA_VERSION),
                    _json(_get(document, "metadata", _get(document, "raw_metadata", {}))),
                ),
            )
            conn.execute("DELETE FROM media_references WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM media WHERE document_id = ?", (doc_id,))

            for chunk in chunk_items:
                chunk_id = str(_get(chunk, "id", _get(chunk, "chunk_id", "")))
                conn.execute(
                    "INSERT INTO chunks(id,document_id,page,text,modality,tokens_json,metadata_json) VALUES (?,?,?,?,?,?,?)",
                    (
                        chunk_id, doc_id, int(_get(chunk, "page", _get(chunk, "metadata", {}).get("page", 1)) or 1),
                        str(_get(chunk, "text", _get(chunk, "content", ""))), str(_get(chunk, "modality", "text")),
                        _json(_get(chunk, "tokens", [])), _json(_get(chunk, "metadata", {})),
                    ),
                )
            for item in media_items:
                conn.execute(
                    """INSERT INTO media
                    (id,document_id,type,page,label,caption,search_text,mime_type,checksum,content,extraction_method,quality,metadata_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(_get(item, "id", "")), doc_id, str(_get(item, "type", "image")),
                        int(_get(item, "page", 1) or 1), str(_get(item, "label", "")), str(_get(item, "caption", "")),
                        str(_get(item, "search_text", _get(item, "caption", ""))), str(_get(item, "mime_type", "")),
                        str(_get(item, "checksum", "")), _get(item, "content", _get(item, "data", None)),
                        str(_get(item, "extraction_method", "")), str(_get(item, "quality", "derived")),
                        _json(_get(item, "metadata", {})),
                    ),
                )
            chunk_ids = {str(_get(item, "id", _get(item, "chunk_id", ""))) for item in chunk_items}
            media_ids = {str(_get(item, "id", "")) for item in media_items}
            for index, ref in enumerate(ref_items):
                chunk_id = str(_get(ref, "chunk_id", ""))
                if chunk_id not in chunk_ids:
                    raise ValueError(f"reference points to unknown chunk: {chunk_id}")
                media_id = str(_get(ref, "media_id", "")) or None
                if media_id and media_id not in media_ids:
                    media_id = None
                ref_id = str(_get(ref, "id", "")) or f"{chunk_id}_ref_{index:04d}"
                conn.execute(
                    """INSERT INTO media_references
                    (id,chunk_id,media_id,document_id,page,label,media_type,offset,confidence,resolution,reason)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ref_id, chunk_id, media_id, doc_id, int(_get(ref, "page", 1) or 1),
                        str(_get(ref, "label", "")), str(_get(ref, "media_type", "image")),
                        int(_get(ref, "offset", 0) or 0), float(_get(ref, "confidence", 0) or 0),
                        str(_get(ref, "resolution", "unresolved")), str(_get(ref, "reason", "")),
                    ),
                )
        return doc_id

    def _document_dict(self, row: sqlite3.Row) -> dict:
        data = dict(row)
        data["metadata"] = _loads(data.pop("metadata_json", "{}"), {})
        return data

    def get_document(self, document_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._document_dict(row) if row else None

    def list_documents(self, include_unsearchable: bool = False) -> List[dict]:
        sql = "SELECT * FROM documents" if include_unsearchable else "SELECT * FROM documents WHERE status != 'deleting'"
        return [self._document_dict(row) for row in self._conn.execute(sql + " ORDER BY created_at")]

    def list_chunks(self, document_id: Optional[str] = None) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunks" + (" WHERE document_id = ?" if document_id else "") + " ORDER BY id",
            (document_id,) if document_id else (),
        )
        output = []
        for row in rows:
            item = dict(row)
            item["tokens"] = _loads(item.pop("tokens_json", "[]"), [])
            item["metadata"] = _loads(item.pop("metadata_json", "{}"), {})
            output.append(item)
        return output

    def list_media(self, document_id: Optional[str] = None, include_content: bool = True) -> List[dict]:
        columns = "*" if include_content else "id,document_id,type,page,label,caption,search_text,mime_type,checksum,extraction_method,quality,metadata_json"
        rows = self._conn.execute(
            f"SELECT {columns} FROM media" + (" WHERE document_id = ?" if document_id else "") + " ORDER BY id",
            (document_id,) if document_id else (),
        )
        output = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _loads(item.pop("metadata_json", "{}"), {})
            output.append(item)
        return output

    def list_references(self, document_id: Optional[str] = None) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM media_references" + (" WHERE document_id = ?" if document_id else "") + " ORDER BY id",
            (document_id,) if document_id else (),
        )
        return [dict(row) for row in rows]

    def mark_unsearchable(self, document_id: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("UPDATE documents SET status='deleting', updated_at=? WHERE id=?", (time.time(), document_id))
        return cur.rowcount > 0

    def delete_document(self, document_id: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cur.rowcount > 0

    def load_retrieval_corpus(self) -> List[dict]:
        ref_rows = self._conn.execute(
            "SELECT * FROM media_references ORDER BY id"
        )
        refs_by_chunk: Dict[str, List[dict]] = {}
        for ref in ref_rows:
            item = dict(ref)
            refs_by_chunk.setdefault(item["chunk_id"], []).append(item)
        rows = self._conn.execute(
            """SELECT c.* FROM chunks c JOIN documents d ON d.id=c.document_id
               WHERE d.status IN ('ready','degraded') ORDER BY c.id"""
        )
        output = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _loads(item.pop("metadata_json", "{}"), {})
            item["metadata"]["media_refs"] = refs_by_chunk.get(item["id"], [])
            item.pop("tokens_json", None)
            output.append(item)
        media_rows = self._conn.execute(
            """SELECT m.* FROM media m JOIN documents d ON d.id=m.document_id
               WHERE d.status IN ('ready','degraded') AND trim(m.search_text) != '' ORDER BY m.id"""
        )
        for row in media_rows:
            item = dict(row)
            output.append({
                "id": item["id"], "document_id": item["document_id"], "page": item["page"],
                "text": item["search_text"], "modality": item["type"], "metadata": {
                    "media_id": item["id"], "label": item["label"], "caption": item["caption"],
                    "quality": item["quality"], "source": "media",
                },
            })
        return output

    def integrity_check(self) -> str:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "unknown"

    def close(self) -> None:
        with self._lock:
            self._conn.close()
