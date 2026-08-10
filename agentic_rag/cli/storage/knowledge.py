"""Transactional knowledge, media metadata, and cloud vector persistence."""

from __future__ import annotations

from array import array
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time
from typing import Any, Iterator

from ..errors import UsageError
from ..models import ChunkRecord, DocumentRecord, MediaRecord
from .migrations import connect_database, migrate


SCHEMA_VERSION = 1
MIGRATIONS = {
    1: """
    CREATE TABLE categories(id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE TABLE documents(
      id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL UNIQUE, title TEXT NOT NULL, source TEXT NOT NULL,
      source_type TEXT NOT NULL, category_id TEXT NOT NULL REFERENCES categories(id), parser TEXT NOT NULL,
      page_count INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('staging','ready','error')),
      metadata_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL, updated_at REAL NOT NULL
    );
    CREATE INDEX idx_documents_category ON documents(category_id,status);
    CREATE TABLE chunks(
      id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      page INTEGER NOT NULL, sequence INTEGER NOT NULL, text TEXT NOT NULL, modality TEXT NOT NULL,
      media_refs_json TEXT NOT NULL DEFAULT '[]', metadata_json TEXT NOT NULL DEFAULT '{}',
      UNIQUE(document_id,sequence)
    );
    CREATE INDEX idx_chunks_document ON chunks(document_id,sequence);
    CREATE TABLE media(
      id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      page INTEGER NOT NULL, media_type TEXT NOT NULL, label TEXT NOT NULL, caption TEXT NOT NULL,
      mime_type TEXT NOT NULL, checksum TEXT NOT NULL, storage_path TEXT NOT NULL, quality TEXT NOT NULL,
      metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE embeddings(
      target_id TEXT NOT NULL, target_type TEXT NOT NULL CHECK(target_type IN ('chunk','media')),
      profile_fingerprint TEXT NOT NULL, dimensions INTEGER NOT NULL, vector_blob BLOB NOT NULL,
      created_at REAL NOT NULL, PRIMARY KEY(target_id,target_type,profile_fingerprint)
    );
    """,
}


class KnowledgeRepository:
    def __init__(self, path: Path, backups_dir: Path, *, now=time.time) -> None:
        self.path, self._now = path, now
        self._lock = RLock()
        self._conn = connect_database(path)
        migrate(self._conn, path, backups_dir, SCHEMA_VERSION, MIGRATIONS)
        now_value = self._now()
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO categories VALUES ('default','Default',?,?)", (now_value, now_value))
            db.execute("UPDATE documents SET status='error',updated_at=? WHERE status='staging'", (now_value,))

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

    @staticmethod
    def _loads(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return fallback

    def create_category(self, name: str) -> dict[str, Any]:
        import uuid
        name = name.strip()
        if not name:
            raise UsageError("Category name is required")
        category_id, now = f"cat_{uuid.uuid4().hex}", self._now()
        with self.transaction() as db:
            db.execute("INSERT INTO categories VALUES (?,?,?,?)", (category_id, name, now, now))
        return {"id": category_id, "name": name}

    def list_categories(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM categories ORDER BY lower(name),id")]

    def rename_category(self, category_id: str, name: str) -> None:
        with self.transaction() as db:
            if not db.execute("UPDATE categories SET name=?,updated_at=? WHERE id=?", (name.strip(), self._now(), category_id)).rowcount:
                raise UsageError(f"Category not found: {category_id}")

    def delete_category(self, category_id: str) -> None:
        if category_id == "default":
            raise UsageError("The default category cannot be deleted")
        if self._conn.execute("SELECT 1 FROM documents WHERE category_id=? LIMIT 1", (category_id,)).fetchone():
            raise UsageError("Category contains documents; remove them first")
        with self.transaction() as db:
            if not db.execute("DELETE FROM categories WHERE id=?", (category_id,)).rowcount:
                raise UsageError(f"Category not found: {category_id}")

    def get_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM documents WHERE fingerprint=?", (fingerprint,)).fetchone()
        return self._document_row(row) if row else None

    def commit_document(self, document: DocumentRecord, chunks: list[ChunkRecord], media: list[MediaRecord], embeddings: list[tuple[str, str, str, list[float]]] | None = None) -> None:
        now = self._now()
        with self.transaction() as db:
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (document.id, document.fingerprint, document.title, document.source, document.source_type, document.category_id, document.parser, document.page_count, "staging", json.dumps(document.metadata, ensure_ascii=False), now, now),
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                [(item.id, item.document_id, item.page, item.sequence, item.text, item.modality, json.dumps(item.media_refs, ensure_ascii=False), json.dumps(item.metadata, ensure_ascii=False)) for item in chunks],
            )
            db.executemany(
                "INSERT INTO media VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(item.id, item.document_id, item.page, item.media_type, item.label, item.caption, item.mime_type, item.checksum, item.storage_path, item.quality, json.dumps(item.metadata, ensure_ascii=False)) for item in media],
            )
            for target_id, target_type, fingerprint, vector in embeddings or []:
                dimensions, blob = self.encode_vector(vector)
                db.execute("INSERT INTO embeddings VALUES (?,?,?,?,?,?)", (target_id, target_type, fingerprint, dimensions, blob, now))
            db.execute("UPDATE documents SET status='ready',updated_at=? WHERE id=?", (now, document.id))

    def mark_error(self, document_id: str) -> None:
        with self.transaction() as db:
            db.execute("UPDATE documents SET status='error',updated_at=? WHERE id=?", (self._now(), document_id))

    def list_documents(self, category_id: str = "all", include_error: bool = True) -> list[dict[str, Any]]:
        clauses, params = [], []
        if category_id != "all":
            clauses.append("d.category_id=?")
            params.append(category_id)
        if not include_error:
            clauses.append("d.status='ready'")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            "SELECT d.*,c.name category_name,(SELECT count(*) FROM chunks x WHERE x.document_id=d.id) chunk_count,(SELECT count(*) FROM media m WHERE m.document_id=d.id) media_count FROM documents d JOIN categories c ON c.id=d.category_id" + where + " ORDER BY d.updated_at DESC,d.id",
            params,
        )
        return [self._document_row(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            return None
        return {**self._document_row(row), "chunks": self.list_chunks(document_id), "media": self.list_media(document_id)}

    def list_chunks(self, document_id: str | None = None, category_id: str = "all") -> list[dict[str, Any]]:
        query = "SELECT x.*,d.title document,d.category_id FROM chunks x JOIN documents d ON d.id=x.document_id WHERE d.status='ready'"
        params: list[Any] = []
        if document_id:
            query += " AND x.document_id=?"
            params.append(document_id)
        if category_id != "all":
            query += " AND d.category_id=?"
            params.append(category_id)
        query += " ORDER BY x.document_id,x.sequence"
        output = []
        for row in self._conn.execute(query, params):
            item = dict(row)
            item["media_refs"] = self._loads(item.pop("media_refs_json"), [])
            item["metadata"] = self._loads(item.pop("metadata_json"), {})
            output.append(item)
        return output

    def list_media(self, document_id: str | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM media", []
        if document_id:
            query += " WHERE document_id=?"
            params.append(document_id)
        output = []
        for row in self._conn.execute(query + " ORDER BY document_id,page,id", params):
            item = dict(row)
            item["metadata"] = self._loads(item.pop("metadata_json"), {})
            output.append(item)
        return output

    def delete_document(self, document_id: str) -> None:
        with self.transaction() as db:
            if not db.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount:
                raise UsageError(f"Document not found: {document_id}")

    def list_embeddings(self, profile_fingerprint: str, category_id: str = "all") -> list[dict[str, Any]]:
        query = "SELECT e.*,x.document_id,x.text,x.page,x.modality,d.title document,d.category_id,x.media_refs_json FROM embeddings e JOIN chunks x ON x.id=e.target_id JOIN documents d ON d.id=x.document_id WHERE e.target_type='chunk' AND e.profile_fingerprint=? AND d.status='ready'"
        params: list[Any] = [profile_fingerprint]
        if category_id != "all":
            query += " AND d.category_id=?"
            params.append(category_id)
        output = []
        for row in self._conn.execute(query, params):
            item = dict(row)
            item["vector"] = self.decode_vector(item.pop("vector_blob"), item["dimensions"])
            item["media_refs"] = self._loads(item.pop("media_refs_json"), [])
            output.append(item)
        return output

    @staticmethod
    def encode_vector(vector: list[float]) -> tuple[int, bytes]:
        if not vector:
            raise UsageError("Embedding vector is empty")
        values = array("f", (float(value) for value in vector))
        norm = sum(value * value for value in values) ** 0.5
        if not norm:
            raise UsageError("Embedding vector has zero norm")
        normalized = array("f", (value / norm for value in values))
        return len(normalized), normalized.tobytes()

    @staticmethod
    def decode_vector(blob: bytes, dimensions: int) -> list[float]:
        values = array("f")
        values.frombytes(blob)
        if len(values) != dimensions:
            raise UsageError("Stored embedding dimension is invalid")
        return list(values)

    def _document_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        if "metadata_json" in item:
            item["metadata"] = self._loads(item.pop("metadata_json"), {})
        return item

    def integrity_check(self) -> str:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "unknown"

    def close(self) -> None:
        with self._lock:
            self._conn.close()
