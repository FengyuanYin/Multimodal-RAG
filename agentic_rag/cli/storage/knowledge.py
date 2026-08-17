"""Transactional knowledge, media metadata, and cloud vector persistence."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time
from typing import Any, Iterator

from ..errors import UsageError
from ..models import ChunkRecord, DocumentArtifactRecord, DocumentRecord, GraphEdgeRecord, GraphNodeRecord, MediaRecord
from .migrations import connect_database, migrate


SCHEMA_VERSION = 5
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
    2: """
    CREATE TABLE graph_nodes(
      id TEXT NOT NULL, knowledge_base_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
      graph_kind TEXT NOT NULL CHECK(graph_kind IN ('entity','reference')), node_type TEXT NOT NULL,
      label TEXT NOT NULL, document_id TEXT REFERENCES documents(id) ON DELETE CASCADE, page INTEGER,
      evidence_chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE, properties_json TEXT NOT NULL DEFAULT '{}',
      PRIMARY KEY(id,graph_kind)
    );
    CREATE INDEX idx_graph_nodes_scope ON graph_nodes(knowledge_base_id,graph_kind,node_type);
    CREATE TABLE graph_edges(
      id TEXT PRIMARY KEY, knowledge_base_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
      graph_kind TEXT NOT NULL CHECK(graph_kind IN ('entity','reference')),
      source_id TEXT NOT NULL, target_id TEXT NOT NULL, relation_type TEXT NOT NULL,
      document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
      evidence_chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
      properties_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX idx_graph_edges_scope ON graph_edges(knowledge_base_id,graph_kind,source_id,target_id);
    CREATE TABLE derived_index_states(
      document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, index_kind TEXT NOT NULL,
      profile_fingerprint TEXT NOT NULL DEFAULT '', version TEXT NOT NULL, status TEXT NOT NULL,
      error_code TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL,
      PRIMARY KEY(document_id,index_kind,profile_fingerprint)
    );
    """,
    3: """
    CREATE TABLE document_artifacts(
      id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      artifact_type TEXT NOT NULL CHECK(artifact_type IN ('source_markdown')),
      relative_path TEXT NOT NULL, mime_type TEXT NOT NULL,
      source TEXT NOT NULL CHECK(source IN ('mineru_original','generated')),
      checksum TEXT NOT NULL, byte_size INTEGER NOT NULL, created_at REAL NOT NULL,
      UNIQUE(document_id,artifact_type)
    );
    CREATE INDEX idx_document_artifacts_document ON document_artifacts(document_id,artifact_type);
    """,
    4: """
    DROP TABLE IF EXISTS embeddings;
    DELETE FROM derived_index_states WHERE index_kind='embedding';
    """,
    5: """
    CREATE TABLE media_vlm_analyses(
      media_id TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
      media_checksum TEXT NOT NULL,
      profile_fingerprint TEXT NOT NULL,
      prompt_version TEXT NOT NULL,
      analysis_json TEXT NOT NULL,
      created_at REAL NOT NULL,
      PRIMARY KEY(media_id,profile_fingerprint,prompt_version)
    );
    CREATE INDEX idx_media_vlm_analyses_checksum ON media_vlm_analyses(media_checksum);
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
        name = self._validate_knowledge_base_name(name)
        category_id, now = f"cat_{uuid.uuid4().hex}", self._now()
        try:
            with self.transaction() as db:
                db.execute("INSERT INTO categories VALUES (?,?,?,?)", (category_id, name, now, now))
        except sqlite3.IntegrityError as exc:
            raise UsageError(f"Knowledge base name already exists: {name}") from exc
        return {"id": category_id, "name": name}

    create_knowledge_base = create_category

    def list_categories(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM categories ORDER BY lower(name),id")]

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT c.*,(SELECT count(*) FROM documents d WHERE d.category_id=c.id) document_count "
            "FROM categories c ORDER BY lower(c.name),c.id"
        )
        return [dict(row) for row in rows]

    def resolve_knowledge_base(self, value: str) -> str:
        value = value.strip()
        rows = self.list_knowledge_bases()
        exact = [item["id"] for item in rows if item["id"] == value or item["name"].casefold() == value.casefold()]
        if len(exact) == 1:
            return exact[0]
        matches = [item["id"] for item in rows if item["id"].startswith(value)]
        if len(matches) == 1:
            return matches[0]
        if not exact and not matches:
            raise UsageError(f"Knowledge base not found: {value}")
        raise UsageError(f"Knowledge base is ambiguous: {value}")

    def rename_category(self, category_id: str, name: str) -> None:
        name = self._validate_knowledge_base_name(name)
        try:
            with self.transaction() as db:
                if not db.execute("UPDATE categories SET name=?,updated_at=? WHERE id=?", (name, self._now(), category_id)).rowcount:
                    raise UsageError(f"Category not found: {category_id}")
        except sqlite3.IntegrityError as exc:
            raise UsageError(f"Knowledge base name already exists: {name}") from exc

    rename_knowledge_base = rename_category

    def delete_category(self, category_id: str) -> None:
        if category_id == "default":
            raise UsageError("The default category cannot be deleted")
        if self._conn.execute("SELECT 1 FROM documents WHERE category_id=? LIMIT 1", (category_id,)).fetchone():
            raise UsageError("Category contains documents; remove them first")
        with self.transaction() as db:
            if not db.execute("DELETE FROM categories WHERE id=?", (category_id,)).rowcount:
                raise UsageError(f"Category not found: {category_id}")

    def delete_knowledge_base(self, category_id: str, *, force: bool = False) -> list[str]:
        if category_id == "default":
            raise UsageError("The default knowledge base cannot be deleted")
        documents = self.list_documents(category_id)
        if documents and not force:
            raise UsageError("Knowledge base contains documents; confirm deletion first")
        media_paths = [item["storage_path"] for doc in documents for item in self.list_media(doc["id"])]
        artifact_paths = [item["relative_path"] for doc in documents for item in self.list_document_artifacts(doc["id"])]
        with self.transaction() as db:
            db.execute("DELETE FROM documents WHERE category_id=?", (category_id,))
            if not db.execute("DELETE FROM categories WHERE id=?", (category_id,)).rowcount:
                raise UsageError(f"Knowledge base not found: {category_id}")
        return media_paths + ["artifact:" + item for item in artifact_paths]

    @staticmethod
    def _validate_knowledge_base_name(name: str) -> str:
        name = name.strip()
        if not name or len(name) > 80 or any(ord(char) < 32 for char in name):
            raise UsageError("Knowledge base name must be 1-80 visible characters")
        return name

    def get_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM documents WHERE fingerprint=?", (fingerprint,)).fetchone()
        return self._document_row(row) if row else None

    def commit_document(self, document: DocumentRecord, chunks: list[ChunkRecord], media: list[MediaRecord], embeddings=None) -> None:
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
        return {**self._document_row(row), "chunks": self.list_chunks(document_id), "media": self.list_media(document_id), "artifacts": self.list_document_artifacts(document_id)}

    def upsert_document_artifact(self, record: DocumentArtifactRecord) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO document_artifacts VALUES (?,?,?,?,?,?,?,?,?)",
                (record.id, record.document_id, record.artifact_type, record.relative_path, record.mime_type, record.source, record.checksum, record.byte_size, self._now()),
            )

    def get_document_artifact(self, document_id: str, artifact_type: str = "source_markdown") -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM document_artifacts WHERE document_id=? AND artifact_type=?", (document_id, artifact_type)).fetchone()
        return dict(row) if row else None

    def list_document_artifacts(self, document_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM document_artifacts WHERE document_id=? ORDER BY artifact_type,id", (document_id,))]

    def get_document_in_base(self, document_id: str, category_id: str) -> dict[str, Any] | None:
        item = self.get_document(document_id)
        return item if item and item["category_id"] == category_id else None

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

    def get_media_vlm_analysis(
        self,
        media_id: str,
        media_checksum: str,
        profile_fingerprint: str,
        prompt_version: str,
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT analysis_json FROM media_vlm_analyses "
            "WHERE media_id=? AND media_checksum=? AND profile_fingerprint=? AND prompt_version=?",
            (media_id, media_checksum, profile_fingerprint, prompt_version),
        ).fetchone()
        if not row:
            return None
        value = self._loads(row["analysis_json"], None)
        return value if isinstance(value, dict) else None

    def upsert_media_vlm_analysis(
        self,
        media_id: str,
        media_checksum: str,
        profile_fingerprint: str,
        prompt_version: str,
        analysis: dict[str, Any],
    ) -> None:
        if not isinstance(analysis, dict):
            raise UsageError("Media VLM analysis must be a JSON object")
        payload = json.dumps(analysis, ensure_ascii=False, allow_nan=False)
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO media_vlm_analyses VALUES (?,?,?,?,?,?)",
                (media_id, media_checksum, profile_fingerprint, prompt_version, payload, self._now()),
            )

    def delete_document(self, document_id: str) -> None:
        with self.transaction() as db:
            if not db.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount:
                raise UsageError(f"Document not found: {document_id}")

    def get_chunk_window(self, chunk_id: str, before: int, after: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        center = self._conn.execute("SELECT document_id,sequence FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if not center:
            return [], []
        rows = self.list_chunks(center["document_id"])
        prior = [item for item in rows if center["sequence"] - before <= item["sequence"] < center["sequence"]]
        following = [item for item in rows if center["sequence"] < item["sequence"] <= center["sequence"] + after]
        return prior, following

    def replace_document_graph(self, document_id: str, kind: str, nodes: list[GraphNodeRecord], edges: list[GraphEdgeRecord]) -> None:
        if kind not in {"entity", "reference"}:
            raise UsageError(f"Unsupported graph kind: {kind}")
        with self.transaction() as db:
            db.execute("DELETE FROM graph_edges WHERE document_id=? AND graph_kind=?", (document_id, kind))
            db.execute("DELETE FROM graph_nodes WHERE document_id=? AND graph_kind=?", (document_id, kind))
            db.executemany(
                "INSERT OR REPLACE INTO graph_nodes VALUES (?,?,?,?,?,?,?,?,?)",
                [(n.id,n.knowledge_base_id,n.graph_kind,n.node_type,n.label,n.document_id,n.page,n.evidence_chunk_id,json.dumps(n.properties,ensure_ascii=False)) for n in nodes],
            )
            db.executemany(
                "INSERT OR REPLACE INTO graph_edges VALUES (?,?,?,?,?,?,?,?,?)",
                [(e.id,e.knowledge_base_id,e.graph_kind,e.source_id,e.target_id,e.relation_type,e.document_id,e.evidence_chunk_id,json.dumps(e.properties,ensure_ascii=False)) for e in edges],
            )

    def load_graph(self, category_id: str, kind: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        clause, params = "knowledge_base_id=?", [category_id]
        if kind and kind != "combined":
            clause += " AND graph_kind=?"
            params.append(kind)
        nodes = []
        for row in self._conn.execute(f"SELECT * FROM graph_nodes WHERE {clause} ORDER BY graph_kind,id", params):
            item = dict(row); item["properties"] = self._loads(item.pop("properties_json"), {}); nodes.append(item)
        edges = []
        for row in self._conn.execute(f"SELECT * FROM graph_edges WHERE {clause} ORDER BY graph_kind,id", params):
            item = dict(row); item["properties"] = self._loads(item.pop("properties_json"), {}); edges.append(item)
        return nodes, edges

    def set_index_state(self, document_id: str, kind: str, fingerprint: str, version: str, status: str, error_code: str = "") -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO derived_index_states VALUES (?,?,?,?,?,?,?)",
                (document_id, kind, fingerprint, version, status, error_code, self._now()),
            )

    def index_states(self, document_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM derived_index_states WHERE document_id=? ORDER BY index_kind", (document_id,))]

    def update_chunk_media_refs(self, chunk_id: str, refs: list[dict[str, Any]]) -> None:
        with self.transaction() as db:
            db.execute("UPDATE chunks SET media_refs_json=? WHERE id=?", (json.dumps(refs, ensure_ascii=False), chunk_id))

    def clear_index_states(self, index_kind: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM derived_index_states WHERE index_kind=?", (index_kind,))

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
