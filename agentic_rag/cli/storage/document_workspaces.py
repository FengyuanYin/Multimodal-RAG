"""Persistence facade for isolated full-document workspaces."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..errors import UsageError


class DocumentWorkspaceRepository:
    def __init__(self, connection, transaction, now) -> None:
        self._conn, self._transaction, self._now = connection, transaction, now

    @staticmethod
    def _row(row):
        if not row:
            return None
        item = dict(row)
        if "metadata_json" in item:
            try:
                item["metadata"] = json.loads(item.pop("metadata_json"))
            except json.JSONDecodeError:
                item["metadata"] = {}
        return item

    def open_or_create(self, conversation_id: str, document_id: str, artifact_id: str, checksum: str, model_fp: str, prompt_version: str) -> dict:
        row = self._conn.execute("SELECT * FROM document_workspaces WHERE conversation_id=? AND document_id=?", (conversation_id, document_id)).fetchone()
        now = self._now()
        with self._transaction() as db:
            db.execute("UPDATE document_workspaces SET active=0 WHERE conversation_id=?", (conversation_id,))
            if row:
                db.execute("UPDATE document_workspaces SET active=1,status='ready',markdown_artifact_id=?,markdown_checksum=?,main_model_fingerprint=?,prompt_version=?,updated_at=? WHERE id=?", (artifact_id, checksum, model_fp, prompt_version, now, row["id"]))
                workspace_id = row["id"]
            else:
                workspace_id = f"ws_{uuid.uuid4().hex}"
                db.execute("INSERT INTO document_workspaces VALUES (?,?,?,?,?,?,?,?,?,?,?)", (workspace_id, conversation_id, document_id, artifact_id, checksum, model_fp, prompt_version, 1, "ready", now, now))
        return self.get(workspace_id)

    def get(self, workspace_id: str) -> dict | None:
        return self._row(self._conn.execute("SELECT * FROM document_workspaces WHERE id=?", (workspace_id,)).fetchone())

    def active_for_conversation(self, conversation_id: str) -> dict | None:
        return self._row(self._conn.execute("SELECT * FROM document_workspaces WHERE conversation_id=? AND active=1 AND status='ready'", (conversation_id,)).fetchone())

    def set_active(self, conversation_id: str, workspace_id: str | None) -> None:
        with self._transaction() as db:
            db.execute("UPDATE document_workspaces SET active=0,updated_at=? WHERE conversation_id=?", (self._now(), conversation_id))
            if workspace_id and not db.execute("UPDATE document_workspaces SET active=1,updated_at=? WHERE id=? AND conversation_id=? AND status='ready'", (self._now(), workspace_id, conversation_id)).rowcount:
                raise UsageError("Document workspace not found or unavailable")

    def append_event(self, workspace_id: str, role: str, content: str, event_kind: str = "message", status: str = "complete", metadata: dict | None = None, file_id: str | None = None) -> dict:
        sequence = int(self._conn.execute("SELECT coalesce(max(sequence),0)+1 FROM workspace_events WHERE workspace_id=?", (workspace_id,)).fetchone()[0])
        event_id, now = f"wse_{uuid.uuid4().hex}", self._now()
        with self._transaction() as db:
            db.execute("INSERT INTO workspace_events VALUES (?,?,?,?,?,?,?,?,?,?,?)", (event_id, workspace_id, sequence, role, event_kind, content, file_id, status, json.dumps(metadata or {}, ensure_ascii=False), now, now))
        return self._row(self._conn.execute("SELECT * FROM workspace_events WHERE id=?", (event_id,)).fetchone())

    def finalize_event(self, event_id: str, status: str, content: str, metadata: dict | None = None, file_id: str | None = None) -> None:
        with self._transaction() as db:
            if not db.execute("UPDATE workspace_events SET status=?,content=?,metadata_json=?,file_id=?,updated_at=? WHERE id=?", (status, content, json.dumps(metadata or {}, ensure_ascii=False), file_id, self._now(), event_id)).rowcount:
                raise UsageError("Workspace event not found")

    def events(self, workspace_id: str) -> list[dict]:
        return [self._row(row) for row in self._conn.execute("SELECT * FROM workspace_events WHERE workspace_id=? AND status='complete' ORDER BY sequence", (workspace_id,))]

    def context_events(self, workspace_id: str) -> list[dict]:
        events = self.events(workspace_id)
        summaries = [item for item in events if item["event_kind"] == "summary"]
        if not summaries:
            return events
        summary = summaries[-1]
        covered = int(summary.get("metadata", {}).get("covers_through_sequence", 0))
        return [summary, *[item for item in events if item["sequence"] > covered and item["event_kind"] != "summary"]]

    def clear_events(self, workspace_id: str) -> None:
        with self._transaction() as db:
            db.execute("DELETE FROM workspace_events WHERE workspace_id=?", (workspace_id,))

    def add_file(self, record: dict) -> None:
        with self._transaction() as db:
            db.execute("INSERT INTO workspace_files VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (record["id"], record["workspace_id"], record["file_kind"], record["relative_path"], record["display_name"], record["checksum"], record["byte_size"], record["token_estimate"], record["purpose"], record.get("status", "ready"), json.dumps(record.get("metadata", {}), ensure_ascii=False), self._now(), self._now()))

    def list_files(self, workspace_id: str) -> list[dict]:
        return [self._row(row) for row in self._conn.execute("SELECT * FROM workspace_files WHERE workspace_id=? ORDER BY created_at,id", (workspace_id,))]

    def get_file(self, workspace_id: str, file_id: str) -> dict | None:
        return self._row(self._conn.execute("SELECT * FROM workspace_files WHERE workspace_id=? AND id=? AND status='ready'", (workspace_id, file_id)).fetchone())

    def mark_file_deleted(self, workspace_id: str, file_id: str) -> None:
        with self._transaction() as db:
            if not db.execute("UPDATE workspace_files SET status='deleted',updated_at=? WHERE workspace_id=? AND id=? AND file_kind!='source_markdown'", (self._now(), workspace_id, file_id)).rowcount:
                raise UsageError("Generated workspace file not found")
            db.execute("UPDATE workspace_events SET file_id=NULL,metadata_json=json_set(metadata_json,'$.file_status','unavailable') WHERE workspace_id=? AND file_id=?", (workspace_id,file_id))

    def cache_get(self, key: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM workspace_tool_cache WHERE cache_key=?", (key,)).fetchone()
        if row:
            with self._transaction() as db:
                db.execute("UPDATE workspace_tool_cache SET last_used_at=? WHERE cache_key=?", (self._now(), key))
        return dict(row) if row else None

    def cache_put(self, key: str, workspace_id: str, tool_name: str, text: str, file_id: str | None = None) -> None:
        now = self._now()
        with self._transaction() as db:
            db.execute("INSERT OR REPLACE INTO workspace_tool_cache VALUES (?,?,?,?,?,?,?)", (key, workspace_id, tool_name, file_id, text, now, now))

    def invalidate_document(self, document_id: str, status: str = "stale") -> None:
        with self._transaction() as db:
            db.execute("UPDATE document_workspaces SET status=?,active=0,updated_at=? WHERE document_id=?", (status, self._now(), document_id))
