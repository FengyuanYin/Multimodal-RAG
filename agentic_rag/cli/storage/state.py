"""Transactional conversations, memories, tasks, and evaluation state."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time
from typing import Any, Iterator
import uuid

from ..errors import UsageError
from ..security import looks_sensitive_text
from .migrations import connect_database, migrate


SCHEMA_VERSION = 1
MIGRATIONS = {
    1: """
    CREATE TABLE metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at REAL NOT NULL);
    CREATE TABLE conversations(id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE TABLE messages(
      id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role TEXT NOT NULL CHECK(role IN ('user','assistant','system')), content TEXT NOT NULL,
      mode TEXT NOT NULL CHECK(mode IN ('direct','rag')),
      status TEXT NOT NULL CHECK(status IN ('streaming','complete','interrupted','error')),
      metadata_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL, updated_at REAL NOT NULL
    );
    CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at, id);
    CREATE TABLE memories(id TEXT PRIMARY KEY, content TEXT NOT NULL, enabled INTEGER NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE TABLE tasks(
      id TEXT PRIMARY KEY, task_type TEXT NOT NULL, status TEXT NOT NULL,
      phase TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
      created_at REAL NOT NULL, updated_at REAL NOT NULL
    );
    CREATE TABLE evaluations(
      id TEXT PRIMARY KEY, dataset_path TEXT NOT NULL, status TEXT NOT NULL,
      config_json TEXT NOT NULL, result_path TEXT NOT NULL DEFAULT '',
      summary_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL, updated_at REAL NOT NULL
    );
    """,
}


class StateRepository:
    def __init__(self, path: Path, backups_dir: Path, *, now=time.time) -> None:
        self.path, self._now = path, now
        self._lock = RLock()
        self._conn = connect_database(path)
        migrate(self._conn, path, backups_dir, SCHEMA_VERSION, MIGRATIONS)
        self.recover_incomplete()

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

    def get_metadata(self, key: str, fallback: Any = None) -> Any:
        row = self._conn.execute("SELECT value_json FROM metadata WHERE key=?", (key,)).fetchone()
        return self._loads(row[0], fallback) if row else fallback

    def set_metadata(self, key: str, value: Any) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO metadata VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), self._now()),
            )

    def create_conversation(self, title: str = "New conversation") -> dict[str, Any]:
        now, conversation_id = self._now(), f"conv_{uuid.uuid4().hex}"
        title = title.strip() or "New conversation"
        with self.transaction() as db:
            db.execute("INSERT INTO conversations VALUES (?,?,?,?)", (conversation_id, title, now, now))
            db.execute(
                "INSERT INTO metadata(key,value_json,updated_at) VALUES ('active_conversation',?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at",
                (json.dumps(conversation_id), now),
            )
        return {"id": conversation_id, "title": title, "created_at": now, "updated_at": now}

    def ensure_active_conversation(self) -> str:
        active = str(self.get_metadata("active_conversation", "") or "")
        if active and self._conn.execute("SELECT 1 FROM conversations WHERE id=?", (active,)).fetchone():
            return active
        rows = self.list_conversations()
        if rows:
            self.set_active_conversation(rows[0]["id"])
            return rows[0]["id"]
        return self.create_conversation()["id"]

    def set_active_conversation(self, conversation_id: str) -> None:
        if not self._conn.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone():
            raise UsageError(f"Conversation not found: {conversation_id}")
        self.set_metadata("active_conversation", conversation_id)

    def list_conversations(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC,id")]

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self.transaction() as db:
            cursor = db.execute("UPDATE conversations SET title=?,updated_at=? WHERE id=?", (title.strip() or "New conversation", self._now(), conversation_id))
            if not cursor.rowcount:
                raise UsageError(f"Conversation not found: {conversation_id}")

    def clear_conversation(self, conversation_id: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            cursor = db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (self._now(), conversation_id))
            if not cursor.rowcount:
                raise UsageError(f"Conversation not found: {conversation_id}")

    def delete_conversation(self, conversation_id: str) -> None:
        with self.transaction() as db:
            cursor = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            if not cursor.rowcount:
                raise UsageError(f"Conversation not found: {conversation_id}")
        if self.get_metadata("active_conversation") == conversation_id:
            self.set_metadata("active_conversation", "")

    def append_message(self, conversation_id: str, role: str, content: str, mode: str, status: str, metadata: dict | None = None) -> dict[str, Any]:
        now, message_id = self._now(), f"msg_{uuid.uuid4().hex}"
        with self.transaction() as db:
            db.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)",
                (message_id, conversation_id, role, content, mode, status, json.dumps(metadata or {}, ensure_ascii=False), now, now),
            )
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
        return {"id": message_id, "conversation_id": conversation_id, "role": role, "content": content, "mode": mode, "status": status, "metadata": metadata or {}, "created_at": now, "updated_at": now}

    def finalize_message(self, message_id: str, status: str, content: str, metadata: dict | None = None) -> None:
        if status not in {"complete", "interrupted", "error"}:
            raise UsageError("Invalid final message status")
        with self.transaction() as db:
            row = db.execute("SELECT status FROM messages WHERE id=?", (message_id,)).fetchone()
            if not row:
                raise UsageError(f"Message not found: {message_id}")
            if row["status"] != "streaming":
                raise UsageError("Only streaming messages can be finalized")
            db.execute("UPDATE messages SET status=?,content=?,metadata_json=?,updated_at=? WHERE id=?", (status, content, json.dumps(metadata or {}, ensure_ascii=False), self._now(), message_id))

    def list_messages(self, conversation_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at DESC,id DESC LIMIT ?) ORDER BY created_at,id",
            (conversation_id, max(1, min(limit, 1000))),
        )
        output = []
        for row in rows:
            item = dict(row)
            item["metadata"] = self._loads(item.pop("metadata_json"), {})
            output.append(item)
        return output

    def add_memory(self, content: str) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise UsageError("Memory content is required")
        if looks_sensitive_text(content):
            raise UsageError("Memory looks like a credential and was not saved")
        now, memory_id = self._now(), f"mem_{uuid.uuid4().hex}"
        with self.transaction() as db:
            db.execute("INSERT INTO memories VALUES (?,?,?,?,?)", (memory_id, content, 1, now, now))
        return {"id": memory_id, "content": content, "enabled": True, "created_at": now, "updated_at": now}

    def list_memories(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM memories" + (" WHERE enabled=1" if enabled_only else "") + " ORDER BY updated_at DESC,id"
        return [{**dict(row), "enabled": bool(row["enabled"])} for row in self._conn.execute(query)]

    def set_memory_enabled(self, memory_id: str, enabled: bool) -> None:
        with self.transaction() as db:
            cursor = db.execute("UPDATE memories SET enabled=?,updated_at=? WHERE id=?", (int(enabled), self._now(), memory_id))
            if not cursor.rowcount:
                raise UsageError(f"Memory not found: {memory_id}")

    def delete_memory(self, memory_id: str) -> None:
        with self.transaction() as db:
            if not db.execute("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount:
                raise UsageError(f"Memory not found: {memory_id}")

    def create_task(self, task_type: str, metadata: dict | None = None) -> str:
        task_id, now = f"task_{uuid.uuid4().hex}", self._now()
        with self.transaction() as db:
            db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?)", (task_id, task_type, "running", "", json.dumps(metadata or {}, ensure_ascii=False), now, now))
        return task_id

    def update_task(self, task_id: str, status: str, phase: str = "", metadata: dict | None = None) -> None:
        with self.transaction() as db:
            cursor = db.execute("UPDATE tasks SET status=?,phase=?,metadata_json=?,updated_at=? WHERE id=?", (status, phase, json.dumps(metadata or {}, ensure_ascii=False), self._now(), task_id))
            if not cursor.rowcount:
                raise UsageError(f"Task not found: {task_id}")

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,))]

    def create_evaluation(self, evaluation_id: str, dataset_path: str, config: dict) -> None:
        now = self._now()
        with self.transaction() as db:
            db.execute(
                "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?)",
                (evaluation_id, dataset_path, "running", json.dumps(config, ensure_ascii=False), "", "{}", now, now),
            )

    def update_evaluation(self, evaluation_id: str, status: str, summary: dict | None = None, result_path: str = "") -> None:
        with self.transaction() as db:
            cursor = db.execute(
                "UPDATE evaluations SET status=?,summary_json=?,result_path=CASE WHEN ?='' THEN result_path ELSE ? END,updated_at=? WHERE id=?",
                (status, json.dumps(summary or {}, ensure_ascii=False), result_path, result_path, self._now(), evaluation_id),
            )
            if not cursor.rowcount:
                raise UsageError(f"Evaluation not found: {evaluation_id}")

    def list_evaluations(self, limit: int = 50) -> list[dict[str, Any]]:
        output = []
        for row in self._conn.execute("SELECT * FROM evaluations ORDER BY updated_at DESC LIMIT ?", (limit,)):
            item = dict(row)
            item["config"] = self._loads(item.pop("config_json"), {})
            item["summary"] = self._loads(item.pop("summary_json"), {})
            output.append(item)
        return output

    def recover_incomplete(self) -> dict[str, int]:
        now = self._now()
        with self.transaction() as db:
            messages = db.execute("UPDATE messages SET status='interrupted',updated_at=? WHERE status='streaming'", (now,)).rowcount
            tasks = db.execute("UPDATE tasks SET status='cancelled',updated_at=? WHERE status IN ('pending','running')", (now,)).rowcount
            evaluations = db.execute("UPDATE evaluations SET status='cancelled',updated_at=? WHERE status='running'", (now,)).rowcount
        return {"messages": messages, "tasks": tasks, "evaluations": evaluations}

    def integrity_check(self) -> str:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "unknown"

    def close(self) -> None:
        with self._lock:
            self._conn.close()
