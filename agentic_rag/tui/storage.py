"""Versioned SQLite persistence for AutoMemory UI state."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time
import uuid
from typing import Any, Iterator

from .models import CategoryRecord, Conversation, MemoryRecord, MessageRecord
from .security import looks_secret_key


SCHEMA_VERSION = 1


class StateRepository:
    def __init__(self, path: str | Path, now=time.time, id_factory=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._id_factory = id_factory or (lambda prefix: f"{prefix}_{uuid.uuid4().hex}")
        self._lock = RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._migrate()
        self.recover_streaming_messages()

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

    def _migrate(self) -> None:
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS categories(id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE, created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS messages(
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content TEXT NOT NULL,
                    chat_mode TEXT NOT NULL CHECK(chat_mode IN ('direct','rag')),
                    status TEXT NOT NULL CHECK(status IN ('streaming','complete','interrupted','error')),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at, id);
                CREATE TABLE IF NOT EXISTS memories(id TEXT PRIMARY KEY, content TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluation_runs(
                    id TEXT PRIMARY KEY, dataset_path TEXT NOT NULL, config_json TEXT NOT NULL,
                    result_path TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
                    created_at REAL NOT NULL, completed_at REAL
                );
            """)
            db.execute("INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)", (SCHEMA_VERSION, self._now()))
            now = self._now()
            db.execute("INSERT OR IGNORE INTO categories(id,name,created_at,updated_at) VALUES ('default','Default',?,?)", (now, now))

    @staticmethod
    def _loads(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return fallback

    def load_settings(self) -> dict[str, Any]:
        result = {}
        for row in self._conn.execute("SELECT key,value_json FROM settings"):
            result[row["key"]] = self._loads(row["value_json"], None)
        return result

    def save_settings(self, payload: dict[str, Any]) -> None:
        forbidden = [key for key in payload if looks_secret_key(key)]
        if forbidden:
            raise ValueError("secret settings cannot be persisted")
        now = self._now()
        with self.transaction() as db:
            for key, value in payload.items():
                db.execute(
                    "INSERT INTO settings(key,value_json,updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at",
                    (key, json.dumps(value, ensure_ascii=False), now),
                )

    def create_category(self, name: str) -> CategoryRecord:
        name = name.strip()
        if not name:
            raise ValueError("category name is required")
        now, category_id = self._now(), self._id_factory("cat")
        with self.transaction() as db:
            db.execute("INSERT INTO categories(id,name,created_at,updated_at) VALUES (?,?,?,?)", (category_id, name, now, now))
        return CategoryRecord(category_id, name, now, now)

    def list_categories(self) -> list[CategoryRecord]:
        return [CategoryRecord(**dict(row)) for row in self._conn.execute("SELECT * FROM categories ORDER BY lower(name), id")]

    def rename_category(self, category_id: str, name: str) -> None:
        if not name.strip():
            raise ValueError("category name is required")
        with self.transaction() as db:
            cur = db.execute("UPDATE categories SET name=?,updated_at=? WHERE id=?", (name.strip(), self._now(), category_id))
            if not cur.rowcount:
                raise KeyError(category_id)

    def delete_category(self, category_id: str) -> None:
        if category_id == "default":
            raise ValueError("the default category cannot be deleted")
        with self.transaction() as db:
            cur = db.execute("DELETE FROM categories WHERE id=?", (category_id,))
            if not cur.rowcount:
                raise KeyError(category_id)

    def create_conversation(self, title: str = "New conversation") -> Conversation:
        now, conv_id = self._now(), self._id_factory("conv")
        with self.transaction() as db:
            db.execute("INSERT INTO conversations(id,title,created_at,updated_at) VALUES (?,?,?,?)", (conv_id, title.strip() or "New conversation", now, now))
        return Conversation(conv_id, title.strip() or "New conversation", now, now)

    def list_conversations(self) -> list[Conversation]:
        return [Conversation(**dict(row)) for row in self._conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC,id")]

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self.transaction() as db:
            cur = db.execute("UPDATE conversations SET title=?,updated_at=? WHERE id=?", (title.strip() or "New conversation", self._now(), conversation_id))
            if not cur.rowcount:
                raise KeyError(conversation_id)

    def clear_conversation(self, conversation_id: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (self._now(), conversation_id))

    def delete_conversation(self, conversation_id: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def append_message(self, conversation_id: str, role: str, content: str, chat_mode: str, status: str, metadata: dict | None = None) -> MessageRecord:
        now, message_id = self._now(), self._id_factory("msg")
        metadata = metadata or {}
        with self.transaction() as db:
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", (message_id, conversation_id, role, content, chat_mode, status, json.dumps(metadata, ensure_ascii=False), now))
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
        return MessageRecord(message_id, conversation_id, role, content, chat_mode, status, metadata, now)

    def update_message(self, message_id: str, status: str, content: str, metadata: dict | None = None) -> None:
        with self.transaction() as db:
            row = db.execute("SELECT status FROM messages WHERE id=?", (message_id,)).fetchone()
            if not row:
                raise KeyError(message_id)
            if row["status"] != "streaming":
                raise ValueError("only streaming messages may transition")
            if status not in {"complete", "interrupted", "error"}:
                raise ValueError("invalid final message status")
            db.execute("UPDATE messages SET status=?,content=?,metadata_json=? WHERE id=?", (status, content, json.dumps(metadata or {}, ensure_ascii=False), message_id))

    def list_messages(self, conversation_id: str, limit: int = 200) -> list[MessageRecord]:
        rows = self._conn.execute(
            "SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at DESC,id DESC LIMIT ?) ORDER BY created_at,id",
            (conversation_id, max(1, min(limit, 1000))),
        )
        output = []
        for row in rows:
            item = dict(row)
            item["metadata"] = self._loads(item.pop("metadata_json"), {})
            output.append(MessageRecord(**item))
        return output

    def recover_streaming_messages(self) -> int:
        with self.transaction() as db:
            cur = db.execute("UPDATE messages SET status='interrupted' WHERE status='streaming'")
        return cur.rowcount

    def create_memory(self, content: str, enabled: bool = True) -> MemoryRecord:
        content = content.strip()
        if not content:
            raise ValueError("memory content is required")
        now, memory_id = self._now(), self._id_factory("mem")
        with self.transaction() as db:
            db.execute("INSERT INTO memories VALUES (?,?,?,?,?)", (memory_id, content, int(enabled), now, now))
        return MemoryRecord(memory_id, content, enabled, now, now)

    def list_memories(self, enabled_only: bool = False) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories" + (" WHERE enabled=1" if enabled_only else "") + " ORDER BY updated_at DESC,id"
        return [MemoryRecord(row["id"], row["content"], bool(row["enabled"]), row["created_at"], row["updated_at"]) for row in self._conn.execute(sql)]

    def update_memory(self, memory_id: str, *, content: str | None = None, enabled: bool | None = None) -> None:
        row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            raise KeyError(memory_id)
        next_content = row["content"] if content is None else content.strip()
        if not next_content:
            raise ValueError("memory content is required")
        next_enabled = row["enabled"] if enabled is None else int(enabled)
        with self.transaction() as db:
            db.execute("UPDATE memories SET content=?,enabled=?,updated_at=? WHERE id=?", (next_content, next_enabled, self._now(), memory_id))

    def delete_memory(self, memory_id: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM memories WHERE id=?", (memory_id,))

    def create_evaluation_run(self, dataset_path: str, config: dict[str, Any]) -> str:
        run_id, now = self._id_factory("eval"), self._now()
        with self.transaction() as db:
            db.execute("INSERT INTO evaluation_runs VALUES (?,?,?,?,?,?,?)", (run_id, dataset_path, json.dumps(config, ensure_ascii=False), "", "running", now, None))
        return run_id

    def finish_evaluation_run(self, run_id: str, status: str, result_path: str = "") -> None:
        with self.transaction() as db:
            db.execute("UPDATE evaluation_runs SET status=?,result_path=?,completed_at=? WHERE id=?", (status, result_path, self._now(), run_id))

    def list_evaluation_runs(self) -> list[dict[str, Any]]:
        output = []
        for row in self._conn.execute("SELECT * FROM evaluation_runs ORDER BY created_at DESC,id"):
            item = dict(row)
            item["config"] = self._loads(item.pop("config_json"), {})
            output.append(item)
        return output

    def integrity_check(self) -> str:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "unknown"

    def close(self) -> None:
        with self._lock:
            self._conn.close()
