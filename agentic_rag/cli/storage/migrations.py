"""SQLite connection, backup, and schema migration helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
import time

from ..errors import ConfigurationError


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False, timeout=20.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 20000")
    return connection


def migrate(connection: sqlite3.Connection, path: Path, backups_dir: Path, target_version: int, scripts: dict[int, str]) -> None:
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > target_version:
        raise ConfigurationError(f"Database schema {current} is newer than supported version {target_version}")
    if current == target_version:
        return
    if path.exists() and path.stat().st_size:
        backups_dir.mkdir(parents=True, exist_ok=True)
        backup = backups_dir / f"{path.stem}-schema-{current}-{int(time.time())}.db"
        destination = sqlite3.connect(backup)
        try:
            connection.backup(destination)
        finally:
            destination.close()
    try:
        connection.execute("BEGIN IMMEDIATE")
        for version in range(current + 1, target_version + 1):
            script = scripts.get(version)
            if script is None:
                raise ConfigurationError(f"Missing database migration {version}")
            connection.executescript(script)
            connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
