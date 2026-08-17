"""Isolated, cross-platform AutoMemory paths."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class AutoMemoryPaths:
    root: Path
    config_file: Path
    state_db: Path
    knowledge_db: Path
    history_file: Path
    media_dir: Path
    logs_dir: Path
    exports_dir: Path
    cache_dir: Path
    backups_dir: Path
    knowledge_assets_dir: Path
    workspaces_dir: Path

    @classmethod
    def resolve(cls, root: str | Path | None = None, *, create: bool = True) -> "AutoMemoryPaths":
        configured = root or os.getenv("AUTOMEMORY_HOME")
        if configured:
            base = Path(configured).expanduser()
            if not base.is_absolute():
                raise ConfigurationError("AUTOMEMORY_HOME must be an absolute path")
        else:
            base = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming")) / "AutoMemory"
        base = base.resolve()
        paths = cls(
            root=base,
            config_file=base / "config.json",
            state_db=base / "state.db",
            knowledge_db=base / "knowledge.db",
            history_file=base / "history.txt",
            media_dir=base / "media",
            logs_dir=base / "logs",
            exports_dir=base / "exports",
            cache_dir=base / "cache",
            backups_dir=base / "backups",
            knowledge_assets_dir=base / "knowledge-assets",
            workspaces_dir=base / "workspaces",
        )
        if create:
            for directory in (base, paths.media_dir, paths.logs_dir, paths.exports_dir, paths.cache_dir, paths.backups_dir, paths.knowledge_assets_dir, paths.workspaces_dir):
                directory.mkdir(parents=True, exist_ok=True)
            probe = base / ".write-test"
            try:
                probe.write_bytes(b"")
                probe.unlink()
            except OSError as exc:
                raise ConfigurationError(f"AutoMemory home is not writable: {base}") from exc
        return paths

    def contains(self, path: str | Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False
