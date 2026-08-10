"""Cross-platform, isolated filesystem locations for AutoMemory."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AutoMemoryPaths:
    root: Path
    state_db: Path
    knowledge_db: Path
    vector_dir: Path
    media_file: Path
    graph_file: Path
    logs_dir: Path
    exports_dir: Path
    cache_dir: Path

    @classmethod
    def resolve(cls, root: str | Path | None = None, create: bool = True) -> "AutoMemoryPaths":
        configured = root or os.getenv("AUTOMEMORY_HOME")
        if configured:
            base = Path(configured).expanduser()
            if not base.is_absolute():
                raise ValueError("AUTOMEMORY_HOME must be an absolute path")
        else:
            try:
                from platformdirs import user_data_path
            except ImportError as exc:  # pragma: no cover - entry point gives install hint
                raise RuntimeError('AutoMemory requires the TUI extra: pip install -e ".[tui]"') from exc
            base = user_data_path("AutoMemory", "AutoMemory", roaming=True)
        base = base.resolve()
        if base.exists() and not base.is_dir():
            raise ValueError(f"AutoMemory data path is not a directory: {base}")
        paths = cls(
            root=base,
            state_db=base / "state" / "automemory.db",
            knowledge_db=base / "knowledge" / "knowledge.db",
            vector_dir=base / "knowledge" / "vectors",
            media_file=base / "knowledge" / "media" / "registry.json",
            graph_file=base / "knowledge" / "graph.json",
            logs_dir=base / "logs",
            exports_dir=base / "exports",
            cache_dir=base / "cache",
        )
        if create:
            for directory in {
                paths.root, paths.state_db.parent, paths.knowledge_db.parent,
                paths.vector_dir, paths.media_file.parent, paths.logs_dir,
                paths.exports_dir, paths.cache_dir,
            }:
                directory.mkdir(parents=True, exist_ok=True)
            probe = paths.root / ".write-test"
            try:
                probe.write_bytes(b"")
                probe.unlink()
            except OSError as exc:
                raise PermissionError(f"AutoMemory data directory is not writable: {paths.root}") from exc
        return paths

    def contains(self, path: str | Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False
