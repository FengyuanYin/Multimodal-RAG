"""Console entry point with a useful optional-dependency error."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="automemory", description="AutoMemory local terminal workspace")
    parser.add_argument("--home", type=Path, help="override the isolated AutoMemory data directory")
    args = parser.parse_args(argv)
    if args.home:
        os.environ["AUTOMEMORY_HOME"] = str(args.home.expanduser().resolve())
    try:
        from .app import AutoMemoryApp
    except ModuleNotFoundError as exc:
        if exc.name == "textual":
            parser.error("Textual is not installed. Run: pip install -e '.[tui]'")
        raise
    AutoMemoryApp().run()
