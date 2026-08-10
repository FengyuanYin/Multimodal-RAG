"""Command-line argument parsing for AutoMemory."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import CLIOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automemory",
        description="AutoMemory cloud-model assistant and local multimodal knowledge CLI.",
    )
    parser.add_argument("-p", "--prompt", help="Run one prompt and exit (use '/s ...' for knowledge retrieval)")
    parser.add_argument("--home", type=Path, help="Override the AutoMemory data directory")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--debug", action="store_true", help="Show unexpected error details")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--plain", action="store_true", help="Disable interactive line editing")
    return parser


def parse_args(argv: list[str] | None = None) -> CLIOptions:
    value = build_parser().parse_args(argv)
    return CLIOptions(
        prompt=value.prompt,
        home=value.home,
        no_color=value.no_color,
        debug=value.debug,
        version=value.version,
        force_plain=value.plain,
    )
