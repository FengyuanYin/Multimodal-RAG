"""Lightweight deterministic local document parsers."""

from pathlib import Path

from ..cancellation import CancellationToken
from ..errors import ConfigurationError
from ..models import ParsedDocument
from .image import parse_image
from .pdf import parse_pdf
from .table import parse_table
from .text import parse_text


def parse_local(path: Path, cancel: CancellationToken) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, cancel)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return parse_image(path, cancel)
    if suffix in {".csv", ".tsv", ".xlsx", ".xlsm"}:
        return parse_table(path, cancel)
    if suffix in {".txt", ".md", ".markdown", ".rst", ".json", ".yaml", ".yml"}:
        return parse_text(path, cancel)
    raise ConfigurationError(f"Unsupported local file type: {suffix or '(none)'}")


__all__ = ["parse_local"]
