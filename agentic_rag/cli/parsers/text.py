"""Text and Markdown parsing."""

from pathlib import Path

from ..cancellation import CancellationToken
from ..errors import ConfigurationError
from ..models import ParsedDocument


def parse_text(path: Path, cancel: CancellationToken) -> ParsedDocument:
    cancel.checkpoint()
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ConfigurationError("Text file exceeds the 20MB limit")
    text = path.read_text("utf-8-sig", errors="replace").strip()
    cancel.checkpoint()
    if not text:
        raise ConfigurationError("Text document is empty")
    return ParsedDocument(path.name, [{"page": 1, "text": text}], [], "local_text")
