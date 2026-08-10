"""PyMuPDF text and bounded embedded-image extraction."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path

from ..cancellation import CancellationToken
from ..errors import ConfigurationError
from ..models import ParsedDocument


def parse_pdf(path: Path, cancel: CancellationToken) -> ParsedDocument:
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ConfigurationError("PDF exceeds the 100MB local parsing limit")
    try:
        import fitz
    except ImportError as exc:
        raise ConfigurationError("PDF support requires PyMuPDF from the CLI dependencies") from exc
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ConfigurationError(f"PDF could not be opened: {type(exc).__name__}") from exc
    pages, media, media_total = [], [], 0
    try:
        if document.page_count > 5000:
            raise ConfigurationError("PDF exceeds the 5000-page limit")
        for page_index in range(document.page_count):
            cancel.checkpoint()
            page = document.load_page(page_index)
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_index + 1, "text": text})
            for image_index, item in enumerate(page.get_images(full=True), 1):
                if len(media) >= 200:
                    break
                try:
                    extracted = document.extract_image(item[0])
                    raw = extracted.get("image") or b""
                except Exception:
                    continue
                if not raw or len(raw) > 8 * 1024 * 1024 or media_total + len(raw) > 64 * 1024 * 1024:
                    continue
                media_total += len(raw)
                extension = str(extracted.get("ext") or "png")
                checksum = hashlib.sha256(raw).hexdigest()
                media.append({
                    "id": f"pdf_image_{checksum[:16]}", "page": page_index + 1, "type": "image",
                    "label": f"figure{len(media) + 1}", "caption": "", "data": base64.b64encode(raw).decode("ascii"),
                    "mime_type": mimetypes.guess_type(f"image.{extension}")[0] or f"image/{extension}",
                    "checksum": checksum, "quality": "exact",
                })
    finally:
        document.close()
    if not pages and not media:
        raise ConfigurationError("PDF contains no locally readable text or images; use /mineru")
    if not pages:
        pages = [{"page": 1, "text": f"[Image-only PDF: {path.name}]"}]
    return ParsedDocument(path.name, pages, media, "local_pymupdf")
